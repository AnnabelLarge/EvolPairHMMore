#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:18:49 2024

@author: annabel

ABOUT:
======
Load aligned pfams, optionally calculate emission and transition 
  counts from pair alignments. Possibility of single or mixture models
  over substitutions, equilibrium distributions, and indel models

"""
import os
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None # this is annoying
import numpy as np
import copy
from tqdm import tqdm
import json

import jax
from jax import numpy as jnp
import optax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# custom imports
from utils.setup_utils import setup_training_dir, model_import_register
from utils.training_testing_fns import train_fn, eval_fn
from utils.init_dataloaders import init_dataloaders
from utils.retrieve_transition_mats import retrieve_transition_mats





def train_pairhmm(args, dataloader_tup):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES    ########################
    ###########################################################################
    print('0: checking config')
    
    # if not provided, assume not in debug mode
    if 'debug' not in vars(args):
        args.debug = False
    
    # previous version of code allowed the option for lazy dataloading, but
    #   since GGI model is so small, just get rid of that
    assert args.loadtype == 'eager'
    
    ### DECIDE MODEL PARTS TO IMPORT, REGISTER AS PYTREES
    out = model_import_register(args)
    subst_model, equl_model, indel_model, logfile_msg = out
    del out
    
    
    ### DECIDE TRAINING MODE
    # Use this training mode if you already have precalculated count matrices
    if args.have_precalculated_counts:
        to_add = ('Reading from precalculated counts matrices before'+
                  ' training\n\n')
        
    # Use this training mode if you need to calculate emission and transition  
    #   counts from pair alignments
    elif not args.have_precalculated_counts:
        to_add = ('Calculating counts matrices from alignments, then'+
                  ' training\n\n')
        
        # Later, clip the alignments to one of four possible alignment lengths, 
        #   thus jit-compiling four versions of summarize_alignment
        #   (saves time by not having to calculate counts for excess 
        #   padding tokens)
        def clip_batch_inputs(batch, global_max_seqlen):
            # unpack briefly to get max len in the batch
            batch_seqlens = batch[-2]
            longest_seqlen = batch_seqlens.max()
            
            # determin a bin
            if longest_seqlen <= 800:
                return 800
            elif longest_seqlen <= 1100:
                return 1100
            elif longest_seqlen <= 1800:
                return 1800
            else:
                return global_max_seqlen
    
    logfile_msg = logfile_msg + to_add
    del to_add
    
        
    ###########################################################################
    ### 1: SETUP   ############################################################
    ###########################################################################
    print(f'1: setup')
    ### rng key, folder setup, etc.
    # setup folders; manually create model checkpoint directory (i.e. what 
    #   orbax would normally do for you)
    setup_training_dir(args)
    if not os.path.exists(f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'):
        os.mkdir(f'{os.getcwd()}/{args.training_wkdir}/model_ckpts')
    os.mkdir(args.model_ckpts_dir)
    
    # rng key
    rngkey = jax.random.key(args.rng_seednum)
    
    # create a new logfile to record training loss in an ascii file
    # can eyeball this faster than a tensorboard thing
    with open(args.logfile_name,'w') as g:
        g.write(logfile_msg)
        g.write('TRAINING PROG:\n')
    
    # setup tensorboard writer
    writer = SummaryWriter(args.tboard_dir)
    
    # if debugging, set up an intermediates folder
    folder_path = f'{os.getcwd()}/{args.training_wkdir}/HMM_INTERMEDIATES'
    subfolder_path = f'{folder_path}/{args.runname}'
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    if not os.path.exists(subfolder_path):
        os.mkdir(subfolder_path)
    
    args.intermediates_folder = subfolder_path
    del folder_path, subfolder_path
    
    
    ### use the helper function to import/initialize dataloaders
    # from "dataloader_tup = init_dataloaders(args)"
    training_dset, training_dl, test_dset, test_dl = dataloader_tup
    del dataloader_tup
    
    training_global_max_seqlen = training_dset.max_seqlen()
    test_global_max_seqlen = test_dset.max_seqlen()
    
    
    ### quantize time in geometric spacing, just like in cherryML
    quantization_grid = range(-(args.t_grid_num_steps-1), 
                              args.t_grid_num_steps, 
                              1)
    t_array = jnp.array([(args.t_grid_center * args.t_grid_step**q_i) for q_i in quantization_grid])
    
    
    ### column header for output eval depends on which probability is
    ###   being calculated
    if args.loss_type == 'conditional':
        eval_col_title = 'logP(A_t|A_0,model)'
    
    elif args.loss_type == 'joint':
        eval_col_title = 'logP(A_t,A_0|model)'
        
    
    
    ###########################################################################
    ### 2: INITIALIZE MODEL   #################################################
    ###########################################################################
    print('2: model init')
    ### initialize the equlibrium distribution(s)
    equl_model_params, equl_model_hparams = equl_model.initialize_params(argparse_obj=args)
    
    # if this is the base model, use the equilibrium distribution from 
    #   TRAINING data
    if args.equl_model_type == 'equl_base':
        equl_model_hparams['equl_vecs_from_train_data'] = training_dset.retrieve_equil_dist()
    
    # if you're not scoring emissions from indels at all, use this placeholder
    elif args.equl_model_type == 'no_equl':
        equl_model_hparams['equl_vecs_from_train_data'] = jnp.zeros((args.alphabet_size))
    
    
    ### initialize the substitution model
    subst_model_params, subst_model_hparams = subst_model.initialize_params(argparse_obj=args)
    
    
    ### initialize the indel model
    indel_model_params, indel_model_hparams = indel_model.initialize_params(argparse_obj=args)
    
    
    ### combine all initialized models above
    # combine all parameters to be passed to optax 
    params = {**equl_model_params, **subst_model_params, **indel_model_params}
    
    # combine all hyperparameters to be passed to training function 
    hparams = {**equl_model_hparams, **subst_model_hparams, **indel_model_hparams}
    
    # if it hasn't already been specified in the JSON file, set the gap_tok
    #   to default value of 63; this is only used for calculating counts
    if 'gap_tok' not in dir(args):
        hparams['gap_tok'] = 63
    else:
        hparams['gap_tok'] = args.gap_tok
    
    # add grid step to hparams dictionary; needed for marginaling over time
    hparams['t_grid_step']= args.t_grid_step
    
    # combine models under one pairHMM
    pairHMM = (equl_model, subst_model, indel_model)
    
    
    ###########################################################################
    ### 3: TRAINING LOOP   ####################################################
    ###########################################################################
    print(f'3: main training loop')
    ### SETUP FOR TRAINING LOOP
    # initialize optax
    tx = optax.adam(args.learning_rate)
    opt_state = tx.init(params)
    
    # jit your functions; there's an extra function if you need to 
    #   summarize the alignment
    jitted_train_fn = jax.jit(train_fn, static_argnames=['loss_type', 'DEBUG_FLAG'])
    jitted_eval_fn = jax.jit(eval_fn, static_argnames=['loss_type', 'DEBUG_FLAG'])
    if not args.have_precalculated_counts:
        summarize_alignment_jitted = jax.jit(summarize_alignment, 
                                             static_argnames=['max_seq_len',
                                                              'alphabet_size',
                                                              'gap_tok',
                                                              'subsOnly'])
        
    # quit training if test loss increases for X epochs in a row
    prev_test_loss = 9999
    early_stopping_counter = 0
    
    # what/when to save a model's parameters
    best_epoch = -1
    best_test_loss = 9999
    best_params = dict()
    rng_at_best_epoch = 0
    for epoch_idx in tqdm(range(args.num_epochs)):
        # top of the epoch, these aren't yet determined
        epoch_train_loss = 9999
        epoch_test_loss = 9999
        
        
        ######################
        ### TRAINING PHASE   #
        ######################
        epoch_train_sum_logP = 0
        for batch_idx, batch in enumerate(training_dl):
            # fold in epoch_idx and batch_idx for training
            rngkey_for_training = jax.random.fold_in(rngkey, epoch_idx+batch_idx)
            
            # if you DON'T have precalculated counts matrices, will need to 
            #   clip the batch inputs
            if not args.have_precalculated_counts:
                batch_max_seqlen = clip_batch_inputs(batch, 
                                                     global_max_seqlen = training_global_max_seqlen)
                allCounts = summarize_alignment_jitted(batch, 
                                                max_seq_len = batch_max_seqlen, 
                                                alphabet_size=hparams['alphabet_size'], 
                                                gap_tok=hparams['gap_tok'],
                                                subsOnly = args.subsOnly)
                del batch_max_seqlen
            
            # if you have these counts, just unpack the batch
            elif args.have_precalculated_counts:
                allCounts = (batch[0], batch[1], batch[2], batch[3])
            
            # take a step using minibatch gradient descent
            out = jitted_train_fn(all_counts = allCounts, 
                                  t_arr = t_array, 
                                  pairHMM = pairHMM, 
                                  params_dict = params, 
                                  hparams_dict = hparams,
                                  training_rngkey = rngkey_for_training,
                                  loss_type = args.loss_type)
            aux_dict, param_grads = out
            del out

            # update the parameters dictionary with optax
            updates, opt_state = tx.update(param_grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # add to total loss for this epoch
            epoch_train_sum_logP += aux_dict['sum_logP']
        
        
        ######################################################
        ### GET THE AVERAGE EPOCH TRAINING LOSS AND RECORD   #
        ######################################################
        # aggregate by dividing the sum by the total number of training samples
        epoch_train_loss = float( -( epoch_train_sum_logP/len(training_dset) ) )
        writer.add_scalar('Loss/training set', epoch_train_loss, epoch_idx)

        # if the training loss is nan, stop training
        if jnp.isnan(epoch_train_loss):
            with open(args.logfile_name,'a') as g:
                g.write(f'NaN training loss at epoch {epoch_idx}')
            raise ValueError(f'NaN training loss at epoch {epoch_idx}')
        
        # free up variables
        del batch, allCounts, epoch_train_sum_logP

    
        ##############################################
        ### CHECK PERFORMANCE ON HELD-OUT TEST SET   #
        ##############################################
        epoch_test_sum_logP = 0
        for batch_idx,batch in enumerate(test_dl):
            # fold in epoch_idx and batch_idx for eval (use negative value, 
            #   to have distinctly different random keys from training)
            rngkey_for_eval = jax.random.fold_in(rngkey, -(epoch_idx+batch_idx))
            
            # if you DON'T have precalculated counts matrices, will need to 
            #   clip the batch inputs
            if not args.have_precalculated_counts:
                batch_max_seqlen = clip_batch_inputs(batch, 
                                                     global_max_seqlen = test_global_max_seqlen)
                allCounts = summarize_alignment_jitted(batch, 
                                                max_seq_len = batch_max_seqlen, 
                                                alphabet_size=hparams['alphabet_size'], 
                                                gap_tok=hparams['gap_tok'],
                                                subsOnly = args.subsOnly)
                del batch_max_seqlen
            
            # if you have these counts, just unpack the batch
            elif args.have_precalculated_counts:
                allCounts = (batch[0], batch[1], batch[2], batch[3])
            
            # evaluate batch loss
            out = jitted_eval_fn(all_counts = allCounts, 
                                 t_arr = t_array, 
                                 pairHMM = pairHMM, 
                                 params_dict = params, 
                                 hparams_dict = hparams,
                                 eval_rngkey = rngkey_for_eval,
                                 loss_type = args.loss_type,
                                 DEBUG_FLAG = args.debug)
            
            aux_dict, batch_test_sum_logP = out
            del out
            
            epoch_test_sum_logP += batch_test_sum_logP
            del batch_test_sum_logP
        
        
        ##################################################
        ### GET THE AVERAGE EPOCH TEST LOSS AND RECORD   #
        ##################################################
        # aggregate by dividing the sum by the total number of training samples
        epoch_test_loss = float( -( epoch_test_sum_logP/len(test_dset) ) )
        writer.add_scalar('Loss/test set', epoch_test_loss, epoch_idx)
        del epoch_test_sum_logP, batch
        
        
        ###############################
        ### MODEL SAVING CONDITIONS   #
        ###############################
        if epoch_test_loss < best_test_loss:
            best_epoch = epoch_idx
            best_test_loss = epoch_test_loss
            best_params = params
            rng_at_best_epoch = rngkey_for_eval
            
            # record performance
            with open(args.logfile_name,'a') as g:
                g.write(f'New best TEST loss at epoch {best_epoch}: {best_test_loss}\n')
            
            
            ### if in debug mode, output the aux dict from the LAST batch of 
            ### the eval set every time you hit a "best loss"
            if args.debug:
                to_add = {'equl_vecs_from_train_data': hparams['equl_vecs_from_train_data']}
                aux_dict = {**to_add, **aux_dict}
                
                with open(f'{args.intermediates_folder}/test_set_intermediates_epoch{epoch_idx}.pkl', 'wb') as g:
                    pickle.dump(aux_dict, g)
                del aux_dict
            
    
        
        ####################################################################
        ### EARLY STOPPING: if test loss increases for X epochs in a row,  #
        ###   stop training; reset counter if the loss decreases again     #
        ###   (this is directly from Ian)                                  #
        ####################################################################
        if (jnp.allclose (prev_test_loss, 
                          jnp.minimum (prev_test_loss, epoch_test_loss), 
                          rtol=args.early_stop_rtol) ):
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        
        if early_stopping_counter == args.patience:
            with open(args.logfile_name,'a') as g:
                g.write(f'\n\nEARLY STOPPING AT EPOCH {epoch_idx}:\n')
                
            # rage quit
            break
        
        # remember this epoch's loss for next iteration
        prev_test_loss = epoch_test_loss
        


    ###########################################################################
    ### 4: POST-TRAINING ACTIONS   ############################################
    ###########################################################################
    print(f'4: post-training actions')
    # don't accidentally use old parameters
    del params
    
    # when you're done with the function, close the tensorboard writer
    writer.close()
    
    # if early stopping was never triggered, record results at last epoch
    if early_stopping_counter != args.patience:
        with open(args.logfile_name,'a') as g:
            g.write(f'\n\nRegular stopping after {epoch_idx} full epochs:\n')
    del epoch_idx
    
    
    ######################
    ### save best params #
    ######################
    # always record WHEN the parameters were saved
    epoch_idx_addition = {'epoch_of_saved_params': best_epoch}
        
    ### save params
    # undo the parameter transformations
    for modelClass in pairHMM:
        untransf_params = modelClass.undo_param_transform(best_params)
    
    # pickle the parameters, both transformed and not
    with open(f'{args.model_ckpts_dir}/params_dict.pkl','wb') as g:
        to_save = {**best_params, **untransf_params}
        to_save = {**epoch_idx_addition, **to_save}
        pickle.dump(to_save, g)
        del to_save
    
    
    ### save hyperparameters and some of the argparse
    # pickle the hyperparameters
    with open(f'{args.model_ckpts_dir}/hparams_dict.pkl','wb') as g:
        to_save = {**epoch_idx_addition, **hparams}
        pickle.dump(to_save, g)
        del to_save
    
    # pickle the entire original argparse object, after removing some 
    # useless things
    forLoad = dict(vars(args))
    
    to_remove = ['training_wkdir', 'runname', 'rng_seednum','have_precalculated_counts',
                 'loadtype', 'data_dir','training_dset_splits','test_dset_splits',
                 'batch_size', 'num_epochs', 'learning_rate', 'patience',
                 'loss_type', 'early_stop_rtol']
    for varname in to_remove:
        if varname in forLoad.keys():
            del forLoad[varname]
    forLoad = {**epoch_idx_addition, **forLoad}
    
    with open(f'{args.model_ckpts_dir}/forLoad_dict.pkl','wb') as g:
        pickle.dump(forLoad, g)
    
    
    ### add to outfile
    with open(args.logfile_name,'a') as g:
        g.write(f'Best epoch: {best_epoch}\n\n')
        g.write(f'Best params: \n')
        
        for key, val in untransf_params.items():
            g.write(f'{key}: {val}\n')
        g.write('\n')
        g.write(f'FINAL LOG-LIKELIHOODS, RE-EVALUATING ALL DATA WITH BEST PARAMS:\n')
    
    
    ###########################################
    ### loop through training dataloader and  #
    ### score with best params                #
    ###########################################    
    # if this blows up the CPU, output in parts instead
    final_loglikes_train_set = []
       
    for batch_idx, batch in enumerate(training_dl):
        rng_for_final_train = jax.random.fold_in(rngkey, (10000 + batch_idx + len(training_dl)))
        
        # if you DON'T have precalculated counts matrices, will need to 
        #   clip the batch inputs
        if not args.have_precalculated_counts:
            batch_max_seqlen = clip_batch_inputs(batch, 
                                                 global_max_seqlen = train_global_max_seqlen)
            allCounts = summarize_alignment_jitted(batch, 
                                            max_seq_len = batch_max_seqlen, 
                                            alphabet_size=hparams['alphabet_size'], 
                                            gap_tok=hparams['gap_tok'],
                                            subsOnly = args.subsOnly)
            del batch_max_seqlen
        
        # if you have these counts, just unpack the batch
        elif args.have_precalculated_counts:
            allCounts = (batch[0], batch[1], batch[2], batch[3])
        
        # evaluate batch loss
        out = jitted_eval_fn(all_counts = allCounts, 
                             t_arr = t_array, 
                             pairHMM = pairHMM, 
                             params_dict = best_params, 
                             hparams_dict = hparams,
                             eval_rngkey = rng_for_final_train,
                             loss_type = args.loss_type)
        
        aux_dict, _ = out
        del out
        
        # using batch_idx, generate the initial loss dataframe
        batch_out_df = training_dset.retrieve_sample_names(batch[-1])
        batch_out_df['logP_perSamp'] = np.array(aux_dict['logP_perSamp'])
        batch_out_df['logP_perSamp_length_normed'] = (batch_out_df['logP_perSamp'] / 
                                                      batch_out_df['desc_seq_len'])
        
        final_loglikes_train_set.append(batch_out_df)
    
    # concatenate values
    final_loglikes_train_set = pd.concat(final_loglikes_train_set)
    final_ave_train_loss = final_loglikes_train_set['logP_perSamp'].mean()
    final_ave_train_loss_seqlen_normed = final_loglikes_train_set['logP_perSamp_length_normed'].mean()
    
    # save whole dataframe and remove from memory
    final_loglikes_train_set.to_csv(f'{args.training_wkdir}/{args.runname}_train-set_loglikes.tsv', sep='\t')
    del final_loglikes_train_set
    
    # update the logfile with final losses
    with open(args.logfile_name,'a') as g:
        g.write(f'Training set average loglike: {final_ave_train_loss}\n')
        g.write(f'Training set average loglike (normed by desc seq len): {final_ave_train_loss_seqlen_normed}\n\n')
    
    # clean up variables
    del training_dl, training_dset, batch, batch_idx, rng_for_final_train
    
    
    ###########################################
    ### loop through test dataloader and      #
    ### score with best params                #
    ###########################################    
    # if this blows up the CPU, output in parts instead
    final_loglikes_test_set = []
       
    for batch_idx, batch in enumerate(test_dl):
        rng_for_final_test = jax.random.fold_in(rngkey, -(10000 + batch_idx + len(test_dl)))
        
        # if you DON'T have precalculated counts matrices, will need to 
        #   clip the batch inputs
        if not args.have_precalculated_counts:
            batch_max_seqlen = clip_batch_inputs(batch, 
                                                 global_max_seqlen = test_global_max_seqlen)
            allCounts = summarize_alignment_jitted(batch, 
                                            max_seq_len = batch_max_seqlen, 
                                            alphabet_size=hparams['alphabet_size'], 
                                            gap_tok=hparams['gap_tok'],
                                            subsOnly = args.subsOnly)
            del batch_max_seqlen
        
        # if you have these counts, just unpack the batch
        elif args.have_precalculated_counts:
            allCounts = (batch[0], batch[1], batch[2], batch[3])
        
        # evaluate batch loss
        out = jitted_eval_fn(all_counts = allCounts, 
                             t_arr = t_array, 
                             pairHMM = pairHMM, 
                             params_dict = best_params, 
                             hparams_dict = hparams,
                             eval_rngkey = rng_for_final_test,
                             loss_type = args.loss_type,
                             DEBUG_FLAG = args.debug)
        
        aux_dict, _ = out
        del out
        
        # using batch_idx, generate the initial loss dataframe
        batch_out_df = test_dset.retrieve_sample_names(batch[-1])
        batch_out_df['logP_perSamp'] = np.array(aux_dict['logP_perSamp'])
        batch_out_df['logP_perSamp_length_normed'] = (batch_out_df['logP_perSamp'] / 
                                                      batch_out_df['desc_seq_len'])
        
        final_loglikes_test_set.append(batch_out_df)
    
    # concatenate values
    final_loglikes_test_set = pd.concat(final_loglikes_test_set)
    final_ave_test_loss = final_loglikes_test_set['logP_perSamp'].mean()
    final_ave_test_loss_seqlen_normed = final_loglikes_test_set['logP_perSamp_length_normed'].mean()
    
    # save whole dataframe and remove from memory
    final_loglikes_test_set.to_csv(f'{args.training_wkdir}/{args.runname}_test-set_loglikes.tsv', sep='\t')
    del final_loglikes_test_set
    
    # update the logfile with final losses
    with open(args.logfile_name,'a') as g:
        g.write(f'Test set average loglike: {final_ave_test_loss}\n')
        g.write(f'Test set average loglike (normed by desc seq len): {final_ave_test_loss_seqlen_normed}\n\n')
    
    # output the aux dict from the final batch, if debugging
    if args.debug:
        to_add = {'equl_vecs_from_train_data': hparams['equl_vecs_from_train_data']}
        aux_dict = {**to_add, **aux_dict}
        
        with open(f'{args.intermediates_folder}/FINAL_test_set_intermediates.pkl', 'wb') as g:
            pickle.dump(aux_dict, g)
            
    
    
##########################################
### BASIC CLI+JSON CONFIG IMPLEMENTATION #
##########################################
if __name__ == '__main__':
    import json
    import argparse 
    import pandas as pd
    import numpy as np
    
    # for now, running models on single GPU
    err_ms = 'SELECT GPU TO RUN THIS COMPUTATION ON with CUDA_VISIBLE_DEVICES=DEVICE_NUM'
    assert len(jax.devices()) == 1, err_ms
    del err_ms
    
    # INITIALIZE PARSER
    parser = argparse.ArgumentParser(prog='train_pairhmm')
    
    
    # config files required to run
    parser.add_argument('--config-file',
                      type = str,
                      required=True,
                      help='Load configs from file in json format.')
    
   
    # parse the arguments
    args = parser.parse_args()
    
    
    with open(args.config_file, 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    
    
    # load data
    dataloader_tup = init_dataloaders(args)
    
    # run training function
    train_pairhmm(args, dataloader_tup)
