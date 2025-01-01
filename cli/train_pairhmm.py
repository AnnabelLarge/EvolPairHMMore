#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:18:49 2024

@author: annabel

ABOUT:
======
Load aligned pfams, optionally calculate emission and transition 
  counts from pair alignments, and fit model params. Possibility 
  of single or mixture models over substitutions, equilibrium 
  distributions, and indel models

"""
import os
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None # this is annoying
import numpy as np
import copy
from tqdm import tqdm
import json
from functools import partial

import jax
from jax import numpy as jnp
import optax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# custom imports
from calcCounts_Train.summarize_alignment import summarize_alignment 
from utils.setup_utils import setup_training_dir, model_import_register
from utils.training_testing_fns import train_fn, eval_fn



def train_pairhmm(args, dataloader_lst):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES    ########################
    ###########################################################################
    print('0: checking config')
    
    # if not provided, assume not in debug mode
    if 'debug' not in vars(args):
        DEBUG_FLAG = False
    
    else:
        DEBUG_FLAG = args.debug
    
    ### DECIDE MODEL PARTS TO IMPORT, REGISTER AS PYTREES
    out = model_import_register(args)
    subst_model, equl_model, indel_model, logfile_msg = out
    del out
    
    
    ### DECIDE TRAINING MODE
    # Use this training mode if you already have precalculated count matrices
    if args.have_precalculated_counts:
        to_add = ('Reading from precalculated counts matrices before'+
                  ' training\n')
        
    # Use this training mode if you need to calculate emission and transition  
    #   counts from pair alignments
    elif not args.have_precalculated_counts:
        to_add = ('Calculating counts matrices from alignments, then'+
                  ' training\n')
        
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
    
    # add which length you're normalizing by
    logfile_msg = logfile_msg + f'Normalizing losses by: {args.norm_loss_by}\n\n'
    
        
    ###########################################################################
    ### 1: SETUP   ############################################################
    ###########################################################################
    print(f'1: setup')
    ### rng key, folder setup, etc.
    # setup folders; manually create model checkpoint directory (i.e. what 
    #   orbax would normally do for you)
    setup_training_dir(args)
    
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
    if DEBUG_FLAG:
        folder_path = f'{os.getcwd()}/{args.training_wkdir}/HMM_INTERMEDIATES'
        
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        args.intermediates_folder = folder_path
        del folder_path
    
    
    ### use the helper function to import/initialize dataloaders
    training_dset, training_dl, test_dset, test_dl = dataloader_lst
    del dataloader_lst
    
    if not args.have_precalculated_counts:
        training_global_max_seqlen = training_dset.max_seqlen()
        test_global_max_seqlen = test_dset.max_seqlen()
    
    
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
    #   to default value of 43; this is only used for calculating counts
    if 'gap_tok' not in dir(args):
        hparams['gap_tok'] = 43
    else:
        hparams['gap_tok'] = args.gap_tok
    
    # add grid step to hparams dictionary; needed for marginaling over time
    num_timepoints = training_dset.retrieve_num_timepoints(times_from = args.times_from)
    if num_timepoints > 1:
        hparams['t_grid_step']= args.t_grid_step
    else:
        hparams['t_grid_step']= 0
    
    # combine models under one pairHMM
    pairHMM = (equl_model, subst_model, indel_model)
    
    
    ###########################################################################
    ### 3: TRAINING LOOP   ####################################################
    ###########################################################################
    print(f'3: main training loop')
    ### SETUP FOR TRAINING LOOP
    # initialize optax
    base_optimizer = optax.adam(args.learning_rate)
    tx = optax.MultiSteps(opt = base_optimizer,
                          every_k_schedule = args.every_k_schedule)
    
    opt_state = tx.init(params)
    
    # jit your functions; there's an extra function if you need to 
    #   summarize the alignment
    # for training function, automatically set debug=False
    parted_train_fn = partial(train_fn,
                              loss_type = args.loss_type,
                              norm_loss_by = args.norm_loss_by,
                              DEBUG_FLAG = False)
    jitted_train_fn = jax.jit(parted_train_fn)
    
    parted_eval_fn = partial(eval_fn,
                             norm_loss_by = args.norm_loss_by,
                             loss_type = args.loss_type)
    jitted_eval_fn = jax.jit(parted_eval_fn,
                             static_argnames = ['DEBUG_FLAG'])  
    
    if not args.have_precalculated_counts:
        parted_summary_fn = partial(summarize_alignment,
                                    alphabet_size = hparams['alphabet_size'],
                                    gap_tok = hparams['gap_tok'],
                                    subsOnly = args.subsOnly)
        
        summarize_alignment_jitted = jax.jit(parted_summary_fn, 
                                             static_argnames=['max_seq_len'])
    
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
                                                max_seq_len = batch_max_seqlen)
                del batch_max_seqlen
            
            # if you have these counts, just unpack the batch
            elif args.have_precalculated_counts:
                allCounts = (batch[0], batch[1], batch[2], batch[3])
            
            # take a step using minibatch gradient descent
            out = jitted_train_fn(all_counts = allCounts, 
                                  t_array = batch[-2],
                                  pairHMM = pairHMM, 
                                  params_dict = params, 
                                  hparams_dict = hparams,
                                  training_rngkey = rngkey_for_training)
            aux_dict, param_grads = out
            del out
            
            """
            ### DEBUG: output the transition matrix, parameters, and time 
            ### array; make sure code is calculating as Ian's code would, 
            ### for given timepoint
            with open(f'{args.training_wkdir}_INITIAL_TRANSITION_MAT.pkl','wb') as g:
                pickle.dump({'pred_logprob_trans': aux_dict['logprob_trans_mat'],
                             't_array': t_array,
                             'params': pairHMM[-1].undo_param_transform(params)}, g)
                
                raise RuntimeError('Stop after one pass; check intermediates')
            """
            # update the parameters dictionary with optax
            updates, opt_state = tx.update(param_grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            
            ### write parameters, gradients, and optimizer update size 
            ###   as you train
            batch_epoch_idx = epoch_idx * len(training_dl) + batch_idx
            for key, val in params.items():
                writer.add_scalar(f'PARAMS | {key}', 
                                  val.mean().item(), 
                                  batch_epoch_idx)
            
            for key, val in param_grads.items():
                writer.add_scalar(f'GRADIENTS | {key}', 
                                  val.mean().item(), 
                                  batch_epoch_idx)
            
            for key, val in updates.items():
                writer.add_scalar(f'UPDATES | {key}', 
                                  val.mean().item(), 
                                  batch_epoch_idx)
                
            # add to total loss for this epoch
            epoch_train_sum_logP += aux_dict['loss']
        
        
        ######################################################
        ### GET THE AVERAGE EPOCH TRAINING LOSS AND RECORD   #
        ######################################################
        # aggregate by dividing the sum by the total number of training samples
        epoch_train_loss = float( ( epoch_train_sum_logP/len(training_dset) ) )
        writer.add_scalar('Loss/training set', epoch_train_loss, epoch_idx)

        # if the training loss is nan, stop training
        if jnp.isnan(epoch_train_loss):
            with open(args.logfile_name,'a') as g:
                g.write(f'NaN training loss at epoch {epoch_idx}\n\n')
                
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
                                                max_seq_len = batch_max_seqlen)
                del batch_max_seqlen
            
            # if you have these counts, just unpack the batch
            elif args.have_precalculated_counts:
                allCounts = (batch[0], batch[1], batch[2], batch[3])
            
            # evaluate batch loss
            out = jitted_eval_fn(all_counts = allCounts, 
                                 t_array = batch[-2],
                                 pairHMM = pairHMM, 
                                 params_dict = params, 
                                 hparams_dict = hparams,
                                 eval_rngkey = rngkey_for_eval,
                                 DEBUG_FLAG = DEBUG_FLAG)
            
            aux_dict, batch_test_sum_logP = out
            del out
            
            epoch_test_sum_logP += batch_test_sum_logP
            del batch_test_sum_logP
        
        
        ##################################################
        ### GET THE AVERAGE EPOCH TEST LOSS AND RECORD   #
        ##################################################
        # aggregate by dividing the sum by the total number of training samples
        epoch_test_loss = float( ( epoch_test_sum_logP/len(test_dset) ) )
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
            if DEBUG_FLAG:
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
    untransf_params = {}
    for modelClass in pairHMM:
        untransf_params = {**untransf_params,
                           **modelClass.undo_param_transform(best_params)}
    
    # pickle the parameters, both transformed and not
    with open(f'{args.model_ckpts_dir}/params_dict.pkl','wb') as g:
        to_save = {**best_params, **untransf_params}
        to_save = {**epoch_idx_addition, **to_save}
        pickle.dump(to_save, g)
        del to_save
    
    # pickle the entire original argparse object
    with open(f'{args.model_ckpts_dir}/forLoad_argparse.pkl','wb') as g:
        pickle.dump(args, g)
    
    
    ### add to outfile
    with open(args.logfile_name,'a') as g:
        g.write(f'Best epoch: {best_epoch}\n\n')
        g.write(f'Best params: \n')
        
        for key, val in untransf_params.items():
            g.write(f'{key}: {val}\n')
        g.write('\n')
        g.write(f'FINAL LOG-LIKELIHOODS, RE-EVALUATING ALL DATA WITH BEST PARAMS:\n\n')
    
    
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
                                                 global_max_seqlen = training_global_max_seqlen)
            allCounts = summarize_alignment_jitted(batch, 
                                            max_seq_len = batch_max_seqlen)
            del batch_max_seqlen
        
        # if you have these counts, just unpack the batch
        elif args.have_precalculated_counts:
            allCounts = (batch[0], batch[1], batch[2], batch[3])
        
        # evaluate batch loss
        out = jitted_eval_fn(all_counts = allCounts, 
                             t_array = batch[-2],
                             pairHMM = pairHMM, 
                             params_dict = best_params, 
                             hparams_dict = hparams,
                             eval_rngkey = rng_for_final_train,
                             DEBUG_FLAG = False)
        
        aux_dict, _ = out
        del out
        
        # using batch_idx, generate the initial loss dataframe
        batch_out_df = training_dset.retrieve_sample_names(batch[-1])
        batch_out_df['logP'] = np.array(aux_dict['logP_perSamp_before_length_norm'])
        batch_out_df['logP/normlength'] = np.array(aux_dict['logP_perSamp'])
        batch_out_df['perplexity'] = np.exp(-batch_out_df['logP/normlength'])
        
        final_loglikes_train_set.append(batch_out_df)
    
    # concatenate values
    final_loglikes_train_set = pd.concat(final_loglikes_train_set)
    final_ave_train_loss_raw = final_loglikes_train_set['logP'].mean()
    final_ave_train_loss = final_loglikes_train_set['logP/normlength'].mean()
    final_ave_train_perpl = final_loglikes_train_set['perplexity'].mean()
    final_train_ece = np.exp(-final_ave_train_loss)
    
    # save whole dataframe and remove from memory
    final_loglikes_train_set.to_csv(f'{args.training_wkdir}/train-set_loglikes.tsv', sep='\t')
    del final_loglikes_train_set
    
    # update the logfile with final losses
    with open(args.logfile_name,'a') as g:
        g.write(f'Training set average loglike (without length normalizing): {final_ave_train_loss_raw}\n')
        g.write(f'Training set average loglike: {final_ave_train_loss}\n')
        g.write(f'Training set average perplexity: {final_ave_train_perpl}\n')
        g.write(f'Training set exponentiated cross entropy: {final_train_ece}\n\n')
    
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
                                            max_seq_len = batch_max_seqlen)
            del batch_max_seqlen
        
        # if you have these counts, just unpack the batch
        elif args.have_precalculated_counts:
            allCounts = (batch[0], batch[1], batch[2], batch[3])
        
        # evaluate batch loss
        out = jitted_eval_fn(all_counts = allCounts, 
                             t_array = batch[-2],
                             pairHMM = pairHMM, 
                             params_dict = best_params, 
                             hparams_dict = hparams,
                             eval_rngkey = rng_for_final_test,
                             DEBUG_FLAG = DEBUG_FLAG)
        
        aux_dict, _ = out
        del out
        
        # using batch_idx, generate the initial loss dataframe
        batch_out_df = test_dset.retrieve_sample_names(batch[-1])
        batch_out_df['logP'] = np.array(aux_dict['logP_perSamp_before_length_norm'])
        batch_out_df['logP/normlength'] = np.array(aux_dict['logP_perSamp'])
        batch_out_df['perplexity'] = np.exp(-batch_out_df['logP/normlength'])
        
        final_loglikes_test_set.append(batch_out_df)
    
    # concatenate values
    final_loglikes_test_set = pd.concat(final_loglikes_test_set)
    final_ave_test_loss_raw = final_loglikes_test_set['logP'].mean()
    final_ave_test_loss = final_loglikes_test_set['logP/normlength'].mean()
    final_ave_test_perpl = final_loglikes_test_set['perplexity'].mean()
    final_test_ece = np.exp(-final_ave_test_loss)
    
    # save whole dataframe and remove from memory
    final_loglikes_test_set.to_csv(f'{args.training_wkdir}/test-set_loglikes.tsv', sep='\t')
    del final_loglikes_test_set
    
    # update the logfile with final losses
    with open(args.logfile_name,'a') as g:
        g.write(f'Test set average loglike (without length normalizing): {final_ave_test_loss_raw}\n')
        g.write(f'Test set average loglike: {final_ave_test_loss}\n')
        g.write(f'Test set average perplexity: {final_ave_test_perpl}\n')
        g.write(f'Test set exponentiated cross entropy: {final_test_ece}\n')
    
    # output the aux dict from the final batch, if debugging
    if DEBUG_FLAG:
        to_add = {'equl_vecs_from_train_data': hparams['equl_vecs_from_train_data']}
        aux_dict = {**to_add, **aux_dict}
        
        with open(f'{args.intermediates_folder}/FINAL_test_set_intermediates.pkl', 'wb') as g:
            pickle.dump(aux_dict, g)
