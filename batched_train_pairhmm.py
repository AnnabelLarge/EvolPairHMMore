#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:18:49 2024

@author: annabel

ABOUT:
======
same as train_pairhmm, but for a list of JSON configs using the same dataset
  (dataset retrieved from first file)


TODO:
=====
medium:
-------
- remove the option to calculate counts on the fly, and just make this a 
  separate pre-processing script (I don't ever use it...)


far future:
-----------
For now, using LG08 exchangeability matrix, but in the future, could use 
  CherryML to calculate a new rate matrix for my specific pfam dataset?
  https://github.com/songlab-cal/CherryML

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
from jax import make_jaxpr
import optax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# custom imports
from utils.setup_utils import setup_training_dir, model_import_register
from utils.training_testing_fns import train_fn, eval_fn



def load_all_data(folder_name, first_config_filename):
    ### ARGPARSE
    with open(f'./{folder_name}/{first_config_filename}', 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
        
        
    ### DECIDE TRAINING MODE
    # previous version of code allowed the option for lazy dataloading, but
    #   since GGI model is so small, just get rid of that
    assert args.loadtype == 'eager'
    
    # Use this training mode if you already have precalculated count matrices
    if args.have_precalculated_counts:
        from onlyTrain.hmm_dataset import HMMDset_PC as hmm_reader
        from onlyTrain.hmm_dataset import jax_collator as collator
        
    # Use this training mode if you need to calculate emission and transition  
    #   counts from pair alignments
    elif not args.have_precalculated_counts:
        from calcCounts_Train.hmm_dataset import HMMDset as hmm_reader
        from calcCounts_Train.hmm_dataset import jax_collator as collator
        from calcCounts_Train.summarize_alignment import summarize_alignment
        
        
    ### READ WITH PYTORCH DATALOADERS    
    # training data
    print(f'Creating DataLoader for training set with {args.train_dset_splits}')
    training_dset = hmm_reader(data_dir = args.data_dir, 
                                split_prefixes = args.train_dset_splits)
    training_dl = DataLoader(training_dset, 
                              batch_size = args.batch_size, 
                              shuffle = True,
                              collate_fn = collator)
    
    # test data
    print(f'Creating DataLoader for test set with {args.test_dset_splits}')
    test_dset = hmm_reader(data_dir = args.data_dir, 
                              split_prefixes = args.test_dset_splits)
    test_dl = DataLoader(test_dset, 
                          batch_size = args.batch_size, 
                          shuffle = False,
                          collate_fn = collator)
    
    # wrap output into a tuple
    out = (training_dset, training_dl, test_dset, test_dl)
    
    return out



def train_batch(args, output_from_loading_func):
    #################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES #
    #################################################
    
    ### 0.1: DECIDE MODEL PARTS TO IMPORT, REGISTER AS PYTREES
    out = model_import_register(args)
    subst_model, equl_model, indel_model, logfile_msg = out
    del out
    
    ### 0.2: DECIDE TRAINING MODE
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
    
        
    ##############
    ### 1: SETUP #
    ##############
    ### 1.1: rng key, folder setup, etc.
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
    
    
    ### 1.2: read data; build pytorch dataloaders 
    # unpack result from previous loading function
    training_dset, training_dl, test_dset, test_dl = output_from_loading_func
    
    # get global max sequence lengths
    training_global_max_seqlen = training_dset.max_seqlen()
    test_global_max_seqlen = test_dset.max_seqlen()
    
    
    ### 1.3: quantize time in geometric spacing, just like in cherryML
    quantization_grid = range(-args.t_grid_num_steps, 
                              args.t_grid_num_steps + 1, 
                              1)
    t_array = jnp.array([(args.t_grid_center * args.t_grid_step**q_i) for q_i in quantization_grid])
    
    
    ###########################
    ### 2: INITIALIZE MODEL   #
    ###########################
    ### 2.1: initialize the equlibrium distribution(s)
    equl_model_params, equl_model_hparams = equl_model.initialize_params(argparse_obj=args)
    
    # if this is the base model or the placeholder, use the equilibrium 
    #   distribution from TRAINING data
    if args.equl_model_type in ['equl_base', 'no_equl']:
        equl_model_hparams['equl_vecs_fromData'] = training_dset.retrieve_equil_dist()
    
    
    ### 2.2: initialize the substitution model
    subst_model_params, subst_model_hparams = subst_model.initialize_params(argparse_obj=args)
    
    
    ### 2.3: initialize the indel model
    indel_model_params, indel_model_hparams = indel_model.initialize_params(argparse_obj=args)
    
    
    ### 2.4: combine all initialized models above
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
    
    # combine models under one pairHMM
    pairHMM = (equl_model, subst_model, indel_model)
    
    
    ########################
    ### 3: TRAINING LOOP   #
    ########################
    ### 3.1: SETUP FOR TRAINING LOOP
    # initialize optax
    tx = optax.adam(args.learning_rate)
    opt_state = tx.init(params)
    
    # jit your functions; there's an extra function if you need to 
    #   summarize the alignment
    jitted_train_fn = jax.jit(train_fn)
    jitted_eval_fn = jax.jit(eval_fn)
    if not args.have_precalculated_counts:
        summarize_alignment_jitted = jax.jit(summarize_alignment, 
                                             static_argnames='max_seq_len')
        
    # quit training if test loss increases for X epochs in a row
    prev_test_loss = 9999
    early_stopping_counter = 0
    
    # when to save a model's parameters
    best_epoch = -1
    best_train_loss = 9999
    
    
    for epoch_idx in tqdm(range(args.num_epochs)):
        # default behavior is to not save model parameters or 
        #   eval set log likelihoods
        record_results = False
        
        ### 3.2: TRAINING PHASE
        epoch_train_loss = 0
        for batch_idx, batch in enumerate(training_dl):
            # fold in epoch_idx and batch_idx for training
            rngkey_for_training = jax.random.fold_in(rngkey, epoch_idx+batch_idx)
            
            # if you DON'T have precalculated counts matrices, will need to 
            #   clip the batch inputs
            if not args.have_precalculated_counts:
                batch_max_seqlen = clip_batch_inputs(batch, 
                                                     global_max_seqlen = training_global_max_seqlen)
                allCounts = summarize_alignment(batch, 
                                                max_seq_len = batch_max_seqlen, 
                                                alphabet_size=hparams['alphabet_size'], 
                                                gap_tok=hparams['gap_tok'])
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
                                  training_rngkey = rngkey_for_training)
            batch_train_loss, param_grads = out
            del out
            
            # update the parameters dictionary with optax
            updates, opt_state = tx.update(param_grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # add to total loss for this epoch
            epoch_train_loss += batch_train_loss
            del batch_train_loss
        
        
        ### 3.3: GET THE AVERAGE EPOCH TRAINING LOSS AND RECORD
        ave_epoch_train_loss = float(epoch_train_loss/len(training_dl))
        writer.add_scalar('Loss/training set', ave_epoch_train_loss, epoch_idx)

        # if the training loss is nan, stop training
        if jnp.isnan(ave_epoch_train_loss):
            with open(args.logfile_name,'a') as g:
                g.write(f'NaN training loss at epoch {epoch_idx}')
            raise ValueError(f'NaN training loss at epoch {epoch_idx}')
        
        # free up variables
        del batch, allCounts, epoch_train_loss


        ### 3.4: IF THE TRAINING LOSS IS BETTER, SAVE MODEL WITH PARAMETERS
        if ave_epoch_train_loss < best_train_loss:
            # swap the flag
            record_results = True
            
            # output all possible things needed to load a model later
            OUT_forLoad = {'subst_model_type': args.subst_model_type,
                           'equl_model_type': args.equl_model_type,
                           'indel_model_type': args.indel_model_type,
                           'norm': args.norm,
                           'alphabet_size': args.alphabet_size,
                           't_grid_center': args.t_grid_center,
                           't_grid_step': args.t_grid_step,
                           't_grid_num_steps': args.t_grad_num_steps
                           }
            
            if 'diffrax_params' in dir(args):
                OUT_forLoad['diffrax_params'] = args.diffrax_params
            
            if 'exch_file' in dir(args):
                OUT_forLoad['exch_file'] = args.exch_file
            
            # add (possibly transformed) parameters
            for key, val in params.items():
                if val.shape == (1,):
                    OUT_forLoad[key] = val.item()
                else:
                    OUT_forLoad[key] = np.array(val)
            
            # undo any possible parameter transformations and add to 
            #   1.) the dictionary of all possible things needed to load a 
            #   model, and 2.) a human-readable JSON of parameters
            OUT_params = {}
            for modelClass in pairHMM:
                params_toWrite = modelClass.undo_param_transform(params)
                OUT_forLoad = {**OUT_forLoad, **params_toWrite}
                OUT_params = {**OUT_params, **params_toWrite}

            OUT_params['epoch_of_training']= epoch_idx
            
            # dump json files
            with open(f'{args.model_ckpts_dir}/toLoad.json', 'w') as g:
                json.dump(OUT_hparams, g, indent="\t", sort_keys=True)
            del OUT_hparams
            
            with open(f'{args.model_ckpts_dir}/params.json', 'w') as g:
                json.dump(OUT_params, g, indent="\t", sort_keys=True)
            del OUT_params
            
            # record to log file
            with open(args.logfile_name,'a') as g:
                g.write(f'New best training loss at epoch {epoch_idx}: {ave_epoch_train_loss}\n')
            
            # update save criteria
            best_train_loss = ave_epoch_train_loss
    
        
        ### 3.5: CHECK PERFORMANCE ON HELD-OUT TEST SET
        eval_df_lst = []
        epoch_test_loss = 0
        for batch_idx,batch in enumerate(test_dl):
            # fold in epoch_idx and batch_idx for eval (use negative value, 
            #   to have distinctly different random keys from training)
            rngkey_for_eval = jax.random.fold_in(rngkey, -(epoch_idx+batch_idx))
            
            # if you DON'T have precalculated counts matrices, will need to 
            #   clip the batch inputs
            if not args.have_precalculated_counts:
                batch_max_seqlen = clip_batch_inputs(batch, 
                                                     global_max_seqlen = test_global_max_seqlen)
                allCounts = summarize_alignment(batch, 
                                                max_seq_len = batch_max_seqlen, 
                                                alphabet_size=hparams['alphabet_size'], 
                                                gap_tok=hparams['gap_tok'])
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
                                 eval_rngkey = rngkey_for_eval)
            
            batch_test_loss, logprob_per_sample = out
            del out
            
            epoch_test_loss += batch_test_loss
            del batch_test_loss
            
            # if record_results is triggered (by section 2.4), also record
            # the log losses per sample
            if record_results:
                # get the batch sample labels, associated metadata
                eval_sample_idxes = batch[-1]
                meta_df_forBatch = test_dset.retrieve_sample_names(eval_sample_idxes)
                
                # add loss terms
                meta_df_forBatch['logP(ONLY_emission_at_subst)'] = logprob_per_sample[:, 0]
                meta_df_forBatch['logP(ONLY_emissions)'] = logprob_per_sample[:, 1]
                meta_df_forBatch['logP(ONLY_transitions)'] = logprob_per_sample[:, 2]
                meta_df_forBatch['logP(anc, desc, align)'] = logprob_per_sample[:, 3]
                
                eval_df_lst.append(meta_df_forBatch)

        # get the average epoch_test_loss; record
        ave_epoch_test_loss = float(epoch_test_loss/len(test_dl))
        writer.add_scalar('Loss/test set', ave_epoch_test_loss, epoch_idx)
        del epoch_test_loss, batch

        # output the metadata + losses dataframe, along with what epoch 
        #   you're recording results; place this outside of folders
        if record_results:
            eval_df = pd.concat(eval_df_lst)
            with open(f'./{args.training_wkdir}/{args.runname}_eval-set-logprobs.tsv','w') as g:
                g.write(f'#Logprobs using model params from epoch{epoch_idx}\n')
                eval_df.to_csv(g, sep='\t')


        ### 3.6: EARLY STOPPING: if test loss increases for X epochs in a row, 
        ###      stop training; reset counter if the loss decreases again 
        ###      (this is directly from Ian)
        if (jnp.allclose (prev_test_loss, 
                          jnp.minimum (prev_test_loss, ave_epoch_test_loss), 
                          rtol=1e-05) ):
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        
        if early_stopping_counter == args.patience:
            # write to logfile
            with open(args.logfile_name,'a') as g:
                g.write(f'\n\nEARLY STOPPING AT {epoch_idx}:\n')
                g.write(f'Final training loss: {ave_epoch_train_loss}\n')
                g.write(f'Final test loss: {ave_epoch_test_loss}\n')
                
            # rage quit
            break
        
        # remember this epoch's loss for next iteration
        prev_test_loss = ave_epoch_test_loss
        
        
    ### when you're done with the function, close the tensorboard writer
    writer.close()
    
    

##########################################
### BASIC CLI+JSON CONFIG IMPLEMENTATION #
##########################################
if __name__ == '__main__':
    import json
    import argparse 
    import os
    
    # for now, running models on single GPU
    err_ms = 'SELECT GPU TO RUN THIS COMPUTATION ON with CUDA_VISIBLE_DEVICES=DEVICE_NUM'
    assert len(jax.devices()) == 1, err_ms
    del err_ms
    
    
    ### INITIALIZE PARSER
    parser = argparse.ArgumentParser(prog='train_batch')
    
    # config files required to run
    parser.add_argument('--config-folder',
                        type=str,
                        required=True,
                        help='Load configs from this folder, in json format.')
    
    # parse the arguments
    init_args = parser.parse_args()
    
    
    ### MAIN PROGRAM
    # find all the json files in the folder
    file_lst = [file for file in os.listdir(init_args.config_folder) if file.endswith('.json')]
    
    # read the first config file to load data
    data_tup = load_all_data(folder_name = init_args.config_folder, 
                             first_config_filename = file_lst[0])
    
    # iterate through all config files with this same data tuple
    for config_file in file_lst:
        print(f'STARTING TRAINING FROM: {config_file}')
        to_open = f'./{init_args.config_folder}/{config_file}'
        with open(to_open, 'r') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            this_config_args = parser.parse_args(namespace=t_args)
        
        # run training function with this config file
        train_batch(args = this_config_args, 
                    output_from_loading_func = data_tup)
        
        print()
