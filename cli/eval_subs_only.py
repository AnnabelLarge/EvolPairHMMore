#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:18:49 2024

@author: annabel

ABOUT:
======
Load aligned pfams and score with a substitution matrix using equilibrium
  from TRAINING SPLIT ONLY

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
from utils.setup_utils import model_import_register
from utils.training_testing_fns import eval_fn



def eval_subs_only(args, dataloader_lst):
    ###########################################################################
    ###  0: FOLDER SETUP, MODEL/PARAM/HPARAM IMPORTS    #######################
    ###########################################################################
    print('0: checking config, loading params')
    
    # make sure config is set up correctly
    assert args.subsOnly == True
    
    # overwrite some defaults
    args.equl_model_type == "equl_base"
    args.indel_model_type == "no_indel"
    
    # create the eval working directory, if it doesn't exist
    if args.eval_wkdir not in os.listdir():
        os.mkdir(args.eval_wkdir)
        
    # if not provided, assume not in debug mode
    if 'debug' not in vars(args):
        DEBUG_FLAG = False
    else:
        DEBUG_FLAG = args.debug
    
    # load models and possibly register as pytrees
    out = model_import_register(args)
    subst_model, equl_model, indel_model, _ = out
    del out
    
    
    ### DECIDE TRAINING MODE
    if not args.have_precalculated_counts:
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
    
        
    ###########################################################################
    ### 1: SETUP   ############################################################
    ###########################################################################
    print(f'1: setup')
    # output file for individual results per sample
    output_persamp_file = f'{args.eval_wkdir}/[replace]_LOGPROB-PER-SAMP.tsv'
    output_ave_file = f'{args.eval_wkdir}/AVE-LOGPROB.tsv'
    
    # if debugging, set up an intermediates folder
    folder_path = f'{os.getcwd()}/{args.eval_wkdir}/HMM_INTERMEDIATES'
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    args.intermediates_folder = folder_path
    del folder_path

    #pretty sure this isn't ever used, but keep for compatibility I guess
    rngkey = jax.random.key(args.rng_seednum)    
    
    ### use the helper function to import/initialize dataloaders
    training_dset, training_dl, test_dset, test_dl = dataloader_lst
    del dataloader_lst
    
    if not args.have_precalculated_counts:
        training_global_max_seqlen = training_dset.max_seqlen()
        test_global_max_seqlen = test_dset.max_seqlen()
    
    
    ### quantize time in geometric spacing, just like in cherryML
    quantization_grid = range(-(args.t_grid_num_steps-1), 
                              args.t_grid_num_steps, 
                              1)
    t_array = jnp.array([(args.t_grid_center * args.t_grid_step**q_i) for q_i in quantization_grid])
    
    
    ###########################################################################
    ### 2: INITIALIZE HYPERPARAMS (BUT KEEP LOADED PARAMS)   ##################
    ###########################################################################
    print('2: hyperparam init')
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
    
    
    ## combine all hyperparameters to be passed to eval function 
    params = {**equl_model_params, **subst_model_params, **indel_model_params}
    hparams = {**equl_model_hparams, **subst_model_hparams, **indel_model_hparams}
    
    # if it hasn't already been specified in the JSON file, set the gap_tok
    #   to default value of 43; this is only used for calculating counts
    if 'gap_tok' not in dir(args):
        hparams['gap_tok'] = 43
    else:
        hparams['gap_tok'] = args.gap_tok
    
    # add grid step to hparams dictionary; needed for marginaling over time
    hparams['t_grid_step']= args.t_grid_step
    
    # combine models under one pairHMM
    pairHMM = (equl_model, subst_model, indel_model)
    
    
    ###########################################################################
    ### 3: EVAL LOOP   ########################################################
    ###########################################################################
    print(f'3: main eval loop')
    # jit your functions; there's an extra function if you need to 
    #   summarize the alignment
    parted_eval_fn = partial(eval_fn,
                             t_arr = t_array,
                             loss_type = args.loss_type,
                             norm_loss_by = args.norm_loss_by,
                             DEBUG_FLAG = DEBUG_FLAG)
    jitted_eval_fn = jax.jit(parted_eval_fn) #, 
                             # static_argnames=['loss_type',
                             #                  'DEBUG_FLAG'])
    
    if not args.have_precalculated_counts:
        parted_summary_fn = partial(summarize_alignment,
                                    alphabet_size = hparams['alphabet_size'],
                                    gap_tok = hparams['gap_tok'],
                                    subsOnly = args.subsOnly)
        
        summarize_alignment_jitted = jax.jit(parted_summary_fn, 
                                             static_argnames=['max_seq_len'])
        
        
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
                                            max_seq_len = batch_max_seqlen)
            del batch_max_seqlen
        
        # if you have these counts, just unpack the batch
        elif args.have_precalculated_counts:
            allCounts = (batch[0], batch[1], batch[2], batch[3])
        
        # evaluate batch loss
        out = jitted_eval_fn(all_counts = allCounts, 
                             pairHMM = pairHMM, 
                             params_dict = params, 
                             hparams_dict = hparams,
                             eval_rngkey = rng_for_final_train)
        
        aux_dict, _ = out
        del out
        
        # using batch_idx, generate the initial loss dataframe
        batch_out_df = training_dset.retrieve_sample_names(batch[-1])
        batch_out_df['logP_perSamp'] = np.array(aux_dict['logP_perSamp'])
        
        final_loglikes_train_set.append(batch_out_df)
        
        
    # concatenate values
    final_loglikes_train_set = pd.concat(final_loglikes_train_set)
    final_ave_train_loss = final_loglikes_train_set['logP_perSamp'].mean()
    with open(output_ave_file, 'w') as g:
        g.write(f'{args.eval_wkdir}\t')
        g.write(f'{final_ave_train_loss}\t')
    
    # save whole dataframe and remove from memory
    outfile = output_persamp_file.replace('[replace]','TRAIN-SPLIT')
    final_loglikes_train_set.to_csv(outfile, sep='\t')
    del final_loglikes_train_set
    
    # output the aux dict from the final batch, if debugging
    if DEBUG_FLAG:
        to_add = {'equl_vecs_from_train_data': hparams['equl_vecs_from_train_data']}
        aux_dict = {**to_add, **aux_dict}
        
        with open(f'{args.intermediates_folder}/TRAIN-SPLIT_intermediates.pkl', 'wb') as g:
            pickle.dump(aux_dict, g)
    
    # clean up variables
    del training_dl, training_dset, batch, batch_idx, rng_for_final_train


    ###########################################
    ### loop through test dataloader and      #
    ### score with best params                #
    ###########################################    
    # if this blows up the CPU, output in parts instead
    final_loglikes_test_set = []
       
    for batch_idx, batch in enumerate(test_dl):
        rng_for_final_test = jax.random.fold_in(rngkey, -(50000 + batch_idx + len(test_dl)))
        
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
                             pairHMM = pairHMM, 
                             params_dict = params, 
                             hparams_dict = hparams,
                             eval_rngkey = rng_for_final_test)
        
        aux_dict, _ = out
        del out
        
        # using batch_idx, generate the initial loss dataframe
        batch_out_df = test_dset.retrieve_sample_names(batch[-1])
        batch_out_df['logP_perSamp'] = np.array(aux_dict['logP_perSamp'])
        
        final_loglikes_test_set.append(batch_out_df)
    
    # concatenate values; save average losses
    final_loglikes_test_set = pd.concat(final_loglikes_test_set)
    final_ave_test_loss = final_loglikes_test_set['logP_perSamp'].mean()
    with open(output_ave_file, 'a') as g:
        g.write(f'{final_ave_test_loss}\n')
    
    # save whole dataframe and remove from memory
    outfile = output_persamp_file.replace('[replace]','TEST-SPLIT')
    final_loglikes_test_set.to_csv(outfile, sep='\t')
    del final_loglikes_test_set
    
    # output the aux dict from the final batch, if debugging
    if DEBUG_FLAG:
        to_add = {'equl_vecs_from_train_data': hparams['equl_vecs_from_train_data']}
        aux_dict = {**to_add, **aux_dict}
        
        with open(f'{args.intermediates_folder}/TEST-SPLIT_intermediates.pkl', 'wb') as g:
            pickle.dump(aux_dict, g)
