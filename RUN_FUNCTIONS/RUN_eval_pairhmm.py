#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:18:49 2024

@author: annabel

ABOUT:
======
Load aligned pfams and model params, optionally calculate emission and transition 
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
from utils.setup_utils import model_import_register
from utils.training_testing_fns import eval_fn
from utils.init_dataloaders import init_dataloaders





def eval_pairhmm(args, dataloader_tup):
    ###########################################################################
    ###  0: FOLDER SETUP, MODEL/PARAM/HPARAM IMPORTS    #######################
    ###########################################################################
    print('0: checking config, loading params')
    
    # create the eval working directory, if it doesn't exist
    if args.eval_wkdir not in os.listdir():
        os.mkdir(args.eval_wkdir)
        
    # locate everything needed to load a model; add to main argparse object
    training_dir = f'{args.training_wkdir}/model_ckpts/'
    
    with open(f'{training_dir}/forLoad_argparse.pkl','rb') as f:
        training_argparse_obj = pickle.load(f)
    
    with open(f'{training_dir}/params_dict.pkl','rb') as f:
        loaded_params = pickle.load(f)
    
    # if not provided, assume not in debug mode
    if 'debug' not in vars(args):
        DEBUG_FLAG = False
    else:
        DEBUG_FLAG = args.debug

    
    # previous version of code allowed the option for lazy dataloading, but
    #   since GGI model is so small, just get rid of that
    assert args.loadtype == 'eager'
    
    # load models and possibly register as pytrees
    out = model_import_register(training_argparse_obj)
    subst_model, equl_model, indel_model, _ = out
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
    
        
    ###########################################################################
    ### 1: SETUP   ############################################################
    ###########################################################################
    print(f'1: setup')
    # output file for individual results per sample
    output_persamp_file = f'{args.eval_wkdir}/LOGPROB-PER-SAMP.tsv'
    output_ave_file = f'{args.eval_wkdir}/AVE-LOGPROB.tsv'
    
    # rng key
    rngkey = jax.random.key(args.rng_seednum)
    
    if DEBUG_FLAG:
        # if debugging, set up an intermediates folder
        folder_path = f'{os.getcwd()}/{args.eval_wkdir}/HMM_INTERMEDIATES'
        
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        args.intermediates_folder = subfolder_path
        del folder_path
    
    
    ### use the helper function to import/initialize dataloaders
    # from "dataloader_lst = init_dataloaders(args)"
    test_dset, test_dl = dataloader_lst
    del dataloader_tup
    
    test_global_max_seqlen = test_dset.max_seqlen()
    
    
    ### quantize time in geometric spacing, just like in cherryML
    quantization_grid = range(-(training_argparse_obj.t_grid_num_steps-1), 
                              training_argparse_obj.t_grid_num_steps, 
                              1)
    t_array = jnp.array([(training_argparse_obj.t_grid_center * training_argparse_obj.t_grid_step**q_i) for q_i in quantization_grid])
    
    
    ### column header for output eval depends on which probability is
    ###   being calculated
    if args.loss_type == 'conditional':
        eval_col_title = 'logP(A_t|A_0,model)'
    
    elif args.loss_type == 'joint':
        eval_col_title = 'logP(A_t,A_0|model)'
        
    
    
    ###########################################################################
    ### 2: INITIALIZE HYPERPARAMS (BUT KEEP LOADED PARAMS)   ##################
    ###########################################################################
    print('2: hyperparam init')
    ### initialize the equlibrium distribution(s)
    _, equl_model_hparams = equl_model.initialize_params(argparse_obj=training_argparse_obj)
    
    # if this is the base model, use the equilibrium distribution from 
    #   TRAINING data
    if training_argparse_obj.equl_model_type == 'equl_base':
        equl_model_hparams['equl_vecs_from_train_data'] = test_dset.retrieve_equil_dist()
    
    # if you're not scoring emissions from indels at all, use this placeholder
    elif training_argparse_obj.equl_model_type == 'no_equl':
        equl_model_hparams['equl_vecs_from_train_data'] = jnp.zeros((training_argparse_obj.alphabet_size))
    
    
    ### initialize the substitution model
    _, subst_model_hparams = subst_model.initialize_params(argparse_obj=training_argparse_obj)
    
    
    ### initialize the indel model
    _, indel_model_hparams = indel_model.initialize_params(argparse_obj=training_argparse_obj)
    
    
    ## combine all hyperparameters to be passed to eval function 
    hparams = {**equl_model_hparams, **subst_model_hparams, **indel_model_hparams}
    
    # if it hasn't already been specified in the JSON file, set the gap_tok
    #   to default value of 43; this is only used for calculating counts
    if 'gap_tok' not in dir(training_argparse_obj):
        hparams['gap_tok'] = 43
    else:
        hparams['gap_tok'] = training_argparse_obj.gap_tok
    
    # add grid step to hparams dictionary; needed for marginaling over time
    hparams['t_grid_step']= training_argparse_obj.t_grid_step
    
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
                             DEBUG_FLAG = DEBUG_FLAG)
    jitted_eval_fn = jax.jit(parted_eval_fn, 
                             static_argnames=['loss_type',
                                              'DEBUG_FLAG'])
    
    if not args.have_precalculated_counts:
        parted_summary_fn = partial(summarize_alignment,
                                    alphabet_size = hparams['alphabet_size'],
                                    gap_tok = hparams['gap_tok'],
                                    subsOnly = args.subsOnly)
        
        summarize_alignment_jitted = jax.jit(parted_summary_fn, 
                                             static_argnames=['max_seq_len'])

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
                             params_dict = loaded_params, 
                             hparams_dict = hparams,
                             eval_rngkey = rng_for_final_test)
        
        aux_dict, _ = out
        del out
        
        # using batch_idx, generate the initial loss dataframe
        batch_out_df = test_dset.retrieve_sample_names(batch[-1])
        batch_out_df['logP_perSamp'] = np.array(aux_dict['logP_perSamp'])
        batch_out_df['logP_perSamp_length_normed'] = (batch_out_df['logP_perSamp'] / 
                                                      batch_out_df['desc_seq_len'])
        
        final_loglikes_test_set.append(batch_out_df)
    
    # concatenate values; save average losses
    final_loglikes_test_set = pd.concat(final_loglikes_test_set)
    final_ave_test_loss = final_loglikes_test_set['logP_perSamp'].mean()
    final_ave_test_loss_seqlen_normed = final_loglikes_test_set['logP_perSamp_length_normed'].mean()
    with open(output_ave_file, 'w') as g:
        g.write(f'{args.eval_wkdir}\t')
        g.write(f'{final_ave_test_loss}\t')
        g.write(f'{final_ave_test_loss_seqlen_normed}\n')
    
    # save whole dataframe and remove from memory
    final_loglikes_test_set.to_csv(output_persamp_file, sep='\t')
    del final_loglikes_test_set
    
    # output the aux dict from the final batch, if debugging
    if DEBUG_FLAG:
        to_add = {'equl_vecs_from_train_data': hparams['equl_vecs_from_train_data']}
        aux_dict = {**to_add, **aux_dict}
        
        with open(f'{args.intermediates_folder}/EVAL_intermediates.pkl', 'wb') as g:
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
    parser = argparse.ArgumentParser(prog='eval_pairhmm')
    
    
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
    dataloader_lst = init_dataloaders(args, onlyTest=True)
    
    # run training function
    eval_pairhmm(args, dataloader_lst)
