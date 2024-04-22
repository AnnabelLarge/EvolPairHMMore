#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:18:49 2024

@author: annabel

ABOUT:
======
Load aligned pfams (and optionally calculate emission and transition 
  counts from pair alignment inputs), load GGI indel parameters, and 
  eval only


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
import optax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# custom imports
from utils.setup_utils import model_import_register
from utils.training_testing_fns import eval_fn


def eval_pairhmm(args):
    ###################################################
    ### 0: FOLDER SETUP, MODEL/PARAM/HPARAM IMPORTS   #
    ###################################################
    ### 0.1: MODEL/PARAM LOADING, WKDIR SETUP
    # create the eval working directory, if it doesn't exist
    if args.eval_wkdir not in os.listdir():
        os.mkdir(args.eval_wkdir)
    
    # locate everything needed to load a model; add to main argparse object
    training_dir = f'{args.training_wkdir}/model_ckpts/{args.training_runname}'
    toLoad_file = f'{training_dir}/toLoad.json'
    
    with open(toLoad_file, 'r') as f:
        args.__dict__.update(json.load(f))
    del f, toLoad_file
    
    # load models and possibly register as pytrees
    out = model_import_register(args)
    subst_model, equl_model, indel_model, _ = out
    del out
    
    
    ### 0.2: DECIDE TRAINING MODE
    # Use this training mode if you already have precalculated count matrices
    if args.have_precalculated_counts:
        to_add = ('Reading from precalculated counts matrices before'+
                  ' training\n\n')
        from onlyTrain.hmm_dataset import HMMDset_PC as hmm_reader
        from onlyTrain.hmm_dataset import jax_collator as collator
        
    # Use this training mode if you need to calculate emission and transition  
    #   counts from pair alignments
    elif not args.have_precalculated_counts:
        to_add = ('Calculating counts matrices from alignments, then'+
                  ' training\n\n')
        from calcCounts_Train.hmm_dataset import HMMDset as hmm_reader
        from calcCounts_Train.hmm_dataset import jax_collator as collator
        from calcCounts_Train.summarize_alignment import summarize_alignment
        
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
    
    
    ##############
    ### 1: SETUP #
    ##############
    ### 1.1: output files, rng key (if needed)
    # rng key
    rngkey = jax.random.key(args.rng_seednum)
    
    # output file for individual results per sample
    output_persamp_file = f'{args.eval_wkdir}/{args.eval_runname}_LOGPROB-PER-SAMP.tsv'
    output_ave_file = f'{args.eval_wkdir}/{args.eval_runname}_AVE-LOGPROB.tsv'
    
    
    ### 1.2: read data; build pytorch dataloaders
    print(f'Creating DataLoader for test set with {args.test_dset_splits}')
    test_dset = hmm_reader(data_dir = args.data_dir, 
                              split_prefixes = args.test_dset_splits,
                              subsOnly = args.subsOnly)
    test_dl = DataLoader(test_dset, 
                         batch_size = args.batch_size, 
                         shuffle = False,
                         collate_fn = collator)
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
    if args.equl_model_type in ['equl_base']:
        equl_model_hparams['equl_vecs_fromData'] = test_dset.retrieve_equil_dist()
    
    
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
    
    # add grid step to hparams dictionary; needed for marginaling over time
    hparams['t_grid_step']= args.t_grid_step
    
    # combine models under one pairHMM
    pairHMM = (equl_model, subst_model, indel_model)
    
    
    ####################
    ### 3: EVAL LOOP   #
    ####################
    # jit your functions; there's an extra function if you need to 
    #   summarize the alignment
    jitted_eval_fn = jax.jit(eval_fn)
    if not args.have_precalculated_counts:
        summarize_alignment_jitted = jax.jit(summarize_alignment, 
                                             static_argnames=['max_seq_len',
                                                              'alphabet_size',
                                                              'gap_tok',
                                                              'subsOnly'])
    
    eval_df_lst = []
    eval_test_loss = 0
    for batch_idx,batch in enumerate(test_dl):
        ### 3.1: EVAL ON ALL SAMPLES IN THE BATCH
        # fold in epoch_idx and batch_idx for eval 
        rngkey_for_eval = jax.random.fold_in(rngkey, batch_idx)
    
        # if you DON'T have precalculated counts matrices, will need to 
        #   clip the batch inputs
        if not args.have_precalculated_counts:
            batch_max_seqlen = clip_batch_inputs(batch, 
                                                  global_max_seqlen = test_global_max_seqlen)
            allCounts = summarize_alignment(batch, 
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
                              eval_rngkey = rngkey_for_eval)
    
        batch_test_loss, logprob_per_sample = out
        
        # add to eval_test_loss
        eval_test_loss += batch_test_loss
        
    
        ### 3.2: RECORD RESULTS IN DATAFRAME
        # get the batch sample labels, associated metadata
        eval_sample_idxes = batch[-1]
        meta_df_forBatch = test_dset.retrieve_sample_names(eval_sample_idxes)
        
        # add loss terms
        meta_df_forBatch['logP(A_t,A_0|model)'] = logprob_per_sample
                
        eval_df_lst.append(meta_df_forBatch)
    
    
    # 3.3: COMBINE DATAFRAMES ACROSS BATCHES
    eval_df = pd.concat(eval_df_lst)
    with open(output_persamp_file,'w') as g:
        eval_df.to_csv(g, sep='\t')
    
    # also output averge loss to a row of a file; can concat from all runs later
    ave_epoch_test_loss = float(eval_test_loss/len(test_dl))
    del eval_test_loss
    
    with open(output_ave_file, 'w') as g:
        g.write(f'{args.eval_runname}\t') 
        g.write(f'{ave_epoch_test_loss}\n')



if __name__ == '__main__':
    import json
    import argparse 

    # for now, running models on single GPU
    err_ms = 'SELECT GPU TO RUN THIS COMPUTATION ON with CUDA_VISIBLE_DEVICES=DEVICE_NUM'
    assert len(jax.devices()) == 1, err_ms
    del err_ms

    ### INITIALIZE PARSER
    parser = argparse.ArgumentParser(prog='train_pairhmm')

    # config files required to run
    parser.add_argument('--config-file',
                        type = str,
                        required=True,
                        help='Load configs from file in json format.')

    # parse the arguments
    args = parser.parse_args()


    ### load the config
    with open(args.config_file, 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    # run evaluation function
    eval_pairhmm(args)
    