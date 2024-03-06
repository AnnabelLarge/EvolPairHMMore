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
Immediate:
----------
- implement a mixture model version (possibly in a different script...)
- make the logprob symmetrical 

far future:
-----------
For now, using LG08 exchangeability matrix, but in the future, could use 
  CherryML to calculate a new rate matrix for my specific pfam dataset?
  https://github.com/songlab-cal/CherryML

Implement automatic parameter initialization at some point?

"""
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import jax
from jax import numpy as jnp
import optax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# custom imports
from utils.setup_training_dir import setup_training_dir
from GGI_funcs.rates_transition_matrices import lg_rate_mat




def eval_ggi(args):
    #################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES #
    #################################################
    # previous version of code allowed the option for lazy dataloading, but
    #   since GGI model is so small, just get rid of that
    assert args.loadtype == 'eager'
        
    # Use this training mode if you already have precalculated count matrices
    if args.have_precalculated_counts:
        logfile_msg = 'Reading from precalculated counts matrices before training\n'
        from onlyTrain.training_testing_fns import train_fn, eval_fn
        from onlyTrain.hmm_dataset import HMMDset_PC as hmm_reader
        from onlyTrain.hmm_dataset import jax_collator as collator
        
    # Use this training mode if you need to calculate emission and transition counts 
    #   from pair alignments
    elif not args.have_precalculated_counts:
        logfile_msg = 'Calculating counts matrices from alignments, then training\n'
        from calcCounts_Train.training_testing_fns import train_fn, eval_fn
        from calcCounts_Train.hmm_dataset import HMMDset as hmm_reader
        from calcCounts_Train.hmm_dataset import jax_collator as collator
        
        # Later, clip the alignments to one of four possible alignment lengths, 
        #   thus jit-compiling four versions of train_fn and eval_fn 
        #   (saves time by not having to calculate counts for excess 
        #   padding tokens)
        def clip_batch_inputs(batch):
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
                return 2324 # the sequence-wide max length
        
        
    ##############
    ### 1: SETUP #
    ##############
    ### 1.1: read model 
    with open(args.model_params_file, 'r') as f:
        contents = [line.strip() for line in f.readlines()]

    lam = float(contents[0].split('\t')[-1])
    mu = float(contents[1].split('\t')[-1])
    x = float(contents[2].split('\t')[-1])
    y = float(contents[3].split('\t')[-1])
        
    
    ### 1.2: read data to eval on; build pytorch dataloaders
    print(f'Creating DataLoader for test set with {args.test_dset_splits}')
    test_dset = hmm_reader(data_dir = args.data_dir, 
                              split_prefixes = args.test_dset_splits)
    test_dl = DataLoader(test_dset, 
                         batch_size = args.batch_size, 
                         shuffle = False,
                         collate_fn = collator)
    
    
    ### 1.3: get a rate matrix; {rate}_ij = {exchange_mat}_ij * pi_i where i != j
    # load this dataset's equilibrium distribution
    equl_pi_mat = test_dset.retrieve_equil_dist()
    equl_pi_mat = jnp.array(equl_pi_mat)
    
    # get the R matrix
    subst_rate_mat = lg_rate_mat(equl_pi_mat,
                                 f'./{args.data_dir}/LG08_exchangeability_r.npy')
    
    # normalize the R matrix by the equilibrium vector, if desired
    if args.norm:
        R_times_pi = -np.diagonal(subst_rate_mat) @ equl_pi_mat
        subst_rate_mat = subst_rate_mat / R_times_pi
    
    
    ### 1.4: quantize time in geometric spacing, just like in cherryML
    quantization_grid = range(-args.t_grid_num_steps, 
                              args.t_grid_num_steps + 1, 
                              1)
    t_array = jnp.array([(args.t_grid_center * args.t_grid_step**q_i) for q_i in quantization_grid])
    
    
    
    ####################
    ### 2: EVAL LOOP   #
    ####################
    ### 2.1: transform the initial indel params from the config file to lie on 
    ### domain (-inf, inf)
    ### originally needed this for training... just keep the same signature for
    ### for now
    lam_transf = jnp.sqrt(lam)
    mu_transf = jnp.sqrt(mu)
    x_transf = jnp.sqrt(-jnp.log(x))
    y_transf = jnp.sqrt(-jnp.log(y))
    model_params = jnp.array([lam_transf, mu_transf, x_transf, y_transf])
    
    # replace any zero arguments with smallest_float32
    smallest_float32 = jnp.finfo('float32').smallest_normal
    model_params = jnp.where(model_params == 0, 
                             smallest_float32, 
                             model_params)
    
    # jit your functions
    eval_fn_jitted = jax.jit(eval_fn, static_argnames='max_seq_len')
    
    ### 2.2: Start eval loop
    eval_df_lst = []
    final_eval_loss = 0
    for batch in test_dl:
        # if you DON'T have precalculated counts matrices, will need to 
        #   clip the batch inputs; otherwise, set this to None
        if not args.have_precalculated_counts:
            batch_max_seqlen = clip_batch_inputs(batch)
        else:
            batch_max_seqlen = None
        
        # evaluate batch loss
        out = eval_fn_jitted(data = batch, 
                             t_arr = t_array, 
                             subst_rate_mat = subst_rate_mat, 
                             equl_pi_mat = equl_pi_mat,
                             indel_params_transformed = model_params,
                             diffrax_params = args.diffrax_params,
                             max_seq_len = batch_max_seqlen)
        batch_eval_loss, logprob_per_sample = out
        del out
        
        final_eval_loss += batch_eval_loss
        del batch_eval_loss
        
        # record the log losses per sample
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
    ave_eval_loss = float(final_eval_loss/len(test_dl))
    del final_eval_loss, batch, batch_max_seqlen
    
    # output the metadata + losses dataframe, along with final eval loss
    eval_df = pd.concat(eval_df_lst)
    with open(f'./{args.eval_wkdir}/{args.runname}_FINAL_EVAL_RESULTS.tsv','w') as g:
        g.write(f'#Final Eval Loss: {ave_eval_loss}\n')
        eval_df.to_csv(g, sep='\t')

    




##########################################
### BASIC CLI+JSON CONFIG IMPLEMENTATION #
##########################################
if __name__ == '__main__':
    import json
    import argparse 
    
    # for now, running models on single GPU
    err_ms = 'SELECT GPU TO RUN THIS COMPUTATION ON with CUDA_VISIBLE_DEVICES=DEVICE_NUM'
    assert len(jax.devices()) == 1, err_ms
    del err_ms
    
    # INITIALIZE PARSER
    parser = argparse.ArgumentParser(prog='GGI')
    
    # config files required to run
    parser.add_argument('--config_file',
                        type = str,
                        required=True,
                        help='Load configs from file in json format.')
    
    # parse the arguments
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    
    # run training function
    eval_ggi(args)
