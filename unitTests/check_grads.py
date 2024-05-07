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


TODO:
=====
medium:
-------
- remove the option to calculate counts on the fly, and just make this a 
  separate pre-processing script (I don't ever use it...)


far future:
-----------
For now, using LG08 exchangeability matrix, but in the future, could use 
  CherryML to calculate a new exchangeability matrix for my specific pfam 
  dataset? https://github.com/songlab-cal/CherryML

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




def check_grads(args):
    #################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES #
    #################################################
    # previous version of code allowed the option for lazy dataloading, but
    #   since GGI model is so small, just get rid of that
    assert args.loadtype == 'eager'
    
    ### 0.1: DECIDE MODEL PARTS TO IMPORT, REGISTER AS PYTREES
    out = model_import_register(args)
    subst_model, equl_model, indel_model, logfile_msg = out
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
    
    logfile_msg = logfile_msg + to_add
    del to_add
    
        
    ##############
    ### 1: SETUP #
    ##############
    ### 1.1: rng key, folder setup, etc.
    # setup folders; manually create model checkpoint directory (i.e. what 
    #   orbax would normally do for you)
    # setup_training_dir(args)
    # if not os.path.exists(f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'):
    #     os.mkdir(f'{os.getcwd()}/{args.training_wkdir}/model_ckpts')
    # os.mkdir(args.model_ckpts_dir)
    
    # rng key
    rngkey = jax.random.key(args.rng_seednum)
    
    # # create a new logfile to record training loss in an ascii file
    # # can eyeball this faster than a tensorboard thing
    # with open(args.logfile_name,'w') as g:
    #     g.write(logfile_msg)
    #     g.write('TRAINING PROG:\n')
    
    # # setup tensorboard writer
    #     writer = SummaryWriter(args.tboard_dir)
    
    
    ### 1.2: read data; build pytorch dataloaders 
    # 1.2.1: training data
    print(f'Creating DataLoader for training set with {args.train_dset_splits}')
    training_dset = hmm_reader(data_dir = args.data_dir, 
                               split_prefixes = args.train_dset_splits,
                               subsOnly = args.subsOnly)
    training_dl = DataLoader(training_dset, 
                             batch_size = args.batch_size, 
                             shuffle = True,
                             collate_fn = collator)
    training_global_max_seqlen = training_dset.max_seqlen()
    
    # 1.2.2: test data
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
    
    
    # ### 1.4: column header for output eval depends on which probability is
    # ###      being calculated
    # if args.loss_type == 'conditional':
    #     eval_col_title = 'logP(A_t|A_0,model)'
    
    # elif args.loss_type == 'joint':
    #     eval_col_title = 'logP(A_t,A_0|model)'
        
    
    
    ###########################
    ### 2: INITIALIZE MODEL   #
    ###########################
    ### 2.1: initialize the equlibrium distribution(s)
    equl_model_params, equl_model_hparams = equl_model.initialize_params(argparse_obj=args)
    
    # if this is the base model or the placeholder, use the equilibrium 
    #   distribution from TRAINING data
    if args.equl_model_type == 'equl_base':
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
    
    # add grid step to hparams dictionary; needed for marginaling over time
    hparams['t_grid_step']= args.t_grid_step
    
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
    jitted_train_fn = jax.jit(train_fn, static_argnames='loss_type')
    # jitted_eval_fn = jax.jit(eval_fn, static_argnames='loss_type')
    if not args.have_precalculated_counts:
        summarize_alignment_jitted = jax.jit(summarize_alignment, 
                                             static_argnames=['max_seq_len',
                                                              'alphabet_size',
                                                              'gap_tok',
                                                              'subsOnly'])
        
    joint_lam_grad = []
    joint_x_grad = []
    cond_lam_grad = []
    cond_x_grad = []
    
    for epoch_idx in tqdm(range(args.num_epochs)):
        # top of the epoch, these aren't yet determined
        epoch_train_loss = 9999
        epoch_test_loss = 9999
        
        # default behavior is to not save model parameters or 
        #   eval set log likelihoods
        record_results = False
        
        ### 3.2: TRAINING PHASE
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
            
            ### GRADIENTS UNDER JOINT LOSS
            joint_out = jitted_train_fn(all_counts = allCounts, 
                                  t_arr = t_array, 
                                  pairHMM = pairHMM, 
                                  params_dict = params, 
                                  hparams_dict = hparams,
                                  training_rngkey = rngkey_for_training,
                                  loss_type = "joint")
            _, _, joint_param_grads = joint_out
            
            ### GRADIENTS UNDER CONDITIONAL LOSS
            cond_out = jitted_train_fn(all_counts = allCounts, 
                                  t_arr = t_array, 
                                  pairHMM = pairHMM, 
                                  params_dict = params, 
                                  hparams_dict = hparams,
                                  training_rngkey = rngkey_for_training,
                                  loss_type = "conditional")
            _, _, cond_param_grads = cond_out
            
            # record gradients
            joint_lam_grad.append(joint_param_grads['lam_transf'].item())
            joint_x_grad.append(joint_param_grads['x_transf'].item())
            cond_lam_grad.append(cond_param_grads['lam_transf'].item())
            cond_x_grad.append(cond_param_grads['x_transf'].item())
            
            # update the parameters dictionary with optax
            # update with conditional param gradients, for now
            updates, opt_state = tx.update(cond_param_grads, opt_state)
            params = optax.apply_updates(params, updates)
            

    df = pd.DataFrame({'joint_lam_grad':joint_lam_grad,
                       'cond_lam_grad':cond_lam_grad,
                       'joint_x_grad':joint_x_grad,
                       'cond_x_grad':cond_x_grad})

    return df

    
    

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
    # parser.add_argument('--config-file',
    #                     type = str,
    #                     required=True,
    #                     help='Load configs from file in json format.')
    
   
    # parse the arguments
    args = parser.parse_args()
    args.config_file = 'unitTests/req_files/CONFIG_singleModels.json'
    
    
    # this is specifically made for GGI-WT
    # assert 'GGI-WT' in args.config_file
    
    
    with open(args.config_file, 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    
    # run training function
    grads = check_grads(args)
    
    # see if gradients match
    # seems like this one doesn't match as well? Is that a bad thing?
    print('Lam_transf gradients match?')
    print((np.abs(grads['joint_lam_grad'] - grads['cond_lam_grad']) < 1e-2).all())
    print()
    
    print('x_transf gradients match?')
    print((np.abs(grads['joint_x_grad'] - grads['cond_x_grad']) < 1e-4).all())
    print()
    
    
    grads.to_csv('GRADIENTS_{args.config_file.replace(".json",".tsv")}', sep='\t')
    
    
    
