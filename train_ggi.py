#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:18:49 2024

@author: annabel

ABOUT:
======
Load aligned pfams, optionally calculate emission and transition 
  counts from pair alignments, and fit a simple GGI model


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
pd.options.mode.chained_assignment = None # this is annoying
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




def train_ggi(args):
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
    # can inspect this faster than a tensorboard thing
    with open(args.logfile_name,'w') as g:
        g.write(logfile_msg)
        g.write('TRAINING PROG:\n')
    
    # setup tensorboard writer
        writer = SummaryWriter(args.tboard_dir)
    
    
    ### 1.2: read data; build pytorch dataloaders 
    # 1.2.1: training data
    print(f'Creating DataLoader for training set with {args.train_dset_splits}')
    training_dset = hmm_reader(data_dir = args.data_dir, 
                               split_prefixes = args.train_dset_splits)
    training_dl = DataLoader(training_dset, 
                             batch_size = args.batch_size, 
                             shuffle = True,
                             collate_fn = collator)
    
    # 1.2.2: test data
    print(f'Creating DataLoader for test set with {args.test_dset_splits}')
    test_dset = hmm_reader(data_dir = args.data_dir, 
                              split_prefixes = args.test_dset_splits)
    test_dl = DataLoader(test_dset, 
                         batch_size = args.batch_size, 
                         shuffle = False,
                         collate_fn = collator)
    
    
    ### 1.3: using the equilibrium distribution from the TRAINING SET, get a
    ###      rate matrix; {rate}_ij = {exchange_mat}_ij * pi_i where i != j
    # load this dataset's equilibrium distribution
    equl_pi_mat = training_dset.retrieve_equil_dist()
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
    
    
    
    ########################
    ### 2: TRAINING LOOP   #
    ########################
    ### 2.1: transform the initial indel params from the config file to lie on 
    ### domain (-inf, inf)
    lam_transf = jnp.sqrt(args.lam)
    mu_transf = jnp.sqrt(args.mu)
    x_transf = jnp.sqrt(-jnp.log(args.x))
    y_transf = jnp.sqrt(-jnp.log(args.y))
    indel_params_transformed = jnp.array([lam_transf, mu_transf, x_transf, y_transf])
    
    # replace any zero arguments with smallest_float32
    smallest_float32 = jnp.finfo('float32').smallest_normal
    indel_params_transformed = jnp.where(indel_params_transformed == 0, 
                                         smallest_float32, 
                                         indel_params_transformed)
    
    # initialize optax
    tx = optax.adam(args.learning_rate)
    opt_state = tx.init(indel_params_transformed)
    
    # jit your functions
    train_fn_jitted = jax.jit(train_fn, static_argnames='max_seq_len')
    eval_fn_jitted = jax.jit(eval_fn, static_argnames='max_seq_len')
    
    # quit training if test loss don't significantly change for X epochs
    prev_test_loss = 9999
    early_stopping_counter = 0
    
    # when to save a model's parameters
    best_epoch = -1
    best_train_loss = 9999
    
    for epoch_idx in tqdm(range(args.num_epochs)):
        # default behavior is to not save model parameters or 
        #   eval set log likelihoods
        record_results = False
        
        ### 2.2: training phase
        epoch_train_loss = 0
        for batch_idx, batch in enumerate(training_dl):
            # if you DON'T have precalculated counts matrices, will need to 
            #   clip the batch inputs; otherwise, set this to None
            if not args.have_precalculated_counts:
                batch_max_seqlen = clip_batch_inputs(batch)
            else:
                batch_max_seqlen = None
            
            # take a step using minibatch gradient descent
            out = train_fn_jitted(data = batch, 
                                  t_arr = t_array, 
                                  subst_rate_mat = subst_rate_mat, 
                                  equl_pi_mat = equl_pi_mat,
                                  indel_params_transformed = indel_params_transformed, 
                                  diffrax_params = args.diffrax_params, 
                                  max_seq_len = batch_max_seqlen)
            batch_train_loss, indel_param_grads = out
            del out
            
            # update the parameters with optax
            updates, opt_state = tx.update(indel_param_grads, opt_state)
            indel_params_transformed = optax.apply_updates(indel_params_transformed, 
                                                           updates)
            
            
            # monitor how params change over batches of training
            idx_for_tboard = (batch_idx) + (epoch_idx) * len(training_dl)
            lam_transf, mu_transf, x_transf, y_transf = indel_params_transformed
            lam_tosave = np.array(jnp.square(lam_transf))
            mu_tosave = np.array(jnp.square(mu_transf))
            x_tosave = np.array(jnp.exp(-jnp.square(x_transf)))
            y_tosave = np.array(jnp.exp(-jnp.square(y_transf)))
            writer.add_scalar('Params/lambda (indel param)', lam_tosave, idx_for_tboard)
            writer.add_scalar('Params/mu (indel param)', mu_tosave, idx_for_tboard)
            writer.add_scalar('Params/x (indel param)', x_tosave, idx_for_tboard)
            writer.add_scalar('Params/y (indel param)', y_tosave, idx_for_tboard)
            del lam_transf, mu_transf, x_transf, y_transf
            del lam_tosave, mu_tosave, x_tosave, y_tosave


#            # monitor how gradients update over batches of training
#            writer.add_scalar('Grads/d(lambda)', np.array(indel_param_grads[0]), idx_for_tboard)
#            writer.add_scalar('Grads/d(mu)', np.array(indel_param_grads[1]), idx_for_tboard)
#            writer.add_scalar('Grads/d(x)', np.array(indel_param_grads[2]), idx_for_tboard)
#            writer.add_scalar('Grads/d(y)', np.array(indel_param_grads[3]), idx_for_tboard)
            del batch_idx, idx_for_tboard

    
            # add to total loss for this epoch
            epoch_train_loss += batch_train_loss
            del batch_train_loss
        
        
        ### 2.3: get the average epoch_train_loss; record
        ave_epoch_train_loss = float(epoch_train_loss/len(training_dl))
        writer.add_scalar('Loss/training set', ave_epoch_train_loss, epoch_idx)

        # if the training loss is nan, stop training
        if jnp.isnan(ave_epoch_train_loss):
            print(f'NaN training loss at epoch {epoch_idx}')
            with open(args.logfile_name,'a') as g:
                g.write(f'NaN training loss at epoch {epoch_idx}')
            break
        
        # free up variables
        del batch, epoch_train_loss, batch_max_seqlen


        ### 2.4: if the train loss is better than last epoch, save the 
        ###   parameters to a dictionary
        if ave_epoch_train_loss < best_train_loss:
            # swap the flag
            record_results = True
            
            # unpack; undo the transformation
            lam_transf, mu_transf, x_transf, y_transf = indel_params_transformed
            lam_tosave = jnp.square(lam_transf)
            mu_tosave = jnp.square(mu_transf)
            x_tosave = jnp.exp(-jnp.square(x_transf))
            y_tosave = jnp.exp(-jnp.square(y_transf))
            
            # build dictionary
            indel_params = {'lam': lam_tosave,
                            'mu': mu_tosave,
                            'x': x_tosave,
                            'y': y_tosave}
            
            # save dictionary to flat text file (could pickle it later...)
            with open(f'{args.model_ckpts_dir}/indelparams.txt', 'w') as g:
                for key, val in indel_params.items():
                    g.write(f'{key}\t{val}\n')
            
            # record to log file
            with open(args.logfile_name,'a') as g:
                g.write(f'New best training loss at epoch {epoch_idx}: {ave_epoch_train_loss}\n')
            
            # update save criteria
            best_train_loss = ave_epoch_train_loss
    
        
        ### 2.5: also check current performance on held-out test set, and write
        ###      this to the tensorboard
        # only comes into play if you want to record the results i.e. record_results=True
        eval_df_lst = []
        epoch_test_loss = 0
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
                                 indel_params_transformed = indel_params_transformed,
                                 diffrax_params = args.diffrax_params,
                                 max_seq_len = batch_max_seqlen)
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
        del epoch_test_loss, batch, batch_max_seqlen

        # output the metadata + losses dataframe, along with what epoch 
        #   you're recording results; place this outside of folders
        if record_results:
            eval_df = pd.concat(eval_df_lst)
            with open(f'./{args.training_wkdir}/{args.runname}_eval-set-logprobs.tsv','w') as g:
                g.write(f'#Logprobs using GGI indel params from epoch{epoch_idx}\n')
                eval_df.to_csv(g, sep='\t')

        ### 2.6: EARLY STOPPING
        ### if test loss hasn't changed after X epochs, stop training
        if (jnp.abs(ave_epoch_test_loss - prev_test_loss) < 1e-4).all():
            early_stopping_counter += 1
        
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
    train_ggi(args)
