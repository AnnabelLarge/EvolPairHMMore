#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:18:49 2024

@author: annabel

ABOUT:
======
Load aligned pfams, train a mixture model, then pick which mixture is 
  most likely for each sample; similar to .fit_predict() from scikit learn

"""
import os
import shutil
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
from utils.eval_best_mixture import eval_best_mixture as eval_best_mixture




def fit_predict_pairhmm_mixture(args):
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
    # training data
    print(f'Creating DataLoader for training set with {args.train_dset_splits}')
    training_dset = hmm_reader(data_dir = args.data_dir, 
                               split_prefixes = args.train_dset_splits,
                               subsOnly = args.subsOnly)
    training_dl = DataLoader(training_dset, 
                             batch_size = args.batch_size, 
                             shuffle = True,
                             collate_fn = collator)
    training_global_max_seqlen = training_dset.max_seqlen()
    
    # same dataset, but don't shuffle for eval
    eval_dl = DataLoader(training_dset, 
                         batch_size = args.batch_size, 
                         shuffle = False,
                         collate_fn = collator)
    eval_global_max_seqlen = training_global_max_seqlen
    
    
    ### 1.3: quantize time in geometric spacing, just like in cherryML
    quantization_grid = range(-args.t_grid_num_steps, 
                              args.t_grid_num_steps + 1, 
                              1)
    t_array = jnp.array([(args.t_grid_center * args.t_grid_step**q_i) for q_i in quantization_grid])
    
    
    ### 1.4: column header for output eval depends on which probability is
    ###      being calculated
    if args.loss_type == 'conditional':
        eval_col_title = 'logP(A_t|A_0,model)'
    
    elif args.loss_type == 'joint':
        eval_col_title = 'logP(A_t,A_0|model)'
        
    
    
    ###########################
    ### 2: INITIALIZE MODEL   #
    ###########################
    ### 2.1: initialize the equlibrium distribution(s)
    equl_model_params, equl_model_hparams = equl_model.initialize_params(argparse_obj=args)
    equl_model_paramnames = equl_model_params.keys()
    
    # if this is the base model or the placeholder, use the equilibrium 
    #   distribution from TRAINING data
    if args.equl_model_type == 'equl_base':
        equl_model_hparams['equl_vecs_fromData'] = training_dset.retrieve_equil_dist()
    
    
    ### 2.2: initialize the substitution model
    subst_model_params, subst_model_hparams = subst_model.initialize_params(argparse_obj=args)
    subs_model_paramnames = subst_model_params.keys()
    
    
    ### 2.3: initialize the indel model
    indel_model_params, indel_model_hparams = indel_model.initialize_params(argparse_obj=args)
    indel_model_paramnames = indel_model_params.keys()
    
    
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
    
    
    ### 2.5: get all possible combinations of mixtures
    if 'subst_mix_logits' in params.keys():
        subs_idx = jnp.arange(0, params['subst_mix_logits'].shape[0])
    else:
        subs_idx = jnp.array([0])
    
    if 'equl_mix_logits' in params.keys():
        equl_idx = jnp.arange(0, params['equl_mix_logits'].shape[0])
    else:
        equl_idx = jnp.array([0])
    
    if 'indel_mix_logits' in params.keys():
        indel_idx = jnp.arange(0, params['indel_mix_logits'].shape[0])
    else:
        indel_idx = jnp.array([0])
    
    X,Y,Z = jnp.meshgrid(subs_idx, equl_idx, indel_idx)
    
    # indices is (num_combos, 3); add to hparams
    # col0 = subs; col1 = equl; col2 = indel
    indices = jnp.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    hparams['mixture_idx_matrix'] = indices
    del X,Y,Z, subs_idx, equl_idx, indel_idx

    
    ########################
    ### 3: TRAINING LOOP   #
    ########################
    ### 3.1: SETUP FOR TRAINING LOOP
    # jit your functions; there's an extra function if you need to 
    #   summarize the alignment
    jitted_train_fn = jax.jit(train_fn, static_argnames='loss_type')
    if not args.have_precalculated_counts:
        summarize_alignment_jitted = jax.jit(summarize_alignment, 
                                             static_argnames=['max_seq_len',
                                                              'alphabet_size',
                                                              'gap_tok',
                                                              'subsOnly'])

    # initialize optax
    tx = optax.adam(args.learning_rate)
    opt_state = tx.init(params)
    
    # quit training if test loss increases for X epochs in a row
    prev_train_loss = 9999
    early_stopping_counter = 0
    
    # when to save a model's parameters
    best_epoch = -1
    best_train_loss = 9999
    best_params = {}
    for epoch_idx in tqdm(range(args.num_epochs)):
        # top of the epoch, these aren't yet determined
        epoch_train_loss = 9999
        
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
            
            # take a step using minibatch gradient descent
            out = jitted_train_fn(all_counts = allCounts, 
                                  t_arr = t_array, 
                                  pairHMM = pairHMM, 
                                  params_dict = params, 
                                  hparams_dict = hparams,
                                  training_rngkey = rngkey_for_training,
                                  loss_type = args.loss_type)
            _, batch_train_sum_logP, param_grads = out
            del out

            
            # update the parameters dictionary with optax
            updates, opt_state = tx.update(param_grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # add to total loss for this epoch
            epoch_train_sum_logP += batch_train_sum_logP
            del batch_train_sum_logP
        
        
        ### 3.3: GET THE AVERAGE EPOCH TRAINING LOSS AND RECORD
        # aggregate with the equilvalent of -jnp.mean()
        epoch_train_loss = float( -( epoch_train_sum_logP/len(training_dset) ) )
        writer.add_scalar('Loss/training set', epoch_train_loss, epoch_idx)

        # if the training loss is nan, stop training
        if jnp.isnan(epoch_train_loss):
            with open(args.logfile_name,'a') as g:
                g.write(f'NaN training loss at epoch {epoch_idx}')
            raise ValueError(f'NaN training loss at epoch {epoch_idx}')
        
        # free up variables
        del batch, allCounts, epoch_train_sum_logP


        ### 3.4: IF THE TRAINING LOSS IS BETTER, SAVE MODEL WITH PARAMETERS
        if epoch_train_loss < best_train_loss:
            best_params = params
            best_epoch = epoch_idx
            best_train_loss = epoch_train_loss

            
            
        ### 3.5: EARLY STOPPING: if loss increases for X epochs in a row, 
        ###      stop training; reset counter if the loss decreases again 
        ###      (this is directly from Ian)
        if (jnp.allclose (prev_train_loss, 
                          jnp.minimum (prev_train_loss, epoch_train_loss), 
                          rtol=1e-05) ):
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        
        if early_stopping_counter == args.patience:
            with open(args.logfile_name,'a') as g:
                g.write(f'\n\nEARLY STOPPING AT EPOCH {epoch_idx}:\n')
                g.write(f'Best epoch: {best_epoch}\n')
                g.write(f'Best training loss: {best_train_loss}\n')
                
            # rage quit
            break
        
        # remember this epoch's loss for next iteration
        prev_train_loss = epoch_train_loss
    
    
    ### when you're done with the function, close the tensorboard writer
    writer.close()
    
    ### if early stopping was never triggered, record results at last epoch
    if early_stopping_counter != args.patience:
        with open(args.logfile_name,'a') as g:
            g.write(f'\n\nRegular stopping after {epoch_idx} full epochs:\n')
            g.write(f'Final training loss: {epoch_train_loss}\n')
    
    # make sure to not use an old version of params
    del params, epoch_idx
    
    
    ##############################
    ### 4: EVALUATE BEST MIXTURE #
    ##############################
    ### will run two different eval functions: the regular eval, and the per-mixture labeling
    jitted_eval_fn = jax.jit(eval_fn, static_argnames='loss_type')
    jitted_eval_best_mixture = jax.jit(eval_best_mixture, static_argnames='loss_type')

    ### 4.2: evaluate in batches
    # regular logP across all mixtures
    eval_df_lst = []

    # per mixture labeling
    eval_mixLabel_df_lst = []
    logP_per_model = []
    for batch_idx,batch in enumerate(eval_dl):
        ### 4.2.1 setup for both evals
        # batch_idx for eval (use negative value, to have distinctly 
        #   different random keys from training)
        rngkey_for_eval = jax.random.fold_in(rngkey, -batch_idx)
        
        # if you DON'T have precalculated counts matrices, will need to 
        #   clip the batch inputs
        if not args.have_precalculated_counts:
            batch_max_seqlen = clip_batch_inputs(batch, 
                                                  global_max_seqlen = eval_global_max_seqlen)
            allCounts = summarize_alignment_jitted(batch, 
                                            max_seq_len = batch_max_seqlen, 
                                            alphabet_size=hparams['alphabet_size'], 
                                            gap_tok=hparams['gap_tok'],
                                            subsOnly = args.subsOnly)
            del batch_max_seqlen
        
        # if you have these counts, just unpack the batch
        elif args.have_precalculated_counts:
            allCounts = (batch[0], batch[1], batch[2], batch[3])
        
        # get the batch sample labels, associated metadata for future dataframes
        eval_sample_idxes = batch[-1]
        label_df_forBatch = training_dset.retrieve_sample_names(eval_sample_idxes)
        

        ### 4.2.1 evaluate regular logP for each candidate model
        out = jitted_eval_fn(all_counts = allCounts, 
                                 t_arr = t_array, 
                                 pairHMM = pairHMM, 
                                 params_dict = best_params, 
                                 hparams_dict = hparams,
                                 eval_rngkey = rngkey_for_eval,
                                 loss_type = args.loss_type)
            
        _, _, logprob_per_sample = out
        del out

        # get the batch sample labels, associated metadata, and add loss terms
        logP_across_mixtures_forBatch = copy.deepcopy(label_df_forBatch)
        logP_across_mixtures_forBatch[eval_col_title] = logprob_per_sample
        
        eval_df_lst.append(logP_across_mixtures_forBatch)


        ### 4.2.2 label with mixtures
        raw_mat = jitted_eval_best_mixture(all_counts = allCounts, 
                                           t_arr = t_array, 
                                           pairHMM = pairHMM, 
                                           params_dict = best_params, 
                                           hparams_dict = hparams,
                                           eval_rngkey = rngkey_for_eval,
                                           loss_type = args.loss_type)
        logP_per_model.append(raw_mat)
        
        # max logP per sample
        mixLabeling_forBatch = copy.deepcopy(label_df_forBatch)
        max_per_sample = raw_mat.max(axis=1, keepdims=True)
        mixLabeling_forBatch['best_logP'] = max_per_sample
        
        # which mixtures have this logP (accounts for ties)
        maxes_per_row = jnp.repeat(max_per_sample, 
                                   raw_mat.shape[1], 
                                   axis=1)
        sample_labels = (raw_mat == maxes_per_row).astype(int)
        mini_df = pd.DataFrame(sample_labels,
                               columns = [f'Mix_{i}' for i in range(indices.shape[0])])
        mixLabeling_forBatch = mixLabeling_forBatch.join(mini_df)
        
        # save
        eval_mixLabel_df_lst.append(mixLabeling_forBatch)
    

    ##############################################
    ### 5: SAVE WHAT OVERALL MODEL PRODUCED THIS #
    ##############################################
    ### save the raw log probabilities
    os.mkdir(f'{os.getcwd()}/{args.training_wkdir}/MIXTURE_RESULTS')
    
    logP_per_model = jnp.concatenate(logP_per_model, axis=0)
    logP_outfile = (f'{os.getcwd()}/{args.training_wkdir}/MIXTURE_RESULTS/'+
                    f'{args.runname}_logprobs_per_mix.npy')
    with open(logP_outfile, 'wb') as g:
        jnp.save(g, logP_per_model)

    del logP_per_model, logP_outfile, g


    ### output all possible things needed to load a model later
    OUT_forLoad = {'subst_model_type': args.subst_model_type,
                    'equl_model_type': args.equl_model_type,
                    'indel_model_type': args.indel_model_type,
                    'norm': args.norm,
                    'alphabet_size': args.alphabet_size,
                    't_grid_center': args.t_grid_center,
                    't_grid_step': args.t_grid_step,
                    't_grid_num_steps': args.t_grid_num_steps,
                    'subsOnly': args.subsOnly,
                    'exch_files': args.exch_files
                    }
    
    if 'diffrax_params' in dir(args):
        OUT_forLoad['diffrax_params'] = args.diffrax_params
    
    # add (possibly transformed) parameters
    for key, val in best_params.items():
        if val.shape == (1,):
            OUT_forLoad[key] = val.item()
        else:
            OUT_forLoad[key] = np.array(val).tolist()
    
    # undo any possible parameter transformations and add to 
    #   1.) the dictionary of all possible things needed to load a 
    #   model, and 2.) a human-readable JSON of parameters
    OUT_params = {}
    for modelClass in pairHMM:
        params_toWrite = modelClass.undo_param_transform(best_params)
        OUT_forLoad = {**OUT_forLoad, **params_toWrite}
        OUT_params = {**OUT_params, **params_toWrite}

    OUT_params['epoch_of_training']= best_epoch
    
    # dump json files
    with open(f'{args.model_ckpts_dir}/toLoad.json', 'w') as g:
        json.dump(OUT_forLoad, g, indent="\t", sort_keys=True)
    del OUT_forLoad
    
    with open(f'{args.model_ckpts_dir}/params.json', 'w') as g:
        json.dump(OUT_params, g, indent="\t", sort_keys=True)
    del OUT_params
    
    ### save the regular logprobs
    eval_df = pd.concat(eval_df_lst)
    with open(f'./{args.training_wkdir}/{args.runname}_eval-set-logprobs.tsv','w') as g:
        g.write(f'#Logprobs using model params from epoch{best_epoch}\n')
        eval_df.to_csv(g, sep='\t')


    ### save the mixture labeling
    # just refer to model/params.json to figure out the actual model values
    # making a fancy output is taking too much time
    eval_mixLabel_df = pd.concat(eval_mixLabel_df_lst)
    eval_mixLabel_df_outfile = (f'{os.getcwd()}/{args.training_wkdir}/MIXTURE_RESULTS/'+
                       f'{args.runname}_sample_labeling.tsv')
    eval_mixLabel_df.to_csv(eval_mixLabel_df_outfile, sep='\t')
    del eval_mixLabel_df, eval_mixLabel_df_outfile
    
    
    ### output the key to the model checkpoint directory
    indices_df = pd.DataFrame(indices, columns=['subs_idx','equl_idx','indel_idx'])
    indices_df = indices_df.set_index('Mix_' + indices_df.index.astype(str))
    indices_df_outfile = (f'{args.model_ckpts_dir}/model_idx_key.tsv')
    with open(indices_df_outfile,'w') as g:
        g.write(f'# indices in this table correspond to 0-based indexing of each model\'s param array\n')
        g.write(f'# subs_idx = index for the parameter arrays associated with substitution mixtures\n')
        g.write(f'# equl_idx = index for the parameter arrays associated with equlibrium mixtures\n')
        g.write(f'# indel_idx = index for the parameter arrays associated with indel model mixtures\n')
        g.write(f'# example: indel_idx=0 means mixture uses indel_mix_probs[0], etc. \n\n')
    
        indices_df.to_csv(g, sep='\t')
    del indices_df, indices_df_outfile
    



    

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
    parser = argparse.ArgumentParser(prog='fit_predict_pairhmm_mixture')
    
    
    # config files required to run
    # parser.add_argument('--config-file',
    #                     type = str,
    #                     required=True,
    #                     help='Load configs from file in json format.')
    
   
    # parse the arguments
    args = parser.parse_args()
    args.config_file = 'example_config_fit_predict_multiple_mixtures.json'
    
    
    with open(args.config_file, 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    
    # fit and predict mixture label
    fit_predict_pairhmm_mixture(args)
