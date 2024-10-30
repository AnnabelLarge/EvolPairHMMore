#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:26:05 2024

@author: annabel

ABOUT:
======
manually step through loglikelihood calculation and check numerical stability,
  einsum recipes, etc.

note: individual models have already been unit-tested

Batch size = 2 (using the first two samples from FiveSamp)
T_array = 5

k_equl = 1
k_indel = 1
k_subst = 1

"""
import os
import pickle
import pandas as pd
import json
import argparse 
pd.options.mode.chained_assignment = None # this is annoying
import numpy as np
import copy
from tqdm import tqdm
import json

import jax
from jax import numpy as jnp
from jax.nn import log_softmax
import optax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from utils.extra_utils import logsumexp_where
from utils.setup_utils import model_import_register


###############################################################################
### HELPER FUNCTIONS   ########################################################
###############################################################################
def logsumexp_withZeros(x, axis):
    """
    wrapper that returns zero if WHOLE logsumexp would result in zero 
      (native behavior is to return -inf)
    """
    # mask for tensor elements that are zero
    nonzero_elems = jnp.where(x != 0,
                              1,
                              0)
    
    # use logsumexp_where only where whole sum would not be zero; otherwise,
    #  return zero
    #  note: if there's any weird point where gradient is -inf or inf, it's
    #  probably this causing a problem...
    out = jnp.where(nonzero_elems.sum(axis=axis) > 0,
                    logsumexp_where(a=x,
                                    axis=axis,
                                    where=nonzero_elems),
                    0)
    return out


def get_mats(t_arr, pairHMM, params_dict, hparams_dict, loss_type):
    # unpack model tuple
    equl_model, subst_model, indel_model = pairHMM
    del pairHMM
    
    # need r, the geometric grid step for generating timepoints
    r = hparams_dict['t_grid_step']
    
    ##############################################
    ### 1.1: logP(insertion) and logP(deletions) #
    ##############################################
    ### 1.1.1: retrieve equlibrium distribution vector; this does NOT 
    ###        depend on time
    equl_vecs, logprob_equl = equl_model.equlVec_logprobs(params_dict,
                                                          hparams_dict)
    
    # overwrite equl_vecs entry in the hparams dictionary
    hparams_dict['equl_vecs'] = equl_vecs
    
    
    ##################################################################
    ### 1.2: logP(substitutions/matches) and logP(state transitions) #
    ##################################################################
    # this function calculates logprob at one time, t; vmap this over t_arr
    def apply_model_at_t(t):
        ### 1.2.1: get the emission/transition matrices
        # retrieve substitution logprobs
        if loss_type == 'joint':
            logprob_substitution_at_t = subst_model.joint_logprobs_at_t(t, 
                                                                  params_dict, 
                                                                  hparams_dict)
        elif loss_type == 'conditional':
            logprob_substitution_at_t = subst_model.conditional_logprobs_at_t(t, 
                                                                  params_dict, 
                                                                  hparams_dict)
            
        
        # retrieve indel logprobs
        logprob_transition_at_t = indel_model.logprobs_at_t(t, 
                                                            params_dict, 
                                                            hparams_dict)
        
        
        ### 1.2.3: also need a per-time constant for later 
        ###        marginalization over time
        marg_const_at_t = jnp.log(t) - ( t / (r-1) ) 
        
        
        return (logprob_substitution_at_t, 
                logprob_transition_at_t,
                marg_const_at_t)
    
    ### 1.2.3: vmap over the time array; return log probabilities 
    ###        PER TIME POINT and PER MIXTURE MODEL
    vmapped_apply_model_at_t = jax.vmap(apply_model_at_t)
    out = vmapped_apply_model_at_t(t_arr)
    
    # (time, batch, k_subst, k_equl)
    logprob_substitution = out[0]
    
    # (time, batch, k_indel)
    logprob_transition = out[1]
    
    # (time,)
    marginalization_consts = out[2]
    
    ### matrices in dict
    out_dict = {'log(Pi)': logprob_equl,
                'log(exp(Rt))': logprob_substitution,
                'GGI_T': logprob_transition,
                'marg_consts': marginalization_consts}
    
    return out_dict


def calc_logprob(loss_type):
    #####################
    # INITIALIZE PARSER #
    #####################
    parser = argparse.ArgumentParser(prog='train_pairhmm')
    args = parser.parse_args()
    args.config_file = './unitTests/req_files/CONFIG_singleModels.json'
    
    
    with open(args.config_file, 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    
    args.loss_type = loss_type
    del f, parser, t_args
    
    
    #################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES #
    #################################################
    # previous version of code allowed the option for lazy dataloading, but
    #   since GGI model is so small, just get rid of that
    
    ### 0.1: DECIDE MODEL PARTS TO IMPORT, REGISTER AS PYTREES
    out = model_import_register(args)
    subst_model, equl_model, indel_model, _ = out
    del out
    
    ### 0.2: DECIDE TRAINING MODE
    # Use this training mode if you need to calculate emission and transition  
    #   counts from pair alignments
    if not args.have_precalculated_counts:
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
    ### 1.2: read data; build pytorch dataloaders, and get batch
    print(f'Creating DataLoader for test set with {args.test_dset_splits}')
    test_dset = hmm_reader(data_dir = args.data_dir, 
                           split_prefixes = args.test_dset_splits,
                           subsOnly = args.subsOnly)
    test_dl = DataLoader(test_dset, 
                         batch_size = args.batch_size, 
                         shuffle = False,
                         collate_fn = collator)
    test_global_max_seqlen = test_dset.max_seqlen()
    batch = list(test_dl)[0]
    
    
    ### 1.3: quantize time in geometric spacing, just like in cherryML
    # in debug mode, only have three timepoints
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
        equl_model_hparams['equl_vecs_from_train_data'] = test_dset.retrieve_equil_dist()
        equl_model_hparams['equl_vecs'] = test_dset.retrieve_equil_dist()
        equl_model_hparams['logP_equl'] = jnp.log(test_dset.retrieve_equil_dist())
    
    
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
    
    
    ### REMOVE unneeded variables
    # for clean variable explorer
    del equl_model_params, subst_model_params, indel_model_params
    del equl_model_hparams, subst_model_hparams, indel_model_hparams
    del quantization_grid
    
    ############################
    ### 3: calculation setup   #
    ############################
    batch_max_seqlen = clip_batch_inputs(batch, 
                                         global_max_seqlen = test_global_max_seqlen)
    allCounts = summarize_alignment(batch, 
                                    max_seq_len = batch_max_seqlen, 
                                    alphabet_size=hparams['alphabet_size'], 
                                    gap_tok=hparams['gap_tok'],
                                    subsOnly = args.subsOnly)
    del batch_max_seqlen
    
    # get transition/emission matrices
    # these have already been unit tested and I'm confident they work
    mat_dict = get_mats(t_arr=t_array, 
                        pairHMM=pairHMM, 
                        params_dict=params, 
                        hparams_dict=hparams,
                        loss_type=args.loss_type)
    
    
    
    
    
    ###########################################################################
    ### 4: TEST EVAL FUNCTION MANUALLY BELOW ##################################
    ###########################################################################
    # unpack counts tuple
    subCounts_persamp = allCounts[0] 
    insCounts_persamp = allCounts[1] 
    delCounts_persamp = allCounts[2]
    transCounts_persamp = allCounts[3]
    del allCounts
    
    # unpack model tuple
    equl_model, subst_model, indel_model = pairHMM
    del pairHMM
    
    # need r, the geometric grid step for generating timepoints
    r = hparams['t_grid_step']
    
    
    ##############################################
    ### 4.1: logP(insertion) and logP(deletions) #
    ##############################################
    ### 4.1.1: retrieve equlibrium distribution vector; this does NOT 
    ###        depend on time
    logprob_equl = mat_dict['log(Pi)']
    
    ### 4.1.2: multiply counts by logprobs; this is logP 
    ###        PER EQUILIBRIUM DISTRIBUTION
    # logP(observed insertions); dim: (batch, k_equl)
    ins_logprobs_perMix = jnp.einsum('bi,iy->by',
                                     insCounts_persamp,
                                     logprob_equl)
    
    # sanity check with a manual non-einsum version
    checkvals = []
    for b in range(insCounts_persamp.shape[0]):
        counts_vec = insCounts_persamp[b, :]
        true_logprob = 0
        
        for i in range(counts_vec.shape[0]):
            count = counts_vec[i]
            logP = logprob_equl[i,0]
            true_logprob += count * logP
        
        checkvals.append(true_logprob.item())
    
    checkvals = jnp.array(checkvals)
    
    assert jnp.allclose(ins_logprobs_perMix[:,0], checkvals)
    del checkvals, b, counts_vec, true_logprob, i, count, logP
    
    if loss_type == 'joint':
        # logP(deletions from ancestor sequence); dim: (batch, k_equl)
        # uses the same einsum recipe, so already been verified
        del_logprobs_perMix = jnp.einsum('bi,iy->by',
                                         delCounts_persamp,
                                         logprob_equl)
    
    
    ###########################################
    ### 4.2: logP(transitions) and logP(subs) #
    ###########################################
    sub_logprobs_perTime_perMix = []
    trans_logprobs_perTime_perMix = []
    for t in range(len(t_array)):
        logprob_substitution_at_t = mat_dict['log(exp(Rt))'][t,:,:,:,:]
        logprob_transition_at_t = mat_dict['GGI_T'][t,:,:,:]
        
        ### logP(observed substitutions)
        logP_counts_sub_at_t = jnp.einsum('bij,ijxy->bxy', 
                                          subCounts_persamp,
                                          logprob_substitution_at_t)
        
        # sanity check this
        checkvals = []
        for b in range(subCounts_persamp.shape[0]):
            true_logprob = 0
            for i in range(subCounts_persamp.shape[1]):
                for j in range(subCounts_persamp.shape[2]):
                    count = subCounts_persamp[b,i,j]
                    logprob = logprob_substitution_at_t[i,j,0,0]
                    true_logprob += count * logprob
            checkvals.append(true_logprob.item())
        checkvals = jnp.array(checkvals)
        
        assert jnp.allclose(checkvals, logP_counts_sub_at_t[:,0,0])
        del checkvals, b, true_logprob, i, j, count, logprob
        
        
        ### logP(observed transitions)
        logP_counts_trans_at_t = jnp.einsum('bmn,mnz->bz',
                                            transCounts_persamp,
                                            logprob_transition_at_t)
        
        # sanity check this
        checkvals = []
        for b in range(transCounts_persamp.shape[0]):
            true_logprob = 0
            for m in range(transCounts_persamp.shape[1]):
                for n in range(transCounts_persamp.shape[2]):
                    count = transCounts_persamp[b,m,n]
                    logprob = logprob_transition_at_t[m,n,0]
                    true_logprob += count * logprob
            checkvals.append(true_logprob.item())
        checkvals = jnp.array(checkvals)
        
        assert jnp.allclose(checkvals, logP_counts_trans_at_t[:,0])
        del checkvals, b, true_logprob, m, n, count, logprob
        
    
        ### add to list arrays
        sub_logprobs_perTime_perMix.append(jnp.expand_dims(logP_counts_sub_at_t,0))
        trans_logprobs_perTime_perMix.append(jnp.expand_dims(logP_counts_trans_at_t,0))
        
        del logprob_substitution_at_t, logprob_transition_at_t
        del logP_counts_sub_at_t, logP_counts_trans_at_t
    
    # concatenate
    sub_logprobs_perTime_perMix = jnp.concatenate(sub_logprobs_perTime_perMix, 0)
    trans_logprobs_perTime_perMix = jnp.concatenate(trans_logprobs_perTime_perMix, 0)
    
    
    ##################################
    ### 4.3: sum over mixture models #
    ##################################
    # there's no mixture models, so this whole section should just produce 
    #   the same vectors as above
    
    ### 4.3.1: log-softmax the mixture logits (if available)
    # get mixture logits from hparams OR pass in a dummy vector of 1
    # (k_subst,)
    subst_mix_logits = params.get('subst_mix_logits', jnp.array([1]))
    
    # (k_equl,)
    equl_mix_logits = params.get('equl_mix_logits', jnp.array([1]))
    
    # (k_indel,)
    indel_mix_logits = params.get('indel_mix_logits', jnp.array([1]))
    
    # log-softmax these (if its a dummy vector, then this will be zero)
    subst_mix_logprobs = log_softmax(subst_mix_logits)
    equl_mix_logprobs = log_softmax(equl_mix_logits)
    indel_mix_logprobs = log_softmax(indel_mix_logits)
    
    
    ### 4.3.2: for substitution model, take care of substitution mixtures 
    ###        THEN equlibrium mixtures 
    # k_subs is at dim=2 (third out of three dimensions)
    with_subst_mix_weights = (sub_logprobs_perTime_perMix +
                              jnp.expand_dims(subst_mix_logprobs, (0,1,3)))
    logsumexp_along_k_subst = logsumexp_withZeros(x=with_subst_mix_weights, 
                                                  axis=2)
    
    # k_equl is now at dim=2 (last one out of four dimensions, but middle 
    #   dimension was removed, so now it's last out of three dimensions)
    with_equl_mix_weights = (logsumexp_along_k_subst +
                             jnp.expand_dims(equl_mix_logprobs, (0,1)))
    logP_subs_perTime = logsumexp_withZeros(x=with_equl_mix_weights, 
                                            axis=2)
    
    assert jnp.allclose(logP_subs_perTime, sub_logprobs_perTime_perMix[:,:,0,0])
    del subst_mix_logits, subst_mix_logprobs, with_subst_mix_weights
    del logsumexp_along_k_subst, with_equl_mix_weights
    
    
    ### 4.3.3: logP(emissions at insert sites) and 
    ###        logP(removed substrings from delete sites)
    # insertions
    with_ins_mix_weights = (ins_logprobs_perMix + 
                            jnp.expand_dims(equl_mix_logprobs, 0))
    logP_ins = logsumexp_withZeros(x=with_ins_mix_weights, 
                                   axis=1)
    
    assert jnp.allclose(logP_ins, ins_logprobs_perMix[:,0])
    del with_ins_mix_weights
    
    if loss_type == 'joint':
        # deletions
        with_dels_mix_weights = (del_logprobs_perMix + 
                                 jnp.expand_dims(equl_mix_logprobs, 0))
        logP_dels = logsumexp_withZeros(x=with_dels_mix_weights, 
                                        axis=1)
        
        assert jnp.allclose(logP_dels, del_logprobs_perMix[:,0])
        del with_dels_mix_weights
    
    
    ### 4.3.4: logP(transitions)
    # k_indel is at dim=2 (third out of three dimensions)
    with_indel_mix_weights = (trans_logprobs_perTime_perMix + 
                              jnp.expand_dims(indel_mix_logprobs, (0,1)))
    logP_trans_perTime = logsumexp_withZeros(x=with_indel_mix_weights, 
                                             axis=2)
    
    assert jnp.allclose(logP_trans_perTime, trans_logprobs_perTime_perMix[:,:,0])
    del with_indel_mix_weights
    
    
    ################################
    ### 4.4: marginalize over time #
    ################################
    marginalization_consts = mat_dict['marg_consts']
    
    ### 4.4.1: add all independent logprobs
    # (time, batch)
    logP_perTime = (logP_subs_perTime +
                    jnp.expand_dims(logP_ins,0) +
                    logP_trans_perTime)
    
    if loss_type == 'joint':
        logP_perTime = logP_perTime + jnp.expand_dims(logP_dels,0)
    
    # manual inspection of these element-wise sums show that jax numpy
    #   broadcasting is working ok
    
    # add marginalization_consts (time,)
    logP_perTime_withConst = (logP_perTime +
                              jnp.expand_dims(marginalization_consts, 1))
    
    # make sure sum is applied to correct dimension (axis=0)
    checksum = []
    for t_point in range(len(marginalization_consts)):
        row_to_add = logP_perTime[t_point,:]+ marginalization_consts[t_point]
        row_to_add = jnp.expand_dims(row_to_add, 0)
        checksum.append(row_to_add)
    checksum = jnp.concatenate(checksum, axis=0)
    
    assert jnp.allclose( logP_perTime_withConst, checksum )
    del checksum

    
    ### 4.4.3: logsumexp across time dimension (dim0)
    logP_perSamp = logsumexp_withZeros(x=logP_perTime_withConst,
                                       axis=0)
    
    
    ### 4.4.4: final loss is -mean(logP_perSamp) of this
    loss = -jnp.mean(logP_perSamp)


def main():
    calc_logprob(loss_type='conditional')
    print('[SUB TEST PASSED] conditional logprob as expected')
    print()
    
    calc_logprob(loss_type='joint')
    print('[SUB TEST PASSED] joint logprob as expected')
    print()
    
    
if __name__ == '__main__':
    main()
    