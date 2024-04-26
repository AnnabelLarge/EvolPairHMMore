#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:56:47 2024

@author: annabel_large
"""
import jax
from jax import numpy as jnp
from jax.nn import log_softmax
from unitTests.generate_fake_inputs import fake_input
from utils.extra_utils import logsumexp_where


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


def eval_best_mixture(all_counts, t_arr, pairHMM, params_dict, hparams_dict, 
                      eval_rngkey, loss_type, indices):
    """
    After training a mixture model, determine which mixture is most optimal
        for each sample; akin to .transform() function from scipy.stats
    
    inputs:
        > all_counts: precomputed counts, in this order- 
              subst, inserts, deleted chars, and transitions
        > t_arr: array of evolutionary times you're evaluating the likelihood
              at; sum them all together for final likelihood
        > pairHMM: tuple tying together equl_model, subst_model, and 
              indel_model (IN THAT ORDER)
        > params_dict: model parameters to update with optax
        > hparams_dict: hyperparams needed for run
        > eval_rngkey: rng key for eval (may or may not be needed)
        > loss_type: either "conditional" or "joint"
        > indices = index matrix from jnp.meshgrid
    
    outputs:
        > loss: negative mean log likelihood
        > logprobs_persamp: different logprobs for each sample
    """   
    # unpack counts tuple
    subCounts_persamp = all_counts[0] 
    insCounts_persamp = all_counts[1] 
    delCounts_persamp = all_counts[2]
    transCounts_persamp = all_counts[3]
    del all_counts
    
    # unpack model tuple
    equl_model, subst_model, indel_model = pairHMM
    del pairHMM
    
    # need r, the geometric grid step for generating timepoints
    r = hparams_dict['t_grid_step']
    
    # if equlibrium distribution is a dirichlet mixture, will need a random
    # key for it
    hparams_dict['dirichlet_samp_key'] = jax.random.split(eval_rngkey, 2)[1]
    
    
    ##############################################
    ### 1.1: logP(insertion) and logP(deletions) #
    ##############################################
    ### 1.1.1: retrieve equlibrium distribution vector; this does NOT 
    ###        depend on time
    equl_vecs, logprob_equl = equl_model.equlVec_logprobs(params_dict,
                                                          hparams_dict)
    
    # overwrite equl_vecs entry in the hparams dictionary
    hparams_dict['equl_vecs'] = equl_vecs
    hparams_dict['logP_equl'] = logprob_equl
    
    ### 1.1.2: multiply counts by logprobs; this is logP 
    ###        PER EQUILIBRIUM DISTRIBUTION
    # logP(observed insertions); dim: (batch, k_equl)
    ins_logprobs_perMix = jnp.einsum('bi,iy->by',
                                     insCounts_persamp,
                                     logprob_equl)
    
    if loss_type == 'joint':
        # logP(deletions from ancestor sequence); dim: (batch, k_equl)
        del_logprobs_perMix = jnp.einsum('bi,iy->by',
                                         delCounts_persamp,
                                         logprob_equl)
    
    
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
        
        ### 1.2.2: multiply counts by logprobs
        # logP(observed substitutions)
        logP_counts_sub_at_t = jnp.einsum('bij,ijxy->bxy', 
                                          subCounts_persamp,
                                          logprob_substitution_at_t)
        
        # logP(observed transitions)
        logP_counts_trans_at_t = jnp.einsum('bmn,mnz->bz',
                                            transCounts_persamp,
                                            logprob_transition_at_t)
        
        
        ### 1.2.3: also need a per-time constant for later 
        ###        marginalization over time
        marg_const_at_t = jnp.log(t) - ( t / (r-1) ) 
        
        
        return (logP_counts_sub_at_t, logP_counts_trans_at_t,
                marg_const_at_t)
    
    ### 1.2.3: vmap over the time array; return log probabilities 
    ###        PER TIME POINT and PER MIXTURE MODEL
    vmapped_apply_model_at_t = jax.vmap(apply_model_at_t)
    out = vmapped_apply_model_at_t(t_arr)
    
    # (time, batch, k_subst, k_equl)
    sub_logprobs_perTime_perMix = out[0]
    
    # (time, batch, k_indel)
    trans_logprobs_perTime_perMix = out[1]
    
    # (time,)
    marginalization_consts = out[2]
    
    
    
    #################################################################
    ### 1.3: calculate logP over different combinations of mixtures #
    #################################################################
    ### 1.3.1: log-softmax the mixture logits (if available)
    # get mixture logits from hparams OR pass in a dummy vector of 1
    # (k_subst,)
    subst_mix_logits = params_dict.get('subst_mix_logits', jnp.array([1]))
    
    # (k_equl,)
    equl_mix_logits = params_dict.get('equl_mix_logits', jnp.array([1]))
    
    # (k_indel,)
    indel_mix_logits = params_dict.get('indel_mix_logits', jnp.array([1]))
    
    # log-softmax these (if its a dummy vector, then this will be zero)
    subst_mix_logprobs = log_softmax(subst_mix_logits)
    equl_mix_logprobs = log_softmax(equl_mix_logits)
    indel_mix_logprobs = log_softmax(indel_mix_logits)
    
    
    ### 1.3.2: evaluate logprob for each specific combinations
    def logprob_one_model(idx_vec):
        s, e, i = idx_vec
        
        # for subs: logP(x|k_subs, k_equl) + logP(k_subs) + logP(k_equl) 
        #           = logP(x, k_subs, k_equl)
        # (time, batch)
        logP_subs_perTime = (sub_logprobs_perTime_perMix[:,:,s,e] + 
                             subst_mix_logprobs[s] + 
                             equl_mix_logprobs[e])
        
        # for ins and del
        # both are (batch,)
        logP_ins = ins_logprobs_perMix[:,e] + equl_mix_logprobs[e]
        
        if loss_type == 'joint':
            logP_dels = del_logprobs_perMix[:,e] + equl_mix_logprobs[e]
        
        
        # for transitions (indel model)
        # (time, batch)
        logP_trans_perTime = trans_logprobs_perTime_perMix[:,:,i]
        
        # sum then marginalize over time
        logP_perTime = (logP_subs_perTime +
                        jnp.expand_dims(logP_ins,0) +
                        logP_trans_perTime)
        
        if loss_type == 'joint':
            logP_perTime = logP_perTime + jnp.expand_dims(logP_dels,0)
        
        # add marginalization_consts (time, batch)
        logP_perTime_withConst = (logP_perTime +
                                  jnp.expand_dims(marginalization_consts, 1))
        
        # 1.4.3: logsumexp across time dimension (dim0)
        logP_perSamp = logsumexp_withZeros(logP_perTime_withConst,
                                            axis=0)
        
        return logP_perSamp
    
    vmapped_logprob_one_model = jax.vmap(logprob_one_model, out_axes=1)
    
    # (batch, num_combos)
    logP_per_model = vmapped_logprob_one_model(indices)
    return logP_per_model
    
