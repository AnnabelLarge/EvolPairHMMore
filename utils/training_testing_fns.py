#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:30:15 2024

@author: annabel


ABOUT:
======
training function and (related) eval function

einsum dimension abbreviations:
===============================
         time: t
        batch: b
alphabet_from: i
  alphabet_to: j
   trans_from: m
     trans_to: n
       k_subs: x
       k_equl: y
      k_indel: z

subCounts: (batch, alphabet_from, alphabet_to) = bij
insCounts:              (batch, alphabet_from) = bi
delCounts:              (batch, alphabet_from) = bi
transCounts:     (batch, trans_from, trans_to) = bmn

logP_sub: (alphabet_from, alphabet_to, k_subst, k_equl) = ijxy
logP_equl:                      (alphabet_from, k_equl) = iy
logP_indel:             (trans_from, trans_to, k_indel) = mnz


universal order of dimensions:
==============================
0. time
1. batch
2. k_subst
3. k_equl
4. k_indel

"""
import jax
from jax import numpy as jnp
from jax.nn import log_softmax
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



###############################################################################
### MAIN FUNCTIONS   ##########################################################
###############################################################################
def train_fn(all_counts, t_arr, pairHMM, params_dict, hparams_dict, 
             training_rngkey):
    """
    Jit-able function to find log-likelihood of both substitutions and indels, 
      and collect gradients
    
    inputs:
        > all_counts: precomputed counts, in this order- 
             subst, inserts, deleted chars, and transitions
        > t_arr: array of evolutionary times you're evaluating the likelihood
             at; sum them all together for final likelihood
        > pairHMM: tuple tying together equl_model, subst_model, and 
             indel_model (IN THAT ORDER)
        > params_dict: model parameters to update with optax
        > hparams_dict: hyperparams needed for run
        > training_rngkey: training rng key to add to hyperparameters 
          dictionary (may or may not be used)
    
    outputs:
        > loss: negative mean log likelihood
        > all_grads: gradients w.r.t. indel parameters (pass this to optax)
        
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
    hparams_dict['dirichlet_samp_key'] = jax.random.split(training_rngkey, 2)[1]
    
    # everything in this function is tracked with jax value_and_grad
    def apply_model(params_toTrack):
        ##############################################
        ### 1.1: logP(insertion) and logP(deletions) #
        ##############################################
        ### 1.1.1: retrieve equlibrium distribution vector; this does NOT 
        ###        depend on time
        equl_vecs, logprob_equl = equl_model.equlVec_logprobs(params_toTrack,
                                                              hparams_dict)
        
        # overwrite equl_vecs entry in the hparams dictionary
        hparams_dict['equl_vecs'] = equl_vecs
        
        ### 1.1.2: multiply counts by logprobs; this is logP 
        ###        PER EQUILIBRIUM DISTRIBUTION
        # logP(observed insertions); dim: (batch, k_equl)
        ins_logprobs_perMix = jnp.einsum('bi,iy->by',
                                         insCounts_persamp,
                                         logprob_equl)
        
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
            logprob_substitution_at_t = subst_model.joint_logprobs_at_t(t, 
                                                                  params_toTrack, 
                                                                  hparams_dict)
            
            # retrieve indel logprobs
            logprob_transition_at_t = indel_model.logprobs_at_t(t, 
                                                                params_toTrack, 
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
        
        
        ##################################
        ### 1.3: sum over mixture models #
        ##################################
        ### 1.3.1: log-softmax the mixture logits (if available)
        # get mixture logits from hparams OR pass in a dummy vector of 1
        # (k_subst,)
        subst_mix_logits = params_toTrack.get('subst_mix_logits', jnp.array([1]))
        
        # (k_equl,)
        equl_mix_logits = params_toTrack.get('equl_mix_logits', jnp.array([1]))
        
        # (k_indel,)
        indel_mix_logits = params_toTrack.get('indel_mix_logits', jnp.array([1]))
        
        # log-softmax these (if its a dummy vector, then this will be zero)
        subst_mix_logprobs = log_softmax(subst_mix_logits)
        equl_mix_logprobs = log_softmax(equl_mix_logits)
        indel_mix_logprobs = log_softmax(indel_mix_logits)
        
        
        ### 1.3.2: for substitution model, take care of substitution mixtures 
        ###        THEN equlibrium mixtures 
        # k_subs is at dim=2 (third out of three dimensions)
        with_subst_mix_weights = (sub_logprobs_perTime_perMix +
                                  jnp.expand_dims(subst_mix_logprobs, (0,1,3)))
        logsumexp_along_k_subst = logsumexp_withZeros(with_subst_mix_weights, axis=2)
        
        # k_equl is now at dim=2 (last one out of four dimensions, but middle 
        #   dimension was removed, so now it's last out of three dimensions)
        with_equl_mix_weights = (logsumexp_along_k_subst +
                                 jnp.expand_dims(equl_mix_logprobs, (0,1)))
        logP_subs_perTime = logsumexp_withZeros(with_equl_mix_weights, axis=2)
        
        
        ### 1.3.3: logP(emissions at insert sites) and 
        ###        logP(removed substrings from delete sites)
        # insertions
        with_ins_mix_weights = (ins_logprobs_perMix + 
                                jnp.expand_dims(equl_mix_logprobs, 0))
        logP_ins = logsumexp_withZeros(with_ins_mix_weights, axis=1)
        
        # deletions
        with_dels_mix_weights = (del_logprobs_perMix + 
                                 jnp.expand_dims(equl_mix_logprobs, 0))
        logP_dels = logsumexp_withZeros(with_dels_mix_weights, axis=1)
        
        
        ### 1.3.4: logP(transitions)
        # k_indel is at dim=2 (third out of three dimensions)
        with_indel_mix_weights = (trans_logprobs_perTime_perMix + 
                                  jnp.expand_dims(indel_mix_logprobs, (0,1)))
        logP_trans_perTime = logsumexp_withZeros(with_indel_mix_weights, 
                                                 axis=2)
        
        
        ##########################################
        ### 1.4: marginalize over mixture models #
        ##########################################
        ### 1.4.1: add all independent logprobs
        # (time, batch)
        logP_perTime = (logP_subs_perTime +
                        jnp.expand_dims(logP_ins,0) +
                        jnp.expand_dims(logP_dels,0) +
                        logP_trans_perTime)
        
        ### add marginalization_consts (time,)
        logP_perTime_withConst = (logP_perTime +
                                  jnp.expand_dims(marginalization_consts, 1))
        
        ### 1.4.3: logsumexp across time dimension (dim0)
        logP = logsumexp_withZeros(logP_perTime_withConst,
                                   axis=0)
        
        ### 1.4.4: final loss is -mean(logP) of this
        loss = -jnp.mean(logP)
        
        return loss
    
    
    ### set up the grad functions, based on above apply function
    ggi_grad_fn = jax.value_and_grad(apply_model, has_aux=False)
    
    # return loss and gradients
    loss, all_grads = ggi_grad_fn(params_dict)
    
    return loss, all_grads
    


def eval_fn(all_counts, t_arr, pairHMM, params_dict, hparams_dict, 
            eval_rngkey):
    """
    Jit-able function to find log-likelihood of both substitutions and indels
    
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
    
    outputs:
        > loss: negative mean log likelihood
        > logprobs_persamp: different logprobs for each sample
    """   
    # unpack counts tuple
    subCounts_persamp = all_counts[0] 
    insCounts_persamp = all_counts[1] 
    delCounts_persamp = all_counts[2]
    transCounts_persamp = all_counts[3]
    breakpoint()
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
    
    ### 1.1.2: multiply counts by logprobs; this is logP 
    ###        PER EQUILIBRIUM DISTRIBUTION
    # logP(observed insertions); dim: (batch, k_equl)
    ins_logprobs_perMix = jnp.einsum('bi,iy->by',
                                     insCounts_persamp,
                                     logprob_equl)
    
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
        logprob_substitution_at_t = subst_model.joint_logprobs_at_t(t, 
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
    
    
    ##################################
    ### 1.3: sum over mixture models #
    ##################################
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
    
    
    ### 1.3.2: for substitution model, take care of substitution mixtures 
    ###        THEN equlibrium mixtures 
    # k_subs is at dim=2 (third out of three dimensions)
    with_subst_mix_weights = (sub_logprobs_perTime_perMix +
                              jnp.expand_dims(subst_mix_logprobs, (0,1,3)))
    logsumexp_along_k_subst = logsumexp_withZeros(with_subst_mix_weights, axis=2)
    
    # k_equl is now at dim=2 (last one out of four dimensions, but middle 
    #   dimension was removed, so now it's last out of three dimensions)
    with_equl_mix_weights = (logsumexp_along_k_subst +
                             jnp.expand_dims(equl_mix_logprobs, (0,1)))
    logP_subs_perTime = logsumexp_withZeros(with_equl_mix_weights, axis=2)
    
    
    ### 1.3.3: logP(emissions at insert sites) and 
    ###        logP(removed substrings from delete sites)
    # insertions
    with_ins_mix_weights = (ins_logprobs_perMix + 
                            jnp.expand_dims(equl_mix_logprobs, 0))
    logP_ins = logsumexp_withZeros(with_ins_mix_weights, axis=1)
    
    # deletions
    with_dels_mix_weights = (del_logprobs_perMix + 
                             jnp.expand_dims(equl_mix_logprobs, 0))
    logP_dels = logsumexp_withZeros(with_dels_mix_weights, axis=1)
    
    
    ### 1.3.4: logP(transitions)
    # k_indel is at dim=2 (third out of three dimensions)
    with_indel_mix_weights = (trans_logprobs_perTime_perMix + 
                              jnp.expand_dims(indel_mix_logprobs, (0,1)))
    logP_trans_perTime = logsumexp_withZeros(with_indel_mix_weights, 
                                             axis=2)
    
    
    ################################
    ### 1.4: marginalize over time #
    ################################
    ### 1.4.1: add all independent logprobs
    # (time, batch)
    logP_perTime = (logP_subs_perTime +
                    jnp.expand_dims(logP_ins,0) +
                    jnp.expand_dims(logP_dels,0) +
                    logP_trans_perTime)
    
    ### add marginalization_consts (time, batch)
    logP_perTime_withConst = (logP_perTime +
                              jnp.expand_dims(marginalization_consts, 1))
    
    ### 1.4.3: logsumexp across time dimension (dim0)
    logP_perSamp = logsumexp_withZeros(logP_perTime_withConst,
                                       axis=0)
    
    ### 1.4.4: final loss is -mean(logP_perSamp) of this
    loss = -jnp.mean(logP_perSamp)
    
    
    return(loss, logP_perSamp)
