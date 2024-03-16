#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:30:15 2024

@author: annabel


ABOUT:
======


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

# We replace zeroes and infinities with small numbers sometimes
# It's sinful but BOY DO I LOVE SIN
smallest_float32 = jnp.finfo('float32').smallest_normal


###############################################################################
### HELPER FUNCTIONS   ########################################################
###############################################################################
def logsumexp_withZeros(x, axis):
    """
    helper function that's basically a logsumexp that isn't affected by zeros
    if the whole sum( exp(x) ) is zero, then also return zero (not log(0))
    """
    zero_val_mask = jnp.where(x != 0,
                              1,
                              0)
    
    exp_x = jnp.exp(x)
    exp_x_masked = exp_x * zero_val_mask
    sumexp_x = jnp.sum(exp_x_masked, axis=axis)
    
    logsumexp_x = jnp.where(sumexp_x != 0,
                            jnp.log(sumexp_x),
                            0)
    return logsumexp_x


# use this to logsumexp across timepoints (includes normalization by 1/N)
def marginalize_across_t(logprob_mat, num_timepoints, norm_axis=0):
    """
    P(A) = sum_t[P(A|t)P(t)]
    logP(A) = logsumexp(logP(A|t) + logP(t))
    
    P(t) = 1/num_timepoints
    """
    ### normalize with logP(t) = log(1/N)
    logprob_mat_normed = logprob_mat + jnp.log(1/num_timepoints)
    
    ### logsumexp
    logsumexp_probs = logsumexp_withZeros(logprob_mat_normed, 
                                          axis=norm_axis)
    return logsumexp_probs



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
    
    # if equlibrium distribution is a dirichlet mixture, will need a random
    # key for it
    hparams_dict['dirichlet_samp_key'] = jax.random.split(training_rngkey, 2)[1]
    
    # everything in this function is tracked with jax value_and_grad
    def apply_model(params_toTrack):
        #########################################################
        ### 1: CALCULATE LOG-PROBABILITIES PER t, PER MIXTURE   #
        #########################################################
        # retrieve equlibrium distribution vector, logP(emissions/omissions)
        #   this does NOT depend on time, so do this outside the time loop
        equl_vecs, logprob_equl = equl_model.equlVec_logprobs(params_toTrack,
                                                              hparams_dict)
        
        # overwrite equl_vecs entry in the hparams dictionary
        hparams_dict['equl_vecs'] = equl_vecs
        
        # this calculates logprob at one time, t; vmap this over t_arr
        def apply_model_at_t(t):
            ######################################
            ### 1.1: LOG PROBABILITIES AT TIME t #
            ######################################
            # retrieve substitution logprobs
            logprob_substitution_at_t = subst_model.logprobs_at_t(t, 
                                                                  params_toTrack, 
                                                                  hparams_dict)
            
            # retrieve indel logprobs
            logprob_transition_at_t = indel_model.logprobs_at_t(t, 
                                                                params_toTrack, 
                                                                hparams_dict)
            
            
            ###############################################################
            ### 1.2: multiply counts by logprobs (let einsum handle this) #
            ###############################################################
            # logP(observed substitutions)
            logP_counts_sub = jnp.einsum('bij,ijxy->bxy', 
                                         subCounts_persamp,
                                         logprob_substitution_at_t)
            
            # logP(observed insertions)
            logP_counts_ins = jnp.einsum('bi,iy->by',
                                         insCounts_persamp,
                                         logprob_equl)
            
            # logP(omitted deletions)
            logP_counts_dels = jnp.einsum('bi,iy->by',
                                          delCounts_persamp,
                                          logprob_equl)
            
            # logP(observed transitions)
            logP_counts_trans = jnp.einsum('bmn,mnz->bz',
                                           transCounts_persamp,
                                           logprob_transition_at_t)
            
            # return tuple of these four to deal with... later
            return (logP_counts_sub, logP_counts_ins, 
                    logP_counts_dels, logP_counts_trans)
        
        ### vmap over the time array; return log probabilities PER TIME POINT,
        ###   and PER MIXTURE MODEL
        vmapped_apply_model_at_t = jax.vmap(apply_model_at_t)
        tuple_logprobs_perTime_perMix = vmapped_apply_model_at_t(t_arr)
        num_timepoints = len(t_arr)
        
        ### unpack tuples; all times will be placed at dim0
        # (time, batch, k_subst, k_equl)
        sub_logprobs_perTime_perMix = tuple_logprobs_perTime_perMix[0]
        
        # (time, batch, k_equl)
        ins_logprobs_perTime_perMix = tuple_logprobs_perTime_perMix[1]
        
        # (time, batch, k_equl)
        del_logprobs_perTime_perMix = tuple_logprobs_perTime_perMix[2]
        
        # (time, batch, k_indel)
        trans_logprobs_perTime_perMix = tuple_logprobs_perTime_perMix[3]
        
        
        ################################################
        ### 2: SUM ACROSS TIME ARRAY, MIXTURE MODELS   #
        ################################################
        ### 2.1: marginalize across t
        # (batch, k_subst, k_equl)
        sub_logprobs_perMix = marginalize_across_t(sub_logprobs_perTime_perMix, 
                                                   num_timepoints,
                                                   norm_axis=0)
        
        # (batch, k_equl)
        ins_logprobs_perMix = marginalize_across_t(ins_logprobs_perTime_perMix, 
                                                   num_timepoints,
                                                   norm_axis=0)
        
        # (batch, k_equl)
        del_logprobs_perMix = marginalize_across_t(del_logprobs_perTime_perMix, 
                                                   num_timepoints,
                                                   norm_axis=0)
        
        # (batch, k_indel)
        trans_logprobs_perMix = marginalize_across_t(trans_logprobs_perTime_perMix, 
                                                     num_timepoints,
                                                     norm_axis=0)
        
        
        ### 2.2: log-softmax the mixture logits (if available)
        # get mixture logits from hparams OR pass in a dummy vector of 1
        # (k_subst,)
        subst_mix_logits = params_toTrack.get('susbt_mix_logits', jnp.array([1]))
        
        # (k_equl,)
        equl_mix_logits = params_toTrack.get('equl_mix_logits', jnp.array([1]))
        
        # (k_indel,)
        indel_mix_logits = params_toTrack.get('indel_mix_logits', jnp.array([1]))
        
        # log-softmax these (if its a dummy vector, then this will be zero)
        subst_mix_logprobs = log_softmax(subst_mix_logits)
        equl_mix_logprobs = log_softmax(equl_mix_logits)
        indel_mix_logprobs = log_softmax(indel_mix_logits)
        
        
        ### 2.3: for substitution model, take care of substitution mixtures 
        ###      THEN equlibrium mixtures 
        # k_subs is at dim=1 (middle one out of three dimensions)
        with_subst_mix_weights = (sub_logprobs_perMix +
                                  jnp.expand_dims(subst_mix_logprobs, (0,2)))
        logsumexp_along_k_subst = logsumexp_withZeros(with_subst_mix_weights, axis=1)
        
        # k_equl is now at dim=1 (last one out of three dimensions, but middle 
        #   dimension was removed, so now it's last out of two dimensions)
        with_equl_mix_weights = (logsumexp_along_k_subst +
                                 jnp.expand_dims(equl_mix_logprobs, -1))
        logP_subs = logsumexp_withZeros(with_equl_mix_weights, axis=1)
        
        
        ### 2.4: logP(emissions at insert sites) and 
        ###      logP(omissions at delete sites)
        # insertions
        with_ins_mix_weights = (ins_logprobs_perMix + 
                                jnp.expand_dims(equl_mix_logprobs, 0))
        logP_ins = logsumexp_withZeros(with_ins_mix_weights, axis=1)
        
        # deletions
        with_dels_mix_weights = (del_logprobs_perMix + 
                                 jnp.expand_dims(equl_mix_logprobs, 0))
        logP_dels = logsumexp_withZeros(with_dels_mix_weights, axis=1)
        
        
        ### 2.5: logP(transitions)
        with_indel_mix_weights = (trans_logprobs_perMix + 
                                  jnp.expand_dims(indel_mix_logprobs, 0))
        logP_trans = logsumexp_withZeros(with_indel_mix_weights, axis=1)
        
        
        # sum all and return
        logP_perSamp = logP_subs + logP_ins + logP_dels + logP_trans
        mean_alignment_logprob = jnp.mean(logP_perSamp)
        return -mean_alignment_logprob
        
    
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
    del all_counts
    
    # unpack model tuple
    equl_model, subst_model, indel_model = pairHMM
    del pairHMM
    
    # if equlibrium distribution is a dirichlet mixture, will need a random
    # key for it
    hparams_dict['dirichlet_samp_key'] = jax.random.split(eval_rngkey, 2)[1]
    
    
    #########################################################
    ### 1: CALCULATE LOG-PROBABILITIES PER t, PER MIXTURE   #
    #########################################################
    # retrieve equlibrium distribution vector, logP(emissions/omissions)
    #   this does NOT depend on time, so do this outside the time loop
    equl_vecs, logprob_equl = equl_model.equlVec_logprobs(params_dict,
                                                          hparams_dict)
    
    # overwrite equl_vecs entry in the hparams dictionary
    hparams_dict['equl_vecs'] = equl_vecs
    
    # this calculates logprob at one time, t; vmap this over t_arr
    def apply_model_at_t(t):
        ######################################
        ### 1.1: LOG PROBABILITIES AT TIME t #
        ######################################
        # retrieve substitution logprobs
        logprob_substitution_at_t = subst_model.logprobs_at_t(t, 
                                                              params_dict, 
                                                              hparams_dict)
        
        # retrieve indel logprobs
        logprob_transition_at_t = indel_model.logprobs_at_t(t, 
                                                            params_dict, 
                                                            hparams_dict)
        
        
        ###############################################################
        ### 1.2: multiply counts by logprobs (let einsum handle this) #
        ###############################################################
        # logP(observed substitutions)
        logP_counts_sub = jnp.einsum('bij,ijxy->bxy', 
                                     subCounts_persamp,
                                     logprob_substitution_at_t)
        
        # logP(observed insertions)
        logP_counts_ins = jnp.einsum('bi,iy->by',
                                     insCounts_persamp,
                                     logprob_equl)
        
        # logP(omitted deletions)
        logP_counts_dels = jnp.einsum('bi,iy->by',
                                      delCounts_persamp,
                                      logprob_equl)
        
        # logP(observed transitions)
        logP_counts_trans = jnp.einsum('bmn,mnz->bz',
                                       transCounts_persamp,
                                       logprob_transition_at_t)
        
        # return tuple of these four to deal with... later
        return (logP_counts_sub, logP_counts_ins, 
                logP_counts_dels, logP_counts_trans)
    
    ### vmap over the time array; return log probabilities PER TIME POINT,
    ###   and PER MIXTURE MODEL
    vmapped_apply_model_at_t = jax.vmap(apply_model_at_t)
    tuple_logprobs_perTime_perMix = vmapped_apply_model_at_t(t_arr)
    num_timepoints = len(t_arr)
    
    
    ### unpack tuples; all times will be placed at dim0
    # (time, batch, k_subst, k_equl)
    sub_logprobs_perTime_perMix = tuple_logprobs_perTime_perMix[0]
    
    # (time, batch, k_equl)
    ins_logprobs_perTime_perMix = tuple_logprobs_perTime_perMix[1]
    
    # (time, batch, k_equl)
    del_logprobs_perTime_perMix = tuple_logprobs_perTime_perMix[2]
    
    # (time, batch, k_indel)
    trans_logprobs_perTime_perMix = tuple_logprobs_perTime_perMix[3]
    
    
    ################################################
    ### 2: SUM ACROSS TIME ARRAY, MIXTURE MODELS   #
    ################################################
    ### 2.1: marginalize across t
    # (batch, k_subst, k_equl)
    sub_logprobs_perMix = marginalize_across_t(sub_logprobs_perTime_perMix, 
                                               num_timepoints,
                                               norm_axis=0)
    
    # (batch, k_equl)
    ins_logprobs_perMix = marginalize_across_t(ins_logprobs_perTime_perMix, 
                                               num_timepoints,
                                               norm_axis=0)
    
    # (batch, k_equl)
    del_logprobs_perMix = marginalize_across_t(del_logprobs_perTime_perMix, 
                                               num_timepoints,
                                               norm_axis=0)
    
    # (batch, k_indel)
    trans_logprobs_perMix = marginalize_across_t(trans_logprobs_perTime_perMix, 
                                                 num_timepoints,
                                                 norm_axis=0)
    
    
    ### 2.2: log-softmax the mixture logits (if available)
    # get mixture logits from hparams OR pass in a dummy vector of 1
    # (k_subst,)
    subst_mix_logits = params_dict.get('susbt_mix_logits', jnp.array([1]))
    
    # (k_equl,)
    equl_mix_logits = params_dict.get('equl_mix_logits', jnp.array([1]))
    
    # (k_indel,)
    indel_mix_logits = params_dict.get('indel_mix_logits', jnp.array([1]))
    
    # log-softmax these (if its a dummy vector, then this will be zero)
    subst_mix_logprobs = log_softmax(subst_mix_logits)
    equl_mix_logprobs = log_softmax(equl_mix_logits)
    indel_mix_logprobs = log_softmax(indel_mix_logits)
    
    
    ### 2.3: for substitution model, take care of substitution mixtures 
    ###      THEN equlibrium mixtures 
    # k_subs is at dim=1 (middle one out of three dimensions)
    with_subst_mix_weights = (sub_logprobs_perMix +
                              jnp.expand_dims(subst_mix_logprobs, (0,2)))
    logsumexp_along_k_subst = logsumexp_withZeros(with_subst_mix_weights, axis=1)
    
    # k_equl is now at dim=1 (last one out of three dimensions, but middle 
    #   dimension was removed, so now it's last out of two dimensions)
    with_equl_mix_weights = (logsumexp_along_k_subst +
                             jnp.expand_dims(equl_mix_logprobs, -1))
    logP_subs = logsumexp_withZeros(with_equl_mix_weights, axis=1)
    
    
    ### 2.4: logP(emissions at insert sites) and 
    ###      logP(omissions at delete sites)
    # insertions
    with_ins_mix_weights = (ins_logprobs_perMix + 
                            jnp.expand_dims(equl_mix_logprobs, 0))
    logP_ins = logsumexp_withZeros(with_ins_mix_weights, axis=1)
    
    # deletions
    with_dels_mix_weights = (del_logprobs_perMix + 
                             jnp.expand_dims(equl_mix_logprobs, 0))
    logP_dels = logsumexp_withZeros(with_dels_mix_weights, axis=1)
    
    
    ### 2.5: logP(transitions)
    with_indel_mix_weights = (trans_logprobs_perMix + 
                              jnp.expand_dims(indel_mix_logprobs, 0))
    logP_trans = logsumexp_withZeros(with_indel_mix_weights, axis=1)
    
    
    # sum all and return
    total_logprobs_persamp = logP_subs + logP_ins + logP_dels + logP_trans
    
    loss = -jnp.mean(total_logprobs_persamp)
    

    ##############################################
    ### 3: RETURN LOSS AND INDIVIDUAL LOGPROBS   #
    ##############################################
    # isolate different terms of the loss
    logprobs_persamp = jnp.concatenate([jnp.expand_dims(logP_subs, 1),
                                        jnp.expand_dims((logP_subs + logP_ins), 1),
                                        jnp.expand_dims(logP_trans, 1),
                                        jnp.expand_dims(total_logprobs_persamp, 1)],
                                       axis=1)
                                          
    return (loss, logprobs_persamp)
