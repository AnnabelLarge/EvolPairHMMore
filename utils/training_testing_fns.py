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
def train_fn(all_counts, 
             t_array, 
             pairHMM, 
             params_dict, 
             hparams_dict, 
             training_rngkey, 
             loss_type='conditional', 
             norm_loss_by='desc_len', 
             DEBUG_FLAG=False):
    """
    Jit-able function to find log-likelihood of both substitutions and indels, 
      and collect gradients
    
    inputs:
        > all_counts: precomputed counts, in this order- 
             subst, inserts, deleted chars, and transitions
        > t_array: array of evolutionary times you're evaluating the likelihood
             at; sum them all together for final likelihood
        > pairHMM: tuple tying together equl_model, subst_model, and 
             indel_model (IN THAT ORDER)
        > params_dict: model parameters to update with optax
        > hparams_dict: hyperparams needed for run
        > training_rngkey: training rng key to add to hyperparameters 
          dictionary (may or may not be used)
        > loss_type: either "conditional" or "joint"
        > norm_loss_by: either "desc_len" or "align_len"
        > DEBUG_FLAG: whether or not to output intermediate values; probably 
          don't do this during training loop
    
    outputs:
        > loss: negative mean log likelihood
        > all_grads: gradients w.r.t. indel parameters (pass this to optax)
        
    """
    # unpack counts tuple
    subCounts_persamp = all_counts[0] #(B, 20, 20)
    insCounts_persamp = all_counts[1] #(B, 20)
    delCounts_persamp = all_counts[2] #(B, 20)
    transCounts_persamp = all_counts[3] #(B, 3, 3)
    del all_counts
    
    # t_array is always determined by first sample in the batch
    # (T,)
    t_array = t_array[0,:]
    
    # decide what length to normalize final loss by (default is
    #   ungapped descendant length)
    num_matches = subCounts_persamp.sum( axis=(1,2) ) #(B, )
    
    if norm_loss_by == 'num_match_pos':
        length_for_normalization = num_matches # (B,)
    
    elif norm_loss_by == 'desc_len':
        num_ins = insCounts_persamp.sum( axis=1 ) #(B, )
        length_for_normalization = num_matches + num_ins #(B, )
    
    elif norm_loss_by == 'align_len':
        num_ins = insCounts_persamp.sum( axis=1 ) #(B, )
        num_dels = delCounts_persamp.sum( axis=1 ) #(B, )
        length_for_normalization = ( num_matches + 
                                     num_ins +
                                     num_dels ) #(B, )
        
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
        # this function calculates logprob at one time, t; vmap this over t_array
        def apply_model_at_t(t):
            ### 1.2.1: get the emission/transition matrices
            # retrieve substitution logprobs
            if loss_type == 'joint':
                logprob_substitution_at_t = subst_model.joint_logprobs_at_t(t, 
                                                                params_toTrack, 
                                                                hparams_dict)
            elif loss_type == 'conditional':
                logprob_substitution_at_t = subst_model.conditional_logprobs_at_t(t, 
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
            
            
            if not DEBUG_FLAG:
                return (logP_counts_sub_at_t, logP_counts_trans_at_t,
                        marg_const_at_t)
            
            else:
                return (logP_counts_sub_at_t, 
                        logP_counts_trans_at_t,
                        marg_const_at_t, 
                        logprob_substitution_at_t, 
                        logprob_transition_at_t)
            
        
        ### 1.2.3: vmap over the time array; return log probabilities 
        ###        PER TIME POINT and PER MIXTURE MODEL
        vmapped_apply_model_at_t = jax.vmap(apply_model_at_t)
        out = vmapped_apply_model_at_t(t_array)
        
        # (time, batch, k_subst, k_equl)
        sub_logprobs_perTime_perMix = out[0]
        
        # (time, batch, k_indel)
        trans_logprobs_perTime_perMix = out[1]
        
        # (time,)
        marginalization_consts = out[2]
        
        # if only one timepoint, don't add any marginalization constants
        if marginalization_consts.shape[0] == 1:
            marginalization_consts = 0 * marginalization_consts
        
        
        if DEBUG_FLAG:
            logprob_subst_mat = out[3]
            logprob_trans_mat = out[4]
            
            intermediate_values = {'logprob_equl_vec':logprob_equl,
                                   'logprob_subst_mat': logprob_subst_mat,
                                   'logprob_trans_mat': logprob_trans_mat,
                                   'marginalization_consts':marginalization_consts}
        
        
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
        
        if DEBUG_FLAG:
            to_add = {'subst_mix_logprobs':subst_mix_logprobs,
                      'equl_mix_logprobs':equl_mix_logprobs,
                      'indel_mix_logprobs':indel_mix_logprobs}
            intermediate_values = {**intermediate_values, **to_add}
        
        
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
        
        
        if loss_type == 'joint':
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
                        logP_trans_perTime)
        
        if loss_type == 'joint':
            logP_perTime = logP_perTime + jnp.expand_dims(logP_dels,0)
        
        
        ### add marginalization_consts (time,)
        logP_perTime_withConst = (logP_perTime +
                                  jnp.expand_dims(marginalization_consts, 1))
        
        ### 1.4.3: logsumexp across time dimension, if more than one time (dim0)
        # (batch, )
        if t_array.shape[0] > 1:
            logP_perSamp = logsumexp_withZeros(logP_perTime_withConst,
                                               axis=0)
        elif t_array.shape[0] == 1:
            logP_perSamp = logP_perTime_withConst[0, :]
        
        # normalize by length of sequence (either ungapped descendant or 
        #   full alignment)
        logP_perSamp = jnp.divide(logP_perSamp, length_for_normalization)

        # output sum to get larger average across all batches
        # (not just this particular batch)
        sum_logP = jnp.sum(logP_perSamp)
        
        ### 1.4.4: final loss is -mean(logP) of this
        loss = -jnp.mean(logP_perSamp)
        
        if not DEBUG_FLAG:
            return loss, {'logP_perSamp': logP_perSamp,
                          'sum_logP': sum_logP}
        
        else:
            to_add = {'logP_perSamp': logP_perSamp,
                      'sum_logP': sum_logP,
                      'logP_perTime_withConst': logP_perTime_withConst}
            intermediate_values = {**intermediate_values, **to_add}
            return loss, intermediate_values
    
    
    ### set up the grad functions, based on above apply function
    ggi_grad_fn = jax.value_and_grad(apply_model, has_aux=True)
    
    # return aux and gradients (gradient is really the only output 
    #   that matters in the training loop)
    out_tup, all_grads = ggi_grad_fn(params_dict)
    loss, aux_dict = out_tup
    aux_dict['loss'] = loss
    
    return aux_dict, all_grads
    


def eval_fn(all_counts, 
            t_array, 
            pairHMM, 
            params_dict, 
            hparams_dict, 
            eval_rngkey, 
            loss_type='conditional', 
            norm_loss_by='desc_len', 
            DEBUG_FLAG=False):
    """
    Jit-able function to find log-likelihood of both substitutions and indels
    
    inputs:
        > all_counts: precomputed counts, in this order- 
             subst, inserts, deleted chars, and transitions
        > t_array: array of evolutionary times you're evaluating the likelihood
             at; sum them all together for final likelihood
        > pairHMM: tuple tying together equl_model, subst_model, and 
             indel_model (IN THAT ORDER)
        > params_dict: model parameters to update with optax
        > hparams_dict: hyperparams needed for run
        > eval_rngkey: rng key for eval (may or may not be needed)
        > loss_type: either "conditional" or "joint"
        > norm_loss_by: either "desc_len" or "align_len"
    
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
    
    # t_array is always determined by first sample in the batch
    # (T,)
    t_array = t_array[0,:]
    
    # decide what length to normalize final loss by (default is
    #   ungapped descendant length)
    num_matches = subCounts_persamp.sum( axis=(1,2) ) #(B, )
    
    if norm_loss_by == 'num_match_pos':
        length_for_normalization = num_matches # (B,)
    
    elif norm_loss_by == 'desc_len':
        num_ins = insCounts_persamp.sum( axis=1 ) #(B, )
        length_for_normalization = num_matches + num_ins #(B, )
    
    elif norm_loss_by == 'align_len':
        num_dels = delCounts_persamp.sum( axis=1 ) #(B, )
        length_for_normalization = ( num_matches + 
                                     num_ins +
                                     num_dels ) #(B, )
    
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
    # use equl_vecs and logprob_equl from TRAINING DATA 
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
    # this function calculates logprob at one time, t; vmap this over t_array
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
        
        
        if not DEBUG_FLAG:
            return (logP_counts_sub_at_t, logP_counts_trans_at_t,
                    marg_const_at_t)
        
        else:
            return (logP_counts_sub_at_t, logP_counts_trans_at_t,
                    marg_const_at_t, logprob_substitution_at_t, 
                    logprob_transition_at_t)
            
    
    ### 1.2.3: vmap over the time array; return log probabilities 
    ###        PER TIME POINT and PER MIXTURE MODEL
    vmapped_apply_model_at_t = jax.vmap(apply_model_at_t)
    out = vmapped_apply_model_at_t(t_array)
    
    # (time, batch, k_subst, k_equl)
    sub_logprobs_perTime_perMix = out[0]
    
    # (time, batch, k_indel)
    trans_logprobs_perTime_perMix = out[1]
    
    # (time,)
    marginalization_consts = out[2]
    
    # if only one timepoint, don't add any marginalization constants
    if marginalization_consts.shape[0] == 1:
        marginalization_consts = 0 * marginalization_consts
        
    
    if DEBUG_FLAG:
        logprob_subst_mat = out[3]
        logprob_trans_mat = out[4]
        
        intermediate_values = {'logprob_equl_vec':logprob_equl,
                               'logprob_subst_mat': logprob_subst_mat,
                               'logprob_trans_mat': logprob_trans_mat,
                               'marginalization_consts':marginalization_consts}
    
    
    
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
    
    if DEBUG_FLAG:
        to_add = {'subst_mix_logprobs':subst_mix_logprobs,
                  'equl_mix_logprobs':equl_mix_logprobs,
                  'indel_mix_logprobs':indel_mix_logprobs}
        intermediate_values = {**intermediate_values, **to_add}
    
    
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
    
    if loss_type == 'joint':
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
                    logP_trans_perTime)
    
    if loss_type == 'joint':
        logP_perTime = logP_perTime + jnp.expand_dims(logP_dels,0)
    
    ### add marginalization_consts (time, batch)
    logP_perTime_withConst = (logP_perTime +
                              jnp.expand_dims(marginalization_consts, 1))
    
    ### 1.4.3: logsumexp across time dimension (dim0)
    # (batch, )
    if t_array.shape[0] > 1:
        logP_perSamp = logsumexp_withZeros(logP_perTime_withConst,
                                           axis=0)
    elif t_array.shape[0] == 1:
        logP_perSamp = logP_perTime_withConst[0, :]
    
    # normalize by length of sequence (either ungapped descendant or 
    #   full alignment)
    logP_perSamp = jnp.divide(logP_perSamp, length_for_normalization)

    # return sum_logP, to do larger average over ALL batches (not just this one)
    sum_logP = jnp.sum(logP_perSamp)
    
    ### 1.4.4: final loss is -mean(logP_perSamp) of this
    loss = -jnp.mean(logP_perSamp)
    
    if not DEBUG_FLAG:
        return ({'logP_perSamp': logP_perSamp,
                 'sum_logP': sum_logP,
                 'loss': loss}, 
                sum_logP)
    
    else:
        to_add = {'logP_perSamp': logP_perSamp,
                  'loss': loss}
        intermediate_values = {**intermediate_values, **to_add}
        return (intermediate_values, sum_logP)

