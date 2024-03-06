#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:04:57 2024

@author: annabel

ABOUT:
======
These functions are used at training and evaluation time for single simple 
  GGI model (no site variation, no indel rate variation, 
  no amino acid distribution variation)

MADE TO WORK WITH PRECACULATED COUNTS MATRICES

"""
import jax
from jax import numpy as jnp
import optax
from jax.scipy.special import logsumexp 

from GGI_funcs.loglike_fns import all_loglike
from GGI_funcs.rates_transition_matrices import transitionMatrix


# We replace zeroes and infinities with small numbers sometimes
# It's sinful but BOY DO I LOVE SIN
smallest_float32 = jnp.finfo('float32').smallest_normal


def train_fn(data, t_arr, subst_rate_mat, equl_pi_mat, indel_params_transformed, 
             diffrax_params, max_seq_len=None):
    """
    Jit-able function to create new transition matrix given the indel
      params, find log-likelihood of both substitutions and indels, 
      and collect gradients
    
    inputs:
        > data: data from a pytorch dataloader
        > t_arr: array of evolutionary times you're evaluating the likelihood at; 
             sum them all together for final likelihood
        > subst_rate_mat: rate matrix Q
        > equl_pi_mat: equilibrium distribution, Pi
        > indel_params_transformed: (lam, mu, x, y) in that order, transformed 
          to domain (-inf, inf)
        > diffrax_params: some parameters to pass to diffrax
        > max_seq_len: not used here, but keeping to preserve function signature
    
    outputs:
        > loss: negative mean log likelihood
        > all_grads: gradients w.r.t. indel parameters (pass this to optax)
        
    """
    ### get count matrices by unpacking some of data input
    subCounts_persamp = data[0]
    insCounts_persamp = data[1]
    delCounts_persamp = data[2]
    transCounts_persamp = data[3]
    all_counts = (subCounts_persamp, insCounts_persamp, 
                  delCounts_persamp, transCounts_persamp)
    del data, subCounts_persamp, insCounts_persamp
    del delCounts_persamp, transCounts_persamp

    
    ### log(P(insertion at time t)) = log(equl_pi_mat)
    # if there are any zeros, replace them with smallest_float32
    equl_pi_mat = jnp.where(equl_pi_mat == 0, smallest_float32, equl_pi_mat)
    logprob_indel = jnp.log(equl_pi_mat)
    
    # TODO: add substitution parameters, weighting parameters for mixture models
    def apply_model(indel_params):
        # lam_t, mu_t, x_t, y_t
        ### unpack indel params; undo the domain transformation
        lam_t, mu_t, x_t, y_t = indel_params
        lam = jnp.square(lam_t)
        mu = jnp.square(mu_t)
        x = jnp.exp(-jnp.square(x_t))
        y = jnp.exp(-jnp.square(y_t))
        
        
        ### this calculates logprob at one time, t
        ### vmap it over time array
        def apply_model_at_time_t(t):
            ### 1: GATHER LOG PROBABILITIES FOR EVENTS
            # 1.1: probability for substitutions
            #      log(P(substitution at time t)) = log(exp(R*t))
            logprob_substitution_at_t = subst_rate_mat * t
            
            # 1.2: probability for transitions
            # transition matrix ((a,b,c),(f,g,h),(p,q,r)); rows sum to 1
            alphabet_size = 20
            indel_params = (lam, mu, x, y)
            transmat = transitionMatrix (t, 
                                          indel_params, 
                                          alphabet_size,
                                          **diffrax_params)
            
            # if any position in transmat is zero, replace with smallest_float32
            transmat = jnp.where(transmat == 0, smallest_float32, transmat)
            logprob_transition_at_t = jnp.log(transmat)
            
            # 1.3: wrap all probability matrices/vectors together
            all_logprobs_at_t = (logprob_substitution_at_t, 
                                 logprob_indel, 
                                 logprob_transition_at_t)
            
            
            ### 2: CALCULATE JOINT LOG LIKELIHOOD
            # independent log probs for all cases: 
            #   order of input: subst, inserts, deleted chars, and transitions
            out_logprobs = all_loglike(all_counts, all_logprobs_at_t)
            
            # alignment logprob is sum of these
            alignment_logprob_persamp_at_t = out_logprobs.sum(axis=1)
            
            return alignment_logprob_persamp_at_t
        
        ### do the log probability calculation for all times t
        ###   output is (num_timepoints, num_samples)
        vmapped_apply_model_at_time_t = jax.vmap(apply_model_at_time_t)
        alignment_logprob_persamp_across_t_arr = vmapped_apply_model_at_time_t(t_arr)
        
        # logsumexp across all geometrically-spaced timepoints (this is 
        #   effectively marginalizing out time)
        alignment_logprob_persamp = logsumexp(alignment_logprob_persamp_across_t_arr,
                                              axis=0)
           
        
        ### get mean log likelihood across the batch
        mean_alignment_logprob = jnp.mean(alignment_logprob_persamp)
        
        # return the NEGATIVE log likelihood (you dolt)
        return -mean_alignment_logprob
    
    
    ### set up the grad functions, based on above apply function
    ggi_grad_fn = jax.value_and_grad(apply_model, has_aux=False)
    
    # return loss and gradients
    loss, all_grads = ggi_grad_fn(indel_params_transformed)
    
    return (loss, all_grads)




def eval_fn(data, t_arr, subst_rate_mat, equl_pi_mat, 
            indel_params_transformed, diffrax_params, max_seq_len=None):
    """
    Jit-able function to create new transition matrix given the indel
      params, find log-likelihood of both substitutions and indels, collect
      gradients, and update model params with gradient descent
    
    inputs:
        > data: data from a pytorch dataloader
        > t_arr: array of evolutionary times you're evaluating the likelihood at; 
             sum them all together for final likelihood
        > subst_rate_mat: rate matrix Q
        > equl_pi_mat: equilibrium distribution, Pi
        > indel_params_transformed: (lam, mu, x, y) in that order, transformed 
          to domain (-inf, inf)
        > diffrax_params: some parameters to pass to diffrax
        > max_seq_len: not used here, but keeping to preserve function signature
    
    outputs:
        > loss: negative mean log likelihood
        > alignment_logprob_persamp: log likelihood per pair, summed over all times in t_arr
        
    """
    ### get count matrices by unpacking some of data input
    subCounts_persamp = data[0]
    insCounts_persamp = data[1]
    delCounts_persamp = data[2]
    transCounts_persamp = data[3]
    all_counts = (subCounts_persamp, insCounts_persamp, 
                  delCounts_persamp, transCounts_persamp)
    del data, subCounts_persamp, insCounts_persamp
    del delCounts_persamp, transCounts_persamp

    
    ### log(P(insertion at time t)) = log(equl_pi_mat)
    # if there are any zeros, replace them with smallest_float32
    equl_pi_mat = jnp.where(equl_pi_mat == 0, smallest_float32, equl_pi_mat)
    logprob_indel = jnp.log(equl_pi_mat)
    
    # TODO: add substitution parameters, weighting parameters for mixture models
    ### unpack indel params; undo the domain transformation
    lam_t, mu_t, x_t, y_t = indel_params_transformed
    lam = jnp.square(lam_t)
    mu = jnp.square(mu_t)
    x = jnp.exp(-jnp.square(x_t))
    y = jnp.exp(-jnp.square(y_t))
    
    
    ### this calculates logprob at one time, t
    ### vmap it over time array
    def apply_model_at_time_t(t):
        ### 1: GATHER LOG PROBABILITIES FOR EVENTS
        # 1.1: probability for substitutions
        #      log(P(substitution at time t)) = log(exp(R*t))
        logprob_substitution_at_t = subst_rate_mat * t
        
        # 1.2: probability for transitions
        # transition matrix ((a,b,c),(f,g,h),(p,q,r)); rows sum to 1
        alphabet_size = 20
        indel_params = (lam, mu, x, y)
        transmat = transitionMatrix (t, 
                                      indel_params, 
                                      alphabet_size,
                                      **diffrax_params)
        
        # if any position in transmat is zero, replace with smallest_float32
        transmat = jnp.where(transmat == 0, smallest_float32, transmat)
        logprob_transition_at_t = jnp.log(transmat)
        
        # 1.3: wrap all probability matrices/vectors together
        all_logprobs_at_t = (logprob_substitution_at_t, 
                             logprob_indel, 
                             logprob_transition_at_t)
        
        
        ### 2: CALCULATE JOINT LOG LIKELIHOOD
        # return all individually; note: not all losses will vary with time
        # (num_samples, 4)
        logprobs_per_type_at_t = all_loglike(all_counts, all_logprobs_at_t)
        
        return logprobs_per_type_at_t
    
    
    ### do the log probability calculation for all times t
    ###   output is (num_timepoints, num_samples, 4)
    vmapped_apply_model_at_time_t = jax.vmap(apply_model_at_time_t)
    logprobs_persamp_across_t_arr = vmapped_apply_model_at_time_t(t_arr)
    
    ### when calculating loss, want to sum across the different loss terms,
    ### THEN logsumexp down timepoints
    alignment_logprob_across_t_arr = logprobs_persamp_across_t_arr.sum(axis=2)
    alignment_logprob_persamp = logsumexp(alignment_logprob_across_t_arr, axis=0)
    mean_alignment_logprob = jnp.mean(alignment_logprob_persamp)
    loss = -mean_alignment_logprob
    
    
    ### record what losses are when masking out different terms of the loss function
    def extract_indv_logprobs(logprobs_tensor, which_cols):
        mask = jnp.zeros(logprobs_tensor.shape)
        mask = mask.at[:, :, which_cols].add(1)
        masked_logprobs_tensor = logprobs_tensor * mask
        logprob_after_sum = masked_logprobs_tensor.sum(axis=2)
        logprob_persamp = logsumexp(logprob_after_sum, axis=0)
        return logprob_persamp
    
    subst_logprobs_persamp = extract_indv_logprobs(logprobs_persamp_across_t_arr, 0)
    all_emission_logprobs_persamp = extract_indv_logprobs(logprobs_persamp_across_t_arr, [0,1])
    trans_logprobs_persamp = extract_indv_logprobs(logprobs_persamp_across_t_arr, 3)
    
    logprobs_persamp = jnp.concatenate([jnp.expand_dims(subst_logprobs_persamp, 1),
                                        jnp.expand_dims(all_emission_logprobs_persamp,1),
                                        jnp.expand_dims(trans_logprobs_persamp,1),
                                        jnp.expand_dims(alignment_logprob_persamp,1)], 1)
    
    return (loss, logprobs_persamp)

