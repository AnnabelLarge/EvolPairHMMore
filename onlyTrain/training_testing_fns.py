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

from GGI_funcs.loglike_fns import all_loglikelihoods 
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
    ### unpack batch input
    subCounts_persamp = data[0]
    insCounts_persamp = data[1]
    transCounts_persamp = data[2] 
    del data
    
    # wrap all three together in a tuple
    all_counts = (subCounts_persamp, insCounts_persamp, transCounts_persamp)

    
    ### log(P(insertion at time t)) = log(equl_pi_mat)
    # if there are any zeros, replace them with smallest_float32
    equl_pi_mat = jnp.where(equl_pi_mat == 0, smallest_float32, equl_pi_mat)
    logprob_insertion = jnp.log(equl_pi_mat)
    
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
            ### probability for substitutions
            # log(P(substitution at time t)) = log(exp(R*t)) = R*t
            logprob_substitution_at_t = subst_rate_mat * t
            
            ### probability for transitions
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
            
            # wrap all probability matrices/vectors together
            all_logprobs_at_t = (logprob_substitution_at_t, 
                            logprob_insertion, 
                            logprob_transition_at_t)
            
            
            ### find loglikelihood of alignment
            alignment_logprob_persamp_at_t = all_loglikelihoods(all_counts, all_logprobs_at_t)
            
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
        
        # return the NEGATIVE log likelihood
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
    ### unpack batch input
    subCounts_persamp = data[0]
    insCounts_persamp = data[1]
    transCounts_persamp = data[2] 
    del data
    
    # wrap all three together in a tuple
    all_counts = (subCounts_persamp, insCounts_persamp, transCounts_persamp)
    
    
    ### log(P(insertion at time t)) = log(equl_pi_mat)
    # if there are any zeros, replace them with smallest_float32
    equl_pi_mat = jnp.where(equl_pi_mat == 0, smallest_float32, equl_pi_mat)
    logprob_insertion = jnp.log(equl_pi_mat)
    
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
        ### probability for substitutions
        # log(P(substitution at time t)) = log(exp(R*t)) = R*t
        logprob_substitution_at_t = subst_rate_mat * t
        
        ### probability for transitions
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
        
        # wrap all probability matrices/vectors together
        all_logprobs_at_t = (logprob_substitution_at_t, 
                        logprob_insertion, 
                        logprob_transition_at_t)
        
        
        ### find loglikelihood of alignment
        alignment_logprob_persamp_at_t = all_loglikelihoods(all_counts, all_logprobs_at_t)
        
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
    
    # return the loss and the logprob per sample
    loss = -mean_alignment_logprob
    return (loss, alignment_logprob_persamp)

