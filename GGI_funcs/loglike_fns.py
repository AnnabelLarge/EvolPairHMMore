#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:28:47 2024

@author: annabel

ABOUT:
======
These function help calculate the likelihood of transitions and emissions


"""
import jax
from jax import numpy as jnp


def conditional_loglike(all_counts, all_logprobs):
    """
    logP(desc, align | anc, t) = 
      1.) logP( emissions from match positions )
      2.) logP( emission from insert positions )
      3.) logP( all transitions {M,I,D} )
    """
    ### UNPACK COUNTS, LOGPROBS    
    subCounts_persamp = all_counts[0]
    insCounts_persamp = all_counts[1]
    transCounts_persamp = all_counts[3]
    del all_counts
    
    sub_logprobs, indel_logprobs, trans_logprobs = all_logprobs
    del all_logprobs
    
    
    ### LOGPROB OF EMISSIONS FROM MATCH STATES 
    # (mult counts by substitution matrix)
    logprobs_of_substitutions = jnp.tensordot(subCounts_persamp, sub_logprobs, 
                                              axes=[(1, 2), (0, 1)])
    
    ### LOGPROB OF EMISSIONS FROM INSERT STATES 
    # (mult counts by stationary distribution)
    logprobs_of_insertions = jnp.tensordot(insCounts_persamp, indel_logprobs,
                                           axes=[(1), (0)])
    
    ### LOGPROB OF TRANSITIONS 
    # (mult counts by transition matrix)
    logprobs_of_transitions = jnp.tensordot(transCounts_persamp, trans_logprobs, 
                                              axes=[(1, 2), (0, 1)])
    
    ### sum and return
    cond_logprob = (logprobs_of_substitutions + 
                     logprobs_of_insertions + 
                     logprobs_of_transitions)
    return cond_logprob


def marginal_loglike_Anc(delCounts_persamp, indel_logprobs, lam=None, mu=None):
    """
    logP(anc) =
      1.) logP( deleted ancestor characters even existing in the first place )
    
      [DON'T INCLUDE THIS TERM; INTRODUCES HEADACHES AND NUMERICAL INSTABILITY']
      2.) logP( number of deleted characters, assumed to follow a geometric 
                distribution )

    """
    # Probability of deleted ancestor characters even existing in the
    # first place
    prob_deleted_anc_chars = jnp.tensordot(delCounts_persamp, indel_logprobs,
                                           axes=[(1), (0)])
    
    # # Probability of number of deletions
    # prob_geometric_del_len = (delCounts.sum(axis=1) * jnp.log(lam/mu) +
    #                           jnp.log( 1 - (lam/mu) ) )
    
    # # add these together
    # marg_logprob_perAnc = prob_deleted_anc_chars + prob_geometric_del_len
    
    return prob_deleted_anc_chars
    # return  marg_logprob_perAnc
