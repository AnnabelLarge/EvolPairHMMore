#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:28:47 2024

@author: annabel

ABOUT:
======
These function calculate the likelihood of transitions and emissions

TODO:
=====
make the log likelihood symmetric

"""
import jax
from jax import numpy as jnp


def all_loglikelihoods(all_counts, all_logprobs):
    """
    Find the different log-likelihoods of 1.) emissions from match positions,
    2.) emission from insert positions, 3.) all transitions {M,I,D}
    """
    ### UNPACK COUNTS, LOGPROBS    
    subCounts_persamp, insCounts_persamp, transCounts_persamp = all_counts
    del all_counts
    
    sub_logprobs, ins_logprobs, trans_logprobs = all_logprobs
    del all_logprobs
    
    
    ### LOGPROB OF EMISSIONS FROM MATCH STATES 
    # (mult counts by substitution matrix)
    logprobs_of_substitutions = jnp.tensordot(subCounts_persamp, sub_logprobs, 
                                              axes=[(1, 2), (0, 1)])
    
    ### LOGPROB OF EMISSIONS FROM INSERT STATES 
    # (mult counts by stationary distribution)
    logprobs_of_insertions = jnp.tensordot(insCounts_persamp, ins_logprobs,
                                           axes=[(1), (0)])
    
    ### LOGPROB OF TRANSITIONS 
    # (mult counts by transition matrix)
    logprobs_of_transitions = jnp.tensordot(transCounts_persamp, trans_logprobs, 
                                              axes=[(1, 2), (0, 1)])
    
    ### sum and return
    total_logprob = (logprobs_of_substitutions + 
                     logprobs_of_insertions + 
                     logprobs_of_transitions)
    return total_logprob
    
