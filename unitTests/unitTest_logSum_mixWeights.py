#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:38:42 2024

@author: annabel

ABOUT:
======
test the recipes used to do the addition when calculating mixtures

to keep numbers straight: 
    batch = 1
    k_subst = 2
    k_equl = 3

"""
import jax
from jax import numpy as jnp


def main():
    k_subst = 2
    subst_mix_logprobs = jnp.array([10, 20])
    
    k_equl = 3
    equl_mix_logprobs = jnp.array([100, 200, 300])
    
    
    #############################################
    ### substitution model with k_subst, k_equl #
    #############################################
    ### example data
    sub_logprobs_perMix = jnp.expand_dims(jnp.array([[1,2,3],
                                                     [4,5,6]]), 
                                          0)
    
    ### true values will have the following addition grid added
    add_grid = jnp.expand_dims(jnp.array([[10+100, 10+200, 10+300],
                                          [20+100, 20+200, 20+300]]),
                               0)
    
    true_vals = sub_logprobs_perMix + add_grid
    
    
    ### below is what's actually used in the training/test functions
    with_subst_mix_weights = (sub_logprobs_perMix +
                              jnp.expand_dims(subst_mix_logprobs, (0,2)))
    
    with_equl_mix_weights = (with_subst_mix_weights +
                              jnp.expand_dims(equl_mix_logprobs, 0))
    
    assert jnp.allclose(with_equl_mix_weights, true_vals)
    del sub_logprobs_perMix, add_grid, true_vals, with_subst_mix_weights
    del with_equl_mix_weights, subst_mix_logprobs
    
    
    
    ###################################################
    ### equl/transition models with k_equl OR k_indel #
    ###################################################
    # this one is way simpler, but just sanity check it anyways
    
    ### example data
    ins_logprobs_perMix = jnp.expand_dims(jnp.array([1,2,3]), 0)
    
    true_value = jnp.array([[101, 202, 303]])
    
    with_ins_mix_weights = (ins_logprobs_perMix + 
                            jnp.expand_dims(equl_mix_logprobs, 0))
    
    assert jnp.allclose(with_ins_mix_weights, true_value)
