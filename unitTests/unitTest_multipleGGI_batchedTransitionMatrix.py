#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:28:30 2024

@author: annabel_large


ABOUT:
======
make sure my version of the H20 model returns the same finite-time transition
  probabilities as Ian's original forward.py

Make sure I get the same transition matrix for a mixture H20 model

k_indel = 2

"""
import jax
from jax import numpy as jnp

from unitTests.req_files.ian_forward import transitionMatrix as orig_fn
from model_blocks.GGI_funcs import transitionMatrix as ext_fn


def main():
    #################
    ### FAKE INPUTS #
    #################
    params = { "lambda": [0.5,0.2],
                   "mu": [0.5,0.2],
                    "x": [0.5,0.2],
                    "y": [0.5,0.2]}
    
    diffraxArgs = { "step": None,
                    "rtol": 1e-3,
                    "atol": 1e-6 }
    
    t = 0.15
    alphabet_size = 20
    k_indel = 2
    
    
    
    ###########################
    ### Ian's function result #
    ###########################
    expected_mat = []
    for k in range(k_indel):
        indelParams = (params["lambda"][k], 
                       params["mu"][k], 
                       params["x"][k], 
                       params["y"][k])
        k_mat = orig_fn(t, 
                        indelParams, 
                        alphabetSize = alphabet_size,
                        **diffraxArgs)
        k_mat = jnp.expand_dims(k_mat, -1)
        expected_mat.append(k_mat)
    expected_mat = jnp.concatenate(expected_mat, axis=-1)
    
    
    ########################
    ### My function result #
    ########################
    indelParams_batched = (jnp.array( params["lambda"] ), 
                           jnp.array( params["mu"] ), 
                           jnp.array( params["x"] ), 
                           jnp.array( params["y"] ) )
    
    test_mat = ext_fn(t, 
                      indelParams_batched, 
                      alphabetSize = alphabet_size,
                      **diffraxArgs)
    
    assert jnp.allclose(expected_mat, test_mat)
    