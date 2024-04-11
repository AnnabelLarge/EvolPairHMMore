#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:28:30 2024

@author: annabel_large


ABOUT:
======
make sure my version of the H20 model returns the same finite-time transition
  probabilities as Ian's original forward.py

Make sure I get the same transition matrix for a single H20 model

"""
import jax
from jax import numpy as jnp

from unitTests.req_files.ian_forward import transitionMatrix as orig_fn
from model_blocks.GGI_funcs import transitionMatrix as ext_fn


def main():
    #################
    ### FAKE INPUTS #
    #################
    params = { "lambda": 0.5,
                   "mu": 0.5,
                    "x": 0.5,
                    "y": 0.5}
    
    diffraxArgs = { "step": None,
                    "rtol": 1e-3,
                    "atol": 1e-6 }
    
    t = 0.15
    alphabet_size = 20
    
    
    
    ###########################
    ### Ian's function result #
    ###########################
    indelParams = (params["lambda"], params["mu"], params["x"], params["y"])
    expected_mat = orig_fn(t, 
                           indelParams, 
                           alphabetSize = alphabet_size,
                           **diffraxArgs)
    
    
    ########################
    ### My function result #
    ########################
    indelParams_batched = (jnp.array( [ params["lambda"] ] ), 
                           jnp.array( [ params["mu"] ] ), 
                           jnp.array( [ params["x"] ] ), 
                           jnp.array( [ params["y"] ] ) )
    
    test_mat = ext_fn(t, 
                      indelParams_batched, 
                      alphabetSize = alphabet_size,
                      **diffraxArgs)
    
    assert jnp.allclose(expected_mat, test_mat[:,:,0])
    