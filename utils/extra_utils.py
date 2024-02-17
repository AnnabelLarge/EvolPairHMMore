#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:50:56 2024

@author: annabel_large


ABOUT:
=======
Misc scripts that I sometimes use; not crucial for training
"""
from jax import numpy as jnp


def make_fake_batch():
    """
    For testing small bits of code
    """
    # has one deletion
    samp1 = jnp.array([[3, 4,  5, 6, 7, 0],
                       [3, 4, 63, 6, 7, 0]])
    
    # has one insertion
    samp2 = jnp.array([[3, 4, 63, 6, 7, 8],
                       [3, 4,  5, 6, 7, 8]])
    
    # has one substitution
    samp3 = jnp.array([[3, 4,  5, 6, 0, 0],
                       [3, 4, 12, 6, 0, 0]])
    
    # wrap in a batch; final size is (3, 2, 6)
    fake_batch = jnp.concatenate([jnp.expand_dims(samp1, 0),
                                  jnp.expand_dims(samp2, 0),
                                  jnp.expand_dims(samp3, 0)], 0)
    return fake_batch