#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:54:03 2024

@author: annabel


ABOUT:
======
unit test to make sure that equilibrium model outputs make sense

"""
import jax
from jax import numpy as jnp

from model_blocks.equl_distr_models import *


def main():
    ### sample data
    observed_equl = jnp.array([0.1, 0.3, 0, 0.6])
    alphabet_size = observed_equl.shape[0]
    
    assert observed_equl.sum() == 1
    
    
    
    ### no_equl
    params = {}
    hparams = {'equl_vecs_fromData': observed_equl}
    model = no_equl()
    equl_out, placeholder_vec = model.equlVec_logprobs(params, hparams)
    
    # did this function return the observed equlibrium distribution?
    assert jnp.allclose(equl_out[:,0], observed_equl)
    
    # is the placeholder vec correctly a zero vector?
    assert (placeholder_vec == 0).sum().item() == placeholder_vec.shape[0]
    
    del params, hparams, model, equl_out, placeholder_vec
    
    
    ### equl_base
    params = {}
    hparams = {'equl_vecs_fromData': observed_equl}
    model = equl_base()
    equl_out, logequl_out = model.equlVec_logprobs(params, hparams)
    
    # did this function return the observed equlibrium distribution?
    assert jnp.allclose(equl_out[:,0], observed_equl)
    
    # do the log-probabilities match what happens when I manually take the log?
    assert jnp.allclose(logequl_out[:,0], 
                        jnp.nan_to_num(jnp.log(observed_equl), 
                                       posinf=0,
                                       neginf=0))
    
    del params, hparams, model, equl_out, logequl_out
    
    
    ### equl_dirichletMixture
    k_equl = 2
    params = {'dirichlet_shape_transf': jnp.array([[1,1,1,1],
                                                   [2,2,2,2]]).T,
              'equl_mix_logits': jnp.array([1,1])}
    hparams = {'equl_vecs_fromData': observed_equl,
               'alphabet_size': alphabet_size,
               'dirichlet_samp_key': jax.random.key(0)}
    model = equl_dirichletMixture()
    equl_out, _ = model.equlVec_logprobs(params, hparams)
    
    # make sure all sampled distributions sum to 1
    assert jnp.allclose(equl_out.sum(axis=0), jnp.ones(k_equl))
