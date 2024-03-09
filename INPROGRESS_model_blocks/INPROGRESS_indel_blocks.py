#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 22:04:27 2024

@author: annabel


universal order of dimensions:
==============================
substitution logprobs:
----------------------
    1. batch
    2. one logprob matrix (alphabet_size, alphabet_size)  
    3. k_subst: however many mixtures for substitution models
    4. k_equl: however many mixtures for equilibrium distribution

equl vec logprobs:
------------------
    1. batch
    2. one equl vector (alphabet_size,)
    3. k_equl: however many mixtures for equilibrium distribution

indel vec logprobs:
--------------------
    1. batch
    2. one logprob matrix (3, 3)  
    3. k_indel: however many mixtures for indel models

timepoints could temporarily be created at dim=0 (doesn't really matter; it 
  will get summed over immediately afterwards')



to add new models:
==================
any future indel model classes NEED the following methods:
1. initialize_model(self, **init_params): initialize all parameters and 
     hyperparameters
2. logprobs_at_t(self, t, **params): calculate logP(indel) 
     at time t
     > this could call any number of other custom functions to help do
       the calculation
"""
import jax
from jax import numpy as jnp

