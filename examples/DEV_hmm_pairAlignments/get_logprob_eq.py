#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 01:44:56 2024

@author: annabel
"""
import numpy as np

# We replace zeroes and infinities with small numbers sometimes
# It's sinful but BOY DO I LOVE SIN
smallest_float32 = jnp.finfo('float32').smallest_normal


with open(f'PF00001_AAcounts.npy','rb') as f:
    counts = np.load(f)

eq_prob = counts/counts.sum()
eq_logprob = np.log(  np.where(eq_prob > 0,
                               eq_prob,
                               smallest_float32)
                    )

with open(f'PF00001_logprob_eq.npy','wb') as g:
    np.save(g, eq_logprob)

