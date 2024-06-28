#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:55:27 2024

@author: annabel
"""
import pickle
import numpy as np

import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp


with open(f'subsOnly_eval/HMM_INTERMEDIATES/TRAIN-SPLIT_intermediates.pkl','rb') as f:
    mydict = pickle.load(f)
print(mydict.keys())


logprob_subst_mat = mydict['logprob_subst_mat'][:,:,:,0,0]

for idx in range(logprob_subst_mat.shape[0]):
    one_time = logprob_subst_mat[idx,:,:]
    rowsums = logsumexp(one_time, axis=1)
    
    assert (jnp.abs(rowsums)<1e-6).all()


logprob_equl_vec = mydict['logprob_equl_vec'][:,0]
logprob_trans_mat = mydict['logprob_trans_mat'][:,:,:,0]


logprob_subst_mat = np.array(logprob_subst_mat)
logprob_equl_vec = np.array(logprob_equl_vec)
logprob_trans_mat = np.array(logprob_trans_mat)

