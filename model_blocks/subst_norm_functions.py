#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:38:49 2024

@author: annabel

Pass these functions in if you want to normalize the rate matrix 
by the equilibrium vector
"""
from jax import numpy as jnp


def norm_rate_matrix(subst_rate_mat, equl_pi_mat):
    ### diag will be (k_subst, k_equl, alphabet_size) (i,j,k)
    diag = jnp.diagonal(subst_rate_mat, axis1=0, axis2=1)
    
    ### equl_dists is (alphabet_size, k_equl) (k, j)
    ###    output is (k_subst, k_equl) (i,j)
    R_times_pi = -jnp.einsum('ijk, kj -> ij', diag, equl_pi_mat)
    
    ### divide each subst_rate_mat with this vec (thanks jnp broadcasting!)
    # (alphabet_size, alphabet_size, k_subst, k_equl) / (k_subst, k_equl)
    out = subst_rate_mat / R_times_pi
    return out


def identity(subst_rate_mat, equl_pi_mat):
    return subst_rate_mat





####################
### TEST FUNCTIONS #
####################
if __name__ == '__main__':
    ### k_subst = 5, k_equl = 2
    mat1 = jnp.array([[5,1,1],
                      [1,5,1],
                      [1,1,5]])
    mat2 = mat1 * 2
    mat3 = mat1 * 3
    mat4 = mat1 * 4
    mat5 = mat1 * 5
    subst_mat1 = jnp.concatenate([jnp.expand_dims(m, -1) 
                                  for m in [mat1, mat2, mat3, mat4, mat5]], 
                                 axis=-1)
    
    subst_mat2 = subst_mat1 * 10
    
    mat = jnp.concatenate([jnp.expand_dims(m, -1) for m in 
                           [subst_mat1, subst_mat2]],
                          axis=-1)
    del subst_mat1, subst_mat2, mat1, mat2, mat3, mat4, mat5
    
    
    vec = jnp.array([[0.5, 0.2, 0.3],
                     [0.1, 0.1, 0.8]]).T

    
    # diag will be (k_subst, k_equl, alphabet_size) (i,j,k)
    diag = jnp.diagonal(mat, axis1=0, axis2=1)
    
    # matmul with equl_dists (alphabet_size, k_equl) (k, j)
    R_times_pi = -jnp.einsum('ijk, kj -> ij', diag, vec)
    
    # divide each subst_rate_mat with this vec
    # (alphabet_size, alphabet_size, k_subst, k_equl) / (k_subst, k_equl)
    out = mat / R_times_pi

