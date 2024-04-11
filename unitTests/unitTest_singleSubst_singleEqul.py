#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:27:59 2024

@author: annabel_large


About:
======
SINGLE SUBSTITUTION, SINGLE EQUILIBRIUM



universal order of dimensions:
==============================
0. logprob matrix/vectors
2. k_subst
3. k_equl
4. k_indel
"""
import jax
from jax import numpy as jnp


def main():
    ##############################
    ### 1.) Generate fake data   #
    ##############################
    # exchangeabilities (i,j)
    with open('./unitTests/req_files/unit_test_exchangeabilities.npy', 'rb') as f:
        chi_mat = jnp.load(f)
    
    # equilibrium vector (i OR j, 1)
    pi_vec = jnp.array([[0.1, 0.3, 0.4, 0.2]]).T
    
    # other
    alphabet_size = pi_vec.shape[0]
    t = 0.015
    norm = True
    
    
    
    ###############################
    ### 2.) Generate true value   #
    ###############################
    import numpy
    from jax.scipy.linalg import expm as matrix_exponential
    
    def manual_calc(chi_mat, pi_vec, alphabet_size, t, norm):    
        """
        Calculate the substitution emission probabilities manually, with loops 
        and numpy
        """
        out_dict = {}
        
        ### make rate matrix R
        # r_ij = \chi_{ij} * \pi_{j}
        raw_R = numpy.zeros((alphabet_size, alphabet_size))
        for row_i in range(alphabet_size):
            for col_j in range(alphabet_size):
                r_ij = chi_mat[row_i, col_j] * pi_vec[col_j]
                raw_R[row_i, col_j] = r_ij.item()
        
        # mask out diagonal elements
        inverse_mask = 1 - numpy.eye(alphabet_size)
        masked_R = raw_R * inverse_mask
        
        # find the rowsums of the masked matrix; add back to masked R such
        #   that rows of R will sum to 0
        R = numpy.zeros((alphabet_size, alphabet_size))
        for i in range(alphabet_size):
            row = masked_R[i, :]
            rowsum = numpy.sum(row)
            row[i] = -rowsum
            R[i,:] = row
        
        # rows of R sum to zero (or close to it)
        assert (numpy.abs(numpy.sum(R, axis=1) - 0) < 1e-6).all()
        
        # clear variables 
        del raw_R, row_i, col_j, r_ij, inverse_mask, masked_R, i, row, rowsum
        
        out_dict['pre_norm_R'] = R
        
        
        ### normalize (if desired)
        if norm:
            # norm_factor = -sum_{i} \pi_{i} * r_{ii}
            norm_factor = 0
            for i in range(alphabet_size):
                norm_factor += pi_vec[i] * R[i,i]
            norm_factor = -norm_factor
            
            R = R/norm_factor
            
            # clear variables
            del norm_factor, i
        
        # rows of R_normed still sum to zero (or close to it)
        assert (numpy.abs(numpy.sum(R, axis=1) - 0) < 1e-6).all()
        
        out_dict['post_norm_R'] = R
        
        
        ### logP(x(t)|x(0)) = log(exp(Rt))
        logP = R * t
        
        # sum of log probabilities in each row is zero
        assert (numpy.abs(numpy.sum(logP, axis=1) - 0) < 1e-6).all()
        
        # P = matrix_exponential(Rt) (not the element-wise exponential)
        P = matrix_exponential(logP)
        
        # rows of P should sum to 1
        assert (numpy.abs(numpy.sum(P, axis=1) - 1) < 1e-6).all()
        
        # clear variables
        del P
        
        out_dict['cond_logprob'] = logP
        
        
        ### logP( x(t), x(0) ) = logP( x(t) | x(0) ) + logP( x(0) )
        joint_logP = numpy.zeros((alphabet_size, alphabet_size))
        for row_i in range(alphabet_size):
            old_row = logP[row_i,:]
            joint_prob_ij = old_row * pi_vec[row_i]
            joint_logP[row_i] = joint_prob_ij
        
        # sum of log probabilities in each row is zero
        assert (numpy.abs(numpy.sum(joint_logP, axis=1) - 0) < 1e-6).all()
        
        # joint_P = pi * matrix_exponential(Rt) (not the element-wise exponential)
        joint_P = matrix_exponential(joint_logP)
        
        # rows of P should sum to 1
        assert (numpy.abs(numpy.sum(joint_P, axis=1) - 1) < 1e-6).all()
        
        # clear variables
        del joint_P
        
        out_dict['joint_logprob'] = joint_logP
        
        
        ### return desired values 
        return out_dict
    
    true_values = manual_calc(chi_mat, pi_vec, alphabet_size, t, norm)
    
    
    ############################
    ### 3.) test my function   #
    ############################
    from model_blocks.protein_subst_models import subst_base
    
    ### initialize class object, dictionaries
    my_model = subst_base(norm)
    
    
    ### make rate matrix R
    test_R = my_model.generate_rate_matrix(equl_vecs = pi_vec, 
                                           exch_mat = chi_mat)
    assert jnp.allclose(test_R[:,:,0,0], true_values['pre_norm_R'])
    
    
    ### normalize R
    test_normed_R = my_model.norm_rate_matrix(subst_rate_mat = test_R, 
                                              equl_pi_mat = pi_vec)
    assert jnp.allclose(test_normed_R[:,:,0,0], true_values['post_norm_R'])
    
    
    ### get conditional logP
    params = {}
    hparams = {'equl_vecs': pi_vec,
               'exch_mat': chi_mat,
               'alphabet_size': alphabet_size}
    
    test_cond_logprob = my_model.conditional_logprobs_at_t(t = t, 
                                                           params_dict = params, 
                                                           hparams_dict = hparams)
    assert jnp.allclose(test_cond_logprob[:,:,0,0], true_values['cond_logprob'])
    
    
    ### get joint logP
    test_joint_logprob = my_model.joint_logprobs_at_t(t = t, 
                                                      params_dict = params, 
                                                      hparams_dict = hparams)
    assert jnp.allclose(test_joint_logprob[:,:,0,0], true_values['joint_logprob'])
    
    
