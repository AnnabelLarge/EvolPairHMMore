#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:27:59 2024

@author: annabel_large


About:
======
SINGLE SUBSTITUTION, 3 EQUILIBRIUM DISTS



universal order of dimensions:
==============================
0. logprob matrix/vectors
2. k_subst
3. k_equl = 3
4. k_indel (NA)
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
    
    # equilibrium vector (i OR j, 3)
    pi_vec = jnp.array([[0.5, 0.5, 0.5, 0.5],
                        [0.4, 0.4, 0.1, 0.1],
                        [0.1, 0.1, 0.4, 0.4]]).T
    k_equl = 3
    
    # other
    alphabet_size = pi_vec.shape[0]
    t = 0.015
    norm = True
    
    
    
    ###############################
    ### 2.) Generate true value   #
    ###############################
    import numpy
    from jax.scipy.linalg import expm as matrix_exponential
    
    def manual_calc(chi_mat, pi_vec, alphabet_size, t, norm, k_equl):    
        """
        Calculate the substitution emission probabilities manually, with loops 
        and numpy
        """
        out_dict = {}
        
        ### make rate matrix R
        # r_ij = \chi_{ij} * \pi_{j}
        raw_R = numpy.zeros((alphabet_size, alphabet_size, 1, k_equl))
        for k in range(k_equl):
            this_pi_vec = pi_vec[:, k]
            for row_i in range(alphabet_size):
                for col_j in range(alphabet_size):
                    r_ij = chi_mat[row_i, col_j] * this_pi_vec[col_j]
                    raw_R[row_i, col_j, 0, k] = r_ij.item()
        
        
        # mask out diagonal elements
        inverse_mask = 1 - numpy.eye(alphabet_size)
        inverse_mask = numpy.expand_dims(inverse_mask, (-1, -2))
        inverse_mask = numpy.repeat(inverse_mask, k_equl, axis=-1)
        masked_R = raw_R * inverse_mask
        
        
        # find the rowsums of the masked matrix; add back to masked R such
        #   that rows of R will sum to 0
        R = numpy.zeros((alphabet_size, alphabet_size, 1, k_equl))
        for k in range(k_equl):
            sub_R = masked_R[:,:,0,k]
            to_fill = numpy.zeros(sub_R.shape)
            for i in range(alphabet_size):
                row = sub_R[i, :]
                rowsum = numpy.sum(row)
                row[i] = -rowsum
                to_fill[i,:] = row
            R[:,:,0,k] = to_fill
        
        
        # rows of R sum to zero (or close to it)
        for k in range(k_equl):
            submat = R[:,:,0,k]
            assert (numpy.abs(numpy.sum(submat, axis=1) - 0) < 1e-6).all()
        
        # clear variables 
        del raw_R, row_i, col_j, r_ij, inverse_mask, masked_R, i, row, rowsum
        del k ,submat, sub_R, to_fill
        
        out_dict['pre_norm_R'] = R
        
        
        ### normalize (if desired)
        if norm:
            normed_R = numpy.zeros(R.shape)
            for k in range(k_equl):
                sub_R = R[:,:,0,k]
                this_pi_vec = pi_vec[:,k]
                # norm_factor = -sum_{i} \pi_{i} * r_{ii}
                norm_factor = 0
                for i in range(alphabet_size):
                    norm_factor += this_pi_vec[i] * sub_R[i,i]
                norm_factor = -norm_factor
                
                sub_R = sub_R/norm_factor
                normed_R[:,:,0,k] = sub_R
                
            R = normed_R
            
            del normed_R, k, sub_R, this_pi_vec, norm_factor      
                
        # rows of R sum to zero (or close to it)
        for k in range(k_equl):
            submat = R[:,:,0,k]
            assert (numpy.abs(numpy.sum(submat, axis=1) - 0) < 1e-6).all()
        
        out_dict['post_norm_R'] = R
        
        
        ### logP(x(t)|x(0)) = log(exp(Rt))
        logP = R * t
        
        # sum of log probabilities in each row is zero
        assert (numpy.abs(numpy.sum(logP, axis=1) - 0) < 1e-6).all()
        
        # P = matrix_exponential(Rt) (not the element-wise exponential)
        P = numpy.zeros(logP.shape)
        for k in range(k_equl):
            to_fill = matrix_exponential(logP[:,:,0,k])
            P[:,:,0,k] = to_fill
        
        # rows of P should sum to 1
        for k in range(k_equl):
            submat = P[:,:,0,k]
            assert (numpy.abs(numpy.sum(submat, axis=1) - 1) < 1e-6).all()
        
        # clear variables
        del P
        
        out_dict['cond_logprob'] = logP
        
        
        ### logP( x(t), x(0) ) = logP( x(t) | x(0) ) + logP( x(0) )
        joint_logP = numpy.zeros(logP.shape)
        for k in range(k_equl):
            old_mat = logP[:,:,0,k]
            this_pi_vec = pi_vec[:, k]
            to_fill = numpy.zeros(old_mat.shape)
            for row_i in range(alphabet_size):
                old_row = old_mat[row_i,:]
                joint_prob_ij = old_row * this_pi_vec[row_i]
                to_fill[row_i] = joint_prob_ij
            joint_logP[:,:,0,k] = to_fill
        
        
        # sum of log probabilities in each row is zero
        for k in range(k_equl):
            submat = joint_logP[:,:,0,k]
            assert (numpy.abs(numpy.sum(submat, axis=1) - 0) < 1e-6).all()
        
            # joint_P = pi * matrix_exponential(Rt) (not the element-wise exponential)
            joint_P = matrix_exponential(submat)
            
            # rows of P should sum to 1
            assert (numpy.abs(numpy.sum(joint_P, axis=1) - 1) < 1e-6).all()
        
        # clear variables
        del joint_P
        
        out_dict['joint_logprob'] = joint_logP
        
        
        ### return desired values 
        return out_dict
    
    true_values = manual_calc(chi_mat, pi_vec, alphabet_size, t, norm, k_equl)
    
    
    
    ############################
    ### 3.) test my function   #
    ############################
    from model_blocks.protein_subst_models import subst_base
    
    ### initialize class object, dictionaries
    my_model = subst_base(norm)
    
    
    ### make rate matrix R
    test_R = my_model.generate_rate_matrix(equl_vecs = pi_vec, 
                                           exch_mat = chi_mat)
    assert jnp.allclose(test_R, true_values['pre_norm_R'])
    
    
    ### normalize R
    test_normed_R = my_model.norm_rate_matrix(subst_rate_mat = test_R, 
                                              equl_pi_mat = pi_vec)
    assert jnp.allclose(test_normed_R, true_values['post_norm_R'])
    
    
    ### get conditional logP
    params = {}
    hparams = {'equl_vecs': pi_vec,
               'exch_mat': chi_mat,
               'alphabet_size': alphabet_size}
    
    test_cond_logprob = my_model.conditional_logprobs_at_t(t = t, 
                                                           params_dict = params, 
                                                           hparams_dict = hparams)
    assert jnp.allclose(test_cond_logprob, true_values['cond_logprob'])
    
    
    ### get joint logP
    test_joint_logprob = my_model.joint_logprobs_at_t(t = t, 
                                                      params_dict = params, 
                                                      hparams_dict = hparams)
    assert jnp.allclose(test_joint_logprob, true_values['joint_logprob'])
    
    