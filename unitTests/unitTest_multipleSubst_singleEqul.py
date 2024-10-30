#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:27:59 2024

@author: annabel_large


About:
======
2 SUBSTITUTION MODELS, SINGLE EQUILIBRIUM



universal order of dimensions:
==============================
0. logprob matrix/vectors
2. k_subst = 2
3. k_equl
4. k_indel (NA)
"""
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp



def main():
    ##############################
    ### 1.) Generate fake data   #
    ##############################
    ### exchangeabilities (i,j)
    with open('./unitTests/req_files/unit_test_exchangeabilities.npy', 'rb') as f:
        chi_mat1 = jnp.expand_dims(jnp.load(f), -1)
    
    chi_mat2 = jnp.abs( chi_mat1 + 
                        jax.random.normal(key=jax.random.key(0), 
                                          shape=chi_mat1.shape) )
    
    chi_mat = jnp.concatenate([chi_mat1, chi_mat2], -1)
    del chi_mat1, chi_mat2
    
    k_subst = 2
    
    
    ### equilibrium vector (i OR j, 1)
    pi_vec = jnp.array([[0.1, 0.3, 0.4, 0.2]]).T
    log_pi = jnp.log(pi_vec)
    
    # other
    alphabet_size = pi_vec.shape[0]
    t = 0.0015
    norm = True
    
    
    
    ###############################
    ### 2.) Generate true value   #
    ###############################
    import numpy
    from jax.scipy.linalg import expm as matrix_exponential
    
    def manual_calc(chi_mat, pi_vec, alphabet_size, t, norm, k_subst):    
        """
        Calculate the substitution emission probabilities manually, with loops 
        and numpy
        """
        out_dict = {}
        
        ### make rate matrix R
        # r_ij = rate_class * \chi_{ij} * \pi_{j}
        raw_R = numpy.zeros((alphabet_size, alphabet_size, k_subst, 1))
        for k in range(k_subst):
            this_chi_mat = chi_mat[:,:,k]
            for row_i in range(alphabet_size):
                for col_j in range(alphabet_size):
                    r_ij = this_chi_mat[row_i, col_j] * pi_vec[col_j]
                    raw_R[row_i, col_j, k, 0] = r_ij.item()
        
        
        # mask out diagonal elements
        inverse_mask = 1 - numpy.eye(alphabet_size)
        inverse_mask = numpy.expand_dims(inverse_mask, (-1, -2))
        inverse_mask = numpy.repeat(inverse_mask, k_subst, axis=-2)
        masked_R = raw_R * inverse_mask
        
        
        # find the rowsums of the masked matrix; add back to masked R such
        #   that rows of R will sum to 0
        R = numpy.zeros((alphabet_size, alphabet_size, k_subst, 1))
        for k in range(k_subst):
            sub_R = masked_R[:,:,k,0]
            to_fill = numpy.zeros(sub_R.shape)
            for i in range(alphabet_size):
                row = sub_R[i, :]
                rowsum = numpy.sum(row)
                row[i] = -rowsum
                to_fill[i,:] = row
            R[:,:,k,0] = to_fill
        
        
        # rows of R sum to zero (or close to it)
        for k in range(k_subst):
            submat = R[:,:,k,0]
            assert (numpy.abs(numpy.sum(submat, axis=1) - 0) < 1e-6).all()
            del submat
        
        
        # clear variables 
        del raw_R, row_i, col_j, r_ij, inverse_mask, masked_R, i, row, rowsum
        del k, sub_R, to_fill
        
        out_dict['pre_norm_R'] = R
        
        
        ### normalize (if desired)
        if norm:
            normed_R = numpy.zeros(R.shape)
            for k in range(k_subst):
                sub_R = R[:,:,k,0]
                # norm_factor = -sum_{i} \pi_{i} * r_{ii}
                norm_factor = 0
                for i in range(alphabet_size):
                    norm_factor += pi_vec[i] * sub_R[i,i]
                norm_factor = -norm_factor
                
                sub_R = sub_R/norm_factor
                normed_R[:,:,k,0] = sub_R
                
            R = normed_R
            
            del normed_R, k, sub_R, norm_factor
                
                
        # rows of R_normed still sum to zero (or close to it)
        for k in range(k_subst):
            submat = R[:,:,k,0]
            assert (numpy.abs(numpy.sum(submat, axis=1) - 0) < 1e-6).all()
            del submat
        
        out_dict['post_norm_R'] = R
        
        
        ### logP(x(t)|x(0)) = log(exp(Rt))
        logP = numpy.zeros(R.shape)
        for k in range(k_subst):
            R_mat_here = R[:,:,k,0]
            to_fill = jnp.log( matrix_exponential( R_mat_here*t ) )
            
            # logsumexp of log probabilities in each row is zero
            checksum = logsumexp( to_fill, axis=1 )
            assert not numpy.isnan(checksum).any()
            assert (numpy.abs(checksum - 0) < 1e-6).all()
            del checksum
            
            logP[:,:,k,0] = to_fill
        
        # P = matrix_exponential(Rt) (not the element-wise exponential)
        P = numpy.zeros(logP.shape)
        for k in range(k_subst):
            to_fill = matrix_exponential( R_mat_here*t )
            
            # rows should sum to 1
            assert (numpy.abs(numpy.sum(to_fill, axis=1) - 1) < 1e-6).all()
            
            P[:,:,k,0] = to_fill
        
        # clear variables
        del P
        
        out_dict['cond_logprob'] = logP
        
        
        ### logP( x(t), x(0) ) = logP( x(t) | x(0) ) + logP( x(0) )
        joint_logP = numpy.zeros(logP.shape)
        for k in range(k_subst):
            prev_chi_mat = logP[:,:,k,0]
            this_logpi_vec = log_pi[:, 0]
            to_fill = numpy.zeros(prev_chi_mat.shape)
            for row_i in range(alphabet_size):
                old_row = prev_chi_mat[row_i,:]
                joint_prob_ij = old_row + this_logpi_vec[row_i]
                to_fill[row_i] = joint_prob_ij
            joint_logP[:,:,k,0] = to_fill
        
        out_dict['joint_logprob'] = joint_logP
        
        ### return desired values 
        return out_dict
    
    true_values = manual_calc(chi_mat, pi_vec, alphabet_size, t, norm, k_subst)
    
    
    
    ############################
    ### 3.) test my function   #
    ############################
    from model_blocks.protein_subst_models import subst_mixture
    
    ### initialize class object, dictionaries
    my_model = subst_mixture(norm)
    
    
    ### make rate matrix R
    test_R = my_model.generate_rate_matrix(equl_vecs=pi_vec, 
                                           exch_mat=chi_mat)
    assert jnp.allclose(test_R, true_values['pre_norm_R'])
    
    
    ### normalize R
    test_normed_R = my_model.norm_rate_matrix(subst_rate_mat = test_R, 
                                              equl_pi_mat = pi_vec)
    assert jnp.allclose(test_normed_R, true_values['post_norm_R'])
    
    
    ### get conditional logP
    params = {'subst_mix_logits':jnp.ones(k_subst)}
    
    hparams = {'equl_vecs': pi_vec,
               'exch_mat': chi_mat,
               'logP_equl': log_pi,
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

