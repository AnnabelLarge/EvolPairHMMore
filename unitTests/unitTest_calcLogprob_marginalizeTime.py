#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:13:29 2024

@author: annabel

ABOUT:
======
do a complete calculation of the log probability of a single sample across 
  some timepoints, with single models

(have already validated that mixture models return expected 
  transition/emission probability matrices)

"""
import jax
from jax import numpy as jnp

from model_blocks.equl_distr_models import equl_base
from model_blocks.protein_subst_models import subst_base
from model_blocks.indel_models import GGI_single
from calcCounts_Train.summarize_alignment import summarize_alignment


### logsumexp with zero values
def logsumexp_withZeros(x, axis):
     """
     helper function that's basically a logsumexp that isn't affected by zeros
     if the whole sum( exp(x) ) is zero, then also return zero (not log(0))
     
     see this about NaN's and jnp.where-
     https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-
       where-using-where:~:text=f32%5B1%2C3%5D%7B1%2C0%7D-,Gradients%20
       contain%20NaN%20where%20using%20where,-%23
     """
     zero_val_mask = jnp.where(x != 0,
                               1,
                               0)
     
     exp_x = jnp.exp(x)
     exp_x_masked = exp_x * zero_val_mask
     sumexp_x = jnp.sum(exp_x_masked, axis=axis)
     
     logsumexp_x = jnp.log(jnp.where(sumexp_x > 0., 
                                     sumexp_x, 
                                     1.))
     return logsumexp_x
 
    
### use this to marginalize across timepoints; change as needed?
def marginalize_across_t(logprob_mat, num_timepoints, norm_axis=0):
    """
    P(A) = sum_t[P(A|t)P(t)c] where c is a constant that takes care of 
      geometrically-spaced times and exponential prior over time
    
    """
    ### logsumexp
    logsumexp_probs = logsumexp_withZeros(logprob_mat, 
                                          axis=norm_axis)
    return logsumexp_probs


def main():
    #################
    ### FAKE INPUTS #
    #################
    ### params and whatnot
    diffraxArgs = { "step": None,
                    "rtol": 1e-3,
                    "atol": 1e-6 }
    
    t_arr = jnp.array([0.1 * 1.1**0, 
                       0.1 * 1.1**1, 
                       0.1 * 1.1**2])
    alphabet_size = 4
    
    with open('./unitTests/req_files/unit_test_exchangeabilities.npy','rb') as g:
        exch_mat = jnp.load(g)
    del g
    
    ### sequences
    # A T G C
    # A G C T
    samp1 = jnp.array([[3, 6, 5, 4, 0],
                       [3, 5, 4, 6, 0]])
    
    
    # wrap in a batch; final size is (1, 2, 5)
    fake_seqs = jnp.expand_dims(samp1, 0)
    del samp1
    
    
    ### align lens
    fake_alignlens = (fake_seqs != 0).sum(axis=2)[:, 0]
    
    
    ### fake batch
    fake_batch = (fake_seqs, fake_alignlens, [0])
    
    
    ### counts
    test_out = summarize_alignment(batch=fake_batch, 
                                    max_seq_len=5, 
                                    alphabet_size=alphabet_size, 
                                    gap_tok=63)
    subCounts_persamp =   test_out[0]
    insCounts_persamp =   test_out[1]
    delCounts_persamp =   test_out[2]
    transCounts_persamp = test_out[3]
    
    del fake_alignlens, fake_batch, test_out, fake_seqs
    
    
    ### equilibrium distributions
    equl_vecs_fromData = jnp.array([0.25, 0.25, 0.25, 0.25])
    
    
    
    ############
    ### MODELS #
    ############
    equl_model = equl_base()
    subst_model = subst_base(norm=True)
    indel_model = GGI_single()
    
    
    params = { "lam_transf": jnp.array([0.5]),
                "mu_transf": jnp.array([0.5]),
                 "x_transf": jnp.array([0.5]),
                 "y_transf": jnp.array([0.5])}
    
    hparams = {'equl_vecs_fromData': equl_vecs_fromData,
                         'exch_mat': exch_mat,
                   'diffrax_params': diffraxArgs,
                    'alphabet_size': alphabet_size}
    
    del equl_vecs_fromData, exch_mat, diffraxArgs
    
    
    #########################################################
    ### 1: CALCULATE LOG-PROBABILITIES PER t, PER MIXTURE   #
    #########################################################
    # retrieve equlibrium distribution vector, logP(emissions/omissions)
    #   this does NOT depend on time, so do this outside the time loop
    equl_vecs, logprob_equl = equl_model.equlVec_logprobs(params,
                                                          hparams)
    
    # overwrite equl_vecs entry in the hparams dictionary
    hparams['equl_vecs'] = equl_vecs
    
    ### this calculates logprob at one time, t
    ###   in real code, this is vmapped over the entire t_array
    sub_logprobs_perTime_perMix = []
    ins_logprobs_perTime_perMix = []
    del_logprobs_perTime_perMix = []
    trans_logprobs_perTime_perMix = []
    for t in t_arr:
        ######################################
        ### 1.1: LOG PROBABILITIES AT TIME t #
        ######################################
        # retrieve substitution logprobs
        logprob_substitution_at_t = subst_model.joint_logprobs_at_t(t, 
                                                              params, 
                                                              hparams)
        
        # retrieve indel logprobs
        logprob_transition_at_t = indel_model.logprobs_at_t(t, 
                                                            params, 
                                                            hparams)
        
        ###############################################################
        ### 1.2: multiply counts by logprobs (let einsum handle this) #
        ###############################################################
        def validate(counts_mat, logP_mat, val_from_code):
            counts_mat = jnp.ravel(counts_mat)
            logP_mat = jnp.ravel(logP_mat)
            
            assert counts_mat.shape == logP_mat.shape
            
            checkval = 0
            for i in range(counts_mat.shape[0]):
                    checkval += counts_mat[i] * logP_mat[i]
            
            err_message = f'Calculated value was {val_from_code}, but true value is {checkval}'
            assert jnp.allclose(checkval, val_from_code)
            
        ### logP(observed substitutions)
        logP_counts_sub = jnp.einsum('bij,ijxy->bxy', 
                                     subCounts_persamp,
                                     logprob_substitution_at_t)
        
        # verify this einsum formula works by checking against a manual loop
        validate(subCounts_persamp[0,:,:], 
                 logprob_substitution_at_t[:,:,0,0], 
                 logP_counts_sub)
                
        
        ### logP(observed insertions)
        logP_counts_ins = jnp.einsum('bi,iy->by',
                                     insCounts_persamp,
                                     logprob_equl)
        
        # verify this einsum formula works by checking against a manual loop
        validate(insCounts_persamp[0,:], 
                 logprob_equl[:,0], 
                 logP_counts_ins)
        
        
        ### logP(omitted deletions)
        logP_counts_dels = jnp.einsum('bi,iy->by',
                                      delCounts_persamp,
                                      logprob_equl)
        
        # verify this einsum formula works by checking against a manual loop
        validate(delCounts_persamp[0,:], 
                 logprob_equl[:,0], 
                 logP_counts_dels)
        
        
        ### logP(observed transitions)
        logP_counts_trans = jnp.einsum('bmn,mnz->bz',
                                       transCounts_persamp,
                                       logprob_transition_at_t)
        
        # verify this einsum formula works by checking against a manual loop
        validate(transCounts_persamp[0,:,:], 
                 logprob_transition_at_t[:,:,0], 
                 logP_counts_trans)
        
        
        ### append to lists
        sub_logprobs_perTime_perMix.append(jnp.expand_dims(logP_counts_sub, 0))
        ins_logprobs_perTime_perMix.append(jnp.expand_dims(logP_counts_ins, 0))
        del_logprobs_perTime_perMix.append(jnp.expand_dims(logP_counts_dels, 0))
        trans_logprobs_perTime_perMix.append(jnp.expand_dims(logP_counts_trans, 0))
    
    
    ### return log probabilities PER TIME POINT and PER MIXTURE MODEL
    num_timepoints = len(t_arr)
    
    
    ### unpack tuples; all times will be placed at dim0
    # (time, batch, k_subst, k_equl)
    sub_logprobs_perTime_perMix = jnp.concatenate(sub_logprobs_perTime_perMix, 0)
    
    # (time, batch, k_equl)
    ins_logprobs_perTime_perMix = jnp.concatenate(ins_logprobs_perTime_perMix, 0)
    
    # (time, batch, k_equl)
    del_logprobs_perTime_perMix = jnp.concatenate(del_logprobs_perTime_perMix, 0)
    
    # (time, batch, k_indel)
    trans_logprobs_perTime_perMix = jnp.concatenate(trans_logprobs_perTime_perMix, 0)
    
    
    
    ################################################
    ### 2: SUM ACROSS TIME ARRAY, MIXTURE MODELS   #
    ################################################
    from jax.scipy.special import logsumexp
    
    ### true values; note that there's no insertions or deletions, so
    ###   return P(ins) = P(del) = 0
    def true_vals(tup):
        sub_logprobs_perTime_perMix = tup[0]
        trans_logprobs_perTime_perMix = tup[3]
        
        # reduce dimensions
        sub_logprobs = sub_logprobs_perTime_perMix[:,0,0,0]
        trans_logprobs = trans_logprobs_perTime_perMix[:,0,0]
        
        # exp
        sub_probs = jnp.exp(sub_logprobs)
        trans_probs = jnp.exp(trans_logprobs)
        
        # sum
        sub_probsSummed = jnp.sum(sub_probs)
        trans_probsSummed = jnp.sum(trans_probs)
        
        # log
        sub_logsumexp = jnp.log(sub_probsSummed)
        ins_logsumexp = 0
        del_logsumexp = 0
        trans_logsumexp = jnp.log(trans_probsSummed)
        
        return (sub_logsumexp, ins_logsumexp, del_logsumexp, trans_logsumexp)
    
    
    checksums = true_vals((sub_logprobs_perTime_perMix,
                           ins_logprobs_perTime_perMix,
                           del_logprobs_perTime_perMix,
                           trans_logprobs_perTime_perMix))
    
    
    ### values as originally calculated
    # (batch, k_subst, k_equl)
    sub_logprobs_perMix = marginalize_across_t(sub_logprobs_perTime_perMix, 
                                                num_timepoints,
                                                norm_axis=0)
    
    assert jnp.allclose(sub_logprobs_perMix.item(), checksums[0])
    
    
    # (batch, k_equl)
    ins_logprobs_perMix = marginalize_across_t(ins_logprobs_perTime_perMix, 
                                                num_timepoints,
                                                norm_axis=0)
    
    assert jnp.allclose(ins_logprobs_perMix.item(), checksums[1])
    
    
    # (batch, k_equl)
    del_logprobs_perMix = marginalize_across_t(del_logprobs_perTime_perMix, 
                                                num_timepoints,
                                                norm_axis=0)
    
    assert jnp.allclose(del_logprobs_perMix.item(), checksums[2])
    
    
    # (batch, k_indel)
    trans_logprobs_perMix = marginalize_across_t(trans_logprobs_perTime_perMix, 
                                                  num_timepoints,
                                                  norm_axis=0)
    
    assert jnp.allclose(trans_logprobs_perMix.item(), checksums[3])
