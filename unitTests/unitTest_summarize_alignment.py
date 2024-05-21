#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:09:08 2024

@author: annabel_large

About:
======
Test summarize_alignment.py on some sample DNA alignments

A: 3
C: 4
G: 5
T: 6
-: 63
"""
import jax
from jax import numpy as jnp

from calcCounts_Train.summarize_alignment import summarize_alignment


def main():
    #################
    ### FAKE INPUTS #
    #################
    ### sequences
    # A T G C
    # A G C T
    samp1 = jnp.array([[3, 6, 5, 4, 0],
                       [3, 5, 4, 6, 0]])
    
    # A A A A A
    # A - - - A
    samp2 = jnp.array([[3,  3,  3,  3, 3],
                       [3, 63, 63, 63, 3]])
    
    # T - -
    # T G C
    samp3 = jnp.array([[6, 63, 63, 0, 0],
                       [6,  5,  4, 0, 0]])
    
    # wrap in a batch; final size is (3, 2, 5)
    fake_seqs = jnp.concatenate([jnp.expand_dims(samp1, 0),
                                 jnp.expand_dims(samp2, 0),
                                 jnp.expand_dims(samp3, 0)], 0)
    del samp1, samp2, samp3
    
    
    ### align lens
    fake_alignlens = (fake_seqs != 0).sum(axis=2)[:, 0]
    
    
    ### fake batch
    fake_batch = (fake_seqs, fake_alignlens, [0])
    del fake_seqs, fake_alignlens
    
    
    
    ########################################
    ### RUN FUNCTION WITH SUBSONLY = FALSE #
    ########################################
    summarize_alignment_jitted = jax.jit(summarize_alignment, 
                                         static_argnames=['max_seq_len',
                                                          'alphabet_size',
                                                          'gap_tok',
                                                          'subsOnly'])
    
    test_out = summarize_alignment_jitted(batch=fake_batch, 
                                          max_seq_len=5, 
                                          alphabet_size=4, 
                                          gap_tok=63,
                                          subsOnly=False)
    
    subCounts_persamp =   test_out[0]
    insCounts_persamp =   test_out[1]
    delCounts_persamp =   test_out[2]
    transCounts_persamp = test_out[3]
    del test_out
    
    
    
    ##################
    ### CHECK VALUES #
    ##################
    # transition counts
    true_transCounts = jnp.array([ [[5, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]],
                                
                                  [[2, 0, 1],
                                   [0, 0, 0],
                                   [1, 0, 2]],
                                 
                                  [[1, 1, 0],
                                   [1, 1, 0],
                                   [0, 0, 0]] ])
    
    assert jnp.allclose(transCounts_persamp, true_transCounts)
    
    
    # substitution counts
    true_subCounts = jnp.array([ [[1, 0, 0, 0],
                                  [0, 0, 0, 1],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0]],
                                
                                 [[2, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]],
                                 
                                 [[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 1]] ])
    
    assert jnp.allclose(subCounts_persamp, true_subCounts)
    
    
    # insertion counts
    true_insCounts = jnp.array([ [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 1, 1, 0] ])
    
    assert jnp.allclose(insCounts_persamp, true_insCounts)
    
    
    # deletion counts
    true_delCounts = jnp.array([ [0, 0, 0, 0],
                                 [3, 0, 0, 0],
                                 [0, 0, 0, 0] ])
    
    assert jnp.allclose(delCounts_persamp, true_delCounts)
    
    
    #######################################
    ### RUN FUNCTION WITH SUBSONLY = TRUE #
    #######################################
    test_out = summarize_alignment_jitted(batch=fake_batch, 
                                          max_seq_len=5, 
                                          alphabet_size=4, 
                                          gap_tok=63,
                                          subsOnly=True)
    
    subCounts_persamp =   test_out[0]
    insCounts_persamp =   test_out[1]
    delCounts_persamp =   test_out[2]
    transCounts_persamp = test_out[3]
    
    batch_size = subCounts_persamp.shape[0]
    alphabet_size = subCounts_persamp.shape[1]
    del test_out
    
    
    
    ##################
    ### CHECK VALUES #
    ##################
    # transition counts
    true_transCounts = jnp.zeros( (batch_size, 3, 3) )
    
    assert jnp.allclose(transCounts_persamp, true_transCounts)
    
    
    # substitution counts
    true_subCounts = jnp.array([ [[1, 0, 0, 0],
                                  [0, 0, 0, 1],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0]],
                                
                                 [[2, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]],
                                 
                                 [[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 1]] ])
    
    assert jnp.allclose(subCounts_persamp, true_subCounts)
    
    
    # insertion counts
    true_insCounts = jnp.zeros( (batch_size, alphabet_size) )
    
    assert jnp.allclose(insCounts_persamp, true_insCounts)
    
    
    # deletion counts
    true_delCounts = jnp.zeros( (batch_size, alphabet_size) )
    
    assert jnp.allclose(delCounts_persamp, true_delCounts)

