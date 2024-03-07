#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:09:57 2024

@author: annabel

ABOUT:
======
These functions replace summarize_alignment in foward.py

"""
import jax
from jax import numpy as jnp


######################
### HELPER FUNCTIONS #
######################
# these get called in summarize_alignment
def count_substitutions(one_pair, match_pos_mask, alphabet_size=20):
    """
    vectorized way to count types of substitutions i.e. emissions at
    match states
    yields an (alphabet_size, alphabet_size) matrix for the sample
     
    this will get vmapped over the batch dimension
    """
    # adjust input shape
    match_pos_mask = jnp.expand_dims(match_pos_mask, 0)
    
    # this gets vmapped down the sequence length
    def identify_sub_pairs(vec_at_pos, bool_at_pos):
        # identify what the pair is, using an indicator matrix: 
        #    (rows = anc, cols = desc)
        indicator_mat = jnp.zeros((alphabet_size, alphabet_size))
        anc_tok, desc_tok = vec_at_pos
        
        # subtract index by 3, because there's three special tokens at beginning 
        # of alphabet: pad, bos, and eos
        indicator_mat = indicator_mat.at[anc_tok-3, desc_tok-3].add(1)
        
        # finally, multiply by sample boolean i.e. mask the whole matrix if 
        # the position is NOT a match position
        masked_indicator_mat = (indicator_mat * bool_at_pos)
        return masked_indicator_mat
    vmapped_identify_sub_pairs = jax.vmap(identify_sub_pairs, in_axes=1)
    
    # apply fn; output is (seq_len, alphabet_size, alphabet_size)
    subCounts_persamp_persite = vmapped_identify_sub_pairs(one_pair, match_pos_mask)
    
    # sum down the length of the sequence
    subCounts_persamp = jnp.sum(subCounts_persamp_persite, axis=0)
    return subCounts_persamp


def count_insertions(seqs, ins_pos_mask, alphabet_size=20):
    """
    count different types of insertions i.e. emissions at insert states
    yields a (alphabet_size,) vector for the whole batch
    
    this can operate on the whole batch at once
    """
    ### what got inserted = what amino acid is in the descendant sequence at insertion site
    all_inserts = seqs[:, 1, :] * ins_pos_mask
    
    ### count the number of valid tokens
    # this gets vmapped over the valid alphabet
    valid_toks = jnp.arange(3,alphabet_size+3)
    def count_ins(tok):
        return (all_inserts == tok).sum(axis=1)
    vmapped_count_ins = jax.vmap(count_ins, in_axes=0)
    
    insCounts = vmapped_count_ins(valid_toks)
    return insCounts.T


def count_deletions(seqs, del_pos_mask, alphabet_size=20):
    """
    count different types of deletions i.e. what gets removed from the ancestor
    yields a (alphabet_size,) vector for the whole batch
    
    this can operate on the whole batch at once
    """
    ### what got inserted = what amino acid is in the ancestor sequence at deletion site
    all_dels = seqs[:, 0, :] * del_pos_mask
    
    ### count the number of valid tokens
    # this gets vmapped over the valid alphabet
    valid_toks = jnp.arange(3,alphabet_size+3)
    def count_dels(tok):
        return (all_dels == tok).sum(axis=1)
    vmapped_count_dels = jax.vmap(count_dels, in_axes=0)
    
    delCounts = vmapped_count_dels(valid_toks)
    return delCounts.T


def count_transitions(one_alignment_path, start_idxes):
    """
    vectorized way to count types of transitions (M, I, D)
    yields a (3, 3) matrix for the sample
     
    this will get vmapped over the batch dimension
    """
    # this is vmapped over the length of start_idxes, to get a sliding
    # window effect on one_alignment_path
    def identify_pair_type(start_idx):
        indicator_mat = jnp.zeros((4,4))
        from_tok, to_tok = one_alignment_path[(start_idx, start_idx+1),]
        indicator_mat = indicator_mat.at[from_tok, to_tok].add(1)
        return indicator_mat
    vmapped_subpairs = jax.vmap(identify_pair_type, in_axes=0)
    
    # indicator matrix is (4,4), but first row and column are transitions
    # to padding characters and don't really count; cut them off
    out = vmapped_subpairs(start_idxes)
    transition_counts = out[:, 1:, 1:]
    
    # sum over whole sequence length to get all transitions for this 
    # alignment path
    transition_counts = jnp.sum(transition_counts, axis=0)
    return transition_counts





###################
### MAIN FUNCTION #
###################
def summarize_alignment(batch, max_seq_len, alphabet_size=20, gap_tok=63):
    """
    the first element of batch should be a tensor of aligned sequences that 
    are categorically encoded, with the following non-alphabet tokens:
         0: <pad>
         1: <bos> (not included in pairHMM data, but still reserved token)
         2: <eos> (not included in pairHMM data, but still reserved token)
        63: default gap char (but could be changed; just needs to be >=23)
    
    sequence tensor is of size (batch_size, 2, max_seq_len), where-
        dim1=0: ancestor sequence, aligned (i.e. sequence contains gaps)
        dim1=1: descendant sequence, aligned (i.e. sequence contains gaps)
    
    the second element of batch is a vector of size (batch_size,) 
        that has the length of the alignments
    
    max_seq_len is how much the sequence tensor should be clipped; I implement 
        "semi-dynamic" padding, where jax jit will cache different versions 
        of this function for different max_seq_len values (designed for inputs 
        with EXCESSIVE padding)
    
    Returns three tensors:
        1. subCounts_persite: counts of emissions from MATCH positions
        3. insCounts_persamp: counts of emissions from INSERT positions
        2. transCounts: counts of transitions in the batch
    """
    ### unpack batch input to get sequences and align_lens
    seqs, align_len, _ = batch
    del batch
    
    # clip to max seq len
    seqs = seqs[:, :, :max_seq_len]
    
    
    #######################################
    ### COMPRESS ALIGNMENT REPRESENTATION #
    #######################################
    ### split into gaps vs not gaps
    non_gaps = jnp.where((seqs != gap_tok) & (seqs != 0), 1, 0)
    gaps = jnp.where((seqs == gap_tok), seqs, 0)
    
    ### find matches, inserts, and deletions
    # matches found using non_gaps vector
    match_pos = jnp.where(jnp.sum(non_gaps, axis=1) == 2, 1, 0)
    
    # inserts mean ancestor char == gap_tok
    ins_pos = jnp.where(gaps[:,0,:] == gap_tok, 1, 0)
    
    # deletions means descendant char == gap_tok
    del_pos = jnp.where(gaps[:,1,:] == gap_tok, 1, 0)
    
    # combine all into one vec for counting transitions later
    # M = 1, I = 2, D = 3; padding is 0
    paths_compressed = (match_pos + (ins_pos * 2) + (del_pos * 3))
    
    
    ### ADD ADDITIONAL MATCH STATES AT BEGINNING AND END OF ALIGNMENT PATHS
    ### this is part of the GGI assumptions
    # add match at the end of the paths
    extra_end_col = jnp.zeros((seqs.shape[0], 1))
    to_adjust = jnp.concatenate([paths_compressed, extra_end_col], axis=1)
    x_idxes = (jnp.arange(0, seqs.shape[0]))
    with_extra_end_match = to_adjust.at[x_idxes, align_len].add(1)
    
    # add extra start at the beginning of the paths
    extra_start_col = jnp.ones((seqs.shape[0], 1))
    paths_with_extra_start_end_matches = jnp.concatenate([extra_start_col, 
                                                          with_extra_end_match], 
                                                          axis=1).astype(int)
    
    
    ### clean up variables
    del non_gaps, gaps, paths_compressed
    
    
    ######################################
    ### COUNT EMISSIONS FROM MATCH STATE #
    ######################################
    ### use count_substitutions function, defined above
    ### vmap it along the batch dimension (dim0)
    countsubs_vmapped = jax.vmap(count_substitutions, 
                                 in_axes = (0,0, None))
    subCounts_persamp = countsubs_vmapped(seqs, 
                                          match_pos,
                                          alphabet_size)
    
    
    #######################################
    ### COUNT EMISSIONS FROM INSERT STATE #
    #######################################
    insCounts_persamp = count_insertions(seqs = seqs, 
                                         ins_pos_mask = ins_pos, 
                                         alphabet_size=alphabet_size)
    del ins_pos
    
    
    #######################################
    ### COUNT DELETED CHARS FROM ANCESTOR #
    #######################################
    delCounts_persamp = count_deletions(seqs = seqs,
                                        del_pos_mask = del_pos,
                                        alphabet_size=alphabet_size)
    del del_pos
    
    
    #######################
    ### COUNT TRANSITIONS #
    #######################
    ### use count_transitions function, defined above
    ### vmap it along the batch dimension (dim0)
    counttrans_vmapped = jax.vmap(count_transitions, 
                                  in_axes=(0, None))
    start_idxes = jnp.arange(start = 0, 
                             stop = paths_with_extra_start_end_matches.shape[1] - 1)
    transCounts_persamp = counttrans_vmapped(paths_with_extra_start_end_matches, 
                                             start_idxes)

    return (subCounts_persamp, insCounts_persamp, delCounts_persamp, transCounts_persamp)
