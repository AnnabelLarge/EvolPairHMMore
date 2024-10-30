#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:54:42 2024

@author: annabel

"""
import numpy as np
import pandas as pd
import os


OFFSET = 3
smallest_float32 = np.finfo('float32').smallest_normal

def safe_log(mat):
    new_mat = np.where(mat != 0,
                       mat,
                       smallest_float32)
    return np.log(new_mat)


def main(data_dir, 
         train_dset_splits, 
         out_prefix,
         alphabet_size=20, 
         gap_tok=43):
    ###############
    ### READ DATA #
    ###############
    all_aligns = []
    for prefix in train_dset_splits:
        with open(f'{data_dir}/{prefix}_pair_alignments.npy','rb') as f:
            all_aligns.append( np.load(f) )
    all_aligns = np.concatenate(all_aligns, axis=0)
    
    out_df = pd.read_csv(f'{data_dir}/{prefix}_metadata.tsv',
                         sep = '\t',
                         usecols = ['pairID', 'ancestor', 
                                    'descendant', 'desc_seq_len'])
    
    
    
    ########################
    ### INITIALIZE BUCKETS #
    ########################
    match_cond_counts = np.zeros( (alphabet_size, alphabet_size) )
    eq_counts = np.zeros( (alphabet_size, 1) )
    eq_counts_subs_only = np.zeros( (alphabet_size, 1) )
    
    
    ###################
    ### GATHER COUNTS #
    ###################
    ### iterate through samples by dump hand loops
    for b in range(all_aligns.shape[0]):
        for l in range(all_aligns.shape[1]):
            anc_tok, desc_tok = all_aligns[b, l, :]
            
            # stop whole loop at padding positions
            if (anc_tok == 0) or (desc_tok == 0):
                break
            
            # ignore deletion positions
            elif (desc_tok == gap_tok):
                continue
            
            # INSERTIONS
            elif (anc_tok == gap_tok) and (desc_tok != gap_tok):
                desc_tok = desc_tok - OFFSET
                eq_counts[desc_tok, 0] += 1
            
            # MATCHES
            elif (anc_tok != gap_tok) and (desc_tok != gap_tok):
                anc_tok = anc_tok - OFFSET
                desc_tok = desc_tok - OFFSET
                match_cond_counts[anc_tok, desc_tok] += 1
                eq_counts[desc_tok, 0] += 1
                eq_counts_subs_only[desc_tok, 0] += 1
    
    
    
    #######################################
    ### TURN INTO WEIGHT MATRICES/VECTORS #
    #######################################
    ### scoring matches with P( desc | anc )
    anc_marginals = match_cond_counts.sum(axis=1)
    match_cond_prob = match_cond_counts/anc_marginals[:, None]
    
    # rows should sum to 1
    assert np.allclose(match_cond_prob.sum(axis=1), 
                       np.ones( (alphabet_size,) )
                       )
    
    match_cond_logprob = safe_log(match_cond_prob)
    
    
    ### scoring insertions with equilibrium distribution (standard)
    equl_logprobs = safe_log(eq_counts/
                             eq_counts.sum())
    equl_logprobs_subs_only = safe_log(eq_counts_subs_only/
                                       eq_counts_subs_only.sum())
    
    
    ### alternatively, could also score matches with P(desc)
    match_marginals_logprob = np.broadcast_to( equl_logprobs.T,
                                               (alphabet_size, alphabet_size)
                                               )
    match_marginals_logprob_subs_only = np.broadcast_to( equl_logprobs_subs_only.T,
                                               (alphabet_size, alphabet_size)
                                               )
    
    
    ### write scoring matrices to use in EvolPairHMMore later
    # match emissions
    with open(f'{out_prefix}_match-cond-logprob.npy','wb') as g:
        np.save(g,match_cond_logprob )
    
    with open(f'{out_prefix}_match-marginals-logprob.npy','wb') as g:
        np.save(g,match_marginals_logprob )
        
    with open(f'{out_prefix}_match-marginals-logprob-subs-only.npy','wb') as g:
        np.save(g,match_marginals_logprob_subs_only )
    
    
    
    ######################################
    ### LOOP THROUGH SEQUENCES AND SCORE #
    ######################################
    ### types of scores to record
    cond_entropy_scores = np.zeros( (all_aligns.shape[0],) )
    raw_cond_scores = np.zeros( (all_aligns.shape[0],) )
    
    desc_entropy_scores = np.zeros( (all_aligns.shape[0],) )
    raw_desc_scores = np.zeros( (all_aligns.shape[0],) )
    
    desc_entropy_subs_only_scores = np.zeros( (all_aligns.shape[0],) )
    raw_desc_subs_only_scores = np.zeros( (all_aligns.shape[0],) )
    
    
    ### again, iterate through samples by dump hand loops
    for b in range(all_aligns.shape[0]):
        
        desc_len = 0
        match_only_len = 0
        sample_raw_cond_entropy_score = 0
        sample_raw_desc_entropy_score = 0
        sample_raw_desc_entropy_subs_only_score = 0
        
        for l in range(all_aligns.shape[1]):
            anc_tok, desc_tok = all_aligns[b, l, :]
            
            # stop whole loop at padding positions
            if (anc_tok == 0) or (desc_tok == 0):
                break
            
            # ignore deletion positions
            elif (desc_tok == gap_tok):
                continue
            
            # INSERTIONS- score with equilibrium distribution
            elif (anc_tok == gap_tok) and (desc_tok != gap_tok):
                desc_tok = desc_tok - OFFSET
                
                sample_raw_cond_entropy_score += equl_logprobs[desc_tok, 0]
                sample_raw_desc_entropy_score += equl_logprobs[desc_tok, 0]
                
                desc_len += 1
            
            # MATCHES
            elif (anc_tok != gap_tok) and (desc_tok != gap_tok):
                anc_tok = anc_tok - OFFSET
                desc_tok = desc_tok - OFFSET
                
                sample_raw_cond_entropy_score += match_cond_logprob[anc_tok, desc_tok]
                sample_raw_desc_entropy_score += match_marginals_logprob[anc_tok, desc_tok]
                sample_raw_desc_entropy_subs_only_score += equl_logprobs_subs_only[desc_tok, 0]
                
                desc_len += 1
                match_only_len += 1
        
        cond_entropy_scores[b] = sample_raw_cond_entropy_score/desc_len
        raw_cond_scores[b] = sample_raw_cond_entropy_score
        
        desc_entropy_scores[b] = sample_raw_desc_entropy_score/desc_len
        raw_desc_scores[b] = sample_raw_desc_entropy_score
        
        desc_entropy_subs_only_scores[b] = sample_raw_desc_entropy_subs_only_score/match_only_len
        raw_desc_subs_only_scores[b] = sample_raw_desc_entropy_subs_only_score
        
    
    
    ### write results to file
    out_df['raw logP(desc|anc) scores'] = raw_cond_scores
    out_df['logP(desc|anc)'] = cond_entropy_scores
    out_df['Perplexity(desc|anc)'] = np.exp( -cond_entropy_scores )
    
    out_df['raw logP(desc) scores'] = raw_desc_scores
    out_df['logP(desc)'] = desc_entropy_scores
    out_df['Perplexity(desc)'] = np.exp( -desc_entropy_scores )
    
    out_df['raw logP(desc) scores (MATCH SITES ONLY)'] = raw_desc_subs_only_scores
    out_df['logP(desc) (MATCH SITES ONLY)'] = desc_entropy_subs_only_scores
    out_df['Perplexity(desc) (MATCH SITES ONLY)'] = np.exp( -desc_entropy_subs_only_scores )
    
    
    # final logfile
    with open(f'{out_prefix}_AVE-SCORES.txt','w') as g:
        g.write('EvolPairHMMore checksums:\n')
        g.write('==============================\n')
        g.write(f'DON\'T INLUCDE <bos>, <eos> as emissions\n')
        g.write(f'Normalize by UNALIGNED DESCENDANT LENGTH\n\n\n')
        
        g.write(f'setup\n')
        g.write('-----------------------------\n')
        g.write(f'data_dir: {data_dir}\n')
        g.write(f'train_dset_splits: {train_dset_splits}\n')
        g.write(f'alphabet_size: {alphabet_size}\n')
        g.write(f'gap_tok: {gap_tok}\n\n\n')
        
        g.write(f'logP( desc_tok | anc_tok )\n')
        g.write('-----------------------------\n')
        g.write(f"Average logprob: {cond_entropy_scores.mean()}\n")
        g.write(f"Average perplexity: {out_df['Perplexity(desc|anc)'].mean()}\n")
        g.write(f"ECE: {np.exp( -cond_entropy_scores.mean() )}\n\n\n")
        
        g.write(f'logP( desc_tok )\n')
        g.write('-----------------------------\n')
        g.write(f"Average logprob: {desc_entropy_scores.mean()}\n")
        g.write(f"Average perplexity: {out_df['Perplexity(desc)'].mean()}\n")
        g.write(f"ECE: {np.exp( -desc_entropy_scores.mean() )}\n\n\n")
        
        g.write(f'logP( desc_tok ) ONLY at match sites\n')
        g.write('------------------------------------------------\n')
        g.write(f"Average logprob: {desc_entropy_subs_only_scores.mean()}\n")
        g.write(f"Average perplexity: {out_df['Perplexity(desc) (MATCH SITES ONLY)'].mean()}\n")
        g.write(f"ECE: {np.exp( -desc_entropy_subs_only_scores.mean() )}\n")
        
    
    out_df.to_csv(f'{out_prefix}_SCORE-PER-SAMP.tsv', sep='\t')
        
    
if __name__ == '__main__':
    data_dir = 'examples/DEV_hmm_pairAlignments'
    train_dset_splits = ['PF00001']
    alphabet_size = 20
    gap_tok = 43
    out_prefix = 'DEV-PF00001'
    
    main(data_dir, 
         train_dset_splits, 
         out_prefix,
         alphabet_size, 
         gap_tok)
    
