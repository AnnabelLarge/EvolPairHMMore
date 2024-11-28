#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:39:48 2023

@author: annabel

About:
======
Custom pytorch dataset object for giving pfam data to PAIR HMM MODELS (like
  the GGI model)


outputs:
========
1. the pair alignments, categorically encoded (batch_size, max_len, 2)
   > dim2=0:ancestor
   > dim2=1: descendant

2. the length of alignments (batch_size, )

3. the sample indices, in order of accession (use retrieve_sample_names to get 
   metadata information)


Data to be read:
=================
1. Numpy matrix of aligned matrix info: a tensor of size (num_pairs, max_len, 4)
   only need dim2=[0,1]
   dim2=0: aligned ancestor (with <bos>, <eos>)
   dim2=0: aligned descendant (with <bos>, <eos>)

  All sequences have been categorically encoded (20 possible aa tokens + gap + pad)
  - <pad> token is 0
  - <bos> is understood to be 1 (will be removed)
  - <eos> is understood to be 2 (will be removed)
  - <gap> token is 43
  
2. Numpy vector of amino acid counts; if this is the training dataloader, 
   you'll calculate the equilibrium distribution of amino acids from this

3. Pandas dataframe of metadata

"""
import torch
from torch.utils.data import Dataset, DataLoader,default_collate
import numpy as np
import jax
from jax import numpy as jnp
from jax.tree_util import tree_map
import pandas as pd


def jax_collator(batch):
    return tree_map(jnp.asarray, default_collate(batch))


class HMMDset(Dataset):
    def __init__(self, data_dir, split_prefixes, subsOnly):
        #######################
        ### SUBSET FULL FILES #
        #######################
        data_mat_lst = []
        metadata_list = []
        self.AAcounts = np.zeros(20, dtype=int)
        
        for split in split_prefixes:
            ### full matrix
            with open(f'./{data_dir}/{split}_aligned_mats.npy', 'rb') as f:
                raw_mat = np.load(f)[:,:,[0,1]]
            
            # remove bos, eos, convert to int32 for pytorch dataloaders
            raw_mat = np.where(raw_mat != 2, raw_mat, 0)
            raw_mat = raw_mat[:, 1:-1, :]
            raw_mat = raw_mat.astype('int32')
            data_mat_lst.append(raw_mat)
            del raw_mat
            
            ### metadata
            cols_to_keep = ['pairID',
                            'ancestor',
                            'descendant',
                            'pfam', 
                            'alignment_len', 
                            'desc_seq_len']
            metadata_list.append( pd.read_csv( f'./{data_dir}/{split}_metadata.tsv', 
                                               sep='\t', 
                                               index_col=0,
                                               usecols=cols_to_keep ) )
            
            ### counts
            if not subsOnly:
                counts_file = f'./{data_dir}/{split}_AAcounts.npy'
            else:
                counts_file = f'./{data_dir}/{split}_AAcounts_subsOnly.npy'
            
            print(f'Equilibrium counts coming from: {counts_file}')
            
            with open(counts_file, 'rb') as f:
                self.AAcounts += np.load(f)
                
            del split
        
        # concatenate all data matrices
        self.data_mat = np.concatenate(data_mat_lst, axis=0)
        del data_mat_lst
        
        # little bit of post-processing after concatenating all dataframes
        self.names_df = pd.concat(metadata_list, axis=0)
        self.names_df = self.names_df.reset_index(drop=True)
        del metadata_list
        
        
    def __len__(self):
        return self.data_mat.shape[0]

    def __getitem__(self, idx):
        sample_seqs = self.data_mat[idx, :, :]
        sample_idx = idx
        return (sample_seqs, sample_idx)
    
    def max_seqlen(self):
        return self.data_mat.shape[1]
    
    def retrieve_sample_names(self, idxes):
        # used the list of sample indices to query the original names_df
        return self.names_df.iloc[idxes]
    
    def write_split_indices(self, idxes):
        # this is a method for early loading, but not lazy loading
        raise NotImplementedError
    
    def retrieve_equil_dist(self):
        return self.AAcounts / self.AAcounts.sum()
    