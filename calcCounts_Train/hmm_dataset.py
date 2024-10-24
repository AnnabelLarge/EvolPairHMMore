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
1. Numpy matrix of sequences: a tensor of size (num_pairs, max_len, 2), where 
   dim1 corresponds to-
    - (dim1=0): aligned ancestor
    - (dim1=1): aligned descendant

  All sequences have been categorically encoded (20 possible aa tokens + pad token)
  
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
            ### sequences
            with open(f'./{data_dir}/{split}_pair_alignments.npy', 'rb') as f:
                data_mat_lst.append(np.load(f))
            
            ### metadata
            cols_to_keep = ['pairID','ancestor','descendant','pfam', 'desc_seq_len']
            metadata_list.append( pd.read_csv( f'./{data_dir}/{split}_metadata.tsv', 
                                               sep='\t', 
                                               index_col=0,
                                               use_cols=cols_to_keep ) )
            
            ### counts
            if not subsOnly:
                counts_file = f'./{data_dir}/{split}_AAcounts.npy'
            else:
                counts_file = f'./{data_dir}/{split}_AAcounts_subsOnly.npy'
            
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
        
        # generate the lengths matrix
        self.lengths_vec = np.sum( (self.data_mat != 0), axis=1)[:,0]
        
        
    def __len__(self):
        return self.data_mat.shape[0]

    def __getitem__(self, idx):
        sample_seqs = self.data_mat[idx, :, :]
        sample_align_len = self.lengths_vec[idx]
        sample_idx = idx
        return (sample_seqs, sample_align_len, sample_idx)
    
    def max_seqlen(self):
        return self.data_mat.shape[1]
    
    def retrieve_sample_names(self, idxes):
        # used the list of sample indices to query the original names_df
        return self.names_df.iloc[idxes]
    
    def retrieve_desc_lens(self, idxes):
        # used to return the length of the descendant sequence
        return self.names_df.iloc[idxes]['desc_seq_len'].to_numpy()
    
    def write_split_indices(self, idxes):
        # this is a method for early loading, but not lazy loading
        raise NotImplementedError
    
    def retrieve_equil_dist(self):
        return self.AAcounts / self.AAcounts.sum()
    

        

##############################
### TEST THE DATALOADER HERE #
##############################
if __name__ == '__main__':
    # just testing that code works when subsOnly = True; FiveSamp_AAcounts is 
    #   the same file as FiveSamp_AAcounts_subsOnly
    subsOnly = True
    data_dir = 'DEV-DATA_pair_alignments'
    split_prefixes = ['fiftySamps']
    
    ### initialize the pytorch dataset object
    dset = HMMDset(data_dir = data_dir,
                   split_prefixes = split_prefixes,
                   subsOnly = subsOnly)
    
    
    ### create a dataloader that returns jax arrays
    batch_size = len(dset)
    dload = DataLoader(dset, 
                       batch_size = batch_size, 
                       shuffle = True,
                       collate_fn = jax_collator)
    
    # sample outputs from the first batch
    seqs, lens, sample_idxes = list(dload)[0]
    
    # use PairDset.retrieve_sample_names to get the ancestor, descendant
    # and pfam names
    names = dset.retrieve_sample_names(sample_idxes)
    
    # test out getting the maximum sequence length
    print(dset.max_seqlen())
    
    # test out getting the equilibrium distribution vector
    print(dset.retrieve_equil_dist())
    
    
    