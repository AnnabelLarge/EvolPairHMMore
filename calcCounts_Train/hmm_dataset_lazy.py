#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:39:48 2023

@author: annabel


About:
======
Custom pytorch dataset object for giving pfam data to PAIR HMM MODELS (like
  the GGI model); rewritten to be a LAZY LOADER 


outputs:
========
1. the pair alignments, categorically encoded (batch_size, 2, max_len)
   > dim1=0:ancestor
   > dim1=1: descendant

2. the length of alignments (batch_size, )

3. the sample indices, in order of accession (use retrieve_sample_names to get 
   metadata information)


Data to be read:
=================
1. Numpy matrix of sequences: a tensor of size (2, num_pairs, max_len), where 
   dim0 corresponds to-
    - (dim0=0): aligned ancestor
    - (dim0=1): aligned descendant

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

class HMMLazyDset(Dataset):
    def __init__(self, data_dir, split_prefixes):
        ### Get the sizes of the splits (i.e. the number of pairs in the split)
        split_sizes = {}
        with open(f'./{data_dir}/split_sizes.txt') as f:
            for line in f:
                splitname, splitsize = line.strip().split('\t')
                split_sizes[splitname] = int(splitsize)
        
        ### self.sample_idx_to_split_info[idx] will return the name of the 
        ###   split, as well as which sample from the split to grab
        self.sample_idx_to_split_info = []
        
        ### also calculate the total number of aas in the full dataset; if this
        ### is the training set, then you'll use the equilibrium distribution
        ### from this
        self.equl_vector = np.zeros(20,)
        
        ### loop through given prefixes
        for pre in split_prefixes:
            # fill in the index mapping
            pre_size = split_sizes[pre]
            tuplst = [(pre, i) for i in range(pre_size)]
            self.sample_idx_to_split_info = self.sample_idx_to_split_info + tuplst
            
            # read the counts to create the equilibrium vector
            with open(f'./{data_dir}/{pre}_AAcounts.npy', 'rb') as f:
                self.equl_vector += np.load(f)
            
            del pre_size, tuplst
        
        self.equl_vector = self.equl_vector / self.equl_vector.sum()
        
        self.data_dir = data_dir
        
        
    def __len__(self):
        return len(self.sample_idx_to_split_info)


    def __getitem__(self, dset_idx):
        """
        dset_idx = index for WHOLE dataset (comprised of multiple data splits)
        split_sample_idx = index within the specific data split
        
        example: say you're reading from 3 datasets- A (with three samples)
                 and B (with two samples)
        
        dset_idx   split_name    split_sample_idx
        0          A             0
        1          A             1
        2          A             2
        3          B             0
        4          B             1
        """
        ########################
        ### Retrieve splitname #
        ########################
        split_name, split_sample_idx = self.sample_idx_to_split_info[dset_idx]
        
        ##########################################
        ### Read files; retrieve specific sample #
        ##########################################
        # sequences (2, max_len)
        #with open(f'./{self.data_dir}/{split_name}_pair_alignments.npy', 'rb') as f:
        #    sample_seqs = np.load(f)[:, split_sample_idx, :]
        sample_seqs = np.load(f'./{self.data_dir}/{split_name}_pair_alignments.npy', mmap_mode='r')[:, split_sample_idx, :]
        sample_seqs = sample_seqs.copy()

        # generate the lengths entry (should just be one element...?
        sample_align_len = (sample_seqs != 0).sum(axis=1)[0]
        return (sample_seqs, sample_align_len, dset_idx)
    
    
    def max_seqlen(self):
        # get the first sample, and read its max length (not the most 
        #   efficient, but whatever)
        samp, _, _, = self.__getitem__(0)
        return samp.shape[1]
    
    
    def retrieve_sample_names(self, idxes):
        # this is a method for early loading, but not lazy loading
        raise NotImplementedError
    
    
    def write_split_indices(self, idxes):
        # given the indices, return a nicely formatted pandas dataframe
        # used these indices to read all the different metadata files, if you
        # want to retrieve sample names
        col1 = []
        col2 = []
        col3 = []
        for dset_idx in idxes:
            split_name, split_sample_idx = self.sample_idx_to_split_info[dset_idx]
            col1.append(dset_idx)
            col2.append(split_name)
            col3.append(split_sample_idx)
        out_df = pd.DataFrame({'dset_idx': col1,
                               'split_name': col2,
                               'split_sample_idx': col3})
        return out_df
    
    
    def retrieve_equil_dist(self):
        return self.equl_vector
    

        

##############################
### TEST THE DATALOADER HERE #
##############################
if __name__ == '__main__':
    data_dir = 'data_hmm'
    split_prefixes = ['KPROT_OOD_VALID']
    
    ### initialize the pytorch dataset object
    dset = HMMLazyDset(data_dir = data_dir,
                       split_prefixes = split_prefixes)
    
    ### create a dataloader that returns jax arrays
    batch_size = 10
    dload = DataLoader(dset, 
                       batch_size = batch_size, 
                       shuffle = True,
                       collate_fn = jax_collator)
   
    # sample outputs from the first batch
    seqs, lens, sample_idxes = list(dload)[0]
    
    # use PairDset.retrieve_sample_names to get the ancestor, descendant
    # and pfam names
    names = dset.retrieve_sample_names(sample_idxes)
    
    
    