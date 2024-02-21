#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:39:48 2023

@author: annabel


About:
======
Custom pytorch dataset object for giving pfam data to PAIR HMM MODELS (like
  the GGI model); rewritten to be a LAZY LOADER

Done with PRECALCULATED COUNTS of emissions and transitions, so.. might not 
  need this anyways?


outputs:
========
1. sample_subCounts: substitution counts
2. sample_insCounts: insert counts
3. sample_transCounts: transition counts
4. sample_align_len: length of this alignment
5. sample_idx: pair index, to retrieve info from metadata.tsv in external 
               postprocessing script


Data to be read:
=================
1. subCounts.npy: (num_pairs, 20, 20)
    counts of emissions at match states across whole alignment length
    (i.e. true matches and substitutions)
    
2. insCounts.npy: (num_pairs, 20)
    counts of emissions at insert states across whole alignment length

3. transCounts.npy: (num_seqs, 3, 3)
    transition counts across whole alignment length

4. AAcounts.npy: (20, )
    equilibrium counts


"""
import torch
from torch.utils.data import Dataset, DataLoader,default_collate
import numpy as np
import jax
from jax import numpy as jnp
from jax.tree_util import tree_map
import pandas as pd
from copy import deepcopy



def jax_collator(batch):
    return tree_map(jnp.asarray, default_collate(batch))


class HMMLazyDset_PC(Dataset):
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
        ### subCounts
        sample_subCounts = np.load(f'./{self.data_dir}/{split_name}_subCounts.npy', 
                                    mmap_mode='r')[split_sample_idx, :]
        sample_subCounts = deepcopy(sample_subCounts)
        
        ### insCounts
        sample_insCounts = np.load(f'./{self.data_dir}/{split_name}_insCounts.npy', 
                                    mmap_mode='r')[split_sample_idx, :]
        sample_insCounts = deepcopy(sample_insCounts)
        
        ### transCounts
        sample_transCounts = np.load(f'./{self.data_dir}/{split_name}_transCounts.npy', 
                                    mmap_mode='r')[split_sample_idx, :, :]
        sample_transCounts = deepcopy(sample_transCounts)
        
        
        ### get alignment length from transition counts
        sample_align_len = sample_transCounts.sum() - 1
        
        return (sample_subCounts, sample_insCounts, sample_transCounts, 
                sample_align_len, dset_idx)
    
    
    def max_seqlen(self):
        # get the first sample, and read its max length (not the most 
        #   efficient, but whatever)
        samp, _, _, _, _ = self.__getitem__(0)
        return samp.shape[0]
    
    
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
    data_dir = 'DATA_precomputed_counts'
    split_prefixes = ['KPROT_OOD_VALID', 'KPROT_split0']
    
    ### initialize the pytorch dataset object
    dset = HMMLazyDset_PC(data_dir = data_dir,
                         split_prefixes = split_prefixes)
    
    ### create a dataloader that returns jax arrays
    batch_size = 10
    dload = DataLoader(dset, 
                        batch_size = batch_size, 
                        shuffle = True,
                        collate_fn = jax_collator)
    
    # sample outputs 
    out = next(iter(dload))
    sample_subCounts = out[0]
    sample_insCounts = out[1]
    sample_transCounts = out[2] 
    sample_align_len = out[3]
    sample_idx = out[4]
    del out
    
    
    # use PairDset.write_split_indices to get the files and indices
    indx_df = dset.write_split_indices(sample_idx)
    
    
    