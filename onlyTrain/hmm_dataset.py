#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:39:48 2023

@author: annabel

About:
======
Custom pytorch dataset object; Done with PRECALCULATED COUNTS of 
  emissions and transitions


outputs:
========
1. sample_subCounts: substitution counts
2. sample_insCounts: insert counts
3. sample_transCounts: transition counts
4. sample_align_len: length of this alignment
5. sample_idx: pair index, to retrieve info from metadata_df


Data to be read:
=================
1. subCounts.npy: (num_pairs, 20, 20)
    counts of emissions at match states across whole alignment length
    (i.e. true matches and substitutions)
    
2. insCounts.npy: (num_pairs, 20)
    counts of emissions at insert states across whole alignment length

3. delCounts.npy: (num_pairs, 20)
    counts of bases that get deleted

4. transCounts.npy: (num_pairs, 3, 3)
    transition counts across whole alignment length

5. AAcounts.npy: (20, )
    equilibrium counts from whole dataset

6. metadata.tsv: [PANDAS DATAFRAME]
    metadata about each sample


"""
import torch
from torch.utils.data import Dataset, DataLoader,default_collate
import numpy as np
import jax
from jax import numpy as jnp
from jax.tree_util import tree_map
import pandas as pd


def safe_convert(mat):
    """
    pytorch doesn't support uint16 :(
    """
    # try int16 first
    int16_dtype_min = -32768
    int16_dtype_max = 32767
    
    cond1 = mat.max() <= int16_dtype_max
    cond2 = mat.min() >= int16_dtype_min
    
    if cond1 and cond2:
        return mat.astype('int16')
    
    # otherwise, return int32
    else:
        return mat.astype('int32')
    

def jax_collator(batch):
    return tree_map(jnp.asarray, default_collate(batch))


class HMMDset_PC(Dataset):
    def __init__(self, data_dir, split_prefixes, subsOnly):
        ### iterate through split prefixes and read files
        subCounts_list = []
        insCounts_list = []
        delCounts_list = []
        transCounts_list = []
        self.AAcounts = np.zeros(20, dtype=int)
        metadata_list = []
        
        for split in split_prefixes:
            # subEncoded
            with open(f'./{data_dir}/{split}_subCounts.npy', 'rb') as f:
                mat = safe_convert( np.load(f) )
                subCounts_list.append( mat )
                del mat
            
            # metadata
            cols_to_keep = ['pairID','ancestor','descendant','pfam', 'alignment_len', 'desc_seq_len']
            metadata_list.append( pd.read_csv( f'./{data_dir}/{split}_metadata.tsv', 
                                               sep='\t', 
                                               index_col=0,
                                               usecols = cols_to_keep) )
            
            if not subsOnly:
                # insCounts
                with open(f'./{data_dir}/{split}_insCounts.npy', 'rb') as f:
                    mat = safe_convert( np.load(f) )
                    insCounts_list.append( mat )
                    del mat
                
                # delCounts
                with open(f'./{data_dir}/{split}_delCounts.npy', 'rb') as f:
                    mat = safe_convert( np.load(f) )
                    delCounts_list.append( mat )
                    del mat
                
                # transCounts
                with open(f'./{data_dir}/{split}_transCounts.npy', 'rb') as f:
                    mat = safe_convert( np.load(f) )
                    transCounts_list.append( mat )
                    del mat       
                
                # counts
                with open(f'./{data_dir}/{split}_AAcounts.npy', 'rb') as f:
                    mat = safe_convert( np.load(f) )
                    self.AAcounts += mat
                    del mat
                   
            else:
                # counts
                with open(f'./{data_dir}/{split}_AAcounts_subsOnly.npy', 'rb') as f:
                    mat = safe_convert( np.load(f) )
                    self.AAcounts += mat
                    del mat
            
            del split
                
            
        ### concatenate all data matrices
        self.subCounts = np.concatenate(subCounts_list, axis=0)
        del subCounts_list
        
        if not subsOnly:
            self.insCounts = np.concatenate(insCounts_list, axis=0)
            del insCounts_list
            
            self.delCounts = np.concatenate(delCounts_list, axis=0)
            del delCounts_list
            
            self.transCounts = np.concatenate(transCounts_list, axis=0)
            del transCounts_list
        
        else:
            del insCounts_list, delCounts_list, transCounts_list
            
            num_samps = self.subCounts.shape[0]
            alphabet_size = self.subCounts.shape[1]
            
            self.insCounts = np.zeros( (num_samps, alphabet_size), 
                                       dtype='int16' )
            self.delCounts = np.zeros( (num_samps, alphabet_size),
                                       dtype='int16')
            self.transCounts = np.zeros( (num_samps, 3, 3),
                                         dtype='int16')
            
        # little bit of post-processing after concatenating all dataframes
        self.names_df = pd.concat(metadata_list, axis=0)
        self.names_df = self.names_df.reset_index(drop=True)
        del metadata_list
        
    def __len__(self):
        return self.insCounts.shape[0]

    def __getitem__(self, idx):
        sample_subCounts = self.subCounts[idx, :]
        sample_insCounts = self.insCounts[idx, :]
        sample_delCounts = self.delCounts[idx, :]
        sample_transCounts = self.transCounts[idx, :]
        sample_idx = idx
        return (sample_subCounts, sample_insCounts, sample_delCounts, 
                sample_transCounts, sample_idx)
    
    def max_seqlen(self):
        # this is only used for pair alignments, but not precalculated counts
        raise NotImplementedError
    
    def retrieve_sample_names(self, idxes):
        # used the list of sample indices to query the original names_df
        return self.names_df.iloc[idxes]
    
    def write_split_indices(self, idxes):
        # this is a method for early loading, but not lazy loading
        raise NotImplementedError
    
    def retrieve_equil_dist(self):
        return self.AAcounts / self.AAcounts.sum()
    

        

# ##############################
# ### TEST THE DATALOADER HERE #
# ##############################
# if __name__ == '__main__':
#     data_dir = 'DEV-DATA_precalculated_counts'
#     split_prefixes = ['KPROT_OOD_VALID']
#     subsOnly = False
    
#     ### initialize the pytorch dataset object
#     dset = HMMDset_PC(data_dir = data_dir,
#                      split_prefixes = split_prefixes,
#                      subsOnly = subsOnly)
    
#     ### create a dataloader that returns jax arrays
#     batch_size = len(dset)
#     dload = DataLoader(dset, 
#                         batch_size = batch_size, 
#                         shuffle = True,
#                         collate_fn = jax_collator)
    
#     # sample outputs from the first batch
#     out = list(dload)[0]
#     sample_subCounts = out[0]
#     sample_insCounts = out[1]
#     sample_delCounts = out[2]
#     sample_transCounts = out[3] 
#     sample_align_len = out[4]
#     sample_idx = out[5]
#     del out
    
#     if subsOnly:
#         assert jnp.allclose( jnp.zeros(sample_insCounts.shape), sample_insCounts)
#         assert jnp.allclose( jnp.zeros(sample_delCounts.shape), sample_delCounts)
#         assert jnp.allclose( jnp.zeros(sample_transCounts.shape), sample_transCounts)
    
#     # use PairDset.retrieve_sample_names to get the ancestor, descendant
#     # and pfam names
#     names = dset.retrieve_sample_names(sample_idx)
    
#     # test out getting the maximum sequence length
#     print(dset.max_seqlen())
    
#     # test out getting the equilibrium distribution vector
#     print(dset.retrieve_equil_dist())
    