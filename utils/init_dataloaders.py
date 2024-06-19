#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:15:09 2024

@author: annabel
"""
from torch.utils.data import DataLoader


def init_dataloaders(args):
    ### DECIDE TRAINING MODE
    # previous version of code allowed the option for lazy dataloading, but
    #   since GGI model is so small, just get rid of that
    assert args.loadtype == 'eager'
    
    # Use this training mode if you already have precalculated count matrices
    if args.have_precalculated_counts:
        from onlyTrain.hmm_dataset import HMMDset_PC as hmm_reader
        from onlyTrain.hmm_dataset import jax_collator as collator
        
    # Use this training mode if you need to calculate emission and transition  
    #   counts from pair alignments
    elif not args.have_precalculated_counts:
        from calcCounts_Train.hmm_dataset import HMMDset as hmm_reader
        from calcCounts_Train.hmm_dataset import jax_collator as collator
        
        
    ### READ WITH PYTORCH DATALOADERS    
    # training data
    print(f'Creating DataLoader for training set with {args.train_dset_splits}')
    training_dset = hmm_reader(data_dir = args.data_dir, 
                                split_prefixes = args.train_dset_splits,
                                subsOnly = args.subsOnly)
    training_dl = DataLoader(training_dset, 
                              batch_size = args.batch_size, 
                              shuffle = True,
                              collate_fn = collator)
    
    # test data
    print(f'Creating DataLoader for test set with {args.test_dset_splits}')
    test_dset = hmm_reader(data_dir = args.data_dir, 
                              split_prefixes = args.test_dset_splits,
                              subsOnly = args.subsOnly)
    test_dl = DataLoader(test_dset, 
                          batch_size = args.batch_size, 
                          shuffle = False,
                          collate_fn = collator)
    
    # wrap output into a tuple
    out_tup = (training_dset, training_dl, test_dset, test_dl)
    
    return out_tup