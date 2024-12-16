#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:15:09 2024

@author: annabel
"""
import torch
import random
import numpy as np
from torch.utils.data import DataLoader


def init_dataloaders(args, onlyTest=False):
    #########################################################
    ### set random seeds for numpy and pytorch separately   #
    #########################################################
    torch.manual_seed(args.rng_seednum)
    random.seed(args.rng_seednum)
    np.random.seed(args.rng_seednum)
    
    
    ############################
    ### DECIDE TRAINING MODE   #
    ############################
    # # previous version of code allowed the option for lazy dataloading, but
    # #   since GGI model is so small, just get rid of that
    # assert args.loadtype == 'eager'
    
    # Use this training mode if you already have precalculated count matrices
    if args.have_precalculated_counts:
        from onlyTrain.hmm_dataset import HMMDset_PC as hmm_reader
        from onlyTrain.hmm_dataset import jax_collator as collator
        
    # Use this training mode if you need to calculate emission and transition  
    #   counts from pair alignments
    elif not args.have_precalculated_counts:
        from calcCounts_Train.hmm_dataset import HMMDset as hmm_reader
        from calcCounts_Train.hmm_dataset import jax_collator as collator
    
    
    ###################
    ### manage time   #
    ###################
    # with geometrically-spaced times
    if args.times_from == 'geometric':
        t_grid_center = args.t_grid_center
        t_grid_step = args.t_grid_step
        t_grid_num_steps = args.t_grid_num_steps
        
        quantization_grid = range( -(t_grid_num_steps-1), 
                                   t_grid_num_steps, 
                                   1
                                  )
        times_from_array = np.array([ (t_grid_center * t_grid_step**q_i) 
                                      for q_i in quantization_grid
                                     ]
                                    )
        single_time_from_file = False
        
    # with one time per sample, read from file
    # need to make sure batch size is one for this mode
    elif args.times_from == 'from_file':
        times_from_array = None
        single_time_from_file = True
        assert args.batch_size == 1
        
        
    ######################################
    ### READ WITH PYTORCH DATALOADERS    #
    ######################################
    # test data
    print(f'Creating DataLoader for test set with {args.test_dset_splits}')
    test_dset = hmm_reader(data_dir = args.data_dir, 
                           split_prefixes = args.test_dset_splits,
                           subsOnly = args.subsOnly,
                           times_from_array = times_from_array,
                           single_time_from_file = single_time_from_file,
                           toss_alignments_longer_than = args.toss_alignments_longer_than)
    test_dl = DataLoader(test_dset, 
                         batch_size = args.batch_size, 
                         shuffle = False,
                         collate_fn = collator)
    
    # wrap output into a list
    out_lst = [test_dset, test_dl]
    
    if not onlyTest:
        # training data
        print(f'Creating DataLoader for training set with {args.train_dset_splits}')
        training_dset = hmm_reader(data_dir = args.data_dir, 
                                   split_prefixes = args.train_dset_splits,
                                   subsOnly = args.subsOnly,
                                   times_from_array = times_from_array,
                                   single_time_from_file = single_time_from_file,
                                   toss_alignments_longer_than = args.toss_alignments_longer_than)
        training_dl = DataLoader(training_dset, 
                                 batch_size = args.batch_size, 
                                 shuffle = True,
                                 collate_fn = collator)
        
        # add to out list
        out_lst = [training_dset, training_dl] + out_lst
    
    return out_lst