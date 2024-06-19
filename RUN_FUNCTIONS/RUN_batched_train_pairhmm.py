#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:18:49 2024

@author: annabel

ABOUT:
======
same as train_pairhmm, but for a list of JSON configs using the same dataset
  (dataset retrieved from first file)

"""

if __name__ == '__main__':
    import json
    import argparse 
    import os
    
    from RUN_FUNCTIONS.RUN_train_pairhmm import train_pairhmm
    from utils.init_dataloaders import init_dataloaders
    
    
    # for now, running models on single GPU
    err_ms = 'SELECT GPU TO RUN THIS COMPUTATION ON with CUDA_VISIBLE_DEVICES=DEVICE_NUM'
    assert len(jax.devices()) == 1, err_ms
    del err_ms
    
    
    ### INITIALIZE PARSER
    parser = argparse.ArgumentParser(prog='train_batch')
    
    # config files required to run
    parser.add_argument('--config-folder',
                        type=str,
                        required=True,
                        help='Load configs from this folder, in json format.')
    
    # parse the arguments
    init_args = parser.parse_args()
    
    
    ### MAIN PROGRAM
    # find all the json files in the folder
    file_lst = [file for file in os.listdir(init_args.config_folder) if file.endswith('.json')]
    
    # read the first config file to load data
    init_config_file = f'./{init_args.config_folder}/{file_lst[0]}'
    print(f'LOADING DATA FROM: {init_config_file}')
    with open(init_config_file, 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        init_config_args = parser.parse_args(namespace=t_args)
    data_tup = load_all_data(init_config_args)
    
    
    # iterate through all config files with this same data tuple
    for config_file in file_lst:
        print(f'STARTING TRAINING FROM: {config_file}')
        to_open = f'./{init_args.config_folder}/{config_file}'
        with open(to_open, 'r') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            this_config_args = parser.parse_args(namespace=t_args)
        
        # run training function with this config file
        train_pairhmm(this_config_args, data_tup)
        
        print()
