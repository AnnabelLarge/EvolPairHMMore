#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:08:13 2024

@author: annabel
"""
import jax
import json
import argparse 
import os

from cli.train_pairhmm import train_pairhmm as train_hmm
from cli.eval_pairhmm import eval_pairhmm as eval_hmm
from cli.eval_subs_only import eval_subs_only as eval_hmm_subs_only
from utils.init_dataloaders import init_dataloaders



# for now, running models on single GPU
err_ms = 'SELECT GPU TO RUN THIS COMPUTATION ON with CUDA_VISIBLE_DEVICES=DEVICE_NUM'
assert len(jax.devices()) == 1, err_ms
del err_ms


#######################
### INITIALIZE PARSER #
#######################
parser = argparse.ArgumentParser(prog='EvolPairHMMore')

## which program do you want to run?
valid_tasks = ['train_hmm','eval_hmm','eval_hmm_subs_only']
valid_tasks = valid_tasks + [elem + '_batched' for elem in valid_tasks]

parser.add_argument( '-task',
                      type=str,
                      required=True,
                      choices = valid_tasks,
                      help='What do you want to do?' )

# config files required to run
parser.add_argument('-configs',
                    type = str,
                    required=True,
                    help='Load configs from file or folder of files, in json format')


### parse the arguments
args = parser.parse_args()

# args.task = 'train_hmm_batched'
# args.configs = 'CONFIGS_pairHMM'


### helper function to open a single config file and extract additional arguments
def read_config_file(args, config_file):
    with open(config_file, 'r') as f:
        contents = json.load(f)
        
        t_args = argparse.Namespace()
        t_args.__dict__.update(contents)
        args = parser.parse_args(namespace=t_args)
    return args


########################################################
### OPTION 1: train_hmm (or in batched mode)   #########
########################################################
if args.task == 'train_hmm':
    assert args.configs.endswith('.json'), print("input is one JSON file")
    print(f'TRAINING WITH: {args.configs}')
    args = read_config_file(args, args.configs)
    
    dataloader_lst = init_dataloaders(args, onlyTest=False)
    train_hmm(args = args, 
              dataloader_lst = dataloader_lst)
        

elif args.task == 'train_hmm_batched':
    file_lst = [file for file in os.listdir(args.configs) if file.endswith(".json")]
    
    # use the first config file to get the training and test dataloaders
    print(f'Creating dataset and dataloaders from: {args.configs}/{file_lst[0]}')
    first_config_args = read_config_file(args, f'./{args.configs}/{file_lst[0]}')
    dataloader_lst = init_dataloaders(first_config_args, onlyTest=False)
    
    # then loop through all the files, providing the dataloaders
    for config_file in file_lst:
        print(f'TRAINING WITH: {args.configs}/{config_file}')
        trial_args = read_config_file(args, f'./{args.configs}/{config_file}')
        train_hmm(args = trial_args, 
                  dataloader_lst = dataloader_lst)
        
        del trial_args
    


########################################################
### OPTION 2: eval_hmm (or in batched mode)   ##########
########################################################
elif args.task == 'eval_hmm':
    assert args.configs.endswith('.json'), print("input is one JSON file")
    print(f'EVALUATING: {args.configs}')
    args = read_config_file(args, args.configs)
    
    dataloader_lst = init_dataloaders(args, onlyTest=True)
    eval_hmm(args = args, 
             dataloader_lst = dataloader_lst)


elif args.task == 'eval_hmm_batched':
    file_lst = [file for file in os.listdir(args.configs) if file.endswith(".json")]
    
    # use the first config file to get the dataloaders
    print(f'Creating dataset and dataloaders from: {args.configs}/{file_lst[0]}')
    first_config_args = read_config_file(args, f'./{args.configs}/{file_lst[0]}')
    dataloader_lst = init_dataloaders(first_config_args, onlyTest=True)
    
    # then loop through all the files, providing the dataloaders
    for config_file in file_lst:
        print(f'EVALUATING: {args.configs}/{config_file}')
        trial_args = read_config_file(args, f'./{args.configs}/{config_file}')
        eval_hmm(args = trial_args, 
                 dataloader_lst = dataloader_lst)
        
        del trial_args


########################################################
### OPTION 3: eval_hmm_subs_only (or in batched mode   #
########################################################
# a special case of eval, where you ignore all indel positions and score
#   substitutions with a specific scoring matrix
# provide both a "train" and "test dataset, but you'll just be evaluating
#   on both of them (no training done)
elif args.task == 'eval_hmm_subs_only':
    assert args.configs.endswith('.json'), print("input is one JSON file")
    print(f'EVALUATING (and ignoring indels): {args.configs}')
    args = read_config_file(args, args.configs)
    
    dataloader_lst = init_dataloaders(args, onlyTest=True)
    eval_hmm_subs_only(args = args, 
                       dataloader_lst = dataloader_lst)


elif args.task == 'eval_hmm_subs_only_batched':
    file_lst = [file for file in os.listdir(args.configs) if file.endswith(".json")]
    
    # use the first config file to get the dataloaders
    print(f'Creating dataset and dataloaders from: {args.configs}/{file_lst[0]}')
    first_config_args = read_config_file(args, f'./{args.configs}/{file_lst[0]}')
    dataloader_lst = init_dataloaders(first_config_args, onlyTest=True)
    
    # then loop through all the files, providing the dataloaders
    for config_file in file_lst:
        print(f'EVALUATING (and ignoring indels): {args.configs}/{config_file}')
        trial_args = read_config_file(args, f'./{args.configs}/{config_file}')
        eval_hmm_subs_only(args = trial_args, 
                           dataloader_lst = dataloader_lst)
        
        del trial_args

