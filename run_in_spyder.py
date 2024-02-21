#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:25:54 2024

@author: annabel

ABOUT:
=======
Load a config file and run training
Basically the same as the CLI, but now I can monitor variables in IDE
"""
import json
import argparse 

from train_ggi import train_ggi


config_file = 'GGI_config_template.json'




# INITIALIZE PARSER
parser = argparse.ArgumentParser(prog='GGI')

# # config files required to run
# parser.add_argument('--config_file',
#                     type = str,
#                     required=True,
#                     help='Load configs from file in json format.')

# parse the arguments
args = parser.parse_args()
args.config_file = config_file

with open(args.config_file, 'r') as f:
    t_args = argparse.Namespace()
    t_args.__dict__.update(json.load(f))
    args = parser.parse_args(namespace=t_args)



# run training function
train_ggi(args)


