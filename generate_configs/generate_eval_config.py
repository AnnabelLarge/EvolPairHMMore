#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:20:17 2024

@author: annabel

Helper script to generate an eval config 
"""
def str_to_bool(my_string):
    if my_string in ['y', 'yes']:
        return True
    elif my_string in ['n', 'no']:
        return False
    else:
        raise ValueError(f'invalid input: {my_string}')


lines_to_write = ['{\n']

#########################
### SAME FOR ALL MODELS #
#########################
# General run setup
lines_to_write.append('\t"eval_wkdir": [STR], \n')
lines_to_write.append('\t"rng_seednum": [INT], \n')
lines_to_write.append('\t"debug": [BOOL], \n')
lines_to_write.append('\n')

# where to read other arguments
lines_to_write.append('\t"training_wkdir": [STR], \n')
lines_to_write.append('\n')

# Dataset arguments, how to score
lines_to_write.append('\t"have_precalculated_counts": [BOOL], \n')
lines_to_write.append('\t"data_dir": [STR], \n')
lines_to_write.append('\t"test_dset_splits": [list of STR], \n')
lines_to_write.append('\t"batch_size": [INT], \n')
lines_to_write.append('\n')

lines_to_write.append('\t"subsOnly": true \n')
lines_to_write.append('\n')
lines_to_write.append('}\n')


with open(f'TEMPLATE.json','w') as g:
    [g.write(line) for line in lines_to_write]


