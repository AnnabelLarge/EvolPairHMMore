#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:20:17 2024

@author: annabel

Helper script to generate an eval config for scoring only substitution sites
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
lines_to_write.append('\t"training_wkdir": [STR], \n')
lines_to_write.append('\t"rng_seednum": [INT], \n')
lines_to_write.append('\t"debug": [BOOL], \n')
lines_to_write.append('\n')

# Dataset arguments, how to score
lines_to_write.append('\t"alphabet_size": [INT], \n')
lines_to_write.append('\t"have_precalculated_counts": [BOOL], \n')
lines_to_write.append('\t"data_dir": [STR], \n')
lines_to_write.append('\t"train_dset_splits": [list of STR], \n')
lines_to_write.append('\t"test_dset_splits": [list of STR], \n')
lines_to_write.append('\t"batch_size": [INT], \n')
lines_to_write.append('\t"loss_type": [STR: "conditional","joint"], \n')
lines_to_write.append('\t"norm_loss_by": [STR: "desc_len","align_len"] \n')
lines_to_write.append('\n')

# time grid
lines_to_write.append('\t"t_grid_center": [FLOAT], \n')
lines_to_write.append('\t"t_grid_step": [FLOAT], \n')
lines_to_write.append('\t"t_grid_num_steps": [INT], \n')
lines_to_write.append('\n')



########################
### SUBSTITUTION MODEL #
########################
valid_options = ['subst_base','subst_mixture']
to_write = ', '.join(valid_options)
subst_model_type = input(f"What substitution model? ({to_write})\n")
assert subst_model_type in valid_options, f'Invalid selection: {subst_model_type}'


lines_to_write.append(f'\t"subst_model_type": "{subst_model_type}", \n')
lines_to_write.append('\t"subsOnly": true, \n')
lines_to_write.append('\t"norm": true, \n')

### single substitution model
if subst_model_type == 'subst_base':
    lines_to_write.append('\t"exch_files": [STR], \n')


### mixture of substitution models
elif susbt_model_type == 'subst_mixture':
    lines_to_write.append('\t"exch_files": [ list of STR; size=(k_subst,) ], \n')
    
    # how to handle mixture logits?
    mixture_argtype = input( ("Will you provide a list of initial logits for "+
                              "mixture components? (y/n)\n") )
    mixture_argtype = str_to_bool(mixture_argtype)
    
    if mixture_argtype:
        lines_to_write.append('\t"subst_mix_logits": [ list of FLOAT; size=(k_subst,) ], \n')
    else:
        lines_to_write.append('\t"k_subst": [INT], \n')
    
        
lines_to_write.append('\t"gap_tok": 43, \n') #my pipeline makes gap tokens 43

lines_to_write.append('\n')


###################
### FINAL LINES   #
###################
lines_to_write.append('\t"equl_model_type": "equl_base", \n')
lines_to_write.append('\n')
lines_to_write.append('\t"indel_model_type": null \n')
lines_to_write.append('\n')
lines_to_write.append('}\n')


with open(f'TEMPLATE.json','w') as g:
    [g.write(line) for line in lines_to_write]


