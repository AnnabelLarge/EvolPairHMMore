#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:20:17 2024

@author: annabel

Helper script to generate a training config template; have to update every
  time I add a new model... but I don't know how else to do this
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

# Dataset arguments
lines_to_write.append('\t"alphabet_size": [INT], \n')
lines_to_write.append('\t"have_precalculated_counts": [BOOL], \n')
lines_to_write.append('\t"data_dir": [STR], \n')
lines_to_write.append('\t"train_dset_splits": [list of STR], \n')
lines_to_write.append('\t"test_dset_splits": [list of STR], \n')
lines_to_write.append('\t"batch_size": [INT], \n')
lines_to_write.append('\n')

# training setup
lines_to_write.append('\t"num_epochs": [INT], \n')
lines_to_write.append('\t"learning_rate": [FLOAT], \n')
lines_to_write.append('\t"patience": [INT], \n')
lines_to_write.append('\t"early_stop_rtol": [FLOAT], \n')
lines_to_write.append('\t"loss_type": [STR: "conditional","joint"], \n')
lines_to_write.append('\t"norm_loss_by": [STR: "num_match_pos","desc_len","align_len"] \n')
lines_to_write.append('\n')

# time grid
lines_to_write.append('\t"t_grid_center": [FLOAT], \n')
lines_to_write.append('\t"t_grid_step": [FLOAT], \n')
lines_to_write.append('\t"t_grid_num_steps": [INT], \n')
lines_to_write.append('\n')



########################
### SUBSTITUTION MODEL #
########################
valid_options = ['subst_base','subst_mixture', 'subst_from_file']
to_write = ', '.join(valid_options)
subst_model_type = input(f"What substitution model? ({to_write})\n")
assert subst_model_type in valid_options, f'Invalid selection: {subst_model_type}'


lines_to_write.append(f'\t"subst_model_type": "{subst_model_type}", \n')
lines_to_write.append('\t"subsOnly": false, \n')
lines_to_write.append('\t"norm": true, \n')

### single substitution model
if subst_model_type == 'subst_base':
    lines_to_write.append('\t"exch_files": [STR], \n')


### mixture of substitution models
elif subst_model_type == 'subst_mixture':
    lines_to_write.append('\t"exch_files": [ list of STR; size=(k_subst,) ], \n')
    
    # how to handle mixture logits?
    mixture_argtype = input( ("Will you provide a list of initial logits for "+
                              "mixture components? (y/n)\n") )
    mixture_argtype = str_to_bool(mixture_argtype)
    
    if mixture_argtype:
        lines_to_write.append('\t"subst_mix_logits": [ list of FLOAT; size=(k_subst,) ], \n')
    else:
        lines_to_write.append('\t"k_subst": [INT], \n')

elif subst_model_type == 'subst_from_file':
    lines_to_write.append('\t"cond_logprobs_file": [STR], \n')
    
        
lines_to_write.append('\t"gap_tok": 43, \n') #my pipeline makes gap tokens 43

lines_to_write.append('\n')


#######################
### EQUILIBRIUM MODEL #
#######################
valid_options = ['equl_base', 'equl_deltaMixture', 'equl_dirichletMixture']
to_write = ', '.join(valid_options)
equl_model_type = input( f"What equilibrium model? ({to_write})\n" )
assert equl_model_type in valid_options, f'Invalid selection: {equl_model_type}'

lines_to_write.append(f'\t"equl_model_type": "{equl_model_type}", \n')


### extra arguments for sampling from delta mixture
if equl_model_type == 'equl_deltaMixture':
    wrote_k_equl = False
    
    # how to handle initial guesses of mixtures?
    equl_init_guess = input( ("Will you provide a list of initial "+
                              "equilibrium distribution guesses? (y/n)\n") )
    equl_init_guess = str_to_bool(equl_init_guess)
    
    if equl_init_guess:
        lines_to_write.append( ('\t"equl_vecs_transf": '+
                                '[ list of FLOAT; size=(alphabet_size, k_equl) ], \n') )
    else:
        lines_to_write.append('\t"k_equl": [INT], \n')
        wrote_k_equl = True
    
    
    # how to handle mixture logits?
    mixture_argtype = input( ("Will you provide a list of initial logits for "+
                              "mixture components? (y/n)\n") )
    mixture_argtype = str_to_bool(mixture_argtype)
    
    if mixture_argtype:
        lines_to_write.append('\t"equl_mix_logits": [ list of FLOAT; size=(k_equl,) ], \n')
    elif (not mixture_argtype) and (not wrote_k_equl):
        lines_to_write.append('\t"k_equl": [INT], \n')
    

### extra arguments for sampling from dirichlet mixture
elif equl_model_type == 'equl_dirichletMixture':
    wrote_k_equl = False
    
    # how to handle dirichlet shapes?
    dirichlet_shape_guess = input( ("Will you provide a list of initial "+
                                    "dirichlet shape guesses? (y/n)\n") )
    dirichlet_shape_guess = str_to_bool(dirichlet_shape_guess)
    
    if dirichlet_shape_guess:
        lines_to_write.append( ('\t"dirichlet_shape": '+
                                '[ list of FLOAT; size=(alphabet_size, k_equl) ], \n') )
    else:
        lines_to_write.append('\t"k_equl": [INT], \n')
        wrote_k_equl = True
        
    
    # how to handle mixture logits?
    mixture_argtype = input( ("Will you provide a list of initial logits for "+
                              "mixture components? (y/n)\n") )
    mixture_argtype = str_to_bool(mixture_argtype)
    
    if mixture_argtype:
        lines_to_write.append('\t"equl_mix_logits": [ list of FLOAT; size=(k_equl,) ], \n')
    elif (not mixture_argtype) and (not wrote_k_equl):
        lines_to_write.append('\t"k_equl": [INT], \n')
    
lines_to_write.append('\n')


###################
### INDEL MODEL   #
###################
valid_options = ['GGI_single', 'GGI_mixture', 'TKF91_single',
                 'TKF92_single', 'otherIndel_single', 'no_indel']
to_write = ', '.join(valid_options)
indel_model_type = input( f"What indel model? ({to_write})\n" )
assert indel_model_type in valid_options, f'Invalid selection: {indel_model_type}'

lines_to_write.append(f'\t"indel_model_type": "{indel_model_type}", \n')


### extra arguments for GGI_single
if indel_model_type == 'GGI_single':
    # indel rates, extension probs
    tie_params = input( "Assume one indel rate, one extension prob? (reversibility) (y/n)\n" )
    tie_params = str_to_bool(tie_params)
    
    if tie_params:
        lines_to_write.append('\t"tie_params": true, \n')
        lines_to_write.append('\t"lam": [FLOAT], \n')
        lines_to_write.append('\t"x": [FLOAT], \n')
    else:
        lines_to_write.append('\t"tie_params": false, \n')
        lines_to_write.append('\t"lam": [FLOAT], \n')
        lines_to_write.append('\t"mu": [FLOAT], \n')
        lines_to_write.append('\t"x": [FLOAT], \n')
        lines_to_write.append('\t"y": [FLOAT], \n')
    
    
    # diffrax options    
    lines_to_write.append('\t"diffrax_params": { \n')
    lines_to_write.append(    '\t\t"step": null, \n')
    lines_to_write.append(    '\t\t"rtol": [FLOAT], \n')
    lines_to_write.append(    '\t\t"atol": [FLOAT] \n')
    lines_to_write.append('\t}\n')


### extra arguments for GGI_mixture
elif indel_model_type == 'GGI_mixture':
    wrote_k_indel = False
    
    # indel rates, extension probs
    tie_params = input( "Assume one indel rate, one extension prob? (reversibility) (y/n)\n" )
    indel_param_guess = input( ("Will you provide a list of initial "+
                                "guesses for indel parameters? (y/n)\n") )
    tie_params = str_to_bool(tie_params)
    indel_param_guess = str_to_bool(indel_param_guess)
    
    if tie_params:
        lines_to_write.append('\t"tie_params": true, \n')
        
        if indel_param_guess:
            lines_to_write.append('\t"lam": [ list of FLOAT; size=(k_indel,) ], \n')
            lines_to_write.append('\t"x": [ list of FLOAT; size=(k_indel,) ], \n')
        elif not indel_param_guess:
            lines_to_write.append('\t"k_indel": [INT], \n')
            wrote_k_indel = True
    
    elif not tie_params:
        lines_to_write.append('\t"tie_params": false, \n')
        
        if indel_param_guess:
            lines_to_write.append('\t"lam": [ list of FLOAT; size=(k_indel,) ], \n')
            lines_to_write.append('\t"mu": [ list of FLOAT; size=(k_indel,) ], \n')
            lines_to_write.append('\t"x": [ list of FLOAT; size=(k_indel,) ], \n')
            lines_to_write.append('\t"y": [ list of FLOAT; size=(k_indel,) ], \n')
        elif (not indel_param_guess) and (not wrote_k_indel):
            lines_to_write.append('\t"k_indel": [INT], \n')
    
    
    # how to handle mixture logits?
    mixture_argtype = input( ("Will you provide a list of initial logits for "+
                              "mixture components? (y/n)\n") )
    mixture_argtype = str_to_bool(mixture_argtype)
    
    if mixture_argtype:
        lines_to_write.append('\t"indel_mix_logits": [ list of FLOAT; size=(k_indel,) ], \n')
    elif (not mixture_argtype) and (not wrote_k_indel):
        lines_to_write.append('\t"k_indel": [INT], \n')
    
    
    # diffrax options    
    lines_to_write.append('\t"diffrax_params": { \n')
    lines_to_write.append(    '\t\t"step": null, \n')
    lines_to_write.append(    '\t\t"rtol": [FLOAT], \n')
    lines_to_write.append(    '\t\t"atol": [FLOAT] \n')
    lines_to_write.append('\t}\n')


### extra args for TKF91
elif indel_model_type == 'TKF91_single':
    # indel rates, extension probs
    tie_params = input( "Assume one indel rate? (reversibility) (y/n)\n" )
    tie_params = str_to_bool(tie_params)
    
    if tie_params:
        lines_to_write.append('\t"tie_params": true, \n')
        lines_to_write.append('\t"lam": [FLOAT] \n')
    else:
        lines_to_write.append('\t"tie_params": false, \n')
        lines_to_write.append('\t"lam": [FLOAT], \n')
        lines_to_write.append('\t"offset": [FLOAT] \n')


### extra arguments for TKF92
elif indel_model_type == 'TKF92_single':
    # indel rates, extension probs
    tie_params = input( "Assume one indel rate, one extension prob? (reversibility) (y/n)\n" )
    tie_params = str_to_bool(tie_params)
    
    if tie_params:
        lines_to_write.append('\t"tie_params": true, \n')
        lines_to_write.append('\t"lam": [FLOAT], \n')
        lines_to_write.append('\t"x": [FLOAT] \n')
    else:
        lines_to_write.append('\t"tie_params": false, \n')
        lines_to_write.append('\t"lam": [FLOAT], \n')
        lines_to_write.append('\t"offset": [FLOAT], \n')
        lines_to_write.append('\t"x": [FLOAT], \n')
        lines_to_write.append('\t"y": [FLOAT] \n')


### extra arguments for TKF92
if indel_model_type == 'otherIndel_single':
    # pick model
    valid_models = ['LG05','RS07','KM03']
    to_write = ', '.join(valid_models)
    model_name = input( f"Which model? ({to_write})\n" )
    assert model_name in valid_models, f'INVALID MODEL: {model_name}'
    lines_to_write.append(f'\t"model_name": {model_name}, \n')
    
    
    # indel rates, extension probs
    tie_params = input( "Assume one indel rate, one extension prob? (reversibility) (y/n)\n" )
    tie_params = str_to_bool(tie_params)
    
    if tie_params:
        lines_to_write.append('\t"tie_params": true, \n')
        lines_to_write.append('\t"lam": [FLOAT], \n')
        lines_to_write.append('\t"x": [FLOAT] \n')
    else:
        lines_to_write.append('\t"tie_params": false, \n')
        lines_to_write.append('\t"lam": [FLOAT], \n')
        lines_to_write.append('\t"mu": [FLOAT], \n')
        lines_to_write.append('\t"x": [FLOAT], \n')
        lines_to_write.append('\t"y": [FLOAT] \n')


lines_to_write.append('\n')
lines_to_write.append('}\n')


with open(f'TEMPLATE_{indel_model_type}.json','w') as g:
    [g.write(line) for line in lines_to_write]


