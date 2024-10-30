#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:19:13 2023

@author: annabel_large


"""
import jax
from jax import tree_util
import os
import shutil


def model_import_register(args):
    """
    from argparse object, figure out which models to import, register them
      as pytrees, and return; also add a logfile message
    """
    ########################
    ### IMPORT SUBST MODEL #
    ########################
    if args.subst_model_type == 'subst_base':
        from model_blocks.protein_subst_models import subst_base as subst_model
        logfile_msg1 = (f'1.) substitution model: subst_base;'+
                        f' (norm: {args.norm})\n')
    
    elif args.subst_model_type == 'subst_mixture':
        from model_blocks.protein_subst_models import subst_mixture as subst_model
        logfile_msg1 = (f'1.) substitution model: subst_mixture;'+
                        f' (norm: {args.norm}, num_mixes: {args.k_subst})\n')
    
    elif args.subst_model_type == 'subst_from_file':
        from model_blocks.protein_subst_models import subst_from_file as subst_model
        logfile_msg1 = (f'1.) substitution model: subst_from_file;'+
                        f' (from file: {args.cond_logprobs_file})\n')

    elif args.subst_model_type == 'hky85':
        from model_blocks.dna_subst_models import hky85 as subst_model
        logfile_msg1 = (f'1.) substitution model: HKY85 (DNA model);'+
                        f' (norm: {args.norm})\n')
        
    ### come back to this model if you make rate classes depend on other 
    ###   factors (this is kind of useless right now)
    # elif args.subst_model_type == 'LG_mixture':
    #     from model_blocks.protein_subst_models import LG_mixture as subst_model
    
    
    ####################################
    ### IMPORT EQUL DISTRIBUTION MODEL #
    ####################################
    # if nothing specified, then use equl_base
    if args.equl_model_type == 'equl_deltaMixture':
        from model_blocks.equl_distr_models import equl_deltaMixture  as equl_model
    
    elif args.equl_model_type == 'equl_dirichletMixture':
        from model_blocks.equl_distr_models import equl_dirichletMixture  as equl_model
    
    else:
        from model_blocks.equl_distr_models import equl_base as equl_model
        args.equl_model_type = 'equl_base'
    
    # all models can use the same logfile template
    logfile_msg2 = (f'2.) equilibrium distribution: {args.equl_model_type}\n')
    
    
    ########################
    ### IMPORT INDEL MODEL #
    ########################
    # if nothing specified, then use no indel model
    if args.indel_model_type == 'GGI_single':
        from model_blocks.indel_models import GGI_single as indel_model
        logfile_msg3 = (f'3.) indel model: GGI_single;'+
                        f'(tie_parms: {args.tie_params})\n')
    
    elif args.indel_model_type == 'TKF91_single':
        from model_blocks.indel_models import TKF91_single as indel_model
        logfile_msg3 = (f'3.) indel model: TKF91_single;\n'+
                        f'(tie_parms: {args.tie_params})\n')
    
    elif args.indel_model_type == 'TKF92_single':
        from model_blocks.indel_models import TKF92_single as indel_model
        logfile_msg3 = (f'3.) indel model: TKF92_single;\n'+
                        f'(tie_parms: {args.tie_params})\n')
    
    elif args.indel_model_type == 'otherIndel_single':
        from model_blocks.indel_models import otherIndel_single as indel_model
        logfile_msg3 = (f'3.) indel model: {args.model_name}_single;\n'+
                        f'(tie_parms: {args.tie_params})\n')
        
    elif args.indel_model_type == 'GGI_mixture':
        from model_blocks.indel_models import GGI_mixture as indel_model
        logfile_msg3 = (f'3.) indel model: GGI_mixture;'+
                        f' (tie_parms: {args.tie_params}, num_mixes: {args.k_indel})\n')
        
    else:
        from model_blocks.indel_models import no_indel as indel_model
        args.indel_model_type = 'no_indel'
        logfile_msg3 = (f'3.) indel model: no_indel\n')
    
    
    #################################
    ### REGISTER CLASSES AS PYTREES #
    #################################
    # use some crude try/except handling, in case these are already registered
    #   this is logged as a ValueError
    for customClass in [subst_model, equl_model, indel_model]:
        try:
            tree_util.register_pytree_node(customClass,
                                           customClass._tree_flatten,
                                           customClass._tree_unflatten)
        except ValueError:
            pass

    
    ################
    ### INITIALIZE #
    ################
    subst_model_instance = subst_model(args.norm)
    equl_model_instance = equl_model()
    
    if args.indel_model_type in ['GGI_single', 'GGI_mixture', 'TKF91_single','TKF92_single']:
        indel_model_instance = indel_model(args.tie_params)
    elif args.indel_model_type == 'otherIndel_single':
        indel_model_instance = indel_model(args.tie_params, args.model_name)        
    else:
        indel_model_instance = indel_model()
    
    # also record what loss you're using to train
    logfile_msg4 = f'likelihood fn: {args.loss_type}\n'
        
    
    # add this model info to the top of the logfile
    out_logfile_msg = ('TRAINING PairHMM composed of:\n' +
                        logfile_msg1 +
                        logfile_msg2 +
                        logfile_msg3 +
                        logfile_msg4
                        )
    
    return (subst_model_instance, equl_model_instance, indel_model_instance, 
            out_logfile_msg)




def setup_training_dir(args):
    """
    Create training directory

    Before training your model, organize your working directory, and make sure you
    aren't overwriting any previous data
    """
    ### create folder/file names
    tboard_dir = f'{os.getcwd()}/{args.training_wkdir}/tboard/{args.training_wkdir}'
    model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
    logfile_dir = f'{os.getcwd()}/{args.training_wkdir}/logfiles'
    
    # create logfile in the logfile_dir
    logfile_filename = f'training-prog.log'
    
    
    ### IF TRAINING WKDIR HAS NOT BEEN CREATED YET
    if not os.path.exists(f'{os.getcwd()}/{args.training_wkdir}'):
        os.mkdir(f'{os.getcwd()}/{args.training_wkdir}')
        os.mkdir(model_ckpts_dir)
        os.mkdir(logfile_dir)
        # tensorboard directory takes care of itself
    
    
    ### IF TRAINING WKDIR ALREAD EXISTS, DON'T OVERWRITE
    elif os.path.exists(f'{os.getcwd()}/{args.training_wkdir}'):
        raise RuntimeError(f'{args.training_wkdir} ALREADY EXISTS; DOES IT HAVE DATA?')
        
        
    ### add these filenames to the args dictionary, to be passed to training
    ### script
    args.tboard_dir = tboard_dir
    args.model_ckpts_dir = model_ckpts_dir
    args.logfile_dir = logfile_dir
    args.logfile_name = f'{logfile_dir}/{logfile_filename}'
            