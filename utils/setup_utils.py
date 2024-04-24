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
    # if nothing specified, then use subst_base
    if args.subst_model_type == 'subs_mixture':
        from model_blocks.protein_subst_models import subs_mixture as subst_model
    
    elif args.subst_model_type == 'subst_base':
        from model_blocks.protein_subst_models import subst_base as subst_model
        
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
    
    
    ########################
    ### IMPORT INDEL MODEL #
    ########################
    # if nothing specified, then use no indel model
    if args.indel_model_type == 'GGI_single':
        from model_blocks.indel_models import GGI_single as indel_model
    
    elif args.indel_model_type == 'TKF91_single':
        from model_blocks.indel_models import TKF91_single as indel_model
    
    elif args.indel_model_type == 'GGI_mixture':
        from model_blocks.indel_models import GGI_mixture as indel_model
        
    else:
        from model_blocks.indel_models import no_indel as indel_model
        args.indel_model_type = 'no_indel'
        
    
    
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
    logfile_msg1 = (f'1.) substitution model: {args.subst_model_type};'+
                    f' (norm: {args.norm})\n')
    
    equl_model_instance = equl_model()
    logfile_msg2 = (f'2.) equilibrium distribution: {args.equl_model_type}\n')
    
    if 'tie_params' in vars(args).keys():
        indel_model_instance = indel_model(args.tie_params)
        logfile_msg3 = (f'3.) indel model: {args.indel_model_type};'+
                        f'(tie_parms: {args.tie_params})\n')
    else:
        indel_model_instance = indel_model()
        logfile_msg3 = (f'3.) indel model: {args.indel_model_type}')
        
    
    # add this model info to the top of the logfile
    out_logfile_msg = ('TRAINING PairHMM composed of:\n' +
                        logfile_msg1 +
                        logfile_msg2 +
                        logfile_msg3 
                        )
    
    return (subst_model_instance, equl_model_instance, indel_model_instance, 
            out_logfile_msg)




def setup_training_dir(args):
    """
    Before training your model, organize your working directory, and make sure you
    aren't overwriting any previous data

    Organization of the working directory for training neural nets:

      args.training_wkdir
      |
      | - tboard
      |   |
      |   |- {args.runname}_1
      |   |- {args.runname}_2
      |   |- {args.runname}_n
      |
      | - model checkpoints
      |   |
      |   |- {args.runname}_1
      |   |  |
      |   |  | - ANC_ENC
      |   |  | - DESC_DEC
      |   |  | - OUT_PROJ
      |   |
      |   |- {args.runname}_2
      |   |  |
      |   |  | - ANC_ENC
      |   |  | - DESC_DEC
      |   |  | - OUT_PROJ
      |   |
      |   |- {args.runname}_n
      |
      | - logfiles
      |   |
      |   |- {args.runname}_1_training-prog.log
      |   |- {args.runname}_2_training-prog.log
      |   |- {args.runname}_n_training-prog.log
      

    If training HMM model, then model checkpoints will not contain the 
      (ANC_ENC, DESC_DEC, OUT_PROJ) sub folders; they will just contain flat 
      tsv files containing model params (or pickles, mayeb change this later)
    """
    ### create folder/file names
    tboard_dir = f'{os.getcwd()}/{args.training_wkdir}/tboard/{args.runname}'
    model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts/{args.runname}'
    logfile_dir = f'{os.getcwd()}/{args.training_wkdir}/logfiles'
    logfile_filename = f'{args.runname}_training-prog.log'
    
    
    ### IF TRAINING WKDIR HAS NOT BEEN CREATED YET
    if not os.path.exists(f'{os.getcwd()}/{args.training_wkdir}'):
        os.mkdir(f'{os.getcwd()}/{args.training_wkdir}')
        os.mkdir(logfile_dir)
    
    
    ### IF TRAINING WKDIR ALREAD EXISTS
    elif os.path.exists(f'{os.getcwd()}/{args.training_wkdir}'):
        ### make sure the tensorboard and orbax directories don't exist
        for path_to_check in [tboard_dir, model_ckpts_dir]:
            err_msg = f'{path_to_check} ALREADY EXISTS; DOES IT HAVE DATA?'
            assert not os.path.exists(path_to_check), err_msg
        
        ### make sure the logfile for the given run doesn't exist
        err_msg = f'{logfile_dir}/{logfile_filename} ALREADY EXISTS; DOES IT HAVE DATA?'
        assert logfile_filename not in os.listdir(logfile_dir), err_msg
      
        
    ### add these filenames to the args dictionary, to be passed to training
    ### script
    args.tboard_dir = tboard_dir
    args.model_ckpts_dir = model_ckpts_dir
    args.logfile_name = f'{logfile_dir}/{logfile_filename}'
            