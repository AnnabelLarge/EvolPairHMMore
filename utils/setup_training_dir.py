#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:19:13 2023

@author: annabel_large

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
import os
import shutil


def setup_training_dir(args):
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
            