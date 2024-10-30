#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:11:42 2024

@author: annabel

Fake argparse object for code development of single indel models
"""
import jax
from jax import numpy as jnp
import numpy as np
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None # this is annoying

from utils.setup_utils import model_import_register



def generate_fake_model(args):
    ### register models as pytrees
    out = model_import_register(args)
    subst_model, equl_model, indel_model, _ = out
    del out
    
    
    ### initialize the equlibrium distribution(s)
    equl_model_params, equl_model_hparams = equl_model.initialize_params(argparse_obj=args)
    
    # if this is the base model, use the equilibrium distribution from 
    #   fake data
    if args.equl_model_type == 'equl_base':
        random_vec = jax.random.normal( key = jax.random.key(0),
                                        shape = jnp.zeros((args.alphabet_size,)).shape )
        fake_equl_vec = jax.nn.softmax(random_vec)
        equl_model_hparams['equl_vecs_fromData'] = fake_equl_vec
    
    # if you're not scoring emissions from indels at all, use this placeholder
    elif args.equl_model_type == 'no_equl':
        equl_model_hparams['equl_vecs_fromData'] = jnp.zeros((args.alphabet_size,))
    
    
    ### initialize the substitution model
    subst_model_params, subst_model_hparams = subst_model.initialize_params(argparse_obj=args)
    
    
    ### initialize the indel model
    indel_model_params, indel_model_hparams = indel_model.initialize_params(argparse_obj=args)
    
    
    ### Combine all into one   
    params = {**equl_model_params, **subst_model_params, **indel_model_params}
    hparams = {**equl_model_hparams, **subst_model_hparams, **indel_model_hparams}
    pairHMM = (equl_model, subst_model, indel_model)

    # add grid step to hparams dictionary; needed for marginaling over time
    hparams['t_grid_step']= args.t_grid_step
    
    return (pairHMM, params, hparams)
    
    
class FakeArgparse:
    def __init__(self):
       	self.training_wkdir= "EXAMPLE_OUT"
       	self.rng_seednum= 0
        self.debug=True
       
       	self.alphabet_size= 20
       	self.batch_size= 40
       	self.have_precalculated_counts= True
           
       	self.data_dir= "DEV_hmm_precalc_counts"
       	self.train_dset_splits= ["PF00001"]
       	self.test_dset_splits= ["PF00001"]
       
       	self.num_epochs= 2
       	self.learning_rate= 0.001
       	self.patience= 2
       	self.loss_type= "conditional"
       	self.norm_loss_by='align_len'
       	self.early_stop_rtol=1e-5
       
       	self.t_grid_center= 0.1
       	self.t_grid_step= 1.1
       	self.t_grid_num_steps= 20
       
       	self.subst_model_type= "subst_base"
       	self.norm= True
       	self.subsOnly= False
       	self.exch_files= "LG08_exchangeability_r.npy"
       	self.gap_tok= 43
       
       	self.equl_model_type= "equl_base"
       
        
        ##############################################
        ### uncomment to test different indel models #
        ##############################################
        # ### TKF91 model
       	# self.indel_model_type= "TKF91_single"
       	# self.tie_params= True
        # self.lam = 0.5
       
        
        # ### TKF92 model
       	# self.indel_model_type= "TKF92_single"
       	# self.tie_params= True
        # self.lam = 0.5
        # self.x=0.5
        
        
        # ### other single models
       	# self.indel_model_type= "otherIndel_single"
        # self.model_name="LG05"
       	# self.tie_params= True
        # self.lam = 0.5
        # self.x=0.5
        
        
        ### H20 model
       	self.indel_model_type= "GGI_single"
       	self.tie_params= True
           
       	self.lam= 0.5
       	self.x= 0.5
       	self.diffrax_params= {'step': None,
                                  'rtol': 1e-3,
                                  'atol': 1e-6}
        
        
        # ### no indel model
        # self.indel_model_type = None





if __name__ == '__main__':
    import os
    import shutil
    
    from cli.train_pairhmm import train_pairhmm
    from utils.init_dataloaders import init_dataloaders
    
    
    ### init a fake model, according to above
    args = FakeArgparse()
    
    if 'EXAMPLE_OUT' in os.listdir():
        shutil.rmtree('EXAMPLE_OUT')
    
    # load data
    dataloader_tup = init_dataloaders(args)
    
    # run training function
    train_pairhmm(args, dataloader_tup)
    
    
    
    
    
    
    
    
    
    
    
    
    
    # out = generate_fake_model(args)
    # pairHMM, params, hparams = out
    # del out
    
    
    # ### load some data
    # from utils.init_dataloaders import init_dataloaders
    # out = init_dataloaders(args)
    # training_dset, training_dl, _, _ = out
    # del out
    
    # # get first batch
    # batch = list(training_dl)[0]
    # allCounts = (batch[0], batch[1], batch[2], batch[3])
    # # example t_array
    # t_array = jnp.array([0.1, 0.2, 0.3, 0.4])
    
    
    # ### test out the eval function on fake data
    # from utils.training_testing_fns import train_fn, eval_fn
    # jitted_fn = jax.jit(eval_fn, static_argnames=['loss_type','DEBUG_FLAG'])
    # out = jitted_fn(all_counts = allCounts, 
    #                t_arr = t_array, 
    #                pairHMM = pairHMM, 
    #                params_dict = params, 
    #                hparams_dict = hparams,
    #                eval_rngkey = jax.random.key(0),
    #                loss_type = args.loss_type,
    #                DEBUG_FLAG = False)
    # aux_dict, _ = out
    # del out
    
    
    # ### post-training action: gather everything into a pandas dataframe
    # batch_out_df = training_dset.retrieve_sample_names(batch[-1])
    # batch_out_df['logP_perSamp'] = np.array(aux_dict['logP_perSamp'])
    # batch_out_df['logP_perSamp_length_normed'] = (batch_out_df['logP_perSamp'] / 
    #                                               batch_out_df['desc_seq_len'])
    
    
    # ### save this to an object
    # out_dict = {'args': args,
    #             'pairHMM': pairHMM,
    #             'params': params,
    #             'hparams': hparams}
    
    # with open(f'fake_objects.pkl', 'wb') as g:
    #     pickle.dump(out_dict, g)
    