#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 22:04:27 2024

@author: annabel


ABOUT:
======
Methods to generate equilibrium distributions:

1. equl_base
   > single equlibrium vector, infer from given data

2. equl_deltaMixture
   > mixture model, assume delta priors over the different equilibrium vectors

3. equl_dirichletMixture
   > mixture model, sample equilibrium vectors from a dirichlet prior


at a minimum, future classes need:
==================================
1. initialize_params(self, argparse_obj): initialize all parameters and 
     hyperparameters; parameters are updated with optax, but hyperparameters
     just help the functions run (i.e. aren't updated)

2. equlVec_logprobs(self, params_dict, hparams_dict): calculate equilibrium 
     vector and logP(emissions/omissions) i.e. log(equlibrium vector)
     NOTE THAT THIS DOES NOT DEPEND ON TIME!!!

3. undo_param_transform(self, params_dict): undo any domain transformations
     and output regular list/ints; mainly used for recording results to
     tensorboard, JSON, or anything else that doesn't like jax arrays
       

universal order of dimensions:
==============================
0. logprob matrix/vectors
2. k_subst
3. k_equl
4. k_indel

"""
import jax
from jax import numpy as jnp
from jax.nn import softmax, log_softmax
import numpy as np

# We replace zeroes and infinities with small numbers sometimes
# It's sinful but BOY DO I LOVE SIN
smallest_float32 = jnp.finfo('float32').smallest_normal


###############################################################################
### single equlibrium vector: infer from given data   #########################
###############################################################################
class equl_base:
    def initialize_params(self, argparse_obj):
        """
        ABOUT: return (possibly transformed) parameters
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit: (none, return empty dictionary)
        hparams to pass on (or infer): (none,return empty dictionary)
        """
        return dict(), dict()
    
    
    def equlVec_logprobs(self, params_dict, hparams_dict):
        """
        ABOUT: calculate the equilibrium distribution; return this and 
               logP(emission/omission) at indel sites (to multiply by 
               indel counts during training)
        JITTED: yes
        WHEN IS THIS CALLED: every training loop iteration
        OUTPUTS: 
            - equl_vec: equilibrium distribution seen in real data
            - logprob_equl: logP(emission/omission)
        """
        # equilibrium distribution of amino acids, probably provided by 
        # the dataloader? make sure to give it an extra k_equl dimension
        equl_vec = jnp.expand_dims(hparams_dict['equl_vecs_from_train_data'], -1)
        equl_vec_noZeros = jnp.where(equl_vec!=0, equl_vec, smallest_float32)
        logprob_equl = jnp.log(equl_vec_noZeros)
        
        return (equl_vec, logprob_equl)


    def undo_param_transform(self, params_dict):
        """
        ABOUT: placeholder function; no parameters in params_dict
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary as-is (empty)
        """
        return dict()

    ###  v__(these allow the class to be passed into a jitted function)__v  ###
    def _tree_flatten(self):
        children = ()
        aux_data = {} 
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls()


###############################################################################
### mixture model: sample from a delta distribution   #########################
###############################################################################
class equl_deltaMixture(equl_base):
    def initialize_params(self, argparse_obj):
        """
        ABOUT: return (possibly transformed) parameters
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit: 
            - equl_vecs_transf (DEFAULT: init from jax.random.normal)
            - equl_mix_logits (DEFAULT: vector of ones)
        """
        provided_args = dir(argparse_obj)
        
        ### PARAMETER: equl_vecs
        # if not provided, generate from jax.random.normal with the same rng key
        if 'equl_vecs_transf' not in provided_args:
            genkey, _ = jax.random.split(jax.random.key(argparse_obj.rng_seednum))
            out_shape = (argparse_obj.alphabet_size, argparse_obj.k_equl)
            equl_vecs_transf = jax.random.normal(genkey, out_shape)
        
        # if provided, just use what's provided
        else:
            equl_vecs_transf = jax.array(argparse_obj.equl_vecs_transf)
        
        
        ### PARAMETER: mixture logits
        # if not provided, generate a logits vector of ones
        if 'equl_mix_logits' not in provided_args:
            equl_mix_logits = jnp.ones(argparse_obj.k_equl)
        
        # if provided, just use what's provided
        else:
            equl_mix_logits = jnp.array(argparse_obj.equl_mix_logits)
        
        
        ### OUTPUT DICTIONARIES
        # dictionary of parameters
        initialized_params = {'equl_vecs_transf': equl_vecs_transf,
                              'equl_mix_logits': equl_mix_logits}
        
        # dictionary of hyperparameters
        hparams = dict()
        
        return initialized_params, hparams
    
    
    def equlVec_logprobs(self, params_dict, hparams_dict):
        """
        ABOUT: calculate the equilibrium distribution; return this and 
               logP(emission/omission) at indel sites (to multiply by 
               indel counts during training)
        JITTED: yes
        WHEN IS THIS CALLED: every training loop iteration
        OUTPUTS: 
            - equl_vec: softmax(equl_vecs_transf)
            - logprob_equl: log_softmax(equl_vecs_transf)
        """
        # equilibrium distribution of amino acids
        equl_vecs_transf = params_dict['equl_vecs_transf']
        equl_vec = softmax(equl_vecs_transf, axis=0)
        logprob_equl = log_softmax(equl_vecs_transf, axis=0)
        return (equl_vec, logprob_equl) 


    def undo_param_transform(self, params_dict):
        """
        ABOUT: if any parameters have domain changes, undo those
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary, with transformed params
        """
        ### unpack parameters
        equl_vecs_transf = params_dict['equl_vecs_transf']
        equl_mix_logits = params_dict['equl_mix_logits']

        
        ### undo the domain transformation
        equl_vecs = softmax(equl_vecs_transf, axis=0)
        equl_mix_params = softmax(equl_mix_logits)
        
        # also turn them into regular lists, for writing JSON
        equl_vecs = np.array(equl_vecs).tolist()
        equl_mix_params = np.array(equl_mix_params).tolist()
        
        
        ### add to parameter dictionary
        out_dict = {}
        out_dict['equl_vecs'] = equl_vecs
        out_dict['equl_mix_params'] = equl_mix_params
        
        return out_dict


###############################################################################
### mixture model: sample from a dirichlet distribution   #####################
###############################################################################
class equl_dirichletMixture(equl_base):
    def initialize_params(self, argparse_obj):
        """
        ABOUT: return (possibly transformed) parameters
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit: 
            - dirichlet_shape_logits:
              > DEFAULT: vector of ones, twos, etc. up to number of k_equl
              > DOMAIN RESTRICTION: all values must be greater than zero
              
            - equl_mix_logits
              > DEFAULT: vector of ones
              
        """
        provided_args = dir(argparse_obj)
        
        ### PARAMETER: dirichlet_shape
        # if not provided, use a vector of equal probabilities
        if 'dirichlet_shape' not in provided_args:
            # generate dirichlet_shape
            shapes = jnp.array(range(1, argparse_obj.k_equl+1))
            to_mult = jnp.ones((argparse_obj.alphabet_size, argparse_obj.k_equl))
            dirichlet_shape = jnp.einsum('ij,j->ij', to_mult, shapes)
            del shapes, to_mult
        
        # if provided, just use what's provided 
        else:
            dirichlet_shape = jnp.array(argparse_obj.dirichlet_shape)
            
            # make sure domain restrictions are satisfied
            err_msg = ('Initial guesses for DIRICHLET SHAPE must be '+
                       'greater than zero; received dirichlet_shape='+
                       f'{dirichlet_shape}')
            assert (dirichlet_shape > 0).all(), err_msg
            del err_msg
        
        # for stochastic gradient descent, transform to (-inf, inf) domain
        dirichlet_shape_transf = jnp.sqrt(dirichlet_shape)
            
        
        ### PARAMETER: mixture logits
        # if not provided, generate a logits vector of ones
        if 'equl_mix_logits' not in provided_args:
            equl_mix_logits = jnp.ones(argparse_obj.k_equl)
        
        # if provided, just use what's provided
        else:
            equl_mix_logits = jnp.array(argparse_obj.equl_mix_logits, dtype=float)

        
        ### OUTPUT DICTIONARIES
        # dictionary of parameters
        initialized_params = {'equl_mix_logits': equl_mix_logits,
                              'dirichlet_shape_transf':dirichlet_shape_transf}
        
        # dictionary of hyperparameters
        hparams = dict()
        
        return initialized_params, hparams
    
    
    def equlVec_logprobs(self, params_dict, hparams_dict):
        """
        ABOUT: calculate the equilibrium distribution; return this and 
               logP(emission/omission) at indel sites (to multiply by 
               indel counts during training)
        JITTED: yes
        WHEN IS THIS CALLED: every training loop iteration
        OUTPUTS: 
            - equl_vec: equilibrium distributions sampled from 
                        dirichlet distribution
            - logprob_equl: logP(emission/omission)
            
        note: make sure rngkey you provide has already been folded_in with
              the epoch idx outside of calling this function
        """
        ### unpack parameters
        dirichlet_shape_transf = params_dict['dirichlet_shape_transf']
        
        ### unpack hyperparameters
        k_equl = params_dict['equl_mix_logits'].shape[0]
        alphabet_size = hparams_dict['alphabet_size']
        dirichlet_samp_key = hparams_dict['dirichlet_samp_key']
        
        ### sample equl_vecs from dirichlet distribution
        # undo domain transformation with softmax across dim0
        dirichlet_shape = softmax(dirichlet_shape_transf, axis=0)
        
        # replace zeros with small numbers
        dirichlet_shape = jnp.where(dirichlet_shape !=0, 
                                    dirichlet_shape, smallest_float32)
        
        # sample; dirichlet_shape is (alphabet_size, k_equl)
        equl_vec = self.sample_dirichlet(dirichlet_samp_key, dirichlet_shape, k_equl)
        
        # replace zeros with small numbers
        equl_vec_noZeros = jnp.where(equl_vec != 0, 
                                     equl_vec, 
                                     smallest_float32)
        
        ### log transform for logprob_equl
        logprob_equl = jnp.log(equl_vec_noZeros)
        
        return (equl_vec, logprob_equl) 
    
    
    def undo_param_transform(self, params_dict):
        """
        ABOUT: if any parameters have domain changes, undo those
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary, with transformed params
        """
        ### unpack parameters
        equl_mix_logits = params_dict['equl_mix_logits']
        dirichlet_shape_transf = params_dict['dirichlet_shape_transf']
        
        
        ### undo the domain transformation
        equl_mix_params = softmax(equl_mix_logits)
        dirichlet_shape = softmax(dirichlet_shape_transf, axis=0)
        
        # also turn them into regular numpy arrays, for writing JSON
        equl_mix_params = np.array(equl_mix_params).tolist()
        dirichlet_shape = np.array(dirichlet_shape).tolist()
        
        
        ### make output dictionary
        out_dict = {}
        out_dict['equl_mix_params'] = equl_mix_params
        out_dict['dirichlet_shape'] = dirichlet_shape
        
        return out_dict
    
    
    ###############   v__(extra functions placed below)__v   ###############  
    def sample_dirichlet (self, key, alpha_vecs, k_equl):
        """
        implemented Ian's version of the dirichlet reparameterization trick 
          such that each mixture model will have its own random key, for 
          independently generated gaussian models
        
        if this slows down code too much, just revert to what Ian had; this
          is probably overly-paranoid
        """
        # assign each mixture a numerical ID
        mixture_idx_vec = jnp.array(range(1, k_equl+1))
        
        def indp_samp(mixture_idx, alpha):
            # fold in the mixture_idx
            one_mixtures_key = jax.random.fold_in(key, mixture_idx)
            
            # sample gaussian with one key
            n = jax.random.normal (one_mixtures_key, shape=alpha.shape)
            
            # use this to sample one dirichlet
            one_samp = self.transform_multivariate_normal_to_dirichlet (n, alpha)
            
            return one_samp
        
        vmapped_sampler = jax.vmap(indp_samp, in_axes=(0, 1))
        all_samples = vmapped_sampler(mixture_idx_vec, alpha_vecs).T
        return all_samples
    
    
    def transform_multivariate_normal_to_dirichlet (self, n, alpha):
        """
        used above in sample_dirichlet
        """
        K = alpha.shape[0]
        mu = jax.lax.log(alpha) - jnp.sum (jax.lax.log(alpha)) / K
        Sigma = (1/alpha) * (1 - 2/K) + (1/K**2) * jnp.sum(1/alpha)
        a = mu + jnp.sqrt(Sigma) * n
        return softmax(a)
