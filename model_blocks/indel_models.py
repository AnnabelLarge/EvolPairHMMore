#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 22:04:27 2024

@author: annabel


ABOUT:
======
Models of TRANSITIONS between match, insert, and delete states of a pairHMM:

1. GGI_single

2. GGI_mixture

3. no_indel
   > placeholder so that training script will run, but indels will not 
     be scored



at a minimum, future classes need:
==================================
0. some built-in method to turn the class into a jit-compatible pytree

1. initialize_params(self, argparse_obj): initialize all parameters and 
     hyperparameters; parameters are updated with optax, but hyperparameters
     just help the functions run (i.e. aren't updated)
     
2. logprobs_at_t(self, t, params_dict, hparams_dict): calculate 
     logP(indels) at time t

3. undo_param_transform(self, params_dict): undo any domain transformations
     and output regular list/ints; mainly used for recording results to
     tensorboard, JSON, or anything else that doesn't like jax arrays
     > if NO transforms are needed, this can be inherited as-is from "no_indel"
       

universal order of dimensions:
==============================
0. logprob matrix/vectors
2. k_subst
3. k_equl
4. k_indel
"""
import numpy as np
import copy
import jax
from jax import numpy as jnp
from jax.nn import softmax

### if training/evaluating GGI model, use the functions from Ian
from model_blocks.GGI_funcs import transitionMatrix

# We replace zeroes and infinities with small numbers sometimes
# It's sinful but BOY DO I LOVE SIN
smallest_float32 = jnp.finfo('float32').smallest_normal


###############################################################################
### no indel model (placeholder class)   ######################################
###############################################################################
# use this to train without an indel model; indel counts will be multiplied
#   by zero and not contribute to loss/logprob
class no_indel:
    def initialize_params(self, argparse_obj):
        """
        ABOUT: return (possibly transformed) parameters
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit: None
        hparams to pass on (or infer): None
        """
        return dict(), dict()
        
        
    def logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: return a placeholder matrix for every time t
        JITTED: yes
        WHEN IS THIS CALLED: every training loop iteration, every point t
        OUTPUTS: empty transition matrix (3,3,1); P(transitions)=0
        """
        return jnp.zeros((3,3,1))
    
    def undo_param_transform(self, params_dict):
        """
        ABOUT: placeholder function; no parameters in params_dict
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary as-is (empty)
        """
        return  dict()
    
    ###  v__(these allow the class to be passed into a jitted function)__v  ###
    def _tree_flatten(self):
        children = ()
        aux_data = {} 
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls()


###############################################################################
### single GGI indel model   ##################################################
###############################################################################
class GGI_single:
    def __init__(self, tie_params=False):
        self.tie_params = tie_params
        
    def initialize_params(self, argparse_obj):
        """
        ABOUT: return (possibly transformed) parameters
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit:
            - lambda (insertion rates)
              > DEFAULT: 0.5
              > DOMAIN RESTRICTION: greater than 0
                  
            - mu (deletion rates)
              > DEFAULT:  0.5
              > DOMAIN RESTRICTION: greater than 0
                  
            - x (extension probability)
              > DEFAULT:  0.5
              > DOMAIN RESTRICTION: (0, 1)
                  
            - y (retraction probability)
              > DEFAULT:  0.5
              > DOMAIN RESTRICTION: (0, 1)
            
        hparams to pass on (or infer): None
        """
        provided_args = dir(argparse_obj)
        
        ### PARAMETERS: ggi params
        # lambda
        if 'lam' not in provided_args:
            lam = 0.5
        else:
            lam = argparse_obj.lam
        
        # mu
        if 'mu' not in provided_args:
            mu = 0.5
        else:
            mu = argparse_obj.mu
            
        # x
        if 'x' not in provided_args:
            x = 0.5
        else:
            x = argparse_obj.x
        
        # y
        if 'y' not in provided_args:
            y = 0.5
        else:
            y = argparse_obj.y
        
        ### keep lambda and x
        # transform to (-inf, inf) domain; also add extra k_indel dimension
        lam_transf = jnp.expand_dims(jnp.sqrt(lam), -1)
        x_transf = jnp.expand_dims(jnp.sqrt(-jnp.log(x)), -1)
        
        # create output param dictionary
        initialized_params = {'lam_transf': lam_transf,
                              'x_transf': x_transf}
        
        
        ### if not tying weights, add mu and y separately
        if not self.tie_params:
            # also transform domain
            mu_transf = jnp.expand_dims(jnp.sqrt(mu), -1)
            y_transf = jnp.expand_dims(jnp.sqrt(-jnp.log(y)), -1)
            
            # add to param dict
            to_add = {'mu_transf': mu_transf,
                      'y_transf': y_transf}
            initialized_params = {**initialized_params, **to_add}
            del to_add
        
        
        ### create hyperparams dictionary
        hparams = {'diffrax_params': argparse_obj.diffrax_params}
        
        return initialized_params, hparams
        
        
    def logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: calculate the transition matrix at time t
        JITTED: yes
        WHEN IS THIS CALLED: every training loop iteration, every point t
        OUTPUTS: logP(transitions); the GGI transition matrix
        """
        ### lambda and x are guaranteed to be there
        # unpack parameters
        lam_transf = params_dict['lam_transf']
        x_transf = params_dict['x_transf']
        
        # undo domain transformation
        lam = jnp.square(lam_transf)
        x = jnp.exp(-jnp.square(x_transf))
        

        if not self.tie_params:
            ### mu and y are independent params
            # unpack params
            mu_transf = params_dict['mu_transf']
            y_transf = params_dict['y_transf']
            
            # undo domain transformation
            mu = jnp.square(mu_transf)
            y = jnp.exp(-jnp.square(y_transf))
        
        else:
            ### mu takes on value of lambda, y takes on value of x
            mu = lam
            y = x
        
        
        ### unpack the hyparpameters
        diffrax_params = hparams_dict['diffrax_params']
        alphabet_size = hparams_dict['alphabet_size']
        
        # indel params is a tuple of four elements; each elem is (k_indel,)
        # in this case, each elem is of size (1,)
        indel_params = (lam, mu, x, y)
        
        # transition matrix ((a,b,c),(f,g,h),(p,q,r)); rows sum to 1
        # (3, 3, k_indel); in this case, k_indel=1
        transmat = transitionMatrix (t, 
                                     indel_params, 
                                     alphabet_size,
                                     **diffrax_params)
        
        # if any position in transmat is zero, replace with 1 such that log(1)=0
        transmat = jnp.where(transmat != 0, transmat, 1)
        logprob_transition_at_t = jnp.log(transmat)
        
        return logprob_transition_at_t
    
    
    def undo_param_transform(self, params_dict):
        """
        ABOUT: if any parameters have domain changes, undo those
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary, with transformed params
        """
        ### lambda and x are guaranteed to be there
        # unpack parameters
        lam_transf = params_dict['lam_transf']
        x_transf = params_dict['x_transf']
        
        # undo domain transformation
        lam = jnp.square(lam_transf)
        x = jnp.exp(-jnp.square(x_transf))
        
        out_dict = {}
        if not self.tie_params:
            ### mu and y are independent params
            # unpack params
            mu_transf = params_dict['mu_transf']
            y_transf = params_dict['y_transf']
            
            # undo domain transformation
            mu = jnp.square(mu_transf)
            y = jnp.exp(-jnp.square(y_transf))
            
            # also turn them into regular integers, for writing JSON
            lam = lam.item()
            mu = mu.item()
            x = x.item()
            y = y.item()
            
            # add to output dictionary
            out_dict['lam'] = lam
            out_dict['mu'] = mu
            out_dict['x'] = x
            out_dict['y'] = y
        
        else:
            ### mu takes on value of lambda, y takes on value of x
            # combine lambda and mu under label "indel rate"
            # combine x and y under label "extension prob"
            indel_rate = lam.item()
            extension_prob = x.item()
            
            # add to output dictionary
            out_dict['indel_rate'] = indel_rate
            out_dict['extension_prob'] = extension_prob
        
        return out_dict
    
    ###  v__(these allow the class to be passed into a jitted function)__v  ###
    def _tree_flatten(self):
        children = ()
        aux_data = {'tie_params': self.tie_params} 
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
    

###############################################################################
### mixture GGI indel model   #################################################
###############################################################################
# inherit __init__, logprobs_at_t, and methods for 
#   tree flattening/unflattening from GGI_single
class GGI_mixture(GGI_single):
    def initialize_params(self, argparse_obj):
        """
        ABOUT: return (possibly transformed) parameters
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit:
            - lambda (insertion rates)
              > DEFAULT: np.linspace(0.1, 0.9, k_indel)
              > DOMAIN RESTRICTION: greater than 0
                  
            - mu (deletion rates)
              > DEFAULT: np.linspace(0.1, 0.9, k_indel)
              > DOMAIN RESTRICTION: greater than 0
                  
            - x (extension probability)
              > DEFAULT: np.linspace(0.1, 0.9, k_indel)
              > DOMAIN RESTRICTION: (0, 1)
                  
            - y (retraction probability)
              > DEFAULT: np.linspace(0.1, 0.9, k_indel)
              > DOMAIN RESTRICTION: (0, 1)
            
            - mixture logits
              > DEFAULT: vector of 1s, length of k_indel 
              
        hparams to pass on (or infer):
            - diffrax params (from argparse)
        """
        ### will use the transitionMatrix function from Ian
        provided_args = dir(argparse_obj)
        
        ### PARAMETERS: ggi params
        # lambda
        if 'lam' not in provided_args:
            lam = jnp.linspace(0.1, 0.9, argparse_obj.k_indel)
        else:
            lam = jnp.array(argparse_obj.lam)
        
        # mu
        if 'mu' not in provided_args:
            mu = jnp.linspace(0.1, 0.9, argparse_obj.k_indel)
        else:
            mu = jnp.array(argparse_obj.mu)
            
        # x
        if 'x' not in provided_args:
            x = jnp.linspace(0.1, 0.9, argparse_obj.k_indel)
        else:
            x = jnp.array(argparse_obj.x)
        
        # y
        if 'y' not in provided_args:
            y = jnp.linspace(0.1, 0.9, argparse_obj.k_indel)
        else:
            y = jnp.array(argparse_obj.y)
        
        ### keep lambda and x
        # transform to (-inf, inf) domain; also add extra k_indel dimension
        lam_transf = jnp.sqrt(lam)
        x_transf = jnp.sqrt(-jnp.log(x))
        
        # create output param dictionary
        initialized_params = {'lam_transf': lam_transf,
                              'x_transf': x_transf}
        
        
        ### if not tying weights, add mu and y separately
        if not self.tie_params:
            # also transform domain
            mu_transf = jnp.sqrt(mu)
            y_transf = jnp.sqrt(-jnp.log(y))
            
            # add to param dict
            to_add = {'mu_transf': mu_transf,
                      'y_transf': y_transf}
            initialized_params = {**initialized_params, **to_add}
            del to_add
        
        
        ### PARAMETER: mixture logits
        if 'indel_mix_logits' not in provided_args:
            indel_mix_logits = jnp.ones(argparse_obj.k_indel)
        else:
            indel_mix_logits = jnp.array(argparse_obj.indel_mix_logits, 
                                         dtype=float)
        to_add = {'indel_mix_logits': indel_mix_logits}
        initialized_params = {**initialized_params, **to_add}
        del to_add
        
        
        ### OUTPUT DICTIONARIES
        # dictionary of hyperparameters
        hparams = {'diffrax_params': argparse_obj.diffrax_params}
        
        return initialized_params, hparams
    
    
    def undo_param_transform(self, params_dict):
        """
        ABOUT: if any parameters have domain changes, undo those
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary, with transformed params
        """
        ### lambda and x are guaranteed to be there
        # unpack parameters
        lam_transf = params_dict['lam_transf']
        x_transf = params_dict['x_transf']
        indel_mix_logits= params_dict['indel_mix_logits']
        
        # undo domain transformation
        lam = jnp.square(lam_transf)
        x = jnp.exp(-jnp.square(x_transf))
        indel_mix_probs = softmax(indel_mix_logits)
        
        out_dict = {}
        if not self.tie_params:
            ### mu and y are independent params
            # unpack params
            mu_transf = params_dict['mu_transf']
            y_transf = params_dict['y_transf']
            
            # undo domain transformation
            mu = jnp.square(mu_transf)
            y = jnp.exp(-jnp.square(y_transf))
            
            # also turn them into regular integers, for writing JSON
            lam = np.array(lam).tolist()
            mu = np.array(mu).tolist()
            x = np.array(x).tolist()
            y = np.array(y).tolist()
            indel_mix_probs = np.array(indel_mix_probs).tolist()
            
            # add to output dictionary
            out_dict['lam'] = lam
            out_dict['mu'] = mu
            out_dict['x'] = x
            out_dict['y'] = y
        
        else:
            ### mu takes on value of lambda, y takes on value of x
            # combine lambda and mu under label "indel rate"
            # combine x and y under label "extension prob"
            indel_rate = np.array(lam).to_list()
            extension_prob = np.array(x).to_list()
            
            # add to output dictionary
            out_dict['indel_rate'] = indel_rate
            out_dict['extension_prob'] = extension_prob
        
        
        ### add indel_mix_probs to the output dictionary too
        out_dict['indel_mix_probs'] = indel_mix_probs
        
        return out_dict
