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



shared class methods:
=====================
1. initialize_params(self, inputs_dict): initialize all parameters and 
     hyperparameters; parameters are updated with optax, but hyperparameters
     just help the functions run (i.e. aren't updated)
     
2. logprobs_at_t(self, t, params_dict, hparams_dict): calculate 
     logP(indels) at time t
       

universal order of dimensions:
==============================
0. logprob matrix/vectors
2. k_subst
3. k_equl
4. k_indel
"""
import jax
from jax import numpy as jnp

### will use the transitionMatrix function from Ian
from model_blocks.GGI_funcs import transitionMatrix

# We replace zeroes and infinities with small numbers sometimes
# It's sinful but BOY DO I LOVE SIN
smallest_float32 = jnp.finfo('float32').smallest_normal


###############################################################################
### single GGI indel model   ##################################################
###############################################################################
class GGI_single:
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
        ### will use the transitionMatrix function from Ian
        from model_blocks.GGI_funcs import transitionMatrix
        
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
        
        # for stochastic gradient descent, transform to (-inf, inf) domain
        # also add extra k_indel dimension
        lam_transf = jnp.expand_dims(jnp.sqrt(lam), -1)
        mu_transf = jnp.expand_dims(jnp.sqrt(mu), -1)
        x_transf = jnp.expand_dims(jnp.sqrt(-jnp.log(x)), -1)
        y_transf = jnp.expand_dims(jnp.sqrt(-jnp.log(y)), -1)
        
        
        ### OUTPUT DICTIONARIES
        initialized_params = {'lam_transf': lam_transf,
                              'mu_transf': mu_transf,
                              'x_transf': x_transf,
                              'y_transf': y_transf}
        
        hparams = {'diffrax_params': argparse_obj.diffrax_params}
        
        return initialized_params, hparams
        
        
    def logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: calculate the transition matrix at time t
        JITTED: yes
        WHEN IS THIS CALLED: every training loop iteration, every point t
        OUTPUTS: logP(transitions); the GGI transition matrix
        """
        ### unpack parameters
        lam_transf = params_dict['lam_transf']
        mu_transf = params_dict['mu_transf']
        x_transf = params_dict['x_transf']
        y_transf = params_dict['y_transf']
        
        ### unpack the hyparpameters
        diffrax_params = hparams_dict['diffrax_params']
        alphabet_size = hparams_dict['alphabet_size']
        
        ### undo the domain transformation
        lam = jnp.square(lam_transf)
        mu = jnp.square(mu_transf)
        x = jnp.exp(-jnp.square(x_transf))
        y = jnp.exp(-jnp.square(y_transf))
        
        # indel params is a tuple of four elements; each elem is (k_indel,)
        # in this case, each elem is of size (1,)
        indel_params = (lam, mu, x, y)
        
        # transition matrix ((a,b,c),(f,g,h),(p,q,r)); rows sum to 1
        # (3, 3, k_indel); in this case, k_indel=1
        transmat = transitionMatrix (t, 
                                     indel_params, 
                                     alphabet_size,
                                     **diffrax_params)
        
        # if any position in transmat is zero, replace with smallest_float32
        transmat = jnp.where(transmat == 0, smallest_float32, transmat)
        logprob_transition_at_t = jnp.log(transmat)
        
        return logprob_transition_at_t
        


###############################################################################
### mixture GGI indel model   #################################################
###############################################################################
class GGI_mixture:
    def initialize_params(self, inputs_dict):
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
            - k_indel
              > DEFAULT: length of mixture logits vector
            
        """
        provided_args = dir(argparse_obj)
        
        ### PARAMETERS: ggi params
        # lambda
        if 'lam' not in provided_args:
            lam = jnp.linspace(0.1, 0.9, k_indel)
        else:
            lam = argparse_obj.lam
        
        # mu
        if 'mu' not in provided_args:
            mu = jnp.linspace(0.1, 0.9, k_indel)
        else:
            mu = argparse_obj.mu
            
        # x
        if 'x' not in provided_args:
            x = jnp.linspace(0.1, 0.9, k_indel)
        else:
            x = argparse_obj.x
        
        # y
        if 'y' not in provided_args:
            y = jnp.linspace(0.1, 0.9, k_indel)
        else:
            y = argparse_obj.y
        
        # for stochastic gradient descent, transform to (-inf, inf) domain
        # also add extra k_indel dimension
        lam_transf = jnp.sqrt(lam)
        mu_transf = jnp.sqrt(mu)
        x_transf = jnp.sqrt(-jnp.log(x))
        y_transf = jnp.sqrt(-jnp.log(y))
        
        
        ### PARAMETER: mixture logits
        if 'indel_mix_logits' not in provided_args:
            indel_mix_logits = jnp.ones(k_indel)
        else:
            indel_mix_logits = jnp.array(argparse_obj.indel_mix_logits)
        
        
        ### HYPERPARAMETER: number of mixtures
        if 'k_indel' not in provided_args:
            k_indel = indel_mix_logits.shape[0]
        else:
            k_indel = argparse_obj.k_indel
        
        
        ### OUTPUT DICTIONARIES
        # parameters to fit with optax
        initialized_params = {'lam_transf': lam_transf,
                              'mu_transf': mu_transf,
                              'x_transf': x_transf,
                              'y_transf': y_transf,
                              'indel_mix_logits': indel_mix_logits}
        
        # dictionary of hyperparameters
        hparams = {'k_indel': k_indel,
                   'diffrax_params': argparse_obj.diffrax_params}
        
        return initialized_params, hparams
    
        
    def logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: calculate the transition matrix at time t
        JITTED: yes
        WHEN IS THIS CALLED: every training loop iteration, every point t
        OUTPUTS: logP(transitions); the GGI transition matrix
        """
        ### unpack parameters
        lam_transf = params_dict['lam_transf']
        mu_transf = params_dict['mu_transf']
        x_transf = params_dict['x_transf']
        y_transf = params_dict['y_transf']
        
        ### unpack the hyparpameters
        diffrax_params = hparams_dict['diffrax_params']
        alphabet_size = hparams_dict['alphabet_size']
        
        ### undo the domain transformation
        lam = jnp.square(lam_transf)
        mu = jnp.square(mu_transf)
        x = jnp.exp(-jnp.square(x_transf))
        y = jnp.exp(-jnp.square(y_transf))
        
        # indel params is a tuple of four elements; each elem is (k_indel,)
        indel_params = (lam, mu, x, y)
        
        # transition matrix ((a,b,c),(f,g,h),(p,q,r)); rows sum to 1
        # (3, 3, k_indel)
        transmat = transitionMatrix (t, 
                                     indel_params, 
                                     alphabet_size,
                                     **diffrax_params)
        
        # if any position in transmat is zero, replace with smallest_float32
        transmat = jnp.where(transmat == 0, smallest_float32, transmat)
        logprob_transition_at_t = jnp.log(transmat)
        
        return logprob_transition_at_t



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
    