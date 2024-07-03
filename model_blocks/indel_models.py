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

3. TKF91_single

4. TKF92_single

5. otherIndel_single (this is for LG05, RS07, and KM03)

6. no_indel
   > placeholder so that training script will run, but P(transitions)=0 for 
     all transitions that are not M->M


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
       

universal order of dimensions:
==============================
0. logprob matrix/vectors
2. k_subst
3. k_equl
4. k_indel


TODO:
=====
  - at the very least, could TKF models be combined now?

  - is there a smarter way to combine all indel models into one flexible (but
    still jit-compatible) class? Things to contend with include:
      > TKF models are fitting an offset, but all others are fitting mu 
        directly; TKF also has alpha_beta function
      > GGI function technically has a different signature, but could be 
        wrapped up?
      > TKF91 doesn't have extension probabilities, but I could just provide 
        placeholder values
"""
import numpy as np
import copy
import jax
from jax import numpy as jnp
from jax.nn import softmax

### import transition functions from Ian
from model_blocks.GGI_funcs import transitionMatrix as GGI_Ftransitions
from model_blocks.other_transition_funcs import LG05_Ftransitions
from model_blocks.other_transition_funcs import RS07_Ftransitions
from model_blocks.other_transition_funcs import KM03_Ftransitions

# We replace zeroes and infinities with small numbers sometimes
# It's sinful but BOY DO I LOVE SIN
smallest_float32 = jnp.finfo('float32').smallest_normal

# used to ensure numerical stability of tkf methods
TKF_STABILITY_ADDITION = 1e-3


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
        OUTPUTS: empty transition matrix (3,3,1); logP(transitions)=0
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
    def __init__(self, tie_params):
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
            
        hparams to pass on (or infer):
            - diffrax params (from argparse)
        """
        provided_args = dir(argparse_obj)
        
        ### PARAMETERS: indel rate
        # first, check if "indel_rate" is provided, meaning you're 
        #   tying weights
        if 'indel_rate' in provided_args:
            lam = argparse_obj.indel_rate
            mu = argparse_obj.indel_rate
        
        # if this isn't provided, see if lam or mu are provided; check these
        # seperately, in the weird case you may provide one but not the other
        else:
            # lam; DEFAULT=0.5
            if 'lam' in provided_args:
                lam = argparse_obj.lam
            else:
                lam = 0.5
            
            # mu; DEFAULT=0.5
            if 'mu' in provided_args:
                mu = argparse_obj.mu
            else:
                mu = 0.5
        
        
        ### PARAMETERS: extension probability
        # first, check if "extension_prob" is provided, meaning you're 
        #   tying weights
        if 'extension_prob' in provided_args:
            x = argparse_obj.extension_prob
            y = argparse_obj.extension_prob
        
        # if this isn't provided, see if x or y are provided; check these
        # seperately, in the weird case you may provide one but not the other
        else:
            # x; DEFAULT=0.5
            if 'x' in provided_args:
                x = argparse_obj.x
            else:
                x = 0.5
            
            # y; DEFAULT=0.5
            if 'y' in provided_args:
                y = argparse_obj.y
            else:
                y = 0.5
                
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
        transmat = GGI_Ftransitions (t, 
                                     indel_params, 
                                     alphabet_size,
                                     **diffrax_params)
        
        # if any position in transmat is zero, replace with smallest float
        transmat = jnp.where(transmat != 0, transmat, smallest_float32)
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
            
            # add to output dictionary
            out_dict['lam'] = lam
            out_dict['mu'] = mu
            out_dict['x'] = x
            out_dict['y'] = y
        
        else:
            ### mu takes on value of lambda, y takes on value of x
            # combine lambda and mu under label "indel rate"
            # combine x and y under label "extension prob"
            indel_rate = lam
            extension_prob = x
            
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
        provided_args = dir(argparse_obj)
        
        ### PARAMETERS: indel rate
        # first, check if "indel_rate" is provided, meaning you're 
        #   tying weights
        if 'indel_rate' in provided_args:
            lam = jnp.array(argparse_obj.indel_rate)
            mu = jnp.array(argparse_obj.indel_rate)
        
        # if this isn't provided, see if lam or mu are provided; check these
        # seperately, in the weird case you may provide one but not the other
        else:
            # lam; DEFAULT=evenly spaced values
            if 'lam' not in provided_args:
                lam = jnp.linspace(0.1, 0.9, argparse_obj.k_indel)
            else:
                lam = jnp.array(argparse_obj.lam)
            
            # mu; DEFAULT=evenly spaced values
            if 'mu' not in provided_args:
                mu = jnp.linspace(0.1, 0.9, argparse_obj.k_indel)
            else:
                mu = jnp.array(argparse_obj.mu)
        
        
        ### PARAMETERS: extension probability
        # first, check if "extension_prob" is provided, meaning you're 
        #   tying weights
        if 'extension_prob' in provided_args:
            x = argparse_obj.extension_prob
            y = argparse_obj.extension_prob
        
        # if this isn't provided, see if x or y are provided; check these
        # seperately, in the weird case you may provide one but not the other
        else:
            # x; DEFAULT=evenly spaced values
            if 'x' not in provided_args:
                x = jnp.linspace(0.1, 0.9, argparse_obj.k_indel)
            else:
                x = jnp.array(argparse_obj.x)
            
            # y; DEFAULT=evenly spaced values
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
            
            # add to output dictionary
            out_dict['lam'] = lam
            out_dict['mu'] = mu
            out_dict['x'] = x
            out_dict['y'] = y
        
        else:
            ### mu takes on value of lambda, y takes on value of x
            # combine lambda and mu under label "indel rate"
            # combine x and y under label "extension prob"
            indel_rate = np.array(lam)
            extension_prob = np.array(x)
            
            # add to output dictionary
            out_dict['indel_rate'] = indel_rate
            out_dict['extension_prob'] = extension_prob
        
        
        ### add indel_mix_probs to the output dictionary too
        out_dict['indel_mix_probs'] = np.array(indel_mix_probs)
        
        return out_dict


###############################################################################
### TKF91_single   ############################################################
###############################################################################
# inherits the following from single_GGI: __init__, pytree 
#   transformation functions
class TKF91_single(GGI_single):
    def initialize_params(self, argparse_obj):
        """
        ABOUT: return (possibly transformed) parameters
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit:
            - lambda (birth rate)
              > DEFAULT: 0.5
              > DOMAIN RESTRICTION: greater than 0
            
            - offset (mu = lambda + offest + TKF_STABILITY_ADDITION)
              > TKF_STABILITY_ADDITION is 1e-3, until something fails...
              > DEFAULT: 0.01
        """
        provided_args = dir(argparse_obj)
        
        ### PARAMETERS: indel rate
        # first, check if "indel_rate" or "lam" are provided
        if 'indel_rate' in provided_args:
            lam = argparse_obj.indel_rate
        
        elif 'lam' in provided_args:
            lam = argparse_obj.lam
            
        # otherwise, default to 0.5
        else:
            lam = 0.5
            
            
        if 'offset' not in provided_args:
            offset = 0.0001
        else:
            offset = argparse_obj.offset
        
        
        ### PARAMETER: birth rate lambda
        # transform to (-inf, inf) domain; also add extra k_indel dimension
        lam_transf = jnp.expand_dims(jnp.sqrt(lam), -1)
        
        # create output param dictionary
        initialized_params = {'lam_transf': lam_transf}
        
        
        ### PARAMETER: offset needed to calculate mu
        if not self.tie_params:
            # also transform domain
            offset_transf = jnp.expand_dims(jnp.sqrt(offset), -1)
            
            # add to param dict
            to_add = {'offset_transf': offset_transf}
            initialized_params = {**initialized_params, **to_add}
            del to_add
        
        return initialized_params, dict()
        
        
    def logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: calculate the transition matrix at time t
        JITTED: yes
        WHEN IS THIS CALLED: every training loop iteration, every point t
        OUTPUTS: logP(transitions); the GGI transition matrix
        """
        ### Lambda: unpack params, undo domain transf
        lam_transf = params_dict['lam_transf']
        lam = jnp.square(lam_transf)
        
        
        ### Offset for mu
        if not self.tie_params:
            offset_transf = params_dict['offset_transf']
            offset = jnp.square(offset_transf)
            mu = lam + offset + TKF_STABILITY_ADDITION
        
        else:
            mu = lam + TKF_STABILITY_ADDITION
        
        ### calculate transition probabilities
        alpha, beta, gamma, one_minus_gamma = self.TKF_coeffs(lam, mu, t)
        transmat = jnp.array ([[(1-beta)*alpha, beta, (1-beta)*(1-alpha)],
                               [(1-beta)*alpha, beta, (1-beta)*(1-alpha)],
                               [(one_minus_gamma)*alpha, gamma, (one_minus_gamma)*(1-alpha)]])
        
        # if any position in transmat is zero, replace with smallest float
        transmat = jnp.where(transmat != 0, transmat, smallest_float32)
        logprob_transition_at_t = jnp.log(transmat)
        
        return logprob_transition_at_t
    
    
    def undo_param_transform(self, params_dict):
        """
        ABOUT: if any parameters have domain changes, undo those
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary, with transformed params
        """
        ### lambda
        lam_transf = params_dict['lam_transf']
        lam = jnp.square(lam_transf)
        
    
        ### offset for mu
        if not self.tie_params:
            offset_transf = params_dict['offset_transf']
            offset = jnp.square(offset_transf)
            mu = lam + offset + TKF_STABILITY_ADDITION
            
            out_dict = {'lam': lam,
                        'mu': mu}
        
        else:
            indel_rate = lam
            out_dict = {'indel_rate': lam}
        return out_dict
    
    
    ################   v__(extra functions placed below)__v   ################
    def TKF_coeffs (self, lam, mu, t):
        alpha = jnp.exp(-mu*t)
        beta = (lam*(jnp.exp(-lam*t)-jnp.exp(-mu*t))) / (mu*jnp.exp(-lam*t)-lam*jnp.exp(-mu*t))
        gamma = 1 - ( (mu * beta) / ( lam * (1-alpha) ) )
        
        # can't have gamma < 0, or you'll be in a personal hell
        gamma, one_minus_gamma = jnp.where(gamma > 0,
                                           jnp.concatenate([ gamma, (1-gamma) ]),
                                           jnp.array([ smallest_float32, (1-1e-7) ]))

        # TODO: not numerically instable, but you get weird results if beta=0 (meaning gamma = 1)
        
        gamma = gamma[None]
        one_minus_gamma = one_minus_gamma[None]
        
        return alpha, beta, gamma, one_minus_gamma
    



###############################################################################
### TKF92_single   ############################################################
###############################################################################
# inherits the following from TKF91_single: __init__, TKF_alpha_beta, pytree 
#   transformation functions
class TKF92_single(TKF91_single):
    def initialize_params(self, argparse_obj):
        """
        ABOUT: return (possibly transformed) parameters
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit:
            - lambda (birth rate)
              > DEFAULT: 0.5
              > DOMAIN RESTRICTION: greater than 0
            
            - offset (mu = lambda + offest + TKF_STABILITY_ADDITION)
              > TKF_STABILITY_ADDITION is 1e-3, until something fails...
              > DEFAULT: 0.01
            
            - x (extension probability)
              > DEFAULT:  0.5
              > DOMAIN RESTRICTION: (0, 1)
                  
            - y (retraction probability)
              > DEFAULT:  0.5
              > DOMAIN RESTRICTION: (0, 1)
        """
        provided_args = dir(argparse_obj)
        
        
        ### PARAMETERS: indel rate
        # first, check if "indel_rate" is provided, meaning you're 
        #   tying weights
        if 'indel_rate' in provided_args:
            lam = argparse_obj.indel_rate
            mu = argparse_obj.indel_rate
        
        # if this isn't provided, see if lam or offset are provided; check these
        # seperately, in the weird case you may provide one but not the other
        else:
            # lam; DEFAULT=0.5
            if 'lam' in provided_args:
                lam = argparse_obj.lam
            else:
                lam = 0.5
            
            # offset (for mu); DEFAULT=0.0001
            if 'offset' in provided_args:
                offset = argparse_obj.offset
            else:
                offset = 0.0001
        
        
        ### PARAMETERS: extension probability
        # first, check if "extension_prob" is provided, meaning you're 
        #   tying weights
        if 'extension_prob' in provided_args:
            x = argparse_obj.extension_prob
            y = argparse_obj.extension_prob
        
        # if this isn't provided, see if x or y are provided; check these
        # seperately, in the weird case you may provide one but not the other
        else:
            # x; DEFAULT=0.5
            if 'x' in provided_args:
                x = argparse_obj.x
            else:
                x = 0.5
            
            # y; DEFAULT=0.5
            if 'y' in provided_args:
                y = argparse_obj.y
            else:
                y = 0.5
        
        
        ### keep lambda and x
        # transform to (-inf, inf) domain; also add extra k_indel dimension
        lam_transf = jnp.expand_dims(jnp.sqrt(lam), -1)
        x_transf = jnp.expand_dims(jnp.sqrt(-jnp.log(x)), -1)
        
        # create output param dictionary
        initialized_params = {'lam_transf': lam_transf,
                              'x_transf': x_transf}
        
        
        ### if not tying weights, add offset and y separately
        if not self.tie_params:
            # also transform domain
            offset_transf = jnp.expand_dims(jnp.sqrt(offset), -1)
            y_transf = jnp.expand_dims(jnp.sqrt(-jnp.log(y)), -1)
            
            # add to param dict
            to_add = {'offset_transf': offset_transf,
                      'y_transf': y_transf}
            initialized_params = {**initialized_params, **to_add}
            del to_add
        
        return initialized_params, dict()
        
        
    def logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: calculate the transition matrix at time t
        JITTED: yes
        WHEN IS THIS CALLED: every training loop iteration, every point t
        OUTPUTS: logP(transitions); the GGI transition matrix
        """
        # make sure TKF mu >= lambda
        #   it's kind of a bad hack...not sure what else to try though
        

        ### lambda and x are guaranteed to be there
        # unpack parameters
        lam_transf = params_dict['lam_transf']
        x_transf = params_dict['x_transf']
        
        # undo domain transformation
        lam = jnp.square(lam_transf)
        x = jnp.exp(-jnp.square(x_transf))
        
        
        if not self.tie_params:
            ### mu and y are independent
            # unpack params
            offset_transf = params_dict['offset_transf']
            y_transf = params_dict['y_transf']
            
            # undo domain transformations; calculate mu
            offset = jnp.square(offset_transf)
            mu = lam + offset + TKF_STABILITY_ADDITION
            y = jnp.exp(-jnp.square(y_transf))
            
        
        else:
            ### mu takes on value of lambda, y takes on value of x
            mu = lam + TKF_STABILITY_ADDITION
            y = x
        
        ### calculate transition probabilities
        alpha, beta, gamma, one_minus_gamma = self.TKF_coeffs(lam, mu, t)
        r = (x + y) / 2
        
        transmat = jnp.array ([[r + (1-r)*(1-beta)*alpha, (1-r)*beta, (1-r)*(1-beta)*(1-alpha)],
                               [(1-r)*(1-beta)*alpha, r + (1-r)*beta, (1-r)*(1-beta)*(1-alpha)],
                               [(1-r)*(one_minus_gamma)*alpha, (1-r)*gamma, r + (1-r)*(one_minus_gamma)*(1-alpha)]])
        
        # if any position in transmat is zero, replace with smallest float
        transmat = jnp.where(transmat != 0, transmat, smallest_float32)
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
            offset_transf = params_dict['offset_transf']
            y_transf = params_dict['y_transf']
            
            # undo domain transformation
            offset = jnp.square(offset_transf)
            mu = lam + offset + TKF_STABILITY_ADDITION
            y = jnp.exp(-jnp.square(y_transf))
            
            # add to output dictionary
            out_dict['lam'] = lam
            out_dict['mu'] = mu
            out_dict['x'] = x
            out_dict['y'] = y
        
        else:
            ### mu takes on value of lambda, y takes on value of x
            # combine lambda and mu under label "indel rate"
            # combine x and y under label "extension prob"
            indel_rate = lam
            extension_prob = x
            
            # add to output dictionary
            out_dict['indel_rate'] = indel_rate
            out_dict['extension_prob'] = extension_prob
        
        return out_dict



###############################################################################
### LG05, RS07, KM03   ########################################################
###############################################################################
# this could be the template to combining all indel models into one class
class otherIndel_single:
    def __init__(self, tie_params, model_name):
        self.tie_params = tie_params
        self.model_name = model_name
        
        if model_name == 'LG05':
            self.transition_function = LG05_Ftransitions
            
        elif model_name == 'RS07':
            self.transition_function = RS07_Ftransitions
        
        elif model_name == 'KM03':
            self.transition_function = KM03_Ftransitions
    
    
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
            
        hparams to pass on (or infer):
            - None
        """
        provided_args = dir(argparse_obj)
        
        ### PARAMETERS: indel rate
        # first, check if "indel_rate" is provided, meaning you're 
        #   tying weights
        if 'indel_rate' in provided_args:
            lam = argparse_obj.indel_rate
            mu = argparse_obj.indel_rate
        
        # if this isn't provided, see if lam or mu are provided; check these
        # seperately, in the weird case you may provide one but not the other
        else:
            # lam; DEFAULT=0.5
            if 'lam' in provided_args:
                lam = argparse_obj.lam
            else:
                lam = 0.5
            
            # mu; DEFAULT=0.5
            if 'mu' in provided_args:
                mu = argparse_obj.mu
            else:
                mu = 0.5
        
        
        ### PARAMETERS: extension probability
        # first, check if "extension_prob" is provided, meaning you're 
        #   tying weights
        if 'extension_prob' in provided_args:
            x = argparse_obj.extension_prob
            y = argparse_obj.extension_prob
        
        # if this isn't provided, see if x or y are provided; check these
        # seperately, in the weird case you may provide one but not the other
        else:
            # x; DEFAULT=0.5
            if 'x' in provided_args:
                x = argparse_obj.x
            else:
                x = 0.5
            
            # y; DEFAULT=0.5
            if 'y' in provided_args:
                y = argparse_obj.y
            else:
                y = 0.5
        
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
        
        
        return initialized_params, dict()
    
    
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
        
        
        ### find transition matrix
        transmat = self.transition_function(lam, mu, x, y, t)
        
        # if any position in transmat is zero, replace with smallest float
        transmat = jnp.where(transmat != 0, transmat, smallest_float32)
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
            
            # add to output dictionary
            out_dict['lam'] = lam
            out_dict['mu'] = mu
            out_dict['x'] = x
            out_dict['y'] = y
        
        else:
            ### mu takes on value of lambda, y takes on value of x
            # combine lambda and mu under label "indel rate"
            # combine x and y under label "extension prob"
            indel_rate = lam
            extension_prob = x
            
            # add to output dictionary
            out_dict['indel_rate'] = indel_rate
            out_dict['extension_prob'] = extension_prob
        
        return out_dict
    
    ###  v__(these allow the class to be passed into a jitted function)__v  ###
    def _tree_flatten(self):
        children = ()
        aux_data = {'tie_params': self.tie_params,
                    'model_name': self.model_name} 
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


