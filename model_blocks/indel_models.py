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
from jax.scipy.special import logsumexp

### import transition functions from Ian
from model_blocks.GGI_funcs import transitionMatrix as GGI_Ftransitions
from model_blocks.other_transition_funcs import LG05_Ftransitions
from model_blocks.other_transition_funcs import RS07_Ftransitions
from model_blocks.other_transition_funcs import KM03_Ftransitions

# We replace zeroes and infinities with small numbers sometimes
# It's sinful but BOY DO I LOVE SIN
smallest_float32 = jnp.finfo('float32').smallest_normal

# offset for tkf models
TKF_ERR = 1e-4

########################
### Helper functions   #
########################
def safe_log(x):
    return jnp.log( jnp.where( x>0, x, smallest_float32 ) )

def concat_along_new_last_axis(arr_lst):
    return jnp.concatenate( [arr[...,None] for arr in arr_lst], 
                             axis = -1 )

def logsumexp_with_arr_lst(arr_lst, coeffs = None):
    """
    concatenate a list of arrays, then use logsumexp
    """
    a_for_logsumexp = concat_along_new_last_axis(arr_lst)
    
    out = logsumexp(a = a_for_logsumexp,
                    b = coeffs,
                    axis=-1)
    return out

def log_one_minus_x(x):
    """
    log( 1 - x )
    """
    a_for_logsumexp = concat_along_new_last_axis( [jnp.zeros(x.shape), x] )
    b_for_logsumexp = jnp.array([1, -1])
    
    out = logsumexp(a = a_for_logsumexp,
                    b = b_for_logsumexp,
                    axis = -1)
    
    return out



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
            
            - offset (mu = lambda*(1+TKF_ERR) + offset)
              > DEFAULT: 0.0001
        """
        provided_args = dir(argparse_obj)
        
        ### PARAMETERS: indel rate
        # first, check if "indel_rate" or "lam" are provided
        if 'indel_rate' in provided_args:
            lam = argparse_obj.indel_rate
            mu = lam / (1 - TKF_ERR)
        
        # if this isn't provided, see if lam or offset are provided; check these
        # seperately, in the weird case you may provide one but not the other
        else:
            # lam; DEFAULT=0.5
            if 'lam' in provided_args:
                lam = argparse_obj.lam
            else:
                lam = 0.5
                
            # offset (for mu); DEFAULT=0.1
            if 'offset' in provided_args:
                offset = argparse_obj.offset
            else:
                offset = 0.1
        
        
        ### PARAMETER: birth rate lambda
        # transform to (-inf, inf) domain; also add extra k_indel dimension
        lam_transf = jnp.sqrt(lam)[..., None]
        
        # create output param dictionary
        initialized_params = {'lam_transf': lam_transf}
        
        
        ### PARAMETER: offset needed to calculate mu
        if not self.tie_params:
            # also transform offset to (-inf, inf)
            offset_transf = jnp.sqrt(-jnp.log(offset))[..., None]
            initialized_params['offset_transf'] = offset_transf
        
        return initialized_params, dict()
        
        
    def logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: calculate the transition matrix at time t
        JITTED: yes
        WHEN IS THIS CALLED: every training loop iteration, every point t
        OUTPUTS: logP(transitions); the GGI transition matrix
        """
        ### indel params
        lam, mu, use_approx = self.logits_to_indel_rates(params_dict)
        out_dict = self.tkf_params(lam, mu, t, use_approx)
        
        
        ### entries in the matrix
        # a_f = (1-beta)*alpha;     log(a_f) = log(1-beta) + log(alpha)
        # b_g = beta;               log(b_g) = log(beta)
        # c_h = (1-beta)*(1-alpha); log(c_h) = log(1-beta) + log(1-alpha)
        log_a_f = out_dict['log_one_minus_beta'] + out_dict['log_alpha']
        log_b_g = out_dict['log_beta']
        log_c_h = out_dict['log_one_minus_beta'] + out_dict['log_one_minus_alpha']

        # p = (1-gamma)*alpha;     log(p) = log(1-gamma) + log(alpha)
        # q = gamma;               log(q) = log(gamma)
        # r = (1-gamma)*(1-alpha); log(r) = log(1-gamma) + log(1-alpha)
        log_p = out_dict['log_one_minus_gamma'] + out_dict['log_alpha']
        log_q = out_dict['log_gamma']
        log_r = out_dict['log_one_minus_gamma'] + out_dict['log_one_minus_alpha']
        
        # (3, 3, k_indel)
        logprob_transition_at_t = jnp.array([ [log_a_f, log_b_g, log_c_h],
                                              [log_a_f, log_b_g, log_c_h],
                                              [log_p,   log_q,   log_r  ] ])
        
        return logprob_transition_at_t
    
    
    def undo_param_transform(self, params_dict):
        """
        ABOUT: if any parameters have domain changes, undo those
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary, with transformed params
        """
        lam, mu, use_approx = self.logits_to_indel_rates(params_dict)
        
        out_dict = {'lam_transf': params_dict['lam_transf'],
                    'use_approx': use_approx}
        
        if self.tie_params:
            # combine lambda and mu under label "indel rate"
            out_dict['indel_rate'] = lam
        
        elif not self.tie_params:
            out_dict['lam'] = lam
            out_dict['mu'] = mu
            out_dict['offset'] = offset
            out_dict['offset_transf'] = params_dict['offset_transf']
        
        return out_dict
    
    
    ################   v__(extra functions placed below)__v   ################
    def logits_to_indel_rates(self, params_dict):
        ### Lambda: unpack params, undo domain transf
        lam_transf = params_dict['lam_transf']
        lam = jnp.square(lam_transf)
        
        
        ### Offset for mu
        if not self.tie_params:
            offset_transf = params_dict['offset_transf']
            offset = jnp.exp(-jnp.square(r_transf))
            mu = lam/(1-offset)
            use_approx = (offset <= TKF_ERR).any()
        
        else:
            mu = lam/(1-TKF_ERR)
            use_approx = True
        
        return lam, mu, use_approx
            
            
    def tkf_params(self, 
                   lam, 
                   mu, 
                   t, 
                   use_approx):
        #############################################
        ### parameters in common with both formulas #
        #############################################
        ### all dims are also (k_indel, )
        log_lam = safe_log(lam) 
        log_mu = safe_log(mu)
        mu_per_t = mu * t 
        lam_per_t = lam * t
        log_mu_t = safe_log(mu_per_t)
        log_lam_t = safe_log(lam_per_t)
        
        ### alpha and one minus alpha IN LOG SPACE; (k_indel, )
        # alpha = jnp.exp(-mu_per_t); log(alpha) = -mu_per_t
        log_alpha = -mu_per_t
        log_one_minus_alpha = log_one_minus_x(log_alpha)
        
        #############################################
        ### original calculation of tkf beta, gamma #
        #############################################
        def orig_tkf_params(lam, mu, t_array):
            ### beta
            # log( exp(-lambda * t) - exp(-mu * t) )
            term2_logsumexp = logsumexp_with_arr_lst( [-lam_per_t, -mu_per_t],
                                                      coeffs = jnp.array([1, -1]) )
            
            # log( mu*exp(-lambda * t) - lambda*exp(-mu * t) )
            mixed_coeffs = concat_along_new_last_axis([mu, -lam])
            term3_logsumexp = logsumexp_with_arr_lst([-lam_per_t, -mu_per_t],
                                                     coeffs = mixed_coeffs)
            del mixed_coeffs
            
            # combine
            log_beta = log_lam + term2_logsumexp - term3_logsumexp
            
            
            ### gamma, one minus gamma IN LOG SPACE
            # numerator = mu * beta; log(numerator) = log(mu) + log(beta)
            gamma_numerator = log_mu + log_beta
            
            # denom = lambda * (1-alpha); log(denom) = log(lam) + log(1-alpha)
            gamma_denom = log_lam + log_one_minus_alpha
            
            # 1 - gamma = num/denom; log(1 - gamma) = log(num) - log(denom)
            log_one_minus_gamma = gamma_numerator - gamma_denom
            
            return(log_beta, log_one_minus_x, False)
            
        
        #################################
        ### approximates of beta, gamma #
        #################################
        def approx_tkf_params(lam, mu, t_array):
            ### beta
            log_beta = ( safe_log(1 - self.tkf_err) + 
                         safe_log(lam_per_t) - 
                         safe_log(lam_per_t + 1) )
            
            
            ### gamma
            # numerator = mu * t; log(numerator) = log(mu * t)
            gamma_numerator = safe_log(mu_per_t)
            
            # denom = (1-alpha) (mu_per_t + 1); log(denom) = log(1-alpha) + log(mu_per_t + 1)
            gamma_denom = log_one_minus_alpha + safe_log(mu_per_t + 1)
            
            # 1 - gamma = num/denom; log(1 - gamma) = log(num) - log(denom)
            log_one_minus_gamma = gamma_numerator - gamma_denom
            
            return(log_beta, log_one_minus_gamma, True)
        
        
        ##############################
        ### finish and output values #
        ##############################
        # if any of the mu values are close, then use the approx. function
        #   not great... but more efficient than scanning over every value
        out = jax.lax.cond( use_approx,
                            orig_tkf_params,
                            approx_tkf_params,
                            lam, mu, t)
        log_beta, log_one_minus_gamma, used_approx = out
        del out
        
        log_gamma = log_one_minus_x(log_one_minus_gamma)
        log_one_minus_beta = log_one_minus_x(log_beta)
            
        out_dict = {'log_alpha': log_alpha,
                    'log_beta': log_beta,
                    'log_gamma': log_gamma,
                    'log_one_minus_alpha': log_one_minus_alpha,
                    'log_one_minus_beta': log_one_minus_beta,
                    'log_one_minus_gamma': log_one_minus_gamma,
                    'used_tkf_approx': used_approx
                    }
        
        return out_dict
        

###############################################################################
### TKF92_single   ############################################################
###############################################################################
# inherits the following from TKF91_single: __init__, tkf_params, 
#   logits_to_indel_rates, pytree transformation functions
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
            
            - offset (mu = lambda*(1+TKF_ERR) + offset)
              > DEFAULT: 0.0001
            
            - x (insertion extension probability)
              > DEFAULT:  0.5
              > DOMAIN RESTRICTION: (0, 1)
                  
            - y (deletion extension probability)
              > DEFAULT:  0.5
              > DOMAIN RESTRICTION: (0, 1)
        """
        provided_args = dir(argparse_obj)
        
        
        ### PARAMETERS: indel rate
        # first, check if "indel_rate" is provided, meaning you're 
        #   tying weights
        if 'indel_rate' in provided_args:
            lam = argparse_obj.indel_rate
            mu = lam / (1 - TKF_ERR)
        
        # if this isn't provided, see if lam or offset are provided; check these
        # seperately, in the weird case you may provide one but not the other
        else:
            # lam; DEFAULT=0.5
            if 'lam' in provided_args:
                lam = argparse_obj.lam
            else:
                lam = 0.5
            
            # offset (for mu); DEFAULT=0.1
            if 'offset' in provided_args:
                offset = argparse_obj.offset
            else:
                offset = 0.1
        
        
        ### PARAMETERS: extension probability
        # first, check if r was provided (extension probability); could be
        #   under several names... which is annoying
        if 'extension_prob' in provided_args:
            r = argparse_obj.extension_prob
        
        elif 'r' in provided_args:
            r = argparse_obj.r
        
        # default is 0.5
        else:
            r = 0.5
                
        ### keep lambda and r
        # transform to (-inf, inf) domain; also add extra k_indel dimension
        lam_transf =  jnp.sqrt(lam)[..., None]
        r_transf = jnp.sqrt(-jnp.log(r))[..., None]
        
        # create output param dictionary
        initialized_params = {'lam_transf': lam_transf,
                              'r_transf': r_transf}
        
        ### if not tying weights, add offset separately
        if not self.tie_params:
            # also transform offset to (-inf, inf)
            offset_transf = jnp.sqrt(-jnp.log(offset))[..., None]
            initialized_params['offset_transf'] = offset_transf
            
        return initialized_params, dict()
        
        
    def logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: calculate the transition matrix at time t
        JITTED: yes
        WHEN IS THIS CALLED: every training loop iteration, every point t
        OUTPUTS: logP(transitions); the GGI transition matrix
        """
        ### indel params
        lam, mu, use_approx = self.logits_to_indel_rates(params_dict)
        r_transf = params_dict['r_transf']
        r_fragment_len = jnp.exp(-jnp.square(r_transf))
        out_dict = self.tkf_params(lam, mu, t, use_approx)
        
        
        ### entries in the matrix
        # need log(r_fragment_len) and log(1 - r_fragment_len) for this
        log_r_fragment_len = safe_log(r_fragment_len)
        log_one_minus_r_fragment_len = log_one_minus_x(log_r_fragment_len)
        
        # a = r_fragment_len + (1-r_fragment_len)*(1-beta)*alpha
        # log(a) = logsumexp([r_fragment_len, 
        #                     log(1-r_fragment_len) + log(1-beta) + log(alpha)
        #                     ]
        #                    )
        log_a_second_half = ( log_one_minus_r_fragment_len + 
                              out_dict['log_one_minus_beta'] +
                              out_dict['log_alpha'] )
        log_a = logsumexp_with_arr_lst([log_r_fragment_len, log_a_second_half])
        
        # b = (1-r_fragment_len)*beta
        # log(b) = log(1-r_fragment_len) + log(beta)
        log_b = log_one_minus_r_fragment_len + out_dict['log_beta']
        
        # c_h = (1-r_fragment_len)*(1-beta)*(1-alpha)
        # log(c_h) = log(1-r_fragment_len) + log(1-beta) + log(1-alpha)
        log_c_h = ( log_one_minus_r_fragment_len +
                    out_dict['log_one_minus_beta'] +
                    out_dict['log_one_minus_alpha'] )

        # f = (1-r_fragment_len)*(1-beta)*alpha
        # log(f) = log(1-r_fragment_len) +log(1-beta) +log(alpha)
        log_f = ( log_one_minus_r_fragment_len +
                  out_dict['log_one_minus_beta'] +
                  out_dict['log_alpha'] )
        
        # g = r_fragment_len + (1-r_fragment_len)*beta
        # log(g) = logsumexp([r_fragment_len, 
        #                     log(1-r_fragment_len) + log(beta)
        #                     ]
        #                    )
        log_g_second_half = log_one_minus_r_fragment_len + out_dict['log_beta']
        log_g = logsumexp_with_arr_lst([log_r_fragment_len, log_g_second_half])
        
        # h and log(h) are the same as c and log(c) 

        # p = (1-r_fragment_len)*(1-gamma)*alpha
        # log(p) = log(1-r_fragment_len) + log(1-gamma) +log(alpha)
        log_p = ( log_one_minus_r_fragment_len +
                  out_dict['log_one_minus_gamma'] +
                  out_dict['log_alpha'] )

        # q = (1-r_fragment_len)*gamma
        # log(q) = log(1-r_fragment_len) + log(gamma)
        log_q = log_one_minus_r_fragment_len + out_dict['log_gamma']

        # r = r_fragment_len + (1-r_fragment_len)*(1-gamma)*(1-alpha)
        # log(r) = logsumexp([r_fragment_len, 
        #                     log(1-r_fragment_len) + log(1-gamma) + log(1-alpha)
        #                     ]
        #                    )
        log_r_second_half = ( log_one_minus_r_fragment_len +
                              out_dict['log_one_minus_gamma'] +
                              out_dict['log_one_minus_alpha'] )
        log_r = logsumexp_with_arr_lst([log_r_fragment_len, log_r_second_half])
        
        # (3, 3, k_indel)
        logprob_transition_at_t = jnp.array([ [log_a, log_b, log_c_h],
                                              [log_f, log_g, log_c_h],
                                              [log_p, log_q, log_r  ] ])
        
        return logprob_transition_at_t
    
    
    def undo_param_transform(self, params_dict):
        """
        ABOUT: if any parameters have domain changes, undo those
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary, with transformed params
        """
        lam, mu, use_approx = self.logits_to_indel_rates(params_dict)
        r_transf = params_dict['r_transf']
        r = jnp.exp(-jnp.square(r_transf))
        
        out_dict = {'lam_transf': params_dict['lam_transf'],
                    'r_transf': r_transf,
                    'extension_prob': r,
                    'use_approx': use_approx}
        
        if self.tie_params:
            # combine lambda and mu under label "indel rate"
            out_dict['indel_rate'] = lam
        
        elif not self.tie_params:
            out_dict['lam'] = lam
            out_dict['mu'] = mu
            out_dict['offset'] = offset
            out_dict['offset_transf'] = params_dict['offset_transf']
        
        return out_dict



###############################################################################
### LG05, RS07, KM03   ########################################################
###############################################################################
# this could be the template to combine all indel models into one class
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


