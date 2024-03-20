
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 22:04:27 2024

@author: annabel

ABOUT:
======
Protein substitution models (i.e. the EMISSIONS from match states):
    
1. subst_base: single substitution model with externally provided 
     exchangeability matrix (I use LG08 matrix)
     
2. LG_mixture: mixture model that creates rate classes from quantiles
     of a gamma distribution, which are multiplied by externally provided
     exchangeability matrix (I'm mostly replicating the LG paper, but
     NOT including zero-inflation param; I use the LG08 matrix)
     > NOTE: this inherits a lot of its methods from subst_base 


at a minimum, future classes need:
==================================
0. something in class init to decide whether or not to normalize the 
     rate matrix by the equilibrium distributions

1. initialize_params(self, argparse_obj): initialize all parameters and 
     hyperparameters; parameters are updated with optax, but hyperparameters
     just help the functions run (i.e. aren't updated)
     
2. logprobs_at_t(self, t, params_dict, hparams_dict): calculate 
     logP(substitutions) at time t

3. norm_rate_matrix(self, subst_rate_mat, equl_pi_mat): normalizes the
     rate matrix (as detailed above)

4. undo_param_transform(self, params_dict): undo any domain transformations
     and output regular list/ints; mainly used for recording results to
     tensorboard, JSON, or anything else that doesn't like jax arrays


universal order of dimensions:
==============================
0. logprob matrix/vectors
2. k_subst
3. k_equl
4. k_indel


todo:
=====
- add a more manual mixture model that loads from multiple exchangeability 
  matrices and fit the mixutre parameters (I don't plan on using it yet, but
  does give me more flexibility in future runs)

"""
import jax
from jax import numpy as jnp
from jax.nn import softmax
from tensorflow_probability.substrates import jax as tfp
import copy
import numpy as np


###############################################################################
### single substitution model   ###############################################
###############################################################################
class subst_base:
    def __init__(self, norm):
        """
        just need this to give myself the option for normalizing the
        rate matrix
        """
        self.norm = norm
        
        
    def initialize_params(self, argparse_obj):
        """
        ABOUT: return (possibly transformed) parameters
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit: (none, return empty dictionary)
        hparams to pass on (or infer): 
            - alphabet_size
            - exchangeabilities matrix
        """
        ### load exchangeabilities file
        file_to_load = f'{argparse_obj.data_dir}/{argparse_obj.exch_file}'
        with open(file_to_load,'rb') as f:
            exch_mat = jnp.load(f)
        
        ### hyperparams dict
        hparams = {'alphabet_size': argparse_obj.alphabet_size,
                   'exch_mat': exch_mat}
        
        return dict(), hparams
    
    
    def logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: calculate the logP(substitutions); multiply by counts 
               during training
        JITTED: yes
        WHEN IS THIS CALLED: every training loop, every timepoint
        OUTPUTS: logP(substitutions)
                 a (alphabet_size, alphabet_size, 1, k_equl) tensor
        """
        # unpack extra hyperparameters
        equl_vecs = hparams_dict['equl_vecs']
        exch_mat = hparams_dict['exch_mat']
        
        # generate the rate matrix
        R_mat = self.generate_rate_matrix(equl_vecs, exch_mat)
    
        # normalize if desired
        if self.norm:
            R_mat = self.norm_rate_matrix(R_mat, equl_vecs)
        
        # multiply by time
        # log(exp(Rt)); (alph, alph, k_subst=1, k_equl)
        logprob_substitution_at_t = R_mat * t
        return logprob_substitution_at_t
    
    
    def norm_rate_matrix(self, subst_rate_mat, equl_pi_mat):
        """
        ABOUT: this normalizes the rate matrix by the equilibrium vectors, 
               if self.norm is true
        JITTED: yes
        WHEN IS THIS CALLED: whenever logprobs_at_t is called, if self.norm 
                             is true
        OUTPUTS: normalized rate matrix
        """
        ### diag will be (k_subst, k_equl, alphabet_size) (i,j,k)
        diag = jnp.diagonal(subst_rate_mat, axis1=0, axis2=1)
        
        ### equl_dists is (alphabet_size, k_equl) (k, j)
        ###    output is (k_subst, k_equl) (i,j)
        R_times_pi = -jnp.einsum('ijk, kj -> ij', diag, equl_pi_mat)
        
        ### divide each subst_rate_mat with this vec (thanks jnp broadcasting!)
        # (alphabet_size, alphabet_size, k_subst, k_equl) / (k_subst, k_equl)
        out = subst_rate_mat / R_times_pi
        return out
    
    
    def undo_param_transform(self, params_dict):
        """
        ABOUT: placeholder function
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary as-is (empty)
        """
        return  dict()
    
    
    ###  v__(these allow the class to be passed into a jitted function)__v  ###
    def _tree_flatten(self):
        children = ()
        aux_data = {'norm': self.norm} 
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
    
    ###############   v__(extra functions placed below)__v   ###############    
    def generate_rate_matrix(self, equl_vecs, exch_mat):
        """
        ABOUT: calculating rate matrix R
        JITTED: yes
        WHEN IS THIS CALLED: whenever logprobs_at_t is called
        OUTPUTS: (alphabet_size, alphabet_size, 1, k_equl) tensor
        """
        ### no rate classes, so just expand dimensions to make k_subst axis
        # (alphabet_size, alphabet_size, 1) (i,j,k)
        exch_mat_perClass = jnp.expand_dims(exch_mat, -1)
        
        ### create rate matrix
        # (alphabet_size, alphabet_size, 1, k_equl) (i,j,k,l)
        # fill in values for i != j 
        raw_rate_mat = jnp.einsum('ijk, il -> ijkl', exch_mat_perClass, equl_vecs)
        
        # mask out only (i,j) diagonals
        mask_inv = jnp.abs(1 - jnp.eye(raw_rate_mat.shape[0]))
        rate_mat_without_diags = jnp.einsum('ijkl,ij->ijkl', raw_rate_mat, mask_inv)

        # find rowsums i.e. sum across columns j
        row_sums = rate_mat_without_diags.sum(axis=1)
        neg_row_sums = -row_sums
        row_sums_repeated = jnp.repeat(a=jnp.expand_dims(neg_row_sums, 1),
                                        repeats=raw_rate_mat.shape[0],
                                        axis=1)
        mask = jnp.eye(raw_rate_mat.shape[0])
        diags_to_add = jnp.einsum('ijkl,ij->ijkl', row_sums_repeated, mask)

        # add both
        subst_rate_mat = rate_mat_without_diags + diags_to_add
        
        return subst_rate_mat
    


###############################################################################
### mixture substitution model (using the LG rate class method)   #############
###############################################################################
class LG_mixture(subst_base):
    def initialize_params(self, argparse_obj):
        """
        ABOUT: return (possibly transformed) parameters and hyperparams
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit:
            - gamma shape
              > DEFAULT: 1
              > DOMAIN RESTRICTION: must be greater than zero
              
            - mixture logits
              > DEFAULT: vector of 1s, length of k_subst
        
        hparams to pass on (or infer):
            - k_subst
              > DEFAULT: length of mixture logits vector
            - alphabet_size
            - exchangeability matrix
        """
        provided_args = dir(argparse_obj)
        
        ### PARAMETER: gamma shape
        # if not provided, set to 1
        if 'gamma_shape' not in provided_args:
            gamma_shape = 1
            
        # otherwise, read from argparse object and make sure the domain
        #   restriction is satisfied
        else:
            gamma_shape = argparse_obj.gamma_shape
            
            # has to be greater than zero
            err_msg = ('Initial guess for GAMMA SHAPE must be greater than zero,'+
                      f' but recieved gamma_shape={gamma_shape}')
            assert gamma_shape > 0, err_msg
            del err_msg
        
        # for stochastic gradient descent, transform to (-inf, inf) domain
        gamma_shape_transf = jnp.sqrt(gamma_shape)
        
        
        ### PARAMETER: mixture logits
        # if not provided, generate a logits vector of ones
        if 'subst_mix_logits' not in provided_args:
            subst_mix_logits = jnp.ones(argparse_obj.k_subst)
        
        # if provided, just use what's provided
        else:
            subst_mix_logits = jnp.array(argparse_obj.subst_mix_logits, dtype=float)
        
        
        ### HYPERPARAMETER: k_subst
        # either provided already, or inferred from length of subst_mix_logits
        if 'k_subst' not in provided_args:
            k_subst = subst_mix_logits.shape[0]
        else:
            k_subst = argparse_obj.k_subst
        
        
        ### load exchangeabilities file
        file_to_load = f'{argparse_obj.data_dir}/{argparse_obj.exch_file}'
        with open(file_to_load,'rb') as f:
            exch_mat = jnp.load(f)
        
        
        ### OUTPUT DICTIONARIES
        # dictionary of parameters
        initialized_params = {'gamma_shape_transf': gamma_shape_transf,
                              'subst_mix_logits': subst_mix_logits}
        
        # dictionary of hyperparameters
        hparams = {'k_subst': k_subst,
                   'alphabet_size': argparse_obj.alphabet_size,
                   'exch_mat': exch_mat}
        
        return initialized_params, hparams
    
    
    def logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: calculate the logP(substitutions); multiply by counts
               during training
        JITTED: yes
        WHEN IS THIS CALLED: every training loop, every timepoint
        OUTPUTS: logP(substitutions)-
                 (alphabet_size, alphabet_size, k_subst, k_equl) tensor
        """
        ### unpack parameters (what gets passed in/out of optax for updating)
        gamma_shape_transf = params_dict['gamma_shape_transf']
        
        ### unpack extra hyperparameters
        equl_vecs = hparams_dict['equl_vecs']
        # k_subst = hparams_dict['k_subst']
        k_subst = params_dict['subst_mix_logits'].shape[0]
        exch_mat = hparams_dict['exch_mat']
        
        ### turn gamma_shape_transf into gamma_shape
        # make sure the gamma_shape_transf is not zero by shifting it to
        # a small (but still detectable) number
        gamma_shape_transf = jnp.where(gamma_shape_transf != 0,
                                       gamma_shape_transf,
                                       1e-10)
        
        # undo domain transformation
        # gamma_shape_transf: (-inf, inf)
        # gamma_shape: (0, inf); NOT inclusive of zero!
        gamma_shape = jnp.square(gamma_shape_transf)
        
        
        # generate the rate matrix
        # instead of k_subst, try passing in a shape of a parameter 
        # (should be the same thing)
        R_mat = self.generate_rate_matrix(equl_vecs, exch_mat, 
                                          gamma_shape, k_subst)
        
        # normalize if desired
        if self.norm:
            R_mat = self.norm_rate_matrix(R_mat, equl_vecs)
        
        # multiply by time
        # log(exp(Rt)); (alph, alph, k_subst, k_equl)
        logprob_substitution_at_t = R_mat * t
        return logprob_substitution_at_t
    
    
    def undo_param_transform(self, params_dict):
        """
        ABOUT: if any parameters have domain changes, undo those
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary, with transformed params
        """
        ### unpack parameters
        gamma_shape_transf = params_dict['gamma_shape_transf']
        subst_mix_logits = params_dict['subst_mix_logits']
        
        
        ### undo the domain transformation
        gamma_shape = jnp.square(gamma_shape_transf)
        subst_mix_probs = softmax(subst_mix_logits)
        
        # also turn them into regular lists, for writing JSON
        gamma_shape = np.array(gamma_shape).tolist()
        subst_mix_probs = np.array(subst_mix_probs).tolist()
        
        
        ### add to parameter dictionary
        out_dict = {}
        out_dict['gamma_shape'] = gamma_shape
        out_dict['subst_mix_probs'] = subst_mix_probs
        
        return out_dict
    
    
    ###############   v__(extra functions placed below)__v   ###############
    def generate_rate_matrix(self, equl_vecs, exch_mat, gamma_shape, k_subst):
        """
        ABOUT: calculating rate matrix R
        JITTED: yes
        WHEN IS THIS CALLED: whenever logprobs_at_t is called
        OUTPUTS: (alphabet_size, alphabet_size, k_subst, k_equl) tensor
        """
        ### get rate classes
        # (alphabet_size, alphabet_size, k_subst) (i,j,k)
        rate_classes = self.generate_rate_classes(gamma_shape, k_subst)
        exch_mat_perClass = jnp.einsum('ij,k->ijk', exch_mat, rate_classes)
        
        ### create rate matrix
        # (alphabet_size, alphabet_size, k_subst, k_equl) (i,j,k,l)
        # fill in values for i != j 
        raw_rate_mat = jnp.einsum('ijk, il -> ijkl', exch_mat_perClass, equl_vecs)
        
        # mask out only (i,j) diagonals
        mask_inv = jnp.abs(1 - jnp.eye(raw_rate_mat.shape[0]))
        rate_mat_without_diags = jnp.einsum('ijkl,ij->ijkl', raw_rate_mat, mask_inv)

        # find rowsums i.e. sum across columns j
        row_sums = rate_mat_without_diags.sum(axis=1)
        row_sums_repeated = jnp.repeat(a=jnp.expand_dims(-row_sums, 1),
                                        repeats=raw_rate_mat.shape[0],
                                        axis=1)
        mask = jnp.eye(raw_rate_mat.shape[0])
        diags_to_add = jnp.einsum('ijkl,ij->ijkl', row_sums_repeated, mask)

        # add both
        subst_rate_mat = rate_mat_without_diags + diags_to_add
        
        return subst_rate_mat
    
    def generate_rate_classes(self, gamma_shape, k_subst):
        """
        ABOUT: given the gamma shape, generate vector of rate classes by
               interpolating between k_subst quantiles
        JITTED: yes
        WHEN IS THIS CALLED: whenever generate_rate_matrix is called
        OUTPUTS: rate class vector, rho
        """
        # generate a one-parameter gamma distribution
        gamma_dist = tfp.distributions.Gamma(concentration=gamma_shape,
                                             rate=gamma_shape)
        
        # determine which quantiles to generate; can't generate the
        # quantile for 0 or 1, so replace these with 0.01 and 0.99 respectively
        middle_quantiles = jnp.linspace(0, 1, k_subst+1)[1:-1]
        points_to_gen = jnp.concatenate( [ jnp.array([0.01]),
                                          middle_quantiles, 
                                          jnp.array([0.99]) ] )
        
        # retrieve the xvalues of the quantiles
        xvals_at_quantiles = gamma_dist.quantile(points_to_gen)
        
        # rate classes are points BETWEEN these
        rate_classes = (xvals_at_quantiles[1:] + xvals_at_quantiles[:-1]) / 2
        
        return rate_classes
