#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 22:04:27 2024

@author: annabel

ABOUT:
======
Protein substitution models:
    
1. subst_base: single substitution model with externally provided 
     exchangeability matrix (I use LG08 matrix)
     
2. LG_mixture: mixture model that creates rate classes from quantiles
     of a gamma distribution, which are multiplied by externally provided
     exchangeability matrix (I'm mostly replicating the LG paper, but
     NOT including zero-inflation param; I use the LG08 matrix)


shared class methods:
=====================
1. initialize_model(self, inputs_dict): initialize all parameters and 
     hyperparameters; parameters are updated with optax, but hyperparameters
     just help the functions run (i.e. aren't updated)
     
2. logprobs_at_t(self, t, params_dict, hparams_dict): calculate 
     logP(substitutions) at time t
       

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
from tensorflow_probability.substrates import jax as tfp


###############################################################################
### single substitution model   ###############################################
###############################################################################
class subst_base:
    def initialize_model(self, inputs_dict):
        """
        ABOUT: return (possibly transformed) parameters
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit: (none, return empty dictionary)
        hparams to pass on (or infer): (none,return empty dictionary)
        """
        return dict(), dict()
    
    
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
        lg_exch_file = hparams_dict['lg_exch_file']
        
        # calculate the log probabilities at time t
        # log(exp(Rt))
        R_mat = self.generate_rate_matrix(equl_vecs, lg_exch_file)
        logprob_substitution_at_t = R_mat * t
        
        return logprob_substitution_at_t
    
    
    ###############   v__(extra functions placed below)__v   ###############    
    def generate_rate_matrix(self, equl_vecs, lg_exch_file):
        """
        ABOUT: calculating rate matrix R
        JITTED: yes
        WHEN IS THIS CALLED: whenever logprobs_at_t is called
        OUTPUTS: (alphabet_size, alphabet_size, 1, k_equl) tensor
        """
        ### load LG exchangeabilities file
        # (alphabet_size, alphabet_size) (i,j)
        with open(lg_exch_file,'rb') as f:
            exch_mat = jnp.load(f)
        
        ### no rate classes, so just expand dimensions to make k_subst axis
        # (alphabet_size, alphabet_size, 1) (i,j,k)
        exch_mat_perClass = jnp.expand_dims(exch_mat, axis=-1)
        
        ### create rate matrix
        # (alphabet_size, alphabet_size, 1, k_equl) (i,j,k,l)
        # fill in values for i != j 
        raw_rate_mat = jnp.einsum('ijk, il -> ijkl', exch_mat, equl_vecs)
        
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



###############################################################################
### mixture substitution model (using the LG rate class method)   #############
###############################################################################
class LG_mixture:
    def initialize_model(self, inputs_dict):
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
        """
        ### PARAMETER: gamma shape
        gamma_shape = inputs_dict.get('gamma_shape', 1)
        
        # has to be greater than zero
        err_msg = ('Initial guess for GAMMA SHAPE must be greater than zero,'+
                  f' but recieved gamma_shape={gamma_shape}')
        assert gamma_shape > 0, err_msg
        del err_msg
        
        # for stochastic gradient descent, transform to (-inf, inf) domain
        gamma_shape_transf = jnp.sqrt(gamma_shape)
        
        ### PARAMETER: mixture logits
        # if not provided, generate a logits vector of ones
        if not inputs_dict.get('subst_mix_logits', None):
            err_msg = ('SUBSTITUTION model underspecifed: If not manually '+
                       'initializing subst_mix_logits, need to specify '+
                       'how many SUBSTITUTION mixtures with k_subst=int')
            assert inputs_dict.get('k_susbt'), err_msg
            del err_msg
            
            subst_mix_logits = jnp.ones(inputs_dict['k_subst'])
        
        # if provided, just use what's provided
        else:
            subst_mix_logits = inputs_dict['subst_mix_logits']
        
        ### HYPERPARAMETER: k_subst
        # either provided already, or inferred from length of subst_mix_logits
        k_subst = inputs_dict.get('k_subst', subst_mix_logits.shape[0])
        
        ### OUTPUT DICTIONARIES
        # dictionary of parameters
        initialized_params = {'gamma_shape_transf': gamma_shape_transf,
                              'subst_mix_logits': subst_mix_logits}
        
        # dictionary of hyperparameters
        hparams = {'k_subst': k_subst}
        
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
        k_subst = hparams_dict['k_subst']
        lg_exch_file = hparams_dict['lg_exch_file']
        
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
        
        
        ### calculate the log probabilities at time t: log(exp(Rt))
        R_mat = self.generate_rate_matrix(equl_vecs, lg_exch_file, 
                                          gamma_shape, k_subst)
        logprob_substitution_at_t = R_mat * t
        
        return logprob_substitution_at_t
    
    
    ###############   v__(extra functions placed below)__v   ###############
    def generate_rate_matrix(self, equl_vecs, lg_exch_file, 
                             gamma_shape, k_subst):
        """
        ABOUT: calculating rate matrix R
        JITTED: yes
        WHEN IS THIS CALLED: whenever logprobs_at_t is called
        OUTPUTS: (alphabet_size, alphabet_size, k_subst, k_equl) tensor
        """
        ### load LG exchangeabilities file
        # (alphabet_size, alphabet_size) (i,j)
        with open(lg_exch_file,'rb') as f:
            exch_mat = jnp.load(f)
        
        ### multiply exchangeabilities by rate classes
        # (alphabet_size, alphabet_size, k_subst) (i,j,k)
        rate_classes = self.generate_rate_classes(gamma_shape, k_subst)
        exch_mat_perClass = jnp.einsum('ij,k->ijk', exch_mat, rate_classes)
        
        ### create rate matrix
        # (alphabet_size, alphabet_size, k_subst, k_equl) (i,j,k,l)
        # fill in values for i != j 
        raw_rate_mat = jnp.einsum('ijk, il -> ijkl', exch_mat, equl_vecs)
        
        # mask out only (i,j) diagonals
        mask_inv = jnp.abs(1 - jnp.eye(raw_rate_mat.shape[0]))
        rate_mat_without_diags = jnp.einsum('ijkl,ij->ijkl', 
                                            raw_rate_mat, mask_inv)

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
        # quantile for 1, but can get pretty close with the 99th percentile
        quantiles_except_last = jnp.linspace(0, 1, k_subst+1)[:-1]
        quantiles = jnp.concatenate([quantiles_except_last, jnp.array([0.99])])
        
        # retrieve the xvalues of the quantiles
        xvals_at_quantiles = gamma_dist.quantile(points_to_gen)
        
        # rate classes are points BETWEEN these
        rate_classes = (xvals_at_quantiles[1:] + xvals_at_quantiles[:-1]) / 2
        
        return rate_classes
