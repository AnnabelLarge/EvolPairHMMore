
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
     
2. subst_mixture: mixture of substitution models, given by set list of 
     exchangeability matrices


at a minimum, future classes need:
==================================
0. something in class init to decide whether or not to normalize the 
     rate matrix by the equilibrium distributions

1. initialize_params(self, argparse_obj): initialize all parameters and 
     hyperparameters; parameters are updated with optax, but hyperparameters
     just help the functions run (i.e. aren't updated)
     
2. conditional_logprobs_at_t(self, t, params_dict, hparams_dict): calculate 
     conditional probability of substitution- logP(x(t) | x(0))

3. joint_logprobs_at_t(self, t, params_dict, hparams_dict): calculate joint
      probability of substitution- 
      logP(x(t), x(0)) = logP(x(t) | x(0)) + logP( x(0) )
      (this could be inherited from base class, if not too involved)

4. norm_rate_matrix(self, subst_rate_mat, equl_pi_mat): normalizes the
     rate matrix (as detailed above)

5. undo_param_transform(self, params_dict): undo any domain transformations
     and output regular list/ints; mainly used for recording results to
     tensorboard, JSON, or anything else that doesn't like jax arrays


universal order of dimensions:
==============================
0. logprob matrix/vectors (i,j)
2. k_subst                (k)
3. k_equl                 (l)
4. k_indel


todo:
=====
- add a more manual mixture model that loads from multiple exchangeability 
  matrices and fit the mixture components (I don't plan on using it yet, but
  does give me more flexibility in future runs)

"""
import jax
from jax import numpy as jnp
from jax.nn import softmax
from jax.scipy.linalg import expm
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
        ### load one exchangeabilities file; argparse_obj.exch_files is a 
        ###   single string; final matrix is size (i,j, 1)
        file_to_load = f'exchangeability_matrices/{argparse_obj.exch_files}'
        with open(file_to_load,'rb') as f:
            exch_mat = jnp.expand_dims(jnp.load(f), -1)
        
        ### hyperparams dict
        hparams = {'alphabet_size': argparse_obj.alphabet_size,
                   'exch_mat': exch_mat}
        
        return dict(), hparams
    
    
    def conditional_logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: calculate logP(x(t)=j | x(0)=i); could multiply by counts 
               during training, if desired
        JITTED: yes
        WHEN IS THIS CALLED: every training loop, every timepoint
        OUTPUTS: logP(x(t)=j | x(0)=i), the conditional log-probability of 
                 substitutions
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
        
        # logP = log(expm(Rt))
        # R is (alph, alph, k_subst=1, k_equl), but expm needs square matrices
        for_mat_expm = R_mat * t
        for_mat_expm_reshaped = jnp.reshape( for_mat_expm, 
                                            (for_mat_expm.shape[0],
                                             for_mat_expm.shape[1],
                                             for_mat_expm.shape[2]*for_mat_expm.shape[3]) )
        exponentiated_raw = jax.vmap(expm, in_axes=2, out_axes=2)(for_mat_expm_reshaped)
        for_log = jnp.reshape(exponentiated_raw, for_mat_expm.shape)
        cond_logprob_substitution_at_t = jnp.log(for_log)
        
        return cond_logprob_substitution_at_t
    
    
    def joint_logprobs_at_t(self, t, params_dict, hparams_dict):
        """
        ABOUT: calculate logP(x(t)=j, x(0)=i); could multiply by counts 
               during training, if desired
        JITTED: yes
        WHEN IS THIS CALLED: every training loop, every timepoint
        OUTPUTS: logP(x(t)=j, x(0)=i), the joint log-probability of 
                 substitutions
                 a (alphabet_size, alphabet_size, 1, k_equl) tensor
        """
        # the conditional logprobability
        # (alph, alph, k_subst=1, k_equl)
        cond_logprob = self.conditional_logprobs_at_t(t, params_dict, hparams_dict)
        
        # unpack extra hyperparameters
        logP_equl = hparams_dict['logP_equl']
        
        # add logP(equilibrium distribution): logP(i,j) = logP(i) + logP(j|i)
        # cond_logprob is (alph, alph, k_subst=1, k_equl)  (i,j,k,l)
        # equl_dists is (alphabet_size, k_equl)            (i,l)
        joint_logprob_substitution_at_t = (cond_logprob +
                                           jnp.expand_dims(logP_equl, (1,2))
                                           )
        
        return joint_logprob_substitution_at_t
        
    
    def norm_rate_matrix(self, subst_rate_mat, equl_pi_mat):
        """
        ABOUT: this normalizes the rate matrix by the equilibrium vectors, 
               if self.norm is true
        JITTED: yes
        WHEN IS THIS CALLED: whenever logprobs_at_t is called, if self.norm 
                             is true
        OUTPUTS: normalized rate matrix
        """
        ### diag will be (k_subst, k_equl, alphabet_size) (k,l,i)
        diag = jnp.diagonal(subst_rate_mat, axis1=0, axis2=1)
        
        ### equl_dists is (alphabet_size, k_equl) (i, l)
        ###    output is (k_subst, k_equl) (k,l)
        norm_factor = -jnp.einsum('kli, il -> kl', diag, equl_pi_mat)
        
        ### divide each subst_rate_mat with this vec (thanks jnp broadcasting!)
        # (alphabet_size, alphabet_size, k_subst, k_equl) / (k_subst, k_equl)
        out = subst_rate_mat / norm_factor
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
        ### create rate matrix
        # (alphabet_size, alphabet_size, 1, k_equl) (i,j,k,l)
        # fill in values for i != j 
        raw_rate_mat = jnp.einsum('ijk, jl -> ijkl', exch_mat, equl_vecs)
        
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
### mixture substitution model (from list of exchang. matrices)   #############
###############################################################################
# inherits the following methods: _init_, conditional_logprobs_at_t, 
#   joint_logprobs_at t, norm_rate_matrix, and jax pytree info
class subst_mixture(subst_base):
    def initialize_params(self, argparse_obj):
        """
        ABOUT: return (possibly transformed) parameters and hyperparams
        JITTED: no
        WHEN IS THIS CALLED: once, upon model instantiation
        OUTPUTS: dictionary of initial parameters for optax (if any)
        
        params to fit:
            - mixture logits
              > DEFAULT: vector of 1s, length of k_subst
        
        hparams to pass on (or infer):
            - alphabet_size
            - exchangeability matrix
        """
        provided_args = dir(argparse_obj)
        
        ### PARAMETER: mixture logits
        # if not provided, generate a logits vector of ones
        if 'subst_mix_logits' not in provided_args:
            subst_mix_logits = jnp.ones(argparse_obj.k_subst)
        
        # if provided, just use what's provided
        else:
            subst_mix_logits = jnp.array(argparse_obj.subst_mix_logits, dtype=float)
        
        
        ### load from LIST of exchangeabilities files
        # (alph, alph, k_subst)
        exch_mat = []
        for file in argparse_obj.exch_files:
            with open(f'exchangeability_matrices/{file}', 'rb') as f:
                one_mat = jnp.expand_dims(jnp.load(f), -1)
            exch_mat.append(one_mat)
        exch_mat = jnp.concatenate(exch_mat, -1)
        
        
        ### OUTPUT DICTIONARIES
        # dictionary of parameters
        initialized_params = {'subst_mix_logits': subst_mix_logits}
        
        # dictionary of hyperparameters
        hparams = {'alphabet_size': argparse_obj.alphabet_size,
                   'exch_mat': exch_mat}
        
        return initialized_params, hparams


    def undo_param_transform(self, params_dict):
        """
        ABOUT: if any parameters have domain changes, undo those
        JITTED: no
        WHEN IS THIS CALLED: when writing params to JSON file
        OUTPUTS: parameter dictionary, with transformed params
        """
        ### unpack parameters
        subst_mix_logits = params_dict['subst_mix_logits']
        
        
        ### undo the domain transformation
        subst_mix_probs = softmax(subst_mix_logits)
        
        # also turn them into regular lists, for writing JSON
        subst_mix_probs = np.array(subst_mix_probs).tolist()
        
        
        ### add to parameter dictionary
        out_dict = {}
        out_dict['subst_mix_probs'] = subst_mix_probs
        
        return out_dict

