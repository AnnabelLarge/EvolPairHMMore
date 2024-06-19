#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:30:30 2024

@author: annabel

Make a class for retrieving hky85

CONFIRMED that this logprob matrix matches Ian's output
"""
from jax import numpy as jnp
from jax.scipy.linalg import expm


class hky85:
    def __init__(self, norm):
        self.norm = norm
    
    def initialize_params(self, argparse_obj):
        ### hyperparams dict
        hparams = {'gc': argparse_obj.gc,
                   'ti': argparse_obj.ti,
                   'tv': argparse_obj.tv}
        
        return dict(), hparams
    
    def conditional_logprobs_at_t(self, t, params_dict, hparams_dict):
        # unpack extra hyperparameters
        gc = hparams_dict['gc']
        ti = hparams_dict['ti']
        tv = hparams_dict['tv']
        
        # generate the rate matrix
        eqm = jnp.array([(1-gc)/2, gc/2, gc/2, (1-gc)/2])
        idx = range(len(eqm))
        raw = [[eqm[j] * (ti if i & 1 == j & 1 else tv) 
                for j in idx] for i in idx]
        R_mat = self.zero_rate_matrix_row_sums (jnp.array (raw))
        
        # normalize if desired
        if self.norm:
            R_mat = self.norm_rate_matrix(R_mat, eqm)
            R_mat = self.zero_rate_matrix_row_sums (jnp.array (R_mat))
        
        # submat = log(exp(Rt)); (alph, alph)
        # if needed, place some condition here for if R_mat * t == 0
        cond_logprob_substitution_at_t = jnp.log(expm(R_mat * t))
        
        # for now, this ignores potential for k_subst or k_equl; just expand
        # these dims
        cond_logprob_substitution_at_t = jnp.expand_dims(cond_logprob_substitution_at_t,
                                                         (-1, -2))
        
        return cond_logprob_substitution_at_t
    
    
    def joint_logprobs_at_t(self, t, params_dict, hparams_dict):
        raise NotImplementedError
    
    
    def norm_rate_matrix(self, subst_rate_mat, equl_pi_mat):
        return subst_rate_mat / self.expected_rate_at_eqm (subst_rate_mat, equl_pi_mat)
    
    
    def undo_param_transform(self, params_dict):
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
    def zero_rate_matrix_row_sums (self, mx):
        # note: this only works when mx is (n, n); no further dims
        mx_abs = jnp.abs (mx)
        mx_diag = jnp.diagonal (mx_abs)
        mx_no_diag = mx_abs - jnp.diag (mx_diag)
        mx_rowsums = mx_no_diag @ jnp.ones_like (mx_diag)
        return mx_no_diag - jnp.diag (mx_rowsums)
    
    def expected_rate_at_eqm (self, submat, eqm):
        # note: this only works when mx is (n, n); no further dims
        submat = self.zero_rate_matrix_row_sums (submat)
        return -jnp.diagonal(submat) @ eqm
    


if __name__ == '__main__':
    class FakeArgparse:
        def __init__(self, gc, ti, tv, alphabet_size=4):
            self.gc = gc
            self.ti = ti
            self.tv = tv
            self.alphabet_size = alphabet_size
    
    args = FakeArgparse(gc=0.44,
                        ti=1,
                        tv=1)
    
    hky_instance = hky85(norm=True)
    _, hky_hyperparams = hky_instance.initialize_params(args)
    logprobs = hky_instance.conditional_logprobs_at_t(t = 0.1, 
                                                      params_dict = dict(), 
                                                      hparams_dict = hky_hyperparams)
    
    print(logprobs[:,:,0,0])
