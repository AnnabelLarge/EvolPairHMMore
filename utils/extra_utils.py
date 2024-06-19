#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:50:56 2024

@author: annabel_large


ABOUT:
=======

in use:
--------
1. logsumexp_new: the same as jax.scipy.special.logsumexp, except you can 
                  include which elements to include in the reduction
                  (this is directly from jax source code; maybe will be in 
                   next jax release?)
 
not in use:
-----------
2. logsumexp_withZeros: old way to do logsumexp with a "where" option

3. make_fake_batch: gives me a fake batch of sequences; used during code dev


"""
import jax
from jax import numpy as jnp
from jax import lax
from jax._src.numpy.reductions import _reduction_dims, Axis
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike
import numpy as np


def logsumexp_where(a, axis, where, 
                    b = None, keepdims = False, return_sign = False):
    """Log-sum-exp reduction with an argument to determine which elems to 
    include (almost directly from latest jax source code)
    
    Computes
    
    .. math::
      \mathrm{logsumexp}(a) = \mathrm{log} \sum_j b \cdot \mathrm{exp}(a_{ij})
    
    where the :math:`j` indices range over one or more dimensions to be reduced.
    
    Args:
      a: the input array
      axis: the axis or axes over which to reduce. May be either ``None``, an
        int, or a tuple of ints.
      b: scaling factors for :math:`\mathrm{exp}(a)`. Must be broadcastable to the
        shape of `a`.
      keepdims: If ``True``, the axes that are reduced are left in the output as
        dimensions of size 1.
      return_sign: If ``True``, the output will be a ``(result, sign)`` pair,
        where ``sign`` is the sign of the sums and ``result`` contains the
        logarithms of their absolute values. If ``False`` only ``result`` is
        returned and it will contain NaN values if the sums are negative.
      where: Elements to include in the reduction.
    
    Returns:
      Either an array ``result`` or a pair of arrays ``(result, sign)``, depending
      on the value of the ``return_sign`` argument.
    """
    if b is not None:
        a_arr, b_arr = promote_args_inexact("logsumexp", a, b)
        a_arr = jnp.where(b_arr != 0, a_arr, -jnp.inf)
    else:
        a_arr, = promote_args_inexact("logsumexp", a)
        b_arr = a_arr  # for type checking
    pos_dims, dims = _reduction_dims(a_arr, axis)
    amax = jnp.max(a_arr.real, axis=dims, keepdims=keepdims, where=where, initial=-jnp.inf)
    amax = lax.stop_gradient(lax.select(jnp.isfinite(amax), amax, lax.full_like(amax, 0)))
    amax_with_dims = amax if keepdims else lax.expand_dims(amax, pos_dims)
    
    exp_a = lax.exp(lax.sub(a_arr, amax_with_dims.astype(a_arr.dtype)))
    if b is not None:
        exp_a = lax.mul(exp_a, b_arr)
    sumexp = exp_a.sum(axis=dims, keepdims=keepdims, where=where)
    sign = lax.sign(sumexp)
    if return_sign or not np.issubdtype(a_arr.dtype, np.complexfloating):
        sumexp = abs(sumexp)
    out = lax.add(lax.log(sumexp), amax.astype(sumexp.dtype))
    
    if return_sign:
        return (out, sign)
    if b is not None and not np.issubdtype(out.dtype, np.complexfloating):
        with jax.debug_nans(False):
            out = jnp.where(sign < 0, jnp.array(np.nan, dtype=out.dtype), out)
    return out


def make_fake_batch():
    """
    For testing small bits of code, if needed
    """
    # has one deletion
    samp1 = jnp.array([[3, 4,  5, 6, 7, 0],
                        [3, 4, 63, 6, 7, 0]])
    
    # has one insertion
    samp2 = jnp.array([[3, 4, 63, 6, 7, 8],
                        [3, 4,  5, 6, 7, 8]])
    
    # has one substitution
    samp3 = jnp.array([[3, 4,  5, 6, 0, 0],
                        [3, 4, 12, 6, 0, 0]])
    
    # wrap in a batch; final size is (3, 2, 6)
    fake_batch = jnp.concatenate([jnp.expand_dims(samp1, 0),
                                  jnp.expand_dims(samp2, 0),
                                  jnp.expand_dims(samp3, 0)], 0)
    return fake_batch





# def logsumexp_OLD(x, axis):
#     """
#     this used to be the version of logsumexp I used (which would ignore values
#       whenever the input was zero)
    
#     problem was: it doesn't keep as many significant digits as 
#                  jax's own logsumexp
    
#     see this about NaN's and jnp.where-
#     https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-
#       where-using-where:~:text=f32%5B1%2C3%5D%7B1%2C0%7D-,Gradients%20
#       contain%20NaN%20where%20using%20where,-%23
#     """
#     zero_val_mask = jnp.where(x != 0,
#                               1,
#                               0)
    
#     exp_x = jnp.exp(x)
#     exp_x_masked = exp_x * zero_val_mask
#     sumexp_x = jnp.sum(exp_x_masked, axis=axis)
    
#     logsumexp_x = jnp.log(jnp.where(sumexp_x > 0., 
#                                     sumexp_x, 
#                                     1.))
#     return logsumexp_x


