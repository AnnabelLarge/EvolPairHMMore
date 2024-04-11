#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:27:33 2024

@author: annabel

Functions to calculate alignment-depending log likelihoods 
for the GGI model (from Ian)
"""
import jax
from jax import numpy as jnp
import diffrax
from diffrax import (diffeqsolve, ODETerm, Dopri5, PIDController, 
                     ConstantStepSize, SaveAt)


# We replace zeroes and infinities with small numbers sometimes
# It's sinful but BOY DO I LOVE SIN
smallest_float32 = jnp.finfo('float32').smallest_normal

#####################################
### TRANSITION MATRIX FUNCTIONS     #
#####################################
### calculate L, M
def lm (t, rate, prob):
    num = -rate * t
    denom = 1. - prob
    frac = (num/denom)
    return jnp.exp (frac)

def indels (t, rate, prob):
    return 1. / lm(t,rate,prob) - 1.

# test whether time is past threshold of alignment signal being undetectable
def alignmentIsProbablyUndetectable (t, indelParams, alphabetSize):
    lam,mu,x,y = indelParams
    expectedMatchRunLength = 1. / (1. - jnp.exp(-mu*t))
    expectedInsertions = indels(t,lam,x)
    expectedDeletions = indels(t,mu,y)
    kappa = 2.
    return jnp.where (t > 0.,
                      ((expectedInsertions + 1) * (expectedDeletions + 1)) > kappa * (alphabetSize ** expectedMatchRunLength),
                      False)

# initial transition matrix
def zeroTimeTransitionMatrix (indelParams):
    """
    matrix output should be:
        [[1.,   0., 0.],
         [1.-x, x,  0.],
         [1.-y, 0., y]]
    """
    lam,mu,x,y = indelParams
    
    # row 1: M -> (M, I, D); (3, k_indel)
    mat_a = jnp.ones( (1, 1, lam.shape[0]))
    mat_b = jnp.zeros((1, 1, lam.shape[0]))
    mat_c = jnp.zeros((1, 1, lam.shape[0]))
    mat_Mrow = jnp.concatenate([mat_a, mat_b, mat_c], axis=1)
    
    # row 2:  I -> (M, I, D); (3, k_indel)
    mat_f = jnp.expand_dims( (1. - x), (0,1))
    mat_g = jnp.expand_dims( x,        (0,1))
    mat_h = jnp.zeros((1, 1, lam.shape[0]))
    mat_Irow = jnp.concatenate([mat_f, mat_g, mat_h], axis=1)
    
    # row 3:  D -> (M, I, D); (3, k_indel)
    mat_p = jnp.expand_dims( (1. - y), (0,1))
    mat_q = jnp.zeros((1, 1, lam.shape[0]))
    mat_r = jnp.expand_dims( y,        (0,1))
    mat_Drow = jnp.concatenate([mat_p, mat_q, mat_r], axis=1)
    
    # concatenate all and output (3, 3, k_indel)
    out_matrix = jnp.concatenate([mat_Mrow, mat_Irow, mat_Drow], axis=0)
    return out_matrix

# convert counts (a,b,u,q) to transition matrix ((a,b,c),(f,g,h),(p,q,r))
def smallTimeTransitionMatrix (t, indelParams, /, **kwargs):
    """
    matrix output should be:
    
    [[a,                     b,                             1-a-b],
     [u*L/one_minus_L,       1-(b+q*(1-M)/M)*L/one_minus_L, (b+q*(1-M)/M-u)*L/one_minus_L],
     [(1-a-u)*M/one_minus_M, q,                             1-q-(1-a-u)*M/one_minus_M]])
    """
    lam,mu,x,y = indelParams
    step = kwargs.get('step', None)
    rtol = kwargs.get('rtol', None)
    atol = kwargs.get('atol', None)
    
    # vmap integrateCounts across the different mixture models
    vmapped_integrateCounts = jax.vmap(integrateCounts, 
                                       in_axes=(None,
                                                (0,0,0,0),
                                                None, None, None))
    
    # run integrateCounts for every mixture model; output is (k_indel, 4)
    out = vmapped_integrateCounts(t, indelParams, step, rtol, atol)
    # To use the non-diffrax version, comment out the previous line and turn 
    #   the following into a vmapped function:
    # a,b,u,q = integrateCounts_RK4(t,indelParams,dt0=.1/jnp.maximum(lam,mu))
    
    # unpack outputs; order of outputs should be: a, b, u, q
    a = out[:, 0]
    b = out[:, 1]
    u = out[:, 2]
    q = out[:, 3]
    
    L = lm(t,lam,x)
    M = lm(t,mu,y)
    one_minus_L = jnp.where (L < 1., 1. - L, smallest_float32)   # avoid NaN gradient at zero
    one_minus_M = jnp.where (M < 1., 1. - M, smallest_float32)   # avoid NaN gradient at zero
    
    # row 1: M -> (M, I, D); (3, k_indel)
    mat_a = jnp.expand_dims( a,     (0,1))
    mat_b = jnp.expand_dims( b,     (0,1))
    mat_c = jnp.expand_dims( 1-a-b, (0,1))
    mat_Mrow = jnp.concatenate([mat_a, mat_b, mat_c], axis=1)
    
    # row 2:  I -> (M, I, D); (3, k_indel)
    mat_f = jnp.expand_dims( u*L/one_minus_L,               (0,1))
    mat_g = jnp.expand_dims( 1-(b+q*(1-M)/M)*L/one_minus_L, (0,1))
    mat_h = jnp.expand_dims( (b+q*(1-M)/M-u)*L/one_minus_L, (0,1))
    mat_Irow = jnp.concatenate([mat_f, mat_g, mat_h], axis=1)
    
    # row 3:  D -> (M, I, D); (3, k_indel)
    mat_p = jnp.expand_dims( (1-a-u)*M/one_minus_M,     (0,1))
    mat_q = jnp.expand_dims( q,                         (0,1))
    mat_r = jnp.expand_dims( 1-q-(1-a-u)*M/one_minus_M, (0,1))
    mat_Drow = jnp.concatenate([mat_p, mat_q, mat_r], axis=1)
    
    # concatenate all and output (3, 3, k_indel)
    out_matrix = jnp.concatenate([mat_Mrow, mat_Irow, mat_Drow], axis=0)
    return out_matrix
    

# get limiting transition matrix for large times
def largeTimeTransitionMatrix (t, indelParams):
    """
    matrix output should be:
        
    [[(1-g)*(1-r), g, (1-g)*r],
     [(1-g)*(1-r), g, (1-g)*r],
     [(1-r),       0, r]]
    """
    lam,mu,x,y = indelParams
    g = 1. - lm(t,lam,x) 
    r = 1. - lm(t,mu,y)  
    
    # row 1: M -> (M, I, D); (3, k_indel)
    mat_a = jnp.expand_dims( (1-g)*(1-r), (0, 1))
    mat_b = jnp.expand_dims( g,           (0, 1))
    mat_c = jnp.expand_dims( (1-g)*r,     (0, 1))
    mat_Mrow = jnp.concatenate([mat_a, mat_b, mat_c], axis=1)
    
    # row 2:  I -> (M, I, D); (3, k_indel)
    mat_f = jnp.expand_dims( (1-g)*(1-r), (0, 1))
    mat_g = jnp.expand_dims( g,           (0, 1))
    mat_h = jnp.expand_dims( (1-g)*r,     (0, 1))
    mat_Irow = jnp.concatenate([mat_f, mat_g, mat_h], axis=1)
    
    # row 3:  D -> (M, I, D); (3, k_indel)
    mat_p = jnp.expand_dims( (1-r),                   (0, 1))
    mat_q = jnp.zeros((1,1,lam.shape[0]))
    mat_r = jnp.expand_dims( r,                       (0, 1))
    mat_Drow = jnp.concatenate([mat_p, mat_q, mat_r], axis=1)
    
    # concatenate all and output (3, 3, k_indel)
    out_matrix = jnp.concatenate([mat_Mrow, mat_Irow, mat_Drow], axis=0)
    
    return out_matrix
    

def transitionMatrix (t, indelParams, /, alphabetSize=20, **kwargs):
    lam,mu,x,y = indelParams
    return jnp.where (t > 0.,
                      jnp.where (alignmentIsProbablyUndetectable(t,indelParams,alphabetSize),
                                 largeTimeTransitionMatrix(t,indelParams),
                                 smallTimeTransitionMatrix(t,indelParams,**kwargs)),
                      zeroTimeTransitionMatrix(indelParams))


##############################################
### FINDING A, B, U, Q                       #
### these are used in finding the small time #
### transition matrix functions              #
### (smallTimeTransitionMatrix)              #
##############################################
# calculate derivatives of (a,b,u,q)
def derivs (t, counts, indelParams):
    lam,mu,x,y = indelParams
    a,b,u,q = counts
    L = lm (t, lam, x)
    M = lm (t, mu, y)
    num = mu * (b*M + q*(1.-M))
    unsafe_denom = M*(1.-y) + L*q*y + L*M*(y*(1.+b-q)-1.)
    denom = jnp.where (unsafe_denom > 0., unsafe_denom, 1.)   # avoid NaN gradient at zero
    one_minus_m = jnp.where (M < 1., 1. - M, smallest_float32)   # avoid NaN gradient at zero
    return jnp.where (unsafe_denom > 0.,
                      jnp.array (((mu*b*u*L*M*(1.-y)/denom - (lam+mu)*a,
                                   -b*num*L/denom + lam*(1.-b),
                                   -u*num*L/denom + lam*a,
                                   ((M*(1.-L)-q*L*(1.-M))*num/denom - q*lam/(1.-y))/one_minus_m))),
                      jnp.array ((-lam-mu,lam,lam,0.)))

# calculate counts (a,b,u,q) by numerical integration
def initCounts(indelParams):
    return jnp.array ((1., 0., 0., 0.))
    
def integrateCounts (t, indelParams, /, step = None, rtol = None, atol = None, **kwargs):
    term = ODETerm(derivs)
    solver = Dopri5()
    if step is None and rtol is None and atol is None:
        raise Exception ("please specify step, rtol, or atol")
    if step is not None:
        stepsize_controller = ConstantStepSize()
    else:
        stepsize_controller = PIDController (rtol, atol)
    y0 = initCounts(indelParams)
    sol = diffeqsolve (term, solver, 0., t, step, y0, args=indelParams,
                       stepsize_controller=stepsize_controller,
                       **kwargs)
    return sol.ys[-1]




###########################
### BACKUP TO RUNGE-KUTTE #
###########################
# Runge-Kutte (RK4) numerical integration routine
# This is retained solely to have a simpler routine independent of diffrax, if needed for debugging
# Currently we just use integrateCounts instead, so this function is never called
def integrateCounts_RK4 (t, indelParams, /, steps=10, dt0=None):
  lam,mu,x,y = indelParams
  def RK4body (y, t_dt):
    t, dt = t_dt
    k1 = derivs(t, y, indelParams)
    k2 = derivs(t+dt/2, y + dt*k1/2, indelParams)
    k3 = derivs(t+dt/2, y + dt*k2/2, indelParams)
    k4 = derivs(t+dt, y + dt*k3, indelParams)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6, None
  y0 = initCounts (indelParams)
  if dt0 is None:
    dt0 = 0.1 / jnp.maximum (lam, mu)
  ts = jnp.geomspace (dt0, t, num=steps)
  ts_with_0 = jnp.concatenate ([jnp.array([0]), ts])
  dts = jnp.ediff1d (ts_with_0)
  y1, _ = jax.lax.scan (RK4body, y0, (ts_with_0[:-1],dts))
  return y1




####################################################################
### test these new versions, made for any number of mixture models #
####################################################################
if __name__=='__main__':
    t = 0.1
    alphabet_size = 20
    
    # k_indel = 5
    lam = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    mu = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    x = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    # k_indel 1
    # lam = jnp.array([0.1])
    # mu = jnp.array([0.1])
    # x = jnp.array([0.1])
    # y = jnp.array([0.1])
    
    indelParams = (lam, mu, x, y)
    
    diffrax_params = {"step": None,
                      "rtol": 1e-3,
                      "atol": 1e-6}
    
    
    out = transitionMatrix (t, 
                            indelParams,
                            alphabetSize=20, 
                            **diffrax_params)
        
        