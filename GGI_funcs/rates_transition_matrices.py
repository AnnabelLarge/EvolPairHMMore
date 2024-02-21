#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:27:33 2024

@author: annabel

Functions to calculate rate matrices, transitions matrices, etc.
"""
import jax
from jax import numpy as jnp
import diffrax
from diffrax import (diffeqsolve, ODETerm, Dopri5, PIDController, 
                     ConstantStepSize, SaveAt)


# We replace zeroes and infinities with small numbers sometimes
# It's sinful but BOY DO I LOVE SIN
smallest_float32 = jnp.finfo('float32').smallest_normal


#############################
### Calculate rate matrices #
#############################
def lg_rate_mat(equl_pi_vec, file_path):
    """
    Loads the LG08 exchangeability parameters, and uses them to calculate 
    the 20x20 substitution rate matrix
    
    Place this file in the data folder, defined by "file_path"
    """
    ### load the preprocessed exchangeability paramters
    with open(file_path,'rb') as f:
        exch_r_mat = jnp.load(f)
    
    ### fill in values for i != j
    # einsum should be doing-
    #   exch_ij = exch_r_mat[i,j]
    #   pi_i = equl_pi_vec[i]
    #   raw_rate_mat_ij = exch_ij * pi_i
    raw_rate_mat = jnp.einsum('ij, i -> ij', exch_r_mat, equl_pi_vec)
    
    ### now mask out diagonals with zeros
    mask = jnp.abs(1 - jnp.eye(equl_pi_vec.shape[0]))
    rate_mat_without_diags = raw_rate_mat * mask
    
    ### fill in values for i == j such that rows sum to zero
    row_sums_without_diag = jnp.expand_dims(rate_mat_without_diags.sum(axis=1), 1)
    rows_sums_copied = jnp.repeat(-row_sums_without_diag, equl_pi_vec.shape[0], axis=1)
    mask = jnp.eye(equl_pi_vec.shape[0])
    diags_to_add = mask * rows_sums_copied
    
    
    ### add them together for final rate matrix
    subst_rate_mat = rate_mat_without_diags + diags_to_add
    
    return subst_rate_mat



#####################################
### TRANSITION MATRIX FUNCTIONS     #
#####################################
### (all this is from Ian directly)
### calculate L, M
def lm (t, rate, prob):
    return jnp.exp (-rate * t / (1. - prob))

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
    lam,mu,x,y = indelParams
    return jnp.array ([[1.,0.,0.],
                       [1.-x,x,0.],
                       [1.-y,0.,y]])

# convert counts (a,b,u,q) to transition matrix ((a,b,c),(f,g,h),(p,q,r))
def smallTimeTransitionMatrix (t, indelParams, /, **kwargs):
    lam,mu,x,y = indelParams
    a,b,u,q = integrateCounts(t,indelParams,**kwargs)
    # To use the non-diffrax version, comment out the previous line and uncomment the following one:
    # a,b,u,q = integrateCounts_RK4(t,indelParams,dt0=.1/jnp.maximum(lam,mu))
    L = lm(t,lam,x)
    M = lm(t,mu,y)
    one_minus_L = jnp.where (L < 1., 1. - L, smallest_float32)   # avoid NaN gradient at zero
    one_minus_M = jnp.where (M < 1., 1. - M, smallest_float32)   # avoid NaN gradient at zero
    return jnp.array ([[a,  b,  1-a-b],
                      [u*L/one_minus_L,  1-(b+q*(1-M)/M)*L/one_minus_L,  (b+q*(1-M)/M-u)*L/one_minus_L],
                       [(1-a-u)*M/one_minus_M,  q,  1-q-(1-a-u)*M/one_minus_M]])

# get limiting transition matrix for large times
def largeTimeTransitionMatrix (t, indelParams):
    lam,mu,x,y = indelParams
    g = 1. - lm(t,lam,x) 
    r = 1. - lm(t,mu,y)  
    return jnp.array ([[(1-g)*(1-r),g,(1-g)*r],
                       [(1-g)*(1-r),g,(1-g)*r],
                       [(1-r),0,r]])

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