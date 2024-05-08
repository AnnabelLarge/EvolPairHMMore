
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 19:13:58 2024

@author: annabel

Additional functions from Ian
"""
import jax.numpy as jnp

# All functions return 3x3 transition matrices of the form
# [[tMM,tMI,tMD],[tIM,tII,tID],[tDM,tDI,tDD]]
# where tXY is the transition probability from state X->Y,
# so that  sum_j tij = 1  for any i.

# # Helper functions for TKF91 and TKF92
# def TKF_alpha_beta (lam, mu, t):
#   alpha = jnp.exp(-mu*t)
#   beta = (lam*(jnp.exp(-lam*t)-jnp.exp(-mu*t))) / (mu*jnp.exp(-lam*t)-lam*jnp.exp(-mu*t))
  
#   return alpha, beta

# TKF91
# Thorne, Kishino, and Felsenstein, 1991
# An evolutionary model for maximum likelihood alignment of DNA sequences.
# J. Mol. Evol.  33: 114–124.
def TKF91_Ftransitions (alpha, beta, lam, mu, t):
  # alpha, beta = TKF_alpha_beta (lam, mu, t)

  ### this approximation from Ian behaves weirdly for small sample sizes
  # gamma = lam / mu

  gamma =  1 - ( (mu * beta) / ( lam * (1-alpha) ) )

  
  return jnp.array ([[(1-beta)*alpha, beta, (1-beta)*(1-alpha)],
                     [(1-beta)*alpha, beta, (1-beta)*(1-alpha)],
                     [(1-gamma)*alpha, gamma, (1-gamma)*(1-alpha)]])

# TKF92
# Thorne, Kishino, and Felsenstein, 1992
# Inching toward reality: an improved likelihood model of sequence evolution.
# J. Mol. Evol.  34: 3–16.
def TKF92_Ftransitions (alpha, beta, lam, mu, x, y, t):
  """
  domain problem if mu <= lam
  """
  r = (x + y) / 2

  ### this approximation from Ian behaves weirdly for small sample sizes
  # gamma = lam / mu

  gamma =  1 - ( (mu * beta) / ( lam * (1-alpha) ) )
  
  return jnp.array ([[r + (1-r)*(1-beta)*alpha, (1-r)*beta, (1-r)*(1-beta)*(1-alpha)],
        		         [(1-r)*(1-beta)*alpha, r + (1-r)*beta, (1-r)*(1-beta)*(1-alpha)],
		                 [(1-r)*(1-gamma)*alpha, (1-r)*gamma, r + (1-r)*(1-gamma)*(1-alpha)]])

# LG05
# Löytynoja and Goldman, 2005
# An algorithm for progressive multiple alignment of sequences with insertions.
# Proc. Natl. Acad. Sci. USA  102: 10557–10562.
def LG05_Ftransitions (lam, mu, x, y, t):
  """
  domain problem if x=y=1, but that's really unlikely
  """
  epsilon = (x + y) / 2
  gamma = epsilon
  maxDelta = .49999
  # delta = jnp.minimum (maxDelta, 1 - jnp.exp(-(lam + mu)*t/(2*(1-gamma))))
  
  # epsilon cannot be zero, or this denom is undefined
  exponentiated =  -(lam + mu)*t/(2*(1-gamma))
  delta = jnp.minimum (maxDelta, 
                       1 - jnp.exp(exponentiated) 
                       )
  
  return jnp.array ([[gamma + (1-gamma)*(1-2*delta), (1-gamma)*delta, (1-gamma)*delta],
                     [(1-epsilon)*(1-2*delta), epsilon + (1-epsilon)*delta, (1-epsilon)*delta],
        		         [(1-epsilon)*(1-2*delta), epsilon + (1-epsilon)*delta, (1-epsilon)*delta]])

# RS07
# Redelings and Suchard, 2007
# Incorporating indel information into phylogeny estimation for rapidly emerging pathogens.
# BMC Evol. Biol.  7: 40.
def RS07_Ftransitions (lam, mu, x, y, t):
  """
  domain problem if x=y=1, but that's really unlikely
  """
  epsilon = (x + y) / 2
  maxDelta = .49999
  # delta = jnp.minimum (maxDelta, 1 / (1 + 1 / (1 - jnp.exp(-(lam + mu)*t/(2*(1-epsilon))))))
  
  # epsilon cannot be zero, or this denom is undefined
  exponentiated = -(lam + mu)*t/(2*(1-epsilon))
  delta = jnp.minimum (maxDelta, 
                       1 / (1 + 1 / (1 - jnp.exp( exponentiated )))
                       )
  
  return jnp.array ([[epsilon + (1-epsilon)*(1-2*delta), (1-epsilon)*delta, (1-epsilon)*delta],
		                 [(1-epsilon)*(1-2*delta), epsilon + (1-epsilon)*delta, (1-epsilon)*delta],
		                 [(1-epsilon)*(1-2*delta), epsilon + (1-epsilon)*delta, (1-epsilon)*delta]])

# KM03
# Knudsen and Miyamoto, 2003
# Sequence Alignments and Pair Hidden Markov Models Using Evolutionary History
# J. Mol. Biol. 333:2, 453-460.
def KM03_Ftransitions (lam, mu, x, y, t):
  """
  domain problem if x=y=1, but that's really unlikely; probably don't have to
    worry about that
  """
  r = (lam + mu) / 2
  a = (x + y) / 2
  
  Pid = 1 - jnp.exp(-2*r*t)
  Pid_prime = 1 - (1 - jnp.exp(-2*r*t)) / (2*r*t)
  
  T00 = 1 - Pid*(1-Pid_prime*(1-a)/(4+4*a))
  T01 = (1-T00)/2
  T02 = T01
  
  E10 = 1-a + Pid_prime*a*(1-a)/(2+2*a) - Pid*(7-7*a)/8
  E11 = a + Pid_prime*a*a/(1-a*a) + Pid*(1-a)/2 # domain problem if a*a=1
  E12 = Pid_prime*a*a/(2+2*a) + Pid*(3-3*a)/8
  E1 = 1 + Pid_prime*a/(2-2*a) # domain problem if a = 1
  
  T10 = E10/E1
  T11 = E11/E1
  T12 = E12/E1
  T20 = T10
  T22 = T11
  T21 = T12
  return jnp.array ([[T00, T01, T02],
                     [T10, T11, T12],
                     [T20, T21, T22]])



if __name__ == '__main__':
    ### All are good for either single or mixture models! :)
    def try_all(lam, mu, x, y, t):
        tkf91 = TKF91_Ftransitions (lam, mu, t)
        tkf92 = TKF92_Ftransitions (lam, mu, x, y, t)
        lg05 = LG05_Ftransitions (lam, mu, x, y, t)
        rs07 = RS07_Ftransitions (lam, mu, x, y, t)
        km03 = KM03_Ftransitions (lam, mu, x, y, t)
        
        for mat in [tkf91, tkf92, lg05,rs07, km03]:
            assert mat.shape == (3,3,lam.shape[-1])
        
        out_dict = {'TKF91': tkf91,
                    'TKF92': tkf92,
                    'LG05': lg05,
                    'RS07': rs07,
                    'KM03': km03}
        
        return out_dict
    
    ### test out the single parameter version of each
    out_mats1 = try_all(lam = jnp.array([0.2]),
                       mu = jnp.array([0.3]),
                       x = jnp.array([0.6]),
                       y = jnp.array([0.6]),
                       t = 0.05)
    
    out_mats2 = try_all(lam = jnp.array([0.5]),
                       mu = jnp.array([0.6]),
                       x = jnp.array([0.5]),
                       y = jnp.array([0.5]),
                       t = 0.05)
    
    
    
    ### see if you can extend to mixture models
    out_mats_mix = try_all(lam = jnp.array([0.2, 0.5]),
                           mu = jnp.array([0.3, 0.6]),
                           x = jnp.array([0.6, 0.5]),
                           y = jnp.array([0.6, 0.5]),
                           t = 0.05)
    
    for mod in ['TKF91', 'TKF92','LG05','RS07', 'KM03']:
        single_mat1 = out_mats1[mod]
        single_mat2 = out_mats2[mod]
        
        mixture_mat = out_mats_mix[mod]
        
        assert jnp.allclose(single_mat1[:,:,0], mixture_mat[:,:,0])
        assert jnp.allclose(single_mat2[:,:,0], mixture_mat[:,:,1])
    
    
