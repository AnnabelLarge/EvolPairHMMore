#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:56:31 2024

@author: annabel


ABOUT:
======
One script to run all unit tests


testing counting in summarize_alignment
---------------------------------------
1. counting insertions, deletions, match types, and transitions


testing substitution models in protein_subst_models
----------------------------------------------------
1. calculating substitution rate matrix: one substitution model, one 
   equilibrium distribution
   
2. calculating substitution rate matrix: one substitution model, mixture of 
   candidate equilibrium distributions
   
3. calculating substitution rate matrix: mixture of substitution models, one 
   equilibrium distribution


testing equilibrium distribution blocks in equl_distr_models
-------------------------------------------------------------
1. calculating equilibrium distribution and log(equilibrium distribution) for
   single and mixtures of equilibriums (basically testing all blocks)


testing batched GGI code from indel_models
------------------------------------------
1. calculation of transition matrix for single H20 model

2. calculation of transition matrix for mixture of H20 models


testing likelihood calculation in utils.training_testing_fns
-------------------------------------------------------------
1. full likelihood calculation with all single models
2. full likelihood calculation with no indel model (only score match sites)
3. recipe for multiplying mixture weights and summing


"""
#############################################
### testing counting in summarize_alignment #
#############################################
from unitTests.unitTest_summarize_alignment import main as counts_test
print('TESTING counting in summarize_alignment')

counts_test()
print('[PASSED] correct counts of alignment events')

print()



#########################################################
### testing substitution models in protein_subst_models #
#########################################################
from unitTests.unitTest_singleSubst_singleEqul import main as subModel_test1
from unitTests.unitTest_singleSubst_multipleEqul import main as subModel_test2
from unitTests.unitTest_multipleSubst_singleEqul import main as subModel_test3
print('TESTING substitution models in protein_subst_models')
      
subModel_test1()
print('[PASSED] correct calculation: single rate matrix, single equilibrium')

subModel_test2()
print('[PASSED] correct calculation: single rate matrix, mixtures of equilibrium')

subModel_test3()
print('[PASSED] correct calculation: mixtures of rate matrices, single equilibrium')

print()



##################################################################
### testing equilibrium distribution blocks in equl_distr_models #
##################################################################
from unitTests.unitTest_equlDistModels import main as equlModel_test
print('TESTING single and mixture equilibrium distribution blocks in equl_distr_models')

equlModel_test()
print('[PASSED] correct equilibrium distributions returned in all cases')

print()



################################################
### testing batched GGI code from indel_models #
################################################
from unitTests.unitTest_singleGGI_batchedTransitionMatrix import main as indelModel_test1
from unitTests.unitTest_multipleGGI_batchedTransitionMatrix import main as indelModel_test2
print('TESTING batched GGI code from indel_models')

indelModel_test1()
print('[PASSED] correct calculation: transition matrix from single H20 model')

indelModel_test2()
print('[PASSED] correct calculation: transition matrices for mixture of H20 models')

print()



##################################################################
### testing likelihood calculation in utils.training_testing_fns #
##################################################################
from unitTests.unitTest_calcLoglike_singleModels import main as trainingFn_test1
from unitTests.unitTest_subst_only_model import main as trainingFN_test2
from unitTests.unitTest_logSum_mixWeights import main as trainingFn_test3
print('TESTING likelihood calculation in utils.training_testing_fns')

trainingFn_test1()
print('[PASSED] correct likelihood calculation for all single models')

trainingFn_test2()
print('[PASSED] correct likelihood calculation for only scoring substitution sites')

trainingFn_test3()
print('[PASSED] correct recipe for adding mixture weights and logsumexp-ing')

print()

