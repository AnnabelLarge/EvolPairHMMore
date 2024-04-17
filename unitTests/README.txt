output on 4/16/24:
==================
TESTING counting in summarize_alignment
[PASSED] correct counts of alignment events

TESTING substitution models in protein_subst_models
[PASSED] correct calculation: single rate matrix, single equilibrium
[PASSED] correct calculation: single rate matrix, mixtures of equilibrium
[PASSED] correct calculation: mixtures of rate matrices, single equilibrium

TESTING single and mixture equilibrium distribution blocks in equl_distr_models
[PASSED] correct equilibrium distributions returned in all cases

TESTING batched GGI code from indel_models
[PASSED] correct calculation: transition matrix from single H20 model
[PASSED] correct calculation: transition matrices for mixture of H20 models

TESTING likelihood calculation in utils.training_testing_fns
Creating DataLoader for test set with ['FiveSamp']
[PASSED] correct likelihood calculation for all single models
[PASSED] correct recipe for adding mixture weights and logsumexp-ing