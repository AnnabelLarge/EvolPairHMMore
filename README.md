# EvolPairHMM 2: Electric Boogaloo

## Main Packages Needed
- Jax + friends (Flax, Optax, Orbax, Diffrax)
- Pytorch (for the dataloaders)
- Tensorboard


## CLI interface
Access the main CLI interface with:
```
python EvolPairHMMore.py -task [TASK] -configs [CONFIG FILE/FOLDER]
```

Tasks include:
- `train_hmm`: train a pairHMM model
- `eval_hmm`: load parameters for previously-trained pairHMM, evaluate likelihoods of new (aligned) data
- `eval_hmm_subs_only`: evaluate likelihoods of aligned data based ONLY on substitution positions; requires exchangeabilities to load from
- `*_batched`: do one of the above commands, using all configs from a folder; all runs will share one dataloader
  - options include: `train_hmm_batched`, `eval_hmm_batched`, `eval_hmm_subs_only_batched`  

For regular tasks, provide a single JSON file for `-configs` flag.  

For batched versions, provide a folder name for `-configs` flag. Code will automatically detect any file in the folder ending in `.json`. Dataloaders will be built from first config file detected.


## Config fields for TRAINING (`train_hmm`)
Templates for config files can be automatically generated using `./generate_configs/generate_training_config.py`  

### general setup
- `training_wkdir [str]`: name of the training directory that will store outputs
- `rng_seednum [int]`: Jax rng seed
- `debug [bool]`: output some intermediates during training

### dataset
- `alphabet_size [int]`: 20 for proteins, 4 for DNA
- `have_precalculated_counts [bool]`: if you've precalculated transition/emission counts or not
- `data_dir [str]`: directory containing all the data
- `train_dset_splits [list of str]`: prefixes for training splits
- `test_dset_splits [list of str]`: prefixes for test/validation splits
- `batch_size [int]`: number of samples per batch

### training
- `num_epochs [int]`: number of epochs 
- `learning_rate [float]`: learning rate for training
- `patience [int]`: patience for early stopping
- `early_stop_rtol [float]`: rtol for early stopping
- `loss_type [str=(conditional, joint)]`: either train with "conditional" loss or "joint" loss (i.e. do deleted ancestor characters contribute to loss or not?)
- `norm_loss_by [str=(desc_len, align_len)]`: either normalize loss by length of (unaligned) descendant, "desc_len", or by the length of the alignment as a whole, "align_len"

### Geometrically-spaced time
Using code from [CherryML](https://github.com/songlab-cal/CherryML) (Prillo et. al. 2022)  
- `t_grid_center [float]`: center for geometric grid
- `t_grid_step [float]`: step for geometric grid
- `t_grid_num_steps [int]`: number of grid points

### model specific things
Each model uses specific config options; check `./model_blocks/README.txt`
- `subst_model_type [str]`: what kind of substitution model to use
- `equl_model_type [str]`: what kind of equilibrium distribution to use
- `indel_model_type [str]`: what indel model to use


## Config fields for EVALUATION (`eval_hmm`)
### general setup
- `eval_wkdir [str]`: name of the eval directory that will store outputs
- `rng_seednum [int]`: Jax rng seed, if a model ever needs that
- `debug [bool]`: output some intermediates during evaluation
- `training_wkdir [str]`: training working directory to find model parameters and original training configuration

### dataset
- `have_precalculated_counts [bool]`: if you've precalculated transtition/emission counts or not
- `data_dir [str]`: directory containing all the data
- `test_dset_splits [str]`: prefixes for test/validation splits
- `batch_size [int]`: batch size; max it out for your GPU/CPU (there's no training involved)

## Config fields for EVALUATING WITH SUBSTITUTION MODEL (`eval_hmm_subs_only`)
### general setup
- `training_wkdir [str]`: name of the training directory that will store outputs
- `rng_seednum [int]`: Jax rng seed, if needed (probably not)
- `debug [bool]`: output some intermediates during training

### dataset
- `alphabet_size [int]`: 20 for proteins, 4 for DNA
- `have_precalculated_counts [bool]`: if you've precalculated transition/emission counts or not
- `data_dir [str]`: directory containing all the data
- `train_dset_splits [list of str]`: prefixes for training splits
- `test_dset_splits [list of str]`: prefixes for test/validation splits
- `batch_size [int]`: number of samples per batch

### loss to evaluate
- `loss_type [str=(conditional, joint)]`: either train with "conditional" loss or "joint" loss (i.e. do deleted ancestor characters contribute to loss or not?)
- `norm_loss_by [str=(desc_len, align_len)]`: either normalize loss by length of (unaligned) descendant, "desc_len", or by the length of the alignment as a whole, "align_len"

### Geometrically-spaced time
Using code from [CherryML](https://github.com/songlab-cal/CherryML) (Prillo et. al. 2022)  
- `t_grid_center [float]`: center for geometric grid
- `t_grid_step [float]`: step for geometric grid
- `t_grid_num_steps [int]`: number of grid points

A couple options for substitution model, depending on which is used. See `./model_blocks/README.txt`


## Input data Format
Use [Pfam-pair-processing](`https://github.com/AnnabelLarge/Pfam-pair-processing`) to create training data. This repo can use tokenized alignments OR matrices of precomputed transition/emission counts.

A folder of tokenized alignments would have:
- `{prefix}_pair_alignments.npy`: tokenized alignments
- `{prefix}_AAcounts.npy`: counts of amino acids in dataset (used for equilibrium distribution)
- `{prefix}_AAcounts_subsOnly.npy`: counts of amino acids ONLY at substitution sites
- `{prefix}_metadata.tsv`: metadata about alignments

A folder of precomputed counts would have:
- `{prefix}_subCounts.npy`: counts of (anc -> desc) emissions at MATCH sites
- `{prefix}_insCounts.npy`: counts of emissions at INSERT sites
- `{prefix}_delCounts.npy`: counts of missing data from DELETE sites (only relevant for joint loss function)
- `{prefix}_transCounts.npy`: counts of transitions, padded with starting MATCH and ending MATCH state
- `{prefix}_AAcounts.npy`: counts of amino acids in dataset (used for equilibrium distribution)
- `{prefix}_AAcounts_subsOnly.npy`: counts of amino acids ONLY at substitution sites
- `{prefix}_metadata.tsv`: metadata about alignments

Possible emissions are indexed in alphabetical order, according to one-letter abbreviation. For a 20-element equilibrium distribution vector over amino acids:
- i=0: Alanine
- i=1: Cysteine
- i=2: Aspartic Acid (Aspartate)
- and so on

I understand this ordering is controversial. Oh well, it's too late to go back.  

Possible transitions are indexed in this order: Match, Insert, Delete.

## Add a new model

### A note about einsum notation
#### Abbreviations used in code
- time: `t ` 
- batch: `b`  
- alphabet_size_from:  `i`  
- alphabet_size_to: `j`  
- number_transitions_from: `m`  
- number_transitions_to: `n`  
- number_substitution_mixtures: `x ` 
- number_equilibrium_mixtures: `y`  
- number_indel_mixtures: `z`  

`i == j` and `m == n`, but distinguishing between the two allows for distinguishing between row/column operations in einsum.

#### universal order of dimensions:
All matrices in the code will adhere to the following order of axes:
1. time
2. batch
3. length
4. *(model-specific indices, like `(i,j)` for substitution models)*
5. number_substitution_mixtures
6. number_equilibrium_mixtures
7. number_indel_mixtures

Matrices won't necessarily have all of the above. This is just a general guide on how to order dimensions.  

For example: for a mixture model that uses 2 substitution models, 3 equilibrium distributions, and 1 indel model, the probability of all alignments at all timepoints will be a matrix of size: `(t, b, 2, 3, 1)`


### substitution model
Models can subclass `subst_base` to get functions to 1.) register classes as pytrees (allows jit compilation) and 2.) calculate/normalize rate matrix. Any model added here will actually be generating the *exchangeabilities*.  

At a minimum, future classes need the following methods (unless otherwise marked "optional"):  

#### `initialize_params(self, argparse_obj)`
  - Purpose: initialize all parameters and hyperparameters
  - Inputs
    - argparse_obj: the higher-level config from argparse
  - Returns
    - initialized_params: dictionary of named parameters; will be updated by optax through training
    - hparams: dictionary of hyperparameters; remain static through training

#### `conditional_logprobs_at_t(self, t, params_dict, hparams_dict)`
  - Purpose: Calculate conditional logprobability matrix of emissions at match sites 
  - Inputs
    - t: which timepoint to calculate for
    - params_dict: dictionary of named parameters
    - hparams_dict: dictionary of hyperparameters
  - Returns
    - cond_logprob_substitution_at_t: conditional log-probability; shape is `(i, j, x, y)`

#### (OPTIONAL) `joint_logprobs_at_t(self, t, params_dict, hparams_dict)`
  - Purpose: Calculate joint log-probability matrix of emissions at match sites 
  - Inputs
    - t: which timepoint to calculate for
    - params_dict: dictionary of named parameters
    - hparams_dict: dictionary of hyperparameters
  - Returns
    - cond_logprob_substitution_at_t: conditional log-probability; shape is `(i, j, x, y)`
  - it's pretty rare to need the joint log probability; could also raise a `NotImplementedError` here and call it a day

#### `undo_param_transform(self, params_dict)`
  - Purpose: parameter dictionary will contain model parameters after some transformation (e.g. logits). This function undoes the transformation. This is useful for writing the values to tensorboard, logfiles, etc.
  - Inputs
    - params_dict: dictionary of named parameters (see above)
  - Returns
    - out_dict: dictinoary of parameters after undoing transformation; could have different names

### equilibrium distribution model
Models can subclass `equl_base` to get functions to register classes as pytrees (allows jit compilation).  

At a minimum, future classes need the following methods:  

#### `equlVec_logprobs(self, params_dict, hparams_dict)`
  - Purpose: calculate the equlibrium distribution(s); later, multiply by observed emission counts to calculate logP(emissions at Insert) and/or logP(missing chars at Delete). Note that this vector won't depend on time!
  - Inputs
    - params_dict: dictionary of named parameters
    - hparams_dict: dictionary of hyperparameters
  - Returns
    - equl_vec: equilibrium distribution(s); shape is `(i, y)`
    - logprob_equl: log of the equilibrium distribution(s); shape is `(i, y)`

#### `initialize_params(self, argparse_obj)` (see above)

#### `undo_param_transform(self, params_dict)` (see above)

### indel model
Models can subclass `no_indel` to get functions to 1.) register classes as pytrees (allows jit compilation)

At a minimum, future classes need the following methods:  

#### `logprobs_at_t(self, t, params_dict, hparams_dict)`
  - Purpose: Calculate log-probability matrix of transitions through pairHMM at one time `t`
    - t: which timepoint to calculate for
    - params_dict: dictionary of named parameters
    - hparams_dict: dictionary of hyperparameters
  - Returns
    - logprob_transition_at_t: log-probability of transitions; shape is `(m, n, z)`

#### `initialize_params(self, argparse_obj)` (see above)

#### `undo_param_transform(self, params_dict)` (see above)

### After creating model class
Need to add imports/initializations to `utils.setup_utils.model_import_register()`

## Unit tests/Baselines
### Basic unit tests
Run a basic suite with:
```
python ./unitTest/RUN_unitTests.py
```
These broadly cover things like:
  - do einsum recipes match hand-done loops?
  - does my implementation of the H20 model match Ian's implementation?
  - is loss being calculated as I expect?

### Conditional/Descendant Entropy Scoring
This work can be found in `./unitTest/check_entropic_counts`. It's kind of a baseline and a unit test in one.  

Steps:  
  1. Move this folder to top-level
  2. Run `hand_calculate_entropy_scores.py` to hand-calculate losses according to either:
     - frequencies of (descendant characters, given aligned ancestor characters); likelihood will match the conditional logprob P(desc | anc, align)
     - frequencies of descendant characters; likelihood will match the marginal logprob P(desc)
     - frequencies of descendant characters ONLY at match sites (i.e. ignore indels)
  3. Do a fake training run using `EvolPairHMMore.py` and the config files in this folder
  4. Manually verify that likelihoods from fake training runs match the hand-calculated likelihoods
  5. Compare any future results to those found here

## Repo organization
Folders
- `calcCounts_train`: use these functions when tokenized alignments are given
  - dataloader
  - function to summarize counts on the fly
- `cli`: code to run tasks
- `examples`: examples of inputs, configs, and training results
- `exchangeability_matrices`: LG08 exchangeability matrices
- `generate_configs`: functions to generate config files
- `model_blocks`: classes for different model parts
- `not_used`: internal scripts I use during code development
- `onlyTrain`: use these functions when precomputed counts are given
  - dataloader
- `unitTests`: folder with suite of unit tests for this codebase
  - note that these tests are specifically configured for the example data in this directory; would need to change configs in this folder if you wanted to try out unit tests on NEW data
- `utils`: scripts to help with training and evaluation

Top-level files
- `README.md`: you're looking at her, baby!
- `EvolPairHMMore.py`: main CLI launcher
