# EvolPairHMM 2: Electric Boogaloo

## Main Packages Needed
- Jax/Flax/Diffrax
- Pytorch (for the dataloaders)
- Tensorboard


## CLI implementation
To fit a GGI model: `python train_ggi.py --config_file [CONFIG].json`  

Two ways to run this (triggered by `have_precalculated_counts`):
  1. Need to calculate counts, then train a model (`have_precalculated_counts = FALSE`):  
    - see `DEV-DATA_pair_alignments` for example of inputs  
    - `train_ggi.py` will import functions found in `calcCounts_Train` (and `GGI_funcs`, `utils`)

  2. Only train  (`have_precalculated_counts = TRUE`):  
    - see `DEV-DATA_precomputed_counts` for example of inputs  
    - `train_ggi.py` will import functions found in `onlyTrain` (and `GGI_funcs`, `utils`)


## Config File Fields
#### workspace setup
- `training_wkdir:` Folder containing all results (potentially from multiple runs)
- `runname:` Folder containing this run's results
- `rng_seednum:` RNG seed for run

#### training/testing data
- `have_precalculated_counts:` **[BOOL]** have you precalculated emission and transition counts for your dataset?
- `loadtype:`  **["eager", "lazy"]** how to load data; eager is faster and way better
- `data_dir:` Folder containing the data
- `train_dset_splits:` file prefix for splits that should be in TRAINING dataset
- `test_dset_splits:` file prefix for splits that should be in TEST dataset

#### general training setup
- `batch_size:` batch size for minibatch gradient descent (and evaluation)
- `num_epochs:` number of epochs to train
- `learning_rate:` ADAM learning rate
- `patience:` for early stopping; how many times will the early stopping condition be triggered before you stop training?

#### GGI specific parameters
- `model_type:` Not sure if this will be used, but set it to "GGI_simple" for now  
- `t_grid_center:` center for geometrically-distributed time (see https://github.com/songlab-cal/CherryML)
- `t_grid_step:` step for geometrically-distributed time
- `t_grid_num_steps:` how many steps for geometrically-distributed time
- `norm:` normalize the substitution rate matrix such that expected substitution rate at equilibrium is 1
- `lam:` rate of insertions
- `mu:` rate of deletions
- `x:` insertion length parameter
- `y:` deletion length parameter
- `diffrax_params:` parameters passed to diffrax, for the Runge–Kutta solver
  - `step:` step size (I think?)
  - `rtol:` Relative Tolerance
  - `atol:` Absolute Tolerance


## To do
- Implement mixture model
