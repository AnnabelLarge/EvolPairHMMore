# EvolPairHMM 2: Electric Boogaloo

## Main Packages Needed
- Jax/Flax/Diffrax
- Pytorch (for the dataloaders)
- Tensorboard


## CLI implementation
To fit a GGI model: `python train_pairhmm.py --config_file [CONFIG].json`  

Two ways to run this (triggered by `have_precalculated_counts`):
  1. Need to calculate counts, then train a model (`have_precalculated_counts = FALSE`):  
    - see `DEV-DATA_pair_alignments` for example of inputs  
    - `train_ggi.py` will import functions found in `calcCounts_Train` (and `GGI_funcs`, `utils`)

  2. Only train  (`have_precalculated_counts = TRUE`):  
    - unpack `DEV-DATA_precomputed_counts.tar.gz` for example of inputs  
    - `train_ggi.py` will import functions found in `onlyTrain` (and `GGI_funcs`, `utils`)


## Config File Fields
NEED TO UPDATE THIS README!!!

## TODO
- make a script that loads model parameters and evaluates likelihood of selected dataset
- update this readme file
