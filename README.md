# EvolPairHMM 2: Electric Boogaloo

## Main Packages Needed
- Jax + friends (Flax, Optax, Orbax, Diffrax)
- Pytorch (for the dataloaders)
- Tensorboard


## Repo contents
- **calcCounts_Train**: dataloader and counting function, for calculating transition/emission counts on the fly
- **example_runs**: example data, along with configs and outputs for all CLI implementations
- **exchangeability_matrices**: The exchangeability matrices used for substitutions models; all from [ATGC Montpellier](http://www.atgc-montpellier.fr/models/).
- **model_blocks**: Blocks for indel models, substitution models, and equilibrium distributions.
- **onlyTrain**: dataloader for reading from precomputed transition/emissions counts
- **RUN_FUNCTIONS**: Main CLI interfaces to run HMM training or eval
- **unitTests**: suite of unit tests for everything here
- **utils**: misc functions

## CLI implementations
To run, move the desired function from `RUN_FUNCTIONS` folder to top-level directory. Also make sure data folder in top-level directory. Syntax for all functions is generally `CUDA_VISIBLE_DEVICES=[n], python [RUN_script.py] [input_flag] [config_location]`. If the input flag is `--config-file`, then provide a JSON. If the input flag is `--config-folder`, then provide a folder name; all JSON files will be read from this folder.

The training functions are:
|                              | saves model params? | input flag        | descr                                                                          |
|------------------------------|---------------------|-------------------|--------------------------------------------------------------------------------|
| RUN_train_pairhmm.py         | YES                 | `--config-file`   | train one HMM model from a single config                                       |
| RUN_batched_train_pairhmm.py | YES                 | `--config-folder` | train many HMM models, from configs in a folder (all must use same dataset)                                |
| RUN_hparam_sweep.py          | no                  | `--config-folder` | run a hyperparameter sweep for HMM models, from configs in a folder (all must use same dataset)            |
| RUN_fit_predict_mixtures.py  | YES                 | `--config-file`   | train one mixture HMM model and predict group membership, from a single config |

The eval functions are:
|                      | input flag      | descr                                                                                   |
|----------------------|-----------------|-----------------------------------------------------------------------------------------|
| RUN_eval_pairhmm.py  | `--config-file` | load from a previously-saved model, and eval on a new dataset                           |
| RUN_eval_subsOnly.py | `--config-file` | evaluate log-likelihoods using one substitution model (with exchangeabilities provided) |


## Config File Fields: training functions 
All training functions will use some variety of these inputs; see `example_runs` for examples of configs.
### general setup
- **training_wkdir** [str]: the larger training directory
- **runname** [str]: the run-specific name; multiple runs can reside in the same training directory
- **rng_seednum** [int]: Jax rng seed

### dataset
- **alphabet_size** [int]: 20 for proteins, 4 for DNA (but I don't have the exchangeabilities or the code for DNA models here!)
- **have_precalculated_counts** [bool]: if you've precalculated transtition/emission counts or not
- **loadtype** [str]: set this to "eager"
- **data_dir** [str]: directory containing all the data
- **train_dset_splits** [str]: prefixes for training splits
- **test_dset_splits** [str]: prefixes for test/validation splits

### training
- **batch_size** [int]: batch size
- **num_epochs** [int]: number of epochs for training
- **learning_rate** [float]: learning rate
- **patience** [int]: patience for early stopping
- **early_stop_rtol** [float]: rtol for early stopping
- **loss_type** [str]: either train with "conditional" loss or "joint" loss

### quantization of time
I use a geometrically spaced time grid, inspired by [CherryML](https://github.com/songlab-cal/CherryML) from Prillo et. al. 2022  
- **t_grid_center** [float]: center for geometric grid
- **t_grid_step** [float]: step for geometric grid
- **t_grid_num_steps** [int]: number of grid points

### model specific things
Requirements vary with model type; see comments in `model_blocks` for more details
- **subst_model_type** [str]: what kind of substitution model to use
- **equl_model_type** [str]: what kind of equilibrium distribution to use
- **indel_model_type** [str]: what indel model to use

## Config File Fields: eval functions 
Both eval functions will use these inputs, or some subset of these inputs; see `example_runs` for examples of configs.
### general setup
- **eval_wkdir** [str]: new eval working directory
- **eval_runname** [str]: new eval run name; multiple runs can exist in the same eval directory
- **training_wkdir** [str]: training working directory to find model parameters from
- **training_runname** [str]: training runname to find model parameters from
- **rng_seednum** [int]: Jax rng seed (but as far as I know, isn't used...?)

### dataset
- **have_precalculated_counts** [bool]: if you've precalculated transtition/emission counts or not
- **loadtype** [str]: set this to "eager"
- **data_dir** [str]: directory containing all the data
- **test_dset_splits** [str]: prefixes for test/validation splits

### misc
- **batch_size** [int]: batch size; max it out for your GPU/CPU (there's no training involved)
- **loss_type** [str]: either train with "conditional" loss or "joint" loss


## TODO
### medium priority
- For now, using LG08 exchangeability matrix, but in the future, could use [CherryML](https://github.com/songlab-cal/CherryML) to calculate a new exchangeability matrix for my specific pfam dataset?

### low priority
- combine all training functions under one general, flexible run function
- combine all eval functions under one general, flexible run function
- remove the option to calculate counts on the fly; I don't ever use it

