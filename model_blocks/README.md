# Configs for model parts
Add these options to the top-level config file, unless otherwise noted

## PROTEIN SUBSTITUTION MODELS
### arguments for all models
All models should have these config options
- `norm [bool]`: normalize the substitution rate matrix (usually set this to `true`)
- `alphabet_size`: (alread defined in top-level `README.md`)


### subst_base
**ABOUT:** Load one exchangeability matrix (I usually use LG08)  

- `exch_files [str]`: name of the file containing the exchangeability matrix. default location to look is `./exchangeability_matrices`. Files should be exported numpy matrices, with names ending in `.npy`


### subst_mixture
**ABOUT:** Load multiple exchangeability matrices to fit a mixture  

These are unique config flags that MUST be provided
- `exch_files [list of str]`: a LIST of filenames, each containing an exchangeability matrix to add to the mixture model. Default location to look is `./exchangeability_matrices`. Files should be exported numpy matrices, with names ending in `.npy`  

Actions associated with multiple config options
- initializing mixture component logits: provide one of the following-
  1. `k_subst [int]`: How many mixture to use (this should match the length of `exch_files`). Mixture logits will be initialized with a vector of ones.
  2. `subst_mix_logits [list of ints]`: The desired initial values. This should be the sample length as `exch_files`


### subst_from_file
**ABOUT:** Load the log probability matrix directly  

These are unique config flags
- `exch_files [str]`: name of the file containing the log probability of each type of substitution. File should be exported numpy matrices whose name ends in `.npy`



## EQUILIBRIUM DISTRIBUTION MODELS
### equl_base
**ABOUT:** Calculate one equilibrium distribution from the training set data  

(no config arguments)


### equl_deltaMixture
**ABOUT:** Use a mixture of equilibrium distributions, drawn from a delta mixture (i.e. directly fit each equilibrium distribution)  

Actions associated with multiple config options
- initializing collection of equilibrium distributions: provide one group of the following-
  1. declare how many mixtures, to automatically initialize
    - `k_equl [int]`: How many mixtures to use. Equilibrium distribution logits will be inititalized from a normal distribution
    - `rng_seednum`: (alread defined in top-level `README.md`) 
    - `alphabet_size`: (alread defined in top-level `README.md`)
  2. manually initialize
    - `equl_vecs_transf [nested list of ints]`: the desired initial values

- initializing mixture logits: provide one of the following-
  1. `k_equl [int]`: How many mixtures to use. Mixture logits will be initialized with a vector of ones.
  2. `equl_mix_logits [list of ints]`: the desired initial values


### equl_dirichletMixture
**ABOUT:** Use a mixture of equilibrium distributions, drawn from a multinomial distribution with a dirichlet prior    

Actions associated with multiple config options
- initializing dirichlet shape parameter: provide one group of the following-
  1. declare how many mixtures, to automatically initialize
    - `k_equl [int]`: How many mixtures to use. Shape parameters will be a `(alphabet_size, k_equl)` matrix. First row will be initialized as a `(k_equl,)` vector of ones, next will be a vector of twos, etc.
    - `alphabet_size`: (alread defined in top-level `README.md`)
  2. manually initialize
    - `dirichlet_shape [nested list of ints]`: the desired initial values

- initializing mixture logits: provide one of the following-
  1. `k_equl [int]`: How many mixtures to use. Mixture logits will be initialized with a vector of ones.
  2. `equl_mix_logits [list of ints]`: the desired initial values


## INDEL MODELS

### no_indel
**ABOUT:** Placeholder to use if you want to ignore indels   

(no config arguments)


### GGI_single
**ABOUT:** Fit one H20 transition matrix  

These are unique config flags that MUST be provided
- `tie_params [bool]`: if true, insertion rate == deletion rate and insertion extension probability == deletion "extension" probability
- `diffrax_params [dict]`: a separate config providing parameters for diffrax solver (see diffrax documentation for more info):
  - `step [float]`
  - `rtol [float]`
  - `atol [float]`

Actions associated with multiple config options
- initialize indel rates: provide one group of the following-
  1. `indel_rate [float]`: initializes insertion and deletion rates with same value
  2. `lam [float = 0.5]` and/or `mu [float = 0.5]`: initialize insertion rate (`lam`) and/or deletion rate (`mu`)
  3. provide nothing: initialize both with default values

- initialize extension probabilities: provide one group of the following-
  1. `extension_prob [float]`: initializes insertion and deletion extension probabilities with same value (this is related to mean insertion and mean deletion length)
  2. `x [float = 0.5]` and/or `y [float = 0.5]`: initialize insertion (`x`) and/or deletion (`y`) extension probabilities
  3. provide nothing: initialize both with default values




### GGI_mixture
**ABOUT:** Fit a mixture of H20 models  

These are unique config flags that MUST be provided (see details in `GGI_single` above)  
- `tie_params [bool]`
- `diffrax_params [dict]`
  - `step [float]`
  - `rtol [float]`
  - `atol [float]` 

Actions associated with multiple config options
- initialize indel rates: provide one group of the following-
  1. `indel_rate [list of float]`: initializes insertion and deletion rates with same value
  2. `lam [list of float]` and/or `mu [list of float]`: initialize insertion rate (`lam`) and/or deletion rate (`mu`) with desired initial values
  3. `k_indel [int]`: How many mixtures to use. `lam` and `mu` will be initialized from equally-spaced values from 0.1 to 0.9.

- initialize extension probabilities: provide one group of the following-
  1. `extension_prob [list of float]`: initializes insertion and deletion extension probabilities with same value (this is related to mean insertion and mean deletion length)
  2. `x [list of float]` and/or `y [list of float]`: initialize insertion (`x`) and/or deletion (`y`) extension probabilities with desired initial values
  3. `k_indel [int]`: How many mixtures to use. `x` and `y` will be initialized from equally-spaced values from 0.1 to 0.9.

- initializing mixture logits: provide one of the following-
  1. `k_indel [int]`: How many mixtures to use. Mixture logits will be initialized with a vector of ones.
  2. `indel_mix_logits [list of ints]`: the desired initial values


### TKF91_single
**ABOUT:** TKF91 indel model  

These are unique config flags that MUST be provided
- `tie_params [bool]`: if true, insertion rate == deletion rate
- `offset [float=1e-4]`: the TKF models technically don't allow equal insertion and deletion rates. Instead, I set `mu = lam * (1+TKF_ERR) + offset`, where `TKF_ERR=1e-4` (if you keep the default value, you'd incorporate `1e-4` twice)

Actions associated with multiple config options
- initialize the insert rate: provide one of the following
  1. `indel_rate [float]` or `lam [float = 0.5]`: either will initialize `lam` with desired value
  2. provide nothing: initialize `lam` with default value


### TKF92_single
**ABOUT:** TKF92 indel model  

These are unique config flags that MUST be provided
- `tie_params [bool]`: see details in `GGI_single` above


Actions associated with multiple config options
- initialize indel rates: provide one group of the following-
  1. `indel_rate [float]`: initializes insertion rate with value provided, and calculates deletion rate as `mu = lam*(1+TKF_ERR)`
  2. `lam [float = 0.5]` and/or `offset [float = 1e-4]`: initialize insertion rate (`lam`) and/or offset (`offset`). See `TKF91_single` for how `offset` is used to calculate deletion rate `mu`.
  3. provide nothing: initialize both with default values

- initialize extension probabilities: provide one group of the following-
  1. `extension_prob [float]`: initializes insertion and deletion extension probabilities with same value (this is related to mean insertion and mean deletion length)
  2. `x [float = 0.5]` and/or `y [float = 0.5]`: initialize insertion (`x`) and/or deletion (`y`) extension probabilities
  3. provide nothing: initialize both with default values



### otherIndel_single
**ABOUT:** Use this broad class to initialize the LG05, RS07, or KM03 indel models  

These are unique config flags that MUST be provided
- `model_name [str]`: string from `[LG05, RS07, KM03]`
- `tie_params [bool]`: see details in `GGI_single` above

Actions associated with multiple config options 
- initialize indel rates: provide one group of the following-
  1. `indel_rate [float]`: initializes insertion and deletion rates with same value
  2. `lam [float = 0.5]` and/or `mu [float = 0.5]`: initialize insertion rate (`lam`) and/or deletion rate (`mu`)
  3. provide nothing: initialize both with default values

- initialize extension probabilities: provide one group of the following-
  1. `extension_prob [float]`: initializes insertion and deletion extension probabilities with same value (this is related to mean insertion and mean deletion length)
  2. `x [float = 0.5]` and/or `y [float = 0.5]`: initialize insertion (`x`) and/or deletion (`y`) extension probabilities
  3. provide nothing: initialize both with default values



## (not used) DNA SUBSTITUTION MODELS

### hky85
**ABOUT:** The HKY85 DNA substitution model; used to check my implementation against an older implementation from Ian  

config should include:
- `norm [bool]`: normalize the substitution rate matrix (usually set this to `true`)
- `gc [float]`: GC content
- `ti [float]`: transition param
- `tv [float]`: transversion param
