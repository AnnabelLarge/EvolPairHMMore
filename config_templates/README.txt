base.json:
===========
substitution model: single
equlibrium distribution model: single
indel model: single ggi

ggiMixr1,2.json:
================
substitution model: single
equlibrium distribution model: single
indel model: MIXTURE ggi

equlMixr1,2.json:
=================
substitution model: single
equlibrium distribution model: MIXTURE of equlibrium distributions, sampled from Dirichlet distribution with reparam. trick
indel model: single ggi

subMixr1,2.json:
================
substitution model: MIXTURE of rate classes, sampled from quantiles of the gamma distribution
equlibrium distribution model: single
indel model: single ggi


todo:
=====
- make a script that automatically validates the json config?
