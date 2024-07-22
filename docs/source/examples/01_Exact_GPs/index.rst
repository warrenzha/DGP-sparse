Exact GPs (Regression)
==============

Regression is the canonical example of Gaussian processes. These examples will work for
small to medium sized datasets (~1,000 data points). All examples here use exact GP inference.

- `Simple GP Regression`_ is the basic tutorial for regression in GPyTorch.
- `Fully Bayesian GP Regression`_ demonstrates how to perform fully Bayesian inference by sampling the DMGP parameters
  using MCMC.

.. toctree::
   :maxdepth: 1
   :hidden:

   Simple_GP_Regression.ipynb
   GP_Regression_Fully_Bayesian.ipynb

.. _Simple GP Regression:
  ./Simple_GP_Regression.ipynb

.. _Fully Bayesian GP Regression:
  ./GP_Regression_Fully_Bayesian.ipynb
