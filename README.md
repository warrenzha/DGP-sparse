# Deep Markov Gaussian Processes (DMGP)

[![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/warrenzha/dmgp)
[![version](https://img.shields.io/badge/release-0.1.0-green)](https://test.pypi.org/project/dmgp/)
[![Documentation Status](https://readthedocs.org/projects/sparse-dgp/badge/?version=latest)](https://dmgp.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DMGP** is a Python library for sparse deep Gaussian processes (DGPs) and uncertainty estimation with GPU acceleration. 
It is built on top of PyTorch and provides a simple and flexible API for building complex deep GP models as trainable 
neural networks.

## Tutorials and Documentation
See our [**documentation, examples, tutorials**](https://dmgp.readthedocs.io/en/latest/) on how to construct all sorts of 
DGP models in DMGP.

## Installation

**Requirements**:
- Python >= 3.8
- PyTorch >= 1.9.0
- SciPy >= 1.7.0

**To install core library using `pip`:**
```
pip install -i https://test.pypi.org/simple/ dmgp
```

**To install latest development version from source:**
```sh
git clone https://github.com/warrenzha/dmgp.git
cd dmgp
pip install -r requirements.txt
```

## Usage
There are two ways to build sparse DGPs using DMGP: 
1. Load a pre-trained model from the library
2. Define your custom model with modules provided in the library

### Load a pre-trained model
```bash
$ cd examples
$ python bayesian_mnist.py --model [additive]
                           --mode [test]
                           --batch-size [batch_size]
                           --epochs [epochs]
                           --lr [learning_rate]
                           --save_dir [save_directory] 
                           --num_monte_carlo [num_monte_carlo_inference]
                           --num_mc [num_monte_carlo_training]
                           --log_dir [logs_directory]
```

### Define your custom model
``` python
import torch
from dmgp.layers import LinearFlipout
from dmgp.kernels import LaplaceProductKernel
from dmgp.utils.sparse_design import HyperbolicCrossDesign
from dmgp.models import DMGP

batch, dim = 1000, 7
x = torch.randn(batch, dim).to("cuda")
model = DMGP(
    input_dim = dim,
    output_dim = 1,
    num_layers = 2,
    num_inducing = 3,
    hidden_dim = 64,
    kernel = LaplaceProductKernel(lengthscale=1.),
    design_class = HyperbolicCrossDesign,
    layer_type = LinearFlipout,
    option = 'additive',
).to("cuda")
y = model(x)
```

## Citing Us
If you use DMGP, please cite as:
```bibtex
@software{zhao2024dgpsparse,
  author       = {Wenyuan Zhao and Haoyuan Chen},               
  title        = {DMGP: Sparse expansion for deep Gaussian processes in PyTorch},
  month        = jul,
  year         = 2024,
  doi          = {},
  url          = {https://dmgp.readthedocs.io/},
  howpublished = {\url{https://github.com/warrenzha/dmgp.git}}
}
```
A Sparse Expansion for Deep Gaussian Processes
```bibtex
@article{ding2024sparse,
  title={A sparse expansion for deep Gaussian processes},
  author={Ding, Liang and Tuo, Rui and Shahrampour, Shahin},
  journal={IISE Transactions},
  volume={56},
  number={5},
  pages={559--572},
  year={2024},
  publisher={Taylor \& Francis}
}
```
Some BNN implementation is based on [Bayesian-Torch](https://github.com/IntelLabs/bayesian-torch)
```bibtex
@software{krishnan2022bayesiantorch,
  author       = {Ranganath Krishnan and Pi Esposito and Mahesh Subedar},               
  title        = {Bayesian-Torch: Bayesian neural network layers for uncertainty estimation},
  month        = jan,
  year         = 2022,
  doi          = {10.5281/zenodo.5908307},
  url          = {https://doi.org/10.5281/zenodo.5908307}
  howpublished = {\url{https://github.com/IntelLabs/bayesian-torch}}
}
```