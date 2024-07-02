# DGP_sparse
**DGP_sparse** is a Python library for sparse deep Gaussian processes (DGPs) with GPU acceleration. It is built on top of 
PyTorch and provides a simple and flexible API for building complex deep GP models as learnable neural networks.

## Tutorials and Documentation
See our [**documentation, examples, tutorials**](https://sparse-dgp.readthedocs.io/) on how to construct all sorts of 
DGP models in DGP-sparse.

## Installation

**Requirements**:
- Python >= 3.8
- PyTorch >= 1.11

**To install core library using `pip`:**
```
pip install dgp-sparse
```

**To install latest development version from source:**
```sh
git clone https://github.com/warrenzha/DGP_sparse.git
cd dgp-sparse
pip install .
```

## Usage
There are two ways to build sparse DGPs using DGP-sparse: 
1. Load a pre-trained model from the library
2. Define your custom model using the Bayesian layers provided in the library

### Load a pre-trained model
```bash
$ cd examples/mnist
$ python bayesian_dtmgp_mnist.py --model [additive_grid_model]
                                 --error-bar [plot_errorbar]
                                 --subset-size [training_subset]
                                 --batch-size [batch_size]
                                 --test-batch-size [test_batch_size]
                                 --epochs [epochs]
                                 --lr [learning_rate]
                                 --gamma [learning_rate_step_gamma]
                                 --no-cuda [disable_cuda]
                                 --seed [random_seed]
                                 --log-interval [num_batches_log]
                                 --save_dir [save_directory]
                                 --mode [train_test_mode]
                                 --num_monte_carlo [num_monte_carlo_inference]
                                 --num_mc [num_monte_carlo_training]
                                 --tensorboard [tensorboard_action]
                                 --log_dir [logs_directory]
```

### Define your custom model
``` python
import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
```

## Citing Us
If you use DGP-sparse, please cite as:
```bibtex
@software{zhao2022dgpsparse,
  author       = {Wenyuan Zhao and Haoyuan Cheng},               
  title        = {DGP-sparse: Sparse Deep Gaussian Processes in PyTorch},
  month        = jul,
  year         = 2024,
  doi          = {},
  url          = {https://sparse-dgp.readthedocs.io/},
  howpublished = {\url{https://https://github.com/warrenzha/DGP_sparse.git}}
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
Bayesian-Torch
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