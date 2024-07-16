Basic Usage
==============

This folder contains notebooks for basic usage of the package, e.g. things like training
and evaluating models, and saving and loading models. Basically, there are two ways to build sparse DGPs using DGP-sparse:

Load a pre-trained model
-----------------------

.. code-block:: console

   $ python examples/bayesian_mnist.py --model [additive_grid_model]
                                       --mode [train_test_mode]
                                       --batch-size [batch_size]
                                       --epochs [epochs]
                                       --lr [learning_rate]
                                       --save_dir [save_directory]
                                       --num_monte_carlo [num_monte_carlo_inference]
                                       --num_mc [num_monte_carlo_training]
                                       --log_dir [logs_directory]

Customize your model
-----------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   from dgp_sparse.models import DMGP
   from dgp_sparse.kernels import LaplaceProductKernel
   from dgp_sparse.utils import HyperbolicCrossDesign
   from dgp_sparse.layers import AMGP, LinearFlipout

   # Define the model
   model = DMGP(
       input_dim=784,
       output_dim=10,
       num_layers=3,
       num_inducing=4,
       hidden_dim=64,
       kernel=LaplaceProductKernel(),
       design_class=HyperbolicCrossDesign,
       layer_type=LinearFlipout,
   )

   # Define the loss
   def loss_fn(model, x, y):
       return -model.elbo(x, y)

   # Define the optimizer
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

   # Train the model
   for epoch in range(100):
       optimizer.zero_grad()
       loss = loss_fn(model, x, y)
       loss.backward()
       optimizer.step()

   # Evaluate the model
   model.eval()
   y_pred = model(x)
   acc = (y_pred.argmax(dim=-1) == y).float().mean()

   print(f"Accuracy: {acc}")


.. toctree::
   :maxdepth: 1
   :hidden:

   Defining_an_Example_Model.ipynb