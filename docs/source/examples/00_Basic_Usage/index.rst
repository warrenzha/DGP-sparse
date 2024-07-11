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
   from dgp_sparse.models import DGP
   from dgp_sparse.likelihoods import Gaussian
   from dgp_sparse.kernels import RBF
   from dgp_sparse.layers import SVGPLayer, SVGPLayer
   from dgp_sparse.models import DGP
   from dgp_sparse.likelihoods import Gaussian
   from dgp_sparse.kernels import RBF
   from dgp_sparse.layers import SVGPLayer, SVGPLayer

   # Define the model
   model = DGP(
       num_layers=3,
       num_inducing=100,
       input_dim=784,
       output_dim=10,
       hidden_dims=[100, 100],
       likelihood=Gaussian(),
       kernel=RBF(),
       inducing_points=torch.randn(100, 784),
       layer_type=SVGPLayer,
       mean_layer=SVGPLayer,
       num_samples=10,
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