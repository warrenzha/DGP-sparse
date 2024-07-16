from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgp_sparse.layers.linear import LinearFlipout, LinearReparameterization
from dgp_sparse.layers.activation import TMGP, AMGP
from dgp_sparse.kernels.laplace_kernel import LaplaceProductKernel
from dgp_sparse.utils.sparse_design.design_class import HyperbolicCrossDesign

__all__ = [
    'DMGP',
]

class GPLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_inducing=4,
                 dense=LinearFlipout,
                 gp_activation=AMGP,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 design_class=HyperbolicCrossDesign):
        super(GPLayer, self).__init__()

        # self.norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.gp = gp_activation(in_features=in_dim, n_level=num_inducing, design_class=design_class, kernel=kernel)
        self.dense = dense(in_features=self.gp.out_features, out_features=out_dim)

    def forward(self, x, return_kl=True):
        kl_sum = 0
        # x = self.norm(x)
        out = self.gp(x)
        out, kl = self.dense(out)
        kl_sum += kl

        if return_kl:
            return out, kl_sum
        else:
            return out


class DMGP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=2,
                 num_inducing=4,
                 hidden_dim=64,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 design_class=HyperbolicCrossDesign,
                 layer_type=LinearFlipout,
                 option='additive'):
        super(DMGP, self).__init__()

        self.embedding = LinearFlipout(input_dim, hidden_dim)
        activation = AMGP if option=='additive' else TMGP
        self.gp_layers = nn.ModuleList(
            [GPLayer(hidden_dim,
                     hidden_dim,
                     num_inducing,
                     layer_type,
                     activation,
                     kernel,
                     design_class,
                     ) for _ in range(num_layers)
             ]
        )
        self.classifier = LinearFlipout(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x, return_kl=True):
        kl_sum = 0
        out = torch.flatten(x, 1)
        out, kl = self.embedding(out)
        kl_sum += kl
        for gp_layer in self.gp_layers:
            out, kl = gp_layer(out)
            kl_sum += kl
        out, kl = self.classifier(out)
        kl_sum += kl
        if self.output_dim == 1:
            out = out.squeeze(-1)

        if return_kl:
            return out, kl_sum
        else:
            return out


