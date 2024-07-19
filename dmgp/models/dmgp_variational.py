# Copyright (c) 2024 Wenyuan Zhao, Haoyuan Chen
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @authors: Wenyuan Zhao, Haoyuan Chen.
#
# ===============================================================================================


from __future__ import print_function
import torch
import torch.nn as nn

from dmgp.layers.linear import LinearFlipout, LinearReparameterization
from dmgp.layers.activation import TMK, AMK
from dmgp.kernels.laplace_kernel import LaplaceProductKernel
from dmgp.utils.sparse_design.design_class import HyperbolicCrossDesign

__all__ = [
    'GPLayer',
    'DMGP',
]

class GPLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_inducing=4,
                 dense=LinearFlipout,
                 gp_activation=AMK,
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
        activation = AMK if option == 'additive' else TMK
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
