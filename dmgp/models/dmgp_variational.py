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
    r"""
    Represents a layer in DMGP where inference is performed via finite-rank approximation, which
    can be represented as a one-layer neural network with correlated Gaussian distributed weights:

    .. math::

        \begin{align*}
            \hat{\mathcal{G}}^{(i)}(\cdot) := & \mu + k(\cdot, \mathbf{U})
            [ k(\mathbf{U}, \mathbf{U})]^{-1} \mathcal{G}^{(i)}(\mathbf{U}), \\
            = & \mu + k(\cdot, \mathbf{U}) R^{-1}_{\mathbf{U}} \mathbf{Z} \\
            = & \mu + \phi^{T}(\cdot) \mathbf{Z}
        \end{align*}

    A GP Layer consists of a GP activation :math:`\phi(\cdot) = k(\cdot, \mathbf{U}) R^{-1}_{\mathbf{U}}`
    and a linear layer with Gaussian weights :math:`\mathbf{Z} = [R^{T}_{\mathbf{U}}]^-1 \mathcal{G}(\mathbf{U})`.
    :math:`\mathbf{U}=\{ \mathbf{u}_i \}_{i=1}^{m}` are the inducing points for approximating GP. :math:`R_{\mathbf{U}}`
    is the Cholesky decomposition of the kernel matrix :math:`k(\mathbf{U}, \mathbf{U})`. The algorithm of Cholesky
    decomposition in DMGP can be found in `dmgp.utils.operators.chol_inv`.

    :param in_features: Input features of :math:`\mathbf x_1`.
    :type in_features: int
    :param out_features: Output features of GP layer.
    :type out_features: int
    :param num_inducing: Level of inducing points for approximating GP. Default: `3`.
    :type num_inducing: int, optional
    :param input_lb: Lower bound of the input space. You can choose any bound you want and apply normalization in the front. Default: `-2.`.
    :type input_lb: float, optional
    :param input_ub: Upper bound of the input space. You can choose any bound you want and apply normalization in the front. Default: `-2.`.
    :type input_ub: float, optional
    :param dense: The dense linear layer for Gaussian weights. Default: `LinearFlipout`.
    :type dense: class, dmgp.layers.linear, optional
    :param gp_activation: The GP activation layer. Default: `AMK`.
    :type gp_activation: class, dmgp.layers.activation, optional
    :param kernel: The GP kernel. Default: `LaplaceProductKernel`.
    :type kernel: class, dmgp.kernels, optional
    :param design_class: The design class of choosing inducing points for approximating GP. Default: `HyperbolicCrossDesign`.
    :type design_class: class, dmgp.utils.sparse_design.design_class, optional
    """
    def __init__(self,
                 in_features,
                 out_features,
                 num_inducing=2,
                 input_lb=-2,
                 input_ub=2,
                 dense=LinearFlipout,
                 gp_activation=AMK,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 design_class=HyperbolicCrossDesign):
        super(GPLayer, self).__init__()

        self.norm = nn.BatchNorm1d(in_features, affine=True)
        self.gp = gp_activation(
            in_features=in_features,
            n_level=num_inducing,
            input_lb=input_lb,
            input_ub=input_ub,
            kernel=kernel,
            design_class=design_class,
        )
        self.dense = dense(in_features=self.gp.out_features, out_features=out_features)

    def forward(self, x, normalize=True, return_kl=True):
        r"""
        Forward the GP inference :math:`\mu + \phi^{T}(x) \mathbf{Z}`.

        :param x: Training data of shape :math:`(n,d)`.
        :type x: torch.Tensor.float
        :param normalize: Apply normalization to fit induced points. Default: `True`.
        :type normalize: bool, optional
        :param return_kl: Return KL-divergence. Default: `True`.
        :type return_kl: bool, optional

        :return: The output of inference and KL-divergence.
        """
        kl_sum = 0
        if normalize:
            x = self.norm(x)
        out = self.gp(x)
        out, kl = self.dense(out)
        kl_sum += kl

        if return_kl:
            return out, kl_sum
        else:
            return out


class DMGP(nn.Module):
    r"""
    A container module to build a Deep GP. This module should contain GPLayer modules, and can also contain other modules as well.

    :param in_features: Input features of :math:`\mathbf x_1`.
    :type in_features: int
    :param out_features: Output features of GP layer.
    :type out_features: int
    :param num_layers: Number of hidden layers in DMGP model. Default: `2`.
    :type num_layers: int
    :param hidden_dim: Dimension of hidden layers in DMGP model. Default: `8`.
    :type hidden_dim: int
    :param num_inducing: Level of inducing points for approximating GP. For "sparse grid" design, we recommend low level of inducing. Default: `3`.
    :type num_inducing: int, optional
    :param input_lb: Lower bound of the input space. You can choose any bound you want and apply normalization in the front. Default: `-2.`.
    :type input_lb: float, optional
    :param input_ub: Upper bound of the input space. You can choose any bound you want and apply normalization in the front. Default: `-2.`.
    :type input_ub: float, optional
    :param kernel: The GP kernel. Default: `LaplaceProductKernel`.
    :type kernel: class, dmgp.kernels, optional
    :param design_class: The design class of choosing inducing points for approximating GP. Default: `HyperbolicCrossDesign`.
    :type design_class: class, dmgp.utils.sparse_design.design_class, optional
    :param linear_layer: The dense linear layer for Gaussian weights. Default: `LinearFlipout`.
    :type linear_layer: class, dmgp.layers.linear, optional
    :param option: The option of DMGP architecture: use sparse grid or additive structure. Default: `additive`.
    :type option: str, optional
    """
    def __init__(self,
                 in_features,
                 out_features,
                 num_layers=2,
                 hidden_dim=8,
                 num_inducing=2,
                 input_lb=-2,
                 input_ub=2,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 design_class=HyperbolicCrossDesign,
                 linear_layer=LinearFlipout,
                 option='additive'):
        super(DMGP, self).__init__()

        self.embedding = LinearFlipout(in_features, hidden_dim)
        activation = AMK if option == 'additive' else TMK
        self.gp_layers = nn.ModuleList(
            [GPLayer(hidden_dim,
                     hidden_dim,
                     num_inducing,
                     input_lb,
                     input_ub,
                     dense=linear_layer,
                     gp_activation=activation,
                     kernel=kernel,
                     design_class=design_class,
                     ) for _ in range(num_layers)
             ]
        )
        self.classifier = LinearFlipout(hidden_dim, out_features)

    def forward(self, x, normalize=True, return_kl=True):
        r"""
        Forward the DMGP inference.

        :param x: Training data of shape :math:`(n,d)`.
        :type x: torch.Tensor.float
        :param normalize: Apply normalization to fit induced points. Default: `True`.
        :type normalize: bool, optional
        :param return_kl: Return KL-divergence. Default: `True`.
        :type return_kl: bool, optional

        :return: The output of inference and KL-divergence.
        """
        kl_sum = 0
        out = torch.flatten(x, 1)
        out, kl = self.embedding(out)
        kl_sum += kl
        for gp_layer in self.gp_layers:
            out, kl = gp_layer(out, normalize)
            kl_sum += kl
        out, kl = self.classifier(out)
        kl_sum += kl

        if return_kl:
            return out, kl_sum
        else:
            return out
