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
# GP activations for deep Gaussian processes
#
# @authors: Haoyuan Chen, Wenyuan Zhao
#
# ===============================================================================================


import torch
import torch.nn as nn
from dmgp.kernels.laplace_kernel import LaplaceProductKernel
from dmgp.utils.sparse_design.design_class import HyperbolicCrossDesign, SparseGridDesign
from dmgp.utils.operators.chol_inv import mk_chol_inv, tmk_chol_inv

__all__ = [
    'TMK',
    'AMK',
]


class TMK(nn.Module):
    r"""
    Implements tensor markov GP as an activation layer using sparse grid structure.

    .. math::

        \begin{equation*}
            k\left( \mathbf{x}, X^{SG} \right)R^{-1}
        \end{equation*}

    :param in_features: Size of each input sample.
    :type in_features: int
    :param n_level: Level of sparse grid design. (Default: `2`.)
    :type n_level: int, optional
    :param input_lb: Input lower boundary. (Default: `-2`.)
    :type input_lb: float, optional
    :param input_ub: Input upper boundary. (Default: `2`.)
    :type input_ub: float, optional
    :param design_class: Base design class of sparse grid. (Default: `HyperbolicCrossDesign`.)
    :type design_class: class, dmgp.utils.sparse_design.design_class, optional
    :param kernel: Kernel function of deep GP. (Default: `LaplaceProductKernel(lengthscale=1.)`.)
    :type kernel: class, dmgp.kernels, optional
    """

    def __init__(self,
                 in_features,
                 n_level=2,
                 input_lb=-2,
                 input_ub=2,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 design_class=HyperbolicCrossDesign,
                 ):
        super().__init__()

        self.kernel = kernel

        if in_features == 1:  # one-dimension TMGP
            dyadic_design = design_class(dyadic_sort=True, return_neighbors=True)(deg=n_level, input_lb=input_lb, input_ub=input_ub)
            chol_inv = mk_chol_inv(dyadic_design=dyadic_design, markov_kernel=kernel, upper=True)
            design_points = dyadic_design.points.reshape(-1, 1)
        else:  # multi-dimension TMGP
            eta = int(in_features + n_level)
            sg = SparseGridDesign(in_features, eta, input_lb=input_lb, input_ub=input_ub, design_class=design_class).gen_sg(
                dyadic_sort=True, return_neighbors=True)
            chol_inv = tmk_chol_inv(sparse_grid_design=sg, tensor_markov_kernel=kernel, upper=True)
            design_points = sg.pts_set

        self.register_buffer('design_points',
                             design_points)  # [m,d] size tensor, sparse grid points X^{SG} of dyadic sort
        self.register_buffer('chol_inv',
                             chol_inv)  # [m,m] size tensor, inverse of Cholesky decomposition of kernel(X^{SG},X^{SG})
        self.out_features = design_points.shape[0]

    def forward(self, x):
        r"""
        Computes the tensor markov kernel activation of :math:`\mathbf x`.

        :param x: [N, C] size tensor, N is the batch size, C is the feature size of input
        :type x: torch.Tensor.float

        :return: [N, M] size tensor, kernel(input, sparse_grid) @ chol_inv
        """

        out = self.kernel(x, self.design_points)  # [N, C] size tensor
        out = out @ self.chol_inv  # [N, M] size tensor

        return out


class AMK(nn.Module):
    r"""
    Implements additive markov GP as an activation layer using additive structure.

    .. math::

        \begin{equation*}
            \left\{ k\left( x_i, X^{SG} \right)R^{-1} \right\}^{d}_{i=1}
        \end{equation*}

    :param in_features: Size of each input sample.
    :type in_features: int
    :param n_level: Level of induced points for approximating GP. (Default: `3`.)
    :type n_level: int, optional
    :param input_lb: Input lower boundary. (Default: `-2`.)
    :type input_lb: float, optional
    :param input_ub: Input upper boundary. (Default: `2`.)
    :type input_ub: float, optional
    :param design_class: Base design class of sparse grid. (Default: `HyperbolicCrossDesign`.)
    :type design_class: class, dmgp.utils.sparse_design.design_class, optional
    :param kernel: Kernel function of deep GP. (Default: `LaplaceProductKernel(lengthscale=1.)`.)
    :type kernel: class, dmgp.kernels, optional
    """

    def __init__(self,
                 in_features,
                 n_level=3,
                 input_lb=-2,
                 input_ub=2,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 design_class=HyperbolicCrossDesign,
                 ):
        super().__init__()

        self.kernel = kernel

        dyadic_design = design_class(dyadic_sort=True, return_neighbors=True)(deg=n_level, input_lb=input_lb, input_ub=input_ub)
        chol_inv = mk_chol_inv(dyadic_design=dyadic_design, markov_kernel=kernel, upper=True)  # [m, m] size tensor
        design_points = dyadic_design.points.reshape(-1, 1)  # [m, 1] size tensor

        self.register_buffer('design_points',
                             design_points)  # [m,d] size tensor, sparse grid points X^{SG} of dyadic sort
        self.register_buffer('chol_inv',
                             chol_inv)  # [m,m] size tensor, inverse of Cholesky decomposition of kernel(X^{SG},X^{SG})
        self.out_features = design_points.shape[0] * in_features  # m*d

    def forward(self, x):
        r"""
        Computes the element-wise markov kernel activation of :math:`\mathbf x`.

        :param x: [N, C] size tensor, N is the batch size, C is the channels of input, L is the sequence length
        :type x: torch.Tensor.float

        :return: [N, C*L*M] size tensor, kernel(input, sparse_grid) @ chol_inv
        """

        out = torch.flatten(x, start_dim=1).unsqueeze(dim=-1)  # reshape x of size [N, C, L] --> size [N, C*L, 1]
        out = self.kernel(out, self.design_points)  # [N, C*L, M] size tenosr
        out = torch.matmul(out, self.chol_inv)  # [N, C*L, M] size tensor
        out = torch.flatten(out, start_dim=1)  # [N, C*L*M] size tensor

        return out
