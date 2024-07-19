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
# @authors: Wenyuan Zhao. Some code snippets borrowed from: Intel Labs Bayeisan-Torch.
#
# ===============================================================================================


import torch
import torch.nn as nn
from itertools import repeat
import collections


def get_kernel_size(x, n):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))


class _BaseVariationalLayer(nn.Module):
    r"""
    The base variational layer is implemented as a :class:`torch.nn.Module` that, when called on two distributions 
    :math:`Q` and :math:`P` returns a :obj:`torch.Tensor` that represents the KL divergence between two gaussians 
    :math:`\left( Q\parallel P \right)`.

    .. math::

        \begin{equation*}
            D_{\text{KL}}\left( Q\parallel P \right)= \sum_{x\in \mathcal{X}}Q(x)\log\left( \frac{Q(x)}{P(x)} \right)
        \end{equation*}
    """

    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        r"""
        Calculates kl divergence between two gaussians (Q || P)

        :param mu_q: mean of distribution Q
        :type mu_q: torch.Tensor
        :sigma_q: deviation of distribution Q
        :type sigma_q: torch.Tensor
        :mu_p: mean of distribution P
        :type mu_p: torch.Tensor
        :sigma_p: deviation of distribution P
        :type sigma_p: torch.Tensor

        :return: the KL divergence between Q and P.
        """
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 *
                                                          (sigma_p**2)) - 0.5
        return kl.mean()