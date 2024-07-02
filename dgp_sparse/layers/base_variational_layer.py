# Copyright (C) 2024 Intel Labs
#
# BSD-3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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