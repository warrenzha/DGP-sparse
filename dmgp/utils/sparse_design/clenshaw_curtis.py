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
# @authors: Haoyuan Chen.
#
# ===============================================================================================


import torch


def cc_design(deg, input_lb=-1, input_ub=1, dyadic_sort=True):
    """
    :param deg: degree of clenshaw curtis (# of points = 2^(deg) - 1)
    :param input_ld: lower bound of input, default=-1
    :param input_ub: upper bound of input, default=1
    :param dyadic_sort: if sort=True, return sorted incremental tensor, default=True

    "return: res: [-cos( 0*pi/ n ), -cos( 1*pi/ n ), ..., -cos( n*pi/ n ) ], where n = 2^deg
         
    """
    x_1 = input_lb
    x_n = input_ub
    n = 2**(deg)

    if dyadic_sort is True:
        res_basis = torch.empty(0)
        for i in range(1, deg+1):
            m_i = 2**i
            increment_set = - torch.cos( torch.pi * torch.arange(start=1, end=m_i, step=2) / m_i)
            res_basis = torch.cat((res_basis,increment_set), dim=0)
    else:
        res_basis = - torch.cos( torch.pi * torch.arange(1, n) / n) # interval on [-1,1]
    
    res = res_basis*(x_n-x_1)/2 + (x_n+x_1)/2 # [n-1] size tensor
    
    return res