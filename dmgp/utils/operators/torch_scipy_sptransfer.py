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
from scipy.sparse import coo_array

def scipy_coo_to_torch_coo(scipy_coo_array):
    """
    convert scipy.sparse.coo_array to torch.sparse_coo_tensor
    """
    row = torch.tensor(scipy_coo_array.row)
    col = torch.tensor(scipy_coo_array.col)
    vals = torch.tensor(scipy_coo_array.data, dtype=torch.float32)
    torch_coo_tensor = torch.sparse_coo_tensor(indices=torch.vstack((row, col)), values=vals, size=scipy_coo_array.shape) 
    return torch_coo_tensor

def torch_coo_to_scipy_coo(torch_coo_tensor):
    """
    convert torch.sparse_coo_tensor to scipy.sparse.coo_array
    """
    ids = torch_coo_tensor._indices().numpy()
    data = torch_coo_tensor._values().numpy()
    scipy_coo_array = coo_array( (data, (ids[0,:], ids[1,:]) ), shape=list(torch_coo_tensor.shape) )
    return scipy_coo_array