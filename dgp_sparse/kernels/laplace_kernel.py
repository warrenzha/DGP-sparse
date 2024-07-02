from typing import Optional

import torch
from torch import Tensor


class LaplaceProductKernel(torch.nn.Module):
    r"""
    Computes a covariance matrix based on the Laplace product kernel 
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

        \begin{equation*}
            k\left( \mathbf{x_1}, \mathbf{x_2} \right) = \exp\left\{ -\frac{\left\| 
            \mathbf{x_1}- \mathbf{x_2} \right\|_1}{\theta} \right\}
        \end{equation*}
    
    where :math:`\theta` is the lengthscale parameter.

    
    :param lengthscale: Set this if you want a customized lengthscale. It should be a [d] size tensor. Default: `None` = d.
    """

    def __init__(self, lengthscale=None):
        super().__init__()
        self.lengthscale = lengthscale

    def forward(self, x1: Tensor, x2: Optional[Tensor] = None, 
                diag: bool = False, **params) -> Tensor:
        r"""
        Computes the covariance between :math:`\mathbf x_1` and :math:`\mathbf x_2`.
        
        :param x1: First set of data.
        :type x1: n x d torch.Tensor.float
        :param x2: Second set of data.
        :type x2: m x d torch.Tensor.float
        :param diag: Should the kernel compute the whole kernel, or just the diag? If `True`, it must be the case that `x1 == x2`. (Default: `False`.)
        :type x1: bool, optional
        
        :return: the kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * 'full_covar`: `n x m`
            * `diag`: `n`
        """
        # Size checking
        if x1.ndimension() == 1:
            x1 = x1.unsqueeze(1)    # Add a last dimension, if necessary
        if x2 is not None:
            if x2.ndimension() == 1:
                x2 = x2.unsqueeze(1)
            if not x1.size(-1) == x2.size(-1):
                raise RuntimeError("x1 and x2 must have the same number of dimensions!")
        else:
            x2 = x1

        # Reshape lengthscale
        d = x1.shape[-1]
        if self.lengthscale is None:
            lengthscale = x1.new_full(size=(d,), fill_value=d, dtype=x1.dtype)
        else:
            lengthscale = self.lengthscale

        # Type checking
        if isinstance(lengthscale, (int, float)):
            lengthscale = x1.new_full(size=(d,), fill_value=lengthscale, dtype=x1.dtype)    # [d,] torch.Tensor([1., 1.,.., 1.])
        
        if isinstance(lengthscale, Tensor):
            if lengthscale.ndimension() == 0 or max(lengthscale.size()) == 1:
                lengthscale = x1.new_full(size=(d,), fill_value=lengthscale.item(), dtype=x1.dtype)
            if not max(lengthscale.size()) == d:
                raise RuntimeError("lengthscale and input must have the same dimension")
        
        lengthscale = lengthscale.reshape(-1)

        adjustment = x1.mean(dim=-2, keepdim=True)  # [d] size tensor
        x1_ = (x1 - adjustment).div(lengthscale)
        x2_ = (x2 - adjustment).div(lengthscale)
        x1_eq_x2 = torch.equal(x1_, x2_)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                distance = torch.zeros(*x1_.shape[:-2], x1_.shape[-2], dtype=x1_.dtype, device=x1.device)
            else:
                distance = torch.sum(torch.abs(x1_-x2_), dim=-1)
        else:
            distance = torch.cdist(x1_, x2_, p=1)
            distance = distance.clamp_min(1e-15)

        res = torch.exp(-distance)
        return res


class LaplaceAdditiveKernel(torch.nn.Module):
    r"""
    Computes a covariance matrix based on the Laplace additive kernel 
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

        \begin{equation*}
            k\left( \mathbf{x_1}, \mathbf{x_2} \right) = \sum_{j=1}^{d}k_j\left( x_{1,j}, x_{2,j} \right)
        \end{equation*}
    
    where :math:`\theta` is the lengthscale parameter.

    :param lengthscale: Set this if you want a customized lengthscale. It should be a [d] size tensor. (Default: `None` = d.)
    :type lengthscale: float or torch.Tensor.float, optional
    """
    def __init__(self, lengthscale=None):
        super().__init__()
        self.lengthscale = lengthscale

    def forward(self, x1: Tensor, x2: Optional[Tensor] = None, 
                diag: bool = False, **params) -> Tensor:
        """
        Computes the covariance between :math:`\mathbf x_1` and :math:`\mathbf x_2`.
        
        :param x1: First set of data.
        :type x1: n x d torch.Tensor.float
        :param x2: Second set of data.
        :type x2: m x d torch.Tensor.float
        :param diag: Should the kernel compute the whole kernel, or just the diag? If `True`, it must be the case that `x1 == x2`. (Default: `False`.)
        :type x1: bool, optional

        :return: the kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * 'full_covar`: `n x m`
            * `diag`: `n`
        """        
        # Size checking
        if x1.ndimension() == 1:
            x1 = x1.unsqueeze(1)    # Add a last dimension, if necessary
        if x2 is not None:
            if x2.ndimension() == 1:
                x2 = x2.unsqueeze(1)
            if not x1.size(-1) == x2.size(-1):
                raise RuntimeError("x1 and x2 must have the same number of dimensions!")
        else:
            x2 = x1

        # Reshape lengthscale
        d = x1.shape[-1]
        if self.lengthscale is None:
            lengthscale = x1.new_full(size=(d,), fill_value=d, dtype=x1.dtype)
        else:
            lengthscale = self.lengthscale

        # Type checking
        if isinstance(lengthscale, (int, float)):
            lengthscale = x1.new_full(size=(d,), fill_value=lengthscale, dtype=x1.dtype)    # [d,] torch.Tensor([1., 1.,.., 1.])
        
        if isinstance(lengthscale, Tensor):
            if lengthscale.ndimension() == 0 or max(lengthscale.size()) == 1:
                lengthscale = x1.new_full(size=(d,), fill_value=lengthscale.item(), dtype=x1.dtype)
            if not max(lengthscale.size()) == d:
                raise RuntimeError("lengthscale and input must have the same dimension")
        
        lengthscale = lengthscale.reshape(-1)

        adjustment = x1.mean(dim=-2, keepdim=True) # [d] size tensor
        x1_ = (x1 - adjustment).div(lengthscale)
        x2_ = (x2 - adjustment).div(lengthscale)
        x1_eq_x2 = torch.equal(x1_, x2_)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                distance = torch.zeros(*x1_.shape[:-2], x1_.shape[-2], dtype=x1_.dtype, device=x1.device)
            else:
                distance = torch.abs(x1_-x2_)
        else:
            distance = x1_.unsqueeze(dim=-2).repeat(*x1_.shape[:-2],1,x2_.shape[-2],1) - x2_.unsqueeze(dim=-3).repeat(*x2_.shape[:-2],x1_.shape[-2],1,1)

        res = torch.sum(torch.exp(-distance), dim=-1)
        return res
