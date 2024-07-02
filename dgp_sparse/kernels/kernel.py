# reference: https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/kernels/kernel.py
import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import ModuleList


def sq_dist(x1, x2, x1_eq_x2=False):
    """Equivalent to the square of `torch.cdist` with p=2."""
    # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        x2, x2_norm, x2_pad = x1, x1_norm, x1_pad
    else:
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        res.diagonal(dim1=-2, dim2=-1).fill_(0)

    # Zero out negative values
    return res.clamp_min_(0)


def dist(x1, x2, x1_eq_x2=False):
    """
    Equivalent to `torch.cdist` with p=2, but clamps the minimum element to 1e-15.
    """
    if not x1_eq_x2:
        res = torch.cdist(x1, x2)
        return res.clamp_min(1e-15)
    res = sq_dist(x1, x2, x1_eq_x2=x1_eq_x2)
    return res.clamp_min_(1e-30).sqrt_()


# only necessary for legacy purposes
class Distance(torch.nn.Module):
    def __init__(self, postprocess: Optional[Callable] = None):
        super().__init__()
        if postprocess is not None:
            warnings.warn(
                "The `postprocess` argument is deprecated. "
                "See https://github.com/cornellius-gp/gpytorch/pull/2205 for details.",
                DeprecationWarning,
            )
        self._postprocess = postprocess

    def _sq_dist(self, x1, x2, x1_eq_x2=False, postprocess=False):
        res = sq_dist(x1, x2, x1_eq_x2=x1_eq_x2)
        return self._postprocess(res) if postprocess else res

    def _dist(self, x1, x2, x1_eq_x2=False, postprocess=False):
        res = dist(x1, x2, x1_eq_x2=x1_eq_x2)
        return self._postprocess(res) if postprocess else res


class Kernel(torch.nn.Module):
    def __init__(
        self,
        ard_num_dims: Optional[int] = None,
        batch_shape: Optional[torch.Size] = None,
        active_dims: Optional[Tuple[int, ...]] = None,
        eps: float = 1e-6,
        **kwargs,
    ):
        super(Kernel, self).__init__()
        self._batch_shape = torch.Size([]) if batch_shape is None else batch_shape
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.register_buffer("active_dims", active_dims)
        self.ard_num_dims = ard_num_dims

        self.eps = eps

        param_transform = kwargs.get("param_transform")


        if param_transform is not None:
            warnings.warn(
                "The 'param_transform' argument is now deprecated. If you want to use a different "
                "transformation, specify a different 'lengthscale_constraint' instead.",
                DeprecationWarning,
            )

        if self.has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                name="raw_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
            )

        self.distance_module = None
        # TODO: Remove this on next official PyTorch release.
        self.__pdist_supports_batch = True

    
    @abstractmethod
    def forward(
        self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> Union[Tensor, LinearOperator]:
        r"""
        Computes the covariance between :math:`\mathbf x_1` and :math:`\mathbf x_2`.
        This method should be imlemented by all Kernel subclasses.

        :param x1: First set of data (... x N x D).
        :param x2: Second set of data (... x M x D).
        :param diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)
        :param last_dim_is_batch: If True, treat the last dimension
            of `x1` and `x2` as another batch dimension.
            (Useful for additive structure over the dimensions). (Default: False.)

        :return: The kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`
        """
        raise NotImplementedError()

    @property
    def batch_shape(self) -> torch.Size:
        kernels = list(self.sub_kernels())
        if len(kernels):
            return torch.broadcast_shapes(self._batch_shape, *[k.batch_shape for k in kernels])
        else:
            return self._batch_shape

    @batch_shape.setter
    def batch_shape(self, val: torch.Size):
        self._batch_shape = val

    @property
    def device(self) -> Optional[torch.device]:
        if self.has_lengthscale:
            return self.lengthscale.device
        devices = {param.device for param in self.parameters()}
        if len(devices) > 1:
            raise RuntimeError(f"The kernel's parameters are on multiple devices: {devices}.")
        elif devices:
            return devices.pop()
        return None

    @property
    def dtype(self) -> torch.dtype:
        if self.has_lengthscale:
            return self.lengthscale.dtype
        dtypes = {param.dtype for param in self.parameters()}
        if len(dtypes) > 1:
            raise RuntimeError(f"The kernel's parameters have multiple dtypes: {dtypes}.")
        elif dtypes:
            return dtypes.pop()
        return torch.get_default_dtype()

    @property
    def lengthscale(self) -> Tensor:
        if self.has_lengthscale:
            return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
        else:
            return None

    @lengthscale.setter
    def lengthscale(self, value: Tensor):
        self._set_lengthscale(value)

    @property
    def is_stationary(self) -> bool:
        return self.has_lengthscale

    def local_load_samples(self, samples_dict: Dict[str, Tensor], memo: set, prefix: str):
        num_samples = next(iter(samples_dict.values())).size(0)
        self.batch_shape = torch.Size([num_samples]) + self.batch_shape
        super().local_load_samples(samples_dict, memo, prefix)

    def covar_dist(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        square_dist: bool = False,
        **params,
    ) -> Tensor:
        r"""
        This is a helper method for computing the Euclidean distance between
        all pairs of points in :math:`\mathbf x_1` and :math:`\mathbf x_2`.

        :param x1: First set of data (... x N x D).
        :param x2: Second set of data (... x M x D).
        :param diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)
        :param last_dim_is_batch: If True, treat the last dimension
            of `x1` and `x2` as another batch dimension.
            (Useful for additive structure over the dimensions). (Default: False.)
        :param square_dist:
            If True, returns the squared distance rather than the standard distance. (Default: False.)
        :return: The kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)
        res = None

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                return torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            else:
                res = torch.linalg.norm(x1 - x2, dim=-1)  # 2-norm by default
                return res.pow(2) if square_dist else res
        else:
            dist_func = sq_dist if square_dist else dist
            return dist_func(x1, x2, x1_eq_x2)

    def named_sub_kernels(self) -> Iterable[Tuple[str, Kernel]]:
        """
        For compositional Kernel classes (e.g. :class:`~gpytorch.kernels.AdditiveKernel`
        or :class:`~gpytorch.kernels.ProductKernel`).

        :return: An iterator over the component kernel objects,
            along with the name of each component kernel.
        """
        for name, module in self.named_modules():
            if module is not self and isinstance(module, Kernel):
                yield name, module

    def num_outputs_per_input(self, x1: Tensor, x2: Tensor) -> int:
        """
        For most kernels, `num_outputs_per_input = 1`.

        However, some kernels (e.g. multitask kernels or interdomain kernels) return a
        `num_outputs_per_input x num_outputs_per_input` matrix of covariance values for
        every pair of data points.

        I.e. if `x1` is size `... x N x D` and `x2` is size `... x M x D`, then the size of the kernel
        will be `... x (N * num_outputs_per_input) x (M * num_outputs_per_input)`.

        :return: `num_outputs_per_input` (usually 1).
        """
        return 1

    def sub_kernels(self) -> Iterable[Kernel]:
        """
        For compositional Kernel classes (e.g. :class:`~gpytorch.kernels.AdditiveKernel`
        or :class:`~gpytorch.kernels.ProductKernel`).

        :return: An iterator over the component kernel objects.
        """
        for _, kernel in self.named_sub_kernels():
            yield kernel

    def __call__(
        self, x1: Tensor, x2: Optional[Tensor] = None, diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> Union[LazyEvaluatedKernelTensor, LinearOperator, Tensor]:
        r"""
        Computes the covariance between :math:`\mathbf x_1` and :math:`\mathbf x_2`.

        .. note::
            Following PyTorch convention, all :class:`~gpytorch.models.GP` objects should use `__call__`
            rather than :py:meth:`~gpytorch.kernels.Kernel.forward`.
            The `__call__` method applies additional pre- and post-processing to the `forward` method,
            and additionally employs a lazy evaluation scheme to reduce memory and computational costs.

        :param x1: First set of data (... x N x D).
        :param x2: Second set of data (... x M x D).
            (If `None`, then `x2` is set to `x1`.)
        :param diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)
        :param last_dim_is_batch: If True, treat the last dimension
            of `x1` and `x2` as another batch dimension.
            (Useful for additive structure over the dimensions). (Default: False.)

        :return: An object that will lazily evaluate to the kernel matrix or vector.
            The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`
        """
        x1_, x2_ = x1, x2

        # Select the active dimensions
        if self.active_dims is not None:
            x1_ = x1_.index_select(-1, self.active_dims)
            if x2_ is not None:
                x2_ = x2_.index_select(-1, self.active_dims)

        # Give x1_ and x2_ a last dimension, if necessary
        if x1_.ndimension() == 1:
            x1_ = x1_.unsqueeze(1)
        if x2_ is not None:
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(1)
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")

        if x2_ is None:
            x2_ = x1_


        if diag:
            res = super(Kernel, self).__call__(x1_, x2_, diag=True, last_dim_is_batch=last_dim_is_batch, **params)
            # Did this Kernel eat the diag option?
            # If it does not return a LazyEvaluatedKernelTensor, we can call diag on the output
            if not isinstance(res, LazyEvaluatedKernelTensor):
                if res.dim() == x1_.dim() and res.shape[-2:] == torch.Size((x1_.size(-2), x2_.size(-2))):
                    res = res.diagonal(dim1=-1, dim2=-2)
            return res

        else:
            if settings.lazily_evaluate_kernels.on():
                res = LazyEvaluatedKernelTensor(x1_, x2_, kernel=self, last_dim_is_batch=last_dim_is_batch, **params)
            else:
                res = to_linear_operator(
                    super(Kernel, self).__call__(x1_, x2_, last_dim_is_batch=last_dim_is_batch, **params)
                )
            return res

    def __getstate__(self):
        # JIT ScriptModules cannot be pickled
        self.distance_module = None
        return self.__dict__

    def __add__(self, other: Kernel) -> Kernel:
        kernels = []
        kernels += self.kernels if isinstance(self, AdditiveKernel) else [self]
        kernels += other.kernels if isinstance(other, AdditiveKernel) else [other]
        return AdditiveKernel(*kernels)

    def __mul__(self, other: Kernel) -> Kernel:
        kernels = []
        kernels += self.kernels if isinstance(self, ProductKernel) else [self]
        kernels += other.kernels if isinstance(other, ProductKernel) else [other]
        return ProductKernel(*kernels)

    def __setstate__(self, d):
        self.__dict__ = d

    def __getitem__(self, index) -> Kernel:
        r"""
        Constructs a new kernel where the lengthscale (and other kernel parameters)
        are modified by an indexing operation.

        :param index: Index to apply to all parameters.
        """

        if len(self.batch_shape) == 0:
            return self

        new_kernel = deepcopy(self)
        # Process the index
        index = index if isinstance(index, tuple) else (index,)

        for param_name, param in self.named_parameters(recurse=False):
            new_param = new_kernel.__getattr__(param_name)
            new_param.data = new_param.__getitem__(index)
            ndim_removed = len(param.shape) - len(new_param.shape)
            new_batch_shape_len = len(self.batch_shape) - ndim_removed
            new_kernel.batch_shape = new_param.shape[:new_batch_shape_len]

        for buffr_name, buffr in self.named_buffers(recurse=False):
            # For a given buffer, get the number of dimensions that do not correspond to the batch shape
            new_buffr = new_kernel.__getattr__(buffr_name)
            new_buffr.data = new_buffr.__getitem__(index)
            ndim_removed = len(buffr.shape) - len(new_buffr.shape)
            new_batch_shape_len = len(self.batch_shape) - ndim_removed
            new_kernel.batch_shape = new_buffr.shape[:new_batch_shape_len]

        for sub_module_name, sub_module in self.named_sub_kernels():
            new_kernel.__setattr__(sub_module_name, sub_module.__getitem__(index))

        return new_kernel
    

class AdditiveKernel(Kernel):
    """
    A Kernel that supports summing over multiple component kernels.

    Example:
        >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) + RBFKernel(active_dims=torch.tensor([2]))
        >>> x1 = torch.randn(50, 2)
        >>> additive_kernel_matrix = covar_module(x1)

    :param kernels: Kernels to add together.
    """
    @property
    def is_stationary(self) -> bool:
        return all(k.is_stationary for k in self.kernels)

    def __init__(self, *kernels: Iterable[Kernel]):
        super(AdditiveKernel, self).__init__()
        self.kernels = ModuleList(kernels)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Union[Tensor, LinearOperator]:
        res = ZeroLinearOperator() if not diag else 0
        for kern in self.kernels:
            next_term = kern(x1, x2, diag=diag, **params)
            if not diag:
                res = res + to_linear_operator(next_term)
            else:
                res = res + next_term

        return res

    def num_outputs_per_input(self, x1, x2):
        return self.kernels[0].num_outputs_per_input(x1, x2)

    def __getitem__(self, index) -> Kernel:
        new_kernel = deepcopy(self)
        for i, kernel in enumerate(self.kernels):
            new_kernel.kernels[i] = kernel.__getitem__(index)

        return new_kernel


class ProductKernel(Kernel):
    r"""
    A Kernel that supports elementwise multiplying multiple component kernels together.

    Example:
        >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) * RBFKernel(active_dims=torch.tensor([2]))
        >>> x1 = torch.randn(50, 2)
        >>> kernel_matrix = covar_module(x1) # The RBF Kernel already decomposes multiplicatively, so this is foolish!

    :param kernels: Kernels to multiply together.
    """

    @property
    def is_stationary(self) -> bool:
        return all(k.is_stationary for k in self.kernels)

    def __init__(self, *kernels: Iterable[Kernel]):
        super(ProductKernel, self).__init__()
        self.kernels = ModuleList(kernels)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Union[Tensor, LinearOperator]:
        x1_eq_x2 = torch.equal(x1, x2)

        if not x1_eq_x2:
            # If x1 != x2, then we can't make a MulLinearOperator because the kernel won't necessarily be
            # square/symmetric
            res = to_dense(self.kernels[0](x1, x2, diag=diag, **params))
        else:
            res = self.kernels[0](x1, x2, diag=diag, **params)

            if not diag:
                res = to_linear_operator(res)

        for kern in self.kernels[1:]:
            next_term = kern(x1, x2, diag=diag, **params)
            if not x1_eq_x2:
                # Again to_dense if x1 != x2
                res = res * to_dense(next_term)
            else:
                if not diag:
                    res = res * to_linear_operator(next_term)
                else:
                    res = res * next_term

        return res

    def num_outputs_per_input(self, x1: Tensor, x2: Tensor) -> int:
        return self.kernels[0].num_outputs_per_input(x1, x2)

    def __getitem__(self, index) -> Kernel:
        new_kernel = deepcopy(self)
        for i, kernel in enumerate(self.kernels):
            new_kernel.kernels[i] = kernel.__getitem__(index)

        return new_kernel