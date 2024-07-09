from .simple_fc_variational import SFC
from .simple_dgp_variational import DMGPgrid, DMGPadditive
from .mnist_dgp_variational import DAMGPmnist, DTMGPmnist
from .cifar_dgp_variational import *

__all__ = [
    "SFC",
    "DMGPadditive",
    "DMGPgrid",
    "DAMGPmnist",
    "DTMGPmnist",
]