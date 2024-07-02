from .simple_fc_variational import SFC
from .simple_cnn_variational import SCNN
from .simple_dgp_variational import SDGPgrid, SDGPadditive
from .mnist_dgp_variational import DAMGPmnist, DTMGPmnist
from .cifar_resnet_variational import *
from .cifar_dgp_variational import *
from .cifar_resgp_variational import *
from .imgnet_resnet_variational import *

__all__ = [
    "SFC",
    "SCNN",
    "SDGPadditive",
    "SDGPgrid",
    "DAMGPmnist",
    "DTMGPmnist",
]