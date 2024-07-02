'''
Bayesian Residual DGP for CIFAR10.

ResNet architecture ref:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from dgp_sparse.layers import LinearReparameterization
from dgp_sparse.layers import AMGP
from dgp_sparse.utils.sparse_activation.design_class import HyperbolicCrossDesign
from dgp_sparse.kernels.laplace_kernel import LaplaceProductKernel

__all__ = [
    'ResGP', 'resgp8', 'resgp20', 'resgp32', 'resgp44', 'resgp56'
]

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_features, features, design_class, kernel, n_level=5, option='B'):
        super(BasicBlock, self).__init__()
        self.gp1 = AMGP(
            in_features=in_features,
            n_level=n_level,
            design_class=design_class,
            kernel=kernel
        )
        self.fc1 = LinearReparameterization(
            in_features=self.gp1.out_features,
            out_features=features,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=False
        )
        self.gp2 = AMGP(
            in_features=features,
            n_level=n_level,
            design_class=design_class,
            kernel=kernel
        )
        self.fc2 = LinearReparameterization(
            in_features=self.gp2.out_features,
            out_features=features,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=False
        )

        self.shortcut = nn.Sequential()
        if in_features != features:
            padding_size = features - in_features
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x,
                    (padding_size // 2, padding_size // 2), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    LinearReparameterization(
                        in_features=in_features,
                        out_features=features,
                        prior_mean=prior_mu,
                        prior_variance=prior_sigma,
                        posterior_mu_init=posterior_mu_init,
                        posterior_rho_init=posterior_rho_init,
                        bias=False,
                        return_kl=False)
                )

    def forward(self, x):
        kl_sum = 0
        out = self.gp1(x)
        out, kl = self.fc1(out)
        kl_sum += kl
        out = self.gp2(out)
        out, kl = self.fc2(out)
        kl_sum += kl
        out += self.shortcut(x)

        return out, kl_sum


class ResGP(nn.Module):
    def __init__(self, input_dim, design_class, kernel, block, num_blocks, num_classes=10):
        super(ResGP, self).__init__()
        self.in_features = 256

        self.fc0 = LinearReparameterization(
            in_features=input_dim,
            out_features=256,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )
        self.layer1 = self._make_layer(design_class, kernel, block, 256, num_blocks[0])
        self.layer2 = self._make_layer(design_class, kernel, block, 128, num_blocks[1])
        self.layer3 = self._make_layer(design_class, kernel, block, 64, num_blocks[2])
        self.gp1 = AMGP(in_features=64, n_level=5, design_class=design_class, kernel=kernel)
        self.fc1 = LinearReparameterization(
            in_features=self.gp1.out_features,
            out_features=num_classes,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True
        )

    def _make_layer(self, design_class, kernel, block, features, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_features, features, design_class, kernel))
            self.in_features = features
        return nn.Sequential(*layers)

    def forward(self, x):
        kl_sum = 0
        out = x.view(x.size(0), -1)
        # out = torch.flatten(x, 1)
        out, kl = self.fc0(out)
        kl_sum += kl
        for layer in self.layer1:
            out, kl = layer(out)
            kl_sum += kl
        for layer in self.layer2:
            out, kl = layer(out)
            kl_sum += kl
        for layer in self.layer3:
            out, kl = layer(out)
            kl_sum += kl
        out = self.gp1(out)
        out, kl = self.fc1(out)
        kl_sum += kl
        return out, kl_sum


def resgp8(input_dim=3072, design_class=HyperbolicCrossDesign, kernel=LaplaceProductKernel(1.), num_classes=10):
    return ResGP(input_dim, design_class, kernel, BasicBlock, [1, 1, 1], num_classes)


def resgp20(input_dim=3072, design_class=HyperbolicCrossDesign, kernel=LaplaceProductKernel(1.), num_classes=10):
    return ResGP(input_dim, design_class, kernel, BasicBlock, [3, 3, 3], num_classes)


def resgp32(input_dim=3072, design_class=HyperbolicCrossDesign, kernel=LaplaceProductKernel(1.), num_classes=10):
    return ResGP(input_dim, design_class, kernel, BasicBlock, [5, 5, 5], num_classes)


def resgp44(input_dim=3072, design_class=HyperbolicCrossDesign, kernel=LaplaceProductKernel(1.), num_classes=10):
    return ResGP(input_dim, design_class, kernel, BasicBlock, [7, 7, 7], num_classes)


def resgp56(input_dim=3072, design_class=HyperbolicCrossDesign, kernel=LaplaceProductKernel(1.), num_classes=10):
    return ResGP(input_dim, design_class, kernel, BasicBlock, [9, 9, 9], num_classes)
