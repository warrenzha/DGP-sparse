from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgp_sparse.layers import LinearReparameterization, Conv2dReparameterization
from dgp_sparse.layers import AMGP, TMGP

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


class DAMGPmnist(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 design_class, 
                 kernel,
                 activation=None):
        super(DAMGPmnist, self).__init__()

        self.activation = activation

        w0 = 64
        self.fc0 = LinearReparameterization(
            in_features=input_dim,
            out_features=w0,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

        #################################################################################
        ## 1st layer of DGP: input:[n, input_dim] size tensor, output:[n, w1] size tensor
        #################################################################################
        # return [n, m1] size tensor for [n, input_dim] size input and [m1, input_dim] size sparse grid
        self.tmk1 = AMGP(in_features=w0, n_level=5, design_class=design_class, kernel=kernel)
        m1 = self.tmk1.out_features # m1 = input_dim*(2^n_level-1)
        w1 = 64
        # return [n, w1] size tensor for [n, m1] size input and [m1, w1] size weights
        self.fc1 = LinearReparameterization(
            in_features=m1,
            out_features=w1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

        #################################################################################
        ## 2nd layer of DGP: input:[n, w1] size tensor, output:[n, w2] size tensor
        #################################################################################
        # return [n, m2] size tensor for [n, w1] size input and [m2, w1] size sparse grid
        self.tmk2 = AMGP(in_features=w1, n_level=5, design_class=design_class, kernel=kernel)
        m2 = self.tmk2.out_features # m2 = w1*(2^n_level-1)
        w2 = output_dim
        # return [n, w2] size tensor for [n, m2] size input and [m2, w2] size weights
        self.fc2 = LinearReparameterization(
            in_features=m2,
            out_features=w2,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

    def forward(self, x):
        kl_sum = 0

        x = torch.flatten(x, 1)
        x, kl = self.fc0(x)
        kl_sum += kl
        # x = F.relu(x)

        x = self.tmk1(x)
        x, kl = self.fc1(x)
        kl_sum += kl

        x = self.tmk2(x)
        x, kl = self.fc2(x)
        kl_sum += kl

        if self.activation is None:
            output = F.log_softmax(x, dim=1)
        else:
            output = self.activation(x)  # attention, this only regress non-negative values TODO XXX
        return output, kl_sum


class DTMGPmnist(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 design_class,
                 kernel,
                 activation=None):
        super(DTMGPmnist, self).__init__()

        self.activation = activation

        w0 = 8
        self.fc0 = LinearReparameterization(
            in_features=input_dim,
            out_features=w0,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

        #################################################################################
        ## 1st layer of DGP: input:[n, input_dim] size tensor, output:[n, w1] size tensor
        #################################################################################
        # return [n, m1] size tensor for [n, input_dim] size input and [m1, input_dim] size sparse grid
        self.tmk1 = TMGP(in_features=w0, n_level=3, design_class=design_class, kernel=kernel)
        m1 = self.tmk1.out_features
        w1 = 8
        # return [n, w1] size tensor for [n, m1] size input and [m1, w1] size weights
        self.fc1 = LinearReparameterization(
            in_features=m1,
            out_features=w1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

        #################################################################################
        ## 2nd layer of DGP: input:[n, w1] size tensor, output:[n, w2] size tensor
        #################################################################################
        # return [n, m2] size tensor for [n, w1] size input and [m2, w1] size sparse grid
        self.tmk2 = TMGP(in_features=w1, n_level=3, design_class=design_class, kernel=kernel)
        m2 = self.tmk2.out_features
        w2 = output_dim
        # return [n, w2] size tensor for [n, m2] size input and [m2, w2] size weights
        self.fc2 = LinearReparameterization(
            in_features=m2,
            out_features=w2,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

    def forward(self, x):
        kl_sum = 0

        x = torch.flatten(x, 1)
        x, kl = self.fc0(x)
        kl_sum += kl
        x = F.relu(x)

        x = self.tmk1(x)
        x, kl = self.fc1(x)
        kl_sum += kl

        x = self.tmk2(x)
        x, kl = self.fc2(x)
        kl_sum += kl

        if self.activation is None:
            output = F.log_softmax(x, dim=1)
        else:
            output = self.activation(x)  # attention, this only regress non-negative values TODO XXX
        return output, kl_sum