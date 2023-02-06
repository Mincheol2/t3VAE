import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import argument
import random
import math
import numpy as np
from loss import *
args = argument.args

class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim,device, df=0):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.df = df
        self.device = device
        
        self.encConv1 = nn.Conv2d(1, 16, 5)
        self.norm1 = nn.BatchNorm2d(16)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.norm2 = nn.BatchNorm2d(32)

        self.latent_mu = nn.Linear(32*20*20, self.z_dim)
        self.latent_var = nn.Linear(32*20*20, self.z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.df == 0:
            eps = torch.randn_like(std) # Normal dist
            return mu + std * eps
        else:
            '''
            Sampling algorithm
            1. Generate v ~ chiq(df) and eps ~ N(0, I), independently.
            2. Caculate x = mu + std * eps / (sqrt(v/df)) 
            '''
            MVN_dist = torch.MultivariateNormal(torch.zeros(self.z_dim), torch.eye(self.z_dim))
            chi_dist = torch.distributions.chi2.Chi2(torch.tensor([self.df]))
            
            
            y = MVNdist.sample(sample_shape = torch.Size(prior_mu.shape)).to(self.device) # Student T dist
            v = chi_dist.sample()
            return mu + std * eps / torch.sqrt(self.df / v)
        
        
        return mu + std * eps

    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.encConv1(x)))
        x = F.leaky_relu(self.norm2(self.encConv2(x)))
        x = x.view(-1, 32*20*20)
        mu = self.latent_mu(x)
        logvar = self.latent_var(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def loss(self, mu, logvar):
        if args.df == 0:
            KL_div = Alpha_Family(mu, logvar,args.prior_mu, args.prior_logvar)
            div_loss = KL_div.KL_loss()

        else:
            Gamma_div = Gamma_Family(mu, logvar,args.df,args.prior_mu, args.prior_logvar)
            div_loss = Gamma_div.gamma_divergence(args.df)
        
        return div_loss * args.beta

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            return m + torch.log(sum_exp)
