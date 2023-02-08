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
    def __init__(self, input_dim, z_dim,device, nu=0):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.nu = nu
        self.device = device
        
        self.encConv1 = nn.Conv2d(1, 16, 5)
        self.norm1 = nn.BatchNorm2d(16)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.norm2 = nn.BatchNorm2d(32)

        self.latent_mu = nn.Linear(32*20*20, self.z_dim)
        self.latent_var = nn.Linear(32*20*20, self.z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.nu == 0:
            eps = torch.randn_like(std) # Normal dist : eps ~ N(0, I)
            return mu + std * eps
        else:
            '''
            
            Sampling algorithm
            1. Generate v ~ chiq(nu) and eps ~ N(0, (nu-2)/nu * var), independently.
            2. Caculate x = mu + eps / (sqrt(v/nu)) 
            (Note that the covariance matrix of MVT is nu/(nu-2)*((nu-2)/nu * var) = var)
            '''
            MVN_dist = torch.MultivariateNormal(torch.zeros(self.z_dim), torch.eye(self.z_dim))
            chi_dist = torch.distributions.chi2.Chi2(torch.tensor([self.nu]))
            eps = MVN_dist.sample(sample_shape = torch.Size(mu.shape)).to(self.device) # Student T dist
            Sigma = torch.tensor(np.sqrt((self.nu - 2) / self.nu) * std)
            v = chi_dist.sample()
            
            return mu + Sigma * eps / torch.sqrt(self.nu / v)

    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.encConv1(x)))
        x = F.leaky_relu(self.norm2(self.encConv2(x)))
        x = x.view(-1, 32*20*20)
        mu = self.latent_mu(x)
        logvar = self.latent_var(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def loss(self, mu, logvar, x, input_dim):
        if args.nu == 0:
            KL_div = Alpha_Family(mu, logvar)
            div_loss = KL_div.KL_loss()

        else:
            div_loss = gamma_neg_entropy(x,logvar,input_dim)
        
        return div_loss * args.beta