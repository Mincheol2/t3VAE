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
    def __init__(self, input_dim, z_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(28*28, 400)
        self.latent_mu = nn.Linear(400, self.z_dim)
        self.latent_var = nn.Linear(400, self.z_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if args.nu == 0:
            eps = torch.randn_like(std) # Normal dist : eps ~ N(0, I)
            return mu + std * eps
        else:
            '''
            Sampling algorithm
            Let nu_prime = nu + p_dim
            1. Generate v ~ chiq(nu_prime) and eps ~ N(0, I), independently.
            2. Caculate x = mu + std * eps / (sqrt(v/nu)), where std = sqrt(nu/(nu_prime) * var)
            '''
            nu_prime = args.nu + self.z_dim
            MVN_dist = torch.MultivariateNormal(torch.zeros(self.z_dim), torch.eye(self.z_dim))
            chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu_prime]))

            eps = MVN_dist.sample(sample_shape = torch.Size(mu.shape)) # Student T dist
            std = torch.tensor(np.sqrt((args.nu / nu_prime) * std))
            v = chi_dist.sample()
            
            return mu + std * eps * torch.sqrt(nu_prime / v)

    def forward(self, x):
        x = x.view(-1,784)
        x = F.relu(self.fc1(x))
        mu = self.latent_mu(x)
        logvar = self.latent_var(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def loss(self, mu, logvar, input_dim):
        if args.nu == 0:
            # Vanila VAE and RVAE
            # KL_div = Alpha_Family(mu, logvar)
            # div_loss = KL_div.KL_loss()
            div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            # gammaAE
            div_loss = gamma_regularizer(mu, logvar, input_dim)
        return div_loss * args.reg_weight
