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
    def __init__(self, input_dim,DEVICE):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.device = DEVICE
        self.fc1 = nn.Linear(self.input_dim, 400)
        self.latent_mu = nn.Linear(400, args.zdim)
        self.latent_var = nn.Linear(400, args.zdim)
            
        #precomputing constants
        if args.nu != 0:
            self.gamma = -2 / (nu + p_dim + q_dim)
        
            const_2bar1_term_1 = (1 + q_dim / (nu + p_dim -2))
            const_2bar1_term_2_log = -gamma / (1+gamma) * (-p_dim * np.log(args.recon_sigma) + log_t_normalizing_const(nu, p_dim) - np.log(nu + p_dim - 2) + np.log(nu-2))
            self.const_2bar1 = const_2bar1_term_1 * const_2bar1_term_2_log.exp()
    
    def reparameterize(self, mu, logvar):
        if args.nu == 0:
            std = torch.exp(0.5 * logvar) # diagonal mat
            eps = torch.randn_like(std) # Normal dist : eps ~ N(0, I)
            return mu + std * eps
        else:
            '''
                Sampling algorithm
                Let nu_prime = nu + p_dim
                1. Generate v ~ chiq(nu_prime) and eps ~ N(0, I), independently.
                2. Caculate x = mu + std * eps / (sqrt(v/nu_prime)), where std = sqrt(nu/(nu_prime) * var)
            '''
            nu_prime = args.nu + args.zdim
            MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(args.zdim), torch.eye(args.zdim))
            chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu_prime]))
            
            # Student T dist : [B, z_dim]
            eps = MVN_dist.sample(sample_shape=torch.tensor([mu.shape[0]])).to(self.device)
            
            std = torch.sqrt((args.nu / nu_prime) * torch.exp(0.5 * logvar))
            v = chi_dist.sample().to(self.device)
            return mu + std * eps * torch.sqrt(nu_prime / v)

    def forward(self, x):
        x = x.view(-1,self.input_dim)
        x = F.relu(self.fc1(x))
        mu = self.latent_mu(x)
        logvar = self.latent_var(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def loss(self, mu, logvar, input_dim):
        if args.nu == 0:
            # KL divergence
            div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            # gammaAE regularizer
            div_loss = gamma_regularizer(mu, logvar, input_dim, self.const_2bar1, self.gamma)
        
        return div_loss * args.reg_weight
