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
    def __init__(self, img_shape, DEVICE):
        super(Encoder, self).__init__()
        _, self.C, self.H, self.W = img_shape
        self.device = DEVICE
        hidden_dims = [128, 256, 512]
        layers = []
        input_ch = self.C
        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(input_ch, dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU())
            )
            input_ch = dim

        self.cnn_layers = nn.Sequential(*layers)
        
        # Linear Layers
        # n : nb of cnn layers. If H, W are even numbers,
        # 2**(2*n) * (self.H // 2**n) * self.W // 2**n = self.H * self.W
        n = len(hidden_dims)
        self.pdim = hidden_dims[-1]* math.ceil(self.H / 2**n) * math.ceil(self.W / 2**n)
        self.mu_layer = nn.Linear(self.pdim , args.zdim) 
        self.logvar_layer = nn.Linear(self.pdim , args.zdim)
            
        #precomputing constants
        if args.nu != 0:
            
            self.qdim = args.zdim
            
            self.gamma = -2 / (args.nu + self.pdim + self.qdim)
            
            log_tau_base = -self.pdim * np.log(args.recon_sigma) + log_t_normalizing_const(args.nu, self.pdim) - np.log(args.nu + self.pdim - 2) + np.log(args.nu-2)
            
            const_2bar1_term_1 = (1 + self.qdim / (args.nu + self.pdim -2))
            const_2bar1_term_2_log = -self.gamma / (1+self.gamma) * log_tau_base
            self.const_2bar1 = const_2bar1_term_1 * const_2bar1_term_2_log.exp()
            
            
            ## 230308 : add new constant nu*tau
            log_tau = 1 / (args.nu + self.pdim - 2) * log_tau_base
            self.tau = log_tau.exp()
    
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
            std = torch.exp(0.5 * logvar)
            std = np.sqrt(args.nu / nu_prime) * std
            v = chi_dist.sample().to(self.device)
            return mu + std * eps * torch.sqrt(nu_prime / v)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)
 
        mu = self.mu_layer(x)
        logvar = self.mu_layer(x)
        z = self.reparameterize(mu, logvar)

        # x = x.view(-1,self.input_dim)
        # x = F.relu(self.fc1(x))
        # mu = self.latent_mu(x)
        # logvar = self.latent_var(x)
        # z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def loss(self, mu, logvar):
        if args.nu == 0 or args.flat != 'y':
            # KL divergence
            reg_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            # gammaAE regularizer
            reg_loss = gamma_regularizer(mu, logvar, self.pdim, self.const_2bar1, self.gamma, self.tau)
        
        return reg_loss * args.reg_weight
