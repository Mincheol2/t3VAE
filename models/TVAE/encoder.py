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
    def __init__(self, img_shape, num_components, DEVICE):
        super(Encoder, self).__init__()
        _, self.C, self.H, self.W = img_shape
        self.device = DEVICE
        self.num_components = num_components
        hidden_dims = [32, 64, 128, 256, 512]
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
        

        # Vampprior : use pseudo input layer
        self.idle_input = torch.eye(self.num_components, requires_grad= False).to(self.device)
        self.pseudo_input_layer = nn.Sequential(nn.Linear(self.num_components, self.pdim),
                                          nn.Hardtanh(min_val=0.0, max_val=1.0)
                                          )
        # the default mean and std value initialization in the VampPrior's GitHub code
        torch.nn.init.normal_(self.pseudo_input_layer.weight, mean=-0.05, std=0.01)
    
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
            nu_prime = args.nu + self.pdim
            MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(args.zdim), torch.eye(args.zdim))
            chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu_prime]))
            
            # Student T dist : [B, z_dim]
            eps = MVN_dist.sample(sample_shape=torch.tensor([mu.shape[0]])).to(self.device)
            std = torch.exp(0.5 * logvar)
            std = np.sqrt(args.nu / nu_prime) * std
            v = chi_dist.sample().to(self.device)
            return mu + std * eps * torch.sqrt(nu_prime / v)

    def forward(self, x):
        #pseudo_input = self.pseudoinput_layer(x)
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)
 
        mu = self.mu_layer(x)
        logvar = self.mu_layer(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def loss(self, z1, mu, logvar):
        prior_mu, prior_logvar = self.make_vampprior()
        prior_var = prior_log_var.exp()
        z1_expand = z1.unsqueeze(1) # why??

        # By default option, torch.sum() operates on the innermost dimension of the original code.
        E_log_q = torch.mean(torch.sum(-0.5 * (log_var + (z1 - mu) ** 2 / prior_var), dim = 1), dim = 0)
        E_log_p = torch.sum(-0.5 * (prior_log_var + (z1_expand - prior_mu) ** 2/ prior_var),
                              dim = 2) - torch.tensor(np.log(self.num_components)).float()

        # Pytorch already implements efficient logsumexp algorithms.
        E_log_p = torch.logsumexp(E_log_p, dim = 1)
        E_log_p = torch.mean(E_log_p, dim = 0)

        #KL div
        reg_loss = E_log_q - E_log_p
        return reg_loss * args.reg_weight

    def make_vampprior(self):
        z2 = self.pseudo_input_layer(self.idle_input)
        z2 = z2.view(-1, self.C, self.H, self.W)
        prior_mu, prior_log_var = self.encode(z2)

        # WhY?? 
        prior_mu = prior_mu.unsqueeze(0)
        prior_log_var = prior_log_var.unsqueeze(0)

        return prior_mu, prior_log_var
