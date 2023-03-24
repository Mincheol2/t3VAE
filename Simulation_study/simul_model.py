import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from simul_loss import log_t_normalizing_const, gamma_regularizer

class Encoder(nn.Module):
    def __init__(self, p_dim, q_dim, nu, DEVICE, num_layers = 32, recon_sigma = 0.5, reg_weight = 1.0):
        super(Encoder, self).__init__()
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.nu = nu
        self.num_layers = 20
        self.device = DEVICE
        self.recon_sigma = recon_sigma
        self.reg_weight = reg_weight

        self.fc = nn.Sequential(
            nn.Linear(self.p_dim, self.num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(self.num_layers, self.num_layers), 
            nn.LeakyReLU()
        )

        self.latent_mu = nn.Linear(self.num_layers, self.q_dim)
        self.latent_var = nn.Linear(self.num_layers, self.q_dim)
        
            
        #precomputing constants
        if self.nu != 0:
            
            self.gamma = -2 / (self.nu + self.p_dim + self.q_dim)
            
            log_tau_base = -self.p_dim * np.log(self.recon_sigma) + log_t_normalizing_const(self.nu, self.p_dim) - np.log(self.nu + self.p_dim - 2) + np.log(self.nu-2)
            
            const_2bar1_term_1 = (1 + self.q_dim / (self.nu + self.p_dim -2))
            const_2bar1_term_2_log = -self.gamma / (1+self.gamma) * log_tau_base
            self.const_2bar1 = const_2bar1_term_1 * const_2bar1_term_2_log.exp()
            
            
            ## 230308 : add new constant nu*tau
            ## 230320 : revise tau
            log_tau = 2 / (self.nu + self.p_dim - 2 ) * log_tau_base
            self.tau = log_tau.exp()
    
    def reparameterize(self, mu, logvar):
        if self.nu == 0:
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
            nu_prime = self.nu + self.q_dim
            MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(self.q_dim), torch.eye(self.q_dim))
            chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu_prime]))
            
            # Student T dist : [B, z_dim]
            eps = MVN_dist.sample(sample_shape=torch.tensor([mu.shape[0]])).to(self.device)
            
            std = np.sqrt(self.nu / nu_prime) * torch.exp(0.5 * logvar)
            v = chi_dist.sample().to(self.device)
            return mu + std * eps * torch.sqrt(nu_prime / v)

    def forward(self, x):
        x = self.fc(x)
        # x = F.leaky_relu(self.fc2(x))
        mu = self.latent_mu(x)
        logvar = self.latent_var(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def loss(self, mu, logvar):
        if self.nu == 0:
            # KL divergence
            reg_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            # gammaAE regularizer
            reg_loss = gamma_regularizer(mu, logvar, self.p_dim, self.const_2bar1, self.gamma, self.tau, self.nu)
        
        return reg_loss * self.reg_weight

class Decoder(nn.Module):
    def __init__(self, p_dim, q_dim, num_layers = 20, recon_sigma = 0.5):
        super(Decoder, self).__init__()
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.num_layers = num_layers
        self.recon_sigma = 0.5
        self.fc = nn.Sequential(
            nn.Linear(self.q_dim, self.num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(self.num_layers, self.num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(self.num_layers, self.p_dim)
        )

    def forward(self, enc_z):
        x = self.fc(enc_z)
        return x

    def loss(self, recon_x, x):
        recon_loss = F.mse_loss(recon_x, x, reduction = 'mean') / self.recon_sigma**2
        
        return recon_loss


class gammaAE():
    def __init__(self, dataset, p_dim, q_dim, nu, DEVICE, num_layers = 20, 
                 recon_sigma = 0.5, reg_weight = 1.0, lr = 5e-4, batch_size = 64):
        self.dataset = dataset
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.nu = nu
        self.DEVICE = DEVICE
        self.num_layers = num_layers
        self.recon_sigma = recon_sigma
        self.reg_weight = reg_weight
        self.lr = lr
        self.batch_size = batch_size

        self.encoder = Encoder(self.p_dim, self.q_dim, self.nu, 
                               self.DEVICE, self.num_layers, self.recon_sigma, self.reg_weight).to(self.DEVICE)
        self.decoder = Decoder(self.p_dim, self.q_dim, self.num_layers, self.recon_sigma).to(self.DEVICE)
        self.opt = optim.Adam(list(self.encoder.parameters()) +
                 list(self.decoder.parameters()), lr=self.lr, eps=1e-6, weight_decay=1e-5)


        self.trainloader = torch.utils.data.DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)

    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()
        total_loss = []
        for batch_idx, data in enumerate(self.trainloader):
            data = data[0].to(self.DEVICE)
            self.opt.zero_grad()
            z, mu, logvar = self.encoder(data)
            reg_loss = self.encoder.loss(mu, logvar)
            recon_data = self.decoder(z)
            recon_loss = self.decoder.loss(recon_data, data.view(-1,self.p_dim))
            current_loss = reg_loss + recon_loss
            current_loss.backward()

            total_loss.append(current_loss.item())
            
            self.opt.step()
