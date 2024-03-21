import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from loss import log_t_normalizing_const

class Disentangled_VAE(nn.Module) : 
    def __init__(self, n_dim=1, m_dim=1, nu=3, recon_sigma=1, reg_weight=1, num_layers=64, device='cpu', sample_size_for_integral = 5):
        super(Disentangled_VAE, self).__init__()
        self.model_name = f"Disentangled_VAE_nu_{nu}"

        self.n_dim = n_dim
        self.m_dim = m_dim
        self.nu = nu
        self.recon_sigma = recon_sigma
        self.reg_weight = reg_weight
        self.num_layers = num_layers
        self.device = device

        self.sample_size_for_integral = sample_size_for_integral

        # define encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_dim, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, num_layers), 
            nn.LeakyReLU()
        )
        self.latent_mu = nn.Linear(num_layers, m_dim)
        self.latent_logvar = nn.Linear(num_layers, m_dim)

        # define decoder
        self.decoder = nn.Sequential(
            nn.Linear(m_dim, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, n_dim)
        )

    def encoder_reparameterize(self, mu, logvar) : 
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def encode(self, x) : 
        x = self.encoder(x)
        mu = self.latent_mu(x)
        logvar = self.latent_logvar(x)
        z = self.encoder_reparameterize(mu, logvar)

        return z, mu, logvar
    
    def decode(self, z) : 
        return self.decoder(z)
    
    def recon_loss(self, x, mu_theta) : 
        return F.mse_loss(mu_theta, x, reduction = 'none').sum(dim = 1).mean(dim = 0) / self.recon_sigma**2

    def reg_loss(self, enc_z, logvar) : 
        # return 2 * KL regularizer
        reg = -torch.sum(np.log(2 * np.pi) + 1 + logvar, dim = 1)
        reg -= 2 * self.m_dim * log_t_normalizing_const(self.nu, 1)
        reg += (self.nu + 1) * torch.sum(torch.log(1 + enc_z.pow(2) / self.nu), dim = 1)
        return torch.mean(reg)
    
    def total_loss(self, x, enc_z, mu_theta, logvar) : 
        recon = self.recon_loss(x, mu_theta)
        reg = self.reg_loss(enc_z, logvar)

        return recon, reg, recon + self.reg_weight * reg
    
    def decoder_sampling(self, z) : 
        mu_theta = self.decode(z)
        eps = torch.randn_like(mu_theta)
        return mu_theta + self.recon_sigma * eps

    def generate(self, N = 1000) : 
        prior = torch.randn(N, self.m_dim).to(self.device)

        chi_dist = torch.distributions.chi2.Chi2(self.nu)
        v = chi_dist.sample(sample_shape=prior.shape).to(self.device)

        prior *= torch.sqrt(self.nu / v)

        return self.decoder_sampling(prior)
    
    def reconstruct(self, x) : 
        return self.decoder_sampling(self.encode(x)[0])

    def forward(self, x) : 
        mean_recon = 0
        mean_reg = 0
        mean_total = 0
        
        for k in range(self.sample_size_for_integral) : 
            enc_z, _, logvar = self.encode(x)
            mu_theta = self.decode(enc_z)
            recon, reg, total = self.total_loss(x, enc_z, mu_theta, logvar)
            mean_recon += recon / self.sample_size_for_integral
            mean_reg += reg / self.sample_size_for_integral
            mean_total += total / self.sample_size_for_integral
        
        return mean_recon, mean_reg, mean_total
        

