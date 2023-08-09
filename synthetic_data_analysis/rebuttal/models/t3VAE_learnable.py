import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from loss import log_t_normalizing_const, gamma_regularizer

class t3VAE_learnable(nn.Module) : 
    
    def __init__(self, n_dim=1, m_dim=1, nu=3, recon_sigma=1, reg_weight=1, num_layers=64, device='cpu'):
        super(t3VAE_learnable, self).__init__()
        self.model_name = f"t3VAE_learnable_nu_{nu}"

        self.n_dim = n_dim
        self.m_dim = m_dim
        self.nu = nu
        self.reg_weight = reg_weight
        self.num_layers = num_layers
        self.device = device

        self.gamma = -2 / (self.nu + self.n_dim + self.m_dim)
        log_tau_base = -n_dim * np.log(recon_sigma) + log_t_normalizing_const(nu, n_dim) - np.log(nu + n_dim - 2) + np.log(nu-2)
        
        const_2bar1_term_1 = (1 + m_dim / (nu + n_dim -2))
        const_2bar1_term_2_log = -self.gamma / (1+self.gamma) * log_tau_base
        self.const_2bar1 = const_2bar1_term_1 * const_2bar1_term_2_log.exp()
        
        log_tau = 2 / (nu + n_dim - 2 ) * log_tau_base
        self.tau = log_tau.exp()

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
            nn.LeakyReLU()
        )
        self.decoder_mu = nn.Linear(num_layers, n_dim)
        self.decoder_logvar = nn.Linear(num_layers, n_dim)

    def encoder_reparameterize(self, mu, logvar) : 
        N_sample = mu.shape[0]
        nu_n = self.nu + self.n_dim

        std = np.sqrt(self.nu / nu_n) * torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu_n]))
        v = chi_dist.sample(torch.tensor([N_sample])).to(self.device)

        return mu + std * eps * torch.sqrt(nu_n / v)
    
    def encode(self, x) : 
        x = self.encoder(x)
        mu = self.latent_mu(x)
        logvar = self.latent_logvar(x)
        z = self.encoder_reparameterize(mu, logvar)

        return z, mu, logvar
    
    def decode(self, z) : 
        z = self.decoder(z)
        mu_theta = self.decoder_mu(z)
        logvar_theta = self.decoder_logvar(z)
        return mu_theta, logvar_theta
    
    def recon_loss(self, x, mu_theta, logvar_theta) : 
        reg = torch.sum(F.mse_loss(mu_theta, x, reduction = 'none') / torch.exp(logvar_theta), dim = 1)
        reg += torch.sum(logvar_theta, dim = 1)
        return torch.mean(reg)

    def reg_loss(self, mu_phi, logvar_phi) : 
        # return gamma regularizer including constant term
        return gamma_regularizer(mu_phi, logvar_phi, self.n_dim, self.const_2bar1, self.gamma, self.tau, self.nu)
    
    def total_loss(self, x, mu_theta, logvar_theta, mu_phi, logvar_phi) : 
        recon = self.recon_loss(x, mu_theta, logvar_theta)
        reg = self.reg_loss(mu_phi, logvar_phi)

        return recon, reg, recon + self.reg_weight * reg
    
    def decoder_sampling(self, z) : 
        N_sample = z.shape[0]
        mu_theta, logvar_theta = self.decode(z)
        eps = torch.randn_like(mu_theta)

        nu_m = self.nu + self.m_dim
        chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu_m]))
        v = chi_dist.sample(sample_shape=torch.tensor([N_sample])).to(self.device)

        std = torch.exp(0.5 * logvar_theta) * torch.sqrt((self.nu + torch.norm(z,dim=1).pow(2)) / nu_m).unsqueeze(1).to(self.device)
        return mu_theta + std * (eps * torch.sqrt(nu_m / v))

    def generate(self, N = 1000) : 
        prior = torch.randn(N, self.m_dim).to(self.device)

        chi_dist = torch.distributions.chi2.Chi2(torch.tensor([self.nu]))
        v = chi_dist.sample(sample_shape=torch.tensor([N])).to(self.device)

        prior *= torch.sqrt(self.nu / v)

        return self.decoder_sampling(prior)
    
    def reconstruct(self, x) : 
        return self.decoder_sampling(self.encode(x)[0])

    def forward(self, x) : 
        enc_z, mu_phi, logvar_phi = self.encode(x)
        mu_theta, logvar_theta = self.decode(enc_z)
        return self.total_loss(x, mu_theta, logvar_theta, mu_phi, logvar_phi)
        

        

