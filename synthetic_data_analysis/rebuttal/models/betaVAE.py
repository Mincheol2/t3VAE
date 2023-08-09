import torch
import torch.nn as nn
from torch.nn import functional as F

class betaVAE(nn.Module) : 
    def __init__(self, n_dim=1, m_dim=1, nu=3, recon_sigma=1, reg_weight=1, num_layers=64, device='cpu'):
        super(betaVAE, self).__init__()
        self.model_name = f"betaVAE_{reg_weight}"

        self.n_dim = n_dim
        self.m_dim = m_dim
        self.recon_sigma = recon_sigma
        self.reg_weight = reg_weight
        self.num_layers = num_layers
        self.device = device

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
    
    def recon_loss(self, x, recon_x) : 
        return F.mse_loss(recon_x, x, reduction = 'none').sum(dim = 1).mean(dim = 0) / self.recon_sigma**2

    def reg_loss(self, mu, logvar) : 
        # return KL regularizer including constant term
        return torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1))
    
    def total_loss(self, x, recon_x, mu, logvar) : 
        recon = self.recon_loss(x, recon_x)
        reg = self.reg_loss(mu, logvar)

        return recon, reg, recon + self.reg_weight * reg
    
    def decoder_sampling(self, z) : 
        mu_theta = self.decode(z)
        eps = torch.randn_like(mu_theta)
        return mu_theta + self.recon_sigma * eps

    def generate(self, N = 1000) : 
        prior = torch.randn(N, self.m_dim).to(self.device)

        return self.decoder_sampling(prior)
    
    def reconstruct(self, x) : 
        return self.decoder_sampling(self.encode(x)[0])

    def forward(self, x, verbose = False) : 
        enc_z, mu, logvar = self.encode(x)
        recon_x = self.decode(enc_z)
        total_loss = self.total_loss(x, recon_x, mu, logvar)
        if verbose is False : 
            return total_loss
        else : 
            return [
                total_loss, 
                mu.detach().flatten().cpu().numpy(), 
                logvar.detach().flatten().exp().cpu().numpy(), 
                total_loss[0] + total_loss[1]
            ]
        

        

