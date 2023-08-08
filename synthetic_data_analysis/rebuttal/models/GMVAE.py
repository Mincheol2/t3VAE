import torch
import torch.nn as nn
from torch.nn import functional as F

class GMVAE(nn.Module) : 
    def __init__(self, l_dim = 1, n_dim=1, m_dim=1, K = 2, nu=3, recon_sigma=1, reg_weight=1, num_layers=64, device='cpu'):
        '''
        In GMVAE, w, z, x are latent variables and y is the data. 
        w ~ N_{l_dim} (0, I)
        z ~ Multinomial (1/K, ..., 1/K)
        x | w, z ~ N_{m_dim} (mu_z(w), Sigma_z(w))
        y | x ~ N_{n_dim} (mu_theta(x), sigma^2 I)
        '''
        super(GMVAE, self).__init__()
        self.model_name = "GMVAE"

        self.l_dim = l_dim
        self.n_dim = n_dim
        self.m_dim = m_dim
        self.K_dim = K
        self.recon_sigma = recon_sigma
        self.reg_weight = reg_weight
        self.num_layers = num_layers
        self.device = device

        # y -> w
        self.w_encoder = nn.Sequential(
            nn.Linear(n_dim, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, num_layers), 
            nn.LeakyReLU()
        )
        self.w_encoder_mu = nn.Linear(num_layers, l_dim)
        self.w_encoder_var = nn.Linear(num_layers, l_dim)

        # y -> x
        self.x_encoder = nn.Sequential(
            nn.Linear(n_dim, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, num_layers), 
            nn.LeakyReLU()
        )
        self.x_encoder_mu = nn.Linear(num_layers, m_dim)
        self.x_encoder_encoder_var = nn.Linear(num_layers, m_dim)

        # w, z -> x
        self.x_decoder_list = [
            nn.Sequential(
                nn.Linear(l_dim, num_layers), 
                nn.LeakyReLU(), 
                nn.Linear(num_layers, num_layers), 
                nn.LeakyReLU()
            ) for _ in range(K)
        ]
        self.x_decoder_mu_list = [nn.Linear(num_layers, m_dim) for _ in range(K)]
        self.x_decoder_logvar_list = [nn.Linear(num_layers, m_dim) for _ in range(K)]

        # x -> y
        self.y_decoder = nn.Sequential(
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
    
    def w_encode(self, y) : 
        y = self.w_encoder(y)
        w_mu = self.w_encoder_mu(y)
        w_logvar = self.w_encoder_logvar(y)
        enc_w = self.encoder_reparameterize(w_mu, w_logvar)
        return enc_w, w_mu, w_logvar

    def x_encode(self, y) : 
        y = self.x_decoder_list_encoder(y)
        x_mu = self.x_encoder_mu(y)
        x_logvar = self.x_encoder_logvar(y)
        enc_x = self.encoder_reparameterize(x_mu, x_logvar)
        return enc_x, x_mu, x_logvar
    
    def x_decode(self, w, z) : 
        return self.decoder(z)
    
    def recon_loss(self, x, recon_x) : 
        return F.mse_loss(recon_x, x, reduction = 'none').sum(dim = 1).mean(dim = 0) / self.recon_sigma**2

    def reg_loss(self, mu, logvar) : 
        # return KL regularizer including constant term
        return torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1))
    
    def total_loss(self, x, recon_x, mu, logvar) : 
        recon = self.recon_loss(recon_x, x)
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

    def forward(self, x) : 
        enc_z, mu, logvar = self.encode(x)
        recon_x = self.decode(enc_z)
        return self.total_loss(x, recon_x, mu, logvar)
        

        

