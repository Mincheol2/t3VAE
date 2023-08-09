import torch
import torch.nn as nn
from torch.nn import functional as F

class VAE_st(nn.Module) : 
    def __init__(self, n_dim=1, m_dim=1, nu=3, recon_sigma=1, reg_weight=1, num_layers=64, device='cpu', sample_size_for_integral = 5):
        super(VAE_st, self).__init__()
        self.model_name = "VAE-st"

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
        self.latent_var = nn.Sequential(
            nn.Linear(num_layers, m_dim), 
            nn.Softplus()
        )
        self.latent_nu = nn.Sequential(
            nn.Linear(num_layers, 1), 
            nn.Softplus()
        )

        # define decoder
        self.decoder = nn.Sequential(
            nn.Linear(m_dim, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, n_dim)
        )

    def encoder_reparameterize(self, mu, var, nu) : 
        std = torch.sqrt(var)
        eps = torch.randn_like(std)

        chi_dist = torch.distributions.chi2.Chi2(nu)
        v = chi_dist.sample(sample_shape=torch.tensor([1])).squeeze(0).to(self.device)
        # print(f'std shape : {std.shape}')
        # print(f'eps shape : {eps.shape}')
        # print(f'nu shape : {nu.shape}')
        # print(f'v shape : {v.shape}')

        '''
        Note that if we use *.sample method for v, it does not update the gradient for nu. 
        However, even if we use *.rsample method, I have no idea about whether it gives us right gradient for nu or not. 
        I will check later...
        '''
        return mu + std * eps * torch.sqrt(nu / v).unsqueeze(1)
    
    def encode(self, x) : 
        x = self.encoder(x)
        mu = self.latent_mu(x)
        var = self.latent_var(x)
        nu = self.latent_nu(x).flatten()
        z = self.encoder_reparameterize(mu, var, nu)

        return z, mu, var, nu
    
    def decode(self, z) : 
        return self.decoder(z)
    
    def recon_loss(self, x, mu_theta) : 
        return F.mse_loss(mu_theta, x, reduction = 'none').sum(dim = 1).mean(dim = 0) / self.recon_sigma**2

    def reg_loss(self, enc_z, mu, var, nu) : 
        # return 2 * KL regularizer including constant term
        logvar = torch.log(var)
        reg = -torch.sum(logvar, dim=1)
        reg -= (nu + self.m_dim) * torch.log(nu + torch.sum((enc_z - mu).pow(2) / var, dim = 1))
        reg += (nu + self.m_dim) * torch.log(nu + torch.sum(enc_z.pow(2), dim = 1))

        return torch.mean(reg)
    
    def total_loss(self, x, enc_z, mu_theta, mu, var, nu) : 
        recon = self.recon_loss(x, mu_theta)
        reg = self.reg_loss(enc_z, mu, var, nu)

        return recon, reg, recon + self.reg_weight * reg
    
    def decoder_sampling(self, z) : 
        mu_theta = self.decode(z)
        eps = torch.randn_like(mu_theta)
        return mu_theta + self.recon_sigma * eps

    def generate(self, N = 1000, nu = 5) : 
        prior = torch.randn(N, self.m_dim).to(self.device)

        chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu]))
        v = chi_dist.sample(sample_shape=torch.tensor([N])).to(self.device)

        prior*= torch.sqrt(v / self.nu)

        return self.decoder_sampling(prior)
    
    def reconstruct(self, x) : 
        return self.decoder_sampling(self.encode(x)[0])

    def forward(self, x) : 
        mean_recon = 0
        mean_reg = 0
        mean_total = 0
        
        for k in range(self.sample_size_for_integral) : 
            enc_z, mu, var, nu = self.encode(x)
            # print(f'mu_theta shape : {enc_z.shape}')
            mu_theta = self.decode(enc_z)
            recon, reg, total = self.total_loss(x, enc_z, mu_theta, mu, var, nu)
            mean_recon += recon / self.sample_size_for_integral
            mean_reg += reg / self.sample_size_for_integral
            mean_total += total / self.sample_size_for_integral

        return mean_recon, mean_reg, mean_total


        

        

