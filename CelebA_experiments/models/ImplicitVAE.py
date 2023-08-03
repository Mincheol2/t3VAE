import torch
import math

from models import baseline
from . import Implicit_prior_Discriminator

class ImplicitVAE(baseline.VAE_Baseline):
    def __init__(self, image_shape, DEVICE, args):
        super(ImplicitVAE, self).__init__(image_shape, DEVICE,args)

        self.discriminator = Implicit_prior_Discriminator.Discriminator(m_dim=args.m_dim,n_h=500).to(DEVICE)

           
    def encoder(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        z = self.reparameterize(mu, logvar)
        return [z, mu, logvar]
    
    def decoder(self, z):
        z = self.linear(z)
        z = z.reshape(-1,self.decoder_hiddens[0],math.ceil(self.H / 2**self.n),math.ceil(self.W / 2**self.n))
        z = self.tp_cnn_layers(z)
        z = self.final_layer(z)
        return z
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # diagonal mat
        eps = torch.randn_like(std) # Normal dist : eps ~ N(0, I)
        return mu + std * eps
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return [x_recon, z, mu, logvar]
        
    @torch.enable_grad()
    def loss(self, x, recon_x, z, mu, logvar):
        N = x.shape[0]
        
        ## Estimate density ratio ##
        density_ratio = Implicit_prior_Discriminator.discriminator(z)

        reg_loss = 2 * self.args.beta_weight * torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1) - density_ratio, dim=0)
        recon_loss = torch.sum((recon_x - x)**2 / (N * self.args.prior_sigma**2))
        total_loss = reg_loss + recon_loss
        return [reg_loss, recon_loss, total_loss]

    def generate(self):
        prior_z = torch.randn(64, self.args.m_dim)
        prior_z = self.args.prior_sigma * prior_z
        VAE_gen = self.decoder(prior_z.to(self.DEVICE)).detach().cpu()
        VAE_gen = VAE_gen
        return VAE_gen