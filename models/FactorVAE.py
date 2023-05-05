import torch
import torch.nn as nn
from . import FactorVAE_Discriminator
from torch.nn import functional as F
import math

from models import baseline

class FactorVAE(baseline.VAE_Baseline):
    def __init__(self, image_shape, DEVICE, args):
        super(FactorVAE, self).__init__(image_shape, DEVICE,args)
        self.discriminator = FactorVAE_Discriminator.Discriminator(q_dim=args.qdim).to(DEVICE)

        # Discriminator network for the Total Correlation loss
        self.D_z_reserve = None
        self.gamma = args.TC_gamma
           
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
        

    def permute_z(self, z):
        permuted_z = torch.zeros_like(z)
        for i in range(z.shape[-1]):
            perms = torch.randperm(z.shape[0]).to(z.device)
            permuted_z[:, i] = z[perms, i]

        return permuted_z
    @torch.enable_grad()
    def loss(self, x, recon_x, z, mu, logvar):
        N = x.shape[0]
        # Update the Factor VAE
        reg_loss = 2 * self.args.reg_weight * torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1), dim=0)
        recon_loss = torch.sum((recon_x - x)**2 / (N * self.args.recon_sigma**2))
    
        self.D_z_reserve = self.discriminator(z)
        vae_tc_loss = (self.D_z_reserve[:, 0] - self.D_z_reserve[:, 1]).mean()

        total_loss = self.args.reg_weight * reg_loss + recon_loss + self.gamma * vae_tc_loss

        return [reg_loss, recon_loss, total_loss, vae_tc_loss]

    
    def TC_loss(self,z):
        ## TC loss for Discriminator ##
        z_perm = self.permute_z(z)
        D_z_perm = self.discriminator(z_perm)
        N = z.shape[0]
        true_labels = torch.ones(N, dtype= torch.long,
                                    requires_grad=False).to(self.DEVICE)
        false_labels = torch.zeros(N, dtype= torch.long,
                                    requires_grad=False).to(self.DEVICE)
        return 0.5 * (F.cross_entropy(self.D_z_reserve, false_labels) + F.cross_entropy(D_z_perm, true_labels)).mean()


    def generate(self):
        prior_z = torch.randn(144, self.args.qdim)
        prior_z = self.args.recon_sigma * prior_z
        VAE_gen = self.decoder(prior_z.to(self.DEVICE)).detach().cpu()
        VAE_gen = VAE_gen
        return VAE_gen
