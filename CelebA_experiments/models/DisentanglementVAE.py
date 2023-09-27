import torch
import math
import numpy as np
from models import baseline

class DisentanglementVAE(baseline.VAE_Baseline):
    def __init__(self, image_shape, DEVICE, args):
        super(DisentanglementVAE, self).__init__(image_shape, DEVICE,args)
        
    def log_t_normalizing_const(self, nu, d):
        nom = torch.lgamma(torch.tensor((nu+d)/2))
        denom = torch.lgamma(torch.tensor(nu/2)) + d/2 * (np.log(nu) + np.log(np.pi))
        return nom - denom

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
        
        '''
            Computes E_{p(x)}[ELBO_{\alpha,\beta}]
            (According to the original paper, alpha = 0 in this experiment.)
        '''

        '''
            Why we multiply 2 on the reg_loss?
            
            If we look at the gamma-bound formula: 1/2 * (||x - recon_x ||^2 / recon_sigma**2 + 2 * regularizer), 
            we can see that we omit the constant 1/2 when calculating the total_loss.
            For comparison, we also multlply 2 with other frameworks (except t3VAE)
        '''
        log_const = self.log_t_normalizing_const(self.args.nu, 1)
        total_reg = 0
        total_recon = 0
        for _ in range(self.args.int_K):
            z, _, logvar = self.encoder(x)
            reg_loss = -torch.sum(np.log(2 * np.pi) + 1 + logvar, dim = 1) # Entropy

            # KL (q_zx || p_z)
            reg_loss -= 2 * self.args.m_dim * log_const # multiply by 2
            reg_loss += (self.args.nu + 1) * torch.sum(torch.log(1 + z.pow(2) / self.args.nu), dim = 1)
            recon_loss = torch.sum((recon_x - x)**2 / (N * self.args.prior_sigma**2))
            total_reg += torch.mean(reg_loss)
            total_recon += recon_loss
        reg_loss = total_reg / self.args.int_K
        recon_loss = total_recon / self.args.int_K
        total_loss = recon_loss + reg_loss

        return [reg_loss, recon_loss, total_loss]

    def generate(self):
        prior_z = torch.randn(64, self.args.m_dim)
        prior_z = self.args.prior_sigma * prior_z
        VAE_gen = self.decoder(prior_z.to(self.DEVICE)).detach().cpu()
        VAE_gen = VAE_gen
        return VAE_gen