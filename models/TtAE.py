import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.nn import functional as F
import math

from models import baseline

class TtAE(baseline.VAE_Baseline):
    def __init__(self, image_shape, DEVICE, args):
        super(TtAE, self).__init__(image_shape, DEVICE, args)
        self.opt = optim.Adam(list(self.parameters()), lr=self.args.lr, eps=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma = 0.99)
        self.pdim = self.C * self.H * self.W
        self.qdim = self.args.qdim
            
        self.gamma = -2 / (self.args.nu + self.pdim + self.qdim)

        '''
            normalizing constant for t-distribution
        '''
        def log_t_normalizing_const(nu, d):
            nom = torch.lgamma(torch.tensor((nu+d)/2))
            denom = torch.lgamma(torch.tensor(nu/2)) + d/2 * (np.log(nu) + np.log(np.pi))
            return nom - denom

        log_tau_base = -self.pdim * np.log(self.args.recon_sigma) + log_t_normalizing_const(self.args.nu,self.pdim)
        log_tau_base += - np.log(self.args.nu + self.pdim - 2) + np.log(self.args.nu-2)
        
        const_2bar1_term_1 = (1 + self.qdim / (self.args.nu + self.pdim -2))
        const_2bar1_term_2_log = -self.gamma / (1+self.gamma) * log_tau_base

        self.const_2bar1 = const_2bar1_term_1 * const_2bar1_term_2_log.exp()
        log_tau = 2 / (self.args.nu + self.pdim - 2) * log_tau_base
        self.tau = log_tau.exp()
        
    
    def encoder(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)
        mu = self.mu_layer(x)
        logvar = self.mu_layer(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def decoder(self, z):
        z = self.linear(z)
        z = z.reshape(-1,self.decoder_hiddens[0],math.ceil(self.H / 2**self.n),math.ceil(self.W / 2**self.n))
        z = self.tp_cnn_layers(z)
        z = self.final_layer(z)
        return z
    
    def reparameterize(self, mu, logvar):
        '''
            Sampling algorithm
            Let nu_prime = nu + pdim
            1. Generate v ~ chiq(nu_prime) and eps ~ N(0, I), independently.
            2. Caculate x = mu + std * eps / (sqrt(v/nu_prime)), where std = sqrt(nu/(nu_prime) * var)
        '''
        nu_prime = self.args.nu + self.pdim
        MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(self.args.qdim), torch.eye(self.args.qdim))
        chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu_prime]))
        
        # Student T dist : [B, z_dim]
        eps = MVN_dist.sample(sample_shape=torch.tensor([mu.shape[0]])).to(self.DEVICE)
        std = torch.exp(0.5 * logvar)
        std = np.sqrt(self.args.nu / nu_prime) * std
        v = chi_dist.sample().to(self.DEVICE)
        return mu + std * eps * torch.sqrt(nu_prime / v)

    
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, mu, logvar


    def loss(self, x, recon_x, z, mu, logvar):
        N = x.shape[0]
        x = x.view(N,-1)
        recon_x = recon_x.view(N,-1)

        ## gamma regularizer ##
        mu_norm_sq = torch.linalg.norm(mu, ord=2, dim=1).pow(2)
        trace_var = self.args.nu / (self.args.nu + self.pdim - 2) * torch.sum(logvar.exp(),dim=1)
        log_det_var = -self.gamma / (2+2*self.gamma) * torch.sum(logvar,dim=1)
        reg_loss = torch.mean(mu_norm_sq + trace_var - self.args.nu * self.const_2bar1 * log_det_var.exp() + self.args.nu * self.tau, dim=0)

        ## recon loss (same as VAE) ##
        recon_loss = F.mse_loss(recon_x, x) / self.args.recon_sigma**2
        total_loss = self.args.reg_weight * reg_loss + recon_loss

        return reg_loss, recon_loss, total_loss

    def generate(self):
        MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(self.args.qdim), torch.eye(self.args.qdim))
        chi_dist = torch.distributions.chi2.Chi2(torch.tensor([self.args.nu]))
        prior_z = MVN_dist.sample(sample_shape=torch.tensor([64])).to(self.DEVICE)
        v = chi_dist.sample().to(self.DEVICE)
        prior_t = self.args.recon_sigma * prior_z * torch.sqrt(self.args.nu / v)
        imgs = self.decoder(prior_t.to(self.DEVICE)).detach().cpu()
        imgs = imgs*0.5 + 0.5
        return imgs