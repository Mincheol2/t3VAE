import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import math

from models import baseline

class TVAE(baseline.VAE_Baseline):
    def __init__(self, image_shape, DEVICE, args):
        super(TVAE, self).__init__(image_shape, DEVICE,args)
        self.opt = optim.Adam(list(self.parameters()), lr=args.lr, eps=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma = 0.99)
        self.pdim = self.C * self.H * self.W
        '''
            T-VAE : add three-level layers and parameter layers : mu, lambda, nu
            Although the original code use one-dimension prior for each pixel,
            we use Multivariate Gamma prior instead.
            -> Therefore, we learn "one-dimensional" nu and lambda.
        ''' 
        n_latent = self.decoder_hiddens[-1] * self.H // 2 * self.W // 2 # 32x32x32
        # init parameters
        self.loglambda = 0
        self.lognu = 0

        # parameter layers
        n_h = n_latent//8
        self.parameter_loglambda_layer = nn.Linear(n_latent, 1) # scale params
        self.parameter_lognu_layer = nn.Linear(n_latent, 1) # degree of freedom

        ## One-dim case
        # self.parameter_loglambda_layer = nn.Linear(n_h, self.pdim) # scale params
        # self.parameter_lognu_layer = nn.Linear(n_h, self.pdim) # degree of freedom
    
        self.MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(self.pdim), torch.eye(self.pdim))

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
        recon_x = self.final_layer(z) 
        z = z.flatten(start_dim=1)
        self.loglambda = self.parameter_loglambda_layer(z)
        self.lognu = self.parameter_lognu_layer(z)
        
        ## parameter layers ##

        nu_z = torch.clamp(self.lognu.exp(), min=1)
        lambda_z = torch.clamp(self.loglambda.exp(), min=1e-8)
        eps = self.MVN_dist.sample(sample_shape=torch.tensor([recon_x.shape[0]])).to(self.DEVICE)
        
        chi_dist = torch.distributions.chi2.Chi2(nu_z)
        v = chi_dist.sample().to(self.DEVICE)
        t_sample = eps * torch.sqrt(nu_z / v)
        t_sample = t_sample / torch.sqrt(lambda_z)

        recon_x = recon_x + t_sample.view(-1,self.C,self.H,self.W)

        return recon_x
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # diagonal mat
        eps = torch.randn_like(std) # Normal dist : eps ~ N(0, I)
        return mu + std * eps
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z, mu, logvar
        
    def loss(self, x, recon_x, z, mu, logvar):
        N = x.shape[0]
        reg_loss = self.args.reg_weight * torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1), dim=0)
        
        lambda_z = torch.clamp(self.loglambda.exp(), min=1e-8)
        nu_z = torch.clamp(self.lognu.exp(), min=1e-8)

        lgamma_term = torch.lgamma((nu_z + self.pdim)/2) - torch.lgamma(nu_z/2)
        log_term = self.pdim/2 * (self.loglambda - self.lognu - torch.log(torch.tensor([np.pi]).to(self.DEVICE)))
        lgamma_term = lgamma_term.to(self.DEVICE)

        x_flat = torch.flatten(x,start_dim = 1)
        locale_mu = recon_x.flatten(start_dim=1)

        x_norm_sq = torch.linalg.norm(x_flat-locale_mu, ord=2, dim=1).pow(2).unsqueeze(1)
        log_recon = (nu_z + self.pdim)/2 * torch.log(1 + lambda_z / nu_z * x_norm_sq)

        recon_loss = - torch.mean(lgamma_term + log_term - log_recon)

        # One dim case
        # recon_loss = - torch.mean(torch.mean(lgamma_term + log_term - log_recon,dim=1),dim=0)

        total_loss = reg_loss + recon_loss
        return reg_loss, recon_loss, total_loss

    def generate(self):

        prior_z = torch.randn(64, self.args.qdim)
        prior_z = self.args.recon_sigma * prior_z
        VAE_gen = self.decoder(prior_z.to(self.DEVICE)).detach().cpu()
        VAE_gen = VAE_gen *0.5 + 0.5
        return VAE_gen