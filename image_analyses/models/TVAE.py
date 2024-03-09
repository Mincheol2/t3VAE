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
        
        self.n_dim = self.C * self.H * self.W
        self.m_dim = self.args.m_dim
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

        # # parameter layers
        # self.parameter_loglambda_layer = nn.Linear(self.pdim, 1) # scale params
        # self.parameter_lognu_layer = nn.Linear(self.pdim, 1) # degree of freedom


        # parameter layers for univatriate
        self.parameter_loglambda_layer = nn.Linear(self.pdim, self.args.m_dim) # scale params for each component
        self.parameter_lognu_layer = nn.Linear(self.pdim, self.args.m_dim) # degree of freedom for each component

    
        self.MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(self.pdim), torch.eye(self.pdim))

    def encoder(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def decoder(self, z):
        z = self.linear(z)
        z = z.reshape(-1,self.decoder_hiddens[0],math.ceil(self.H / 2**self.n),math.ceil(self.W / 2**self.n))
        z = self.tp_cnn_layers(z)
        recon_mu = self.final_layer(z)
        
        ## parameter layers ##
        z = z.flatten(start_dim=1)
        self.loglambda = self.parameter_loglambda_layer(recon_mu.reshape(-1,self.pdim)) # [B, 1]
        self.lognu = self.parameter_lognu_layer(recon_mu.reshape(-1,self.pdim))  # [B, 1]
         
        return recon_mu
    
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

        '''
            Why we multiply 2 on the reg_loss?
            
            If we look at the gamma-bound formula: 1/2 * (||x - recon_x ||^2 / recon_sigma**2 + 2 * regularizer), 
            we can see that we omit the constant 1/2 when calculating the total_loss.
            For comparison, we also multlply 2 with other frameworks (except t3VAE)
        '''

        # multiply by 2
        reg_loss = 2 * self.args.beta_weight * torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1), dim=0)
        
        lambda_z = self.loglambda.exp()
        nu_z = self.lognu.exp()
        

        # Multivariate
        # lgamma_term = torch.lgamma((nu_z + self.pdim)/2) - torch.lgamma(nu_z/2)
        # log_term = self.pdim/2 * (self.loglambda - self.lognu - torch.log(torch.tensor([np.pi]).to(self.DEVICE)))
        # lgamma_term = lgamma_term.to(self.DEVICE)

        # x_flat = torch.flatten(x,start_dim = 1)
        # locale_mu = recon_x.flatten(start_dim=1)

        # x_norm_sq = torch.linalg.norm(x_flat-locale_mu, ord=2, dim=1).pow(2).unsqueeze(1)
        # log_recon = (nu_z + self.pdim)/2 * torch.log(1 + lambda_z / nu_z * x_norm_sq)

        # # One dim case
        lgamma_term = torch.lgamma((nu_z +1)/2) - torch.lgamma(nu_z/2)
        log_term = 1/2 * (self.loglambda - self.lognu - torch.log(torch.tensor([np.pi]).to(self.DEVICE)))
        lgamma_term = lgamma_term.to(self.DEVICE)

        x_flat = x.flatten(start_dim = 1)
        locale_mu = recon_x.flatten(start_dim=1)

        x_norm_sq = torch.linalg.norm(x_flat-locale_mu, ord=2, dim=1).pow(2).unsqueeze(1)
        log_recon = (nu_z + 1)/2 * torch.log(1 + lambda_z / nu_z * x_norm_sq)

        # multiply by 2
        recon_loss = - 2 * torch.mean(torch.sum(lgamma_term + log_term - log_recon,dim=1),dim=0)

        total_loss = reg_loss + recon_loss
        return reg_loss, recon_loss, total_loss

    def generate(self, N = 64):
        prior_z = torch.randn(N, self.m_dim)
        prior_z = self.args.prior_sigma * prior_z
        TVAE_gen = self.decoder(prior_z.to(self.DEVICE)).detach().cpu()
        return TVAE_gen