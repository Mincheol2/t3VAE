import numpy as np
import torch
import torch.nn as nn
import math

from models import VAE

class HVAE(VAE.VAE):
    def __init__(self, image_shape, DEVICE, args):
        super(HVAE, self).__init__(image_shape, DEVICE, args)
        
        self.L = 2 # L-level HVAE model

        # Hierarchical latent layers.
        self.mu_layers = []
        self.logvar_layers = []
        self.prior_layers = []
        self.h_latent_dim_list = []

        
        self.m1 = args.m_dim
        self.m2 = self.m1 // 2
        # For simplicity, fix all recon_sigma to 1.
        input_dim = self.cnn_lineardim

        # Construct ith MLP layer
        h_latent_dim = args.m_dim
        for i in range(self.L):
            self.h_latent_dim_list.append(h_latent_dim)
            self.mu_layers.append(nn.Sequential(nn.Linear(input_dim, h_latent_dim),
                                  nn.ReLU()).to(self.DEVICE))
            self.logvar_layers.append(nn.Sequential(nn.Linear(input_dim, h_latent_dim),
                                    nn.ReLU()).to(self.DEVICE))
            if i > 0:
                self.prior_layers.append(nn.Sequential(nn.Linear(input_dim - self.cnn_lineardim, h_latent_dim),
                                    nn.ReLU()).to(self.DEVICE))

            input_dim += h_latent_dim
            h_latent_dim = h_latent_dim // 2 if i % 2 == 0 else h_latent_dim # e.g.) 64 -> 32 -> 32 -> 16

        # decoder layer
        decoder_dim = input_dim - self.cnn_lineardim
        self.linear = nn.Sequential(
                        nn.Linear(decoder_dim, self.cnn_lineardim),
                        nn.ReLU(),
        )
        
        # define samplers for t
        self.MVN_dists = []
        for i in range(2):
            MVN_dim = self.h_latent_dim_list[i]
            self.MVN_dists.append(torch.distributions.MultivariateNormal(torch.zeros(MVN_dim), torch.eye(MVN_dim)))

    def encoder(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)
        mu_list = []
        logvar_list = []
        z_list = []

        for L in range(1,3):
            input_x = x if L == 1 else torch.concat([z, x], dim=1)
            mu = self.mu_layers[L-1](input_x)
            logvar = self.logvar_layers[L-1](input_x)
            z = self.reparameterize(mu, logvar)
            mu_list.append(mu)
            logvar_list.append(logvar)
            z_list.append(z)

        return z_list, mu_list, logvar_list
        
    def decoder(self, z_list):
        z = torch.concat(z_list, dim=1).to(self.DEVICE)
        z = self.linear(z)
        z = z.reshape(-1,self.decoder_hiddens[0],math.ceil(self.H / 2**self.n),math.ceil(self.W / 2**self.n))
        z = self.tp_cnn_layers(z)
        z = self.final_layer(z)
        return z
    

    def forward(self, x):
        z_list, mu_list, logvar_list = self.encoder(x)
        x_recon = self.decoder(z_list)
        return x_recon, z_list, mu_list, logvar_list

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # diagonal mat
        eps = torch.randn_like(std) # Normal dist : eps ~ N(0, I)
        return mu + std * eps

    def loss(self, x, recon_x, z_list, mu_list, logvar_list):
        N = x.shape[0]

        ## gamma regularizers ##
        
        
        # regularizer for q(z_1|x)
        reg_loss = 2 * self.args.beta_weight * torch.mean(-0.5 * torch.sum(1 + logvar_list[0] - mu_list[0].pow(2) - logvar_list[0].exp(),dim=1), dim=0)

        # regularizer for q(z_2 |x, z_1)

        prior_z2 = self.prior_layers[0](z_list[0]) #p(z2|z1)
        reg_loss2 = 2 * self.args.beta_weight * torch.mean(-0.5 * torch.sum(1 + logvar_list[1] - (mu_list[1] - prior_z2).pow(2) - logvar_list[1].exp(),dim=1), dim=0)


        ## recon loss (same as VAE) ##
        recon_loss = torch.sum((recon_x - x)**2 / N)
        total_loss = recon_loss + reg_loss + reg_loss2
        return reg_loss, reg_loss2, recon_loss, total_loss

    def generate(self, N = 64):
        '''
        There are two alternative t-priors to generate.
        '''

        z_samples = []
        # L = 1
        prior_z =  torch.randn(N, self.m1).to(self.DEVICE)
        z_samples.append(prior_z)
        
        mu_1 = self.prior_layers[0](prior_z)
        prior_z2 =  torch.randn(N, self.m2).to(self.DEVICE) + mu_1
        z_samples.append(prior_z2)

        imgs = self.decoder(z_samples).detach().cpu()

        return imgs