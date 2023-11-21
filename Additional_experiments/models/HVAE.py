import numpy as np
import torch
import torch.nn as nn
import math

from models import VAE

class HVAE(VAE.VAE):
    def __init__(self, image_shape, DEVICE, args):
        super(HVAE, self).__init__(image_shape, DEVICE, args)
        self.m1 = args.m_dim
        self.m2 = self.m1 // 2
        self.h_latent_dim_list = [self.m1, self.m2]
        self.L = 2 # L-level HVAE model

        input_dim = self.cnn_lineardim + self.m1
        
        self.mu_layer2 = nn.Linear(input_dim, self.m2).to(self.DEVICE)
        self.logvar_layer2 = nn.Linear(input_dim, self.m2).to(self.DEVICE)
        self.priormu_layer = nn.Linear(self.m1,self.m2).to(self.DEVICE)


        # decoder feature layer
        self.linear = nn.Sequential(
                        nn.Linear(self.m1 +  self.m2, self.cnn_lineardim),
                        nn.ReLU()
        ).to(self.DEVICE)

        # decoder feature layer
        self.linear = nn.Sequential(
                        nn.Linear(self.m1 +  self.m2, self.cnn_lineardim),
                        nn.ReLU()
        ).to(self.DEVICE)
        
    def encoder(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)

        # L=1
        mu1 = self.mu_layer(x)
        logvar1 = self.logvar_layer(x)
        z1 = self.reparameterize(mu1, logvar1)

        # L=2 
        input_x = torch.cat([z1, x], dim=1).to(self.DEVICE)
        
        mu2 = self.mu_layer2(input_x)
        logvar2 = self.logvar_layer2(input_x)
        z2 = self.reparameterize(mu2, logvar2)

        return z1, z2, mu1, mu2, logvar1, logvar2
        
    def decoder(self, z1, z2):
        z = torch.cat([z1, z2], dim=1).to(self.DEVICE)
        z = self.linear(z)
        z = z.reshape(-1,self.decoder_hiddens[0],math.ceil(self.H / 2**self.n),math.ceil(self.W / 2**self.n))
        z = self.tp_cnn_layers(z)
        z = self.final_layer(z)
        return z
    

    def forward(self, x):
        z1, z2, mu1, mu2, logvar1, logvar2 = self.encoder(x)
        recon_x  = self.decoder(z1,z2)
        return recon_x ,z1, z2, mu1, mu2, logvar1, logvar2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # diagonal mat
        eps = torch.randn_like(std) # Normal dist : eps ~ N(0, I)
        return mu + std * eps

    def loss(self, x, recon_x, z1, mu1, mu2, logvar1, logvar2):
        N = x.shape[0]

        # regularizer for q(z_1|x)
        reg_loss = 2 * self.args.beta_weight * torch.mean(-0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(),dim=1), dim=0)

        # regularizer for q(z_2 |x, z_1)

        prior_z2 = self.priormu_layer(z1) #p(z2|z1)
        reg_loss2 = 2 * self.args.beta_weight * torch.mean(-0.5 * torch.sum(1 + logvar2 - (mu2 - prior_z2).pow(2) - logvar2.exp(),dim=1), dim=0)


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


        mu_1 = self.priormu_layer(prior_z)
        prior_z2 =  torch.randn(N, self.m2).to(self.DEVICE) + mu_1


        imgs = self.decoder(prior_z, prior_z2).detach().cpu()

        return imgs