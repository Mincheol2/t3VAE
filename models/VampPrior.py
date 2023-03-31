import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import math

from models import baseline

class VampPrior(baseline.VAE_Baseline):
    def __init__(self, image_shape, DEVICE, args):
        super(VampPrior, self).__init__(image_shape, DEVICE,args)
        self.opt = optim.Adam(list(self.parameters()), lr=self.args.lr, eps=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma = 0.99)
        self.pdim = self.C * self.H * self.W
        
        # Vampprior : use pseudo input layer
        
        # deterministic input
        self.idle_input = torch.eye(self.args.num_components, requires_grad= False).to(self.DEVICE)
        self.pseudo_input_layer = nn.Sequential(nn.Linear(self.args.num_components, self.pdim),
                                          nn.Hardtanh(min_val=0.0, max_val=1.0)
                                          )
        
        # the default mean and std value initialization in the VampPrior's GitHub code
        torch.nn.init.normal_(self.pseudo_input_layer[0].weight, mean=-0.05, std=0.01)

    
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
        std = torch.exp(0.5 * logvar) # diagonal mat
        eps = torch.randn_like(std) # Normal dist : eps ~ N(0, I)
        return mu + std * eps
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, mu, logvar
        
    def loss(self, x, recon_x, z, mu, logvar):
        prior_mu, prior_logvar = self.make_vampprior()
        prior_var = prior_logvar.exp()
        E_log_q = torch.mean(torch.sum(-0.5 * (logvar + (z - mu) ** 2 / logvar.exp()), dim = 1), dim = 0)

        z = z.unsqueeze(dim=1)
        E_log_p = torch.sum(-0.5 * (prior_logvar + (z - prior_mu) ** 2/ prior_var),
                              dim = 2) - torch.tensor(np.log(self.args.num_components)).float()
        E_log_p = torch.logsumexp(E_log_p, dim = 1)
        
        reg_loss = torch.mean(E_log_q, dim=0) - torch.mean(E_log_p, dim=0)

        recon_loss = F.mse_loss(recon_x, x) / self.args.recon_sigma**2
        total_loss = reg_loss * self.args.reg_weight + recon_loss
        
        return reg_loss, recon_loss, total_loss

    def make_vampprior(self):
        z2 = self.pseudo_input_layer(self.idle_input)
        z2 = z2.view(-1, self.C, self.H, self.W)
        *_, prior_mu, prior_logvar = self.forward(z2)

        prior_mu = prior_mu.unsqueeze(0)
        prior_logvar = prior_logvar.unsqueeze(0)
        
        return prior_mu, prior_logvar # dim : [1, K, qdim]

    def generate(self):
        prior_z = torch.randn(64, self.args.qdim)
        prior_z = self.args.recon_sigma * prior_z
        VAE_gen = self.decoder(prior_z.to(self.DEVICE)).detach().cpu()
        VAE_gen = VAE_gen *0.5 + 0.5
        return VAE_gen