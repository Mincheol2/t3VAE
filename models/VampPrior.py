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
        self.pdim = self.C * self.H * self.W
        
        # Vampprior : use pseudo input layer
        
        # deterministic input
        self.idle_input = torch.eye(self.args.num_components, requires_grad=False).to(self.DEVICE)
        self.pseudo_input_layer = nn.Sequential(nn.Linear(self.args.num_components, self.pdim),
                                          nn.Hardtanh(min_val=0.0, max_val=1.0)
                                          )
        
        # the default mean and std value initialization in the VampPrior's GitHub code
        # torch.nn.init.normal_(self.pseudo_input_layer[0].weight, mean=-0.05, std=0.01)

    
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
        N = x.shape[0]
        prior_mu, prior_logvar = self.make_vampprior() # [1, K, qdim]
        prior_var = prior_logvar.exp()
        E_log_q = torch.sum(-0.5 * (logvar + (z - mu) ** 2 / logvar.exp()), dim = 1)
        
        ## Compute E[logp(x)]
        z = z.unsqueeze(dim=1) # [B, 1, qdim]
        # compute dim2 : qdim
        E_log_p = torch.sum(-0.5 * (prior_logvar + (z - prior_mu) ** 2/ prior_var),
                              dim = 2) - torch.tensor(np.log(self.args.num_components)).float()
        
        # compute dim1 : 1 -> K (dimension broadcasting)
        E_log_p = torch.logsumexp(E_log_p, dim = 1) # For numerical stability

        reg_loss = 2 * torch.mean(E_log_q - E_log_p, dim=0)
        recon_loss = torch.sum((recon_x - x)**2 / (N * self.args.recon_sigma**2))
        total_loss = self.args.reg_weight * reg_loss + recon_loss
        return reg_loss, recon_loss, total_loss

    def make_vampprior(self):
        x = self.pseudo_input_layer(self.idle_input)
        x = x.view(-1, self.C, self.H, self.W)
        *_, prior_mu, prior_logvar = self.encoder(x)

        prior_mu = prior_mu.unsqueeze(0)
        prior_logvar = prior_logvar.unsqueeze(0)
        
        return prior_mu, prior_logvar # dim : [1, K, qdim], K = num_components

    def generate(self):
        # make pseudo input prior with pseudo_input_layer
        x = self.pseudo_input_layer(self.idle_input)[:144]
        x = x.view(-1, self.C, self.H, self.W)
        prior_z , *_ = self.encoder(x)



        VAE_gen = self.decoder(prior_z.to(self.DEVICE)).detach().cpu()
        VAE_gen = VAE_gen * 0.5 + 0.5
        return VAE_gen