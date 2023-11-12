import numpy as np
import torch
import torch.nn as nn
import math

from models import t3VAE

class t3HVAE(t3VAE.t3VAE):
    def __init__(self, image_shape, DEVICE, args):
        super(t3HVAE, self).__init__(image_shape, DEVICE, args)
        
        def log_t_normalizing_C(nu, d):
            nom = torch.lgamma(torch.tensor((nu+d)/2))
            denom = torch.lgamma(torch.tensor(nu/2)) + d/2 * (np.log(nu) + np.log(np.pi))
            return nom - denom

        self.nu = args.nu
        self.L = 2 # L-level t3HVAE model

        # Hierarchical latent layers.
        self.mu_layers = []
        self.logvar_layers = []
        self.prior_layers = [0] # for starting index 1
        self.h_latent_dim_list = []

        
        # For simplicity, fix all recon_sigma to 1.

        h_latent_dim = args.m_dim
        input_dim = self.cnn_lineardim


        # 1. Construct ith MLP layer

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

        
        # 2. define Cants 
        
        # L=2 case
        m2 = self.h_latent_dim_list[-1]
        M_plus_n = (input_dim - m2) + self.n_dim  # m1 + n
        log_tau_base = log_t_normalizing_C(self.nu, M_plus_n ) - np.log(M_plus_n + self.nu - 2) + np.log(self.nu-2)
        
        log_tau_sq = 2 / (self.nu + M_plus_n - 2) * log_tau_base
        self.tau_sq = self.nu / (self.nu + M_plus_n) * log_tau_sq.exp()

        # Derive C1/C2 from tau_sq
        self.gamma_exponent = self.gamma / (1+self.gamma)

        self.log_C_1_over_2 = 0
        self.log_C_1_over_2 += np.log(self.nu + M_plus_n) + (2 + self.gamma_exponent * m2) * log_tau_sq / 2
        self.log_C_1_over_2 += m2 /2 * self.gamma_exponent * np.log(1 + M_plus_n / self.nu)
        self.log_C_1_over_2 += np.log(1 + m2 / (self.nu + M_plus_n - 2)) + np.log(self.nu)

    def encoder(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)
        mu_list = []
        logvar_list = []
        z_list = []

        for i in range(self.L):
            if i != 0:
                z_concat = torch.concat(z_list, dim=1)
                zx_concat = torch.concat([z_concat, x], dim=1)
            else:
                zx_concat = x
            
            mu = self.mu_layers[i](zx_concat)
            logvar = self.logvar_layers[i](zx_concat)

            z = self.reparameterize(mu, logvar)
            
            mu_list.append(z)
            logvar_list.append(z)
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
        '''
            t-reparametrization trick

            Let nu_prime = nu + n_dim
            1. Generate v ~ chiq(nu_prime) and eps ~ N(0, I), independently.
            2. Caculate x = mu + std * eps / (sqrt(v/nu_prime)), where std = sqrt(nu/(nu_prime) * var)
        '''
        MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(mu.shape[1]), torch.eye(mu.shape[1]))
        
        # Student T dist : [B, z_dim]
        eps = MVN_dist.sample(sample_shape=torch.tensor([mu.shape[0]])).to(self.DEVICE)
        std = torch.exp(0.5 * logvar)
        std = torch.tensor(self.nu / self.nu_prime).sqrt() * std
        v = self.chi_dist.sample(sample_shape=torch.tensor([mu.shape[0]])).to(self.DEVICE)
        return mu + std * eps * torch.sqrt(self.nu_prime / v)

    def loss(self, x, recon_x, z_list, mu_list, logvar_list):
        N = x.shape[0]
        #TODO : L >= 3?


        ## gamma regularizers ##
        trace_denom = self.nu + self.n_dim - 2
        

        # for i in range(self.L):
        
        # regularizer for q(z_1|x)
        mu_norm_sq = torch.linalg.norm(mu_list[0]-0, ord=2, dim=1).pow(2)
        trace_var = self.nu / trace_denom * torch.sum(logvar_list[0].exp(),dim=1)
        
        log_det_var = torch.sum(logvar_list[0],dim=1)  # log(|Lamma(x)|)

        reg_loss = torch.mean(mu_norm_sq + trace_var + self.gamma / 2 * self.log_C_1_over_2.exp() * log_det_var, dim=0) + self.nu_prime * self.tau_sq

        trace_denom += self.h_latent_dim_list[0]
                
        # regularizer for q(z_2 |x, z_1)

        prior_z2 = self.prior_layers[1](z_list[0]) #p(z2|z1)

        mu_norm_sq = torch.linalg.norm(mu_list[1] - prior_z2, ord=2, dim=1).pow(2)
        trace_var = self.nu / trace_denom * torch.sum(logvar_list[1].exp(),dim=1)
        log_det_var = - self.gamma_exponent / 2 * torch.sum(logvar_list[1],dim=1)
        reg_loss2 = torch.mean(mu_norm_sq + trace_var - (self.log_C_1_over_2 + log_det_var).exp(), dim=0)

        ## recon loss (same as VAE) ##
        recon_loss = torch.sum((recon_x - x)**2 / N)
        total_loss = recon_loss + reg_loss + reg_loss2
        return reg_loss, reg_loss2, recon_loss, total_loss

    def generate(self, N = 64):
        '''
        Hierarchical t-priors
        '''
        prior_chi_dist = torch.distributions.chi2.Chi2(torch.tensor([self.nu]))
        prior_z = self.MVN_dist.sample(sample_shape=torch.tensor([N])).to(self.DEVICE)
        v = prior_chi_dist.sample(sample_shape=torch.tensor([N])).to(self.DEVICE)
        prior_t = self.args.prior_sigma * prior_z * torch.sqrt(self.nu / v)
        prev_t_samples = []
        for i in range(1, self.L+1):
            if i != 1:
                prev_t_concat = torch.concat(prev_t_samples,dim=1)
                input_t = torch.concat(torch.concat([prev_t_concat, prior_t], dim=1))
            else:
                input_t = prior_t
            
            prior_t = self.prior_layers[i](input_t)
            prev_t_samples.append(prior_t)

        
        imgs = self.decoder(prev_t_samples).detach().cpu()

        return imgs