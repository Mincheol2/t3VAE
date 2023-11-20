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
        self.m1 = args.m_dim
        self.m2 = self.m1 // 2
        self.h_latent_dim_list = [self.m1, self.m2]
        self.gamma = -2 / (self.args.nu + self.n_dim + self.m1 + self.m2)
        self.nu = args.nu

        self.L = 2 # L-level t3HVAE model

        # additional Hierarchical latent layers.
        
        input_dim = self.cnn_lineardim + self.m1
        
        self.mu_layer2 = nn.Linear(input_dim, self.m2).to(self.DEVICE)
        self.logvar_layer2 = nn.Linear(input_dim, self.m2).to(self.DEVICE)
        self.priormu_layer = nn.Linear(self.m1,self.m2).to(self.DEVICE)

        

        # decoder feature layer
        self.linear = nn.Sequential(
                        nn.Linear(self.m1 +  self.m2, self.cnn_lineardim),
                        nn.ReLU()
        ).to(self.DEVICE)

        
        # 2. define Constants 

        self.gamma_exponent = self.gamma / (1+self.gamma)
        self.M_plus_n = (input_dim - self.m2) + self.n_dim  # m1 + n

        # compute C1/C2 and taus (for the alternative priors)
        self.log_nu_C1overC2 = - self.gamma_exponent * (log_t_normalizing_C(self.nu, self.M_plus_n) + np.log(self.nu - 2))
        self.log_nu_C1overC2 += - 1 / (self.gamma + 1) * (np.log(self.nu + self.M_plus_n - 2) ) + np.log(self.nu)
        self.log_nu_C1overC2 -= np.log(self.nu + self.M_plus_n + self.m2 - 2)
        
        self.log_tau_1 = log_t_normalizing_C(self.nu, self.M_plus_n) - np.log(self.M_plus_n + self.nu - 2) + np.log(self.nu - 2)
        self.log_tau_1 -= (np.log(self.nu + self.M_plus_n- 2) -  np.log(self.nu + self.n_dim - 2))/self.gamma
        self.log_tau_1 /= (self.nu + self.M_plus_n - 2)
        self.log_tau_1 += (np.log(self.nu) - np.log(self.nu + self.n_dim)) / 2
        
        self.log_tau_2 = log_t_normalizing_C(self.nu, self.M_plus_n) - np.log(self.M_plus_n + self.nu - 2) + np.log(self.nu - 2)
        self.log_tau_2 /= (self.nu + self.M_plus_n - 2)
        self.log_tau_2 += (np.log(self.nu) - np.log(self.nu + self.M_plus_n)) / 2

        # define samplers for t
        self.MVN_dists = []
        self.chi_dists = []
        for i in range(2):
            MVN_dim = self.h_latent_dim_list[i]
            chi_dim = self.nu+self.n_dim if i == 0 else self.nu + self.M_plus_n
            self.MVN_dists.append(torch.distributions.MultivariateNormal(torch.zeros(MVN_dim), torch.eye(MVN_dim)))
            self.chi_dists.append(torch.distributions.chi2.Chi2(torch.tensor([chi_dim])))
        

    def encoder(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)

        # L=1
        mu1 = self.mu_layer(x)
        logvar1 = self.logvar_layer(x)
        z1 = self.reparameterize(mu1, logvar1, 1)

        # L=2 
        input_x = torch.cat([z1, x], dim=1).to(self.DEVICE)
        
        mu2 = self.mu_layer2(input_x)
        logvar2 = self.logvar_layer2(input_x)
        z2 = self.reparameterize(mu2, logvar2, 2)

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

    def reparameterize(self, mu, logvar, L):
        '''
            t-reparametrization trick (L = hierarchy parameter)

            Let nu_prime = nu + n_dim
            1. Generate v ~ chiq(nu_prime) and eps ~ N(0, I), independently.
            2. Caculate x = mu + std * eps / (sqrt(v/nu_prime)), where std = sqrt(nu/(nu_prime) * var)
        '''
        nu_prime = self.nu + self.n_dim + (L-1)*self.m1 # L=1 -> nu+n // L=2 -> nu+m1+n

        # Student T dist : [B, z_dim]
        eps = self.MVN_dists[L-1].sample(sample_shape=torch.tensor([mu.shape[0]])).to(self.DEVICE)
        std = torch.exp(0.5 * logvar)
        std = torch.tensor(self.nu / nu_prime).sqrt() * std
        v = self.chi_dists[L-1].sample(sample_shape=torch.tensor([mu.shape[0]])).to(self.DEVICE)
        
        return mu + std * eps * torch.sqrt(nu_prime / v)

    def loss(self, x, recon_x, z1, mu1, mu2, logvar1, logvar2):
        N = x.shape[0]

        ## gamma regularizers ##
        trace_denom = self.nu + self.n_dim - 2
        

        # for i in range(self.L):
        
        # regularizer for q(z_1|x)
        mu_norm_sq = torch.linalg.norm(mu1, ord=2, dim=1).pow(2)
        trace_var = self.nu / trace_denom * torch.sum(logvar1.exp(),dim=1)
        log_det_var = torch.sum(logvar1,dim=1)
        reg_loss = torch.mean(mu_norm_sq + trace_var + self.gamma / 2 * self.log_nu_C1overC2.exp() * log_det_var, dim=0)

        trace_denom += self.m1
                
        # regularizer for q(z_2 |x, z_1)   
        
        prior_mu = self.priormu_layer(z1) #p(z2|z1)
        mu_norm_sq = torch.linalg.norm(mu2 - prior_mu, ord=2, dim=1).pow(2)
        trace_var = self.nu / trace_denom * torch.sum(logvar2.exp(),dim=1)
        log_det_var = - self.gamma_exponent / 2 * torch.sum(logvar2,dim=1)
        reg_loss2 = torch.mean(mu_norm_sq + trace_var - (self.log_nu_C1overC2 + log_det_var).exp(), dim=0)
        
        ## recon loss (same as VAE) ##
        recon_loss = torch.sum((recon_x - x)**2 / N)
        total_loss = recon_loss + reg_loss + reg_loss2
        return reg_loss, reg_loss2, recon_loss, total_loss

    def generate(self, N = 64):
        '''
        There are two alternative t-priors to generate.
        '''

        # L = 1
        nu_prime = self.nu
        mu_1 = torch.zeros((N,self.m1)).to(self.DEVICE)
        logvar_1 = torch.zeros((N,self.m1)).to(self.DEVICE)
        eps = self.MVN_dists[0].sample(sample_shape=torch.tensor([mu_1.shape[0]])).to(self.DEVICE)
        std = torch.exp(0.5 * logvar_1)
        std = torch.tensor(self.nu / nu_prime).sqrt() * std
        v = self.chi_dists[0].sample(sample_shape=torch.tensor([mu_1.shape[0]])).to(self.DEVICE)
        
        prior_t1 = mu_1 + std * eps * torch.sqrt(nu_prime / v)

        
        mu_2 = self.priormu_layer(prior_t1)
        logvar_2 = (2* self.log_tau_2)*torch.ones((N, self.m2)).to(self.DEVICE)
        prior_t2 = self.reparameterize(mu_2, logvar_2, 2)

        
        imgs = self.decoder(prior_t1,prior_t2).detach().cpu()

        return imgs