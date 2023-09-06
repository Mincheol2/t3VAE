import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

class GMVAE(nn.Module) : 
    def __init__(self, l_dim = 1, n_dim=1, m_dim=1, K = 2, nu=3, recon_sigma=1, reg_weight=1, num_layers=64, device='cpu', sample_size_for_integral = 1):
        '''
        In GMVAE, w, z, x are latent variables and y is the data. 
        w ~ N_{l_dim} (0, I)
        z ~ Multinomial (1/K, ..., 1/K)
        x | w, z ~ N_{m_dim} (mu_z(w), Sigma_z(w))
        y | x ~ N_{n_dim} (mu_theta(x), sigma^2 I)
        '''
        super(GMVAE, self).__init__()
        self.model_name = "GMVAE"

        self.l_dim = l_dim
        self.n_dim = n_dim
        self.m_dim = m_dim
        self.K = K
        self.recon_sigma = recon_sigma
        self.reg_weight = reg_weight
        self.num_layers = num_layers
        self.device = device

        self.sample_size_for_integral = sample_size_for_integral

        self.z_prob = torch.tensor([1/K for _ in range(K)])

        # y -> w
        self.w_encoder = nn.Sequential(
            nn.Linear(n_dim, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, num_layers), 
            nn.LeakyReLU()
        )
        self.w_encoder_mu = nn.Linear(num_layers, l_dim)
        self.w_encoder_logvar = nn.Linear(num_layers, l_dim)

        # y -> x
        self.x_encoder = nn.Sequential(
            nn.Linear(n_dim, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, num_layers), 
            nn.LeakyReLU()
        )
        self.x_encoder_mu = nn.Linear(num_layers, m_dim)
        self.x_encoder_logvar = nn.Linear(num_layers, m_dim)

        # w, z -> x
        self.x_decoder = nn.Sequential(
            nn.Linear(l_dim, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, num_layers), 
            nn.LeakyReLU()
        )
        self.x_decoder_mu = nn.Linear(num_layers, K * m_dim)
        self.x_decoder_logvar = nn.Linear(num_layers, K * m_dim)

        # x -> y
        self.y_decoder = nn.Sequential(
            nn.Linear(m_dim, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, n_dim)
        )

    def encoder_reparameterize(self, mu, logvar) : 
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + std * eps
    
    def w_encode(self, y) : 
        y = self.w_encoder(y)
        w_mu = self.w_encoder_mu(y)
        w_logvar = self.w_encoder_logvar(y)
        enc_w = self.encoder_reparameterize(w_mu, w_logvar)
        return enc_w, w_mu, w_logvar


    def x_encode(self, y) : 
        y = self.x_decoder(y)
        x_mu = self.x_encoder_mu(y)
        x_logvar = self.x_encoder_logvar(y)
        enc_x = self.encoder_reparameterize(x_mu, x_logvar)
        return enc_x, x_mu, x_logvar
    


    def x_decode(self, w, z) : 
        # only for m_dim = 1, 

        w = self.x_decoder(w)
        mu = self.x_decoder_mu(w)
        mu = torch.gather(mu, 1, z.unsqueeze(1))

        logvar = self.x_decoder_logvar(w)
        logvar = torch.gather(logvar, 1, z.unsqueeze(1))


        # indice = z.repeat(self.m_dim, 1, 1)
        # print(f'indice : {indice.shape}')
        # indice = indice.T
        # print(f'indice : {indice.shape}')
        # indice = indice.reshape(-1, self.K, self.m_dim)
        # print(f'indice : {indice.shape}')
        # print(f'indice : {indice[0:5]}')

        # w = self.x_decoder(w)
        # mu = self.x_decoder_mu(w).reshape(-1, self.K, self.m_dim)
        # print(f'mu : {mu.shape}')
        # print(f'mu : {mu[0:5]}')
        # logvar = self.x_decoder_logvar(w).reshape(-1, self.K, self.m_dim)

        # mu = torch.gather(mu, 1, indice)
        # logvar = torch.gather(logvar, 1, indice)

        return mu, logvar


    def y_decode(self, x) : 
        return self.y_decoder(x)
    


    def z_posterior(self, w, x) : 
        w = self.x_decoder(w)
        mu = self.x_decoder_mu(w).reshape(-1, self.K, self.m_dim)
        logvar = self.x_decoder_logvar(w).reshape(-1, self.K, self.m_dim)

        std = torch.exp(0.5 * logvar)

        term_1 = torch.sum(logvar + torch.pow((mu - x.unsqueeze(1)) / std, 2), dim = 2)
        term_2 = torch.exp(-0.5 * term_1)
        z_posterior = term_2 / torch.sum(term_2, dim = 1).unsqueeze(1)

        return z_posterior
    
    def x_decoder_sampling(self, w, z) : 
        mu_beta, logvar_beta = self.x_decode(w,z)
        eps = torch.randn_like(mu_beta)
        return mu_beta + torch.exp(0.5 * logvar_beta) * eps
    
    def y_decoder_sampling(self, x) : 
        mu_theta = self.y_decode(x)
        eps = torch.randn_like(mu_theta)
        return mu_theta + self.recon_sigma * eps

    def generate(self, N = 1000) : 
        w_prior = torch.randn(N, self.l_dim).to(self.device)
        z_prior = torch.multinomial(self.z_prob, N, replacement=True).to(self.device)

        x_gen = self.x_decoder_sampling(w_prior,z_prior)
        y_gen = self.y_decoder_sampling(x_gen)

        return y_gen

    # def reconstruct_1(self, y) : 
    #     enc_x, _, _ = self.x_encode(y)
    #     y_recon = self.y_decoder_sampling(enc_x)

    #     return y_recon
    
    # def reconstruct_2(self, y) : 
    #     enc_x, _, _ = self.x_encode(y)
    #     enc_w, _, _ = self.w_encode(y)

    #     z_posterior = self.z_posterior(enc_w, enc_x)
    #     enc_z = torch.multinomial(z_posterior, 1, replacement=True).squeeze(1)

    #     x_recon = self.x_decoder_sampling(enc_w, enc_z)
    #     y_recon = self.y_decoder_sampling(x_recon)

    #     return y_recon

    def kl_div_continuous(self, mu_0, logvar_0, mu_1 = None, logvar_1 = None, reduction = True) : 
        # 2 * D_KL (N(mu_0, logvar_0) || N(mu_1, logvar_1))

        if mu_1 is None : 
            mu_1 = torch.zeros_like(mu_0).to(self.device)
        if logvar_1 is None : 
            logvar_1 = torch.zeros_like(mu_0).to(self.device)

        div = torch.sum(logvar_1 - logvar_0 - 1 + torch.exp(logvar_0 - logvar_1) + (mu_0 - mu_1).pow(2) / torch.exp(logvar_1), dim = 1)

        if reduction is True : 
            return torch. mean(div)
        else : 
            return div

    def kl_div_discrete(self, prob) : 
        # 2 * D_KL( prob_0 || uniform muitinomial)
        div = torch.sum(prob * torch.log(prob), dim = 1) + np.log(self.K)
        return 2 * torch.mean(div)
        
    def forward(self, y) : 
        sample_size = y.shape[0]

        enc_x, x_mu, x_logvar = self.x_encode(y)
        enc_w, w_mu, w_logvar = self.w_encode(y)
        z_posterior = self.z_posterior(enc_w, enc_x)

        y_mu = self.y_decode(enc_x)

        # 2 * reconstruction term
        # torch.mean(torch.sum(torch.pow((y - y_mu) / self.recon_sigma, 2), dim = 1))
        
        recon_term = torch.mean(F.mse_loss(y_mu, y, reduction = 'none').sum(1))  / (self.recon_sigma ** 2)

        conditional_prior_term = 0
        for k in range(self.K) : 
            prob = z_posterior[:, k]
            mu_beta_i, logvar_beta_i = self.x_decode(enc_w, z= torch.ones(sample_size, dtype=torch.int64).to(self.device))
            kl_div = self.kl_div_continuous(x_mu, x_logvar, mu_beta_i, logvar_beta_i, reduction = False)
            conditional_prior_term += torch.mean(prob * kl_div) / self.sample_size_for_integral
        w_prior_term = self.kl_div_continuous(w_mu, w_logvar) / self.sample_size_for_integral
        z_prior_term = self.kl_div_discrete(z_posterior) / self.sample_size_for_integral

        reg_term = conditional_prior_term + w_prior_term + z_prior_term

        return recon_term, reg_term, recon_term + self.reg_weight * reg_term

        # for _ in range(self.sample_size_for_integral) : 
        #     enc_x, x_mu, x_logvar = self.x_encode(y)
        #     enc_w, w_mu, w_logvar = self.w_encode(y)
        #     z_posterior = self.z_posterior(enc_w, enc_x)

        #     y_mu = self.y_decode(enc_x)

        #     # 2 * reconstruction term
        #     # torch.mean(torch.sum(torch.pow((y - y_mu) / self.recon_sigma, 2), dim = 1))
            
        #     mean_recon_term += torch.mean(F.mse_loss(y_mu, y, reduction = 'none').sum(1) / (self.recon_sigma ** 2)) / self.sample_size_for_integral

        #     for k in range(self.K) : 
        #         prob = z_posterior[:, k]
        #         mu_beta_i, logvar_beta_i = self.x_decode(enc_w, z= torch.ones(sample_size, dtype=torch.int64).to(self.device))
        #         kl_div = self.kl_div_continuous(x_mu, x_logvar, mu_beta_i, logvar_beta_i, reduction = False)
        #         mean_conditional_prior_term += torch.mean(prob * kl_div) / self.sample_size_for_integral
        #     mean_w_prior_term += self.kl_div_continuous(w_mu, w_logvar) / self.sample_size_for_integral
        #     mean_z_prior_term += self.kl_div_discrete(z_posterior) / self.sample_size_for_integral


            # enc_x, x_mu, x_logvar = self.x_encode(y)
            # enc_w, w_mu, w_logvar = self.w_encode(y)
            # z_posterior = self.z_posterior(enc_w, enc_x)

            # y_mu = self.y_decode(enc_x)

            # # 2 * reconstruction term
            # recon_term = torch.mean(torch.sum(torch.pow((y - y_mu) / self.recon_sigma, 2), dim = 1))
            # w_prior_term = self.kl_div_continuous(w_mu, w_logvar)
            # z_prior_term = self.kl_div_discrete(z_posterior)
            # conditional_prior_term = 0
            # for k in range(self.K) : 
            #     prob = z_posterior[:, k]
            #     mu_beta_i, logvar_beta_i = self.x_decode(enc_w, z= torch.ones(sample_size, dtype=torch.int64).to(self.device))
            #     kl_div = self.kl_div_continuous(x_mu, x_logvar, mu_beta_i, logvar_beta_i, reduction = False)
            #     conditional_prior_term += torch.mean(prob * kl_div)

            # mean_recon_term += recon_term / self.sample_size_for_integral
            # mean_conditiona_prior_term += conditional_prior_term / self.sample_size_for_integral
            # mean_w_prior_term += w_prior_term / self.sample_size_for_integral
            # mean_z_prior_term += z_prior_term / self.sample_size_for_integral

        # mean_reg_term = mean_conditional_prior_term + mean_w_prior_term + mean_z_prior_term

        # return mean_recon_term, mean_reg_term, mean_recon_term * self.reg_weight * mean_reg_term



        



    # def recon_loss(self, y, mu_theta) : 
    #     recon = torch.sum((y - mu_theta).pow(2), dim = 1) / self.recon_sigma**2
    #     return torch.mean(recon)

    # def reg_loss() : 
    #     # return KL regularizer including constant term
    #     return torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1))
    
    # def total_loss(self, x, recon_x, mu, logvar) : 
    #     recon = self.recon_loss(recon_x, x)
    #     reg = self.reg_loss(mu, logvar)

    #     return recon, reg, recon + self.reg_weight * reg

        

