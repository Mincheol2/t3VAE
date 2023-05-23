import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from simul_loss import log_t_normalizing_const, gamma_regularizer

class Encoder(nn.Module):
    def __init__(self, p_dim, q_dim, nu, device, num_layers, recon_sigma):
        super(Encoder, self).__init__()
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.nu = nu
        self.num_layers = num_layers
        self.device = device
        self.recon_sigma = recon_sigma

        self.fc = nn.Sequential(
            nn.Linear(self.p_dim, self.num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(self.num_layers, self.num_layers), 
            nn.LeakyReLU()
        )

        self.latent_mu = nn.Linear(self.num_layers, self.q_dim)
        self.latent_var = nn.Linear(self.num_layers, self.q_dim)
        
            
        # precomputing constants
        if self.nu != 0:
            
            self.gamma = -2 / (self.nu + self.p_dim + self.q_dim)
            
            log_tau_base = -self.p_dim * np.log(self.recon_sigma) + log_t_normalizing_const(self.nu, self.p_dim) - np.log(self.nu + self.p_dim - 2) + np.log(self.nu-2)
            
            const_2bar1_term_1 = (1 + self.q_dim / (self.nu + self.p_dim -2))
            const_2bar1_term_2_log = -self.gamma / (1+self.gamma) * log_tau_base
            self.const_2bar1 = const_2bar1_term_1 * const_2bar1_term_2_log.exp()
            
            log_tau = 2 / (self.nu + self.p_dim - 2 ) * log_tau_base
            self.tau = log_tau.exp()
    
    def reparameterize(self, mu, logvar):
        if self.nu == 0:
            std = torch.exp(0.5 * logvar) # diagonal mat
            eps = torch.randn_like(std) # Normal dist : eps ~ N(0, I)
            return mu + std * eps
        else:
            '''
                Sampling algorithm
                Let nu_prime = nu + p_dim
                1. Generate v ~ chiq(nu_prime) and eps ~ N(0, I), independently.
                2. Caculate x = mu + std * eps / (sqrt(nu_prime/nu)), where std = sqrt(nu/(nu_prime) * var)
            '''
            nu_prime = self.nu + self.q_dim
            MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(self.q_dim), torch.eye(self.q_dim))
            chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu_prime]))
            
            # Student T dist : [B, z_dim]
            eps = MVN_dist.sample(sample_shape=torch.tensor([mu.shape[0]])).to(self.device)
            
            std = np.sqrt(self.nu / nu_prime) * torch.exp(0.5 * logvar)
            v = chi_dist.sample(torch.tensor([mu.shape[0]])).to(self.device)
            return mu + std * eps * torch.sqrt(nu_prime / v)

    def forward(self, x):
        x = self.fc(x)
        mu = self.latent_mu(x)
        logvar = self.latent_var(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def loss(self, mu, logvar):
        if self.nu == 0:
            reg_loss = torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1))
        else:
            reg_loss = gamma_regularizer(mu, logvar, self.p_dim, self.const_2bar1, self.gamma, self.tau, self.nu)
        
        return reg_loss

class Decoder(nn.Module):
    def __init__(self, p_dim, q_dim, nu, device, num_layers, recon_sigma):
        super(Decoder, self).__init__()
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.nu = nu
        self.device = device
        self.num_layers = num_layers
        self.recon_sigma = recon_sigma
        self.fc = nn.Sequential(
            nn.Linear(self.q_dim, self.num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(self.num_layers, self.num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(self.num_layers, self.p_dim)
        )

    def forward(self, enc_z):
        x = self.fc(enc_z)
        return x
    
    
    def sampling(self, z):
        '''
        For given z_1,..., z_B \in R^q, we wish to sample x_1,...,x_B from
        x_i ~ t_p(f_theta(z_i),  ((nu + ||z_i||^2) / (nu+q)) * sigma^2 * I_p,  nu+q)
        '''
        f_theta = self.forward(z)

        if self.nu == 0:
            eps = torch.randn_like(f_theta) # Normal dist : eps ~ N(0, I)
            return f_theta + self.recon_sigma * eps
        else:
            nu_prime = self.nu + self.q_dim
            MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(self.p_dim), torch.eye(self.p_dim))
            chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu_prime]))
            
            eps = MVN_dist.sample(sample_shape=torch.tensor([f_theta.shape[0]])).to(self.device)
            std_const = torch.sqrt((self.nu * torch.ones(f_theta.shape[0]).to(self.device) + torch.norm(z,dim=1).pow(2)) / nu_prime)
            std_const = std_const.unsqueeze(1).repeat(1,self.p_dim).to(self.device)
            std = self.recon_sigma * std_const
            v = chi_dist.sample(sample_shape=torch.tensor([f_theta.shape[0]])).to(self.device)
            return f_theta + std * (eps * torch.sqrt(nu_prime / v))

    def loss(self, recon_x, x):
        recon_loss = F.mse_loss(recon_x, x, reduction = 'none').sum(dim = 1).mean(dim = 0) / self.recon_sigma**2
        
        return recon_loss


class gammaAE():
    def __init__(self, train_dataset, p_dim, q_dim, nu, recon_sigma, device, num_layers,  
                 lr = 1e-3, batch_size = 64, eps = 1e-6, weight_decay = 1e-5):
        self.train_dataset = train_dataset
        self.p_dim = p_dim
        self.q_dim = q_dim
        self.nu = nu
        self.recon_sigma = recon_sigma
        self.device = device
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size
        self.eps = eps
        self.weight_decay = weight_decay

        self.encoder = Encoder(p_dim, q_dim, nu, device, num_layers, recon_sigma).to(device)
        self.decoder = Decoder(p_dim, q_dim, nu, device, num_layers, recon_sigma).to(device)
        self.opt = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                              lr=lr, eps=eps, weight_decay=weight_decay)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def train(self, epoch, writer):
        self.encoder.train()
        self.decoder.train()

        denom_train = int(len(self.train_loader.dataset)/self.batch_size) + 1

        for batch_idx, data in enumerate(self.train_loader):
            data = data[0].to(self.device)
            self.opt.zero_grad()
            z, mu, logvar = self.encoder(data)
            reg_loss = self.encoder.loss(mu, logvar)
            recon_data = self.decoder(z)
            recon_loss = self.decoder.loss(recon_data, data.view(-1,self.p_dim))
            total_loss = reg_loss + recon_loss
            total_loss.backward()

            current_step_train = epoch * denom_train + batch_idx
            writer.add_scalar("Train/Reconstruction Error", recon_loss.item(), current_step_train)
            writer.add_scalar("Train/Regularizer", reg_loss.item(), current_step_train)
            writer.add_scalar("Train/Total Loss" , total_loss.item(), current_step_train)

            self.opt.step()

    def validation(self, data, epoch, writer):
        self.encoder.eval()
        self.decoder.eval()

        data = data.to(self.device)

        z, mu, logvar = self.encoder(data)
        reg_loss = self.encoder.loss(mu, logvar)
        recon_data = self.decoder(z)
        recon_loss = self.decoder.loss(recon_data, data.view(-1,self.p_dim))
        total_loss = reg_loss + recon_loss

        writer.add_scalar("Validation/Reconstruction Error", recon_loss.item(), epoch)
        writer.add_scalar("Validation/Regularizer", reg_loss.item(), epoch)
        writer.add_scalar("Validation/Total Loss" , total_loss.item(), epoch)

        return total_loss.item()
    
    def test(self, data, epoch, writer, tail_cut = 5):
        self.encoder.eval()
        self.decoder.eval()

        data = data.to(self.device)

        z, mu, logvar = self.encoder(data)
        reg_loss = self.encoder.loss(mu, logvar)
        recon_data = self.decoder(z)
        recon_loss = self.decoder.loss(recon_data, data.view(-1,self.p_dim))
        total_loss = reg_loss + recon_loss

        writer.add_scalar("Test/Reconstruction Error", recon_loss.item(), epoch)
        writer.add_scalar("Test/Regularizer", reg_loss.item(), epoch)
        writer.add_scalar("Test/Total Loss" , total_loss.item(), epoch)

        return total_loss.item()

    def generate(self, N = 1000) : 
        MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(self.q_dim), torch.eye(self.q_dim))
        prior = MVN_dist.sample(sample_shape=torch.tensor([N]))

        if self.nu != 0 : 
            chi_dist = torch.distributions.chi2.Chi2(torch.tensor([self.nu]))
            v = chi_dist.sample(sample_shape=torch.tensor([N]))
            prior *= torch.sqrt(self.nu/v)
        
        prior = prior.to(self.device)
        return self.decoder.sampling(prior)

    def reconstruct(self, x) : 
        return self.decoder.sampling(self.encoder(x)[0])
