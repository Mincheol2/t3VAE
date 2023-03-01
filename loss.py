import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import argument

args = argument.args

'''
    beta divergence loss (ref to RVAE)
'''
def beta_div_loss(recon_x, x, beta, sigma=0.5):
    p_dim = recon_x.shape[1] # data dim
    sigma_sq = sigma**2
    const2 = 1 / pow((2 * np.pi * (sigma**2)), (beta * p_dim / 2))
    recon_norm = torch.sum((x - recon_x)**2,dim=1)
    internal_term = torch.exp(-(beta / (2 * sigma_sq )) * recon_norm)
    loss = torch.sum(-((1 + beta) / beta) * (const2 * internal_term - 1))
    return loss


'''
    Gamma divergence assumes that
    prior p ~ t_q(0,I,nu) and posterior q ~ t_q(mu, Sigma, nu + p_dim).
    (Student's T distribution)
    
    For numerical stablity, we use an argument 'logvar' instead of var.
    Note that var is a diagonal matrix.
'''

'''
    log version of normalizing constant for t-distribution
'''
def log_t_normalizing_const(nu, d):
    nom = torch.lgamma(torch.tensor((nu+d)/2)) 
    denom = torch.lgamma(torch.tensor(nu/2)) + d/2 * (np.log(nu) + np.log(np.pi))
    return nom - denom


def gamma_regularizer(mu, logvar, p_dim):
    '''
        p_dim : data dim
        q_dim : latent dim

        output : 1/N sum_{i=1}^{N} ||mu(X_i)||^2 + Sigma(X_i)|^{-gamma /2}
    '''
    q_dim = args.zdim
    nu = args.nu
    gamma = -2 / (nu + p_dim + q_dim)

    mu_norm_sq = torch.linalg.norm(mu, ord=2, dim=1).pow(2)
    trace_var = args.nu / (nu + p_dim - 2) * torch.sum(logvar.exp(),dim=1)
    log_det_var = -gamma / (2+2*gamma) * torch.sum(logvar,dim=1)
    const_2bar1_term_1 = (1 + q_dim / (nu + p_dim -2))
    const_2bar1_term_2_log = -gamma / (1+gamma) * (-p_dim + log_t_normalizing_const(nu, p_dim) - np.log(nu + p_dim - 2) + np.log(nu-2))
    const_2bar1 = const_2bar1_term_1 * const_2bar1_term_2_log.exp()
    
    return torch.sum(mu_norm_sq + trace_var - args.nu * const_2bar1 * log_det_var.exp())




class Gamma_Family():
    def __init__(self, post_mu, post_logvar, nu, prior_mu=None, prior_logvar=None):
        self.post_mu = post_mu
        self.post_logvar = post_logvar
        self.nu = nu
        
        self.prior_mu = torch.zeros_like(post_mu)
        self.prior_logvar = torch.zeros_like(post_logvar)
        self.post_var = self.post_logvar.exp()
        self.prior_var = self.prior_logvar.exp()

    def gamma_divergence(self, nu):
        '''
        Generalized gamma divergence.
        paramter : nu > 2
        (This is because the variance of T dist : nu/(nu-2), v>2)
        cf) Instead of gamma, we use nu(degree of freedom) as a paramter. (gamma = -2 /(1+nu))
        '''
        # Check the well-definedness
        if nu <= 2:
            raise Exception(f'the degree of freedom is not larger than 2. Divergence is not well-defined.')


        # dimension : [B : batch size, D : zdim]
        zdim = self.post_mu.shape[1]
        log_det_ratio = (nu + zdim) / (2*(nu + zdim - 2)) * (torch.sum(self.prior_logvar,dim=1) - torch.sum(self.post_logvar,dim=1))
        log_term = (nu + zdim)/2 * torch.log(1 + 1/(nu-2) * torch.sum( self.post_var / self.prior_var,dim=1) + 1/nu * torch.sum( (self.prior_mu-self.post_mu).pow(2) / self.prior_var,dim=1))
        
        gamma_div = torch.sum(log_det_ratio + log_term) # Batch mean
        return gamma_div
    


'''
    Alpha divergence assumes that
    prior p ~ N(0,I) and posterior q ~ N(mu, var). (Note that var is diagonal.)
    For numerical stablity, we use an argument 'logvar' instead of var.
    You can change prior's mean and variance by modifying the argument 'prior_mu' and 'prior_logvar'.
'''

class Alpha_Family():
    def __init__(self, post_mu, post_logvar):
        self.post_mu = post_mu
        self.post_logvar = post_logvar
        self.prior_mu = torch.zeros_like(post_mu) 
        self.prior_logvar = torch.zeros_like(post_logvar)
        self.post_var = self.post_logvar.exp()
        self.prior_var = self.prior_logvar.exp()


    def KL_loss(self, is_reversed=False):
        kl_div = 0
        logvar_diff = self.post_logvar - self.prior_logvar
        mu_square = (self.post_mu - self.prior_mu).pow(2)
        if is_reversed:
            sq_term = (self.prior_var + mu_square) / self.post_var
            kl_div = -0.5 * torch.sum(- logvar_diff + 1.0 - sq_term)
        else:
            sq_term = (self.post_var + mu_square) / self.prior_var
            kl_div = -0.5 * torch.sum(logvar_diff + 1.0 - sq_term)
        return kl_div


    def alpha_divergence(self, alpha):
        '''
        Generalized alpha divergence
        
        cf) Special cases
        * KL Divergence (alpha = 1, 0)
        * Hellinger distance (alpha = 0.5)
        * Pearson divergence (alpha = 2)
        * Neyman divergence (alpha = -1)
        '''
        if alpha == 1:
            return self.KL_loss()
        elif alpha == 0:
            return self.KL_loss(is_reversed=True)
        
        else:
            var_denom = (1-alpha) * self.post_var + alpha * self.prior_var
            # Check the well-definedness
            if torch.min(var_denom) <= 0:
                raise Exception(f'min(var_denom) = {torch.min(var_denom)} is not positive. Divergence may not be well-defined.')
            
            const_alpha = 1 / (alpha * (1-alpha))
            prod_const = 0.5 * ((1-alpha) * self.post_logvar + alpha * self.prior_logvar - var_denom.log())
            exp_term = -0.5 * alpha * (1-alpha) * (self.prior_mu - self.post_mu).pow(2) / var_denom
            
            log_prodterm = torch.sum(prod_const + exp_term,dim=1) # 
            alpha_div = torch.sum(const_alpha * (1 - log_prodterm.exp())) # batch, sen mean
            
            return alpha_div
    
    
    def renyi_divergence(self, alpha):
        if alpha == 1:
            raise Exception('The divergence is not well-defined when alpha = 1')
        
        exp_renyi = 1 + alpha * (alpha - 1 ) * self.alpha_divergence(alpha)

        return exp_renyi.log() / (alpha - 1)
