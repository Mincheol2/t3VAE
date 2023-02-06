import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

'''
    Gamma divergence assumes that
    prior p ~ T_{df}(0,I) and posterior q ~ T_{df}(mu, var).
    (Student's T distribution)
    * Note that var is diagonal.
    
    For numerical stablity, we use an argument 'logvar' instead of var.
    You can change prior's mean and variance by modifying the argument 'prior_mu' and 'prior_logvar'.
'''

class Gamma_Power_Family():
    def __init__(self, post_mu, post_logvar, nu, prior_mu=None, prior_logvar=None):
        self.post_mu = torch.tensor(post_mu)
        self.post_logvar = torch.tensor(post_logvar)
        self.prior_mu = torch.zeros_like(post_mu) if prior_mu is None else torch.tensor(prior_mu)
        self.prior_logvar = torch.zeros_like(post_logvar) if prior_logvar is None else torch.tensor(prior_logvar)
        self.post_var = self.post_logvar.exp()
        self.prior_var = self.prior_logvar.exp()
        
        # dimension & params
        self.nu = nu
        self.p = self.prior_mu.shape[1]
        self.q = self.post_mu.shape[1]
        self.gamma = - 2 / (self.nu + self.p + self.q) # gamma < 0
        
    def gamma_entropy(p):
        '''
            parameter : p (distribution)
            Return : H_{\gamma}(p) = -||p||_{1+\gamma}
        '''
        p_power = torch.pow(p, 1+self.gamma)
        norm_value = torch.pow(torch.mean(p_power, dim=1), 1/(1+self.gamma))
        return - norm_value
    def gamma_ce(p, q):
        '''
            parameter : p, q (distributions)
            Return : C_{\gamma}(p,q)
            = - 1/||q||_{1+\gamma} E_p[q^r]
        '''
        
        q_negent_pow = torch.pow(- gamma_entropy(q), self.gamma)
        integrand = p * torch.pow(q,gamma)
        return - torch.mean(integrand, dim=1) / q_entropy_pow
        
    def gamma_pow_div(p, q):
        return gamma_ce(p, q) - gamma_entropy(p)
    


class Gamma_Family():
    def __init__(self, post_mu, post_logvar, df, prior_mu=None, prior_logvar=None):
        self.post_mu = torch.tensor(post_mu)
        self.post_logvar = torch.tensor(post_logvar)
        self.df = df
        
        self.prior_mu = torch.zeros_like(post_mu) if prior_mu is None else torch.tensor(prior_mu)
        self.prior_logvar = torch.zeros_like(post_logvar) if prior_logvar is None else torch.tensor(prior_logvar)
        self.post_var = self.post_logvar.exp()
        self.prior_var = self.prior_logvar.exp()

    def gamma_divergence(self, df):
        '''
        Generalized gamma divergence.
        paramter : df > 2
        (This is because the variance of T dist : df/(df-2), v>2)
        cf) Instead of gamma, we use df(degree of freedom) as a paramter. (gamma = -2 /(1+df))
        '''
        # Check the well-definedness
        if df <= 2:
            raise Exception(f'the degree of freedom is not larger than 2. Divergence is not well-defined.')


        # dimension : [B : batch size, D : zdim]
        zdim = self.post_mu.shape[1]
        log_det_ratio = (df + zdim) / (2*(df + zdim - 2)) * (torch.sum(self.prior_logvar,dim=1) - torch.sum(self.post_logvar,dim=1))
        log_term = (df + zdim)/2 * torch.log(1 + 1/(df-2) * torch.sum( self.post_var / self.prior_var,dim=1) + 1/df * torch.sum( (self.prior_mu-self.post_mu).pow(2) / self.prior_var,dim=1))
        
        gamma_div = torch.mean(log_det_ratio + log_term) # Batch mean
        return gamma_div
    


'''
    Alpha divergence assumes that
    prior p ~ N(0,I) and posterior q ~ N(mu, var). (Note that var is diagonal.)
    For numerical stablity, we use an argument 'logvar' instead of var.
    You can change prior's mean and variance by modifying the argument 'prior_mu' and 'prior_logvar'.
'''

class Alpha_Family():
    def __init__(self, post_mu, post_logvar, prior_mu=None, prior_logvar=None):
        self.post_mu = torch.tensor(post_mu)
        self.post_logvar = torch.tensor(post_logvar)
        self.prior_mu = torch.zeros_like(post_mu) if prior_mu is None else torch.tensor(prior_mu)
        self.prior_logvar = torch.zeros_like(post_logvar) if prior_logvar is None else torch.tensor(prior_logvar)
        self.post_var = self.post_logvar.exp()
        self.prior_var = self.prior_logvar.exp()


    def KL_loss(self, is_reversed=False):
        kl_div = 0
        logvar_diff = self.post_logvar - self.prior_logvar
        mu_square = (self.post_mu - self.prior_mu).pow(2)
        if is_reversed:
            sq_term = (self.prior_var + mu_square) / self.post_var
            kl_div = -0.5 * torch.mean(- logvar_diff + 1.0 - sq_term)
        else:
            sq_term = (self.post_var + mu_square) / self.prior_var
            kl_div = -0.5 * torch.mean(logvar_diff + 1.0 - sq_term)
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
            alpha_div = torch.mean(const_alpha * (1 - log_prodterm.exp())) # batch, sen mean
            
            return alpha_div
    
    
    def renyi_divergence(self, alpha):
        if alpha == 1:
            raise Exception('The divergence is not well-defined when alpha = 1')
        
        exp_renyi = 1 + alpha * (alpha - 1 ) * self.alpha_divergence(alpha)

        return exp_renyi.log() / (alpha - 1)
