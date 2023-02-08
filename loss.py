import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import argument

args = argument.args
'''
    Gamma divergence assumes that
    prior p ~ t_q(0,I,nu) and posterior q ~ t_q(mu, Sigma, nu + p_dim).
    (Student's T distribution)
    
    For numerical stablity, we use an argument 'logvar' instead of var.
    Note that var is a diagonal matrix.
    '''


'''
    log ver of normalizing constant for t-distribution
'''

def log_t_normalizing_const(nu, d):
    nom = torch.lgamma(torch.tensor(nu+d))
    denom = torch.lgamma(torch.tensor(nu/2)) + d/2 * torch.log(torch.tensor(nu * np.pi))
    return nom - denom

'''
    gamma_power divergence : C_1 gamma reconstruction term - C_2 (gamma cross entropy term)
'''
def gamma_recon_error(recon_x, z, x, p_dim):
    '''
        parameters
        f_z : reconstructed data
        z : latent var
        x : original data
        
        Return : closed form
        E_{X~p_data}E_{Z_q} [ 1 + 1/nu||Z||^2 + 1/(v+q) 1/sigma^2 ||X-f_{theta}(Z)||^2]
    '''
    f_z = torch.flatten(recon_x, start_dim = 1)
    z = torch.flatten(z, start_dim = 1) # dim : [B, input_dim]
    x = torch.flatten(x, start_dim = 1)
    q_dim = args.zdim
    gamma = -2 / (args.nu + p_dim + q_dim)


    z_norm = torch.linalg.norm(z, ord=2, dim=1) #ok
    x_diff_z_norm = torch.linalg.norm(x-f_z, ord=2, dim=1) #ok

    normalizing_term = log_t_normalizing_const(args.nu, p_dim + q_dim)
    const_pow_log = - np.log(normalizing_term) +  p_dim/2 * np.log(1+ q_dim/args.nu) + np.log(1+ (p_dim+q_dim)/(args.nu-2))
    
    const = -1/gamma * torch.exp(- gamma / (1+gamma) * const_pow_log)
    return const * torch.mean(1 + 1/args.nu * z_norm + 1/((args.nu+q_dim) * (args.recon_sigma)**2 * x_diff_z_norm))

def gamma_neg_entropy(logvar, p_dim):
    '''

        method 1 : ignoring gamma above p_data(x).
        So, we calculate 1/N sum_{i=1}^{N} |Sigma(X_i)|^{-gamma /2}

        method 2 : use convergence of L^{1+u} quasi norm.
        In this case, we can also ignore the global exponent.

    '''

    q_dim = args.zdim
    gamma = -2 / (args.nu + p_dim + q_dim)

    const_1_log = log_t_normalizing_const(args.nu + p_dim, q_dim) * (gamma/ (1+gamma))
    const_2 = (1 + q_dim/(args.nu+p_dim-2)) ** (1/(1+gamma)) # almost 1
    
    const = -1/gamma * torch.exp(const_1_log) * const_2

    if args.method == 1:
        eps = 1e-10
        # Sigma_X_det = torch.prod(torch.exp(logvar),dim=1) # determinant
        # det -> 0으로 수렴하는 문제 발생.. why?

        # Sigma_X_pow = torch.pow(Sigma_X_det,-gamma/2)
        Sigma_X_pow = torch.exp(-gamma/2 * torch.sum(logvar, dim=1))
        return const * torch.pow(torch.mean(Sigma_X_pow), 1 / (1+ gamma))

    elif args.method == 2:
        # Sigma_X_det = torch.prod(torch.exp(logvar), dim=1)
        # Sigma_X_pow = torch.pow(Sigma_X_det,-gamma/2)

        Sigma_X_pow = torch.exp(-gamma/2 * torch.sum(logvar,dim=1))
        return const * torch.mean(Sigma_X_pow)

    elif args.method == 3:
        pass
    else:
        raise Exception('Please choose one of the appropriate methods.')



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
        
        gamma_div = torch.mean(log_det_ratio + log_term) # Batch mean
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
