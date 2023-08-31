import os
import torch
import random
import numpy as np

from loss_1D import log_t_normalizing_const
from util_1D import make_reproducibility

def t_sampling(N, mu, cov, nu, device) :
    MVN_dist = torch.distributions.MultivariateNormal(torch.zeros_like(mu), cov)
    eps = MVN_dist.sample(sample_shape=torch.tensor([N]))
    
    if nu != 0 : 
        chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu]))
        v = chi_dist.sample(sample_shape=torch.tensor([N]))
        eps *= torch.sqrt(nu/v)

    return (mu + eps).to(device)

def sample_generation(device, SEED = None, 
                      K = 1, N = 1000, ratio_list = [1.0], mu_list = None, var_list = None, nu_list = None) : 
    if SEED is not None : 
        make_reproducibility(SEED)

    N_list = np.random.multinomial(N, ratio_list)
    result_list = [t_sampling(N_list[ind], mu_list[ind], var_list[ind], nu_list[ind], device) for ind in range(K)]
    result = torch.cat(result_list)
    shuffled_ind = torch.randperm(result.shape[0])
    return result[shuffled_ind]

def t_density(x, nu, mu = torch.zeros(1), var = torch.ones(1,1)) : 
    if nu == 0 : 
        const_term = - 0.5 * np.log(2 * np.pi)
        exp_term = - 0.5 * (mu - x).pow(2) / var
        return torch.exp(const_term + exp_term) / torch.sqrt(var)
    else : 
        const_term = log_t_normalizing_const(nu, 1)
        power_term = -torch.log(1 + (mu - x).pow(2) / (nu * var)) * (nu + 1) / 2
        return torch.exp(const_term + power_term) / torch.sqrt(var)

def t_density_contour(x, K, sample_nu_list, mu_list, var_list, ratio_list) : 
    output = 0
    for ind in range(K) : 
        output += ratio_list[ind] * t_density(x, sample_nu_list[ind], mu_list[ind], var_list[ind])
    return output
    
