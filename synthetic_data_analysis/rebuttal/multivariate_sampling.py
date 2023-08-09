import os
import torch
import random
import numpy as np

from loss import log_t_normalizing_const
from util import make_reproducibility

def multivariate_t_sampling(N, mu, cov, nu, device) :
    MVN_dist = torch.distributions.MultivariateNormal(torch.zeros_like(mu), cov)
    eps = MVN_dist.sample(sample_shape=torch.tensor([N]))
    
    chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu]))
    v = chi_dist.sample(sample_shape=torch.tensor([N]))
    eps *= torch.sqrt(nu/v)

    return (mu + eps).to(device)

def multivariate_sample_generation(device, SEED = None, 
        K = 1, N = 1000, ratio_list = [1.0], mu_list = [None], var_list = [None], nu_list = [3.0]) : 
    if SEED is not None : 
        make_reproducibility(SEED)

    N_list = np.random.multinomial(N, ratio_list)
    result_list = [multivariate_t_sampling(N_list[ind], mu_list[ind], var_list[ind], nu_list[ind], device) for ind in range(K)]
    result = torch.cat(result_list)
    shuffled_ind = torch.randperm(result.shape[0])
    return result[shuffled_ind]

def multivariate_t_density(x, nu, mu = torch.zeros(1), var = torch.ones(1,1)) : 
    d = x.shape[1]

    precision = torch.linalg.inv(var)
    mahalanobis_dist = torch.sum(torch.matmul(mu - x, precision) * (mu - x), dim = 1)

    log_const_term = log_t_normalizing_const(nu, d)
    deter_term = torch.linalg.det(var)
    power_term = -torch.log(1 + mahalanobis_dist / nu) * (nu + d) / 2
    return torch.exp(log_const_term + power_term) / torch.sqrt(deter_term)

def multivariate_t_density_contour(x, K, sample_nu_list, mu_list, var_list, ratio_list) : 
    output = 0
    for ind in range(K) : 
        output += ratio_list[ind] * multivariate_t_density(x, sample_nu_list[ind], mu_list[ind], var_list[ind])
    return output
    
