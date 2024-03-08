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

def nonlinear_sampling(device, SEED = None, K = 1, N = 1000, ratio_list = [1.0], mu_list = [None], var_list = [None], nu_list = [3.0]) : 
    x = multivariate_sample_generation(device, SEED , K, N, ratio_list, mu_list, var_list, nu_list)
    y = x + 2*torch.sin(x * torch.pi / 4)
    res = torch.cat([x,y], dim = 1)
    eps = torch.randn_like(res).to(device)
    v = torch.distributions.chi2.Chi2(torch.tensor([6])).sample(sample_shape=[x.shape[0]]).to(device)
    res += eps * torch.sqrt(6 / v)
    return res