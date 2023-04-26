import os
import torch
import random
import numpy as np

from simul_loss import log_t_normalizing_const

def make_result_dir(dirname):
    os.makedirs(dirname, exist_ok=True)
    os.makedirs(dirname + '/gAE', exist_ok=True)
    os.makedirs(dirname + '/VAE', exist_ok=True)
    os.makedirs(dirname + '/generations', exist_ok=True)

def make_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def t_sampling(N, mu, cov, nu, device) :
    MVN_dist = torch.distributions.MultivariateNormal(torch.zeros_like(mu), cov)
    eps = MVN_dist.sample(sample_shape=torch.tensor([N]))
    
    if nu != 0 : 
        chi_dist = torch.distributions.chi2.Chi2(torch.tensor([nu]))
        # v = chi_dist.sample()
        v = chi_dist.sample(sample_shape=torch.tensor([N]))
        eps *= torch.sqrt(nu/v)

    return (mu + eps).to(device)

def sample_generation(device, p_dim = 3, SEED = None, K = 1, default_N = 1000, default_nu = 5, N_list = None, mu_list = None, var_list = None, nu_list = None) : 
    if SEED is not None : 
        make_reproducibility(SEED)
    
    result_list = [t_sampling(N_list[ind], mu_list[ind], var_list[ind], nu_list[ind], device) for ind in range(K)]
    return torch.cat(result_list)

def t_density(x, nu, mu = torch.zeros(1), var = torch.ones(1,1)) : 
    const_term = log_t_normalizing_const(nu, 1)
    power_term = -torch.log(1 + (mu - x).pow(2) / (nu * var)) * (nu + 1) / 2
    return torch.exp(const_term + power_term)

def density_contour(x, K, sample_nu_list, mu_list, var_list, ratio_list) : 
    output = 0
    for ind in range(K) : 
        output += ratio_list[ind] * t_density(x, sample_nu_list[ind], mu_list[ind], var_list[ind])
    return output


class MYTensorDataset(torch.utils.data.Dataset) :
    def __init__(self, *tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
