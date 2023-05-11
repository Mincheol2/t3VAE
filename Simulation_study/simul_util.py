import os
import torch
import random
import numpy as np
import scipy.stats as stats

from simul_loss import log_t_normalizing_const

def make_result_dir(dirname):
    os.makedirs(dirname, exist_ok=True)
    # os.makedirs(dirname + '/gAE', exist_ok=True)
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

def sample_generation(device, p_dim = 3, SEED = None, K = 1, default_N = 1000, default_nu = 5, N_list = None, mu_list = None, var_list = None, nu_list = None, sample_type = "t") : 
    if SEED is not None : 
        make_reproducibility(SEED)
    
    if sample_type == "t" : 
        result_list = [t_sampling(N_list[ind], mu_list[ind], var_list[ind], nu_list[ind], device) for ind in range(K)]
        return torch.cat(result_list)

    if sample_type == "lognormal" : 
        result_list = [torch.exp(t_sampling(N_list[ind], mu_list[ind], var_list[ind], 0, device)) for ind in range(K)]
        return torch.cat(result_list)

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

def latent_generate(sample_N = 10000, nu = 5, SEED = None, device = 'cpu') : 
    if SEED is not None : 
        make_reproducibility(SEED)

    nu = 5
    e1 = torch.tensor([1,0])
    z1 = torch.randn(sample_N, 2) - 2.5 * e1
    z2 = torch.randn(sample_N, 2) + 2.5 * e1

    xy = torch.concat([z1 * 3, z2 * 3])
    x = xy[:,0]
    y = xy[:,1]
    result = x * y * torch.sin(x)
    noise = torch.randn_like(result) / torch.tensor(np.sqrt(np.random.chisquare(nu, result.shape) / nu))
    w = torch.concat([xy, (result - noise).unsqueeze(1)], dim = 1)

    return xy.to(device).float(), w.to(device).float()

# def latent_generate(sample_N, DEVICE, nu = 5, sigma = 1, seed = None) : 
#     if seed is not None : 
#         make_reproducibility(seed)
#     z1 = np.random.normal(size = int(sample_N / 2)) + 1
#     z2 = np.random.normal(size = int(sample_N / 2)) - 3

#     z = np.concatenate([np.exp(z1), z2])
#     z = torch.tensor(z).to(DEVICE)

#     noise = stats.t.rvs(nu, size = sample_N) * sigma
#     x = z + torch.tensor(noise).to(DEVICE)
#     return z.unsqueeze(1).float(), x.unsqueeze(1).float()

class MYTensorDataset(torch.utils.data.Dataset) :
    def __init__(self, *tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
    
