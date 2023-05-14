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

def sample_generation(device, p_dim = 3, SEED = None, 
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
    '''z = x sin y + noise'''
    # result = x * torch.sin(y)
    # noise = torch.randn(2*sample_N, 3) / torch.tensor(np.sqrt(np.random.chisquare(nu, 2*sample_N) / nu)).unsqueeze(1)
    # w = torch.concat([xy, result.unsqueeze(1)], dim = 1) - noise
    result = x * torch.sin(y)
    w = torch.concat([xy, result.unsqueeze(1)], dim = 1)

    return xy.to(device).float() / 5., w.to(device).float()/5.

def latent_generate_2D(sample_N = 10000, nu = 5, SEED = None, device = 'cpu') : 
    if SEED is not None : 
        make_reproducibility(SEED)

    nu = 5
    e1 = torch.tensor([1])
    z1 = torch.randn(sample_N, 1) - 2.5 * e1
    z2 = torch.randn(sample_N, 1) + 2.5 * e1

    xy = torch.concat([z1 * 3, z2 * 3])
    x = xy[:,0]
    result = x * torch.sin(x)
    w = torch.concat([xy, result.unsqueeze(1)], dim = 1)

    return xy.to(device).float() / 5., w.to(device).float()/5.

class MYTensorDataset(torch.utils.data.Dataset) :
    def __init__(self, *tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
    
