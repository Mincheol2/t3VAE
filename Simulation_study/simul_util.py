import os
import torch
import random
import numpy as np

def make_result_dir(dirname):
    os.makedirs(dirname,exist_ok=True)
    os.makedirs(dirname + '/gAE',exist_ok=True)
    os.makedirs(dirname + '/VAE',exist_ok=True)
    os.makedirs(dirname + '/generations',exist_ok=True)

def make_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def t_sampling(N, mu, cov, nu, device):
    '''
    N      : sample size
    D      : dimension
    mu     : torch tensor [D, ]
    cov    : torch tensor [D x D] (positive definite matrix)
    nu     : degree of freedom 
    device : 'cuda:0', 'cpu', etc
    '''

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
    if N_list is None : 
        N_list = [default_N for ind in range(K)]
    if mu_list is None : 
        mu_list = [torch.randn(p_dim) for ind in range(K)]
    if var_list is None : 
        var_list = [torch.randn(2 * p_dim, p_dim) for ind in range(K)]
        var_list = [X.T @ X for X in var_list]
    if nu_list is None : 
        nu_list = [default_nu for ind in range(K)]
    
    result_list = [t_sampling(N_list[ind], mu_list[ind], var_list[ind], nu_list[ind], device) for ind in range(K)]
    return torch.cat(result_list)

class MYTensorDataset(torch.utils.data.Dataset) :
    def __init__(self, *tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
