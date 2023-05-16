import torch
import numpy as np
from tqdm import tqdm

def mmd_unbiased_sq(z_hat, z, device, sigma2_k = None):
    n = z.shape[0]
    # zdim = z.shape[1]
    half_size = int((n * n - n)/2)
    
    norms_z = z.pow(2).sum(1).unsqueeze(1)
    dots_z = torch.mm(z, z.t())
    dists_z = (norms_z + norms_z.t() - 2. * dots_z).abs()
    
    norms_zh = z_hat.pow(2).sum(1).unsqueeze(1)
    dots_zh = torch.mm(z_hat, z_hat.t())
    dists_zh = (norms_zh + norms_zh.t() - 2. * dots_zh).abs()
    
    dots = torch.mm(z_hat, z.t())
    dists = (norms_zh + norms_z.t() - 2. * dots).abs()
    
    if sigma2_k is None : 
        sigma2_k = torch.topk(dists.reshape(-1), half_size)[0][-1] + torch.topk(dists_zh.reshape(-1), half_size)[0][-1]
    
    res1 = torch.exp(-dists_zh/2./sigma2_k)
    res1 = res1 + torch.exp(-dists_z/2./sigma2_k)
    res1 = torch.mul(res1, 1. - torch.eye(n).to(device))
    res1 = res1.sum() / (n*n-n)
    res2 = torch.exp(-dists/2./sigma2_k)
    res2 = res2.sum()*2./(n*n)
    stat = res1 - res2
    return stat

def mmd_linear(z_hat, z, sigma2_k = None):
    n = int(z.shape[0] / 2)
    z_hat_1 = z_hat[0:n]
    z_hat_2 = z_hat[n:2*n]

    z_1 = z[0:n]
    z_2 = z[n:2*n]

    term_1 = (z_hat_1 - z_hat_2).pow(2).sum(1)
    term_2 = (z_1 - z_2).pow(2).sum(1)
    term_3 = (z_hat_1 - z_2).pow(2).sum(1)
    term_4 = (z_hat_2 - z_1).pow(2).sum(1)

    if sigma2_k is None : 
        sigma2_k = torch.cat([term_1, term_2, term_3, term_4]).topk(2 * n)[0][-1]

    res1 = torch.mean(torch.exp(-term_1/2./sigma2_k))
    res2 = torch.mean(torch.exp(-term_2/2./sigma2_k))
    res3 = torch.mean(torch.exp(-term_3/2./sigma2_k))
    res4 = torch.mean(torch.exp(-term_4/2./sigma2_k))
    return res1 + res2 - res3 - res4

def make_masking(n) : 
    indice = np.arange(0,2*n)
    mask = np.zeros(2*n,dtype=bool)
    rand_indice = np.random.choice(2*n, n, replace = False)
    mask[rand_indice] = True
    
    return indice[mask], indice[~mask]

def mmd_unbiased_bootstrap_test(z_hat, z, device, sigma2=None, iteration=1999) : 
    n = z.shape[0]
    half_size = int((n * n - n)/2)
    
    norms_z = z.pow(2).sum(1).unsqueeze(1)
    norms_zh = z_hat.pow(2).sum(1).unsqueeze(1)

    dots_zh = torch.mm(z_hat, z_hat.t())
    dists_zh = (norms_zh + norms_zh.t() - 2. * dots_zh).abs()
    
    dots = torch.mm(z_hat, z.t())
    dists = (norms_zh + norms_z.t() - 2. * dots).abs()
    
    if sigma2 is None : 
        sigma2 = torch.topk(dists.reshape(-1), half_size)[0][-1] + torch.topk(dists_zh.reshape(-1), half_size)[0][-1]

    mmd_stat = mmd_unbiased_sq(z_hat, z, device, sigma2).item()
    mmd_bootstrap_list = []
    full_data = torch.cat([z_hat, z], dim = 0)

    for _ in range(iteration) : 
        ind_1, ind_2 = make_masking(n)
        z_1 = full_data[ind_1]
        z_2 = full_data[ind_2]
        mmd_bootstrap_list.append(mmd_unbiased_sq(z_1, z_2, device, sigma2).item())

    sum([int(stat > mmd_stat) for stat in mmd_bootstrap_list])
    p_value = (1 + sum([int(stat > mmd_stat) for stat in mmd_bootstrap_list])) / (1 + iteration)

    return mmd_stat, p_value, mmd_bootstrap_list

def mmd_linear_bootstrap_test(z_hat, z, device, sigma2=None, iteration=1999) : 
    n = int(z.shape[0] / 2)
    z_hat_1 = z_hat[0:n]
    z_hat_2 = z_hat[n:2*n]

    z_1 = z[0:n]
    z_2 = z[n:2*n]

    term_1 = (z_hat_1 - z_hat_2).pow(2).sum(1)
    term_2 = (z_1 - z_2).pow(2).sum(1)
    term_3 = (z_hat_1 - z_2).pow(2).sum(1)
    term_4 = (z_hat_2 - z_1).pow(2).sum(1)

    if sigma2 is None : 
        sigma2 = torch.cat([term_1, term_2, term_3, term_4]).topk(2 * n)[0][-1]

    mmd_stat = mmd_linear(z_hat, z, sigma2).item()
    mmd_bootstrap_list = []
    full_data = torch.cat([z_hat, z], dim = 0)

    for _ in range(iteration) : 
        ind_1, ind_2 = make_masking(n)
        z_1 = full_data[ind_1]
        z_2 = full_data[ind_2]
        mmd_bootstrap_list.append(mmd_linear(z_1, z_2, sigma2).item())

    sum([int(stat > mmd_stat) for stat in mmd_bootstrap_list])
    p_value = (1 + sum([int(stat > mmd_stat) for stat in mmd_bootstrap_list])) / (1 + iteration)

    return mmd_stat, p_value, mmd_bootstrap_list
