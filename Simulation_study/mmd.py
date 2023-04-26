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


def mmd_uniform_bound(n = 1000, alpha=0.05) : 
    return 4. * np.sqrt(-np.log(alpha) / n)


def make_masking(n) : 
    indice = np.arange(0,2*n)
    mask = np.zeros(2*n,dtype=bool)
    rand_indice = np.random.choice(2*n, n, replace = False)
    mask[rand_indice] = True
    
    return indice[mask], indice[~mask]


def mmd_bootstrap_test(z_hat, z, device, sigma2=None, iteration=1999) : 
    n = z.shape[0]
    half_size = int((n * n - n)/2)
    
    norms_z = z.pow(2).sum(1).unsqueeze(1)
    dots_z = torch.mm(z, z.t())
    dists_z = (norms_z + norms_z.t() - 2. * dots_z).abs()
    
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
