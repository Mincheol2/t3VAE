import torch

def mmd_penalty(z_hat, z, kernel="RBF", sigma2_p=1):
    n = z.shape[0]
    zdim = z.shape[1]
    half_size = int((n * n - n)/2)
    
    norms_z = z.pow(2).sum(1).unsqueeze(1)
    dots_z = torch.mm(z, z.t())
    dists_z = (norms_z + norms_z.t() - 2. * dots_z).abs()
    
    norms_zh = z_hat.pow(2).sum(1).unsqueeze(1)
    dots_zh = torch.mm(z_hat, z_hat.t())
    dists_zh = (norms_zh + norms_zh.t() - 2. * dots_zh).abs()
    
    dots = torch.mm(z_hat, z.t())
    dists = (norms_zh + norms_z.t() - 2. * dots).abs()
    
    if kernel == "RBF":
        sigma2_k = torch.topk(dists.reshape(-1), half_size)[0][-1]
        sigma2_k = sigma2_k + torch.topk(dists_zh.reshape(-1), half_size)[0][-1]
        
        res1 = torch.exp(-dists_zh/2./sigma2_k)
        res1 = res1 + torch.exp(-dists_z/2./sigma2_k)
        res1 = torch.mul(res1, 1. - torch.eye(n))
        res1 = res1.sum() / (n*n-n)
        res2 = torch.exp(-dists/2./sigma2_k)
        res2 = res2.sum()*2./(n*n)
        stat = res1 - res2
        return stat
    elif kernel == "IMQ":
        Cbase = 2 * zdim * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + dists_z) + C / (C + dists_zh)
            res1 = torch.mul(res1, 1. - torch.eye(n))
            res1 = res1.sum() / (n*n-n)
            res2 = C / (C + dists)
            res2 = res2.sum()*2./(n*n)
            stat = stat + res1 - res2
        return stat