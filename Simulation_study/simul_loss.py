import torch
import numpy as np

def log_t_normalizing_const(nu, d):
    nom = torch.lgamma(torch.tensor((nu+d)/2))
    denom = torch.lgamma(torch.tensor(nu/2)) + d/2 * (np.log(nu) + np.log(np.pi))
    return nom - denom

def gamma_regularizer(mu, logvar, n_dim, const_2bar1, gamma, tau, nu):
    mu_norm_sq = torch.linalg.norm(mu, ord=2, dim=1).pow(2)
    trace_var = nu / (nu + n_dim - 2) * torch.sum(logvar.exp(),dim=1)
    log_det_var = -gamma / (2+2*gamma) * torch.sum(logvar,dim=1)

    return torch.mean(mu_norm_sq + trace_var - nu * const_2bar1 * log_det_var.exp() + nu * tau)