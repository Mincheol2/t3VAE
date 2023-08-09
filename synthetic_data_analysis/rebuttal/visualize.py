import torch
import numpy as np
import matplotlib.pyplot as plt

from loss import log_t_normalizing_const
from sampling import t_density, t_density_contour

def visualize_density(model_title_list, model_gen_list, 
                      K, sample_nu_list, sample_mu_list, sample_var_list, ratio_list, xlim) :
    model_gen_list = [gen[torch.isfinite(gen)].cpu().numpy() for gen in model_gen_list]

    M = len(model_gen_list)
    input = np.arange(-xlim * 100, xlim * 100 + 1) * 0.01
    contour = t_density_contour(input, K, sample_nu_list, sample_mu_list, sample_var_list, ratio_list).squeeze().numpy()

    # plot
    fig = plt.figure(figsize = (3.5 * M, 7))

    for m in range(M) : 
        ax = fig.add_subplot(2,M,m+1)
        plt.plot(input, contour, color='black')
        plt.hist(model_gen_list[m], bins = 100, range = [-10, 10], density=True, alpha = 0.5, color='dodgerblue')
        plt.xlim(-10, 10)
        plt.title(f'{model_title_list[m]}')

        ax = fig.add_subplot(2,M,M+m+1)
        plt.plot(input, contour, color='black')
        plt.hist(model_gen_list[m], bins = 100, range = [-xlim, xlim], density=True, alpha = 0.5, color='dodgerblue')
        plt.xlim(-xlim, xlim)
        plt.yscale("log")
        plt.ylim(1e-6, 1)

    return fig


def visualize_verbose(model_title_list, model_var_phi) : 

    M = len(model_title_list)

    fig = plt.figure(figsize = (3.5 * M, 7))

    for m in range(M) : 
        ax = fig.add_subplot(2,M,m+1)
        plt.hist(model_var_phi[m], bins = 100, density=True, alpha = 0.5, color='dodgerblue')
        plt.title(f'var_phi of {model_title_list[m]}')

        ax = fig.add_subplot(2,M,M+m+1)
        plt.hist(model_var_phi[m], bins = 100, density=True, alpha = 0.5, color='dodgerblue')
        plt.yscale("log")
        plt.title(f'log scale histogram')

    return fig