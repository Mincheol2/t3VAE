import torch
import numpy as np
import matplotlib.pyplot as plt

from loss_1D import log_t_normalizing_const
from sampling_1D import t_density, t_density_contour

def visualize_density(model_nu_list, t3VAE_gen_list, VAE_gen, 
                      K, sample_nu_list, mu_list, var_list, ratio_list, xlim) :
    t3VAE_gen_list = [t3VAE_gen[torch.isfinite(t3VAE_gen)].cpu().numpy() for t3VAE_gen in t3VAE_gen_list]
    VAE_gen = VAE_gen[torch.isfinite(VAE_gen)].cpu().numpy()

    M = len(t3VAE_gen_list)
    input = np.arange(-xlim * 100, xlim * 100 + 1) * 0.01
    contour = t_density_contour(input, K, sample_nu_list, mu_list, var_list, ratio_list).squeeze().numpy()

    # plot
    fig = plt.figure(figsize = (3.5 * (M+1), 7))

    ax = fig.add_subplot(2,M+1,1)
    plt.plot(input, contour, color='black')
    plt.hist(VAE_gen, bins = 100, range = [-10, 10], density=True, alpha = 0.5, color='dodgerblue')
    plt.xlim(-10, 10)
    plt.title('VAE')

    ax = fig.add_subplot(2,M+1,M+2)
    plt.plot(input, contour, color='black')
    plt.hist(VAE_gen, bins = 100, range = [-xlim, xlim], density=True, alpha = 0.5, color='dodgerblue')
    plt.xlim(-xlim, xlim)
    plt.yscale("log")
    plt.ylim(1e-6, 1)

    for m in range(M) : 
        ax = fig.add_subplot(2,M+1,m+2)
        plt.plot(input, contour, color='black')
        plt.hist(t3VAE_gen_list[m], bins = 100, range = [-10, 10], density=True, alpha = 0.5, color='dodgerblue')
        plt.xlim(-10, 10)
        plt.title(f't3VAE (nu = {model_nu_list[m]})')

        ax = fig.add_subplot(2,M+1,M+m+3)
        plt.plot(input, contour, color='black')
        plt.hist(t3VAE_gen_list[m], bins = 100, range = [-xlim, xlim], density=True, alpha = 0.5, color='dodgerblue')
        plt.xlim(-xlim, xlim)
        plt.yscale("log")
        plt.ylim(1e-6, 1)

    return fig