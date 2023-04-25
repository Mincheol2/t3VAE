import os
import random
import numpy as np
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mmd import mmd_unbiased_sq, mmd_uniform_bound, make_masking, mmd_bootstrap_test
from simul_util import make_result_dir, make_reproducibility, t_sampling, sample_generation, t_density, density_contour, MYTensorDataset
from simul_loss import log_t_normalizing_const, gamma_regularizer
from simul_model import Encoder, Decoder, gammaAE
from simul_visualize import visualize, visualize_PCA, visualize_3D, visualize_2D, visualize_density

def simulation_1D(p_dim, q_dim, model_nu_list, recon_sigma, 
                  K, sample_nu_list, train_N_list, test_N_list, 
                  dir_name, device, 
                  epochs, num_layers, batch_size, lr, eps, weight_decay, 
                  train_data_seed, test_data_seed, model_init_seed, 
                  mu_list = None, var_list = None, param_seed = None, bootstrap_iter = 1999, gen_N = 100000) : 

    # Step 0. Environment setup
    M = len(model_nu_list)

    test_N = sum(test_N_list)
    ratio_list = [n_k / test_N for n_k in test_N_list]

    dirname = f'./{dir_name}'
    make_result_dir(dirname)
    generation_writer = SummaryWriter(dirname + '/generations')
    gAE_writer_list = [SummaryWriter(dirname + f'/gAE_nu{model_nu}') for model_nu in model_nu_list]
    # gAE_writer = SummaryWriter(dirname + '/gAE')
    VAE_writer = SummaryWriter(dirname + '/VAE')

    # Step 1. Sampling data
    if mu_list is None and var_list is None and param_seed is not None : 
        make_reproducibility(param_seed)
        mu_list = [torch.randn(p_dim) * 3 for ind in range(K)]
        var_list = [torch.eye(p_dim) for ind in range(K)]
        # A_list = [torch.randn(p_dim, q_dim) for ind in range(K)]
        # var_list = [recon_sigma**2 * torch.eye(p_dim) + A @ A.T for A in A_list]

    train_data = sample_generation(
        device, p_dim=p_dim, SEED=train_data_seed,
        K=K, N_list=train_N_list, mu_list=mu_list, var_list=var_list, nu_list=sample_nu_list
    )

    test_data = sample_generation(
        device, p_dim=p_dim, SEED=test_data_seed,
        K=K, N_list=test_N_list, mu_list=mu_list, var_list=var_list, nu_list=sample_nu_list
    )

    train_dataset = MYTensorDataset(train_data)

    # Step 2. Model initialization
    make_reproducibility(model_init_seed)

    gAE_list = [
        gammaAE(train_dataset, test_data, p_dim, q_dim, model_nu, recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay) for model_nu in model_nu_list
    ]
    # gAE = gammaAE(train_dataset, test_data, p_dim, q_dim, model_nu, recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay)
    VAE = gammaAE(train_dataset, test_data, p_dim, q_dim, 0,  recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay)

    # Step 3. Model training
    epoch_tqdm = tqdm(range(0, epochs))
    for epoch in epoch_tqdm : 

        # print(f'\nEpoch {epoch}')
        [gAE_list[m].train(epoch, gAE_writer_list[m]) for m in range(M)]
        VAE.train(epoch, VAE_writer)

        [gAE_list[m].test(epoch, gAE_writer_list[m]) for m in range(M)]
        VAE.test(epoch, VAE_writer)

        if epoch % 5 == 0:
            # Generation & Randomly reconstruction
            gAE_gen_list = [gAE.generate(gen_N).detach() for gAE in gAE_list]
            VAE_gen = VAE.generate(gen_N).detach()

            # Visualization
            visualization = visualize_density(train_data, test_data, model_nu_list, gAE_gen_list, VAE_gen, K, sample_nu_list, mu_list, var_list, ratio_list)

            generation_writer.add_figure("Generation", visualization, epoch)
            filename = f'{dirname}/generations/epoch{epoch}.png'
            visualization.savefig(filename)

            # MMD score
            gAE_mmd_result = [mmd_bootstrap_test(gAE_gen[0:test_N], test_data, device = device, iteration = bootstrap_iter) for gAE_gen in gAE_gen_list]
            gAE_stat_list = [result[0] for result in gAE_mmd_result]
            gAE_p_value_list = [result[1] for result in gAE_mmd_result]
            VAE_stat, VAE_p_value, _ = mmd_bootstrap_test(VAE_gen[0:test_N], test_data, device = device, iteration = bootstrap_iter)
            # gAE_stat, gAE_p_value, _ = mmd_bootstrap_test(gAE_gen[0:test_N], test_data, device = device, iteration = bootstrap_iter)

            for m in range(M) : 
                gAE_writer_list[m].add_scalar("Test/MMD score", gAE_stat_list[m], epoch)
                gAE_writer_list[m].add_scalar("Test/MMD p-value", gAE_p_value_list[m], epoch)

            VAE_writer.add_scalar("Test/MMD score", VAE_stat, epoch)
            VAE_writer.add_scalar("Test/MMD p-value", VAE_p_value, epoch)

    return None


# def simulation(dir_name, K, nu_list, train_N_list, test_N_list, train_data_seed, test_data_seed, 
#                p_dim, q_dim, nu, recon_sigma, model_init_seed, device, 
#                epochs, num_layers, lr, batch_size, eps, weight_decay, 
#                b_list = None, var_list = None, param_seed = None, bootstrap_iter = 1999) : 

#     # Step 0. Environment setup
#     test_N = sum(test_N_list)

#     dirname = f'./{dir_name}'
#     make_result_dir(dirname)
#     generation_writer = SummaryWriter(dirname + '/generations')
#     # criterion_writer = SummaryWriter(dirname + '/criterion')
#     gAE_writer = SummaryWriter(dirname + '/gAE')
#     VAE_writer = SummaryWriter(dirname + '/VAE')

#     visualize_module = visualize(p_dim)

#     # Step 1. Sampling data
#     if b_list is None and var_list is None and param_seed is not None : 
#         make_reproducibility(param_seed)
#         b_list = [torch.randn(p_dim) for ind in range(K)]
#         A_list = [torch.randn(p_dim, q_dim) for ind in range(K)]
#         var_list = [recon_sigma**2 * torch.eye(p_dim) + A @ A.T for A in A_list]

#     train_data = sample_generation(
#         device, p_dim=p_dim, SEED=train_data_seed,
#         K=K, N_list=train_N_list, mu_list=b_list, var_list=var_list, nu_list=nu_list
#     )

#     test_data = sample_generation(
#         device, p_dim=p_dim, SEED=test_data_seed,
#         K=K, N_list=test_N_list, mu_list=b_list, var_list=var_list, nu_list=nu_list
#     )

#     train_dataset = MYTensorDataset(train_data)


#     # Step 2. Model initialization
#     make_reproducibility(model_init_seed)

#     gAE = gammaAE(train_dataset, test_data, p_dim, q_dim, nu, recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay)
#     VAE = gammaAE(train_dataset, test_data, p_dim, q_dim, 0,  recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay)


#     # Step 3. Model training
#     epoch_tqdm = tqdm(range(0, epochs))
#     for epoch in epoch_tqdm : 

#         # print(f'\nEpoch {epoch}')
#         gAE.train(epoch, gAE_writer)
#         VAE.train(epoch, VAE_writer)

#         gAE.test(epoch, gAE_writer)
#         VAE.test(epoch, VAE_writer)

#         if epoch % 10 == 0:
#             # Generation & Randomly reconstruction
#             gAE_gen = gAE.generate(test_N).detach()
#             VAE_gen = VAE.generate(test_N).detach()

#             gAE_recon = gAE.reconstruct(test_data).detach()
#             VAE_recon = VAE.reconstruct(test_data).detach()

#             # Visualization
#             visualization = visualize_module.visualize(train_data, test_data, gAE_gen, VAE_gen, gAE_recon, VAE_recon)

#             generation_writer.add_figure("Generation", visualization, epoch)
#             filename = f'{dirname}/generations/epoch{epoch}.png'
#             visualization.savefig(filename)

#             # MMD score
#             gAE_stat, gAE_p_value = mmd_bootstrap_test(gAE_gen, test_data, device = device, iteration = bootstrap_iter)
#             VAE_stat, VAE_p_value = mmd_bootstrap_test(VAE_gen, test_data, device = device, iteration = bootstrap_iter)

#             # mmd_criterion = mmd_acceptance_region(test_N)

#             gAE_writer.add_scalar("Test/MMD score", gAE_stat, epoch)
#             gAE_writer.add_scalar("Test/MMD p-value", gAE_p_value, epoch)


#             VAE_writer.add_scalar("Test/MMD score", VAE_stat, epoch)
#             VAE_writer.add_scalar("Test/MMD p-value", VAE_p_value, epoch)
#             # criterion_writer.add_scalar("Test/MMD score", mmd_criterion, epoch)


#     return None