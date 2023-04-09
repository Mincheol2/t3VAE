import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mmd import mmd_penalty, mmd_acceptance_region, mmd_prob_bound
from simul_util import make_result_dir, make_reproducibility, t_sampling, sample_generation, MYTensorDataset
from simul_loss import log_t_normalizing_const, gamma_regularizer
from simul_model import Encoder, Decoder, gammaAE
from simul_visualize import visualize, visualize_PCA, visualize_3D, visualize_2D

def simulation(index, K, nu_list, train_N_list, test_N_list, train_data_seed, test_data_seed, 
               p_dim, q_dim, nu, recon_sigma, model_init_seed, device, 
               epochs, num_layers, lr, batch_size, eps, weight_decay, 
               b_list = None, var_list = None, param_seed = None) : 

    # Step 0. Environment setup
    test_N = sum(test_N_list)

    dirname = f'./Results_{index}'
    make_result_dir(dirname)
    generation_writer = SummaryWriter(dirname + '/generations')
    # criterion_writer = SummaryWriter(dirname + '/criterion')
    gAE_writer = SummaryWriter(dirname + '/gAE')
    VAE_writer = SummaryWriter(dirname + '/VAE')

    visualize_module = visualize(p_dim)

    # Step 1. Sampling data
    if b_list is None and var_list is None and param_seed is not None : 
        make_reproducibility(param_seed)
        b_list = [torch.randn(p_dim) for ind in range(K)]
        A_list = [torch.randn(p_dim, q_dim) for ind in range(K)]
        var_list = [recon_sigma**2 * torch.eye(p_dim) + A @ A.T for A in A_list]

    train_data = sample_generation(
        device, p_dim=p_dim, SEED=train_data_seed,
        K=K, N_list=train_N_list, mu_list=b_list, var_list=var_list, nu_list=nu_list
    )

    test_data = sample_generation(
        device, p_dim=p_dim, SEED=test_data_seed,
        K=K, N_list=test_N_list, mu_list=b_list, var_list=var_list, nu_list=nu_list
    )

    train_dataset = MYTensorDataset(train_data)


    # Step 2. Model initialization
    make_reproducibility(model_init_seed)

    gAE = gammaAE(train_dataset, test_data, p_dim, q_dim, nu, recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay)
    VAE = gammaAE(train_dataset, test_data, p_dim, q_dim, 0,  recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay)


    # Step 3. Model training
    epoch_tqdm = tqdm(range(0, epochs))
    for epoch in epoch_tqdm : 

        # print(f'\nEpoch {epoch}')
        gAE.train(epoch, gAE_writer)
        VAE.train(epoch, VAE_writer)

        gAE.test(epoch, gAE_writer)
        VAE.test(epoch, VAE_writer)

        if epoch % 10 == 0:
            # Generation & Randomly reconstruction
            gAE_gen = gAE.generate(test_N).detach().cpu()
            VAE_gen = VAE.generate(test_N).detach().cpu()

            gAE_recon = gAE.reconstruct(test_data).detach().cpu()
            VAE_recon = VAE.reconstruct(test_data).detach().cpu()

            # Visualization
            visualization = visualize_module.visualize(train_data, test_data, gAE_gen, VAE_gen, gAE_recon, VAE_recon)

            generation_writer.add_figure("Generation", visualization, epoch)
            filename = f'{dirname}/generations/epoch{epoch}.png'
            visualization.savefig(filename)

            # MMD score
            gAE_mmd = mmd_penalty(torch.as_tensor(gAE_gen), test_data.cpu())
            VAE_mmd = mmd_penalty(torch.as_tensor(VAE_gen), test_data.cpu())

            # mmd_criterion = mmd_acceptance_region(test_N)

            gAE_writer.add_scalar("Test/MMD score", gAE_mmd.item(), epoch)
            VAE_writer.add_scalar("Test/MMD score", VAE_mmd.item(), epoch)
            # criterion_writer.add_scalar("Test/MMD score", mmd_criterion, epoch)


    return None