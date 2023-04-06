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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from simul_util import make_reproducibility, sampling, simulation, MYTensorDataset
from simul_loss import log_t_normalizing_const, gamma_regularizer
from simul_model import Encoder, Decoder, gammaAE

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device(f'cuda:0' if USE_CUDA else "cpu")

index = 158
K = 2
p_dim = 3
q_dim = 2
nu = 10
recon_sigma = 0.1
nu_list = [5, 2]
train_N_list = [4000, 1000]
test_N_list = [800, 200]
sample_N = 5000
epochs = 120
num_layers = 128
batch_size = 128
make_reproducibility(500)
sample_seed = 100
b_list = [torch.randn(p_dim) for ind in range(K)]
A_list = [torch.randn(p_dim, q_dim) for ind in range(K)]
var_list = [recon_sigma**2 * torch.eye(p_dim) + A @ A.T for A in A_list]
module_type = gammaAE


def train(index, K, nu_list, train_N_list, test_N_list, train_data_seed, test_data_seed, 
          p_dim, q_dim, nu, recon_sigma, model_init_seed, device, generation_seed, 
          num_layers, lr, batch_size, eps, weight_decay, 
          b_list = None, var_list = None, param_seed = None) : 
    
    # Step 1. Sampling data
    if b_list is None and N_list is None and param_seed is not None : 
        make_reproducibility(param_seed)
        b_list = [torch.randn(p_dim) for ind in range(K)]
        A_list = [torch.randn(p_dim, q_dim) for ind in range(K)]
        var_list = [recon_sigma**2 * torch.eye(p_dim) + A @ A.T for A in A_list]

    train_data = simulation(
        device, p_dim=p_dim, SEED=train_data_seed,
        K=K, N_list=train_N_list, mu_list=b_list, var_list=var_list, nu_list=nu_list
    )

    test_data = simulation(
        device, p_dim=p_dim, SEED=test_data_seed,
        K=K, N_list=test_N_list, mu_list=b_list, var_list=var_list, nu_list=nu_list
    )

    train_dataset = MYTensorDataset(train_data)

    # Step 2. Model initialization
    make_reproducibility(model_init_seed)

    gAE = gammaAE(train_dataset, test_data, p_dim, q_dim, nu, recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay)
    VAE = gammaAE(train_dataset, test_data, p_dim, q_dim, 0,  recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay)


    # Step 3. Model training
    train_gAE_total_loss = []
    train_gAE_recon_loss = []
    train_gAE_regul_loss = []    

    train_VAE_total_loss = []
    train_VAE_recon_loss = []
    train_VAE_regul_loss = []    
    
    test_gAE_total_loss = []
    test_gAE_recon_loss = []
    test_gAE_regul_loss = []

    test_VAE_total_loss = []
    test_VAE_recon_loss = []
    test_VAE_regul_loss = []

    epoch_tqdm = tqdm(range(0, epochs))
    for epoch in epoch_tqdm : 
        # print(f'\nEpoch {epoch}')
        gAE_total, gAE_recon, gAE_reg = gAE.train()
        VAE_total, VAE_recon, VAE_reg = VAE.train()

        train_gAE_total_loss.extend(gAE_total)
        train_gAE_recon_loss.extend(gAE_recon)
        train_gAE_regul_loss.extend(gAE_reg) 

        train_VAE_total_loss.extend(VAE_total)
        train_VAE_recon_loss.extend(VAE_recon)
        train_VAE_regul_loss.extend(VAE_reg)

        gAE_total, gAE_recon, gAE_reg = gAE.test()
        VAE_total, VAE_recon, VAE_reg = VAE.test()

        test_gAE_total_loss.append(gAE_total)
        test_gAE_recon_loss.append(gAE_recon)
        test_gAE_regul_loss.append(gAE_reg) 

        test_VAE_total_loss.append(VAE_total)
        test_VAE_recon_loss.append(VAE_recon)
        test_VAE_regul_loss.append(VAE_reg)


    # Step 4. Generation from trained models
    make_reproducibility(generation_seed)
    MVT_prior = sampling(N=sum(test_N_list), torch.zeros(q_dim), torch.eye(q_dim), nu, device)
    MVN_prior = sampling(N=sum(test_N_list), torch.zeros(q_dim), torch.eye(q_dim), 0,  device)

    gAE_reconstruction = gAE.decoder.sampling(gAE.encoder(test_data)[0]).detach().cpu().numpy()
    VAE_reconstruction = VAE.decoder.sampling(VAE.encoder(test_data)[0]).detach().cpu().numpy()

    gAE_generation = gAE.decoder.sampling(MVT_prior).detach().cpu().numpy()
    VAE_generation = VAE.decoder.sampling(MVN_prior).detach().cpu().numpy()

    # Step 5. Visualization
    figure_1 = None
    if p_dim == 2 : 
        figure = total_visualize_2D(t_sample, gAE_reconstruction, gAE_generation, VAE_reconstruction, VAE_generation)
    elif p_dim == 3: 
        figure = total_visualize_3D(t_sample, gAE_reconstruction, gAE_generation, VAE_reconstruction, VAE_generation)
    elif p_dim == 3: 
        figure = total_visualize_PCA(t_sample, gAE_reconstruction, gAE_generation, VAE_reconstruction, VAE_generation)


    return None