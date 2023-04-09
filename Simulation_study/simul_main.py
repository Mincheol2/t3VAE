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

from simul_train import simulation

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device(f'cuda:0' if USE_CUDA else "cpu")


device = DEVICE
p_dim = 3
q_dim = 2
batch_size = 128
epochs = 100
num_layers = 128

train_data_seed = 100
test_data_seed = 200
model_init_seed = 1000
b_list = None
var_list = None
eps = 1e-6
weight_decay = 1e-4
lr = 1e-3

K = 1
train_N_list = [5000]
test_N_list = [1000]

# param_seed_list = [300, 500, 700, 1100, 1300]
# recon_sigma_list = [0.1, 1, 3]
# sample_nu_list = [[1.5], [1.8], [2], [3], [5]]
# model_nu_list = [2.1, 3, 5, 10, 20]

# # recon_Sigma
# for i in range(1) : 
#     # param_seed_list
#     for j in range(1) : 
#         # sample_nu_list
#         for k in range(5) : 
#             # model_nu_list
#             for l in range(5) : 
#                 index = 1111 + 1000*j +100*j + 10*k + l
#                 simulation(
#                     index, K, sample_nu_list[k], train_N_list, test_N_list, train_data_seed, test_data_seed, 
#                     p_dim, q_dim, model_nu_list[l], recon_sigma_list[i], model_init_seed, device, 
#                     epochs, num_layers, lr, batch_size, eps, weight_decay, 
#                     b_list = None, var_list = None, param_seed = param_seed_list[j]
#                 )


param_seed_list = [500, 700]
recon_sigma_list = [0.1, 1, 3]
sample_nu_list = [[1.8], [2.1], [3], [5], [10]]
model_nu_list = [2.1, 3, 5, 10]


# recon_Sigma
for i in range(2) : 
    # param_seed_list
    for j in range(3) : 
        # sample_nu_list
        for k in range(5) : 
            # model_nu_list
            for l in range(4) : 
                index = 11111 + 1000*j +100*j + 10*k + l
                simulation(
                    index, K, sample_nu_list[k], train_N_list, test_N_list, train_data_seed, test_data_seed, 
                    p_dim, q_dim, model_nu_list[l], recon_sigma_list[i], model_init_seed, device, 
                    epochs, num_layers, lr, batch_size, eps, weight_decay, 
                    b_list = None, var_list = None, param_seed = param_seed_list[j]
                )