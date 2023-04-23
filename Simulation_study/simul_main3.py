import os
import random
import numpy as np
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


os.chdir('./gammaAE/Simulation_study')

from mmd import mmd_unbiased_sq, mmd_uniform_bound, make_masking, mmd_bootstrap_test
from simul_util import make_result_dir, make_reproducibility, t_sampling, sample_generation, MYTensorDataset
from simul_loss import log_t_normalizing_const, gamma_regularizer
from simul_model import Encoder, Decoder, gammaAE
from simul_visualize import visualize, visualize_PCA, visualize_3D, visualize_2D, visualize_density

from simul_train import simulation

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device(f'cuda:0' if USE_CUDA else "cpu")

device = DEVICE
p_dim = 1
q_dim = 1
batch_size = 128
epochs = 100
num_layers = 128

train_data_seed = 100
test_data_seed = 200
model_init_seed = 1000

K = 1
train_N_list = [5000]
test_N_list = [1000]

index = 2
b_list = None
var_list = None
eps = 1e-6
weight_decay = 1e-4
lr = 1e-3
param_seed = 5000
recon_sigma = 0.2
sample_nu = [3]
model_nu= 3

simulation(index, K, sample_nu, train_N_list, test_N_list, train_data_seed, test_data_seed, 
            p_dim, q_dim, model_nu, recon_sigma, model_init_seed, device, 
            epochs, num_layers, lr, batch_size, eps, weight_decay, 
            b_list = None, var_list = None, param_seed = param_seed, bootstrap_iter = 1999)