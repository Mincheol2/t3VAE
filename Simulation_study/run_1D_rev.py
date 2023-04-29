import os
import random
import argparse
import numpy as np
# import seaborn as sns
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

from mmd import mmd_unbiased_sq, make_masking, mmd_bootstrap_test, mmd_linear, mmd_linear_bootstrap_test
from simul_util import make_result_dir, make_reproducibility, t_sampling, sample_generation, t_density, density_contour, MYTensorDataset
from simul_loss import log_t_normalizing_const, gamma_regularizer
from simul_model import Encoder, Decoder, gammaAE
from simul_visualize import visualize_density

from simul_train import simulation_1D

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device(f'cuda:0' if USE_CUDA else "cpu")

parser = argparse.ArgumentParser(description="gammaAE")
# parser.add_argument('--dirname', type=str, default='simul')
parser.add_argument('--dirname',        type=str,   default='Results', help='Name of experiments')

parser.add_argument('--p_dim',          type=int,   default=1,      help='data dimension')
parser.add_argument('--q_dim',          type=int,   default=1,      help='Latent dimension')
parser.add_argument('--model_nu_list',  nargs='+',  type=float,     default=[3.0],    help='Degree of freedom in model')
parser.add_argument('--recon_sigma',    type=float, default=0.25,   help='Sigma value in decoder')

parser.add_argument('--epochs',         type=int,   default=50,     help='Train epoch')
parser.add_argument('--num_layers',     type=int,   default=128,    help='Number of nodes in layers of neural networks')
parser.add_argument('--batch_size',     type=int,   default=512,    help='Batch size')
parser.add_argument('--lr',             type=float, default=1e-3,   help='Learning rate')
parser.add_argument('--eps',            type=float, default=1e-8,   help="Epsilon for Adam optimizer")
parser.add_argument('--weight_decay',   type=float, default=1e-4,   help='Weight decay')

parser.add_argument('--train_data_seed',type=int,   default=10000,  help="Seed for sampling train data")
parser.add_argument('--test_data_seed', type=int,   default=20000,  help="Seed for sampling test data")
parser.add_argument('--model_init_seed',type=int,   default=1000,   help="Seed for model parameter initialization")
parser.add_argument('--param_seed',     type=int,   default=5000,   help="Seed for random initialization of parameters for train and test data")

parser.add_argument('--K',              type=int,   default=2,      help="Number of mixture distribution in data distribution")
parser.add_argument('--sample_nu_list', nargs='+',  type=float,     default=[2.0, 2.0],     help='Degree of freedom from each cluster')
parser.add_argument('--train_N_list',   nargs='+',  type=int,       default=[60000, 40000],   help="Number of sample size from each cluster")
parser.add_argument('--test_N_list',    nargs='+',  type=int,       default=[60000, 40000],     help="Number of sample size from each cluster")
parser.add_argument('--mu_list',        nargs='+',  type=float,     default=[-2.0, 2.0],    help="Mean parameter for each cluster")
parser.add_argument('--var_list',       nargs='+',  type=float,     default=[1.0, 1.0],     help="Dispersion parameter for each cluster")

parser.add_argument('--boot_iter',      type=int,   default=9999,   help="Number of iterations in bootstrap MMD test")
parser.add_argument('--gen_N',          type=int,   default=1000000,help="Number of generations")
parser.add_argument('--xlim',           type=float, default=200.0,  help="Maximum value of x-axis in log-scale plot")
# parser.add_argument('--b_list') : How to?

args = parser.parse_args()

mu_list = args.mu_list
var_list = args.var_list

if mu_list is not None : 
    mu_list = [mu * torch.ones(1) for mu in mu_list]

if var_list is not None : 
    var_list = [var * torch.ones(1,1) for var in var_list]

device = DEVICE

dirname = f'{args.dirname}_data{args.sample_nu_list}_sigma{args.recon_sigma}'

simulation_1D(args.p_dim, args.q_dim, args.model_nu_list, args.recon_sigma, 
              args.K, args.sample_nu_list, args.train_N_list, args.test_N_list, 
              dirname, device, 
              args.epochs, args.num_layers, args.batch_size, args.lr, args.eps, args.weight_decay, 
              args.train_data_seed, args.test_data_seed, args.model_init_seed, 
              mu_list = mu_list, var_list = var_list, param_seed = args.param_seed, 
              xlim = args.xlim, 
              bootstrap_iter = args.boot_iter, gen_N = args.gen_N)
