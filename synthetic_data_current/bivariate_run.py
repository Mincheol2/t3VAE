import os
import copy
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from tqdm import tqdm
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import *
from mmd import make_masking, mmd_linear, mmd_linear_bootstrap_test
from loss import log_t_normalizing_const, gamma_regularizer
from util import make_result_dir, make_reproducibility, TensorDataset
from bivariate.sampling import multivariate_sample_generation, multivariate_t_sampling, nonlinear_sampling
from bivariate.visualize import drawing
from bivariate.train import bivariate_simulation


parser = argparse.ArgumentParser(description="t3VAE")
parser.add_argument('--dirname',        type=str,   default='exp', help='Name of experiments')

parser.add_argument('--nu',             type=float, default=40.0,    help='degree of freedom')
parser.add_argument('--recon_sigma',    type=float, default=1.0,    help='sigma value in decoder')
parser.add_argument('--reg_weight',     type=float, default=1.0,    help='weight for regularizer term (beta)')

parser.add_argument('--epochs',         type=int,   default=100,    help='Train epoch')
parser.add_argument('--num_layers',     type=int,   default=64,     help='Number of nodes in layers of neural networks')
parser.add_argument('--batch_size',     type=int,   default=1024,   help='Batch size')
parser.add_argument('--lr',             type=float, default=1e-2,   help='Learning rate')
parser.add_argument('--eps',            type=float, default=1e-8,   help="Epsilon for Adam optimizer")
parser.add_argument('--weight_decay',   type=float, default=1e-4,   help='Weight decay')

parser.add_argument('--train_data_seed',type=int,   default=1,      help="Seed for sampling train data")
parser.add_argument('--validation_data_seed', type=int, default=2,  help="Seed for sampling test data")
parser.add_argument('--test_data_seed', type=int,   default=3,      help="Seed for sampling test data")
parser.add_argument('--model_init_seed',type=int,   default=42,     help="Seed for model parameter initialization")

# parser.add_argument('--K',              type=int,   default=2,      help="Number of mixture distribution in data distribution")
parser.add_argument('--train_N',        type=int,   default=200000, help="Number of sample size of train data")
parser.add_argument('--val_N',          type=int,   default=200000, help="Number of sample size of train data")
parser.add_argument('--test_N',         type=int,   default=500000, help="Number of sample size of train data")
# parser.add_argument('--sample_nu_list', nargs='+',  type=float,     default=[5.0, 5.0],     help='Degree of freedom from each cluster')
# parser.add_argument('--ratio_list',     nargs='+',  type=float,     default=[0.6, 0.4],     help='Mixture density of each cluster')
# parser.add_argument('--mu_list',        nargs='+',  type=float,     default=[-2.0, 2.0],    help="Mean parameter for each cluster")
# parser.add_argument('--var_list',       nargs='+',  type=float,     default=[1.0, 1.0],     help="Dispersion parameter for each cluster")

parser.add_argument('--boot_iter',      type=int,   default=999,    help="Number of iterations in bootstrap MMD test")
parser.add_argument('--gen_N',          type=int,   default=500000,help="Number of generations")
parser.add_argument('--MMD_test_N',     type=int,   default=100000, help="Number of generations")
parser.add_argument('--xmin',           type=float, default=-10.0,   help="Minimum value of x-axis")
parser.add_argument('--xmax',           type=float, default=15.0,   help="Maximum value of y-axis")
parser.add_argument('--ymin',           type=float, default=-5.0,   help="Minimum value of x-axis")
parser.add_argument('--ymax',           type=float, default=5.0,   help="Maximum value of y-axis")
parser.add_argument('--bins_x',         type=int,   default=30,     help="Number of bins of x-axis")
parser.add_argument('--bins_y',         type=int,   default=30,     help="Number of bins of y-axis")
parser.add_argument('--patience',       type=int,   default=10,     help="Patience for Early stopping")

args = parser.parse_args()

n_dim = 2
m_dim = 1

K = 2
ratio_list = [0.75, 0.25]
sample_nu_list = [5.0, 10.0]

sample_mu_list = [
    torch.tensor([0.0]), 
    torch.tensor([7.0])
]

sample_var_list = [
    torch.tensor([[2.0]]), 
    torch.tensor([[1.0]])
]

# mu_list = args.mu_list
# var_list = args.var_list
# if mu_list is not None : 
#     mu_list = [mu * torch.ones(1) for mu in mu_list]
# if var_list is not None : 
#     var_list = [var * torch.ones(1,1) for var in var_list]

device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")
dirname = f'2D_results/{args.dirname}'

make_reproducibility(args.model_init_seed)

model_list = [
    VAE.VAE(n_dim, m_dim, device = device).to(device),  
    t3VAE.t3VAE(n_dim, m_dim, nu = args.nu, device = device).to(device),  
    TVAE.TVAE(n_dim, m_dim, device = device).to(device)
]


bivariate_simulation(
    model_list, [model.model_name for model in model_list], 
    K, args.train_N, args.val_N, args.test_N, ratio_list,
    sample_nu_list, sample_mu_list, sample_var_list, 
    dirname, device, args.xmin, args.xmax, args.ymin, args.ymax, args.bins_x, args.bins_y, 
    args.epochs, args.batch_size, args.lr, args.eps, args.weight_decay, 
    args.train_data_seed, args.validation_data_seed, args.test_data_seed, 
    bootstrap_iter = args.boot_iter, gen_N = args.gen_N, MMD_test_N = args.MMD_test_N, patience = args.patience
)

