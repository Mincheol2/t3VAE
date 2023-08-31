import os
import copy
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss_1D import log_t_normalizing_const, gamma_regularizer
from util_1D import make_result_dir, make_reproducibility, TensorDataset
from sampling_1D import t_sampling, sample_generation, t_density, t_density_contour
from mmd import mmd_unbiased_sq, make_masking, mmd_unbiased_bootstrap_test, mmd_linear, mmd_linear_bootstrap_test
from visualize_1D import visualize_density
from model_1D import Encoder, Decoder, t3VAE

from train_1D import simulation_1D

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device(f'cuda:0' if USE_CUDA else "cpu")

parser = argparse.ArgumentParser(description="t3VAE")
parser.add_argument('--dirname',        type=str,   default='result', help='Name of experiments')

parser.add_argument('--n_dim',          type=int,   default=1,      help='data dimension')
parser.add_argument('--m_dim',          type=int,   default=1,      help='Latent dimension')
parser.add_argument('--model_nu_list',  nargs='+',  type=float,     default=[20.0, 16.0, 12.0, 8.0],    help='Degree of freedom for model')
parser.add_argument('--recon_sigma',    type=float, default=0.1,    help='sigma value in decoder')

parser.add_argument('--epochs',         type=int,   default=100,    help='Train epoch')
parser.add_argument('--num_layers',     type=int,   default=64,     help='Number of nodes in layers of neural networks')
parser.add_argument('--batch_size',     type=int,   default=1024,   help='Batch size')
parser.add_argument('--lr',             type=float, default=1e-3,   help='Learning rate')
parser.add_argument('--eps',            type=float, default=1e-8,   help="Epsilon for Adam optimizer")
parser.add_argument('--weight_decay',   type=float, default=1e-4,   help='Weight decay')

parser.add_argument('--train_data_seed',type=int,   default=1,      help="Seed for sampling train data")
parser.add_argument('--validation_data_seed', type=int, default=2,  help="Seed for sampling test data")
parser.add_argument('--test_data_seed', type=int,   default=3,      help="Seed for sampling test data")
parser.add_argument('--model_init_seed',type=int,   default=42,     help="Seed for model parameter initialization")
parser.add_argument('--param_seed',     type=int,   default=5000,   help="Seed for random initialization of parameters for train and test data")

parser.add_argument('--K',              type=int,   default=2,      help="Number of mixture distribution in data distribution")
parser.add_argument('--train_N',        type=int,   default=200000, help="Number of sample size of train data")
parser.add_argument('--val_N',          type=int,   default=200000, help="Number of sample size of train data")
parser.add_argument('--test_N',         type=int,   default=500000,help="Number of sample size of train data")
parser.add_argument('--sample_nu_list', nargs='+',  type=float,     default=[5.0, 5.0],     help='Degree of freedom from each cluster')
parser.add_argument('--ratio_list',     nargs='+',  type=float,     default=[0.6, 0.4],     help='Mixture density of each cluster')
parser.add_argument('--mu_list',        nargs='+',  type=float,     default=[-2.0, 2.0],    help="Mean parameter for each cluster")
parser.add_argument('--var_list',       nargs='+',  type=float,     default=[1.0, 1.0],     help="Dispersion parameter for each cluster")

parser.add_argument('--boot_iter',      type=int,   default=999,    help="Number of iterations in bootstrap MMD test")
parser.add_argument('--gen_N',          type=int,   default=500000,help="Number of generations")
parser.add_argument('--MMD_test_N',     type=int,   default=100000, help="Number of generations")
parser.add_argument('--xlim',           type=float, default=20.0,   help="Maximum value of x-axis in log-scale plot")
parser.add_argument('--patience',       type=int,   default=10,     help="Patience for Early stopping")

args = parser.parse_args()

mu_list = args.mu_list
var_list = args.var_list

if mu_list is not None : 
    mu_list = [mu * torch.ones(1) for mu in mu_list]

if var_list is not None : 
    var_list = [var * torch.ones(1,1) for var in var_list]

device = DEVICE

dirname = args.dirname

simulation_1D(args.n_dim, args.m_dim, args.model_nu_list, args.recon_sigma, 
              args.K, args.train_N, args.val_N, args.test_N, args.sample_nu_list, args.ratio_list,
              dirname, device, 
              args.epochs, args.num_layers, args.batch_size, args.lr, args.eps, args.weight_decay, 
              args.train_data_seed, args.validation_data_seed, args.test_data_seed, args.model_init_seed, 
              xlim = args.xlim, mmd_type = 'linear', 
              mu_list = mu_list, var_list = var_list, param_seed = args.param_seed, 
              bootstrap_iter = args.boot_iter, gen_N = args.gen_N, MMD_test_N = args.MMD_test_N, patience = args.patience)
