import os
import copy
import random
import argparse
import numpy as np

# import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from tqdm import tqdm
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mmd import mmd_unbiased_sq, make_masking, mmd_bootstrap_test, mmd_linear, mmd_linear_bootstrap_test
from simul_util import make_result_dir, make_reproducibility, t_sampling, sample_generation, t_density, t_density_contour, latent_generate, MYTensorDataset
from simul_loss import log_t_normalizing_const, gamma_regularizer
from simul_model import Encoder, Decoder, gammaAE
from simul_visualize import visualize_3D, visualize_density, visualize_latent

from simul_train_latent import latent_simulation

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device(f'cuda:0' if USE_CUDA else "cpu")

parser = argparse.ArgumentParser(description="gammaAE")

parser.add_argument('--dirname',        type=str,   default='Results', help='Name of experiments')

# parser.add_argument('--p_dim',          type=int,   default=1,      help='data dimension')
# parser.add_argument('--q_dim',          type=int,   default=1,      help='Latent dimension')
parser.add_argument('--model_nu_list',  nargs='+',  type=float,     default=[3.0],    help='Degree of freedom in model')
parser.add_argument('--recon_sigma',    type=float, default=0.25,   help='Sigma value in decoder')

parser.add_argument('--epochs',         type=int,   default=100,     help='Train epoch')
parser.add_argument('--num_layers',     type=int,   default=64,    help='Number of nodes in layers of neural networks')
parser.add_argument('--batch_size',     type=int,   default=1024,    help='Batch size')
parser.add_argument('--lr',             type=float, default=1e-3,   help='Learning rate')
parser.add_argument('--eps',            type=float, default=1e-8,   help="Epsilon for Adam optimizer")
parser.add_argument('--weight_decay',   type=float, default=1e-4,   help='Weight decay')

parser.add_argument('--train_data_seed',type=int,   default=10000,  help="Seed for sampling train data")
parser.add_argument('--validation_data_seed', type=int,   default=20000,  help="Seed for sampling test data")
parser.add_argument('--test_data_seed', type=int,   default=30000,  help="Seed for sampling test data")
parser.add_argument('--model_init_seed',type=int,   default=1000,   help="Seed for model parameter initialization")

parser.add_argument('--sample_nu',      type=float, default=5.0,     help='Degree of freedom from each cluster')
parser.add_argument('--train_N',        type=float, default=100000,  help="Number of sample size from each cluster")
parser.add_argument('--test_N',         type=float, default=100000,  help="Number of sample size from each cluster")

parser.add_argument('--boot_iter',      type=int,   default=999,    help="Number of iterations in bootstrap MMD test")
parser.add_argument('--gen_N',          type=int,   default=200000, help="Number of generations")
parser.add_argument('--xlim',           type=float, default=200.0,  help="Maximum value of x-axis in log-scale plot")
parser.add_argument('--patience',       type=int,   default=10,     help="Patience for Early stopping")

args = parser.parse_args()

device = DEVICE

dirname = f'{args.dirname}_data{args.sample_nu}_sigma{args.recon_sigma}'

latent_simulation(args.model_nu_list, args.recon_sigma, 
                  args.sample_nu, args.train_N, args.test_N, 
                  dirname, device, 
                  args.epochs, args.num_layers, args.batch_size, args.lr, args.eps, args.weight_decay, 
                  args.train_data_seed, args.validation_data_seed, args.test_data_seed, args.model_init_seed, 
                  xlim = args.xlim, mmd_type = 'linear', 
                  bootstrap_iter = args.boot_iter, gen_N = args.gen_N, patience = args.patience)