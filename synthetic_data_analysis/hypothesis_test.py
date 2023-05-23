
import os
import torch
import argparse
import numpy as np
import pandas as pd

from util_1D import make_reproducibility
from mmd import make_masking, mmd_linear, mmd_linear_bootstrap_test

USE_CUDA = torch.cuda.is_available()
device = torch.device(f'cuda:0' if USE_CUDA else "cpu")

parser = argparse.ArgumentParser(description="t3VAE")
parser.add_argument('--dirname',        type=str,   default='result', help='Name of experiments')
parser.add_argument('--model_nu_list',  nargs='+',  type=float,     default=[20.0, 16.0, 12.0, 8.0],    help='Degree of freedom for model')

parser.add_argument('--K',              type=int,   default=2,      help="Number of mixture distribution in data distribution")
parser.add_argument('--sample_nu_list', nargs='+',  type=float,     default=[5.0, 5.0],     help='Degree of freedom from each cluster')
parser.add_argument('--ratio_list',     nargs='+',  type=float,     default=[0.6, 0.4],     help='Mixture density of each cluster')
parser.add_argument('--mu_list',        nargs='+',  type=float,     default=[-2.0, 2.0],    help="Mean parameter for each cluster")
parser.add_argument('--var_list',       nargs='+',  type=float,     default=[1.0, 1.0],     help="Dispersion parameter for each cluster")

parser.add_argument('--boot_seed',      type=int,   default=10,     help="Random seed for bootstrap MMD test")
parser.add_argument('--boot_iter',      type=int,   default=999,    help="Number of iterations in bootstrap MMD test")
parser.add_argument('--MMD_test_N',     type=int,   default=100000, help="Number of generations")

args = parser.parse_args()

mu_list = args.mu_list
var_list = args.var_list

if mu_list is not None : 
    mu_list = [mu * torch.ones(1) for mu in mu_list]

if var_list is not None : 
    var_list = [var * torch.ones(1,1) for var in var_list]


test_data = torch.tensor(np.asarray(pd.read_csv(f'./{args.dirname}/test_data.csv', header = None))).to(device)
VAE_gen = torch.tensor(np.asarray(pd.read_csv(f'./{args.dirname}/VAE_gen.csv', header = None))).to(device)
t3VAE_gen_list = [torch.tensor(np.asarray(pd.read_csv(f'./{args.dirname}/t3VAE_gen_{nu}.csv', header = None))).to(device) for nu in [20.0, 16.0, 12.0, 8.0]]

# SEED for bootstrap test
make_reproducibility(args.boot_seed)

# We do not report this p-value. Instead, we report the recorded MMD p-value in last training step. 
mmd_linear_bootstrap_test(VAE_gen[0:args.MMD_test_N], test_data[0:args.MMD_test_N], device = device, iteration = args.boot_iter)[1]
for m in range(4) : 
    mmd_linear_bootstrap_test(t3VAE_gen_list[m][0:args.MMD_test_N], test_data[0:args.MMD_test_N], device = device, iteration = args.boot_iter)[1]

tail_cut = 5
# right tail
large_VAE_sample = VAE_gen[(VAE_gen > tail_cut).flatten()]
large_t3VAE_sample_list = [t3VAE_gen[(t3VAE_gen > tail_cut).flatten()] for t3VAE_gen in t3VAE_gen_list]
large_test_data = test_data[(test_data > tail_cut).flatten()]
print(f'p-value for VAE (right tail) :{mmd_linear_bootstrap_test(large_VAE_sample, large_test_data, device = device, iteration = args.boot_iter)[1]}')
for m in range(4) : 
    print(f'p-value for t3VAE with nu = {args.model_nu_list[m]} (right tail) :{mmd_linear_bootstrap_test(large_t3VAE_sample_list[m], large_test_data, device = device, iteration = args.boot_iter)[1]}')

# left tail
small_VAE_sample = VAE_gen[(VAE_gen < -tail_cut).flatten()]
small_t3VAE_sample_list = [t3VAE_gen[(t3VAE_gen <- tail_cut).flatten()] for t3VAE_gen in t3VAE_gen_list]
small_test_data = test_data[(test_data < -tail_cut).flatten()]
print(f'p-value for VAE (left tail) :{mmd_linear_bootstrap_test(small_VAE_sample, small_test_data, device = device, iteration = args.boot_iter)[1]}')
for m in range(4) : 
    print(f'p-value for t3VAE with nu = {args.model_nu_list[m]} (left tail) :{mmd_linear_bootstrap_test(small_t3VAE_sample_list[m], small_test_data, device = device, iteration = args.boot_iter)[1]}')
