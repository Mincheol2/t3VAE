
import os
import numpy as np
import torch
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# os.getcwd()
# os.chdir('./gammaAE/Simulation_study')

from simul_util import make_reproducibility
from simul_synthesize import t_density, t_density_contour
from mmd import make_masking, mmd_linear, mmd_linear_bootstrap_test

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device(f'cuda:0' if USE_CUDA else "cpu")
bootstrap_iter = 999

K = 2
sample_nu_list = [5.0, 5.0]
ratio_list = [0.6, 0.4]
mu_list = [-2.0 , 2.0]
var_list = [1.0, 1.0]
model_nu_list = [20.0, 16.0, 12.0, 8.0]

device = DEVICE
MMD_test_N = 100000
mmd_test = mmd_linear_bootstrap_test

if mu_list is not None : 
    mu_list = [mu * torch.ones(1) for mu in mu_list]

if var_list is not None : 
    var_list = [var * torch.ones(1,1) for var in var_list]

test_data = torch.tensor(np.asarray(pd.read_csv('./result/test_data.csv', header = None))).to(DEVICE)
VAE_gen = torch.tensor(np.asarray(pd.read_csv('./result/VAE_gen.csv', header = None))).to(DEVICE)
t3VAE_gen_list = [torch.tensor(np.asarray(pd.read_csv(f'./result/t3VAE_gen_{nu}.csv', header = None))).to(DEVICE) for nu in [20.0, 16.0, 12.0, 8.0]]

# SEED for bootstrap test
make_reproducibility(10)

# We do not report this p-value. Instead, we report the recorded MMD p-value in last training step. 
mmd_linear_bootstrap_test(VAE_gen[0:MMD_test_N], test_data[0:MMD_test_N], device = device, iteration = bootstrap_iter)[1]
for m in range(4) : 
    mmd_linear_bootstrap_test(t3VAE_gen_list[m][0:MMD_test_N], test_data[0:MMD_test_N], device = device, iteration = 999)[1]

tail_cut = 5
# right tail
large_VAE_sample = VAE_gen[(VAE_gen > tail_cut).flatten()]
large_t3VAE_sample_list = [t3VAE_gen[(t3VAE_gen > tail_cut).flatten()] for t3VAE_gen in t3VAE_gen_list]
large_test_data = test_data[(test_data > tail_cut).flatten()]
# print(f'VAE count : {large_VAE_sample.shape}'); print(f't3VAE count : {[a.shape[0] for a in large_t3VAE_sample_list]}'); print(f'Test data count :  {large_test_data.shape[0]}')
print(f'p-value for VAE (right tail) :{mmd_linear_bootstrap_test(large_VAE_sample, large_test_data, device = device, iteration = bootstrap_iter)[1]}')
for m in range(4) : 
    print(f'p-value for t3VAE with nu = {model_nu_list[m]} (right tail) :{mmd_linear_bootstrap_test(large_t3VAE_sample_list[m], large_test_data, device = device, iteration = 999)[1]}')

# left tail
small_VAE_sample = VAE_gen[(VAE_gen < - tail_cut).flatten()]
small_t3VAE_sample_list = [t3VAE_gen[(t3VAE_gen <- tail_cut).flatten()] for t3VAE_gen in t3VAE_gen_list]
small_test_data = test_data[(test_data <- tail_cut).flatten()]
# print(f'VAE count : {small_VAE_sample.shape}'); print(f't3VAE count : {[a.shape[0] for a in small_t3VAE_sample_list]}'); print(f'Test data count :  {small_test_data.shape[0]}')
print(f'p-value for VAE (left tail) :{mmd_linear_bootstrap_test(small_VAE_sample, small_test_data, device = device, iteration = bootstrap_iter)[1]}')
for m in range(4) : 
    print(f'p-value for t3VAE with nu = {model_nu_list[m]} (left tail) :{mmd_linear_bootstrap_test(small_t3VAE_sample_list[m], small_test_data, device = device, iteration = 999)[1]}')


# mmd_linear_bootstrap_test(small_VAE_sample, small_test_data, device = device, iteration = bootstrap_iter)[1]
# for m in range(4) : 
#     print(mmd_linear_bootstrap_test(small_t3VAE_sample_list[m], small_test_data, device = device, iteration = 999)[1])
