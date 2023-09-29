import os
import torch
import argparse
import numpy as np
import pandas as pd

from util import make_reproducibility
from mmd import make_masking, mmd_linear, mmd_linear_bootstrap_test

USE_CUDA = torch.cuda.is_available()
device = torch.device(f'cuda:0' if USE_CUDA else "cpu")

parser = argparse.ArgumentParser(description="t3VAE")
parser.add_argument('--dirname',        type=str,   default='results', help='Name of experiments')
parser.add_argument('--boot_seed',      type=int,   default=10,     help="Random seed for bootstrap MMD test")
parser.add_argument('--boot_iter',      type=int,   default=999,    help="Number of iterations in bootstrap MMD test")
parser.add_argument('--MMD_test_N',     type=int,   default=100000, help="Number of generations")
parser.add_argument('--tail_cut',       type=float, default=6.0,    help="Tail criterion")

args = parser.parse_args()

file_list = np.asarray(os.listdir(f'./1D_results/{args.dirname}'))
csv_list = file_list[np.where(['.csv' in name for name in  file_list])[0]]
csv_list = np.asarray([name[0:-4] for name in csv_list])
csv_list = csv_list[np.where(csv_list != 'test_data')[0]]
M = len(csv_list)

test_data = torch.tensor(np.asarray(pd.read_csv(f'./1D_results/{args.dirname}/test_data.csv', header = None))).to(device)
gen_list = [torch.tensor(np.asarray(pd.read_csv(f'./1D_results/{args.dirname}/{csv_name}.csv', header = None))).to(device) for csv_name in csv_list]

# SEED for bootstrap test
make_reproducibility(args.boot_seed)

# We do not report this p-value. Instead, we report the recorded MMD p-value in last training step. 
full_results = [
    mmd_linear_bootstrap_test(gen_list[m][0:args.MMD_test_N], test_data[0:args.MMD_test_N], device = device, iteration = args.boot_iter)
    for m in range(M)
]

for m in range(M) : 
    print(f'p-value for {csv_list[m]} : {full_results[m][1]}')

# right tail
right_test_data = test_data[(test_data > args.tail_cut).flatten()]
right_sample = [gen[(gen > args.tail_cut).flatten()] for gen in gen_list] 
right_results = [
    mmd_linear_bootstrap_test(right_sample[m], right_test_data, device = device, iteration = args.boot_iter)
    for m in range(M)
]

# print(f'The number of samples (right tail) : {len(right_test_data)}')
# for m in range(M) : 
#     print(f'The number of samples from {csv_list[m]} (right tail) : {len(right_sample[m])}')
for m in range(M) : 
    print(f'p-value for {csv_list[m]} (right tail) : {right_results[m][1]}')

# left tail
left_test_data = test_data[(test_data < -args.tail_cut).flatten()]
left_sample = [gen[(gen < -args.tail_cut).flatten()] for gen in gen_list]
left_results = [
    mmd_linear_bootstrap_test(left_sample[m], left_test_data, device = device, iteration = args.boot_iter)
    for m in range(M)
]

# print(f'The number of samples (left tail) : {len(left_test_data)}')
# for m in range(M) : 
#     print(f'The number of samples from {csv_list[m]} (left tail) : {len(left_sample[m])}')
for m in range(M) : 
    print(f'p-value for {csv_list[m]} (left tail) : {left_results[m][1]}')
