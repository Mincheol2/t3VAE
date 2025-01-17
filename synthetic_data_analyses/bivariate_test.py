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
parser.add_argument('--tail_cut',       type=float, default=5.0,    help="Tail criterion")

args = parser.parse_args()


file_list = np.asarray(os.listdir(f'./2D_results/{args.dirname}'))
csv_list = file_list[np.where(['.csv' in name for name in  file_list])[0]]
csv_list = np.asarray([name[0:-4] for name in csv_list])
csv_list = csv_list[np.where(csv_list != 'test_data')[0]]
M = len(csv_list)

test_data = torch.tensor(np.asarray(pd.read_csv(f'./2D_results/{args.dirname}/test_data.csv', header = None))).to(device)
gen_list = [torch.tensor(np.asarray(pd.read_csv(f'./2D_results/{args.dirname}/{csv_name}.csv', header = None))).to(device) for csv_name in csv_list]

# SEED for bootstrap test
make_reproducibility(args.boot_seed)

full_results = [
    mmd_linear_bootstrap_test(gen_list[m][0:args.MMD_test_N], test_data[0:args.MMD_test_N], device = device, iteration = args.boot_iter)
    for m in range(M)
]

# tail
test_data_norm = torch.norm(test_data, dim = 1)
gen_data_norm = [torch.norm(gen, dim = 1) for gen in gen_list]
tail_test_data = test_data[test_data_norm > args.tail_cut]
tail_sample = [gen_list[i][gen_data_norm[i] > args.tail_cut] for i in range(len(gen_list))] 

right_test_data = tail_test_data[tail_test_data[:,0] > 0]
left_test_data = tail_test_data[tail_test_data[:,0] < 0]

right_sample = [gen[gen[:,0] > 0] for gen in tail_sample] 
left_sample = [gen[gen[:,0] < 0] for gen in tail_sample] 
right_results = [
    mmd_linear_bootstrap_test(right_sample[m], right_test_data, device = device, iteration = args.boot_iter)
    for m in range(M)
]
left_results = [
    mmd_linear_bootstrap_test(left_sample[m], left_test_data, device = device, iteration = args.boot_iter)
    for m in range(M)
]

for m in range(M) : 
    print(f'p-value for {csv_list[m]} : {full_results[m][1]}')

for m in range(M) : 
    print(f'p-value for {csv_list[m]} (right) : {right_results[m][1]}')

for m in range(M) : 
    print(f'p-value for {csv_list[m]} (left) : {left_results[m][1]}')