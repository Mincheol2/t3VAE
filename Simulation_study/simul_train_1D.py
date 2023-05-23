import os
import copy
import random
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mmd import mmd_unbiased_sq, make_masking, mmd_unbiased_bootstrap_test, mmd_linear, mmd_linear_bootstrap_test
from simul_util import make_result_dir, make_reproducibility, TensorDataset
from simul_synthesize import t_sampling, sample_generation, t_density, t_density_contour
from simul_loss import log_t_normalizing_const, gamma_regularizer
from simul_model import Encoder, Decoder, t3VAE
from simul_visualize import visualize_density

def simulation_1D(n_dim, m_dim, model_nu_list, recon_sigma, 
                  K,  train_N, val_N, test_N, sample_nu_list, ratio_list,
                  dir_name, device, 
                  epochs, num_layers, batch_size, lr, eps, weight_decay, 
                  train_data_seed, validation_data_seed, test_data_seed, model_init_seed, 
                  xlim, mmd_type = 'linear', 
                  mu_list = None, var_list = None, param_seed = None, bootstrap_iter = 1999, 
                  gen_N = 100000, MMD_test_N = 100000, patience = 10) : 

    # Step 0. Environment setup
    M = len(model_nu_list)

    mmd_test = mmd_linear_bootstrap_test
    if mmd_type != 'linear' : 
        mmd_test = mmd_unbiased_bootstrap_test

    dirname = f'./{dir_name}'
    make_result_dir(dirname)
    generation_writer = SummaryWriter(dirname + '/generations')
    t3VAE_writer_list = [SummaryWriter(dirname + f'/t3VAE_nu{model_nu}') for model_nu in model_nu_list]
    VAE_writer = SummaryWriter(dirname + '/VAE')

    # Step 1. Sampling data
    # This code is not used
    # if mu_list is None and var_list is None and param_seed is not None : 
    #     make_reproducibility(param_seed)
    #     mu_list = [torch.randn(n_dim) * 3 for _ in range(K)]
    #     var_list = [torch.eye(n_dim) for _ in range(K)]

    train_data = sample_generation(
        device, SEED=train_data_seed,
        K=K, N=train_N, ratio_list = ratio_list, mu_list=mu_list, var_list=var_list, nu_list=sample_nu_list
    )

    validation_data = sample_generation(
        device, SEED=validation_data_seed,
        K=K, N=val_N, ratio_list = ratio_list, mu_list=mu_list, var_list=var_list, nu_list=sample_nu_list
    )

    test_data = sample_generation(
        device, SEED=test_data_seed,
        K=K, N=test_N, ratio_list = ratio_list, mu_list=mu_list, var_list=var_list, nu_list=sample_nu_list
    )

    train_dataset = TensorDataset(train_data)

    # Step 2. Model initialization
    make_reproducibility(model_init_seed)

    t3VAE_list = [
        t3VAE(train_dataset, n_dim, m_dim, model_nu, recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay) for model_nu in model_nu_list
    ]
    VAE = t3VAE(train_dataset, n_dim, m_dim, 0,  recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay)

    # Step 3. Model training

    t3VAE_best_loss = [10^6 for _ in range(M)]
    t3VAE_best_model = copy.deepcopy(t3VAE_list)
    t3VAE_count = [0 for _ in range(M)]
    t3VAE_stop = [False for _ in range(M)]

    VAE_best_loss = 10^6
    VAE_best_model = copy.deepcopy(VAE)
    VAE_count = 0
    VAE_stop = False

    for epoch in tqdm(range(0, epochs)) : 
        
        # If all models had finished training, then stop the loop
        if all(t3VAE_stop) & VAE_stop : 
            break 

        if VAE_stop is not True : 
            VAE.train(epoch, VAE_writer)

            VAE_val_loss = VAE.validation(validation_data, epoch, VAE_writer)
            if VAE_val_loss < VAE_best_loss : 
                VAE_best_loss = VAE_val_loss
                VAE_best_model = copy.deepcopy(VAE)
                VAE_count = 0
            else : 
                VAE_count += 1

            if VAE_count == patience : 
                VAE_stop = True
                print(f"VAE stopped training at {epoch}th epoch")

            VAE_best_model.test(test_data, epoch, VAE_writer)

        for m in range(M) : 
            if t3VAE_stop[m] is not True : 
                t3VAE_list[m].train(epoch, VAE_writer)
                t3VAE_val_loss = t3VAE_list[m].validation(validation_data, epoch, t3VAE_writer_list[m])
                if t3VAE_val_loss < t3VAE_best_loss[m] : 
                    t3VAE_best_loss[m] = t3VAE_val_loss
                    t3VAE_best_model[m] = copy.deepcopy(t3VAE_list[m])
                    t3VAE_count[m] = 0
                else : 
                    t3VAE_count[m] += 1

                if t3VAE_count[m] == patience :
                    t3VAE_stop[m] = True
                    print(f"t3VAE with nu {model_nu_list[m]} stopped training at {epoch}th epoch")

                t3VAE_best_model[m].test(test_data, epoch, t3VAE_writer_list[m])

        # Record generation, MMD/KS stat, and loss
        if epoch % 5 == 0 or (all(t3VAE_stop) & VAE_stop):
            # Generation
            t3VAE_gen_list = [t3VAE.generate(gen_N).detach() for t3VAE in t3VAE_best_model]
            VAE_gen = VAE_best_model.generate(gen_N).detach()
            
            visualization = visualize_density(
                model_nu_list, t3VAE_gen_list, VAE_gen, 
                K, sample_nu_list, mu_list, var_list, ratio_list, xlim
                )

            generation_writer.add_figure("Generation", visualization, epoch)
            filename = f'{dirname}/generations/epoch{epoch}.png'
            visualization.savefig(filename)

            # MMD test results (statistic and p-value)
            t3VAE_mmd_result = [mmd_test(t3VAE_gen[0:MMD_test_N], test_data[0:MMD_test_N], device = device, iteration = bootstrap_iter) for t3VAE_gen in t3VAE_gen_list]
            t3VAE_stat_list = [result[0] for result in t3VAE_mmd_result]
            t3VAE_p_value_list = [result[1] for result in t3VAE_mmd_result]
            VAE_stat, VAE_p_value, _ = mmd_test(VAE_gen[0:MMD_test_N], test_data[0:MMD_test_N], device = device, iteration = bootstrap_iter)

            for m in range(M) : 
                t3VAE_writer_list[m].add_scalar("Test/MMD score", t3VAE_stat_list[m], epoch)
                t3VAE_writer_list[m].add_scalar("Test/MMD p-value", t3VAE_p_value_list[m], epoch)

            VAE_writer.add_scalar("Test/MMD score", VAE_stat, epoch)
            VAE_writer.add_scalar("Test/MMD p-value", VAE_p_value, epoch)

    np.savetxt(f'{dirname}/test_data.csv', test_data.cpu().numpy(), delimiter=',')
    np.savetxt(f'{dirname}/VAE_gen.csv', VAE_gen.cpu().numpy(), delimiter=',')
    for m in range(M) : 
        np.savetxt(f'{dirname}/t3VAE_gen_{model_nu_list[m]}.csv', t3VAE_gen_list[m].cpu().numpy(), delimiter = ',')

    return None