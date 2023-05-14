import os
import copy
import random
# import argparse
import numpy as np

import scipy.stats as stats
import matplotlib.pyplot as plt

from tqdm import tqdm

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
from simul_visualize import visualize_density, visualize_latent, visualize_latent_2D

def latent_simulation(model_nu_list, recon_sigma, 
                      sample_nu, train_N, test_N, 
                      dir_name, device, 
                      epochs, num_layers, batch_size, lr, eps, weight_decay, 
                      train_data_seed, validation_data_seed, test_data_seed, model_init_seed, 
                      xlim, mmd_type = 'linear', bootstrap_iter = 999, gen_N = 100000, patience = 10) : 

    # Step 0. Environment setup
    # Here we may assume that p_dim = 3, q_dim = 2
    p_dim = 3
    q_dim = 2
    M = len(model_nu_list)

    mmd_test = mmd_linear_bootstrap_test
    if mmd_type != 'linear' : 
        mmd_test = mmd_bootstrap_test

    dirname = f'./{dir_name}'
    make_result_dir(dirname)
    latent_writer = SummaryWriter(dirname + '/latent_space')
    # generation_writer = SummaryWriter(dirname + '/generations')
    gAE_writer_list = [SummaryWriter(dirname + f'/gAE_nu{model_nu}') for model_nu in model_nu_list]
    VAE_writer = SummaryWriter(dirname + '/VAE')


    # Step 1. Sampling data
    _, train_data = latent_generate(train_N, nu = sample_nu, SEED = train_data_seed, device = device)
    _, validation_data = latent_generate(test_N, nu = sample_nu, SEED = validation_data_seed, device = device)
    test_latent, test_data = latent_generate(test_N, nu = sample_nu, SEED = test_data_seed, device = device)

    train_dataset = MYTensorDataset(train_data)

    # Step 2. Model initialization
    make_reproducibility(model_init_seed)

    gAE_list = [
        gammaAE(train_dataset, p_dim, q_dim, model_nu, recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay) for model_nu in model_nu_list
    ]
    VAE = gammaAE(train_dataset, p_dim, q_dim, 0,  recon_sigma, device, num_layers, lr, batch_size, eps, weight_decay)

    # Step 3. Model training
    epoch_tqdm = tqdm(range(0, epochs))

    gAE_best_loss = [10^6 for _ in range(M)]
    VAE_best_loss = 10^6
    gAE_best_model = copy.deepcopy(gAE_list)
    VAE_best_model = copy.deepcopy(VAE)
    gAE_count = [0 for _ in range(M)]
    VAE_count = 0

    gAE_stop = [False for _ in range(M)]
    VAE_stop = False

    for epoch in epoch_tqdm : 
        
        # If all models had finished training, then stop the loop
        if all(gAE_stop) & VAE_stop : 
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
            if gAE_stop[m] is not True : 
                gAE_list[m].train(epoch, VAE_writer)
                gAE_val_loss = gAE_list[m].validation(validation_data, epoch, gAE_writer_list[m])
                if gAE_val_loss < gAE_best_loss[m] : 
                    gAE_best_loss[m] = gAE_val_loss
                    gAE_best_model[m] = copy.deepcopy(gAE_list[m])
                    gAE_count[m] = 0
                else : 
                    gAE_count[m] += 1

                if gAE_count[m] == patience :
                    gAE_stop[m] = True
                    print(f"gAE with nu {model_nu_list[m]} stopped training at {epoch}th epoch")

                gAE_best_model[m].test(test_data, epoch, gAE_writer_list[m])

        # Record generation, MMD/KS stat, and latent representation
        if epoch % 5 == 0 or (all(gAE_stop) & VAE_stop):
            # Generation and visualization
            gAE_gen_list = [gAE.generate(gen_N).detach() for gAE in gAE_best_model]
            VAE_gen = VAE_best_model.generate(gen_N).detach()
            
            # visualization = visualize_density(train_data, test_data, model_nu_list, gAE_gen_list, VAE_gen, K, sample_nu_list, mu_list, var_list, ratio_list, xlim)
            # generation_writer.add_figure("Generation", visualization, epoch)
            # filename = f'{dirname}/generations/epoch{epoch}.png'
            # visualization.savefig(filename)

            # MMD / KS test results (statistic and p-value)
            gAE_mmd_result = [mmd_test(gAE_gen[0:(2*test_N)], test_data, device = device, iteration = bootstrap_iter) for gAE_gen in gAE_gen_list]
            gAE_stat_list = [result[0] for result in gAE_mmd_result]
            gAE_p_value_list = [result[1] for result in gAE_mmd_result]
            VAE_stat, VAE_p_value, _ = mmd_test(VAE_gen[0:(2*test_N)], test_data, device = device, iteration = bootstrap_iter)

            for m in range(M) : 
                gAE_writer_list[m].add_scalar("Test/MMD score", gAE_stat_list[m], epoch)
                gAE_writer_list[m].add_scalar("Test/MMD p-value", gAE_p_value_list[m], epoch)

            VAE_writer.add_scalar("Test/MMD score", VAE_stat, epoch)
            VAE_writer.add_scalar("Test/MMD p-value", VAE_p_value, epoch)

            # Latent representation of test data
                
            gAE_latent_list = [gAE.encoder(test_data)[0].detach() for gAE in gAE_best_model]
            VAE_latent = VAE_best_model.encoder(test_data)[0].detach()

            gAE_recon_list = [gAE_best_model[m].decoder.sampling(gAE_latent_list[m]).detach() for m in range(M)]
            VAE_recon = VAE_best_model.decoder.sampling(VAE_latent).detach()

            # latent_representation = visualize_latent(
            #     sample_nu, test_latent, test_data, model_nu_list, gAE_latent_list, VAE_latent, gAE_recon_list, VAE_recon, xlim
            # )

            latent_representation = visualize_latent_2D(
                sample_nu, test_latent, test_data, model_nu_list, gAE_latent_list, VAE_latent, gAE_recon_list, VAE_recon, xlim
            )

            latent_writer.add_figure("Latent representation", latent_representation, epoch)
            filename = f'{dirname}/latent_space/epoch{epoch}.png'
            latent_representation.savefig(filename)

    return None