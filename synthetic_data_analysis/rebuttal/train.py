import os
import copy
import random
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

from loss import log_t_normalizing_const, gamma_regularizer
from util import make_result_dir, make_reproducibility, TensorDataset
from sampling import t_sampling, sample_generation, t_density, t_density_contour
from mmd import make_masking, mmd_linear, mmd_linear_bootstrap_test
from visualize import visualize_density

# from VAE import VAE
# from t3VAE import t3VAE
# from TVAE import TVAE
# from TVAE_modified import TVAE_modified

def simul(
    model_list, model_title_list, 
    K, train_N, val_N, test_N, ratio_list, 
    sample_nu_list, sample_mu_list, sample_var_list, 
    dir_name, device, xlim, 
    epochs, batch_size, lr, eps, weight_decay, 
    train_data_seed, validation_data_seed, test_data_seed, 
    bootstrap_iter = 1999, gen_N = 100000, MMD_test_N = 100000, patience = 10
) : 
    M = len(model_list)

    dirname = f'./{dir_name}'
    make_result_dir(dirname)

    generation_writer = SummaryWriter(dirname + '/generations')
    model_writer_list = [SummaryWriter(dirname + f'/{title}') for title in model_title_list]

    # Generate dataset
    train_data = sample_generation(
        device, SEED=train_data_seed,
        K=K, N=train_N, ratio_list = ratio_list, mu_list=sample_mu_list, var_list=sample_var_list, nu_list=sample_nu_list
    )

    validation_data = sample_generation(
        device, SEED=validation_data_seed,
        K=K, N=val_N, ratio_list = ratio_list, mu_list=sample_mu_list, var_list=sample_var_list, nu_list=sample_nu_list
    )

    test_data = sample_generation(
        device, SEED=test_data_seed,
        K=K, N=test_N, ratio_list = ratio_list, mu_list=sample_mu_list, var_list=sample_var_list, nu_list=sample_nu_list
    )

    train_dataset = TensorDataset(train_data)

    # Model training

    model_best_loss = [10^6 for _ in range(M)]
    model_best_model = copy.deepcopy(model_list)
    model_count = [0 for _ in range(M)]
    model_stop = [False for _ in range(M)]

    opt_list = [
        optim.Adam(model.parameters(), lr = lr, eps = eps, weight_decay=weight_decay) for model in model_list
    ]

    train_loader_list = [
        torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) for _ in range(M)
    ]

    for epoch in tqdm(range(0, epochs)) : 

        if all(model_stop) : 
            break

        # 1 epoch train
        for m in range(M) : 
            if model_stop[m] is not True : 
                model_list[m].train()

                denom_train = int(len(train_dataset)/batch_size) + 1

                for batch_idx, data in enumerate(train_loader_list[m]) : 
                    data = data[0].to(device)
                    opt_list[m].zero_grad()
                    recon_loss, reg_loss, train_loss = model_list[m](data)
                    train_loss.backward()

                    current_step_train = epoch * denom_train + batch_idx
                    model_writer_list[m].add_scalar("Train/Reconstruction Error", recon_loss.item(), current_step_train)
                    model_writer_list[m].add_scalar("Train/Regularizer", reg_loss.item(), current_step_train)
                    model_writer_list[m].add_scalar("Train/Total Loss" , train_loss.item(), current_step_train)

                    opt_list[m].step()

                # valiation step
                model_list[m].eval()
                data = validation_data.to(device)
                recon_loss, reg_loss, validation_loss = model_list[m](data)

                model_writer_list[m].add_scalar("Validation/Reconstruction Error", recon_loss.item(), epoch)
                model_writer_list[m].add_scalar("Validation/Regularizer", reg_loss.item(), epoch)
                model_writer_list[m].add_scalar("Validation/Total Loss" , validation_loss.item(), epoch)

                if validation_loss < model_best_loss[m] : 
                    model_best_loss[m] = validation_loss
                    model_best_model[m] = copy.deepcopy(model_list[m])
                    model_count[m] = 0
                else : 
                    model_count[m] += 1
                
                if model_count[m] == patience : 
                    model_stop[m] = True
                    print(f"{model_title_list[m]} stopped training at {epoch-patience}th epoch")

                # test step
                model_best_model[m].eval()
                data = test_data.to(device)
                recon_loss, reg_loss, test_loss = model_best_model[m](data)

                model_writer_list[m].add_scalar("Test/Reconstruction Error", recon_loss.item(), epoch)
                model_writer_list[m].add_scalar("Test/Regularizer", reg_loss.item(), epoch)
                model_writer_list[m].add_scalar("Test/Total Loss" , test_loss.item(), epoch)

        if epoch % 5 == 0 or all(model_stop): 
            # Generation
            model_gen_list = [model.generate(gen_N).detach() for model in model_best_model]

            visualization = visualize_density(
                model_title_list, model_gen_list, 
                K, sample_nu_list, sample_mu_list, sample_var_list, ratio_list, xlim
            )

            generation_writer.add_figure("Generation", visualization, epoch)
            filename = f'{dirname}/generations/epoch{epoch}.png'
            visualization.savefig(filename)

            mmd_result = [mmd_linear_bootstrap_test(gen[0:MMD_test_N], test_data[0:MMD_test_N], device = device, iteration = bootstrap_iter) for gen in model_gen_list]
            mmd_stat_list = [result[0] for result in mmd_result]
            mmd_p_value_list = [result[1] for result in mmd_result]

            for m in range(M) : 
                model_writer_list[m].add_scalar("Test/MMD score", mmd_stat_list[m], epoch)
                model_writer_list[m].add_scalar("Test/MMD p-value", mmd_p_value_list[m], epoch)

    np.savetxt(f'{dirname}/test_data.csv', test_data.cpu().numpy(), delimiter=',')
    [np.savetxt(f'{dirname}/{model_title_list[m]}.csv', model_gen_list[m].cpu().numpy(), delimiter = ',') for m in range(M)]

    return None