import argument
import gamma_ae
import torch
import os
from util import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

## init ##
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
SEED = args.seed
make_reproducibility(SEED)
args = argument.args
beta = args.beta
nu = args.nu
model_dir = None

if nu != 0 and beta == 0:
    print("Current model : gammaAE ")
    print(f'nu : {nu}')
    model_dir = './'+args.dataset+ f'_gammaAE_nu:{nu}/'
elif nu == 0 and beta != 0:
    print("Current model : RVAE")
    print(f'beta : {beta}')
    model_dir = './'+args.dataset+ f'_RVAE_beta:{beta}/'
elif nu == 0 and beta == 0:
    print("Current model : Vanilla VAE ")
    model_dir = './'+args.dataset+ f'_VAE/'
else:
    print("Please define valid parameters. (Either nu or beta must be 0)")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

## For tensorboard data ##
writer = SummaryWriter(model_dir + 'Tensorboard_results')

## INIT ##
image_size = 28
input_dim = 784 # 28**2

gammaAE = gamma_ae.gammaAE(input_dim, image_size, DEVICE)

for epoch in tqdm(range(0, args.epochs)):
    gammaAE.train(epoch,writer)
    gammaAE.test(epoch,writer)
    writer.close()

## t-sne ##
if args.tsne == 1:
    print("Draw a tsne plot..")
    make_tsne_plot(gammaAE,DEVICE)
    print("Done!")
