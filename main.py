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
DEVICE = torch.device(f'cuda:{args.gpu_id}' if USE_CUDA else "cpu")
make_reproducibility(args.seed)
args = argument.args

model_dir = None

if args.nu != 0 and args.beta == 0:
    print("Current framework : gammaAE ")
    print(f'nu : {args.nu}')
    model_dir = './'+args.dataset+ f'_gammaAE_nu:{args.nu}_seed:{args.seed}/'
elif args.nu == 0 and args.beta != 0:
    print("Current framework : RVAE")
    print(f'beta : {args.beta}')
    model_dir = './'+args.dataset+ f'_RVAE_beta:{args.beta}_seed:{args.seed}/'
elif args.nu == 0 and args.beta == 0:
    print("Current framework : Vanilla VAE ")
    model_dir = './'+args.dataset+ f'_VAE_seed:{args.seed}/'
else:
    print("Please define valid parameters. (Either nu or beta must be 0)")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

## For tensorboard data ##
writer = SummaryWriter(model_dir + 'Tensorboard_results')

## INIT ##

gammaAE = gamma_ae.gammaAE(DEVICE)

epoch_tqdm = tqdm(range(0, args.epochs))
for epoch in epoch_tqdm:
    # print(f'\nEpoch {epoch}')
    reg_loss, recon_loss, total_loss = gammaAE.train(epoch,writer)
    print(f'\nTrain) reg_loss={reg_loss:.4f} recon_loss={recon_loss:.4f} total_loss={total_loss:.4f}')
    
    reg_loss, recon_loss, total_loss = gammaAE.test(epoch,writer)
    print(f'Test) reg_loss={reg_loss:.4f} recon_loss={recon_loss:.4f} total_loss={total_loss:.4f}\n')
    


writer.close()
## t-sne ##
if args.tsne == 1:
    print("Draw a tsne plot..")
    make_tsne_plot(gammaAE,model_dir, DEVICE)
    print("Done!")
