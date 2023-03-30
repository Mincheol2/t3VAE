import argument
import gamma_ae
from vampprior import vampprior_model
import torch
import os
from util import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

## init ##
args = argument.args
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device(f'cuda:{args.gpu_id}' if USE_CUDA else "cpu")

make_reproducibility(args.seed)


if args.nu != 0:
    print("Current framework : {gammaAE}")
    print(f'nu : {args.nu}')
    if args.dirname == "":
        args.dirname = './'+args.dataset+ f'_gammaAE_nu:{args.nu}_seed:{args.seed}/'
# elif args.nu == 0 and args.beta != 0:
#     print("Current framework : RVAE")
#     print(f'beta : {args.beta}')
#     model_dir = './'+args.dataset+ f'_RVAE_beta:{args.beta}_seed:{args.seed}/'
else:
    if args.model == "vampprior":
        print("Current framework : VAE + Vampprior")
    else:
        print("Current framework : Vanilla VAE ")
    if args.dirname == "":
        args.dirname = './'+args.dataset+ f'_VAE_seed:{args.seed}/'
        
if args.dirname != "":
    args.dirname = f'./{args.dirname}/'

if args.flat != 'y':
    print("We'll use KL divergence, despite gammaAE")


os.makedirs(args.dirname,exist_ok=True)
os.makedirs(args.dirname + 'interpolations',exist_ok=True)
os.makedirs(args.dirname + 'generations',exist_ok=True)
writer = SummaryWriter(args.dirname + 'Tensorboard_results')

## INIT ##
if args.model == "vampprior":
    model = vampprior_model.VAE_vampprior(DEVICE)
else:
    model = gamma_ae.gammaAE(DEVICE)

epoch_tqdm = tqdm(range(0, args.epochs))
for epoch in epoch_tqdm:
    # print(f'\nEpoch {epoch}')
    reg_loss, recon_loss, total_loss = model.train(epoch,writer)
    print(f'\nTrain) reg_loss={reg_loss:.4f} recon_loss={recon_loss:.4f} total_loss={total_loss:.4f}')
    
    reg_loss, recon_loss, total_loss = model.test(epoch,writer)
    print(f'Test) reg_loss={reg_loss:.4f} recon_loss={recon_loss:.4f} total_loss={total_loss:.4f}\n')
    

writer.close()

## t-sne ##
# if args.tsne == 1:
#     print("Draw a tsne plot..")
#     make_tsne_plot(gammaAE,model_dir, DEVICE)
#     print("Done!")
