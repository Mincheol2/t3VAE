import argument
import gamma_ae
import torch
import torch.nn as nn
import dataloader

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
from util import *
import numpy as np
from encoder import Encoder
from decoder import Decoder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
args = argument.args

## Make reproducibility ##
SEED = args.seed
make_reproducibility(SEED)

beta = args.beta
df = args.df

print(f'Current beta : {beta}')
print(f'Current df : {df}')

if args.load:
    model_dir = args.model_dir
else:
    model_dir = './'+args.dataset+ f'_model_save_beta{beta}_df{df}/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


## For tensorboard data ##
writer = SummaryWriter(model_dir + 'Tensorboard_results')

## INIT ##
image_size = 28
input_dim = 784 # 28**2 : MNIST (I'll generalize this param for any dataset)


gammaAE = gamma_ae.gammaAE(input_dim, image_size, DEVICE)
# ============== run ==============
# if args.load:
#     state = torch.load(model_dir+'Vanilavae.tch')
#     encoder.load_state_dict(state["encoder"])
#     decoder.load_state_dict(state["decoder"])
#     state2 = torch.load(model_dir+'Vanilavae.tchopt')
#     ep = state2["ep"]+1
#     opt.load_state_dict(state2["opt"])
#     test(testloader, encoder, decoder, ep, beta)
#     writer.close()

# else:
for epoch in tqdm(range(0, args.epochs)):
    gammaAE.train(epoch)
    gammaAE.test(epoch)
    
    writer.close()


## t-sne ##
if args.tsne == 1:
    print("Draw a tsne plot..")
    make_tsne_plot(gammaAE,DEVICE)
    print("Done!")
