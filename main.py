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

# for args.nu in [0,2.5,3,4,5]:
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

if args.flat != 'y':
    print("We use KL divergence, despite gammaAE")
if args.dirname != '':
    model_dir= args.dirname + "/"

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
    
# generation 

## interpolation images
os.makedirs(model_dir + 'interpolations')
sample_z, _, _ = gammaAE.encoder(gammaAE.sample_imgs)
test_imgs = gammaAE.decoder(sample_z).detach().cpu()

loop_iter = 3
num_steps = 8
for k in range(loop_iter):
    inter_z = []
    idx1, idx2, idx3, idx4 = np.random.choice(args.zdim, 4, replace=False)
    for j in range(num_steps):
        for i in range(num_steps):
            t = i / (num_steps -1)
            s = j / (num_steps -1)
            result1 = t * sample_z[idx1] + (1-t) * sample_z[idx2]
            result2 = t * sample_z[idx3] + (1-t) * sample_z[idx4]
            result = s * result1 + (1-s) * result2
            inter_z.append(result.tolist())
    inter_img = 0.5 * gammaAE.decoder(torch.tensor(inter_z).to(DEVICE)) + 0.5
    inter_grid = torchvision.utils.make_grid(inter_img.cpu())
    filename = f'{model_dir}/interpolations/interpolation_{k}.png'
    torchvision.utils.save_image(inter_grid, filename)

## generation images

# prior_z = 
# gamma_AE.decoder(sample_z).detach().cpu()
writer.close()
## t-sne ##
# if args.tsne == 1:
#     print("Draw a tsne plot..")
#     make_tsne_plot(gammaAE,model_dir, DEVICE)
#     print("Done!")
