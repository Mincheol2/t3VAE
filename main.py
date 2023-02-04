import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import random
import os
from util import *
from mnistc_dataset import *
import numpy as np
from encoder import Encoder
from decoder import Decoder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
import numpy as np

parser = argparse.ArgumentParser(description='vanila VAE')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='alpha div parameter (default: 1.0)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='div weight parameter (default: 1.0)')
parser.add_argument('--df', type=float, default=0.0,
                    help='gamma div parameter (default: 0)')
parser.add_argument('--prior_mu', type=float, default=0,
                    help='prior_mu')
parser.add_argument('--prior_logvar', type=float, default=0,
                    help='prior_logvar')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--seed', type=int, default=999,
                    help='set seed number (default: 999)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--zdim',  type=int, default=32,
                    help='the z size for training (default: 512)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--frac', type=float, default=0.5,
                    help='fraction of noisy dataset')
parser.add_argument('--no_cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('-s', '--save', action='store_true', default=True,
                    help='save model every epoch')
parser.add_argument('-l', '--load', action='store_true',
                    help='load model at the begining')
parser.add_argument('-dt','--dataset', type=str, default="mnist",
                    help='Dataset name')
parser.add_argument('--model_dir', type=str, default='',
                    help='model storing path')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')

args = parser.parse_args()

SEED = args.seed
## Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
## Deterministic operations are often slower than nondeterministic operations.
torch.backends.cudnn.deterministic = True


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

alpha = args.alpha
beta = args.beta
df = args.df

print(f'Current alpha : {alpha}')
print(f'Current beta : {beta}')
print(f'Current df : {df}')

if args.load:
    model_dir = args.model_dir
else:
    model_dir = './'+args.dataset+ f'_model_save_alpha{alpha}_beta{beta}_df{df}/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
## For tensorboard ##
writer = SummaryWriter(model_dir + 'Tensorboard_results')
def train(train_loader, encoder, decoder, opt, epoch, prior_mu, prior_logvar, alpha, beta, df):
    encoder.train()
    decoder.train()
    total_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        opt.zero_grad()
        z, mu, logvar = encoder(data)
        div_loss = encoder.loss(mu, logvar, prior_mu, prior_logvar, alpha, beta, df)
        recon_img = decoder(z)
        recon_loss = decoder.loss(recon_img, data, input_dim)
        current_loss = div_loss + recon_loss
        current_loss.backward()

        total_loss.append(current_loss.item())
        opt.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       current_loss.item() / len(data)))

            writer.add_scalar("Train/Reconstruction Error", recon_loss.item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )
            writer.add_scalar("Train/KL-Divergence", div_loss.item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )
            writer.add_scalar("Train/Total Loss" , current_loss.item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )

        
    return total_loss


def reconstruction(test_loader, encoder, decoder, ep, prior_mu, prior_logvar, alpha, beta, df=0):
    encoder.eval()
    decoder.eval()
    vectors = []

    for batch_idx, (data, labels) in enumerate(test_loader):
        with torch.no_grad():
            data = data.to(DEVICE)
            z, mu, logvar = encoder(data)
                        
            div_loss = encoder.loss(mu, logvar, prior_mu, prior_logvar, alpha, beta, df)
            recon_img = decoder(z)
            recon_loss = decoder.loss(recon_img, data, input_dim)

            current_loss = div_loss + recon_loss

            ## Caculate SSIM ##
            img1 = data.cpu()
            img2 = recon_img.cpu().view_as(data)
            ssim = StructuralSimilarityIndexMeasure()
            ssim_test = ssim(img1, img2)
            ##
            writer.add_scalar("Test/SSIM", ssim_test.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
            writer.add_scalar("Test/Reconstruction Error", recon_loss.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
            writer.add_scalar("Test/KL-Divergence", div_loss.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
            writer.add_scalar("Test/Total Loss" , current_loss.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
            
            recon_img = recon_img.view(-1, 1, image_size, image_size)


            if batch_idx % 20 == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(test_loader.dataset),
                           100. * batch_idx / len(test_loader),
                           current_loss.item() / len(data)))
        if batch_idx == 0:
            n = min(data.size(0), 32)
            comparison = torch.cat([data[:n], recon_img.view(args.batch_size, 1, 28, 28)[:n]]) # (16, 1, 28, 28)
            grid = torchvision.utils.make_grid(comparison.cpu()) # (3, 62, 242)
            writer.add_image("Test image - Above: Real data, below: reconstruction data", grid, epoch)

    return




## Load trainset, testset and trainloader, testloader ###
# transform.Totensor() is used to normalize mnist data. Range : [0, 255] -> [0,1]

MNISTC = MNISTC_Dataset()
trainset, testset = MNISTC.get_dataset('identity')
                              
# Load MNIST-C dataset
if args.dataset != "mnist":
    train_N = 60000 # Total : 60000
    test_N = 10000
    noise_trainset, noise_testset = MNISTC.get_dataset(args.dataset)

    def make_masking(N,frac):

        indice = np.arange(0,N)
        mask = np.zeros(N,dtype=bool)
        rand_indice = np.random.choice(N, int(frac*N))
        mask[rand_indice] = True
        
        return indice[mask], indice[~mask]

    I1, I2 = make_masking(train_N,args.frac)
    trainset = torch.utils.data.Subset(trainset, indices=I1)       
    noise_trainset = torch.utils.data.Subset(noise_trainset, indices=I2)                                          
    
    i1, i2 = make_masking(test_N,args.frac)                                  
    testset = torch.utils.data.Subset(testset, indices=i1)
    noise_testset = torch.utils.data.Subset(noise_testset, indices=i2)
   
    # Train with MNISTC, and reconstruct MNIST data
    
    trainset = torch.utils.data.ConcatDataset([trainset, noise_trainset])
    testset = torch.utils.data.ConcatDataset([testset, noise_testset])



trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)



## INIT ##
image_size = 28
input_dim = 784 # 28**2 : MNIST (I'll generalize this param for any dataset)

encoder = Encoder(input_dim, args.zdim, DEVICE).to(DEVICE)
decoder = Decoder(input_dim, args.zdim, device=DEVICE).to(DEVICE)
lr = args.lr
opt = optim.Adam(list(encoder.parameters()) +
                 list(decoder.parameters()), lr=lr, eps=1e-6, weight_decay=1e-5)

# ============== run ==============
if args.load:
    state = torch.load(model_dir+'Vanilavae.tch')
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    state2 = torch.load(model_dir+'Vanilavae.tchopt')
    ep = state2["ep"]+1
    opt.load_state_dict(state2["opt"])
    reconstruction(testloader, encoder, decoder, ep,alpha, beta)
    writer.close()
else:
    for epoch in tqdm(range(0, args.epochs)):
        train(trainloader, encoder, decoder, opt, epoch, args.prior_mu, args.prior_logvar, alpha, beta, df)
        reconstruction(testloader, encoder, decoder, epoch, args.prior_mu, args.prior_logvar,alpha, beta, df)
    writer.close()

