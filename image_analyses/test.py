import numpy as np
import torch
import os
import random
import argparse
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from models import *
from dataloader import *

parser = argparse.ArgumentParser(description='t3VAE')
parser.add_argument('--dataset', type=str, default="celebA",
                    help='Dataset name')
parser.add_argument('--dirname', type=str, default="",
                    help='directory name')
parser.add_argument('--seed', type=int, default=2023,
                    help='set seed number')
parser.add_argument('--model_path', type=str, default="exp_env/HVAE_lr8e5/HVAE_best.pt",
                    help='model path')
parser.add_argument('--datapath', type=str, default="/data_intern2",
                    help='Dataset path')
    
def make_result_dir(dirname):
    os.makedirs(dirname,exist_ok=True)
    os.makedirs(dirname + '/reconstructions',exist_ok=True)
    

def make_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    ## init ##
    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device(f'cuda:0' if USE_CUDA else "cpu")
    make_reproducibility(args.seed)
    if args.dirname != '':
        make_result_dir(args.dirname)
    transform = transforms.Compose(
            [
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.ToTensor(),
            ]
        )
    testset = CustomCelebA(
    root=args.datapath,
    split='test',
    transform=transform,
    download=False,
    )

    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

    print(f'the number of batches : {len(testloader)}')
    img_shape = torch.tensor([64,3,64,64]) # img : [B, C, H, W]

    model_path = args.model_path
    model = torch.load(model_path, map_location=DEVICE)
    model.DEVICE = DEVICE
    print(f"load the best model from {model_path}")
    fid_recon = FrechetInceptionDistance(normalize=True).to(DEVICE)
    tqdm_testloader = tqdm(testloader)
    model.eval()
    test_z = []
    test_label = []
    sharpnesses = []
    filename = args.model_path + '_result.txt'
    with open(filename, 'w') as f:
        with torch.no_grad():
            tqdm_testloader = tqdm(testloader)
            for batch_idx, (x, label) in enumerate(tqdm_testloader):
                x = x.to(DEVICE)
                recon_x, *_ = model.forward(x)
                fid_recon.update(x, real=True)
                fid_recon.update(recon_x, real=False)

        sharpnesses = np.array(sharpnesses).mean()
        print("caculating recon fid scores....")
        fid_recon_result = fid_recon.compute()
        print(f'FID_RECON:{fid_recon_result}')
        f.write(f'FID_RECON:{fid_recon_result}\n')
        j = 1
        for i in range(40):
            # for j in range(2):
            fid_recon = FrechetInceptionDistance(normalize=True).to(DEVICE)
            indice = torch.where(testset.attr[:,i] == 1)[0]
            if i == 39:
                j = 0 
                indice = torch.where(testset.attr[:,i] == 0)[0] # Not Young (old)
            subset = torch.utils.data.Subset(testset,indice)
            testloader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=False)
            tqdm_testloader = tqdm(testloader)
            for batch_idx, (x, label) in enumerate(tqdm_testloader):
                x = x.to(DEVICE)
                recon_x, *_ = model.forward(x)
                fid_recon.update(x, real=True)
                fid_recon.update(recon_x, real=False)
            fid_recon_result = fid_recon.compute()
            
            print("caculating recon fid scores....")

            print(f'{testset.attr_names[i]}=={j} -> FID_RECON:{fid_recon_result}')
            f.write(f'{testset.attr_names[i]}=={j} -> FID_RECON:{fid_recon_result}\n')
