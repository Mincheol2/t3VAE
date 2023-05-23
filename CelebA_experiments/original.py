import numpy as np
import torch
import cv2
import torchvision
import os
import random
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from torchmetrics.image.fid import FrechetInceptionDistance
from models import *
from dataloader import *

  
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

def measure_sharpness(imgs):
    if len(imgs.shape) == 3: #[C, H, W]
        imgs = imgs.unsqueeze(0)

    N = imgs.shape[0]
    sharpness = 0 
    for img in imgs:
        # 1. convert img to greyscale
        grey_img = torchvision.transforms.functional.rgb_to_grayscale(img).numpy()

        # 2. convolved with the laplace filter
        mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        laplacian = cv2.filter2D(grey_img, -1, mask)
        # 3.compute var of filtered img.
        sharpness += np.var(laplacian)

    return sharpness / N 

if __name__ == "__main__":
    ## init ##
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device(f'cuda' if USE_CUDA else "cpu")
    transform = transforms.Compose(
            [
            # transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(64),
            # transforms.Resize(128),
            transforms.ToTensor(),
            ]
        )
    testset = MyCelebA(
    root="/data_intern/",
    split='test',
    transform=transform,
    download=False,
    )


    # i = 18 # Heavy_Makeup
    i = 12 # Bush eyebrow
    logical = torch.logical_and(testset.attr[:,11] == 1, testset.attr[:,17] == 1) # Doublechin & Pale skinindice = torch.where(logical)[0]
    indice = torch.where(logical)[0]
    subset = torch.utils.data.Subset(testset,indice)
    testloader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=False)
          
    
    make_reproducibility(2023)
    make_result_dir(testset.attr_names[i])
    with torch.no_grad():
        tqdm_testloader = tqdm(testloader)
        for batch_idx, (x, label) in enumerate(tqdm_testloader):
            filename = f"./browngray/origin_{batch_idx}.png" 
            torchvision.utils.save_image(x, filename,normalize=True, nrow=8)
            # fid_recon.update(x, real=True)
            # fid_recon.update(recon_x, real=False)
            if batch_idx > 20:
                break