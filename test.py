import numpy as np
import torch
import cv2
import torchvision
import os
import random
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error as mse
import torch.optim as optim
import seaborn as sns
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from models import *
from dataloader import *

parser = argparse.ArgumentParser(description='gammaAE')
parser.add_argument('--dataset', type=str, default="celebA",
                    help='Dataset name')
parser.add_argument('--dirname', type=str, default="",
                    help='directory name')
parser.add_argument('--seed', type=int, default=2023,
                    help='set seed number')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--model_path', type=str, default="",
                    help='model path')


    
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

def origin_sharpness(testloader):
    origin_sharpness_list = []
    for images, _ in testloader:
        for img in images:
            origin_sharpness_list.append(measure_sharpness(img))
    
    # 1e2 scale
    return 100 *np.array(origin_sharpness_list)

if __name__ == "__main__":
    ## init ##
    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device(f'cuda' if USE_CUDA else "cpu")
    make_reproducibility(args.seed)
    make_result_dir(args.dirname)
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

    indices = testset.attr[:,31] == 1
    subset = torch.utils.data.Subset(testset, 
    torch.tensor(np.arange(len(indices))[indices]))
    testloader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=False)
        
    img_shape = torch.tensor([64,3,64,64]) # img : [B, C, H, W]
    ## Load Model ##
    # model = load_model(args.model,img_shape, DEVICE, args).to(DEVICE)

    # model = torch.load("./new_VAE_FID2/VAE_best.pt")
    model_path = args.model_path
    model = torch.load(model_path)
    print(f"load the best model from {model_path}")
    fid_recon = FrechetInceptionDistance(normalize=True).to(DEVICE)
    
    fid_gen = FrechetInceptionDistance(normalize=True).to(DEVICE)
    ## Test ##
    model.eval()
    with torch.no_grad():
        with open(f'{args.dirname}/fid_score.txt', 'w') as f:
            tqdm_testloader = tqdm(testloader)
            for batch_idx, (x, _) in enumerate(tqdm_testloader):
                x = x.to(DEVICE)
                recon_x, z, mu, logvar = model.forward(x)
                ### Reconstruction ###
                fid_recon.update(x, real=True)
                fid_recon.update(recon_x, real=False)

                ### Generation ###
                gen_x = model.generate().to(DEVICE)
                fid_gen.update(x, real=True)
                fid_gen.update(gen_x, real=False)
            
                comparison = torch.cat([x.cpu() , recon_x.cpu()])
                recon_imgs = torchvision.utils.make_grid(comparison.cpu())
                filename = f'{args.dirname}/reconstructions/reconstruction_{batch_idx}.png'
                torchvision.utils.save_image(recon_imgs, filename,normalize=True, nrow=4)

                if batch_idx == 150:
                    break


            print("caculating fid scores....")
            fid_recon_result = fid_recon.compute()
            print(f'FID_RECON:{fid_recon_result}')
            fid_gen_result = fid_gen.compute()
            print(f'FID_GEN:{fid_gen_result}')
            f.write(f'FID_RECON:{fid_recon_result}')
            f.write(f'FID_GEN:{fid_gen_result}')

