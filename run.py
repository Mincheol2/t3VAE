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

from models import *
from dataloader import *


parser = argparse.ArgumentParser(description='gammaAE')
parser.add_argument('--model', type=str, default="VAE",
                    help='model name')
parser.add_argument('--dataset', type=str, default="celebA",
                    help='Dataset name')
parser.add_argument('--dirname', type=str, default="",
                    help='directory name')
parser.add_argument('--nu', type=float, default=0.0,
                    help='gamma pow div parameter')
parser.add_argument('--epoch', type=int, default=20,
                    help='Train epoch')
parser.add_argument('--seed', type=int, default=2023,
                    help='set seed number')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--qdim',  type=int, default=64,
                    help='latent_dimension')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--reg_weight', type=float, default=1.0,
                    help='weight for regularizer')
parser.add_argument('--recon_sigma', type=float, default=1.0,
                    help='sigma value in reconstruction term')
parser.add_argument('--flat', type=str, default='y',
                    help='use gamma-pow regularizer')
parser.add_argument('--num_components', type=int, default=500,
                    help='number of pseudoinput components (Only used in VampPrior)')
                    
def load_model(model_name,img_shape,DEVICE, args):
    if model_name == 'VAE':
       return VAE.VAE(img_shape, DEVICE,args)
    elif model_name == 'TtAE':
        return TtAE.TtAE(img_shape, DEVICE, args)
    elif model_name == 'VampPrior':
        return VampPrior.VampPrior(img_shape, DEVICE, args)
    elif model_name == "TVAE":
        return TVAE.TVAE(img_shape, DEVICE, args)
    else:
        raise Exception("Please use appropriate model!", ['VAE', 'TtAE', 'VampPrior'])
    
def make_result_dir(dirname):
    os.makedirs(dirname,exist_ok=True)
    os.makedirs(dirname + '/interpolations',exist_ok=True)
    os.makedirs(dirname + '/generations',exist_ok=True)
    

def make_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def measure_sharpness(imgs):
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
    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device(f'cuda' if USE_CUDA else "cpu")
    make_reproducibility(args.seed)
    if args.dirname == "":
        args.dirname = './'+args.dataset+ f'_{args.model}_seed:{args.seed}_qdim:{args.qdim}'
        if args.model == 'TtAE':
            args.dirname += f'nu:{args.nu}'
    
    make_result_dir(args.dirname)
    writer = SummaryWriter(args.dirname + '/Tensorboard_results')
    
    print(f"Current framework : {args.model}")
    if args.model == 'TtAE':
        if args.nu <= 2:
            raise Exception("Degree of freedom nu must be larger than 2")
        print(f'nu : {args.nu}')
    # if args.flat != 'y':
        # print("We'll use KL divergence, despite TtAE")
    
    ## Load Dataset ##
    dataloader_setup = load_dataset(args.batch_size,args.dataset)
    trainloader, testloader, sample_imgs = dataloader_setup.select_dataloader()
    sample_imgs = sample_imgs.to(DEVICE)
    img_shape = sample_imgs.shape # img : [B, C, H, W]
        
    
    ## Load Model ##
    model = load_model(args.model,img_shape, DEVICE, args).to(DEVICE)
    model_best_loss = 1e8

    epoch_tqdm = tqdm(range(0, args.epoch))
    denom_train = len(trainloader.dataset)/args.batch_size
    denom_test = len(testloader.dataset)/args.batch_size

    ## Train & Test ##
    for epoch in epoch_tqdm:
        ## Train ##
        model.train()
        total_loss = []
        tqdm_trainloader = tqdm(trainloader)
        for batch_idx, (x, _) in enumerate(tqdm_trainloader):
            x = x.to(DEVICE)
            model.opt.zero_grad()
            recon_x, z, mu, logvar = model.forward(x)
            reg_loss, recon_loss, total_loss = model.loss(x, recon_x, z, mu, logvar)
            total_loss.backward()
            model.opt.step()
            if batch_idx % 200 == 0:
                current_step_train = batch_idx + epoch *denom_train
                writer.add_scalar("Train/Reconstruction Error", recon_loss.item(), current_step_train)
                writer.add_scalar("Train/Regularizer", reg_loss.item(), current_step_train)
                writer.add_scalar("Train/Total Loss" , total_loss.item(), current_step_train)
            tqdm_trainloader.set_description(f'train {epoch} : reg={reg_loss:.6f} recon={recon_loss:.6f} total={total_loss:.6f}')
        if model.scheduler is not None:
            model.scheduler.step()        
        
        
        ## Test ##
        model.eval()
        with torch.no_grad():
            tqdm_testloader = tqdm(testloader)
            for batch_idx, (x, _) in enumerate(tqdm_testloader):
                x = x.to(DEVICE)
                recon_x, z, mu, logvar = model.forward(x)
                reg_loss, recon_loss, total_loss = model.loss(x, recon_x, z, mu, logvar)
                            
                ## Add metrics to tensorboard ##
                if batch_idx % 200 == 0:
                    ## Caculate SSIM, PSNR, RMSE ##
                    img1 = x.cpu().numpy()
                    img2 = recon_x.cpu().numpy()
                    ssim_test = 0
                    psnr_test = 0
                    mse_test = 0
                    N = img1.shape[0]
                    for i in range(N):
                        # torch : [C, H, W] --> numpy : [H, W, C]
                        ssim_test += ssim(img1[i], img2[i], channel_axis=0, data_range=2.0)
                        psnr_test += psnr(img1[i], img2[i])
                        mse_test += mse(img1[i].flatten(), img2[i].flatten())
                    ssim_test /= N
                    psnr_test /= N
                    mse_test /= N
                    current_step = batch_idx + epoch * denom_test
                    writer.add_scalar("Test/SSIM", ssim_test.item(), current_step)
                    writer.add_scalar("Test/PSNR", psnr_test.item(), current_step)
                    writer.add_scalar("Test/MSE", mse_test.item(), current_step )
                    writer.add_scalar("Test/Reconstruction Error", recon_loss.item(), current_step )
                    writer.add_scalar("Test/Regularizer", reg_loss.item(), current_step)
                    writer.add_scalar("Test/Total Loss" , total_loss.item(), current_step)

                tqdm_testloader.set_description(f'test {epoch} :reg={reg_loss:.6f} recon={recon_loss:.6f} total={total_loss:.6f}')
            
            ## Save the best model ##
            if total_loss < model_best_loss:
                model_best_loss = total_loss
                torch.save(model, f'{args.dirname}/{args.model}_best.pt')


            ### Reconstruction ###
            nb_recons = 32
            test_imgs, *_ = model.forward(sample_imgs)
            test_imgs = test_imgs.detach().cpu()

            writer.add_scalar("Test/Reconstruction Sharpness", measure_sharpness(test_imgs), epoch)
            
            sample_img_board = sample_imgs[:nb_recons] *0.5 +0.5
            test_img_board = test_imgs[:nb_recons] *0.5 +0.5
            comparison = torch.cat([sample_img_board.cpu() , test_img_board])

            grid = torchvision.utils.make_grid(comparison.cpu())
            writer.add_image(f"Test image - Above {nb_recons}: Real images, below {nb_recons} : reconstruction images", grid, epoch)
        
            
            ## generation ##
            gen_imgs = model.generate()
            gen_grid = torchvision.utils.make_grid(gen_imgs)
            writer.add_scalar("Test/Generation Sharpness", measure_sharpness(gen_imgs), epoch)
            filename = f'{args.dirname}/generations/generation_{epoch}.png'
            torchvision.utils.save_image(gen_grid, filename)
    writer.close()
