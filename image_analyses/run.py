import numpy as np
import torch
from datetime import datetime
import torchvision
import os
import random
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from models import *
from dataloader import *


parser = argparse.ArgumentParser(description='t3VAE')
parser.add_argument('--model', type=str, default="VAE",
                    help='model name')
parser.add_argument('--dataset', type=str, default="celebA",
                    help='Dataset name')
parser.add_argument('--datapath', type=str, default="./",
                    help='Dataset path')
parser.add_argument('--dirname', type=str, default="",
                    help='directory name')
parser.add_argument('--nu', type=float, default=0.0,
                    help='gamma pow div parameter')
parser.add_argument('--epoch', type=int, default=50,
                    help='Train epoch')
parser.add_argument('--seed', type=int, default=2023,
                    help='set seed number')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--m_dim',  type=int, default=64,
                    help='latent_dimension')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--beta_weight', type=float, default=1.0,
                    help='weight for regularizer')
parser.add_argument('--prior_sigma', type=float, default=1.0,
                    help='sigma value used in reconstruction term')
parser.add_argument('--tilt', type=float, default=40,
                    help='tilting parameter (Only used in TitledVAE)')
parser.add_argument('--TC_gamma', type=float, default=6.4,
                    help='TC regularizer weight (Only used in FactorVAE)')
parser.add_argument('--lr_D', default=1e-5, type=float, help='learning rate of the discriminator(Only used in FactorVAE)')
parser.add_argument('--int_K', type=float, default=1,
                    help='nb of numerical integral in DisentanglementVAE')
parser.add_argument('--imb', type=float, default=100,
                    help='tail imbalance factor(CIFAR)')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu number')

def load_model(model_name,img_shape,DEVICE, args):
    if model_name == 'VAE':
       return VAE.VAE(img_shape, DEVICE,args).to(DEVICE)
    elif model_name == 't3VAE':
        return t3VAE.t3VAE(img_shape, DEVICE, args).to(DEVICE)
    elif model_name == "TVAE":
        return TVAE.TVAE(img_shape, DEVICE, args).to(DEVICE)
    elif model_name == "TiltedVAE":
        return TiltedVAE.TiltedVAE(img_shape, DEVICE, args).to(DEVICE)
    elif model_name == "FactorVAE":
        return FactorVAE.FactorVAE(img_shape, DEVICE, args).to(DEVICE)
    elif model_name == "DEVAE":
        return DisentanglementVAE.DisentanglementVAE(img_shape, DEVICE, args).to(DEVICE)
    elif model_name == "t3HVAE":
        return t3HVAE.t3HVAE(img_shape, DEVICE, args).to(DEVICE)
    elif model_name == "HVAE":
        return HVAE.HVAE(img_shape, DEVICE, args).to(DEVICE)
    
    else:
        raise Exception("Please use one of the available model type!")
    
def make_result_dir(dirname):
    os.makedirs(dirname,exist_ok=True)
    os.makedirs(dirname + '/imgs',exist_ok=True)
    

def make_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device(f'cuda:{args.gpu}' if USE_CUDA else "cpu")

    make_reproducibility(args.seed)
    if args.dirname == "":
        args.dirname = './'+args.dataset+ f'_{args.model}_seed:{args.seed}_{datetime.today().strftime("%Y%m%d%H%M%S")}'
        if args.model == 't3VAE':
            args.dirname += f'nu:{args.nu}'
    
    make_result_dir(args.dirname)
    print(f'Current directory name : {args.dirname}')
    writer = SummaryWriter(args.dirname + '/Tensorboard_results')

    print(f"Current framework : {args.model}, lr : {args.lr}")
    if args.model == 't3VAE':
        if args.nu <= 2:
            raise Exception("Degree of freedom nu must be larger than 2")
        print(f'nu : {args.nu}')
    
    ## Load Dataset ##
    dataloader_setup = load_dataset(args.batch_size,args.dataset, args.datapath,args.imb)
    trainloader, testloader, sample_imgs = dataloader_setup.select_dataloader()
    sample_imgs = sample_imgs.to(DEVICE)
    img_shape = sample_imgs.shape # img shape : [B, C, H, W]
        
    
    ## Load Model ##
    model = load_model(args.model,img_shape, DEVICE, args)
    model_best_loss = 1e8

    epoch_tqdm = tqdm(range(0, args.epoch))
    denom_train = len(trainloader.dataset)/args.batch_size
    denom_test = len(testloader.dataset)/args.batch_size


    opt = optim.Adam(model.parameters(), lr=args.lr)

    # Use discriminator
    if args.model in ["FactorVAE"]: 
        discriminator_opt = optim.Adam(model.discriminator.parameters(), lr=args.lr_D)


    ## Train & Test ##
    for epoch in epoch_tqdm:
        ## Train ##
        model.train()
        total_loss = []
        tqdm_trainloader = tqdm(trainloader)
        total_time = []
        for batch_idx, (x, _) in enumerate(tqdm_trainloader):
            opt.zero_grad()
            if "HVAE" in args.model:
                recon_x,z1, z2, mu1, mu2, logvar1, logvar2 = model.forward(x.to(DEVICE))
            else:
                recon_x, z, mu, logvar = model.forward(x.to(DEVICE))

            
            if args.model == "FactorVAE":
                reg_loss, recon_loss, total_loss, vae_tcloss = model.loss(x.to(DEVICE), recon_x, z, mu, logvar)
                total_loss.backward(retain_graph=True)
                z = z.detach()
                discriminator_opt.zero_grad()
                TC_loss = model.TC_loss(z)
                TC_loss.backward()
                opt.step()
                discriminator_opt.step()
                tqdm_trainloader.set_description(f'train {epoch} : reg={reg_loss:.4f} recon={recon_loss:.4f} vae_tc={vae_tcloss:.4f} total={total_loss:.4f}')
            
            elif "HVAE" in args.model:
                reg_loss, reg_loss2, recon_loss, total_loss = model.loss(x.to(DEVICE), recon_x, z1, mu1, mu2, logvar1, logvar2)
                total_loss.backward()
                opt.step()
                tqdm_trainloader.set_description(f'train {epoch} : reg1={reg_loss:.4f} reg2={reg_loss2:.4f} recon={recon_loss:.4f} total={total_loss:.4f}')
            
            else:
                reg_loss, recon_loss, total_loss = model.loss(x.to(DEVICE), recon_x, z, mu, logvar)
                total_loss.backward()
                opt.step()
                tqdm_trainloader.set_description(f'train {epoch} : reg={reg_loss:.4f} recon={recon_loss:.4f} total={total_loss:.4f}')
            # break NAN ##
            if torch.isnan(total_loss):
                print('WARNING: finding nan loss.. stop the current train!')
                exit()

            if args.model == "TiltedVAE":
                # clip gradients with max_grad_norm = 100
                for group in opt.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], 100, norm_type=2)
        
        ## Test ##
        model.eval()
        cnt = 0
        total_loss_final = 0 
        with torch.no_grad():
            tqdm_testloader = tqdm(testloader)
            # total_loss_list = []
            for batch_idx, (x, _) in enumerate(tqdm_testloader):
                
                N = x.shape[0]
                cnt += 1
                if "HVAE" in args.model:
                    recon_x, z1, z2, mu1, mu2, logvar1, logvar2 = model.forward(x.to(DEVICE))
                else:
                    recon_x, z, mu, logvar = model.forward(x.to(DEVICE))

                
                if args.model == "FactorVAE":
                    reg_loss, recon_loss, total_loss, vae_tcloss = model.loss(x.to(DEVICE), recon_x, z, mu, logvar)
                    tqdm_testloader.set_description(f'test {epoch} : reg={reg_loss:.4f} recon={recon_loss:.4f} vae_tc={vae_tcloss:.4f} total={total_loss:.4f}')
                elif "HVAE" in args.model:
                    reg_loss, reg_loss2, recon_loss, total_loss = model.loss(x.to(DEVICE), recon_x, z1, mu1, mu2, logvar1, logvar2)
                    tqdm_testloader.set_description(f'test {epoch} : reg1={reg_loss:.4f} reg2={reg_loss2:.4f} recon={recon_loss:.4f} total={total_loss:.4f}')
                else:
                    reg_loss, recon_loss, total_loss = model.loss(x.to(DEVICE), recon_x, z, mu, logvar)
                    tqdm_testloader.set_description(f'test {epoch} :reg={reg_loss:.4f} recon={recon_loss:.4f} total={total_loss:.4f}')

                total_loss_final += total_loss.item()

                current_step = batch_idx + epoch * denom_test
                if batch_idx % 200 == 0:    
                    writer.add_scalar("Test/Reconstruction Error", recon_loss.item(), current_step )
                    writer.add_scalar("Test/Regularizer", reg_loss.item(), current_step)
                    writer.add_scalar("Test/Total Loss" , total_loss.item(), current_step)
                
                for images, _ in testloader:
                    real_imgs = images
                    break
                recon_imgs, *_ = model.forward(real_imgs[:64].to(DEVICE))     
                filename = f'{args.dirname}/imgs/reconstructions_{epoch}.png'         
                torchvision.utils.save_image(recon_imgs, filename,normalize=True, nrow=8)


                gen_imgs = model.generate()   
                filename = f'{args.dirname}/imgs/generations_{epoch}.png'         
                torchvision.utils.save_image(gen_imgs, filename,normalize=True, nrow=8)


            ## Save the best model ##
            total_loss_final /= cnt
            if total_loss_final < model_best_loss:
                print("Update the best model..!\n")
                model_best_loss = total_loss_final
                torch.save(model, f'{args.dirname}/{args.model}_best.pt')
                    
               
    model.eval()
    for images, _ in testloader:
        real_imgs = images
        break
    recon_imgs, *_ = model.forward(real_imgs[:64].to(DEVICE))     
    filename = f'{args.dirname}/imgs/reconstructions.png'         
    torchvision.utils.save_image(recon_imgs, filename,normalize=True, nrow=8)


    gen_imgs = model.generate()   
    filename = f'{args.dirname}/imgs/generations.png'         
    torchvision.utils.save_image(gen_imgs, filename,normalize=True, nrow=8)
    writer.close()
