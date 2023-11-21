import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import os
import gc
import random
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from models import *
from torch.utils.data import DataLoader, random_split

parser = argparse.ArgumentParser(description='t3VAE')
parser.add_argument('--model', type=str, default="VAE",
                    help='model name')
parser.add_argument('--dataset', type=str, default="snp",
                    help='Dataset name')
parser.add_argument('--datapath', type=str, default="/data_intern2",
                    help='Dataset path')
parser.add_argument('--dirname', type=str, default="",
                    help='directory name')
parser.add_argument('--nu', type=float, default=0.0,
                    help='gamma pow div parameter')
parser.add_argument('--epoch', type=int, default=50,
                    help='Train epoch')
parser.add_argument('--seed', type=int, default=2023,
                    help='set seed number')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--n_dim',  type=int, default=1,
                    help='data_dimension')
parser.add_argument('--m_dim',  type=int, default=1,
                    help='latent_dimension')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='learning rate')
parser.add_argument('--beta_weight', type=float, default=1.0,
                    help='weight for regularizer')
parser.add_argument('--prior_sigma', type=float, default=1.0,
                    help='sigma value used in reconstruction term')
parser.add_argument('--int_K', type=float, default=1,
                    help='nb of numerical integral in DisentanglementVAE')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu number')

def load_model(model_name,DEVICE, args):
    if model_name == 'VAE':
       return VAE.VAE(DEVICE,args).to(DEVICE)
    elif model_name == 't3VAE':
        return t3VAE.t3VAE(DEVICE, args).to(DEVICE)
    elif model_name == "TVAE":
        return TVAE.TVAE(DEVICE, args).to(DEVICE)
    elif model_name == "TiltedVAE":
        return TiltedVAE.TiltedVAE(DEVICE, args).to(DEVICE)
    elif model_name == "FactorVAE":
        return FactorVAE.FactorVAE(DEVICE, args).to(DEVICE)
    elif model_name == "DEVAE":
        return DisentanglementVAE.DisentanglementVAE(DEVICE, args).to(DEVICE)
    elif model_name == "t3HVAE":
        return t3HVAE.t3HVAE(DEVICE, args).to(DEVICE)
    elif model_name == "HVAE":
        return HVAE.HVAE(DEVICE, args).to(DEVICE)
    
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

    
class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.file_list = np.load(filename)
        self.length = len(self.file_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = torch.tensor(self.file_list[index],dtype=torch.float32)
        return data, 0 # We don't use any label now.

if __name__ == "__main__":
    ## init ##

    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device(f'cuda:{args.gpu}' if USE_CUDA else "cpu")

    make_reproducibility(args.seed)
    if args.dirname == "":
        args.dirname = './'+args.dataset+ f'_{args.model}_seed:{args.seed}_{datetime.today().strftime("%Y%m%d%H%M%S")}'
        if args.model == 't3VAE':
            args.dirname += f'nu:{args.nu}'
    else:
        args.dirname += f"_lr:{args.lr}_nu:{args.nu}"
    
    make_result_dir(args.dirname)
    print(f'Current directory name : {args.dirname}')
    writer = SummaryWriter(args.dirname + '/Tensorboard_results')
    print(f"Current framework : {args.model}, lr : {args.lr}")
    if args.model == 't3VAE':
        if args.nu <= 2:
            raise Exception("Degree of freedom nu must be larger than 2")
        print(f'nu : {args.nu}')
    
    ## Load Dataset ##
    dataset = Custom_Dataset(f"data/{args.dataset}.npy")
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.4)
    validation_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - validation_size

    train_set,val_set, test_set = random_split(dataset, [train_size, validation_size, test_size])

    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validationloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)    
    
    ## Load Model ##
    model = load_model(args.model, DEVICE, args)
    model_best_loss = 1e8

    epoch_tqdm = tqdm(range(0, args.epoch))
    denom_train = len(trainloader.dataset)/args.batch_size
    denom_test = len(testloader.dataset)/args.batch_size


    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in epoch_tqdm:
    
        ## Train ##
        model.train()
        total_loss = []
        tqdm_trainloader = tqdm(trainloader)
        total_time = []
        for batch_idx, (x, _) in enumerate(tqdm_trainloader):
            opt.zero_grad()
            recon_x, z, mu, logvar = model.forward(x.to(DEVICE))
             
            reg_loss, recon_loss, total_loss = model.loss(x.to(DEVICE), recon_x, z, mu, logvar)
            total_loss.backward()
            opt.step()
            tqdm_trainloader.set_description(f'train {epoch} : reg={reg_loss:.4f} recon={recon_loss:.4f} total={total_loss:.4f}')
            
            # break NAN ##
            if torch.isnan(total_loss):
                print('WARNING: finding nan loss.. stop the current train!')
                exit()

        
        ## Val & Test ##
        gc.collect()
        model.eval()
        cnt = 0
        total_loss_final = 0 
        with torch.no_grad():
            tqdm_valloader = tqdm(validationloader)
            for batch_idx, (x, _) in enumerate(tqdm_valloader):
                N = x.shape[0]
                cnt += 1
                recon_x, z, mu, logvar = model.forward(x.to(DEVICE))

                reg_loss, recon_loss, total_loss = model.loss(x.to(DEVICE), recon_x, z, mu, logvar)
                tqdm_valloader.set_description(f'val {epoch} :reg={reg_loss:.4f} recon={recon_loss:.4f} total={total_loss:.4f}')

                total_loss_final += total_loss.item()

                current_step = batch_idx + epoch * denom_test
                if batch_idx % 200 == 0:    
                    writer.add_scalar("Val/Reconstruction Error", recon_loss.item(), current_step )
                    writer.add_scalar("Val/Regularizer", reg_loss.item(), current_step)
                    writer.add_scalar("Val/Total Loss" , total_loss.item(), current_step)
             
            ## Save the best model ##
            total_loss_final /= cnt
            if total_loss_final < model_best_loss:
                print("Update the best model..!\n")
                model_best_loss = total_loss_final
                torch.save(model, f'{args.dirname}/{args.model}_best.pt')


            tqdm_testloader = tqdm(testloader)
            x_list = []
            recon_x_list = []
            for batch_idx, (x, _) in enumerate(testloader):
                cnt += 1
                recon_x, z, mu, logvar = model.forward(x.to(DEVICE))
                x_list.append(x)
                recon_x_list.append(recon_x)

                reg_loss, recon_loss, total_loss = model.loss(x.to(DEVICE), recon_x, z, mu, logvar)
                tqdm_valloader.set_description(f'test {epoch} :reg={reg_loss:.4f} recon={recon_loss:.4f} total={total_loss:.4f}')

                total_loss_final += total_loss.item()

                current_step = batch_idx + epoch * denom_test
                if batch_idx % 200 == 0:    
                    writer.add_scalar("Test/Reconstruction Error", recon_loss.item(), current_step )
                    writer.add_scalar("Test/Regularizer", reg_loss.item(), current_step)
                    writer.add_scalar("Test/Total Loss" , total_loss.item(), current_step)
            if epoch % 5 == 0:
                plt.clf()
                plt.plot(torch.concat(x_list).cpu().numpy(), label='origin', alpha=0.5)
                plt.plot(torch.concat(recon_x_list).cpu().numpy(), label='recon', alpha=0.5)
                plt.legend()
                plt.title(f"nu {args.nu}, lr {args.lr}")
                plt.savefig(args.dirname + f'/imgs/test_{epoch}.png')
    writer.close()
