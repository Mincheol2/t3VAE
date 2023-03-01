from encoder import Encoder
from decoder import Decoder
import argument
import dataloader
import torch
import torchvision
import torch.optim as optim
from util import *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error as mse

args = argument.args
class gammaAE():
    def __init__(self, input_dim, image_size, DEVICE):
        self.input_dim = input_dim
        self.image_size = image_size
        self.DEVICE = DEVICE
        self.encoder = Encoder(self.input_dim, args.zdim, args.nu).to(DEVICE)
        self.decoder = Decoder(self.input_dim, args.zdim, args.nu).to(DEVICE)
        self.opt = optim.Adam(list(self.encoder.parameters()) +
                 list(self.decoder.parameters()), lr=args.lr, eps=1e-6, weight_decay=1e-5)

        if 'mnist' in args.dataset:
            self.trainloader, self.testloader = dataloader.load_mnist_dataset(args.dataset)
        else:
            self.trainloader, self.testloader = dataloader.load_fashion_dataset(args.dataset)

    def train(self,epoch,writer):
        self.encoder.train()
        self.decoder.train()
        total_loss = []
        for batch_idx, (data, _) in enumerate(dataloader.trainloader):
            data = data.to(self.DEVICE) 
            self.opt.zero_grad()
            z, mu, logvar = self.encoder(data)
            div_loss = self.encoder.loss(mu, logvar, self.input_dim)
            recon_data = self.decoder(z)
            recon_loss = self.decoder.loss(recon_data, data)
            current_loss = div_loss + recon_loss
            current_loss.backward()

            total_loss.append(current_loss.item())
            
            self.opt.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.trainloader.dataset),
                           100. * batch_idx / len(dataloader.trainloader),
                           current_loss.item() / len(data)))
                writer.add_scalar("Train/Reconstruction Error", recon_loss.item(), batch_idx + epoch * (len(dataloader.trainloader.dataset)/args.batch_size) )
                writer.add_scalar("Train/Divergence", div_loss.item(), batch_idx + epoch * (len(dataloader.trainloader.dataset)/args.batch_size) )
                writer.add_scalar("Train/Total Loss" , current_loss.item(), batch_idx + epoch * (len(dataloader.trainloader.dataset)/args.batch_size) )
        return

    def test(self,epoch,writer):
        self.encoder.eval()
        self.decoder.eval()

        for batch_idx, (data, labels) in enumerate(dataloader.testloader):
            with torch.no_grad():
                data = data.to(self.DEVICE)
                z, mu, logvar = self.encoder(data)
                            
                div_loss = self.encoder.loss(mu, logvar, self.input_dim)
                recon_img = self.decoder(z)
                recon_loss = self.decoder.loss(recon_img,data)
                current_loss = div_loss + recon_loss

                ## Caculate SSIM, PSNR, RMSE ##
                img1 = data.cpu().squeeze(dim=1).numpy()
                img2 = recon_img.cpu().view_as(data).squeeze(dim=1).numpy()
                ssim_test = 0
                psnr_test = 0 
                N = img1.shape[0]
                for i in range(N):
                    ssim_test += ssim(img1[i], img2[i])
                    psnr_test += psnr(img1[i], img2[i])
                    rmse_test = mse(img1[i], img2[i]) ** 0.5
                ssim_test /= N
                psnr_test /= N   
                rmse_test /= N
                ## Add metrics to tensorboard ##
                writer.add_scalar("Test/SSIM", ssim_test.item(), batch_idx + epoch * (len(dataloader.testloader.dataset)/args.batch_size) )
                writer.add_scalar("Test/PSNR", psnr_test.item(), batch_idx + epoch * (len(dataloader.testloader.dataset)/args.batch_size) )
                writer.add_scalar("Test/RMSE", rmse_test.item(), batch_idx + epoch * (len(dataloader.testloader.dataset)/args.batch_size) )
                
                writer.add_scalar("Test/Reconstruction Error", recon_loss.item(), batch_idx + epoch * (len(dataloader.testloader.dataset)/args.batch_size) )
                writer.add_scalar("Test/Divergence", div_loss.item(), batch_idx + epoch * (len(dataloader.testloader.dataset)/args.batch_size) )
                writer.add_scalar("Test/Total Loss" , current_loss.item(), batch_idx + epoch * (len(dataloader.testloader.dataset)/args.batch_size) )
                
                recon_img = recon_img.view(-1, 1, self.image_size, self.image_size)


                if batch_idx % 100 == 0:
                    print('Test Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.4f}'.format(
                        epoch, batch_idx * len(data), len(dataloader.testloader.dataset),
                               100. * batch_idx / len(dataloader.testloader),
                               current_loss.item() / len(data)))
                    
            if batch_idx == 0:
                n = min(data.size(0), 32)
                comparison = torch.cat([data[:n], recon_img.view(args.batch_size, 1, 28, 28)[:n]]) # (16, 1, 28, 28)
                grid = torchvision.utils.make_grid(comparison.cpu()) # (3, 62, 242)
                writer.add_image("Test image - Above: Real data, below: reconstruction data", grid, epoch)

        return
