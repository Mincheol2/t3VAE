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


        self.trainloader, self.testloader, self.sample_imgs = dataloader.select_dataloader()
        self.sample_imgs = self.sample_imgs.to(DEVICE)
    def train(self,epoch,writer):
        self.encoder.train()
        self.decoder.train()
        total_loss = []
        for batch_idx, (data, _) in enumerate(self.trainloader):
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

            if batch_idx % 200 == 0:
                N = data.shape[0]
                denom = len(self.trainloader.dataset)/args.batch_size
                writer.add_scalar("Train/Reconstruction Error", recon_loss.item() / N, batch_idx + epoch * denom )
                writer.add_scalar("Train/Regularizer", div_loss.item() / N, batch_idx + epoch * denom )
                writer.add_scalar("Train/Total Loss" , current_loss.item() / N, batch_idx + epoch * denom )
        return div_loss.item() / N, recon_loss.item() / N, current_loss.item() / N

    def test(self,epoch,writer):
        self.encoder.eval()
        self.decoder.eval()

        for batch_idx, (data, labels) in enumerate(self.testloader):
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
            if batch_idx % 200 == 0:
                denom = len(self.testloader.dataset)/args.batch_size
                writer.add_scalar("Test/SSIM", ssim_test.item(), batch_idx + epoch * denom )
                writer.add_scalar("Test/PSNR", psnr_test.item(), batch_idx + epoch * denom )
                writer.add_scalar("Test/RMSE", rmse_test.item(), batch_idx + epoch * denom )
                
                writer.add_scalar("Test/Reconstruction Error", recon_loss.item() /N, batch_idx + epoch * denom )
                writer.add_scalar("Test/Regularizer", div_loss.item() / N, batch_idx + epoch * denom )
                writer.add_scalar("Test/Total Loss" , current_loss.item() /N, batch_idx + epoch * denom)
                
        n = min(self.sample_imgs.shape[0], 32)
        sample_z, _, _ = self.encoder(self.sample_imgs[:n])
        recon_imgs = self.decoder(sample_z)
        comparison = torch.cat([self.sample_imgs[:n], recon_imgs.view(n, 1, 28, 28)[:n]]) # (N, 1, 28, 28)
        grid = torchvision.utils.make_grid(comparison.cpu()) # (3, 62, 242)
        writer.add_image("Test image - Above: Real data, below: reconstruction data", grid, epoch)
        return div_loss.item() / len(data), recon_loss.item() / len(data), current_loss.item() / len(data)
