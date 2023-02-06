from encoder import Encoder
from decoder import Decoder
import argument
import dataloader
import torch
import torchvision
import torch.optim as optim
from util import *
from skimage.metrics import structural_similarity
'''
params 따로 oop화 할 것
writer
'''
args = argument.args
class gammaAE():
    def __init__(self, input_dim, image_size,DEVICE):
        self.input_dim = input_dim
        self.image_size = image_size
        self.DEVICE = DEVICE
        self.encoder = Encoder(self.input_dim, args.zdim, args.df).to(DEVICE)
        self.decoder = Decoder(self.input_dim, args.zdim, args.df).to(DEVICE)
        self.opt = optim.Adam(list(self.encoder.parameters()) +
                 list(self.decoder.parameters()), lr=args.lr, eps=1e-6, weight_decay=1e-5)
    
    def train(self,epoch):
        self.encoder.train()
        self.decoder.train()
        total_loss = []
        for batch_idx, (data, _) in enumerate(dataloader.trainloader):
            data = data.to(self.DEVICE)
            self.opt.zero_grad()
            z, mu, logvar = self.encoder(data)
            div_loss = self.encoder.loss(mu, logvar)
            recon_data = self.decoder(z)
            recon_loss = self.decoder.loss(recon_data, data, self.input_dim)
            current_loss = div_loss + recon_loss
            current_loss.backward()

            total_loss.append(current_loss.item())
            self.opt.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.trainloader.dataset),
                           100. * batch_idx / len(dataloader.trainloader),
                           current_loss.item() / len(data)))
                # writer.add_scalar("Train/Reconstruction Error", recon_loss.item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )
                # writer.add_scalar("Train/Divergence", div_loss.item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )
                # writer.add_scalar("Train/Total Loss" , current_loss.item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )
        return

    def test(self,epoch):
        self.encoder.eval()
        self.decoder.eval()
        vectors = []
        for batch_idx, (data, labels) in enumerate(dataloader.testloader):
            with torch.no_grad():
                data = data.to(self.DEVICE)
                z, mu, logvar = self.encoder(data)
                            
                div_loss = self.encoder.loss(mu, logvar)
                recon_img = self.decoder(z)
                recon_loss = self.decoder.loss(recon_img, data, self.input_dim)

                current_loss = div_loss + recon_loss

                ## Caculate SSIM ##
                img1 = data.cpu()
                img2 = recon_img.cpu().view_as(data)
                ssim = StructuralSimilarityIndexMeasure()
                ssim_test = ssim(img1, img2)
                ##
                # writer.add_scalar("Test/SSIM", ssim_test.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
                # writer.add_scalar("Test/Reconstruction Error", recon_loss.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
                # writer.add_scalar("Test/KL-Divergence", div_loss.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
                # writer.add_scalar("Test/Total Loss" , current_loss.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
                
                recon_img = recon_img.view(-1, 1, self.image_size, self.image_size)


                if batch_idx % 20 == 0:
                    print('Test Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.4f}'.format(
                        epoch, batch_idx * len(data), len(dataloader.testloader.dataset),
                               100. * batch_idx / len(dataloader.testloader),
                               current_loss.item() / len(data)))
            if batch_idx == 0:
                n = min(data.size(0), 32)
                comparison = torch.cat([data[:n], recon_img.view(args.batch_size, 1, 28, 28)[:n]]) # (16, 1, 28, 28)
                grid = torchvision.utils.make_grid(comparison.cpu()) # (3, 62, 242)
                # writer.add_image("Test image - Above: Real data, below: reconstruction data", grid, args.epoch)

        return
