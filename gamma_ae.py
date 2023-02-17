from encoder import Encoder
from decoder import Decoder
import argument
import dataloader
import torch
import torchvision
import torch.optim as optim
from util import *
from skimage.metrics import structural_similarity


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
    
    def train(self,epoch,writer):
        self.encoder.train()
        self.decoder.train()
        total_loss = []
        for batch_idx, (data, _) in enumerate(dataloader.trainloader):
            if args.dataset == 'fashion':
                len_data = len(data)
                rand_indices = np.random.choice(len_data, int(len_data * args.train_frac))
                data[rand_indices] = make_masking(data[rand_indices], 0.5) # [B, C, H, W]
            data = data.to(self.DEVICE) 
            self.opt.zero_grad()
            z, mu, logvar = self.encoder(data)
            div_loss = self.encoder.loss(mu, logvar, self.input_dim)
            recon_data = self.decoder(z)
            recon_loss = self.decoder.loss(recon_data, z, data, mu, logvar, self.input_dim)

            # # ### TEST CODE ###
            # print(f'div_loss : {div_loss}')
            # print(f'recon_loss : {recon_loss}')
            # if np.isnan(recon_loss.cpu().detach().numpy()):
            #     print("NAN!")
            # ####
            current_loss = div_loss + recon_loss
            current_loss.backward()

            total_loss.append(current_loss.item())
            self.opt.step()

            if batch_idx % 50 == 0:
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
        vectors = []
        for batch_idx, (data, labels) in enumerate(dataloader.testloader):
            with torch.no_grad():
                data = data.to(self.DEVICE)
                z, mu, logvar = self.encoder(data)
                            
                div_loss = self.encoder.loss(mu, logvar, self.input_dim)
                recon_img = self.decoder(z)
                recon_loss = self.decoder.loss(recon_img, z, data, mu, logvar, self.input_dim)
                current_loss = div_loss + recon_loss

                ## Caculate SSIM ##
                img1 = data.cpu()
                img2 = recon_img.cpu().view_as(data)
                ssim = StructuralSimilarityIndexMeasure()
                ssim_test = ssim(img1, img2)
                ##
                writer.add_scalar("Test/SSIM", ssim_test.item(), batch_idx + epoch * (len(dataloader.testloader.dataset)/args.batch_size) )
                writer.add_scalar("Test/Reconstruction Error", recon_loss.item(), batch_idx + epoch * (len(dataloader.testloader.dataset)/args.batch_size) )
                writer.add_scalar("Test/KL-Divergence", div_loss.item(), batch_idx + epoch * (len(dataloader.testloader.dataset)/args.batch_size) )
                writer.add_scalar("Test/Total Loss" , current_loss.item(), batch_idx + epoch * (len(dataloader.testloader.dataset)/args.batch_size) )
                
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
                writer.add_image("Test image - Above: Real data, below: reconstruction data", grid, epoch)

        return
