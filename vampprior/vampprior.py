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
class VAE_vampprior():
    def __init__(self, DEVICE):
        
        self.DEVICE = DEVICE
        dataloader_setup = dataloader.load_dataset()
        self.trainloader, self.testloader, self.sample_imgs = dataloader_setup.select_dataloader()
        self.sample_imgs = self.sample_imgs.to(DEVICE)
        self.img_shape = self.sample_imgs.shape # img : [B, C, H, W]
        
        self.encoder = Encoder(self.img_shape,DEVICE).to(DEVICE)
        self.decoder = Decoder(self.img_shape).to(DEVICE)
        self.opt = optim.Adam(list(self.encoder.parameters()) +
                 list(self.decoder.parameters()), lr=args.lr, eps=1e-6, weight_decay=1e-5)

        
    def add_pseudoinputs(self,x):
        x = nn.tanh(self.mean_layer(x))
        return x
        
    def train(self,epoch,writer):
        self.encoder.train()
        self.decoder.train()
        total_loss = []
        for batch_idx, (data, _) in enumerate(self.trainloader):
            data = data.to(self.DEVICE)
            self.opt.zero_grad()
            z, mu, logvar = self.encoder(data)
            reg_loss = self.encoder.loss(mu, logvar)
            recon_data = self.decoder(z)
            B = data.shape[0]
            data = data.view(B,-1)
            recon_data = recon_data.view(B,-1)
            recon_loss = self.decoder.loss(recon_data, data)
            current_loss = reg_loss + recon_loss
            current_loss.backward()

            total_loss.append(current_loss.item())
            
            self.opt.step()

            if batch_idx % 200 == 0:
                N = data.shape[0]
                denom = len(self.trainloader.dataset)/args.batch_size
                writer.add_scalar("Train/Reconstruction Error", recon_loss.item() / N, batch_idx + epoch * denom )
                writer.add_scalar("Train/Regularizer", reg_loss.item() / N, batch_idx + epoch * denom )
                writer.add_scalar("Train/Total Loss" , current_loss.item() / N, batch_idx + epoch * denom )
        
        return reg_loss.item() / N, recon_loss.item() / N, current_loss.item() / N

    def test(self,epoch,writer):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.testloader):
                data = data.to(self.DEVICE)
                z, mu, logvar = self.encoder(data)
                            
                reg_loss = self.encoder.loss(mu, logvar)
                recon_data = self.decoder(z)
                B = data.shape[0]
                data = data.view(B,-1)
                recon_data = recon_data.view(B,-1)
                recon_loss = self.decoder.loss(data, recon_data)
                current_loss = reg_loss + recon_loss

                ## Caculate SSIM, PSNR, RMSE ##
                img1 = data.cpu().numpy()
                img2 = recon_data.cpu().numpy()
                ssim_test = 0
                psnr_test = 0
                mse_test = 0
                N = img1.shape[0]
                for i in range(N):
                    ssim_test += ssim(img1[i], img2[i],data_range=2)
                    psnr_test += psnr(img1[i], img2[i])
                    mse_test += mse(img1[i], img2[i])
                ssim_test /= N
                psnr_test /= N
                mse_test /= N
                ## Add metrics to tensorboard ##
                if batch_idx % 200 == 0:
                    denom = len(self.testloader.dataset)/args.batch_size
                    writer.add_scalar("Test/SSIM", ssim_test.item(), batch_idx + epoch * denom )
                    writer.add_scalar("Test/PSNR", psnr_test.item(), batch_idx + epoch * denom )
                    writer.add_scalar("Test/MSE", mse_test.item(), batch_idx + epoch * denom )
                    
                    writer.add_scalar("Test/Reconstruction Error", recon_loss.item() / N, batch_idx + epoch * denom )
                    writer.add_scalar("Test/Regularizer", reg_loss.item() / N, batch_idx + epoch * denom )
                    writer.add_scalar("Test/Total Loss" , current_loss.item() / N, batch_idx + epoch * denom)
                
            n = min(self.img_shape[0], 32)


            ### Reconstruction ###
            sample_z, _, _ = self.encoder(self.sample_imgs)
            test_imgs = self.decoder(sample_z).detach().cpu()

            writer.add_scalar("Test/Reconstruction Sharpness", measure_sharpness(test_imgs), epoch)
            
            sample_img_board = self.sample_imgs[:n] *0.5 +0.5
            test_img_board = test_imgs[:n] *0.5 +0.5
            comparison = torch.cat([sample_img_board.cpu() , test_img_board]) 

            grid = torchvision.utils.make_grid(comparison.cpu())
            writer.add_image("Test image - Above: Real data, below: reconstruction data", grid, epoch)
        
            ## Interpolation ##
            loop_iter = 1
            num_steps = 8
            for k in range(loop_iter):
                inter_z = []
                idx1, idx2, idx3, idx4 = np.random.choice(sample_z.shape[0], 4, replace=False)
                for j in range(num_steps):
                    for i in range(num_steps):
                        t = i / (num_steps -1)
                        s = j / (num_steps -1)
                        result1 = t * sample_z[idx1] + (1-t) * sample_z[idx2]
                        result2 = t * sample_z[idx3] + (1-t) * sample_z[idx4]
                        result = s * result1 + (1-s) * result2
                        inter_z.append(result.tolist())
                inter_img = 0.5 * self.decoder(torch.tensor(inter_z).to(self.DEVICE)) + 0.5
                inter_grid = torchvision.utils.make_grid(inter_img.cpu())
                filename = f'{args.dirname}interpolations/interpolation_{epoch}.png'
                torchvision.utils.save_image(inter_grid, filename)

            ## generation ##
            if args.nu == 0:
                prior_z = torch.randn(sample_z.shape[0], args.zdim)
                VAE_gen = self.decoder(prior_z.to(self.DEVICE)).detach().cpu()
                VAE_gen = VAE_gen *0.5 +0.5
                gen_grid = torchvision.utils.make_grid(VAE_gen)
                writer.add_scalar("Test/Generation Sharpness", measure_sharpness(VAE_gen), epoch)
            else:
                MVN_dist = torch.distributions.MultivariateNormal(torch.zeros(args.zdim), torch.eye(args.zdim))
                chi_dist = torch.distributions.chi2.Chi2(torch.tensor([args.nu]))
                prior_z = MVN_dist.sample(sample_shape=torch.tensor([sample_z.shape[0]])).to(self.DEVICE)
                v = chi_dist.sample().to(self.DEVICE)
                prior_t = prior_z * torch.sqrt(args.nu / v)
                gammaAE_gen = self.decoder(prior_t.to(self.DEVICE)).detach().cpu()
                gammaAE_gen = gammaAE_gen *0.5 +0.5
                gen_grid = torchvision.utils.make_grid(gammaAE_gen)
                writer.add_scalar("Test/Generation Sharpness", measure_sharpness(gammaAE_gen), epoch)

            filename = f'{args.dirname}generations/generation_{epoch}.png'
            torchvision.utils.save_image(gen_grid, filename)

            
        return reg_loss.item() / len(data), recon_loss.item() / len(data), current_loss.item() / len(data)