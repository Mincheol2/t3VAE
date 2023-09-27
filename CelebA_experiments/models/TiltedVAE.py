import torch
from scipy.special import eval_genlaguerre as L 
import math
import numpy as np

from models import baseline

class TiltedVAE(baseline.VAE_Baseline):
    def __init__(self, image_shape, DEVICE, args):
        super(TiltedVAE, self).__init__(image_shape, DEVICE,args)
        self.n_dim = self.C * self.H * self.W
        self.m_dim = self.args.m_dim
        self.tilt = torch.tensor(float(self.args.tilt)) # tilting hyperparameter
        self.d = self.m_dim # size of the latent z vector
        self.DEVICE = DEVICE
        # Default option : use L2 loss, not use OOD and burn_in arguments.
        
        # optimizing for min kld
        def kld(mu, d):
            # no need to include z, since we run gradient descent...
            return -self.tilt*np.sqrt(np.pi/2)*L(1/2, d/2 -1, -(mu**2)/2) + (mu**2)/2

        steps = [1e-1, 1e-2, 1e-3, 1e-4]
        dx = 5e-3

        # inital guess (very close to optimal value)
        self.mu_star = np.sqrt(max(self.tilt**2 - self.d, 0))

        # run gradient descent (kld is convex)
        for step in steps:
            for _ in range(100):
                y1 = kld(self.mu_star-dx/2, self.d)
                y2 = kld(self.mu_star+dx/2, self.d)

                grad = (y2-y1)/dx
                self.mu_star -= grad*step

        #self.mu_star = util.kld_min(tilt, nz) tilt ->tau, d-> nz
        print('mu_star: {:.3f}'.format(self.mu_star))
             
    
    def encoder(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)
        mu = self.mu_layer(x)
 
        logvar = torch.zeros_like(mu) # The only difference.
        
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def decoder(self, z): 
        # We use L2 loss.

        z = self.linear(z)
        z = z.reshape(-1,self.decoder_hiddens[0],math.ceil(self.H / 2**self.n),math.ceil(self.W / 2**self.n))
        z = self.tp_cnn_layers(z)
        z = self.final_layer(z)
        return z
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # diagonal mat
        eps = torch.randn_like(std) # Normal dist : eps ~ N(0, I)
        return mu + std * eps
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, mu, logvar
        
    def loss(self, x, recon_x, z, mu, logvar):
        N = x.shape[0]
        x = x.view(N,-1)
        recon_x = recon_x.view(N,-1)
        
        mu_norm = torch.linalg.norm(mu, dim=1)
        
        # 2 * Original loss.
        reg_loss = 2 * (1/2 * torch.square(mu_norm - self.mu_star)).mean()
        recon_loss = self.args.beta_weight * torch.sum((recon_x - x)**2 / (N * self.args.prior_sigma**2))
        
        total_loss = reg_loss + recon_loss
        return reg_loss, recon_loss, total_loss

    
    def generate(self, N=64):
        
        # 1. generate random vectors on the unit sphere.
        sample_z = torch.randn(N, self.m_dim).to(self.DEVICE)
        spherical_z = torch.nn.functional.normalize(sample_z,dim=1)

        # 2. scale the length by a sample from the distribution to be length
        # r ~ N(\bar{||z||}, 1) \in R, where \bar{||z||} = \frac{1}{N} \sum_{1}^{N} ||z^i||
        # We approximated \bar{||z||} to mu_star

        tilted_prior = (spherical_z * self.mu_star).to(self.DEVICE)
        VAE_gen = self.decoder(tilted_prior)
        return VAE_gen