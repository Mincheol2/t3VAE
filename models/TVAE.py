import torch
import torchvision
import torch.optim as optim
from torch.nn import functional as F
import math

from models import baseline

class TVAE(baseline.VAE_Baseline):
    def __init__(self, image_shape, DEVICE, args):
        super(TVAE, self).__init__(image_shape, DEVICE,args)
        self.opt = optim.Adam(list(self.parameters()), lr=args.lr, eps=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma = 0.99)
        
        '''
            T-VAE : add three-level layers and parameter layers : mu, lambda, nu
            Although the original code use one-dimension prior for each pixel,
            we use Multivariate Gamma prior instead.
            -> Therefore, we learn "one-dimensional" nu and lambda.
        ''' 
        n_h = 500
        n_latent = self.pdim
        self.linear_layers = nn.Sequential(
            nn.Linear(n_latent, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
        )
        # init parameters
        self.mu = 0
        self.loglambda = 0
        self.lognu = 0

        # parameter layers
        self.mu_layer = nn.Linear(n_h, self.pdim) # locale params
        self.lambda_layer = nn.Linear(n_h, 1) # scale params
        self.nu_layer = nn.Linear(n_h, 1) # degree of freedom
    
    def encoder(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)
        mu = self.mu_layer(x)
        logvar = self.mu_layer(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def decoder(self, z):
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
        reg_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1), dim=0)
        

        lambda_z = self.loglambda.exp()
        nu = self.lognu.exp()
        lgamma_term = torch.lgamma((nu + self.pdim)/2) - torch.lgamma(nu/2)
        log_term = self.pdim/2 * (torch.log(lambda_z/np) - torch.log(np.pi * self.nu_z))
        log_recon = (nu + p)/2 * torch.log(1 + self.lambda_z / self.nu_z * torch.linalg.norm(x-self.mu, ord=2, dim=1).pow(2))
        
        recon_loss = torch.mean(torch.sum(lgamma_term + log_term - log_recon,dim=1),dim=0)

        total_loss = self.args.reg_weight * reg_loss + recon_loss
        return reg_loss, recon_loss, total_loss

    def generate(self):
        prior_z = torch.randn(64, self.args.qdim)
        prior_z = self.args.recon_sigma * prior_z
        VAE_gen = self.decoder(prior_z.to(self.DEVICE)).detach().cpu()
        VAE_gen = VAE_gen *0.5 + 0.5
        return VAE_gen