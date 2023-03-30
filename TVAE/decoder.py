import torch
import torch.nn as nn
from torch.nn import functional as F
import argument
import loss
import math

args = argument.args



class Decoder(nn.Module):
    def __init__(self, img_shape):
        super(Decoder, self).__init__(), 
        self.B, self.C, self.H, self.W = img_shape
        self.hidden_dims = [512,256,128, 64,32]
        self.n = len(self.hidden_dims)
        self.pdim = self.hidden_dims[0]* math.ceil(self.H / 2**self.n) * math.ceil(self.W / 2**self.n)
        self.linear = nn.Sequential(
                        nn.Linear(args.zdim, self.pdim),
                        nn.ReLU(),
        )
        layers = []
        input_ch = self.hidden_dims[0]
        for dim in self.hidden_dims[1:]:
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(input_ch, dim, 
                    kernel_size = 3, stride = 2, padding=1, output_padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU())
            )
            input_ch = dim

        self.tp_cnn_layers = nn.Sequential(*layers)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[-1],
                                                            kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.BatchNorm2d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.ConvTranspose2d(self.hidden_dims[-1], self.C,
                                               kernel_size=3, padding=1),
                            nn.Tanh()
                            )
        
        # T-VAE : add three-level layers and parameter layers : mu, lambda, nu
        # In the original code, n_h is 500. But note that n_latent is 2.. :()
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
        self.lambda_layer = nn.Linear(n_h, self.pdim) # scale params
        self.nu_layer = nn.Linear(n_h, self.pdim) # degree of freedom

    def forward(self, z):
        # parameter learning
        z_params = self.linear_layers(z)
        self.mu = self.mu_layer(z_params)
        self.loglambda = self.lambda_layer(z_params)
        self.lognu = self.nu_layer(z_params)

        z = self.linear(z)
        z = z.reshape(-1,self.hidden_dims[0],math.ceil(self.H / 2**self.n),math.ceil(self.W / 2**self.n))
        z = self.tp_cnn_layers(z)
        z = self.final_layer(z)
        return z

    def loss(self, x):
        # Refer to the original code... But I don't understand this algorithm..!
        lambda_z = self.loglambda.exp()
        nu = self.lognu.exp()
        lgamma_term = torch.lgamma((nu + self.pdim)/2) - torch.lgamma(nu/2)
        log_term = 0.5 * (torch.log(lambda_z) - torch.log(np.pi * self.mu_z))
        log_recon = (nu + 1)/2 * torch.log(1+ self.lambda_z / self.nu_z * (x-self.mu)**2)
        recon_loss = - torch.mean(torch.sum(lgamma_term + log_term - log_recon,dim=1),dim=0)
        return recon_loss
