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
        self.pdim  = self.hidden_dims[0]* math.ceil(self.H / 2**self.n) * math.ceil(self.W / 2**self.n)
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
        
        # self.fc3 = nn.Linear(args.zdim, 400)
        # self.fc4 = nn.Linear(400,output_dim)
    def forward(self, z):
        z = self.linear(z)
        z = z.reshape(-1,self.hidden_dims[0],math.ceil(self.H / 2**self.n),math.ceil(self.W / 2**self.n))
        z = self.tp_cnn_layers(z)
        z = self.final_layer(z)
        return z

    def loss(self, recon_x, x):
        recon_loss = F.mse_loss(recon_x, x) / args.recon_sigma**2
        
        return recon_loss
