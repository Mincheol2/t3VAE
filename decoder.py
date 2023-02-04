import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import random
import math
import numpy as np



class Decoder(nn.Module):
    def __init__(self, output_dim, z_dim, device):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.device = device
        self.decFC1 = nn.Linear(z_dim, 32*20*20)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.norm1 = nn.BatchNorm2d(16)

        self.decConv2 = nn.ConvTranspose2d(16, 1, 5)

    def forward(self, enc_z):
        z = F.relu(self.decFC1(enc_z))
        z = z.view(-1, 32, 20, 20)
        z = F.leaky_relu(self.norm1(self.decConv1(z)))
        prediction = torch.sigmoid(self.decConv2(z))

        return prediction


    def loss(self, recon_x, x, input_dim):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction = 'sum')
        
        # MSE loss is very high when trianing MNIST Dataset.
        #recon_loss = F.mse_loss(recon_x, x, reduction = 'mean')
        return recon_loss

    