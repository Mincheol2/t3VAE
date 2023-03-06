import torch
import torch.nn as nn
from torch.nn import functional as F
import argument
import loss

args = argument.args



class Decoder(nn.Module):
    def __init__(self, output_dim):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.fc3 = nn.Linear(args.zdim, 400)
        self.fc4 = nn.Linear(400,output_dim)
    def forward(self, enc_z):
        z = F.relu(self.fc3(enc_z))
        prediction = torch.sigmoid(self.fc4(z))
        return prediction

    def loss(self, recon_x, x):
        if args.beta == 0:
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction = 'sum') / args.recon_sigma**2
            # recon_loss = F.mse_loss(recon_x, x, reduction = 'mean') / args.recon_sigma**2
        else:
            # RVAE : According to the original code, the sigma value is 0.5.
            recon_loss = loss.beta_div_loss(recon_x, x, args.beta, sigma=0.2)

        return recon_loss
