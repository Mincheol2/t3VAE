import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, n_latent, n_h):
        super(Discriminator, self).__init__()

        self.n_latent = n_latent
        self.n_h = n_h

        # Layer
        self.layers = nn.Sequential(
            nn.Linear(n_latent, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h), nn.Tanh(), nn.Dropout(),
            nn.Linear(n_h, 1)
        )

        # loss
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, z):
        return self.layers(z).squeeze()
    
    def loss(self, logits_inferred, logits_sampled):

        ll_infered = self.criterion(logits_inferred, torch.ones_like(logits_inferred))
        ll_sampled = self.criterion(logits_sampled, torch.zeros_like(logits_sampled))

        return ll_infered + ll_sampled