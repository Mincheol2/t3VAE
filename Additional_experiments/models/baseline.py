import torch.nn as nn
class VAE_Baseline(nn.Module):
    def __init__(self, DEVICE, args):
        super().__init__()
        self.args = args
        self.n_dim = self.args.n_dim
        self.m_dim = self.args.m_dim
        self.DEVICE = DEVICE
        self.num_layers=16

        # Encoder
        self.encoder_net = nn.Sequential(
            nn.Linear(self.n_dim, self.num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(self.num_layers, self.num_layers), 
            nn.LeakyReLU()
        )
        self.mu_layer = nn.Linear(self.num_layers, self.m_dim)
        self.logvar_layer = nn.Linear(self.num_layers, self.m_dim)

        # Decoder
        self.decoder_net = nn.Sequential(
            nn.Linear(self.m_dim, self.num_layers), 
            nn.LeakyReLU(),
            nn.Linear(self.num_layers, self.n_dim)
        )


    