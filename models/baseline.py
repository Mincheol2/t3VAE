import torch.nn as nn
import math

class VAE_Baseline(nn.Module):
    def __init__(self, img_shape, DEVICE, args):
        super().__init__()
        self.args = args
        self.img_shape = img_shape
        self.DEVICE = DEVICE
        self.B, self.C, self.H, self.W = self.img_shape
        self.opt = None
        self.scheduler = None
        ### Encoder layers ##
        
        encoder_hiddens = [128, 256, 512,1024, 2048]
        layers = []
        input_ch = self.C
        for dim in encoder_hiddens:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(input_ch, dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU())
            )
            input_ch = dim

        self.cnn_layers = nn.Sequential(*layers)
        
        n = len(encoder_hiddens)
        self.cnn_lineardim = encoder_hiddens[-1]* math.ceil(self.H / 2**n) * math.ceil(self.W / 2**n)

        self.mu_layer = nn.Linear(self.cnn_lineardim , args.m_dim)
        self.logvar_layer = nn.Linear(self.cnn_lineardim , args.m_dim)
        
        ## Decoder layers ##
        self.decoder_hiddens = encoder_hiddens[::-1]
        self.n = len(self.decoder_hiddens)
        self.linear = nn.Sequential(
                        nn.Linear(args.m_dim, self.cnn_lineardim),
                        nn.ReLU(),
        )
        layers = []
        input_ch = self.decoder_hiddens[0]
        for dim in self.decoder_hiddens[1:]:
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(input_ch, dim,
                    kernel_size = 3, stride = 2, padding=1, output_padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU())
            )
            input_ch = dim

        self.tp_cnn_layers = nn.Sequential(*layers)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.decoder_hiddens[-1], self.decoder_hiddens[-1],
                                                            kernel_size=3, stride=2, padding=1, output_padding=1),
                            nn.BatchNorm2d(self.decoder_hiddens[-1]),
                            nn.ReLU(),
                            nn.ConvTranspose2d(self.decoder_hiddens[-1], self.C,
                                               kernel_size=3, padding=1),
                            nn.Tanh()
                            )

    