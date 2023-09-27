# I'm lazy...
# import torch
# import math
# import torch.nn as nn
# from models import baseline

# class VAE(baseline.VAE_Baseline):
#     def __init__(self, image_shape, DEVICE, args):
#         super(VAE, self).__init__(image_shape, DEVICE,args)

#         # encoder parameter nu layers
        
#         self.lognu_layer = nn.Linear(self.cnn_lineardim , args.m_dim)
#         self.mu = 0
#         self.lognu = 0
#         self.logvar = 0
           
#     def encoder(self, x):
#         x = self.cnn_layers(x)
#         x = torch.flatten(x, start_dim = 1)
#         self.mu = self.mu_layer(x)
#         self.logvar = self.logvar_layer(x)
#         self.lognu = self.lognu_layer(x)
#         z = self.reparameterize(self.mu, self.logvar,self.lognu.exp())
#         return [z, self.mu, self.logvar]
    
#     def decoder(self, z):
#         z = self.linear(z)
#         z = z.reshape(-1,self.decoder_hiddens[0],math.ceil(self.H / 2**self.n),math.ceil(self.W / 2**self.n))
#         z = self.tp_cnn_layers(z)
#         z = self.final_layer(z)
#         return z
    
#     def reparameterize(self, mu, logvar, nu):
#         # Student T dist : [B, z_dim]
#         eps = self.MVN_dist.sample(sample_shape=torch.tensor([mu.shape[0]])).to(self.DEVICE)
#         std = torch.exp(0.5 * logvar)
#         std = torch.tensor(nu / self.nu_prime).sqrt() * std
#         v = self.chi_dist.sample(sample_shape=torch.tensor([mu.shape[0]])).to(self.DEVICE)
#         return mu + std * eps / torch.sqrt(v)
    
    
#     def forward(self, x):
#         z, mu, logvar = self.encoder(x)
#         x_recon = self.decoder(z)
#         return [x_recon, z, mu, logvar]
        
#     @torch.enable_grad()
#     def loss(self, x, recon_x, z, mu, logvar):
#         N = x.shape[0]
        
#         '''
#             Why we multiply 2 on the reg_loss?
            
#             If we look at the gamma-bound formula: 1/2 * (||x - recon_x ||^2 / recon_sigma**2 + 2 * regularizer), 
#             we can see that we omit the constant 1/2 when calculating the total_loss.
#             For comparison, we also multlply 2 with other frameworks (except t3VAE)
#         '''
#         encoder_q_zx = self.
#         reg_loss = 2 * self.args.beta_weight * torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1), dim=0)
#         recon_loss = torch.sum((recon_x - x)**2 / (N * self.args.prior_sigma**2))
#         total_loss = reg_loss + recon_loss
#         return [reg_loss, recon_loss, total_loss]

#     def generate(self):
#         prior_z = torch.randn(64, self.args.m_dim)
#         prior_z = self.args.prior_sigma * prior_z
#         VAE_gen = self.decoder(prior_z.to(self.DEVICE)).detach().cpu()
#         VAE_gen = VAE_gen
#         return VAE_gen