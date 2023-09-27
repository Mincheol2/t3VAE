import torch
import math

from models import baseline

class DisentanglementVAE(baseline.VAE_Baseline):
    def __init__(self, image_shape, DEVICE, args):
        super(DisentanglementVAE, self).__init__(image_shape, DEVICE,args)
           
    def encoder(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim = 1)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        z = self.reparameterize(mu, logvar)
        return [z, mu, logvar]
    
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
        return [x_recon, z, mu, logvar]
        
    @torch.enable_grad()
    def loss(self, x, recon_x, z, mu, logvar):
        N = x.shape[0]
        
        '''
            Computes E_{p(x)}[ELBO_{\alpha,\beta}]
            (According to the original paper, alpha = 0 in this experiment.)
        '''

        qz_x = torch.distributions.normal.Normal(mu,torch.exp(0.5 * logvar))


        '''
            KL regularizer from Gaussian to Student-t is not a closed form.
            Instead, estimate - E_{q(z|x)}[log q(z|x)] and E_{q(z|x)}[log p(x)] separately.
        '''
        # pixelwise one-dim t-dist
        pz = torch.distributions.studentT.StudentT(df=self.args.nu,loc=torch.zeros(self.args.m_dim),scale=torch.ones(self.args.m_dim)) 
        zs = self.reparameterize(mu, logvar) # m_dim samples from repara trick
        ent = - qz_x.log_prob(zs) 
        reg_loss = (ent + pz.log_prob(zs)).mean(0)

        '''
            Why we multiply 2 on the reg_loss?
            
            If we look at the gamma-bound formula: 1/2 * (||x - recon_x ||^2 / recon_sigma**2 + 2 * regularizer), 
            we can see that we omit the constant 1/2 when calculating the total_loss.
            For comparison, we also multlply 2 with other frameworks (except t3VAE)
        '''
        reg_loss = 2 * reg_loss
        recon_loss = torch.sum((recon_x - x)**2 / (N * self.args.prior_sigma**2))
        total_loss = reg_loss + recon_loss
        return [reg_loss, recon_loss, total_loss]

    def generate(self):
        prior_z = torch.randn(64, self.args.m_dim)
        prior_z = self.args.prior_sigma * prior_z
        VAE_gen = self.decoder(prior_z.to(self.DEVICE)).detach().cpu()
        VAE_gen = VAE_gen
        return VAE_gen