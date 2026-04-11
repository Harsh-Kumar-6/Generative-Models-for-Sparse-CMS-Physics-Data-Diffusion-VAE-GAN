import math
import torch
import torch.nn as nn


class SimpleVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 16, 4, 2, 1),   # 128 → 64
            nn.ReLU(),

            nn.Conv2d(16, 32, 4, 2, 1),  # 64 → 32
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),  # 32 → 16
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64*16*16, latent_dim)
        self.fc_logvar = nn.Linear(64*16*16, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 64*16*16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 → 32
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 32 → 64
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, 4, 2, 1),   # 64 → 128
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 64, 16, 16)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x)

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + kl_loss

class VAELossWrapper:
    def __init__(self, beta=10.0):
        self.beta = beta

    def __call__(self, pred, x):
        recon, mu, logvar = pred

        recon_loss = F.mse_loss(recon, x)

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return recon_loss + self.beta * kl_loss
    
class BetaSchedulerCB(Callback):
    def __init__(self, start_beta=0.0, target_beta=0.05, anneal_epochs=20):
        fc.store_attr()
        
    def before_fit(self, learn):
        learn.loss_func.beta = self.start_beta
        
    def before_epoch(self, learn):
        if learn.epoch < self.anneal_epochs:
            progress = learn.epoch / self.anneal_epochs
            
            # Smooth curve instead of linear
            new_beta = self.target_beta * (1 - math.exp(-5 * progress))
            
            learn.loss_func.beta = new_beta
        else:
            learn.loss_func.beta = self.target_beta
            
        print(f"--- Epoch {learn.epoch}: Beta = {learn.loss_func.beta:.5f} ---")
