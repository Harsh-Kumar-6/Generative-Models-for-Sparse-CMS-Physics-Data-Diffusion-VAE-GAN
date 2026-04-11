import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.main = nn.Sequential(
            # Input: Latent Vector -> 8x8 Feature Map
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.Unflatten(1, (256, 8, 8)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample blocks: 8->16, 16->32, 32->64, 64->128
            self._block(256, 128),
            self._block(128, 64),
            self._block(64, 32),
            self._block(32, 16),

            # Final layer to 8 Channels (Detector Layers)
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.Tanh() # Output range [-1, 1]
        )

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, z): return self.main(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Standard CNN for classification
            nn.Conv2d(8, 64, 4, 2, 1), # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 64 -> 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), # 32 -> 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1) # No Sigmoid! We use BCEWithLogitsLoss
        )
    def forward(self, x): return self.main(x)


class GAN(nn.Module):
    def __init__(self, gen, disc):
        super().__init__()
        self.gen, self.disc = gen, disc

    def forward(self, x):
        return torch.tensor(0.0, requires_grad=True).to(x.device)

class GANLoss(nn.Module):
    def __init__(self, disc):
        super().__init__()
        self.disc = disc
        self.loss_func = nn.BCELoss()
        
    def __call__(self, pred, target):
        # Must be on the same device as 'target' to avoid Device Mismatch errors
        return torch.tensor(0.0, requires_grad=True).to(target.device)
    
class GANTrainer(Callback):
    def __init__(self, latent_dim=128): 
        self.latent_dim = latent_dim
    
    def before_fit(self):
        # Generator learns faster (4e-4) than Discriminator (1e-4)
        self.opt_g = torch.optim.Adam(self.learn.model.gen.parameters(), lr=4e-4, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.learn.model.disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.crit = nn.BCEWithLogitsLoss()

    def before_batch(self):
        x = self.learn.batch[0]
        device = x.device
    
        self.opt_d.zero_grad()
        
        real_logits = self.learn.model.disc(x)
        loss_real = self.crit(real_logits, torch.ones_like(real_logits) * 0.9)
        
        z = torch.randn(x.size(0), self.latent_dim).to(device)
        fake = self.learn.model.gen(z)
        fake_logits = self.learn.model.disc(fake.detach())
        loss_fake = self.crit(fake_logits, torch.zeros_like(fake_logits))
        
        d_loss = (loss_real + loss_fake) / 2
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.learn.model.disc.parameters(), 1.0)
        self.opt_d.step()

        self.opt_g.zero_grad()
        gen_logits = self.learn.model.disc(fake)
        g_loss = self.crit(gen_logits, torch.ones_like(gen_logits))
        
        sparsity = fake.abs().mean() * 0.001
        total_g_loss = g_loss + sparsity
        
        total_g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.learn.model.gen.parameters(), 1.0)
        self.opt_g.step()

        self.learn.loss = (d_loss + g_loss).detach()
        self.learn.xb = x
        self.learn.preds = fake.detach()
        raise CancelBatchException()
