import torch
import torch.nn as nn
from diffusers import UNet2DModel

def noisify(x0, ᾱ):
    device = x0.device
    n = len(x0)
    t = torch.randint(0, n_steps, (n,), dtype=torch.long)
    ε = torch.randn(x0.shape, device=device)
    ᾱ_t = ᾱ[t].reshape(-1, 1, 1, 1).to(device)
    xt = ᾱ_t.sqrt()*x0 + (1-ᾱ_t).sqrt()*ε
    return (xt, t.to(device)), ε


@torch.no_grad()
def sample(model, sz, alpha, alphabar, sigma, n_steps):
    device = next(model.parameters()).device
    x_t = torch.randn(sz, device=device)
    preds = []
    for t in reversed(range(n_steps)):
        t_batch = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
        z = (torch.randn(x_t.shape) if t > 0 else torch.zeros(x_t.shape)).to(device)
        ᾱ_t1 = alphabar[t-1]  if t > 0 else torch.tensor(1)
        b̄_t = 1 - alphabar[t]
        b̄_t1 = 1 - ᾱ_t1
        x_0_hat = ((x_t - b̄_t.sqrt() * learn.model((x_t, t_batch)))/alphabar[t].sqrt()).clamp(-1,1)
        x_t = x_0_hat * ᾱ_t1.sqrt()*(1-alpha[t])/b̄_t + x_t * alpha[t].sqrt()*b̄_t1/b̄_t + sigma[t]*z
        preds.append(x_t.cpu())
    return preds

class DDPMCB(Callback):
    order = DeviceCB.order+1
    def __init__(self, n_steps, beta_min, beta_max):
        super().__init__()
        fc.store_attr()
        self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
        self.α = 1. - self.beta
        self.ᾱ = torch.cumprod(self.α, dim=0)
        self.σ = self.beta.sqrt()

    def before_batch(self): self.learn.batch = noisify(self.learn.batch[0], self.ᾱ)
    def sample(self, model, sz): return sample(model, sz, self.α, self.ᾱ, self.σ, self.n_steps)


class UNet(UNet2DModel):
    def forward(self, xt_and_t):
        xt, t = xt_and_t
        # pass timestep as keyword exactly as the API expects
        return super().forward(sample=xt, timestep=t).sample
    

def init_ddpm(model):
    for o in model.down_blocks:
        for p in o.resnets:
            p.conv2.weight.data.zero_()
            for p in fc.L(o.downsamplers): init.orthogonal_(p.conv.weight)

    for o in model.up_blocks:
        for p in o.resnets: p.conv2.weight.data.zero_()

    model.conv_out.weight.data.zero_()


from torch.utils.data import DataLoader, default_collate

def collate_ddpm_h5(batch):
    # batch is a list of (x0, x0) from dataset
    x0 = torch.stack([b[0] for b in batch])
    (xt, t), ε = noisify(x0, alphabar)   # xt: [B,C,H,W], t: [B], ε: [B,C,H,W]
    return (xt, t), ε

def dl_ddpm(ds, bs=128):
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_ddpm_h5)


