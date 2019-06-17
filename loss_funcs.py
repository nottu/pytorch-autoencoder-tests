import torch
from torch import nn

class VAE_Loss:
    def __init__(self, recon_ls = nn.BCELoss(reduction='sum')):
        self.recon_ls = recon_ls
    def kld(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    def __call__(self, res, img):
        batch_sz = len(img)
        x, mu, logvar = res
        recon = self.recon_ls(x, img).div(batch_sz) #res -> x, mu, var
        return recon, self.kld(mu, logvar).div(batch_sz)

class B_VAE_Loss(VAE_Loss):
    def __init__(self, gamma, recon_ls = nn.BCELoss(reduction='sum')):
        super(B_VAE_Loss, self).__init__(recon_ls)
        self.gamma = gamma
    def __call__(self, res, img):
        recon, kld = super(B_VAE_Loss, self).__call__(res, img)
        return recon, self.gamma * kld

class B_VAE_Loss_decay(B_VAE_Loss):
    def __init__(self, gamma, recon_ls = nn.BCELoss(reduction='sum'), decay=0):
        super(B_VAE_Loss_decay, self).__init__(gamma, recon_ls)
        self.decay = decay
    def update(self):
        self.gamma *= (1 - self.decay)

class B_VAE_Loss_cap(B_VAE_Loss):
    def __init__(self, gamma, max_capacity, epochs, recon_ls = nn.BCELoss(reduction='sum')):
        super(B_VAE_Loss_cap, self).__init__(gamma, recon_ls)
        self.capacity = 0
        self.delta = max_capacity / float(epochs)
        self.max_capacity = max_capacity
    def update(self):
        self.capacity = min(self.max_capacity, self.capacity + self.delta)
        return self.capacity
    def __call__(self, res, img):
        recon, kld = super(B_VAE_Loss_cap, self).__call__(res, img)
        return self.capacity, recon, (kld - self.gamma * self.capacity).abs()

