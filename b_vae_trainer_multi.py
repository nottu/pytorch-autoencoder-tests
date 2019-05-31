import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data
from torch.optim import Adam

from torchvision import transforms
from torchvision import datasets

import numpy as np

from pandas import read_fwf, DataFrame
from tqdm   import tqdm_notebook as tqdm


from rg_dataset import LRG, UNLRG, UNLRG_C

lrg_data_set = LRG(use_kittler=True, blur=True)
unlrg_data_set = UNLRG_C(use_kittler=True)

data_loader_lrg   = data.DataLoader(lrg_data_set, batch_size=64, shuffle=False)
data_loader_unlrg = data.DataLoader(unlrg_data_set, batch_size=64, shuffle=False)

class VAE(nn.Module):
    def __init__(self, lt_dim=4):
        super(VAE, self).__init__()
        # self.k = [1, 16, 32, 64, 128, 256]
        self.k = [1, 64, 128, 128, 258, 256]
        encoder_layers = []
        decoder_layers = []
        
        for i in range(len(self.k) - 1):
            layer = nn.Conv2d(self.k[i], self.k[i+1], 3, 2, 1, 1)
            encoder_layers.append(layer)
            encoder_layers.append(nn.ReLU())
        
        for i in range(len(self.k) - 1, 0, -1):
            layer = nn.ConvTranspose2d(self.k[i], self.k[i-1], 3, 2, 1, 1)
            decoder_layers.append(layer)
            decoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers[:-1])
        
        self.fc_mu = nn.Linear(self.k[-1]*2*2, lt_dim)
        self.fc_ep = nn.Linear(self.k[-1]*2*2, lt_dim)
        
        self.fc_dc = nn.Linear(lt_dim, self.k[-1]*2*2)
    def encode(self, x):
        encoded = self.encoder(x).view(-1, self.k[-1]*2*2)
        return self.fc_mu(encoded), self.fc_ep(encoded)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x):
        x = F.relu(self.fc_dc(x)).view(-1, self.k[-1], 2, 2)
        return torch.sigmoid(self.decoder(x))
    
    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        d = self.decode(z)
        return d, mu, var

class B_VAE_Loss:
    def __init__(self, gamma, max_capacity, epochs):
        self.gamma = gamma
        self.recon_ls = nn.BCELoss(reduction='sum')
        self.capacity = 0
        self.delta = max_capacity / float(epochs)
        self.max_capacity = max_capacity
    def update(self):
        self.capacity = min(self.max_capacity, self.capacity + self.delta)
        return self.capacity
    def __call__(self, res, img):
        batch_sz = len(img)
        x, mu, logvar = res
        recon = self.recon_ls(x, img).div(batch_sz) #res -> x, mu, var
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).div(batch_sz)
        return self.capacity, recon, self.gamma * (kld - self.capacity).abs()


def train_step(model, device, data_loader, optim, epoch, loss_fun, log_interval=5):
    model.train()
    s = ''
    r_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = Variable(target, requires_grad=False).to(device)
        #Forward Pass
        optim.zero_grad()
        output = model(data)
        
        #############################################
        ######### check against known class #########
        #############################################
        #########    Compact vs Extended    #########
        extended = target > 1
        extended = Variable(extended.float().to(device), requires_grad=False)
        pred_ext = torch.sigmoid(output[1][:,0])
        ext_loss = F.binary_cross_entropy(pred_ext, extended, reduction='sum').div(len(pred_ext))
        #########       FRI vs FRII         #########
        o = torch.sigmoid(output[1][:, 1])[target > 1]
        c = target[target > 1]
        o = o[c < 4]
        c = c[c < 4]
        c = Variable( (c == 3).float().to(device), requires_grad=False)
        fr_loss = F.binary_cross_entropy(o, c, reduction='sum').div(len(c))
        #########   Regular vs Irregular   #########
        # BCE Loss
        c, r_loss , g_loss = loss_fun(output, data)
        loss = r_loss + g_loss + 10 * (ext_loss + fr_loss)# + reg_loss)
        #Backpropagation
        loss.backward()
        optim.step()
        s = 'Train Epoch: {:3d} [{:5d}/{:5d} ({:3.0f}%)]   Loss: {:4.4f}   R_Loss: {:4.4f}   Capacity: {:4.2f}'
        s = s.format(epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item(), r_loss.item(), c)
        if batch_idx % log_interval == 0:
            sys.stdout.write('{}\r'.format(s))
            sys.stdout.flush()
    return s, r_loss

def test_step(model, device, data_loader, loss_fun):
    model.train()
    r_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            data = data.to(device)
            #Forward Pass
            output = model(data)

            # BCE Loss
            c, r_loss , g_loss = loss_fun(output, data)
    return r_loss


def do_training(lt_dim):
  device = 'cuda'
  vae = VAE(lt_dim=lt_dim).to(device)
  optimizer = Adam(vae.parameters(), lr=0.001, weight_decay=1E-5)
  epochs = 100
  gam = 10
  cap = 10
  beta_vae_loss = B_VAE_Loss(gamma=gam, max_capacity=cap, epochs=epochs)
  train_loss = []
  test_loss  = []
  for epoch in range(1, epochs+1):
      #LRG, forced params
      start = time.time()
      s, l = train_step(vae, 'cuda', data_loader_unlrg, optimizer, epoch, loss_fun=beta_vae_loss)
      loss = test_step(vae, 'cuda', data_loader_lrg, loss_fun=beta_vae_loss)
      train_loss.append(l)
      test_loss.append(loss)
      t = time.time() - start
      sys.stdout.write('{}   Test Loss : {:4.4f}   Time : {:.2f}s\n'.format(s, loss, t))
      beta_vae_loss.update()
  torch.save(vae, 'b_vae_norot_unlrg_g{}_ld{}_epochs{}_cap{}'.format(gam, lt_dim, epochs, cap))
  np.save('train_loss_lt_{}'.format(lt_dim), train_loss)
  np.save('test_loss_lt_{}'.format(lt_dim), test_loss)


for i in range(2, 11):
  do_training(i)