import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, lt_dim=4, k=[1, 64, 128, 128, 256, 256]):
        super(VAE, self).__init__()
        # self.k = [1, 16, 32, 64, 128, 256]
        self.k = k
        encoder_layers = []
        decoder_layers = []
        
        for i in range(len(self.k) - 1):
            layer = nn.Conv2d(self.k[i], self.k[i + 1], 3, 2, 1, 1)
            encoder_layers.append(layer)
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm2d(self.k[i + 1]))
        
        for i in range(len(self.k) - 1, 0, -1):
            layer = nn.ConvTranspose2d(self.k[i], self.k[i - 1], 3, 2, 1, 1)
            decoder_layers.append(layer)
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm2d(self.k[i - 1]))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers[:-2])
        
        self.fc_mu = nn.Sequential(
                      nn.Linear(self.k[-1]*2*2, lt_dim*2),
                      nn.Linear(lt_dim*2, lt_dim)
        )
        self.fc_ep = nn.Sequential(
                      nn.Linear(self.k[-1]*2*2, lt_dim*2),
                      nn.Linear(lt_dim*2, lt_dim)
        )
        
        self.fc_dc = nn.Linear(lt_dim, self.k[-1]*2*2)
    def encode(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(-1, self.k[-1]*2*2)
        return self.fc_mu(encoded), self.fc_ep(encoded)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x):
        x = F.relu(self.fc_dc(x))
        x = x.view(-1, self.k[-1], 2, 2) #reshape
        return torch.sigmoid(self.decoder(x))
    
    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        d = self.decode(z)
        return d, mu, var

class RotVAE(VAE):
    def __init__(self, lt_dim, k):
        super(RotVAE, self).__init__(lt_dim, k)
        self.fc_mu = nn.Sequential(
                      nn.Linear(self.k[-1] * 2 * 2, (lt_dim + 1) * 2),
                      nn.Linear((lt_dim + 1) * 2, lt_dim + 1)
        )
        self.fc_ep = nn.Sequential(
                      nn.Linear(self.k[-1] * 2 * 2, (lt_dim + 1) * 2),
                      nn.Linear((lt_dim + 1) * 2, lt_dim + 1)
        )
    
    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        d = self.decode(z[:,:-1])
        return d, mu, var


class Discriminator(nn.Module):
    def __init__(self, sz=32*4*4):
        super(Discriminator, self).__init__()
        self.sz = sz
        self.encoder = nn.Sequential(
            nn.Conv2d( 1,  8, 3, 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d( 8,  8, 3, 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d( 8, 16, 3, 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1, 1),
            nn.ReLU()
        )
        self.main = nn.Sequential(
            nn.Linear(self.sz, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x).view(-1, self.sz)
        x = self.main(x)
        return x