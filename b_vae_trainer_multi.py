import sys
import time
import numpy as np
import torch

from torch.utils import data
from torch.optim import Adam

from VAE.rg_dataset import LRG, BasicDataset
from VAE.vae_models import VAE, RotVAE
from VAE.loss_funcs import B_VAE_Loss_cap, B_VAE_Loss_decay
from VAE.vae_trainers import train_step_b_vae_cap, train_step_b_vae, test_step_b_vae_cap, test_step_b_vae


def do_training(lt_dim, data_loader_lrg, data_loader_unlrg, model_prefix, beta_vae_loss, epochs=50,
                learn_rot=False, model_dir='trained_models', loss_data_dir='model_loss_data'):
  device = 'cuda'
  vae = VAE(lt_dim=lt_dim).to(device) if not learn_rot else RotVAE(lt_dim=lt_dim).to(device)
  optimizer = Adam(vae.parameters(), lr=0.001, weight_decay=1E-5)
  train_loss = []
  test_loss  = []
  for epoch in range(1, epochs+1):
      #LRG, forced params
      start = time.time()
      s, l = train_step_b_vae_cap(vae, device, data_loader_unlrg, optimizer, epoch, loss_fun=beta_vae_loss, learn_rot=learn_rot)
      loss = test_step_b_vae_cap(vae, device, data_loader_lrg, loss_fun=beta_vae_loss, learn_rot=learn_rot)
      train_loss.append(l)
      test_loss.append(loss)
      t = time.time() - start
      sys.stdout.write('{}   Test Loss : {:4.4f}   Time : {:.2f}s\n'.format(s, loss, t))
      beta_vae_loss.update()
  # b_vae_proc_unlrg
  torch.save(vae, '{}/{}_ld{}_epochs{}'.format(model_dir, model_prefix, lt_dim, epochs))
  np.save('{}/train_loss_{}_lt_{}'.format(loss_data_dir, model_prefix, lt_dim), train_loss)
  np.save('{}/test_loss_{}_lt_{}'.format( loss_data_dir, model_prefix, lt_dim), test_loss)



if __name__ == '__main__':
  
  # lrg_images   = np.load(  'data/proc_aligned/lrg_norm_proc.npy', allow_pickle=True)
  # unlrg_images = np.load('data/proc_aligned/unlrg_norm_proc.npy', allow_pickle=True)

  lrg_data_set   = LRG(use_kittler=True, n_aug=10, blur=True, catalog_dir='data/catalog/mrt-table3.txt', file_dir='data/lrg')
  unlrg_data_set = LRG(use_kittler=True, n_aug=10, blur=True, catalog_dir='data/catalog/mrt-table4.txt', file_dir='data/unlrg')

  print("{} UNLRG images and {} LRG images".format(len(unlrg_data_set.get_data()[0]), len(lrg_data_set.get_data()[0]) ))
  # my_lrg_dataset  = BasicDataset(lrg_images, lrg_data_set.labels, n_aug=10) #
  data_loader_lrg = data.DataLoader(lrg_data_set, batch_size=128, shuffle=False)

  # my_unlrg_dataset  = BasicDataset(unlrg_images, unlrg_data_set.labels, n_aug=10)
  data_loader_unlrg = data.DataLoader(unlrg_data_set, batch_size=128, shuffle=False)

  for i in range(7, 8):
    net_type = 'capacity'
    gam = 12
    cap = 10
    decay  = 0.015
    epochs = 50

    if net_type == 'decay' :
      beta_vae_loss = B_VAE_Loss_decay(gamma=gam, decay=decay)
      prefix = 'bvae_decay_g_{}_decay_{}'.format(gam, decay)
    elif net_type == 'capacity' :
      beta_vae_loss = B_VAE_Loss_cap(gamma=gam, max_capacity=cap, epochs=epochs)
      prefix = 'bvae_full_cap_{}_g_{}'.format(cap, gam)
    else:
      beta_vae_loss = VAE_Loss(gamma=gam)
      prefix = 'bvae_g_{}'.format(gam)

    do_training(i, data_loader_lrg, data_loader_unlrg, beta_vae_loss=beta_vae_loss, 
                model_prefix=prefix, epochs=epochs, learn_rot=False)