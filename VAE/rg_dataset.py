from pandas import read_fwf, DataFrame
import math

from utils.radioreader import *
from utils.methods import *
from utils.kittler import kittler_float

from torch import utils
from torchvision import transforms

import PIL.Image as Image
from PIL import ImageFilter

from skimage import measure
import numpy as np
from sklearn.model_selection import train_test_split

class LRG(utils.data.Dataset):
    def __init__(self, sz=64, rd_sz=128, use_kittler=False, n_aug=10, crop_factor=0.8, catalog_dir='catalog/mrt-table3.txt', file_dir='lrg', file_ext='fits', blur=False, remove_noisy=True):
        self.blur = blur
        self.rd_sz = rd_sz
        self.remove_noisy = remove_noisy
        self.catalog_dir = catalog_dir
        self.file_dir = file_dir
        self.file_ext = file_ext
        self.use_kittler = use_kittler
        self.n_aug = n_aug
        self.__load_data()
        self.data_len = len(self.data)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.CenterCrop(int( rd_sz * crop_factor)),
            transforms.Resize(sz),
            transforms.ToTensor()])

    def __load_data(self):
        lrg = read_fwf(self.catalog_dir, skiprows=41, header=None)
        l_idx = 7 if(self.file_dir.endswith('/lrg')) else 6
        labeled = DataFrame({'Name':lrg[0], 'Label':lrg[l_idx]})
        names = labeled['Name'].tolist()
        labels_v = labeled['Label'].tolist()
        images = []
        directory, ext = self.file_dir, self.file_ext
        # print(l_idx)
        labels = []
        for i in range(len(names)):
          l = labels_v[i]
          if l == 'F' or l == '1F' : l = 1

          try:
            l = float(l)
          except Exception as the_exception:
            continue
          if(math.isnan( l )): continue

          f_name = '{0}/{1}.{2}'.format(directory, names[i].replace('.','_'), ext)
          im = readImg(f_name, normalize=True, sz=self.rd_sz)
          if self.use_kittler :
            
            im = closing(im, disk(2))
            im = opening(im, disk(2))
            im = kittler_float(im, copy=False)
            im = opening(im, disk(2))

            if(self.remove_noisy):
              if(np.median(im) > 0.0): continue
              ls = measure.label(im > 0)
              ngroups = len(np.bincount(ls.flat))
              if ngroups > 10 : continue

          images.append(im.T)
          labels.append(int(l) - 1) #since label starts at 1, '-1'

          sys.stdout.write('{}:\t{}/{}\r'.format(self.file_dir, i + 1, len(names)))
          sys.stdout.flush()
        print('')
        images = np.array(images)

        self.data = images
        self.labels = labels

    def get_data(self):
      return self.data, self.labels

    def __getitem__(self, index):
        index = index % self.data_len
        np_arr = self.data[index, :]
        y = self.labels[index]

        np_arr = np_arr.reshape(self.rd_sz, self.rd_sz)
        ## convert to PIL-image
        img = Image.fromarray((np_arr*255).astype('uint8'))
        if self.blur : img = img.filter(ImageFilter.BLUR)
        #apply the transformations and return tensors
        return self.transform(img), y
    def __len__(self):
        return self.data_len * self.n_aug


def get_datasets(sz=64, rd_sz=128):
  data_path = '../data/'
  lrg_data_set   = LRG(sz=sz, rd_sz=rd_sz,use_kittler=True, n_aug=1, blur=True, catalog_dir=data_path + 'catalog/mrt-table3.txt', file_dir=data_path + 'lrg')

  x, y = lrg_data_set.get_data()
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

  def get_balanced_set(data, labels):

    l = np.array(labels)
    sort_order = np.argsort(l)

    #sort_stuff!
    l = l[sort_order]
    data = data[sort_order]

    compact  = l == 0
    extended = l > 0

    labels_compact = (1 * (l > 0)).tolist()

    return BalancedDataSet(data, labels_compact, sz=sz)

  return {'train':get_balanced_set(X_train, y_train), 'test': get_balanced_set(X_test, y_test), 'full': lrg_data_set}

# data_loader_lrg   = utils.data.DataLoader(compact_v_extended_set,   batch_size=128, shuffle=False)
# for i, (data, target) in enumerate(data_loader_lrg):
class BalancedDataSet(utils.data.Dataset):
  ''' 
    Made for just two classes that need to be balanced i.e. have a similar number of items
    Params similar to LRG
    images : loaded images, should be sorted by label
    labels : labels sorted
    n_aug : base/minimum number of augmentations
  '''
  def __init__(self, images, labels, sz=64, n_aug=10, rotation=180, crop_factor=0.8):
    super(BalancedDataSet, self).__init__()
    self.sz = sz
    self.n_aug = n_aug
    self.data = images
    self.labels = labels
    self.data_len = len(self.data)
    self.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(rotation),
        transforms.CenterCrop(self.data.shape[1] * crop_factor),
        transforms.Resize(self.sz),
        transforms.ToTensor()])

    self.num_pos = np.sum(labels)
    self.num_neg = len(labels) - self.num_pos
    neg_aug = (self.num_pos * n_aug) // self.num_neg

    self.num_pos_aug = self.num_pos * n_aug 
    self.num_neg_aug = neg_aug * self.num_neg


  def get_data(self):
    return self.data, self.labels

  def __getitem__(self, index):
    if index < self.num_neg_aug:
      index = index % self.num_neg
      if self.labels[index] :
        print('Warning: Selecting negative when it should be positive')
    else:
      index = index - self.num_neg
      index = index % self.num_pos
      index = index + self.num_neg
      if self.labels[index] == 0:
        print('Warning: Selecting positive when it should be negative')

    np_arr = self.data[index, :]
    y = self.labels[index]

    np_arr = np_arr.reshape(self.data.shape[1], self.data.shape[2])

    ## convert to PIL-image
    img = Image.fromarray((np_arr*255).astype('uint8'))

    return self.transform(img), y

  def __len__(self):
      return self.num_pos_aug + self.num_neg_aug

class BasicDataset(utils.data.Dataset):
  def __init__(self, images, labels, sz=64, n_aug=10, rotation=180, crop_factor=0.8):
    self.sz = sz
    self.n_aug = n_aug
    self.data = images
    self.labels = labels
    self.data_len = len(self.data)
    self.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(rotation),
        transforms.CenterCrop(self.data.shape[1] * crop_factor),
        transforms.Resize(self.sz),
        transforms.ToTensor()])

  def get_data(self):
    return self.data, self.labels

  def __getitem__(self, index):
      index = index % self.data_len
      np_arr = self.data[index, :]
      y = self.labels[index]

      np_arr = np_arr.reshape(self.data.shape[1], self.data.shape[2])

      ## convert to PIL-image
      img = Image.fromarray((np_arr*255).astype('uint8'))

      return self.transform(img), y

  def __len__(self):
      return self.data_len * self.n_aug