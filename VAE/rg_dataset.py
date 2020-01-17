from pandas import read_fwf, DataFrame
import math

from utils.radioreader import *
from utils.methods import *
from utils.kittler import kittler_float

from torch.utils import data
from torchvision import transforms

import PIL.Image as Image
from PIL import ImageFilter

from skimage import measure



class LRG(data.Dataset):
    def __init__(self, sz=64, rd_sz=128, use_kittler=False, n_aug=10, catalog_dir='catalog/mrt-table3.txt', file_dir='lrg', file_ext='fits', blur=False, remove_noisy=True):
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
            # transforms.CenterCrop(80),
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
        ## reshape np_arr to 28x28
        np_arr = np_arr.reshape(128, 128)

        ## convert to PIL-image
        img = Image.fromarray((np_arr*255).astype('uint8'))
        if self.blur : img = img.filter(ImageFilter.BLUR)
        #apply the transformations and return tensors
        return self.transform(img), y
    def __len__(self):
        return self.data_len * self.n_aug

class BasicDataset(data.Dataset):
  def __init__(self, images, labels, sz=64, n_aug=10, rotation=10):
    self.sz = sz
    self.n_aug = n_aug
    self.data = images
    self.labels = labels
    self.data_len = len(self.data)
    self.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(rotation),
        # transforms.CenterCrop(80),
        transforms.Resize(self.sz),
        transforms.ToTensor()])

  def get_data(self):
    return self.data, self.labels

  def __getitem__(self, index):
      index = index % self.data_len
      np_arr = self.data[index, :]
      y = self.labels[index]
      ## reshape np_arr to 28x28
      # print(self.data.shape)
      np_arr = np_arr.reshape(self.data.shape[1], self.data.shape[2])

      ## convert to PIL-image
      img = Image.fromarray((np_arr*255).astype('uint8'))

      return self.transform(img), y

  def __len__(self):
      return self.data_len * self.n_aug