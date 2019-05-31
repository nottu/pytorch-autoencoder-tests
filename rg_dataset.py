from pandas import read_fwf, DataFrame
import math
from radioreader import *
from methods import *
from kittler import kittler_float

from torch.utils import data
from torchvision import transforms

import PIL.Image as Image
from PIL import ImageFilter

class LRG(data.Dataset):
    def __init__(self, sz=64, use_kittler=False, n_aug=10, twice=False, catalog_dir='catalog/mrt-table3.txt', file_dir='lrg', file_ext='fits', blur=False):
        self.blur = blur
        self.catalog_dir = catalog_dir
        self.file_dir = file_dir
        self.file_ext = file_ext
        self.use_kittler = use_kittler
        self.twice = twice
        self.n_aug = n_aug
        self.__load_data()
        self.data_len = len(self.data)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.CenterCrop(80),
            transforms.Resize(sz),
            transforms.ToTensor()])

    def __load_data(self):
        lrg = read_fwf(self.catalog_dir, skiprows=41, header=None)
        labeled = DataFrame({'Name':lrg[0], 'Label':lrg[7]})
        names = labeled['Name'].tolist()
        labels = labeled['Label'].tolist()
        images = []
        directory, ext = self.file_dir, self.file_ext

        for i in range(len(names)):
          f_name = '{0}/{1}.{2}'.format(directory, names[i].replace('.','_'), ext)
          im = readImg(f_name, normalize=True, sz=128)
          if self.use_kittler : im = kittler_float(im, copy=False)
          images.append(im.T)
          sys.stdout.write('LRG:\t{}/{}\r'.format(i + 1, len(names)))
          sys.stdout.flush()
        print('')
        images = np.array(images)
        labels = [ 1 if (l == '1' or l == '1F') else int(l) for l in labels]

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
        if self.twice:
            return self.transform(img), self.transform(img), y
        return self.transform(img), y
    def __len__(self):
        return self.data_len * self.n_aug

class UNLRG_C(data.Dataset):
    def __init__(self, sz=64, use_kittler=False, n_aug=10, catalog_dir='catalog/mrt-table4.txt', file_dir='unlrg', file_ext='fits'):
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
            transforms.CenterCrop(80),
            transforms.Resize(sz),
            transforms.ToTensor()])

    def __load_data(self):
        lrg = read_fwf(self.catalog_dir, skiprows=41, header=None)
        labeled = DataFrame({'Name':lrg[0], 'Label':lrg[8], 'Cert':lrg[11]})
        labeled[labeled['Cert']==1]
        names = labeled['Name'].tolist()
        labels_v = labeled['Label'].tolist()
        images = []
        directory, ext = self.file_dir, self.file_ext

        labels = []
        for i in range(len(names)):
          l = labels_v[i]
          if l == 'F' : l = 1
          l = float(l)
          if(math.isnan( l )): continue
          labels.append(int(l))
          f_name = '{0}/{1}.{2}'.format(directory, names[i].replace('.','_'), ext)
          im = readImg(f_name, normalize=True, sz=128)
          if self.use_kittler : im = kittler_float(im, copy=False)
          images.append(im.T)
          sys.stdout.write('LRG:\t{}/{}\r'.format(i + 1, len(names)))
          sys.stdout.flush()
        print('')
        images = np.array(images)
        # labels = [ 1 if (l == '1' or l == 'F') else int(l) for l in labels]

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

        #apply the transformations and return tensors
        return self.transform(img), y
    def __len__(self):
        return self.data_len * self.n_aug


class UNLRG(data.Dataset):
    def __init__(self, sz=64, use_kittler=False, n_aug=10, catalog_dir='catalog/mrt-table4.txt', file_dir='unlrg', file_ext='fits'):
        self.catalog_dir = catalog_dir
        self.file_dir = file_dir
        self.file_ext = file_ext
        self.use_kittler = use_kittler
        self.n_aug = n_aug
        self.__load_data()
        self.data_len = len(self.data)
        self.sz = sz
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.CenterCrop(80),
            transforms.Resize(self.sz),
            transforms.ToTensor()])

    def __load_data(self):
        directory, ext = self.file_dir, self.file_ext
        names = glob.glob('{0}/*.{1}*'.format(directory, ext))
        images = []
        for n in range(len(names)):
            im = readImg(names[n], normalize=True, sz=128)
            if self.use_kittler : im = kittler_float(im, copy=False)
            images.append( np.expand_dims(im, axis=0) )
            sys.stdout.write('UNLRG:\t{}/{}\r'.format(n + 1, len(names)))
            sys.stdout.flush()
        print('')
        self.data = np.array(images)
            
    def __getitem__(self, index):
        index = index % self.data_len
        np_arr = self.data[index, :]
        ## reshape np_arr to 28x28
        np_arr = np_arr.reshape(128, 128)

        ## convert to PIL-image
        img = Image.fromarray((np_arr*255).astype('uint8'))

        #apply the transformations and return tensors
        return self.transform(img)

    def __len__(self):
        return self.data_len * self.n_aug

    def __repr__(self) -> str:
        return 'unLRG dataset'