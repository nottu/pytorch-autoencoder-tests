import numpy as np
from astropy.io import fits
import glob, sys
from skimage.transform import rescale as sk_rescale

__first_size = 256
__nvss_size = 64

def __min_max_img(img):
  min_b, max_b = 10, -100
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if(img[i,j] < min_b): min_b = img[i,j]
      if(img[i,j] > max_b): max_b = img[i,j]
  return (min_b, max_b)

def __normalize_img(img, as_int=False, max_val=255):
  min_b, max_b = __min_max_img(img)
  range_b = max_b - min_b
  n_img = np.zeros_like(img, dtype=float)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      n_img[i, j] = (img[i, j] - min_b)/ range_b
  if(as_int): n_img = sk.img_as_int(n_img * max_val)
  return n_img

def readImg(img, normalize=False, sz=False):
  f = fits.open(img)[0].data
  while (len(f.shape) > 2): f = f[0]

  if normalize : f = __normalize_img(f) if normalize else f
  if sz : f = sk_rescale(f, (sz/f.shape[0], sz/f.shape[1]), mode='reflect', multichannel=None, anti_aliasing=False)

  return np.nan_to_num(f, copy=False)[::-1]

def __transpose_mtx(mtx):
    return [[mtx[i][j] for i in range(len(mtx))] for j in range(len(mtx[0]))]
def __get_file_list(directory, ext):
  return sorted([f for f in sorted(glob.glob('{0}/*.{1}*'.format(directory, ext)))])

def readImagesFromDirs(dirs, normalize=False, rescale=False):
  files = __transpose_mtx([__get_file_list(dirs[i], 'fit') for i in range(len(dirs))])
  images = []
  names = []
  types = (True, False) if files[0][0].find('first') else (False, True)
  for im in files:
      try:
          images.append([readImg(im[0], normalize, rescale, types[0]),
                         readImg(im[1], normalize, rescale, types[1])])
          names.append(im)
      except:
          print('error reading image {0} or {1}...'.format(im[0], im[1], sys.exc_info()[0]))
  files = names
  return files, np.array(images)