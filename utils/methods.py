import numpy as np
import scipy.stats as sps

from skimage.transform import rotate
from skimage.morphology import disk, opening, closing

def get_max_proy(img, theta=False, circle=False, f1=np.sum, f2=np.sum):
    cvs = []
    img = np.array(img, copy=True).T
    if (theta is False): theta = np.linspace(0., 180., 181, endpoint=False)
    if (circle):
        d = 1 - disk( int((img.shape[0] - 1) / 2))
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                img[i,j] = img[i,j] * d[i,j]
    for i in theta:
        im2 = rotate(img, i)
        c = [f1(l) for l in im2]
        cvs.append(c)
    s = np.arange(len(cvs[0]))
    a = [f2(l) for l in cvs]
    idx = a.index(min(a))
    g = cvs[idx] / np.sum(cvs[idx]) # we want g to be left heavy (could be right, doesn't matter)
    l = int(len(g)/2)
    g = g if sum(g[:l]) > sum(g[l:]) else g[::-1]
    cvs = np.array(cvs).T
    return g, np.roll(cvs,idx), idx

def get_all_proy(images, f1=np.max, f2=np.sum):
    gs = []
    for img in images:
        g, c, idx = get_max_proy(img, f1=f1, f2=f2)
        gs.append(g)
    return gs

def get_proy_dst_mtx(data, metric=sps.wasserstein_distance):
    d = np.zeros((len(data), len(data)))
    for i in range(0, len(data) - 1):
        for j in range(i + 1, len(data)):
            d[i, j] = d[j, i] = metric(data[i], data[j])
    return d
'''
    Morphology methods
    Opening
    Closing
'''

def __sphere_elem(rad):
    l      = 2 * rad + 1
    elem   = np.zeros((l , l), dtype='float')
    center = np.array((rad, rad))
    for i in range(l):
        for j in range(l):
            d = np.linalg.norm( center - np.array([i, j]) )
            elem[i, j] = d
    return elem

def __erosion_sphere_m(img, r):
    m_img = np.copy(img)
    elem = __sphere_elem(r)
    a, b = elem.shape
    l, m = m_img.shape
    scale = img.mean()
    elem *= scale
    for i in range(l):
        for j in range(m):
            h = img[i, j] - elem[r, r]
            for x in range(a):
                i_x = i + (x - r) #x - r gives us absolute pos for -r to r
                if (i_x < 0 or i_x >= l): continue #don't allow negative indexes
                for y in range(b):
                    j_y = j + (y - r) #y - r gives us absolute pos for -r to r
                    if(j_y < 0 or j_y >= m): continue #don't allow negative indexes
                    val = img[i_x, j_y]
                    if(val < elem[x, y] + h):
                        h = val - elem[x, y]
            n_val = h + elem[r, r]
            m_img[i, j] = n_val if n_val > 0 else 0
    return m_img

def __dilation_sphere_m(img, r):
    m_img = np.copy(img)
    elem = __sphere_elem(r)
    a, b = elem.shape
    l, m = m_img.shape
    scale = img.mean()
    elem *= scale
    for i in range(l):
        for j in range(m):
            h = img[i, j] - elem[r, r]
            for x in range(a):
                i_x = i + (x - r) #x - r gives us absolute pos for -r to r
                if (i_x < 0 or i_x >= l): continue #don't allow negative indexes
                for y in range(b):
                    j_y = j + (y - r) #y - r gives us absolute pos for -r to r
                    if(j_y < 0 or j_y >= m): continue #don't allow negative indexes
                    val = img[i_x, j_y]
                    if(val > elem[x, y] + h):
                        h = val - elem[x, y]
            n_val = h + elem[r, r]
            m_img[i, j] = n_val if n_val > 0 else 0
    return m_img

def __img_area(img):
    area = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            area += img[i,j]
    return area

def opening_sphere(img, r):
    # return __erosion_sphere_m(__dilation_sphere_m(img, r), r)
    return  opening(img, disk(r))

def closing_sphere(img, r):
    # return __dilation_sphere_m(__erosion_sphere_m(img, r), r)
    return  closing(img, disk(r))

def get_open_close_info(im, sz):
  o_area = __img_area(im)
  a = np.zeros((2 * sz))
  for i in range(sz):
    im_c = opening_sphere(im, i + 1)
    im_a = __img_area(im_c)
    a[i] = (o_area - im_a) / o_area
    # im_c = closing_sphere(im, i + 1)
    # a[i + sz] = (o_area - im_a) / o_area
  return a

'''
    Procrustes Analysis
'''

def __img_distance(im1, im2):
    dif = im1 - im2
    return np.linalg.norm(dif)

def __best_rotate(im1, im2, img_distance=__img_distance):
    min_dist = img_distance(im1, im2)
    angle = 0
    for i in range(1, 360):
        im_r = rotate(im2, i)
        ndist = img_distance(im1, im_r)
        if(ndist < min_dist):
            min_dist = ndist
            angle = i
    return angle

def __best_rotate_flip(im1, im2, img_distance=__img_distance):
    angle = __best_rotate(im1, im2)
    im2_r = rotate(im2, angle)
    min_dist = img_distance(im1, im2_r)
    angle_f = __best_rotate(im1, im2[:,::-1])
    im2_rf = rotate(im2[:,::-1], angle)
    min_dist_f = img_distance(im1, im2_rf)
    if(min_dist_f < min_dist): return (angle_f, True)
    return (angle, False)

def get_procrustes_dists(images_r, n_iter = 5):
    avg_img = images_r[0]
    new_avg = np.zeros_like(avg_img)
    for it in range(n_iter):
        for i in range(len(images_r)):
            angle, flip = __best_rotate_flip(avg_img, images_r[i])
            images_r[i] = rotate(images_r[i], angle)
            if(flip): images_r[i] = images_r[i][:,::-1]
            new_avg += images_r[i] / len(images_r)
        avg_img = new_avg
    return images_r, avg_img

'''
    Local Binary Patterns
'''

