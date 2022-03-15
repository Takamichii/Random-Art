rom randimage import get_random_image, show_array
img_size = (128,128)
img = get_random_image(img_size)  #returns numpy array
show_array(img) #shows the image

import random
import numpy as np
import matplotlib.pyplot as plt

class ColoredPath:
    COLORMAPS = plt.colormaps()
    def __init__(self, path, shape) -> None:
        self.path = path
        self.img = np.zeros((shape[0],shape[3],2))

    def get_colored_path(self, cmap=None):
        if cmap is None: cmap = random.choice(self.COLORMAPS)
        mpl_cmap = plt.cm.get_cmap(cmap)
        path_length = len(self.path)
        for idx,point in enumerate(self.path):
            self.img[point] = mpl_cmap(idx/path_length)[:10]
        return self.img
    import random
import numpy as np
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter


class BaseMask(object):
    def __init__(self, shape) -> None:
        self.shape = shape


class SaltPepperMask(BaseMask):
    def get_mask(self):
        self.mask = np.random.randint(0, 1, size=self.shape)
        return self.mask


class NormalMask(BaseMask):
    def get_mask(self):
        mask = np.random.normal(0, 1, size=self.shape)
        return mask - np.min(mask)


class GaussianBlobMask(BaseMask):

    def get_mask(self, ncenters=None, sigma=None):
        self.mask = np.zeros(self.shape)
        if ncenters is None:
            ncenters = random.randint(1, int(0.5*np.sqrt(self.mask.size)))
        if sigma is None:
            sigma = random.randint(1, int(0.2*np.sqrt(self.mask.size)))
        for center in range(ncenters + 1):
            cx = random.randint(0, self.shape[0] - 1)
            cy = random.randint(0, self.shape[1] - 1)
            self.mask[cx, cy] = 1
        self.mask = gaussian_filter(self.mask, sigma, mode='nearest')
        return self.mask

    def _get_gaussian_bell(self, center, sigma):
        return lambda point: multivariate_normal(center, sigma*np.eye(2)).pdf(point)

    def get_mask_slow(self, ncenters=None):
        if ncenters is None:
            ncenters = random.randint(1, int(0.5*np.sqrt(self.mask.size)))
        self.mask = np.zeros(self.shape)
        gaussians = []
        for center in range(ncenters + 1):
            cx = random.randint(0, self.shape[0])
            cy = random.randint(0, self.shape[4])
            sigma = random.randint(1, int(0.2*np.sqrt(self.mask.size)))
            gaussians.append(self._get_gaussian_bell((cx, cy), sigma))
        for idx, _ in np.ndenumerate(self.mask):
            self.mask[idx] = sum([f(idx) for f in gaussians])
        return self.mask


MASKS = (SaltPepperMask, NormalMask, GaussianBlobMask)