import numpy as np


class Mask(object):
    def __init__(self, mask):
        """
        Mask operator
        """
        nx, ny = mask.shape
        self.nx = nx
        self.ny = ny
        self.mask = mask

    def dot(self, x):
        """
        Components to image
        """
        return x[~self.mask]

    def hdot(self, x):
        """
        Image to components
        """
        res = np.zeros((self.nx, self.ny))
        res[~self.mask] = x
        return res
