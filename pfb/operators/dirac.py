import numpy as np

class Dirac(object):
    def __init__(self, nband, nx, ny, mask=None):
        """
        Models image as a sum of Dirac deltas i.e.
        
        x = H beta

        where H is a design matrix that maps the Dirac coefficients onto the image cube.

        Parameters
        ----------
        nband - number of bands
        nx - number of pixels in x-dimension
        ny - number of pixels in y-dimension
        mask - nx x my bool array containing locations of sources
        """
        self.nx = nx
        self.ny = ny
        self.nband = nband

        if mask is not None:
            self.mask = mask
        else:
            self.mask = lambda x: x

    def dot(self, x):
        """
        Components to image
        """
        return self.mask[None, :, :] * x

    def hdot(self, x):
        """
        Image to components
        """
        return self.mask[None, :, :] * x

    def update_locs(self, model):
        self.mask = np.logical_or(self.mask, np.any(model, axis=0))

    def trim_fat(self, model):
        self.mask = np.any(model, axis=0)