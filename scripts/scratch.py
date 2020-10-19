
import numpy as np
from pfb.opt import hogbom
from pfb.utils import give_edges, load_fits
import matplotlib.pyplot as plt


if __name__=="__main__":
    psf = load_fits('/home/landman/Data/VLA/CYGA/pfbclean_out/config-D_psf.fits')
    dirty = load_fits('/home/landman/Data/VLA/CYGA/pfbclean_out/config-D_dirty.fits')
    nband, nx, ny = dirty.shape
    psf = psf[:, nx//2:3*nx//2, ny//2:3*ny//2]
    psf_max = psf_max = np.amax(psf.reshape(nband, nx*ny), axis=1)[:, None, None]

    dirty /= psf_max
    psf /= psf_max

    model, residual = hogbom(dirty, psf, pf=0.1)


    plt.figure('dirty')
    plt.imshow(dirty[0])
    plt.colorbar()

    plt.figure('psf')
    plt.imshow(psf[0])
    plt.colorbar()

    plt.figure('model')
    plt.imshow(model[0], vmin=0.0, vmax=0.01)
    plt.colorbar()

    plt.figure('residual')
    plt.imshow(residual[0], vmin=0.0, vmax=0.01)
    plt.colorbar()
    plt.show()

    