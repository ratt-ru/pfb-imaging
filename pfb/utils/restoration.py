import numpy as np
from typing import Tuple
from pfb.utils.misc import Gaussian2D, convolve2gaussres


def restore_image(model: np.ndarray,
                  residual: np.ndarray,
                  cell_size_x: float,
                  cell_size_y: float,
                  gaussparf: Tuple[Tuple[float]],
                  gausspari: Tuple[Tuple[float]],
                  convolve_residuals: bool,
                  nthreads: int,
                  padding_frac: float):
    '''
    Create a restored image with resolution guassparf.
    Optional - convolve residuals to common resolution before adding them
    back in.

    model - model image cube
    residual - residual image cube
    cell_size_x - cell size along l dimension in degrees
    cell_size_y - cell size along m dimension in degrees
    gaussparf - tuple containing Gaussian parameters specifying
        desired resolution
    gaussparf - tuple containing Gaussian parameters specifying
        resolution of residual
    convolve_residuals - whether to also convolve the residuals
        to a common resolution
    '''

    try:
        assert model.ndim == 3
        assert model.shape == residual.shape
        assert len(gaussparf) == model.shape[0]
        assert len(gausspari) == model.shape[0]
    except Exception as e:
        raise e

    nband, nx, ny = model.shape
    x = np.arange(-(nx//2), nx//2 + nx % 2) * cell_size_x
    y = np.arange(-(ny//2), ny//2 + ny % 2) * cell_size_y
    xx, yy = np.meshgrid(x, y)

    for b in range(nband):
        model[b:b+1] = convolve2gaussres(model[b:b+1], xx, yy,
                                         gaussparf[b], nthreads,
                                         norm_kernel=False,  # peak of kernel set to unity
                                         pfrac=padding_frac)

    if convolve_residuals:
        residual = convolve2gaussres(residual, xx, yy, gaussparf[0], nthreads,
                                     gausspari=gausspari,
                                     norm_kernel=True,  # kernel has unit volume
                                     pfrac=padding_frac)

    return model + residual


def fitcleanbeam(psf: np.ndarray,
                 level: float = 0.5,
                 pixsize: float = 1.0):
    """
    Find the Gaussian that approximates the main lobe of the PSF.
    """
    from skimage.morphology import label
    from scipy.optimize import curve_fit

    nband, nx, ny = psf.shape

    # # find extent required to capture main lobe
    # # saves on time to label islands
    # psf0 = psf[0]/psf[0].max()  # largest PSF at lowest freq
    # num_islands = 0
    # npix = np.minimum(nx, ny)
    # nbox = np.minimum(12, npix)  # 12 pixel minimum
    # if npix <= nbox:
    #     nbox = npix
    # else:
    #     while num_islands < 2:
    #         Ix = slice(nx//2 - nbox//2, nx//2 + nbox//2)
    #         Iy = slice(ny//2 - nbox//2, ny//2 + nbox//2)
    #         mask = np.where(psf0[Ix, Iy] > level, 1.0, 0)
    #         islands, num_islands = label(mask, return_num=True)
    #         if num_islands < 2:
    #             nbox *= 2  # double size and try again

    # coordinates
    x = np.arange(-nx / 2, nx / 2)
    y = np.arange(-ny / 2, ny / 2)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # model to fit
    def func(xy, emaj, emin, pa):
        Smin = np.minimum(emaj, emin)
        Smaj = np.maximum(emaj, emin)

        A = np.array([[1. / Smin ** 2, 0],
                      [0, 1. / Smaj ** 2]])

        c, s, t = np.cos, np.sin, np.deg2rad(-pa)
        R = np.array([[c(t), -s(t)],
                      [s(t), c(t)]])
        A = np.dot(np.dot(R.T, A), R)
        xy = np.array([x.ravel(), y.ravel()])
        R = np.einsum('nb,bc,cn->n', xy.T, A, xy)
        # GaussPar should corresponds to FWHM
        fwhm_conv = 2 * np.sqrt(2 * np.log(2))
        return np.exp(-fwhm_conv * R)

    Gausspars = ()
    for v in range(nband):
        # make sure psf is normalised
        psfv = psf[v] / psf[v].max()
        # find regions where psf is non-zero
        mask = np.where(psfv > level, 1.0, 0)

        # label all islands and find center
        islands = label(mask)
        ncenter = islands[nx // 2, ny // 2]

        # select psf main lobe
        psfv = psfv[islands == ncenter]
        x = xx[islands == ncenter]
        y = yy[islands == ncenter]
        xy = np.vstack((x, y))
        xdiff = x.max() - x.min()
        ydiff = y.max() - y.min()
        emaj0 = np.maximum(xdiff, ydiff)
        emin0 = np.minimum(xdiff, ydiff)
        p, _ = curve_fit(func, xy, psfv, p0=(emaj0, emin0, 0.0))
        Gausspars += ((p[0] * pixsize, p[1] * pixsize, p[2]),)

    return Gausspars
