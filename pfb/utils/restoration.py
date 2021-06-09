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


