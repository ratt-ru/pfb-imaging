import itertools
import numpy as np
import pytest
pmp = pytest.mark.parametrize
from ducc0.wgridder import dirty2vis, vis2dirty
from ducc0.fft import c2r, r2c
iFs = np.fft.ifftshift
Fs = np.fft.fftshift
from pfb.operators.hessian import _hessian_impl as hessian
from pfb.operators.psf import psf_convolve_slice
from ducc0.misc import make_noncritical
from ducc0.wgridder import vis2dirty
from ducc0.wgridder import dirty2vis



# @pytest.mark.parametrize("center_offset", [(0.0, 0.0), (0.1, -0.17), (0.2, 0.5)])
def test_hessian():
    # np.random.seed(42)
    nx, ny = 128, 128
    nant = 10
    nchan = 2

    pixsize = 0.5 * np.pi / 180 / 3600.  # 1 arcsec ~ 4 pixels / beam, so we'll avoid aliasing
    l0, m0 = 0.0, 0.0
    dl = pixsize
    dm = pixsize

    ant1, ant2 = np.asarray(list(itertools.combinations(range(nant), 2))).T
    antennas = 10e3 * np.random.normal(size=(nant, 3))
    antennas[:, 2] *= 0.001
    uvw = antennas[ant2] - antennas[ant1]
    nrow = uvw.shape[0]
    freqs = np.linspace(700e6, 2000e6, nchan)

    epsilon = 1e-12
    uvwneg = uvw.copy()
    uvwneg[:, 2] *= -1
    psf = vis2dirty(
        uvw=uvwneg,
        freq=freqs,
        vis=np.ones((nrow, nchan), dtype='c16'),
        wgt=None,
        npix_x=2*nx,
        npix_y=2*ny,
        pixsize_x=dl,
        pixsize_y=dm,
        center_x=l0,
        center_y=m0,
        epsilon=epsilon,
        do_wgridding=False,
        flip_v=False,
        divide_by_n=False,  # else we also need it in the PSF convolve
        nthreads=1,
        verbosity=0,
    )

    psfhat = r2c(iFs(psf, axes=(0, 1)), axes=(0, 1),
                     nthreads=8,
                     forward=True, inorm=0)

    x = np.random.normal(size=(nx, ny))
    # x[...] = 0.0
    # x[nx//2, ny//2] = 1.0
    # x[nx//2-1, ny//2] = 0.5
    # x[nx//2, ny//2-1] = 0.5
    beam = np.ones((nx, ny))
    res1 = hessian(
        x,
        uvw,
        np.ones((nrow, nchan), dtype='f8'),
        np.ones((nrow, nchan), dtype=np.uint8),
        freqs,
        beam,
        x0=l0, y0=m0,
        cell=pixsize,
        do_wgridding=False,
        epsilon=epsilon,
        double_accum=True,
        nthreads=8
    )

    res2 = psf_convolve_slice(np.zeros((2*nx, 2*ny)),
                              np.zeros_like(psfhat),
                              np.zeros_like(x),
                              psfhat,
                              2*ny,
                              x*beam,
                              nthreads=8)

    scale = np.abs(res2).max()

    diff = res2-res1
    assert np.allclose(1 + diff, 1)

