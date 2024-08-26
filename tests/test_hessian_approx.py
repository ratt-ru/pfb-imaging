import itertools
import numpy as np
import pytest
from pathlib import Path
from pfb.operators.gridder import wgridder_conventions
from pfb.operators.hessian import _hessian_impl as hessian
from pfb.operators.psf import psf_convolve_slice
from pfb.utils.misc import set_image_size
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from scipy.constants import c as lightspeed
from daskms import xds_from_ms, xds_from_table
from ducc0.fft import c2r, r2c
iFs = np.fft.ifftshift
Fs = np.fft.fftshift
pmp = pytest.mark.parametrize

@pmp("center_offset", [(0.0, 0.0), (0.1, -0.17), (0.2, 0.5)])
def test_psfvis(center_offset, ms_name):
    test_dir = Path(ms_name).resolve().parent
    xds = xds_from_ms(ms_name,
                      chunks={'row': -1, 'chan': -1})[0]
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]
    uvw = xds.UVW.values
    freq = spw.CHAN_FREQ.values.squeeze()

    # uvw = ms.getcol('UVW')
    nrow = uvw.shape[0]
    nchan = freq.size

    umax = np.abs(uvw[:, 0]).max()
    vmax = np.abs(uvw[:, 1]).max()
    uv_max = np.maximum(umax, vmax)
    max_freq = freq.max()

    nx, ny, nx_psf, ny_psf, cell_N, cell_rad = set_image_size(
                    uv_max,
                    max_freq,
                    1.0,
                    2.0)
    x0, y0 = center_offset
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(x0, y0)
    epsilon = 1e-10
    signu = -1.0 if flip_u else 1.0
    signv = -1.0 if flip_v else 1.0
    # we need these in the test because of flipped wgridder convention
    # https://github.com/mreineck/ducc/issues/34
    signx = -1.0 if flip_u else 1.0
    signy = -1.0 if flip_v else 1.0

    # produce PSF visibilities centered at x0, y0
    n = np.sqrt(1 - x0**2 - y0**2)
    freqfactor = -2j*np.pi*freq[None, :]/lightspeed
    psf_vis = np.exp(freqfactor*(signu*uvw[:, 0:1]*x0*signx +
                                 signv*uvw[:, 1:2]*y0*signy -
                                 uvw[:, 2:]*(n-1)))
    x = np.zeros((nx, ny), dtype='f8')
    x[nx//2, ny//2] = 1.0
    psf_vis2 = dirty2vis(
                    uvw=uvw,
                    freq=freq,
                    dirty=x,
                    pixsize_x=cell_rad,
                    pixsize_y=cell_rad,
                    center_x=x0,
                    center_y=y0,
                    flip_u=flip_u,
                    flip_v=flip_v,
                    flip_w=flip_w,
                    epsilon=epsilon,
                    nthreads=8,
                    do_wgridding=True,
                    divide_by_n=False)

    assert np.abs(psf_vis - psf_vis2).max() <= epsilon



@pmp("center_offset", [(0.0, 0.0), (0.1, -0.17), (0.2, 0.5)])
def test_hessian(center_offset, ms_name):
    test_dir = Path(ms_name).resolve().parent
    xds = xds_from_ms(ms_name,
                      chunks={'row': -1, 'chan': -1})[0]
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]
    uvw = xds.UVW.values
    freq = spw.CHAN_FREQ.values.squeeze()

    nrow = uvw.shape[0]
    nchan = freq.size

    umax = np.abs(uvw[:, 0]).max()
    vmax = np.abs(uvw[:, 1]).max()
    uv_max = np.maximum(umax, vmax)
    max_freq = freq.max()

    x0, y0 = center_offset
    nx, ny, nx_psf, ny_psf, cell_N, cell_rad = set_image_size(
                    uv_max,
                    max_freq,
                    1.5,
                    2.0)

    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(x0, y0)
    epsilon = 1e-10
    signu = -1.0 if flip_u else 1.0
    signv = -1.0 if flip_v else 1.0
    # we need these in the test because of flipped wgridder convention
    # https://github.com/mreineck/ducc/issues/34
    signx = -1.0 if flip_u else 1.0
    signy = -1.0 if flip_v else 1.0

    # produce PSF visibilities centered at x0, y0
    n = np.sqrt(1 - x0**2 - y0**2)
    freqfactor = -2j*np.pi*freq[None, :]/lightspeed
    psf_vis = np.exp(freqfactor*(signu*uvw[:, 0:1]*x0*signx +
                                 signv*uvw[:, 1:2]*y0*signy))


    x = np.zeros((nx, ny), dtype='f8')
    x[nx//2, ny//2] = 1.0
    psf = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=psf_vis,
        wgt=None,
        npix_x=nx_psf,
        npix_y=ny_psf,
        pixsize_x=cell_rad,
        pixsize_y=cell_rad,
        center_x=x0,
        center_y=y0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        epsilon=epsilon,
        do_wgridding=False,
        divide_by_n=False,  # else we also need it in PSF convolve
        nthreads=8,
        verbosity=0,
    )

    psfhat = r2c(iFs(psf, axes=(0, 1)), axes=(0, 1),
                     nthreads=8,
                     forward=True, inorm=0)


    res1 = hessian(
        x,
        uvw=uvw,
        weight=np.ones((nrow, nchan), dtype='f8'),
        vis_mask=np.ones((nrow, nchan), dtype=np.uint8),
        freq=freq,
        cell=cell_rad,
        x0=x0,
        y0=y0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        do_wgridding=False,
        epsilon=epsilon,
        double_accum=True,
        nthreads=8
    )

    res2 = psf_convolve_slice(np.zeros((nx_psf, ny_psf)),
                              np.zeros_like(psfhat),
                              np.zeros_like(x),
                              psfhat,
                              ny_psf,
                              x,
                              nthreads=8)

    scale = np.abs(res2).max()
    diff = (res2-res1)/scale
    assert np.allclose(1 + diff, 1)
