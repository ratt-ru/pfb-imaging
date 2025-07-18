import itertools
from functools import partial
import numpy as np
import pytest
from pathlib import Path
from pfb.operators.gridder import wgridder_conventions
from pfb.operators.hessian import _hessian_slice, hessian_slice_jax
from pfb.operators.psf import (psf_convolve_slice,
                               psf_convolve_slice_jax) 
from pfb.utils.misc import set_image_size
from pfb.utils.stokes import stokes_to_corr, corr_to_stokes
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from scipy.constants import c as lightspeed
from daskms import xds_from_ms, xds_from_table
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from ducc0.fft import r2c, c2c, c2r
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


    res1 = _hessian_slice(
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


def test_hessian_jax(ms_name):
    xds = xds_from_ms(ms_name,
                      chunks={'row': -1, 'chan': -1})[0]
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]
    uvw = xds.UVW.values
    freq = spw.CHAN_FREQ.values.squeeze()

    umax = np.abs(uvw[:, 0]).max()
    vmax = np.abs(uvw[:, 1]).max()
    uv_max = np.maximum(umax, vmax)
    max_freq = freq.max()

    x0, y0 = 0.0, 0.0
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

    res1 = psf_convolve_slice(np.zeros((nx_psf, ny_psf)),
                              np.zeros_like(psfhat),
                              np.zeros_like(x),
                              psfhat,
                              ny_psf,
                              x,
                              nthreads=8)

    x = jnp.array(x)
    psfhat = jnp.array(psfhat)

    res2 = psf_convolve_slice_jax(
        nx, ny,
        nx_psf, ny_psf,
        psfhat,
        x)

    scale = np.abs(res2).max()
    diff = (res2-res1)/scale
    assert np.allclose(1 + diff, 1)


def test_hessian_inv_jax(ms_name):
    xds = xds_from_ms(ms_name,
                      chunks={'row': -1, 'chan': -1})[0]
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]
    uvw = xds.UVW.values
    freq = spw.CHAN_FREQ.values.squeeze()

    umax = np.abs(uvw[:, 0]).max()
    vmax = np.abs(uvw[:, 1]).max()
    uv_max = np.maximum(umax, vmax)
    max_freq = freq.max()

    x0, y0 = 0.0, 0.0
    nx, ny, nx_psf, ny_psf, cell_N, cell_rad = set_image_size(
                    uv_max,
                    max_freq,
                    1.0,
                    1.1)

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
    
    abspsf = jnp.array(jnp.abs(psfhat)) #, dtype=jnp.float32)

    dirty = jnp.array(psf[nx_psf//4:3*nx_psf//4, ny_psf//4:3*ny_psf//4]) #, dtype=jnp.float32)

    eta = 1e-5
    A = partial(hessian_slice_jax, nx, ny, nx_psf, ny_psf, eta, abspsf)

    xrec = cg(A, dirty, tol=1e-8, maxiter=nx_psf)

    res1 = psf_convolve_slice(np.zeros((nx_psf, ny_psf)),
                              np.zeros_like(psfhat),
                              np.zeros_like(x),
                              psfhat,
                              ny_psf,
                              xrec[0],
                              nthreads=8)
    eps = (np.abs(res1 - dirty)/np.abs(dirty).max()).max()
    assert eps < 0.05  # max diff is less than 5%


def test_complex_convolve(ms_name):
    xds = xds_from_ms(ms_name,
                      chunks={'row': -1, 'chan': -1})[0]
    spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]
    uvw = xds.UVW.values
    freq = spw.CHAN_FREQ.values.squeeze()

    umax = np.abs(uvw[:, 0]).max()
    vmax = np.abs(uvw[:, 1]).max()
    uv_max = np.maximum(umax, vmax)
    max_freq = freq.max()

    x0, y0 = 0.0, 0.0
    nx, ny, nx_psf, ny_psf, cell_N, cell_rad = set_image_size(
                    uv_max,
                    max_freq,
                    1.0,
                    1.1)

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
    
    # create complex psf
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
        verbosity=0) + \
        1j*vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=-1j*psf_vis,
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
        verbosity=0)
    
    # create a sky model
    I = np.zeros((nx, ny))
    Q = np.zeros((nx, ny))
    U = np.zeros((nx, ny))
    V = np.zeros((nx, ny))

    np.random.seed(42)
    idx = np.random.randint(10, nx-10, 10)
    idy = np.random.randint(10, ny-10, 10)

    Q[idx, idy] = 1
    U[idx, idy] = -1
    V[idx, idy] = 0.1
    p = 0.7
    I = np.sqrt(Q**2 + U**2 + V**2)/p

    # TODO - real and imaginary separately with beam
    # convert Stokes to vis
    vis_I = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=I,
        pixsize_x=cell_rad,
        pixsize_y=cell_rad,
        epsilon=epsilon,
        do_wgridding=False,
        divide_by_n=False,  # else we also need it in PSF convolve
        nthreads=8,
        verbosity=0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        center_x=x0,
        center_y=y0
    )

    vis_Q = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=Q,
        pixsize_x=cell_rad,
        pixsize_y=cell_rad,
        epsilon=epsilon,
        do_wgridding=False,
        divide_by_n=False,  # else we also need it in PSF convolve
        nthreads=8,
        verbosity=0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        center_x=x0,
        center_y=y0
    )

    vis_U = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=U,
        pixsize_x=cell_rad,
        pixsize_y=cell_rad,
        epsilon=epsilon,
        do_wgridding=False,
        divide_by_n=False,  # else we also need it in PSF convolve
        nthreads=8,
        verbosity=0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        center_x=x0,
        center_y=y0
    )

    vis_V = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=V,
        pixsize_x=cell_rad,
        pixsize_y=cell_rad,
        epsilon=epsilon,
        do_wgridding=False,
        divide_by_n=False,  # else we also need it in PSF convolve
        nthreads=8,
        verbosity=0,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=flip_w,
        center_x=x0,
        center_y=y0
    )

    # convert Stokes vis to corr
    stokes_vis = np.stack((vis_I, vis_Q, vis_U, vis_V), axis=-1)
    vis = stokes_to_corr(stokes_vis, axis=-1)

    # create dirty images
    dirty0 = vis2dirty(uvw=uvw,
                       freq=freq,
                       vis=vis[:, :, 0],
                       npix_x=nx, npix_y=ny,
                       pixsize_x=cell_rad, pixsize_y=cell_rad,
                       epsilon=epsilon,
                       do_wgridding=False,
                       nthreads=8,
                       flip_u=flip_u,
                       flip_v=flip_v,
                       flip_w=flip_w,
                       center_x=x0,
                       center_y=y0).astype("f8")\
                +1j * vis2dirty(uvw=uvw,
                                freq=freq,
                                vis=-1j*vis[:, :, 0],
                                npix_x=nx, npix_y=ny,
                                pixsize_x=cell_rad, pixsize_y=cell_rad,
                                epsilon=epsilon,
                                do_wgridding=False,
                                nthreads=8,
                                flip_u=flip_u,
                                flip_v=flip_v,
                                flip_w=flip_w,
                                center_x=x0,
                                center_y=y0).astype("f8")
    dirty1 = vis2dirty(uvw=uvw,
                       freq=freq,
                       vis=vis[:, :, 1],
                       npix_x=nx, npix_y=ny,
                       pixsize_x=cell_rad, pixsize_y=cell_rad,
                       epsilon=epsilon,
                       do_wgridding=False,
                       nthreads=8,
                       flip_u=flip_u,
                       flip_v=flip_v,
                       flip_w=flip_w,
                       center_x=x0,
                       center_y=y0).astype("f8") \
                +1j * vis2dirty(uvw=uvw,
                                freq=freq,
                                vis=-1j*vis[:, :, 1],
                                npix_x=nx, npix_y=ny,
                                pixsize_x=cell_rad, pixsize_y=cell_rad,
                                epsilon=epsilon,
                                do_wgridding=False,
                                nthreads=8,
                                flip_u=flip_u,
                                flip_v=flip_v,
                                flip_w=flip_w,
                                center_x=x0,
                                center_y=y0).astype("f8")
    dirty2 = vis2dirty(uvw=uvw,
                       freq=freq,
                       vis=vis[:, :, 2],
                       npix_x=nx, npix_y=ny,
                       pixsize_x=cell_rad, pixsize_y=cell_rad,
                       epsilon=epsilon,
                       do_wgridding=False,
                       nthreads=8,
                       flip_u=flip_u,
                       flip_v=flip_v,
                       flip_w=flip_w,
                       center_x=x0,
                       center_y=y0).astype("f8") \
                +1j * vis2dirty(uvw=uvw,
                                freq=freq,
                                vis=-1j*vis[:, :, 2],
                                npix_x=nx, npix_y=ny,
                                pixsize_x=cell_rad, pixsize_y=cell_rad,
                                epsilon=epsilon,
                                do_wgridding=False,
                                nthreads=8,
                                flip_u=flip_u,
                                flip_v=flip_v,
                                flip_w=flip_w,
                                center_x=x0,
                                center_y=y0).astype("f8")
    dirty3 = vis2dirty(uvw=uvw,
                       freq=freq,
                       vis=vis[:, :, 3],
                       npix_x=nx, npix_y=ny,
                       pixsize_x=cell_rad, pixsize_y=cell_rad,
                       epsilon=epsilon,
                       do_wgridding=False,
                       nthreads=8,
                       flip_u=flip_u,
                       flip_v=flip_v,
                       flip_w=flip_w,
                       center_x=x0,
                       center_y=y0).astype("f8")\
                +1j * vis2dirty(uvw=uvw,
                                freq=freq,
                                vis=-1j*vis[:, :, 3],
                                npix_x=nx, npix_y=ny,
                                pixsize_x=cell_rad, pixsize_y=cell_rad,
                                epsilon=epsilon,
                                do_wgridding=False,
                                nthreads=8,
                                flip_u=flip_u,
                                flip_v=flip_v,
                                flip_w=flip_w,
                                center_x=x0,
                                center_y=y0).astype("f8")
    
    ID = np.stack((dirty0, dirty1, dirty2, dirty3), axis=0)

    # complex convolution
    psfhat = c2c(iFs(psf), axes=(0, 1), forward=True, inorm=0)
    
    # since there are no weights or gains the PSF is the same for all Stokes parameters
    psfhat = np.tile(psfhat[None, :, :], (4, 1, 1))

    # Convolution
    S = np.stack([I, Q, U, V], axis=0)
    B = stokes_to_corr(S, axis=0)
    Bpad = np.pad(B, ((0, 0), (0, nx), (0, ny)), mode='constant')
    Bhat = c2c(Bpad,
               axes=(1, 2),
               forward=True,
               inorm=0)
    Bconv = c2c(Bhat * psfhat,
                axes=(1, 2),
                forward=False,
                inorm=2)[:, 0:nx, 0:ny]

    assert np.allclose(Bconv, ID, atol=epsilon)