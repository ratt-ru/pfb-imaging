'''
This is an attempt to develop a minor cycle for the resolve polarisation model.

The rationale here is that primary beam patterns are actually for corellations, not Stokes parameters.

As a starting point, we try to figure out how to compute the dirty image as
a convolution of the model with the PSF i.e.

ID = PSF.conv(B) = R.H Ninv V

where B is the brightness matrix.

'''


import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax._src.scipy.sparse.linalg import _vdot_tree
import nifty8.re as jft
from functools import partial
from ducc0.wgridder import dirty2vis, vis2dirty
from ducc0.fft import c2c, r2c, c2r
from ducc0.nufft import u2nu, nu2u
from numpy.fft import fftshift, ifftshift
Fs = fftshift
iFs = ifftshift
import matplotlib.pyplot as plt
from pfb_imaging.utils.stokes import stokes_to_corr, corr_to_stokes


@jax.jit
def pol_energy_approx(ID, psfhat, x):
    """
    x - (4, nx, nx) - signal
    ID - (nx, nx, 4) - dirty image
    psfhat - (nx_psf, ny_psf, 4) FFT of point spread function
    """
    psfh = jax.lax.stop_gradient(psfhat)
    Id = jax.lax.stop_gradient(ID)
    nx_psf, ny_psf, _ = psfh.shape
    nx, ny, _ = Id.shape
    # Convert to brightness
    a = x[0:n*n].reshape(n, n)
    b = x[n*n:2*n*n].reshape(n, n)
    c = x[2*n*n:3*n*n].reshape(n, n)
    d = x[3*n*n:].reshape(n, n)
    y = jnp.stack((a+b, c+1j*d, c-1j*d, a-b), axis=-1).reshape(nx, ny, 2, 2)
    B = expm(y).reshape(nx, ny, 4)
    # Convolution
    Bhat = jnp.fft.fft2(B,
                        s=(nx_psf, ny_psf),
                        axes=(0, 1),
                        norm='backward')
    Bconv = jnp.fft.ifft2(Bhat * psfh,
                         s=(nx_psf, ny_psf),
                         axes=(0, 1),
                         norm='backward')[0:nx, 0:ny, :]
    # # _vdot_tree should promote to double during vdot
    # return (_vdot_tree(B, Bconv).real -
    #         _vdot_tree(B, Id).real*2 +
    #         _vdot_tree(a, a) + 
    #         _vdot_tree(b, b) +
    #         _vdot_tree(c, c) +
    #         _vdot_tree(d, d))
    
    return (jnp.vdot(B, Bconv).real -
            jnp.vdot(B, Id).real*2)
            # jnp.sum(a*a) + 
            # jnp.sum(b*b) +
            # jnp.sum(c*c) +
            # jnp.sum(d*d))


def stokes_from_abcd(a, b, c, d):
    p = np.sqrt(b**2 + c**2 + d**2)
    I = np.exp(a)*np.cosh(p)
    Q = b*np.exp(a)*np.sinh(p)/p
    U = c*np.exp(a)*np.sinh(p)/p
    V = d*np.exp(a)*np.sinh(p)/p
    return I, Q, U, V


if __name__=="__main__":
    n = 128
    I = np.zeros((n, n))
    Q = np.zeros((n, n))
    U = np.zeros((n, n))
    V = np.zeros((n, n))

    np.random.seed(42)
    idx = np.random.randint(10, n-10, 10)
    idy = np.random.randint(10, n-10, 10)

    Q[idx, idy] = 1
    U[idx, idy] = -1
    V[idx, idy] = 0.1
    p = 0.7
    I = p*np.sqrt(Q**2 + U**2 + V**2)/p

    # make vis
    m = 10000
    u = np.random.uniform(-1, 1, m)
    v = np.random.uniform(-1, 1, m)
    w = np.zeros(m)
    uvw = np.stack((u, v, w), axis=-1)

    freq = np.array([1e9])

    lightspeed = 299792458.0
    uv_max = 2.0  # in (-1, 1)
    cell_N = 1.0 / (2 * uv_max * freq[0] / lightspeed)
    epsilon = 1e-6  # gridding precision

    # convert Stokes to vis
    vis_I = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=I,
        pixsize_x=cell_N,
        pixsize_y=cell_N,
        epsilon=epsilon,
    )

    vis_Q = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=Q,
        pixsize_x=cell_N,
        pixsize_y=cell_N,
        epsilon=epsilon,
    )

    vis_U = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=U,
        pixsize_x=cell_N,
        pixsize_y=cell_N,
        epsilon=epsilon,
    )

    vis_V = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=V,
        pixsize_x=cell_N,
        pixsize_y=cell_N,
        epsilon=epsilon,
    )

    # convert Stokes vis to corr
    vis = np.zeros((m, 1, 4), dtype=np.complex128)
    vis[:, :, 0] = vis_I + vis_Q
    vis[:, :, 1] = vis_U + 1j*vis_V
    vis[:, :, 2] = vis_U - 1j*vis_V
    vis[:, :, 3] = vis_I - vis_Q


    # create dirty images (only off diagonal elements are expected to be complex)
    dirty0 = vis2dirty(uvw=uvw, freq=freq, vis=vis[:, :, 0],
                       npix_x=n, npix_y=n,
                       pixsize_x=cell_N, pixsize_y=cell_N,
                       epsilon=epsilon, do_wgridding=False,
                       nthreads=8).astype("f8")
    dirty1 = vis2dirty(uvw=uvw, freq=freq, vis=vis[:, :, 1],
                       npix_x=n, npix_y=n,
                       pixsize_x=cell_N, pixsize_y=cell_N,
                       epsilon=epsilon, do_wgridding=False,
                       nthreads=8).astype("f8") \
                +1j * vis2dirty(uvw=uvw, freq=freq, vis=-1j*vis[:, :, 1],
                                npix_x=n, npix_y=n,
                                pixsize_x=cell_N, pixsize_y=cell_N,
                                epsilon=epsilon, do_wgridding=False,
                                nthreads=8).astype("f8")
    dirty2 = vis2dirty(uvw=uvw, freq=freq, vis=vis[:, :, 2],
                       npix_x=n, npix_y=n,
                       pixsize_x=cell_N, pixsize_y=cell_N,
                       epsilon=epsilon, do_wgridding=False,
                       nthreads=8).astype("f8") \
                +1j * vis2dirty(uvw=uvw, freq=freq, vis=-1j*vis[:, :, 2],
                                npix_x=n, npix_y=n,
                                pixsize_x=cell_N, pixsize_y=cell_N,
                                epsilon=epsilon, do_wgridding=False,
                                nthreads=8).astype("f8")
    dirty3 = vis2dirty(uvw=uvw, freq=freq, vis=vis[:, :, 3],
                       npix_x=n, npix_y=n,
                       pixsize_x=cell_N, pixsize_y=cell_N,
                       epsilon=epsilon, do_wgridding=False,
                       nthreads=8).astype("f8")
    
    ID = np.stack((dirty0, dirty1, dirty2, dirty3), axis=-1)

    # in the absence of gains the PSF for each correlation should be the same
    psf = vis2dirty(uvw=uvw, freq=freq, vis=2*np.ones((m, 1), dtype=np.complex128),
                    npix_x=2*n, npix_y=2*n,
                    pixsize_x=cell_N, pixsize_y=cell_N,
                    epsilon=epsilon, do_wgridding=False,
                    nthreads=8).astype("f8")
    
    wsum = psf.max()

    dirtyI, dirtyQ, dirtyU, dirtyV = corr_to_stokes(ID, wsum=wsum)


    plt.figure('1')
    plt.imshow(dirtyI)
    plt.colorbar()

    plt.figure('2')
    plt.imshow(dirtyQ)
    plt.colorbar()

    plt.figure('3')
    plt.imshow(dirtyU)
    plt.colorbar()

    plt.figure('4')
    plt.imshow(dirtyV)
    plt.colorbar()

    plt.show()


    # don't use rfft since corrs could be complex
    psfhat = np.fft.fft2(iFs(psf), axes=(0, 1), norm='backward')
    psfhat = np.tile(psfhat[:, :, None], (1, 1, 4))


    # test convolve
    S = np.stack([I, Q, U, V], axis=-1)
    B = np.stack(stokes_to_corr(S), axis=-1)
    # Convolution
    Bhat = jnp.fft.fft2(B,
                        s=(2*n, 2*n),
                        axes=(0, 1),
                        norm='backward')
    Bconv = jnp.fft.ifft2(Bhat * psfhat,
                          s=(2*n, 2*n),
                          axes=(0, 1),
                          norm='backward')[0:n, 0:n, :]/2
    
    # convert back to stokes
    dirtyI2, dirtyQ2, dirtyU2, dirtyV2 = corr_to_stokes(Bconv, wsum=wsum)

    # compare Stokes dirty images
    resI = np.abs(dirtyI - dirtyI2)
    resQ = np.abs(dirtyQ - dirtyQ2)
    resU = np.abs(dirtyU - dirtyU2)
    resV = np.abs(dirtyV - dirtyV2)

    assert np.allclose(resI, 0, atol=epsilon)
    assert np.allclose(resQ, 0, atol=epsilon)
    assert np.allclose(resU, 0, atol=epsilon)
    assert np.allclose(resV, 0, atol=epsilon)

    # compare corr dirty images
    res0 = ID[:, :, 0] - Bconv[:, :, 0]
    res1 = ID[:, :, 1] - Bconv[:, :, 1]
    res2 = ID[:, :, 2] - Bconv[:, :, 2]
    res3 = ID[:, :, 3] - Bconv[:, :, 3]
