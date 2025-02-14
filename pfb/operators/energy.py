import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax._src.scipy.sparse.linalg import _vdot_tree
import nifty8.re as jft
from functools import partial
from ducc0.wgridder import dirty2vis, vis2dirty
from numpy.fft import fftshift, ifftshift
Fs = fftshift
iFs = ifftshift


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
    # # Brightness
    # a = x[:, :, 0]
    # b = x[:, :, 1]
    # c = x[:, :, 2]
    # d = x[:, :, 3]
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


def corr_to_stokes(x, wsum):
    '''
    x = [I+Q, U+1jV, U-1jV, I-Q]
    out = [I, Q, U, V]
    '''
    dirtyI = (x[:, :, 0] + x[:, :, -1]).real/wsum
    dirtyU = (x[:, :, 1] + x[:, :, 2]).real/wsum
    dirtyV = (x[:, :, 1] - x[:, :, 2]).imag/wsum
    dirtyQ = (x[:, :, 0] - x[:, :, -1]).real/wsum
    return dirtyI, dirtyQ, dirtyU, dirtyV

def stokes_to_corr(x):
    '''
    x = [I, Q, U, V]
    out = [I+Q, U+1jV, U-1jV, I-Q]
    '''
    dirty0 = x[:, :, 0] + x[:, :, 1]
    dirty1 = x[:, :, 2] + 1j*x[:, :, 3]
    dirty2 = x[:, :, 2] - 1j*x[:, :, 3]
    dirty3 = x[:, :, 0] - x[:, :, 1]
    return dirty0, dirty1, dirty2, dirty3


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
    p = 1.5
    I = p*np.sqrt(Q**2 + U**2 + V**2)

    # make vis
    m = 10000
    u = np.random.uniform(-1, 1, m)
    v = np.random.uniform(-1, 1, m)
    w = np.zeros(m)
    uvw = np.stack((u, v, w), axis=-1)

    freq = np.array([1e9])

    lightspeed = 299792458.0
    cell_N = 1.0 / (2 * 2 * freq[0] / lightspeed)

    vis_I = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=I,
        pixsize_x=cell_N,
        pixsize_y=cell_N,
        epsilon=1e-6,
    )

    vis_Q = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=Q,
        pixsize_x=cell_N,
        pixsize_y=cell_N,
        epsilon=1e-6,
    )

    vis_U = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=U,
        pixsize_x=cell_N,
        pixsize_y=cell_N,
        epsilon=1e-6,
    )

    vis_V = dirty2vis(
        uvw=uvw,
        freq=freq,
        dirty=V,
        pixsize_x=cell_N,
        pixsize_y=cell_N,
        epsilon=1e-6,
    )

    vis = np.zeros((m, 1, 4), dtype=np.complex128)
    vis[:, :, 0] = vis_I + vis_Q
    vis[:, :, 1] = vis_U + 1j*vis_V
    vis[:, :, 2] = vis_U - 1j*vis_V
    vis[:, :, 3] = vis_I - vis_Q


    # create dirty images (only off diagonal elements are expected to be complex)
    dirty0 = vis2dirty(uvw=uvw, freq=freq, vis=vis[:, :, 0],
                       npix_x=n, npix_y=n,
                       pixsize_x=cell_N, pixsize_y=cell_N,
                       epsilon=1e-6, do_wgridding=False,
                       nthreads=8).astype("f8")
    dirty1 = vis2dirty(uvw=uvw, freq=freq, vis=vis[:, :, 1],
                       npix_x=n, npix_y=n,
                       pixsize_x=cell_N, pixsize_y=cell_N,
                       epsilon=1e-6, do_wgridding=False,
                       nthreads=8).astype("f8") \
                +1j * vis2dirty(uvw=uvw, freq=freq, vis=-1j*vis[:, :, 1],
                                npix_x=n, npix_y=n,
                                pixsize_x=cell_N, pixsize_y=cell_N,
                                epsilon=1e-6, do_wgridding=False,
                                nthreads=8).astype("f8")
    dirty2 = vis2dirty(uvw=uvw, freq=freq, vis=vis[:, :, 2],
                       npix_x=n, npix_y=n,
                       pixsize_x=cell_N, pixsize_y=cell_N,
                       epsilon=1e-6, do_wgridding=False,
                       nthreads=8).astype("f8") \
                +1j * vis2dirty(uvw=uvw, freq=freq, vis=-1j*vis[:, :, 2],
                                npix_x=n, npix_y=n,
                                pixsize_x=cell_N, pixsize_y=cell_N,
                                epsilon=1e-6, do_wgridding=False,
                                nthreads=8).astype("f8")
    dirty3 = vis2dirty(uvw=uvw, freq=freq, vis=vis[:, :, 3],
                       npix_x=n, npix_y=n,
                       pixsize_x=cell_N, pixsize_y=cell_N,
                       epsilon=1e-6, do_wgridding=False,
                       nthreads=8).astype("f8")
    
    ID = np.stack((dirty0, dirty1, dirty2, dirty3), axis=-1)

    psf = vis2dirty(uvw=uvw, freq=freq, vis=2*np.ones((m, 1), dtype=np.complex128),
                    npix_x=2*n, npix_y=2*n,
                    pixsize_x=cell_N, pixsize_y=cell_N,
                    epsilon=1e-6, do_wgridding=False,
                    nthreads=8).astype("f8")
    
    wsum = psf.max()

    dirtyI, dirtyQ, dirtyU, dirtyV = corr_to_stokes(ID, wsum=wsum)


    import matplotlib.pyplot as plt
    # plt.figure('1')
    # plt.imshow(dirtyI)
    # plt.colorbar()

    # plt.figure('2')
    # plt.imshow(dirtyQ)
    # plt.colorbar()

    # plt.figure('3')
    # plt.imshow(dirtyU)
    # plt.colorbar()

    # plt.figure('4')
    # plt.imshow(dirtyV)
    # plt.colorbar()


    # plt.show()

    # psfhat = np.fft.fft2(iFs(psf), axes=(0, 1), norm='backward')
    # psfhat = np.tile(psfhat[:, :, None], (1, 1, 4))


    # # test convolve
    # S = np.stack([I, Q, U, V], axis=-1)
    # B = np.stack(stokes_to_corr(S), axis=-1)
    # # Convolution
    # Bhat = jnp.fft.fft2(B,
    #                     s=(2*n, 2*n),
    #                     axes=(0, 1),
    #                     norm='backward')
    # Bconv = jnp.fft.ifft2(Bhat * psfhat,
    #                       s=(2*n, 2*n),
    #                       axes=(0, 1),
    #                       norm='backward')[0:n, 0:n, :]/2
    

    psfhat = jnp.fft.rfft2(iFs(psf),
                           s=(2*n, 2*n),
                           axes=(0, 1), norm='backward')


    # test convolve
    Ihat = jnp.fft.rfft2(I,
                         s=(2*n, 2*n),
                         axes=(0, 1),
                         norm='backward')
    Qhat = jnp.fft.rfft2(Q,
                         s=(2*n, 2*n),
                         axes=(0, 1),
                         norm='backward')
    Uhat = jnp.fft.rfft2(U,
                         s=(2*n, 2*n),
                         axes=(0, 1),
                         norm='backward')
    Vhat = jnp.fft.rfft2(V,
                         s=(2*n, 2*n),
                         axes=(0, 1),
                         norm='backward')
    
    hat0 = Ihat + Qhat
    hat1 = Uhat + 1j*Vhat
    hat2 = Uhat - 1j*Vhat
    hat3 = Ihat - Qhat

    # import ipdb; ipdb.set_trace()

    conv0 = jnp.fft.irfft2(hat0 * psfhat,
                          s=(2*n, 2*n),
                          axes=(0, 1),
                          norm='backward')[0:n, 0:n]
    conv1 = jnp.fft.irfft2(hat1 * psfhat,
                          s=(2*n, 2*n),
                          axes=(0, 1),
                          norm='backward')[0:n, 0:n] + \
            1j*jnp.fft.irfft2(-1j*hat1 * psfhat,
                          s=(2*n, 2*n),
                          axes=(0, 1),
                          norm='backward')[0:n, 0:n]
    conv2 = jnp.fft.irfft2(hat2 * psfhat,
                          s=(2*n, 2*n),
                          axes=(0, 1),
                          norm='backward')[0:n, 0:n] + \
            1j*jnp.fft.irfft2(-1j*hat2 * psfhat,
                          s=(2*n, 2*n),
                          axes=(0, 1),
                          norm='backward')[0:n, 0:n]
    conv3 = jnp.fft.irfft2(hat3 * psfhat,
                          s=(2*n, 2*n),
                          axes=(0, 1),
                          norm='backward')[0:n, 0:n]
    
    Bconv = np.stack((conv0, conv1, conv2, conv3), axis=-1)/2
    
    dirtyI2, dirtyQ2, dirtyU2, dirtyV2 = corr_to_stokes(Bconv, wsum=wsum)

    print(np.abs(dirtyI - dirtyI2).max())
    print(np.abs(dirtyQ - dirtyQ2).max())
    print(np.abs(dirtyU - dirtyU2).max())
    print(np.abs(dirtyV - dirtyV2).max())


    # import matplotlib.pyplot as plt
    # plt.figure('1')
    # plt.imshow(dirtyI2)
    # plt.colorbar()

    # plt.figure('2')
    # plt.imshow(dirtyQ2)
    # plt.colorbar()

    # plt.figure('3')
    # plt.imshow(dirtyU2)
    # plt.colorbar()

    # plt.figure('4')
    # plt.imshow(dirtyV2)
    # plt.colorbar()


    # plt.show()

    dd = ID - Bconv
    
    
    # Plot the dirty images
    import matplotlib.pyplot as plt
    plt.figure('Dirty 0 real')
    plt.imshow(dd[:, :, 0].real/wsum)
    plt.colorbar()

    plt.figure('Dirty 0 imag')
    plt.imshow(dd[:, :, 0].imag/wsum)
    plt.colorbar()

    plt.figure('Dirty 1 real')
    plt.imshow(dd[:, :, 1].real/wsum)
    plt.colorbar()

    plt.figure('Dirty 1 imag')
    plt.imshow(dd[:, :, 1].imag/wsum)
    plt.colorbar()

    plt.figure('Dirty 2 real')
    plt.imshow(dd[:, :, 2].real/wsum)
    plt.colorbar()

    plt.figure('Dirty 2 imag')
    plt.imshow(dd[:, :, 2].imag/wsum)
    plt.colorbar()

    plt.figure('Dirty 3 real')
    plt.imshow(dd[:, :, 3].real/wsum)
    plt.colorbar()

    plt.figure('Dirty 3 imag')
    plt.imshow(dd[:, :, 3].imag/wsum)
    plt.colorbar()

    plt.show()

    # Plot the
    import ipdb; ipdb.set_trace()



    ID = jnp.array(ID)
    psfhat = jnp.array(psfhat)
    # x = jnp.zeros((n, n, 4), dtype=float)
    x = jnp.zeros((n*n*4), dtype=float)

    phi = pol_energy_approx(ID, psfhat, x)

    fun = partial(pol_energy_approx, ID, psfhat)

    print(phi, fun(x))

    x = jft.newton_cg(fun=fun, x0=x)