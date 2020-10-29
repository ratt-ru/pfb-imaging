
import numpy as np
from pfb.operators import Dirac, Prior, PSF
from pfb.opt import pcg
from ducc0.wgridder import ms2dirty, dirty2ms
import matplotlib.pyplot as plt
from africanus.constants import c as lightspeed
import pywt

nband = 1
nx = 128
ny = 128

def psi_func(y, H):
    alpha = y[:, 0:nx*ny].reshape(nband, nx, ny)
    beta = y[:, nx*ny::] 
    return alpha + H.dot(beta)

def psih_func(x, H):
    y = x.reshape(nband, nx*ny)
    y = np.append(y, H.hdot(x), axis=1)
    return y

def hess_func(x, A, psf, psi, psih, sig_b):
    alpha = x[:, 0:nx*ny].reshape(nband, nx, ny)
    tmp1 = A.idot(alpha).reshape(nband, nx*ny)

    beta = x[:, nx*ny::]
    tmp2 = beta/sig_b**2
    return  psih(psf.convolve(psi(x))) + np.append(tmp1, tmp2, axis=1)

def M_func(x, A, sig_b):
    alpha = x[:, 0:nx*ny].reshape(nband, nx, ny)
    tmp1 = A.convolve(alpha).reshape(nband, nx*ny)

    beta = x[:, nx*ny::]
    tmp2 = beta*sig_b**2

    return np.append(tmp1, tmp2, axis=1)


if __name__=="__main__":
    nrow = 100000
    uvw = np.random.randn(nrow, 3)
    uvw[:, 2] = 0.0

    u_max = abs(uvw[:, 0]).max()
    v_max = abs(uvw[:, 1]).max()
    uv_max = np.maximum(u_max, v_max)

    cell = 0.9/(2*uv_max)

    freq = np.array([lightspeed])

    model = np.zeros((nband, nx, ny))
    npoints = 10
    Ix = np.random.randint(0, nx, npoints)
    Iy = np.random.randint(0, ny, npoints)
    model[:, Ix, Iy] = np.abs(1.0 + np.random.randn(npoints))
    mask = np.zeros((nx, ny))
    mask[Ix, Iy] = 1.0

    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    locs = ((nx//4, ny//4), (nx//2, ny//2), (3*nx//4, ny//4), (nx//4, 3*ny//4), (3*nx//4, 3*ny//4))
    
    for loc in locs: 
        model += np.exp(-(xx-loc[0])**2/15**2 - (yy-loc[1])**2/15**2)[None, :, :]


    vis = dirty2ms(uvw=uvw, freq=freq, dirty=model[0], pixsize_x=cell, pixsize_y=cell, epsilon=1e-7, do_wstacking=False, nthreads=8)
    noise = np.random.randn(nrow, 1)/np.sqrt(2) + 1.0j*np.random.randn(nrow, 1)/np.sqrt(2)

    data = vis + noise

    dirty = ms2dirty(uvw=uvw, freq=freq, ms=data, npix_x=nx, npix_y=ny, pixsize_x=cell, pixsize_y=cell, epsilon=1e-7, do_wstacking=False, nthreads=8)[None, :, :]

    psf_array = ms2dirty(uvw=uvw, freq=freq, ms=np.ones(data.shape, dtype=np.complex128),
                         npix_x=2*nx, npix_y=2*ny, pixsize_x=cell, pixsize_y=cell, epsilon=1e-7, do_wstacking=False, nthreads=8)[None, :, :]

    plt.figure('dirty')
    plt.imshow(dirty[0])
    plt.colorbar()

    plt.figure('psf')
    plt.imshow(psf_array[0])
    plt.colorbar()

    psf = PSF(psf_array, 8)

    A = Prior(1.0, nband, nx, ny)

    H = Dirac(nband, nx, ny, mask=mask)

    psi = lambda y:psi_func(y, H)
    psih = lambda x:psih_func(x, H)

    sig_b = 10

    hess = lambda x:hess_func(x, A, psf, psi, psih, sig_b)

    augmented_dirty = psih(dirty)
    M = lambda x:M_func(x, A, sig_b)
    x = pcg(hess, augmented_dirty, np.zeros(augmented_dirty.shape), M=M, tol=1e-7, maxit=100, verbosity=1)

    model_rec = psi(x)

    plt.figure('rec')
    plt.imshow(model_rec[0])
    plt.colorbar()

    plt.figure('model')
    plt.imshow(model[0])
    plt.colorbar()

    # get normal Wiener filter solution
    def hess(x):
        return psf.convolve(x) + A.idot(x)

    x = pcg(hess, dirty, np.zeros(dirty.shape), M=A.dot, tol=1e-7, maxit=100, verbosity=1)

    plt.figure('wiener')
    plt.imshow(x[0])
    plt.colorbar()


    # # filter with mexican hat wavelet
    # wavelet = pywt.ContinuousWavelet('mexh')
    # psi, g = wavelet.wavefun(25)
    # print(g)
    # print(psi)

    # plt.figure('mexh')
    # plt.plot(g, psi)




    plt.show()













    