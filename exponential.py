import numpy as np
np.random.seed(420)
from numpy.testing import assert_array_almost_equal
import scipy.linalg as scl
from scipy.optimize import fmin_l_bfgs_b as bfgs
from africanus.constants import c as lightspeed
from africanus.gps.kernels import exponential_squared as expsq
import matplotlib.pyplot as plt
from pfb.opt import pcg
from ducc0.wgridder import dirty2ms, ms2dirty
from ducc0.fft import r2c, c2r
iFs = np.fft.ifftshift
Fs = np.fft.fftshift
from pfb.operators import PSF, Prior
from astropy.io import fits
from pfb.utils import load_fits, save_fits, data_from_header, prox_21, str2bool, compare_headers

def main1D():
    # signal
    N = 128
    l = np.linspace(-0.5,0.5,N)
    cell = l[1]-l[0]
    K = expsq(l, l, 1.0, 0.05)
    L = np.linalg.cholesky(K + 1e-12*np.eye(N))
    Kinv = np.linalg.inv(K + 1e-12*np.eye(N))

    m = L.dot(np.random.randn(N))
    p = np.zeros(N, dtype=np.float64)
    p[10] = 1.0
    p[53] = 2.0
    p[88] = 0.5
    I = p # np.exp(m + p)

    # plt.figure()
    # plt.plot(l, m, 'b')
    # plt.plot(l, I, 'k')
    # plt.show()


    # log-normal data model
    nu = np.fft.fftfreq(N, d=cell)
    numax = np.abs(nu).max()
    M = 5000
    u = np.random.randn(M)
    umax = np.abs(u).max()
    u = 1.5 * numax*u/umax 

    R = np.exp(-2.0j*np.pi * np.outer(u, l))
    RH = R.conj().T
    Mop = RH @ R 
    
    # plt.figure('real')
    # plt.imshow(Mop.real)
    # plt.colorbar()

    # plt.figure('imag')
    # plt.imshow(Mop.imag)
    # plt.colorbar()

    # plt.show()

    # white noise
    noise = np.random.randn(M)/np.sqrt(2) + 1.0j*np.random.randn(M)/np.sqrt(2)
    model_data = R.dot(I)
    data = model_data + noise

    def hess1(x):
        return Mop.dot(x).real + x

    # # Hess operator
    # def hess2(x, s, Mop, Kinv):
    #     I = np.exp(s)
    #     return I[:, None] * Mop @ I[:, None] * x + Kinv @ x 

    # dirty image and psf
    dirty = RH.dot(data).real
    psf = RH.dot(np.ones(M, dtype=np.float64)).real

    # plt.figure('psf')
    # plt.plot(l, psf, 'k')
    # plt.show()

    # plt.figure()
    # plt.plot(l, dirty, 'b')
    # plt.plot(l, I, 'k')
    # plt.show()

    # get Wiener filter soln
    xhat1 = pcg(hess1, dirty, np.zeros(N, dtype=np.float64), M=lambda x:x, tol=1e-4)

    # get log-normal soln    
    def fprime(x, dirty, Mop, Kinv):
        I = np.exp(x)
        tmp1 = Mop.dot(I).real
        tmp2 = Kinv.dot(x)
        return np.vdot(I, tmp1 - 2*dirty) + np.vdot(x, tmp2), 2 * I * (tmp1 - dirty) + 2*tmp2

    x0 = np.zeros(N, dtype=np.float64) # np.where(xhat1 > 1e-15, np.log(xhat1), 0.0)
    xhat2, f, d = bfgs(fprime, x0, args=(dirty, Mop, np.eye(N)), approx_grad=False, factr=1e10)

    for key in d.keys():
        if key != 'grad':
            print(key, d[key])

    plt.figure('sol')
    plt.plot(l, xhat1, 'k')
    plt.plot(l, np.exp(xhat2), 'r')
    # plt.plot(l, np.exp(m + p), 'b')
    plt.plot(l, p, 'b')
    plt.show()

def R_func(x, uvw, freq, cell):
    return dirty2ms(uvw=uvw, freq=freq, dirty=x, 
                       pixsize_x=cell, pixsize_y=cell, 
                       epsilon=1e-10, nthreads=8, do_wstacking=False)

def RH_func(x, uvw, freq, cell, nx, ny):
    return ms2dirty(uvw=uvw, freq=freq, ms=x,
                       npix_x=nx, npix_y=ny,
                       pixsize_x=cell, pixsize_y=cell, 
                       epsilon=1e-10, nthreads=8, do_wstacking=False)

def main2D():
    # signal
    nx = 128
    ny = 128
    I = np.zeros((nx, ny), dtype=np.float64)
    npoints = 25
    indx = np.random.randint(5, nx-5, npoints)
    indy = np.random.randint(5, ny-5, npoints)
    I[indx, indy] = np.exp(np.random.randn(npoints))

    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(y, y)
    gshape = 2 * np.exp(-(xx - nx/2)**2/50 - (yy - ny/2)**2/50)

    I += gshape


    # log-normal data model
    M = 150000
    u = np.random.randn(M)
    v = np.random.randn(M)
    w = np.zeros(M, dtype=np.float64)

    uvw = np.vstack((u, v, w)).T

    freq = np.array([lightspeed])

    uvmax = np.maximum(np.abs(u).max(), np.abs(v).max())

    cell = 0.99/(2*uvmax)

    R = lambda x: R_func(x, uvw, freq, cell)
    RH = lambda x: RH_func(x, uvw, freq, cell, nx, ny)

    model_data = R(I)
    noise = np.random.randn(M, 1)/np.sqrt(2) + 1.0j*np.random.randn(M,1)/np.sqrt(2)
    data = model_data + 10*noise

    dirty = RH(data)
    noise_im = RH(noise)

    psf = ms2dirty(uvw=uvw, freq=freq, ms=np.ones((M, 1), dtype=np.complex128),
                      npix_x=2*nx, npix_y=2*ny,
                      pixsize_x=cell, pixsize_y=cell, 
                      epsilon=1e-10, nthreads=8, do_wstacking=False)

    # plt.figure('psf')
    # plt.imshow(psf)
    # plt.colorbar()

    plt.figure('dirty')
    plt.imshow(dirty)
    plt.colorbar()

    # plt.figure('noise')
    # plt.imshow(noise_im)
    # plt.colorbar()

    plt.figure('model')
    plt.imshow(I)
    plt.colorbar()
    # plt.show()


    # plt.show()
    # quit()


    nx_psf, ny_psf = psf.shape
    npad_x = (nx_psf - nx)//2
    npad_y = (ny_psf - ny)//2
    padding = ((npad_x, npad_x), (npad_y, npad_y))
    unpad_x = slice(npad_x, -npad_x)
    unpad_y = slice(npad_y, -npad_y)
    lastsize = ny + np.sum(padding[-1])
    ax = (0,1)
    psf_pad = iFs(psf, axes=ax)
    psfhat = r2c(psf_pad, axes=ax, forward=True, nthreads=8, inorm=0)

    def convolve(x):
        xhat = iFs(np.pad(x, padding, mode='constant'), axes=ax)
        xhat = r2c(xhat, axes=ax, nthreads=8, forward=True, inorm=0)
        xhat = c2r(xhat * psfhat, axes=ax, forward=False, lastsize=lastsize, inorm=2, nthreads=8)
        return Fs(xhat, axes=ax)[unpad_x, unpad_y] 

    def hess(x):
        return convolve(x) + 1000*x

    # get Wiener filter soln
    xhat1 = pcg(hess, dirty, np.zeros((nx, ny), dtype=np.float64), M=lambda x:x, tol=1e-8)

    plt.figure('wsoln')
    plt.imshow(xhat1)
    plt.colorbar()

    # get log-normal soln    
    def fprime(x):
        I = np.exp(x.reshape(nx, ny))
        tmp1 = convolve(I)
        tmp2 = x.reshape(nx, ny)
        return np.vdot(I, tmp1 - 2*dirty) + 1000*np.vdot(x, tmp2), (2 * I * (tmp1 - dirty) + 1000*2*tmp2).flatten()

    x0 = np.where(xhat1 > 1e-15, np.log(xhat1), 0.0)
    xhat2, f, d = bfgs(fprime, x0, approx_grad=False, factr=1e8)

    for key in d.keys():
        if key != 'grad':
            print(key, d[key])

    plt.figure('logsol')
    plt.imshow(np.exp(xhat2.reshape(nx, ny)))
    plt.colorbar()
    plt.show()

def main3D():
    dirty = load_fits('/home/landman/Data/pfb-testing/output/combined_image_dirty.fits')
    real_type = dirty.dtype
    hdr = fits.getheader('/home/landman/Data/pfb-testing/output/combined_image_dirty.fits')
    freq = data_from_header(hdr, axis=3)
    
    nchan, nx, ny = dirty.shape
    psf_array = load_fits('/home/landman/Data/pfb-testing/output/combined_image_psf.fits')
    hdr_psf = fits.getheader('/home/landman/Data/pfb-testing/output/combined_image_psf.fits')
    try:
        assert np.array_equal(freq, data_from_header(hdr_psf, axis=3))
    except:
        raise ValueError("Fits frequency axes dont match")
    
    psf_max = np.amax(psf_array.reshape(nchan, 4*nx*ny), axis=1)
    wsum = np.sum(psf_max)
    psf_max[psf_max < 1e-15] = 1e-15

    print("Image size = ", nchan, nx, ny)

    # plt.figure('psf')
    # plt.imshow(psf_array[0])
    # plt.colorbar()

    # plt.figure('dirty')
    # plt.imshow(dirty[0])
    # plt.colorbar()

    # plt.show()

    psf = PSF(psf_array, 8, 1.0)
    
    # x = np.zeros((nchan, nx, ny), dtype=np.float64)
    # x[:, nx//4:3*nx//4, ny//4:3*ny//4] = np.random.randn(nchan, nx//2, ny//2)

    # res = psf.hess(x)


    rec = pcg(psf.hess, dirty, np.zeros((nchan, nx, ny), dtype=np.float64), tol=1e-10, maxit=30)

    for i in range(2):
        plt.imshow(rec[i])
        plt.colorbar()
        plt.show()

    

    quit()
    
    K = Prior(0.0001, nchan, nx, ny, 8)

    # Tikhonov
    def hess(x):
        return psf.convolve(x) + K.iconvolve(x)

    # get Wiener filter soln
    xhat1 = pcg(hess, dirty, np.zeros((nchan, nx, ny), dtype=np.float64), M=K.convolve, tol=1e-3, maxit=30)

    x0 = np.where(xhat1 > 1e-5, np.log(xhat1), 1e-10)
    plt.figure('cgsol')
    plt.imshow(xhat1[0], vmax=0.05)
    plt.colorbar()

    # get log-normal soln    
    def fprime(x):
        I = np.exp(x.reshape(nchan, nx, ny))
        tmp1 = psf.convolve(I)
        tmp2 = K.iconvolve(x.reshape(nchan, nx, ny))
        return np.vdot(I, tmp1 - 2*dirty) + np.vdot(x, tmp2), (2 * I * (tmp1 - dirty) + 2*tmp2).flatten()

    
    xhat2, f, d = bfgs(fprime, x0, approx_grad=False, factr=1e12)

    for key in d.keys():
        if key != 'grad':
            print(key, d[key])

    plt.figure('logsol')
    plt.imshow(np.exp(xhat2.reshape(nchan, nx, ny)[0]), vmax=0.05)
    plt.colorbar()
    plt.show()


if __name__=='__main__':
    main3D()
    

