import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.fftpack import next_fast_len
from pyrap.tables import table
from astropy.io import fits
from africanus.constants import c as lightspeed
from pfb.opt import fista, power_method
from pfb.operators import Gridder, Prior, PSF

ms_name = "test_data/point_gauss_nb.MS_p0"
result_name = 'ref_images/fista_minor.npz'

def gen_ref_images():
    np.random.seed(420)
    ms = table(ms_name)
    uvw = ms.getcol('UVW').astype(np.float64)
    data = ms.getcol('DATA')[:, :, 0].astype(np.complex128)  # Stokes I
    ms.close()

    nrow, nchan = data.shape

    data += np.random.randn(nrow, nchan)/np.sqrt(2) + 1.0j*np.random.randn(nrow, nchan)/np.sqrt(2)

    # set weights
    weight = np.ones((nrow, nchan), dtype=data.real.dtype)
    sqrtW = np.sqrt(weight)

    tbl = table(ms_name + "::SPECTRAL_WINDOW")
    freq = tbl.getcol("CHAN_FREQ").squeeze()
    tbl.close()

    u_max = np.abs(uvw[:, 0]).max()
    v_max = np.abs(uvw[:, 1]).max()
    
    uv_max = np.maximum(u_max, v_max)
    cell_N = 1.0/(2*uv_max*freq.max()/lightspeed)
    super_resolution_factor = 1.2
    cell = cell_N/super_resolution_factor
    cell_size = cell*60*60*180/np.pi
    print("Cell size set to %5.5e arcseconds" % cell_size)
    
    fov_d = 1.5
    fov = fov_d*3600
    nx = int(fov/cell_size)
    nx = next_fast_len(nx)
    ny = nx

    nband = nchan

    print("Image size set to (%i, %i, %i)"%(nband, nx, ny))
    
    
    R = Gridder(uvw, freq, sqrtW, nx, ny, cell_size, nband=None, precision=1e-7, ncpu=8, do_wstacking=1)    
    imsize = (R.nchan, R.nx, R.ny)
    nchan, nx, ny = imsize

    hdu = fits.PrimaryHDU()
    hdr = hdu.header

    # save dirty and psf images
    dirty = R.hdot(data)
    psf_array = R.make_psf()
    

    psf = PSF(psf_array, 6)
    K = Prior(freq, 1.0, 0.25, R.nx, R.ny, nthreads=6)

    def op(x):
        return psf.convolve(x) + K.idot(x)

    beta = power_method(op, imsize, tol=1e-15, maxit=25)
    # beta = 214341236.1895035
    print("beta = ", beta)

    # fidelity and gradient term
    def fprime(x):
        diff = psf.convolve(x) - dirty
        tmp = K.idot(x)
        return 0.5*np.vdot(x, diff).real - 0.5*np.vdot(x, dirty) + 0.5*np.vdot(x, tmp), diff + tmp

    x0 = np.zeros(imsize, dtype=np.float64)
    y0 = np.zeros(imsize, dtype=np.float64)
    maxit = 100
    model, objhist, fidhist, fidupper = fista(fprime, x0, y0, beta, 0.01, 1e-5, maxit=maxit, positivity=True)

    residual_vis = data - R.dot(model)
    residual = R.hdot(residual_vis)

    np.savez(result_name, dirty=dirty, psf=psf_array, model=model, residual=residual)

def run_test():
    # get ref images
    ref_images = np.load(result_name)

    np.random.seed(420)
    ms = table(ms_name)
    uvw = ms.getcol('UVW').astype(np.float64)
    data = ms.getcol('DATA')[:, :, 0].astype(np.complex128)  # Stokes I
    ms.close()

    nrow, nchan = data.shape
    # add noise so weight are unity
    data += np.random.randn(nrow, nchan)/np.sqrt(2) + 1.0j*np.random.randn(nrow, nchan)/np.sqrt(2)

    # set weights
    weight = np.ones((nrow, nchan), dtype=data.real.dtype)
    sqrtW = np.sqrt(weight)

    tbl = table(ms_name + "::SPECTRAL_WINDOW")
    freq = tbl.getcol("CHAN_FREQ").squeeze()
    tbl.close()

    u_max = np.abs(uvw[:, 0]).max()
    v_max = np.abs(uvw[:, 1]).max()
    
    uv_max = np.maximum(u_max, v_max)
    cell_N = 1.0/(2*uv_max*freq.max()/lightspeed)
    super_resolution_factor = 1.2
    cell = cell_N/super_resolution_factor
    cell_size = cell*60*60*180/np.pi
    print("Cell size set to %5.5e arcseconds" % cell_size)
    
    fov_d = 1.5
    fov = fov_d*3600
    nx = int(fov/cell_size)
    nx = next_fast_len(nx)
    ny = nx

    nband = nchan

    print("Image size set to (%i, %i, %i)"%(nband, nx, ny))
    
    
    R = Gridder(uvw, freq, sqrtW, nx, ny, cell_size)    
    imsize = (R.nchan, R.nx, R.ny)
    nchan, nx, ny = imsize

    # save dirty and psf images
    dirty = R.hdot(data)
    psf_array = R.make_psf()
    
    assert_array_almost_equal(dirty, ref_images['dirty'])
    assert_array_almost_equal(psf_array, ref_images['psf'])

    psf = PSF(psf_array, 6)
    K = Prior(freq, 1.0, 0.25, R.nx, R.ny, nthreads=6)

    def op(x):
        return psf.convolve(x) + K.idot(x)

    beta = power_method(op, imsize, tol=1e-15, maxit=25)
    # beta = 214341236.1895035
    print("beta = ", beta)

    # fidelity and gradient term
    def fprime(x):
        diff = psf.convolve(x) - dirty
        tmp = K.idot(x)
        return 0.5*np.vdot(x, diff).real - 0.5*np.vdot(x, dirty) + 0.5*np.vdot(x, tmp), diff + tmp

    x0 = np.zeros(imsize, dtype=np.float64)
    y0 = np.zeros(imsize, dtype=np.float64)
    maxit = 100
    model, objhist, fidhist, fidupper = fista(fprime, x0, y0, beta, 0.01, 1e-5, maxit=maxit, positivity=True)

    residual_vis = data - R.dot(model)
    residual = R.hdot(residual_vis)

    assert_array_almost_equal(model, ref_images['model'])
    assert_array_almost_equal(residual, ref_images['residual'])