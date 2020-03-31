# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from pyrap.tables import table
from daskms import xds_from_ms
import dask
import dask.array as da
from opt import power_method, solve_x0, hpd
from scipy.fftpack import next_fast_len
from africanus.linalg import kronecker_tools as kt
from time import time
import argparse
from astropy.io import fits
from utils import test_adjoint, test_convolve, init_data, str2bool, robust_reweight
from operators import Gridder, PSF, Prior

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str, nargs='+')
    p.add_argument("--data_column", default="CORRECTED_DATA", type=str,
                   help="The column to image.")
    p.add_argument("--model_column", default="MODEL_DATA", type=str,
                   help="Column to write model visibilities to.")
    p.add_argument("--weight_column", default='WEIGHT_SPECTRUM', type=str,
                   help="Weight column to use. Will use WEIGHT if no WEIGHT_SPECTRUM is found")
    p.add_argument("--nx", type=int, default=None,
                   help="Number of x pixels.")
    p.add_argument("--ny", type=int, default=None,
                   help="Number of y pixels.")
    p.add_argument('--cell_size', type=float, default=None,
                   help="Cell size in arcseconds.")
    p.add_argument("--fov", type=float, default=None,
                   help="Field of view")
    p.add_argument("--ref_image", type=str, default=None,
                   help='A reference image to pinch header from.')
    p.add_argument("--outfile", type=str,
                   help='Directory in which to place the output.')
    p.add_argument("--outname", default='image', type=str,
                   help='base name of output.')
    p.add_argument("--ncpu", default=0, type=int,
                   help='Number of threads to use.')
    p.add_argument("--numba_threads", type=int, default=8,
                   help='Number of threads to use for numba parallelism.')
    p.add_argument('--precision', default=1e-7, type=float,
                   help='Precision of the gridder.')
    p.add_argument("--do_wstacking", type=str2bool, nargs='?', const=True, default=False,
                   help="Whether to do wide-field correction or not")
    p.add_argument("--channels_out", default=None, type=int,
                   help="Number of channels in output cube")
    p.add_argument("--nmiter", default=10, type=int,
                   help="Number of major cycles to run")
    p.add_argument("--sig_l21", default=0.1, type=float,
                   help="The strength of the l21 norm regulariser")
    p.add_argument("--sig_n", default=1.0, type=float,
                   help="The strength of the nuclear norm regulariser")
    p.add_argument("--make_restored", type=str2bool, nargs='?', const=True, default=False,
                   help="Add final update to model and compute residual based on that")
    p.add_argument("--cache_name", type=str, default='',
                   help="Will cache reduced data and final results to this file. "
                   "If empty string will auto-generate name using outfile and options")
    p.add_argument("--from_cache", type=str, default='',
                   help="Name of cache to restart from. "
                   "If empty string will use auto-genrated cache_name")
    p.add_argument("--method", type=str, default='FISTA',
                   help="Deconvolution method to use")
    p.add_argument("--positivity", type=str2bool, nargs='?', const=True, default=True,
                   help="Apply positivity constraint")
    p.add_argument("--continue_from", type=str, default='',
                   help='File containing results to continue from')
    p.add_argument("--mtol", type=float, default=1e-5, 
                   help="Major cycle tolerance")
    p.add_argument("--report_freq", type=int, default=2,
                   help="How often to save output images during deconvolution")
    p.add_argument("--cgtol", type=float, default=1e-4,
                   help="Tolerance for cg updates")
    p.add_argument("--cgmaxit", type=int, default=100,
                   help="Maximum number of iterations for the cg updates")
    p.add_argument("--pmtol", type=float, default=1e-14,
                   help="Tolerance for cg updates")
    p.add_argument("--pmmaxit", type=int, default=25,
                   help="Maximum number of iterations for the cg updates")
    p.add_argument("--row_chunks", type=int, default=100000,
                   help="Row chunking when loading in data")
    p.add_argument("--field", type=int, default=0,
                   help="Which field to image")
    p.add_argument("--verify_convolve", type=str2bool, nargs='?', const=True, default=False,
                   help="Apply positivity constraint")
    p.add_argument("--test_adjoint", type=str2bool, nargs='?', const=True, default=False,
                   help="Apply positivity constraint")
    p.add_argument("--super_resolution_factor", type=float, default=1.2,
                   help="Pixel sizes will be set to Nyquist divided by this factor")
    p.add_argument("--tidy", type=str2bool, nargs='?', const=True, default=False,
                   help="Use update operator in prox step")
    p.add_argument("--reweight0", type=str2bool, nargs='?', const=True, default=False,
                   help="Reweight after loading result cache")
    return p

def main(args, ddict):
    data = ddict['data']
    uvw = ddict['uvw']
    freq = ddict['freq']
    sqrtW = ddict['sqrtW']

    from africanus.constants import c as lightspeed
    u_max = np.abs(uvw[:, 0]).max()
    v_max = np.abs(uvw[:, 1]).max()
    
    uv_max = np.maximum(u_max, v_max)
    cell_N = 1.0/(2*uv_max*freq.max()/lightspeed)

    if args.cell_size is not None:
        cell_rad = args.cell_size * np.pi/60/60/180
        print("Super resolution factor = ", cell_N/cell_rad)
    else:
        cell_rad = cell_N/args.super_resolution_factor
        args.cell_size = cell_rad*60*60*180/np.pi
        print("Cell size set to %5.5e arcseconds" % args.cell_size)
    
    if args.nx is None:
        fov = args.fov*3600
        nx = int(fov/args.cell_size)
        from scipy.fftpack import next_fast_len
        args.nx = next_fast_len(nx)

    if args.ny is None:
        fov = args.fov*3600
        ny = int(fov/args.cell_size)
        from scipy.fftpack import next_fast_len
        args.ny = next_fast_len(ny)

    print("Image size set to (%i, %i, %i)"%(args.channels_out, args.nx, args.ny))

    # init gridder
    R = Gridder(uvw, freq, sqrtW, args)
    freq_out = R.freq_out
    if args.test_adjoint:
        print("Testing consistency of operator")
        test_adjoint(R)

    if args.ref_image is not None:
        hdr = fits.getheader(args.ref_image)
    else:
        hdu = fits.PrimaryHDU()
        hdr = hdu.header
    
    # check for cached dirty and psf
    GD = vars(args)
    dirty_hash = ''
    dirty_hash_options = ['nx', 'ny', 'cell_size', 'channels_out', 'do_wstacking', 'precision']
    for opt in dirty_hash_options:
        dirty_hash += '%s=%s_'%(opt, str(GD[opt]))
    dirty_hash = dirty_hash[0:-1]
    dirty_cache_name = args.outfile + dirty_hash + '.npz'
    try:
        pdict = np.load(dirty_cache_name)
        dirty = pdict['dirty']
        psf_array = pdict['psf']
        psf = PSF(psf_array, args.ncpu)
        nband = freq_out.size
        psf_max = np.amax(psf_array.reshape(nband, 4*args.nx*args.ny), axis=1)
        print("Found valid dirty and psf cache")
    except: 
        psf_array = R.make_psf()
        nband = R.nband
        psf_max = np.amax(psf_array.reshape(nband, 4*args.nx*args.ny), axis=1)
        psf = PSF(psf_array, args.ncpu)
    
        # make dirty (data hasn't been whitened)
        dirty = R.hdot(R.wgt * data)

        # save cache
        np.savez(dirty_cache_name, dirty=dirty, psf=psf_array)

        # save dirty and psf images
        hdu = fits.PrimaryHDU(header=hdr)
        hdu.data = np.transpose(dirty/psf_max[:, None, None], axes=(0, 2, 1))[:, ::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_dirty.fits', overwrite=True)


        psf_crop = np.ascontiguousarray(psf_array[:, args.nx//2:-(args.nx//2), args.ny//2:-(args.ny//2)])
        hdu = fits.PrimaryHDU(header=hdr)
        hdu.data = np.transpose(psf_crop/psf_max[:, None, None], axes=(0, 2, 1))[:, ::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_psf.fits', overwrite=True)
        del psf_crop

        if args.verify_convolve:
            print("Verifying convolution")
            test_convolve(R, psf, args)
    
    wsum = np.sum(psf_max) # to normalise mfs results
    
    # load in previous result
    result_hash = ''
    result_hash_options = ['nx', 'ny', 'cell_size', 'channels_out']
    for opt in result_hash_options:
        result_hash += '%s=%s_'%(opt, str(GD[opt]))
    result_hash = result_hash[0:-1]
    result_cache_name = args.outfile + result_hash + '.npz'
    try:
        rdict = np.load(result_cache_name)
        print("Found result cache at %s"%result_cache_name)
        model = rdict['model']
        residual = rdict['residual']
        L = rdict['L']
        # dual = rdict['dual']
        # weights_l1 = rdict['weights_l1']
        sigma0 = rdict['sigma0']
        
        # set prior
        l = 0.25 * (freq_out.max() - freq_out.min())/np.mean(freq_out)  # length scale in terms of fractional bandwidth
        print("Length scale set to ", l)
        K = Prior(freq_out, sigma0, l, args.nx, args.ny)
        def op(x):  # used for stabalised updates
            return K.sqrthdot(psf.convolve(K.sqrtdot(x)))
        print(" L = %5.5e "%L)
    except:
        print("Result cache does not exist or is invalid. Starting from scratch")
        # weights_l1 = np.zeros((nbasis, args.nx*args.ny), dtype=data.real.dtype)
        sigma0 = 1.0

        print("Getting x0")
        # set prior
        l = 0.25 * (freq_out.max() - freq_out.min())/np.mean(freq_out)  # length scale in terms of fractional bandwidth
        print("Length scale set to ", l)
        K = Prior(freq_out, sigma0, l, args.nx, args.ny)
        
        def op(x):  # used for stabalised updates
            return K.sqrthdot(psf.convolve(K.sqrtdot(x)))
        
        print("Getting spectral norm for update operator")
        L = power_method(op, dirty.shape, tol=args.pmtol, maxit=args.pmmaxit)
        print(" L = %5.5e "%L)
        soln, _, _ = solve_x0(op, K.sqrthdot(dirty), 
                              np.zeros_like(dirty), np.zeros_like(dirty), 
                              L, 1.0, maxit=2*args.cgmaxit, positivity=True)
        model = K.sqrtdot(soln)

        hdu = fits.PrimaryHDU(header=hdr)
        hdu.data = np.transpose(model, axes=(0, 2, 1))[:, ::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_first_soln.fits', overwrite=True)

        # get residual
        residual_vis = R.wgt*data - R.dot(model)
        residual = R.hdot(residual_vis)

        hdu = fits.PrimaryHDU(header=hdr)
        hdu.data = np.transpose(residual/psf_max[:, None, None], axes=(0, 2, 1))[:, ::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_first_residual.fits', overwrite=True)

    # set approx posterior metric
    if args.tidy:
        A = lambda x: psf.convolve(x) + K.idot(x)
        nuA = L
    else: 
        A = lambda x: x
        nuA = 1.0

    # deconvolve
    xi = np.zeros_like(dirty)
    yi = np.zeros_like(dirty)
    v21 = np.zeros((nband, args.nx*args.ny), dtype=np.float64)
    vn = np.zeros((nband, args.nx*args.ny), dtype=np.float64)
    rmax = np.abs(residual/psf_max[:, None, None]).max()
    eps = 1.0
    i = 0
    rms = np.std(residual/psf_max[:, None, None])
    v_dof = 5.0
    reweight_iters = []
    # reweight_alpha = 0.05
    # reweight_alpha_ff = 0.5
    print("At iteration 0 peak of residual is %f and rms is %f" % (rmax, rms))
    report_iters = list(np.arange(0, args.nmiter, args.report_freq))
    if report_iters[-1] != args.nmiter-1:
        report_iters.append(args.nmiter-1)

    while eps > args.mtol and i < args.nmiter:
        # get the (flat) Wiener filter soln
        xi, yi, L = solve_x0(op, K.sqrthdot(residual), xi, yi, L, 1.0, tol=args.cgtol, maxit=args.cgmaxit, positivity=False)
        x = K.sqrtdot(xi)
        # x = pcg(A, residual, x, M=K.dot, tol=1e-10, maxit=args.cgmaxit)
        
        # update model
        modelp = model
        model = modelp + x

        if args.tidy:
            nuA = L
        else:
            nuA = 1.0
            args.sig_l21 = 2*rms

        # compute prox
        model, v21, vn = hpd(A, model, 
                             modelp, v21, vn,  # initial guesses
                             nuA, 1.0, 1.0,    # spectral norms
                             args.sig_l21, args.sig_n,       # regularisers
                             1.0, 1.0,      # step sizes
                             1.0, 1.0, 1.0,  # extrapolation
                             tol=args.cgtol, maxit=args.cgmaxit, positivity=args.positivity) 

        # get residual
        residual_vis = R.wgt*data - R.dot(model)

        if i in reweight_iters:
            # # l1 reweighting 
            # weights_l21 = reweight_alpha / (reweight_alpha + np.abs(v21))

            # # nuclear norm reweighting
            # weights_l21 = reweight_alpha / (reweight_alpha + np.abs(vn))
            
            # reweight_alpha = reweight_alpha_ff * reweight_alpha
            
            # l2 reweighting compute new weights based on current residuals
            v_dof, wgt = robust_reweight(v_dof, residual_vis)

            print("v_dof = ", v_dof)

            # update gridder weigths (note sqrtW)
            R.wgt *= wgt

            # apply new sqrtW to residual visibilities
            residual_vis *= wgt   

            del wgt

            # compute new psf
            psf_array = R.make_psf()
            psf_max = np.amax(psf_array.reshape(nband, 4*args.nx*args.ny), axis=1)
            psf = PSF(psf_array, args.ncpu)

            def op(x):  # used for stabalised updates
                return K.sqrthdot(psf.convolve(K.sqrtdot(x)))

            print("Getting spectral norm for update operator")
            L = power_method(op, residual.shape, tol=args.pmtol, maxit=args.pmmaxit)
            print("L = %5.5e"%L)
       
        # residual image
        residual = R.hdot(residual_vis)
       
        # check stopping criteria
        rmax = np.abs(residual/psf_max[:, None, None]).max()
        rms = np.std(residual/psf_max[:, None, None])
        eps = np.linalg.norm(model - modelp)/np.linalg.norm(model)

        if i in report_iters:
            # save current iteration
            hdu = fits.PrimaryHDU(header=hdr)
            hdu.data = np.transpose(model, axes=(0, 2, 1))[:, ::-1].astype(np.float32)
            hdu.writeto(args.outfile + args.outname + str(i) + '.fits', overwrite=True)

            mfs_model = np.mean(model, axis=0)
            hdu = fits.PrimaryHDU(header=hdr)
            hdu.data = mfs_model.T[::-1].astype(np.float32)
            hdu.writeto(args.outfile + args.outname + str(i) + '_model_mfs.fits', overwrite=True)

            hdu = fits.PrimaryHDU(header=hdr)
            hdu.data = np.transpose(x, axes=(0, 2, 1))[:, ::-1].astype(np.float32)
            hdu.writeto(args.outfile + args.outname + str(i) + '_update.fits', overwrite=True)

            hdu = fits.PrimaryHDU(header=hdr)
            hdu.data = np.transpose(residual/psf_max[:, None, None], axes=(0, 2, 1))[:, ::-1].astype(np.float32)
            hdu.writeto(args.outfile + args.outname + str(i) + '_residual.fits', overwrite=True)

            mfs_residual = np.sum(residual, axis=0)/wsum
            hdu = fits.PrimaryHDU(header=hdr)
            hdu.data = mfs_residual.T[::-1].astype(np.float32)
            hdu.writeto(args.outfile + args.outname + str(i) + '_residual_mfs.fits', overwrite=True)

        i += 1
        print("At iteration %i peak of residual is %f, rms is %f, current eps is %f" % (i, rmax, rms, eps))

    # cache results so we can resume if needs be
    np.savez(result_cache_name, model=model, psf=psf_array, dirty=dirty, L=L, residual=residual, sigma0=sigma0, imaging_weights=R.wgt)

    if args.make_restored:
        # get the (flat) Wiener filter soln
        xi, yi, L = solve_x0(op, K.sqrthdot(residual), xi, yi, L, 1.0, 
                      positivity=False, tol=args.cgtol, maxit=args.cgmaxit)
        x = K.sqrtdot(xi)
        # x = pcg(A, residual, x, M=K.dot, tol=1e-10, maxit=args.cgmaxit)
        model += x
        # get residual
        residual_vis = R.wgt*data - R.dot(model)
        residual = R.hdot(residual_vis)  # negative of gradient term  

        rmax = np.abs(residual/psf_max[:, None, None]).max()
        rms = np.std(residual/psf_max[:, None, None])

        print("After restoring peak of residual is %f and rms is %f" % (rmax, rms))

        # save current iteration
        hdu = fits.PrimaryHDU(header=hdr)
        hdu.data = np.transpose(model, axes=(0, 2, 1))[:, ::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_restored.fits', overwrite=True)

        mfs_restored = np.mean(model, axis=0)
        hdu = fits.PrimaryHDU(header=hdr)
        hdu.data = mfs_restored.T[::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_restored_mfs.fits', overwrite=True)

        hdu = fits.PrimaryHDU(header=hdr)
        hdu.data = np.transpose(residual/psf_max[:, None, None], axes=(0, 2, 1))[:, ::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_restored_residual.fits', overwrite=True)

        mfs_residual = np.sum(residual, axis=0)/wsum
        hdu = fits.PrimaryHDU(header=hdr)
        hdu.data = mfs_residual.T[::-1].astype(np.float32)
        hdu.writeto(args.outfile + args.outname + '_restored_residual_mfs.fits', overwrite=True)


if __name__=="__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    if args.numba_threads:
        import os
        os.environ.update(NUMBA_NUM_THREADS = str(args.numba_threads))
    else:
        os.environ.update(NUMBA_NUM_THREADS = str(args.ncpu))

    if args.outfile[-1] != '/':
        args.outfile += '/'

    print("Using %i threads"%args.ncpu)

    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])

    # create hash and cache_name
    data_hash = ''
    data_hash_options = ['data_column', 'weight_column', 'channels_out']
    for opt in data_hash_options:
        data_hash += '%s=%s_'%(opt, str(GD[opt]))
    data_hash = data_hash[0:-1]
    data_hash = data_hash.replace('/', '.')
    
    if args.cache_name == '':
        cache_name = args.outfile + data_hash + '.npz'
    else:
        if args.cache_name[-4:] != '.npz':
            cache_name = args.cache_name + '.npz'
        else:    
            cache_name = args.cache_name
    
    # try load in cache if it exists
    try:
        ddict = np.load(cache_name)
        assert ddict['data_hash']==data_hash
        print("Successfully loaded data cache")
    except:
        print("Data cache is either invalid or does not exist. Starting from scratch")
        # initialise data
        ddict = init_data(args)
        ddict['data_hash'] = data_hash

        # save cache
        np.savez(cache_name, **ddict)
    finally:
        main(args, ddict)