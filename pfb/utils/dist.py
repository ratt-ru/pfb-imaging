import inspect
import numpy as np
import numba
import xarray as xr
from pfb.opt.pcg import pcg
from pfb.operators.gridder import image_data_products, compute_residual
from pfb.operators.hessian import _hessian_impl
from pfb.operators.psf import psf_convolve_slice
from pfb.utils.weighting import (compute_counts,
                                 counts_to_weights,
                                 filter_extreme_counts)
from pfb.utils.misc import taperf, set_image_size
from pfb.utils.naming import xds_from_url, xds_from_list
from uuid import uuid4
import pywt
from pfb.wavelets.wavelets_jsk import  (get_buffer_size,
                                       dwt2d, idwt2d)
from ducc0.misc import empty_noncritical, resize_thread_pool
import concurrent.futures as cf
from africanus.constants import c as lightspeed
from ducc0.wgridder import vis2dirty, dirty2vis
from ducc0.fft import c2r, r2c, c2c, good_size
iFs = np.fft.ifftshift
Fs = np.fft.fftshift


class fake_future(object):
    '''
    This is just a class that has a fake result method.
    Useful for testing and when not running in distributed mode.
    '''
    def __init__(self, r):
        self.r = r

    def result(self):
        return self.r

# class fake_actor(object):
#     def __init__(self, r):
#         self.r = r

#     def result(self):
#         import ipdb; ipdb.set_trace()
#         return self.r

class fake_client(object):
    '''
    This is just a class that has a fake submit method.
    Useful for testing and when not running in distributed mode.
    '''
    def __init__(self, nworkers=1, nvthreads=1):
        # TODO - use cf to fake these for horizontal parallelism
        self.nworkers = nworkers
        self.nvthreads = nvthreads

    def submit(self, *args, **kwargs):
        func = args[0]  # by convention
        params = inspect.signature(func).parameters
        fkw = {}
        for name, param in params.items():
            if name in kwargs:
                fkw[name] = kwargs[name]
        res = func(*args[1:], **fkw)
        # if 'actor' in kwargs and kwargs['actor']:  # return instantiated class
        #     return fake_actor(res)
        # else:  # return fake future with result
        return fake_future(res)

    def gather(self, futures):
        return [f.result() for f in futures]


def l1reweight_func(actors, rmsfactor, alpha=4):
    '''
    The logic here is that weights should remain the same for model
    components that are rmsfactor times larger than the rms of the updates.
    High SNR values should experience relatively small thresholding
    whereas small values should be strongly thresholded
    '''
    # project model
    futures = list(map(lambda a: a.psi_dot(mode='model'), actors))
    outvar = np.sum(list(map(lambda f: f.result(), futures)), axis=0)
    mcomps = np.abs(outvar)
    # project updates
    futures = list(map(lambda a: a.psi_dot(mode='update'), actors))
    outvar = np.sum(list(map(lambda f: f.result(), futures)), axis=0)
    if not outvar.any():
        raise ValueError("Cannot reweight before any updates have been performed")
    rms_comps = np.std(outvar)
    # the **alpha here results in more agressive reweighting
    return (1 + rmsfactor)/(1 + mcomps**alpha/rms_comps**alpha)


# the context manager should allow both vertical and horizontal scaling
# within a single python process
def dwt(x, buffer, dec_lo, dec_hi, nlevel, nthreads, i):
    if nthreads > 1:
        with numba.config.THREADING_LAYER.override('omp'), numba.config.NUMBA_NUM_THREADS.override(nthreads):
            dwt2d(x, buffer, dec_lo, dec_hi, nlevel)
    else:
        dwt2d(x, buffer, dec_lo, dec_hi, nlevel)
    return buffer, i

def idwt(buffer, rec_lo, rec_hi, nlevel, nx, ny, nthreads, i):
    image = np.zeros((nx, ny), dtype=buffer.dtype)
    if nthreads > 1:
        with numba.config.THREADING_LAYER.override('omp'), numba.config.NUMBA_NUM_THREADS.override(nthreads):
            idwt2d(buffer, image, rec_lo, rec_hi, nlevel)
    else:
        idwt2d(buffer, image, rec_lo, rec_hi, nlevel)
    return image, i

class band_actor(object):
    def __init__(self, xds_list, opts, bandid, cache_path, max_freq, uv_max):
        self.opts = opts
        self.bandid = bandid
        self.cache_path = f'{cache_path}/time0000_band{bandid:04d}.zarr'
        self.xds_list = xds_list
        self.nthreads = opts.nthreads
        self.real_type = np.float64
        self.complex_type = np.complex128
        # hess_approx determines whether we use the vis of image space
        # approximation of the hessian.
        self.hess_approx = opts.hess_approx
        resize_thread_pool(self.nthreads)

        dsl = xds_from_list(xds_list, drop_vars=['VIS', 'BEAM'],
                            nthreads=self.nthreads)

        times_in = []
        self.freq = dsl[0].FREQ.values
        for ds in dsl:
            times_in.append(ds.time_out)
            if (ds.FREQ.values != self.freq).all():
                raise NotImplementedError("Freqs per band currently assumed to be the same")

        times_in = np.unique(times_in)

        # drop before concat
        for i, ds in enumerate(dsl):
            ds = ds.drop_vars('FREQ')
            dsl[i] = ds.drop_dims(('l_beam', 'm_beam'))

        # concat and check if band is empty
        xds = xr.concat(dsl, dim='row')
        wsumb = np.sum(xds.WEIGHT.values*xds.MASK.values)
        if wsumb == 0:
            raise ValueError('Got empty dataset. This is a bug!')

        # set cell sizes
        self.max_freq = max_freq
        self.uv_max = uv_max
        nx, ny, nx_psf, ny_psf, cell_N, cell_rad = set_image_size(uv_max,
                                                                  max_freq,
                                                                  opts)
        cell_deg = np.rad2deg(cell_rad)
        cell_size = cell_deg * 3600
        # print(f"Super resolution factor = {cell_N/cell_rad}", file=log)
        # print(f"Cell size set to {cell_size:.5e} arcseconds", file=log)
        # print(f"Field of view is ({nx*cell_deg:.3e},{ny*cell_deg:.3e}) degrees",
        #     file=log)

        self.cell_rad = cell_rad
        self.nx = nx
        self.ny = ny
        self.nx_psf = nx_psf
        self.ny_psf = ny_psf
        if nx_psf > nx or ny_psf > ny:
            npad_xl = (nx_psf - nx) // 2
            npad_xr = nx_psf - nx - npad_xl
            npad_yl = (ny_psf - ny) // 2
            npad_yr = ny_psf - ny - npad_yl
            self.padding = ((npad_xl, npad_xr), (npad_yl, npad_yr))
            self.unpad_x = slice(npad_xl, -npad_xr)
            self.unpad_y = slice(npad_yl, -npad_yr)
        else:
            self.padding = ((0, 0), (0, 0))
            self.unpad_x = slice(None)
            self.unpad_y = slice(None)

        self.ra = dsl[0].ra
        self.dec = dsl[0].dec
        self.x0 = 0.0
        self.y0 = 0.0

        self.freq_out = np.mean(self.freq)
        self.time_out = np.mean(times_in)

        # set up wavelet dictionaries
        self.bases = opts.bases.split(',')
        self.nbasis = len(self.bases)
        # do the wavelet transform in parallel over basis
        if 'self' in self.bases:
            self.nhthreads = np.minimum(self.nthreads, self.nbasis-1)
        else:
            self.nhthreads = np.minimum(self.nthreads, self.nbasis)
        print(f"Will process {self.nhthreads} bases in parallel")
        # use additional threads to prange each basis
        self.nvthreads = np.maximum(1, self.nthreads//self.nhthreads)
        numba.set_num_threads(self.nvthreads)
        print(f"numba is using {numba.get_num_threads()} vertical threads")
        self.nlevel = opts.nlevels
        self.dec_lo ={}
        self.dec_hi ={}
        self.rec_lo ={}
        self.rec_hi ={}
        self.buffer_size = {}
        self.buffer = {}
        self.nmax = 0
        for wavelet in self.bases:
            if wavelet=='self':
                self.buffer_size[wavelet] = self.nx*self.ny
                self.nmax = np.maximum(self.nmax, self.buffer_size[wavelet])
                continue
            # Get the required filter banks from pywt.
            wvlt = pywt.Wavelet(wavelet)
            self.dec_lo[wavelet] = np.array(wvlt.filter_bank[0])  # Low pass, decomposition.
            self.dec_hi[wavelet] = np.array(wvlt.filter_bank[1])  # Hi pass, decomposition.
            self.rec_lo[wavelet] = np.array(wvlt.filter_bank[2])  # Low pass, recon.
            self.rec_hi[wavelet] = np.array(wvlt.filter_bank[3])  # Hi pass, recon.

            self.buffer_size[wavelet] = get_buffer_size((self.nx, self.ny),
                                                        self.dec_lo[wavelet].size,
                                                        self.nlevel)
            self.buffer[wavelet] = np.zeros(self.buffer_size[wavelet],
                                            dtype=np.float64)

            self.nmax = np.maximum(self.nmax, self.buffer_size[wavelet])

        # tmp optimisation vars
        self.b = np.zeros((self.nx, self.ny), dtype=self.real_type)
        self.bp = np.zeros((self.nx, self.ny), dtype=self.real_type)
        self.xp = np.zeros((self.nx, self.ny), dtype=self.real_type)
        self.vp = np.zeros((self.nbasis, self.nmax), dtype=self.real_type)
        self.vtilde = np.zeros((self.nbasis, self.nmax), dtype=self.real_type)

        # always initialised to zero
        self.dual = np.zeros((self.nbasis, self.nmax), dtype=self.real_type)

        # this will be overwritten if loading from cache
        # important for L1-reweighting!
        self.update = np.zeros((self.nx, self.ny), dtype=self.real_type)

        # pre-allocate tmp array required for hess_psf
        # make_noncritical for efficiency of FFT
        self.xhat = empty_noncritical((self.nx_psf, self.ny_psf),
                                       dtype=self.complex_type)

        self.sigmainvsq = opts.sigmainvsq
        if opts.hess_approx=='wgt':
            self.hess = self.hess_wgt
        elif opts.hess_approx=='psf':
            self.hess = self.hess_psf
        elif opts.hess_approx=='direct':
            self.hess = self.hess_direct
            if self.sigmainvsq < 1:
                print(f'Warning - using dangerously low value of '
                      f'{self.sigmainvsq} with direct Hessian approximation')
        npix = np.maximum(self.nx, self.ny)
        self.taperxy = taperf((self.nx, self.ny), int(0.1*npix))

        self.attrs = {
            'ra': self.ra,
            'dec': self.dec,
            'x0': self.x0,
            'y0': self.y0,
            'cell_rad': self.cell_rad,
            'bandid': self.bandid,
            'timeid': '0000',
            'freq_out': self.freq_out,
            'time_out': self.time_out,
            'robustness': self.opts.robustness,
            'super_resolution_factor': self.opts.super_resolution_factor,
            'field_of_view': self.opts.field_of_view,
            'product': self.opts.product
        }

    def get_image_info(self):
        return (self.nx, self.ny, self.nmax, self.cell_rad,
                self.ra, self.dec, self.x0, self.y0,
                self.freq_out, self.time_out)


    def set_image_data_products(self, model, k, dof=None, from_cache=False):
        '''
        This initialises the psf and residual.
        Also used to perform L2 reweighting so wsum can change.
        This is why we can't normalise by wsum.
        Returns the per-band resid and wsum so that we can
        compute the MFS images (and hence wsum, rms and rmax etc.)
        on the runner.
        '''

        self.model = model

        if from_cache:
            self.dds = xr.open_zarr(self.cache_path, chunks=None)
            self.update = self.dds.UPDATE.values
            residual = self.dds.RESIDUAL.values
            self.wsumb = self.dds.wsum
        else:
            # attempt to save memory
            if hasattr(self, 'dds'):
                self.dds = None
            residual, wsumb = image_data_products(
                                self.xds_list,
                                None,  # counts
                                self.nx, self.ny,
                                self.nx_psf, self.ny_psf,
                                self.cell_rad, self.cell_rad,
                                self.cache_path,
                                self.attrs,
                                model=model,
                                robustness=self.opts.robustness,
                                x0=self.x0, y0=self.y0,
                                nthreads=self.nthreads,
                                epsilon=self.opts.epsilon,
                                do_wgridding=self.opts.do_wgridding,
                                double_accum=self.opts.double_accum,
                                # divide_by_n=False,
                                l2reweight_dof=dof,
                                do_dirty=True,
                                do_psf=True,
                                do_residual=True,
                                do_weight=True,
                                do_noise=False)

            self.wsumb = wsumb
            # cache vars affected by major cycle
            # we cache the weights in the
            data_vars = {
                'UPDATE': (('x','y'), self.update)
            }
            self.attrs['niters'] = k
            self.attrs['wsum'] = wsumb
            dds = xr.Dataset(data_vars, attrs=self.attrs)
            dds.to_zarr(self.cache_path, mode='a')

            # these are read in each major cycle anyway
            drop_vars = ['PSF','MODEL','DIRTY']
            if self.opts.hess_approx != 'wgt':
                drop_vars += ['WEIGHT','UVW','MASK']
            self.dds = xds_from_list([self.cache_path],
                                     drop_vars=drop_vars)[0]

        # return the per band resid since resid MFS required on runner
        return residual, self.wsumb

    def set_wsum(self, wsum):
        self.wsum = wsum  # wsum over all bands
        # we can only normalise once we have wsum from all bands
        # only required if wsum changes
        self.psfhat = self.dds.PSFHAT.values/wsum
        self.residual = self.dds.RESIDUAL.values/wsum
        if self.opts.hess_approx == 'direct':
            self.psfhat = np.abs(self.psfhat)


    def set_residual(self, k, x=None):
        if x is None:
            x = self.model

        if self.wsumb == 0:
            self.data = np.zeros_like(x)
            return np.zeros_like(x)

        residual = compute_residual(
                                self.cache_path,
                                self.nx, self.ny,
                                self.cell_rad, self.cell_rad,
                                self.cache_path,  # output_name (same as dsl names?)
                                x,
                                x0=self.x0, y0=self.y0,
                                nthreads=self.nthreads,
                                epsilon=self.opts.epsilon,
                                do_wgridding=self.opts.do_wgridding,
                                double_accum=self.opts.double_accum)

        # is this even necessary?
        data_vars = {
            'UPDATE': (('x','y'), self.update)
        }
        self.attrs['niters'] = k
        dds = xr.Dataset(data_vars, attrs=self.attrs)
        dds.to_zarr(self.cache_path, mode='a')

        # these are read in each major cycle anyway
        drop_vars = ['PSF','MODEL','DIRTY']
        if self.opts.hess_approx != 'wgt':
            drop_vars += ['WEIGHT','UVW','MASK']
        self.dds = xds_from_list([self.cache_path],
                                    drop_vars=drop_vars)[0]

        # wsum doesn't change so we can normalise
        self.residual = residual/self.wsum

        # return residual since we need the MFS residual on the runner
        return residual

    def hess_direct(self, x, mode='forward', sigma=None):
        if sigma is None:
            sigma = self.sigmainvsq
        if self.wsumb == 0:  # should not happen
            return np.zeros((self.nx, self.ny), dtype=self.real_type)
        self.xhat[...] = np.pad(x*self.taperxy, self.padding, mode='constant')
        c2c(self.xhat, out=self.xhat,
            forward=True, inorm=0, nthreads=self.nthreads)
        if mode=='forward':
            self.xhat *= (self.psfhat + sigma)
        elif mode=='backward':
            self.xhat /= (self.psfhat + sigma)
        c2c(self.xhat, out=self.xhat,
            forward=False, inorm=2, nthreads=self.nthreads)
        # TODO - do we need the copy here?
        # c2c out must not overlap input
        return self.xhat[self.unpad_x, self.unpad_y].real * self.taperxy

    def hess_psf(self, x, mode='forward'):
        raise NotImplementedError('Not yet')


    def hess_wgt(self, x, mode='forward'):
        convx = np.zeros_like(x)
        if self.wsumb == 0:
            return convx

        if mode=='forward':
            convx += _hessian_impl(x,
                                self.dds.UVW.values,
                                self.dds.WEIGHT.values,
                                self.dds.MASK.values,
                                self.freq,
                                None, # self.beam
                                x0=0.0, y0=0.0,
                                cell=self.cell_rad,
                                do_wgridding=self.opts.do_wgridding,
                                epsilon=5e-4,  # we don't need much accuracy here
                                double_accum=self.opts.double_accum,
                                nthreads=self.nthreads)

            return convx/self.wsum + self.sigmainvsq * x
        else:
            raise NotImplementedError('Not yet')

    def cg_update(self):
        if self.hess_approx == 'wgt':
            # precond = lambda x: self.hess_direct(x,
            #                                      mode='backward',
            #                                      sigma=1.0)
            x = pcg(
                A=self.hess_wgt,
                b=self.residual,
                # x0=precond(self.residual),
                # M=precond,
                tol=self.opts.cg_tol,
                maxit=self.opts.cg_maxit,
                minit=self.opts.cg_minit,
                verbosity=self.opts.cg_verbose,
                report_freq=self.opts.cg_report_freq,
                backtrack=self.opts.backtrack,
                return_resid=False
            )
        elif self.hess_approx == "psf":
            # the information source differs because we need a Hermitian
            # postive definite operator for the pcg

            raise NotImplementedError
        elif self.hess_approx == "direct":
            x = self.hess_direct(self.residual, mode='backward')
        else:
            raise ValueError(f'Unknown hess-approx {self.hess_approx}')

        self.update = x
        self.xtilde = self.model + x

        return x

    def backward_grad(self, x):
        '''
        The gradient function during the backward step
        '''
        return -self.hess(self.xtilde - x, mode='forward')

    def psi_dot(self, x=None, mode='model'):
        '''
        signal to coeffs

        Input:
            x       - (nx, ny) input signal

        Output:
            alpha   - (nbasis, Nnmax) output coeffs

        If x==None we apply psi_dot to the update.
        Used for L1-reweighting
        '''
        # TODO - this is a bit of a kludge that we need
        # for performing L1 reweights
        if x is None:
            if mode == 'model':
                x = self.model
            else:
                x = self.update
        alpha = np.zeros((self.nbasis, self.nmax),
                         dtype=x.dtype)

        with cf.ThreadPoolExecutor(max_workers=self.nhthreads) as executor:
            futures = []
            for i, wavelet in enumerate(self.bases):
                if wavelet=='self':
                    alpha[i, 0:self.buffer_size[wavelet]] = x.ravel()
                    continue

                f = executor.submit(dwt, x,
                                    self.buffer[wavelet],
                                    self.dec_lo[wavelet],
                                    self.dec_hi[wavelet],
                                    self.nlevel,
                                    self.nvthreads,
                                    i)

                futures.append(f)

            for f in cf.as_completed(futures):
                buffer, i = f.result()
                alpha[i, 0:self.buffer_size[self.bases[i]]] = buffer

        return alpha

    def psi_hdot(self, alpha):
        '''
        coeffs to signal

        Input:
            alpha   - (nbasis, Nxmax, Nymax) per basis output coeffs

        Output:
            x       - (nx, ny) output signal
        '''
        nx = self.nx
        ny = self.ny
        x = np.zeros((nx, ny), dtype=alpha.dtype)

        with cf.ThreadPoolExecutor(max_workers=self.nhthreads) as executor:
            futures = []
            for i, wavelet in enumerate(self.bases):
                nmax = self.buffer_size[wavelet]
                if wavelet=='self':
                    x += alpha[i, 0:nmax].reshape(nx, ny)
                    continue

                f = executor.submit(idwt,
                                    alpha[i, 0:nmax],
                                    self.rec_lo[wavelet],
                                    self.rec_hi[wavelet],
                                    self.nlevel,
                                    self.nx,
                                    self.ny,
                                    self.nvthreads,
                                    i)


                futures.append(f)

            for f in cf.as_completed(futures):
                image, i = f.result()
                x += image

        return x

    def init_pd_params(self, hess_norm, nu, gamma=1):
        self.hess_norm = hess_norm
        self.sigma = hess_norm/(2*gamma*nu)
        self.tau = 0.9 / (hess_norm / (2.0 * gamma) + self.sigma * nu**2)
        self.vtilde = self.dual + self.sigma * self.psi_dot(x=self.model)
        return self.vtilde

    def pd_update(self, ratio):
        # ratio - (nbasis, nmax)
        self.xp[...] = self.model[...]
        self.vp[...] = self.dual[...]

        self.dual[...] = self.vtilde * (1 - ratio)

        grad = self.backward_grad(self.xp)
        self.model[...] = self.xp - self.tau * (self.psi_hdot(2*self.dual - self.vp) + grad)

        if self.opts.positivity:
            self.model[self.model < 0] = 0.0

        self.vtilde[...] = self.dual + self.sigma * self.psi_dot(x=self.model)

        eps_num = np.sum((self.model-self.xp)**2)
        eps_den = np.sum(self.model**2)

        # vtilde - (nband, nbasis, nmax)
        return self.vtilde, eps_num, eps_den


    def init_random(self):
        self.b = np.random.randn(self.nx, self.ny)
        return np.sum(self.b**2)


    def pm_update(self, bnorm):
        self.bp[...] = self.b/bnorm
        self.b[...] = self.hess(self.bp)
        bsumsq = np.sum(self.b**2)
        beta_num = np.vdot(self.b, self.bp)
        beta_den = np.vdot(self.bp, self.bp)

        return bsumsq, beta_num, beta_den

    def give_model(self):
        return self.model, self.bandid
