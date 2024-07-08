import inspect
import numpy as np
import xarray as xr
from distributed import get_client, worker_client, wait
from pfb.opt.pcg import pcg
from pfb.operators.hessian import _hessian_impl
from pfb.operators.psf import psf_convolve_slice
from pfb.utils.weighting import (compute_counts,
                                 counts_to_weights,
                                 filter_extreme_counts)
from uuid import uuid4
import pywt
from pfb.wavelets.wavelets_jsk import  (get_buffer_size,
                                       dwt2d, idwt2d)
from ducc0.misc import make_noncritical
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
        if 'actor' in kwargs and kwargs['actor']:  # return instantiated class
            return res
        else:  # return fake future with result
            return fake_future(res)

    def wait(self, futures):
        return

    def gather(self, futures):
        return [f.result() for f in futures]


def l1reweight_func(actors, rmsfactor, rms_comps, alpha=4):
    '''
    The logic here is that weights should remain the same for model
    components that are rmsfactor times larger than the rms.
    High SNR values should experience relatively small thresholding
    whereas small values should be strongly thresholded
    '''
    futures = list(map(lambda a: a.psi_dot(), actors))
    outvar = list(map(lambda f: f.result(), futures))
    mcomps = np.abs(np.sum(outvar, axis=0))
    # the **alpha here results in more agressive reweighting
    return (1 + rmsfactor)/(1 + mcomps**alpha/rms_comps**alpha)


def dwt(x, buffer, dec_lo, dec_hi, nlevel, i):
    dwt2d(x, buffer, dec_lo, dec_hi, nlevel)
    return buffer, i


def idwt(buffer, rec_lo, rec_hi, nlevel, nx, ny, i):
    image = np.zeros((nx, ny), dtype=buffer.dtype)
    idwt2d(buffer, image, rec_lo, rec_hi, nlevel)
    return image, i

class band_actor(object):
    def __init__(self, ds_names, opts, bandid, cache_path):
        self.opts = opts
        self.bandid = bandid
        self.cache_path = f'{cache_path}/time0000_band{bandid:04d}.zarr'
        self.ds_names = ds_names
        self.nhthreads = opts.nthreads_dask
        self.nvthreads = opts.nvthreads
        self.real_type = np.float64
        self.complex_type = np.complex128
        # hess_approx determines whether we use the vis of image space
        # approximation of the hessian.
        self.hess_approx = opts.hess_approx

        dsl = []
        for dsn in ds_names:
            dsl.append(xr.open_zarr(dsn, chunks=None))

        times_in = []
        self.freq = dsl[0].FREQ.values
        for ds in dsl:
            times_in.append(ds.time_out)
            if (ds.FREQ.values != self.freq).all():
                raise NotImplementedError("Freqs per band currently assumed to be the same")

        times_in = np.unique(times_in)

        # need to take weighted sum of beam before concat
        beam = np.zeros((ds.l_beam.size, ds.m_beam.size), dtype=float)
        self.wsumb = 0.0  # wsum for the band
        # we need to drop these before concat (needed for beam interpolation)
        l_beam = dsl[0].l_beam
        m_beam = dsl[0].m_beam
        for i, ds in enumerate(dsl):
            wgt = ds.WEIGHT.values
            mask = ds.MASK.values
            wsumt = (wgt*mask).sum()
            self.wsumb += wsumt
            beam += ds.BEAM.values * wsumt
            ds = ds.drop_vars('BEAM')
            dsl[i] = ds.drop_dims(('l_beam', 'm_beam'))

        beam /= self.wsumb

        # set cell sizes
        self.max_freq = dsl[0].max_freq
        self.uv_max = dsl[0].uv_max
        # Nyquist cell size
        cell_N = 1.0 / (2 * self.uv_max * self.max_freq / lightspeed)

        if opts.cell_size is not None:
            cell_size = opts.cell_size
            cell_rad = cell_size * np.pi / 60 / 60 / 180
            if cell_N / cell_rad < 1:
                raise ValueError("Requested cell size too large. "
                                "Super resolution factor = ", cell_N / cell_rad)
        else:
            cell_rad = cell_N / opts.super_resolution_factor
            cell_size = cell_rad * 60 * 60 * 180 / np.pi

        if opts.nx is None:
            fov = opts.field_of_view * 3600
            npix = int(fov / cell_size)
            npix = good_size(npix)
            while npix % 2:
                npix += 1
                npix = good_size(npix)
            nx = npix
            ny = npix
        else:
            nx = opts.nx
            ny = opts.ny if opts.ny is not None else nx
            cell_deg = np.rad2deg(cell_rad)
            fovx = nx*cell_deg
            fovy = ny*cell_deg

        nx_psf = good_size(int(opts.psf_oversize * nx))
        while nx_psf % 2:
            nx_psf += 1
            nx_psf = good_size(nx_psf)

        ny_psf = good_size(int(opts.psf_oversize * ny))
        while ny_psf % 2:
            ny_psf += 1
            ny_psf = good_size(ny_psf)


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

        # TODO - interpolate beam to size of image
        # interpo = RegularGridInterpolator((l_beam, m_beam), beam,
        #                                   bounds_error=True, method='linear')
        self.beam = np.ones((nx, ny), dtype=self.real_type)

        self.ra = dsl[0].ra
        self.dec = dsl[0].dec
        self.x0 = 0.0
        self.y0 = 0.0

        # now there should only be a single one
        self.xds = xr.concat(dsl, dim='row')
        self.freq_out = np.mean(self.freq)
        self.time_out = np.mean(times_in)

        self.nhthreads = opts.nthreads_dask
        self.nvthreads = opts.nvthreads

        # set up wavelet dictionaries
        self.bases = opts.bases.split(',')
        self.nbasis = len(self.bases)
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

        # pre-allocate tmp array required for hess_psf
        # make_noncritical for efficiency of FFT
        self.xhat = make_noncritical(np.zeros((self.nx_psf, self.ny_psf),
                                              dtype=self.complex_type))

        # TODO - tune using evidence and/or psf sse?
        self.sigma = 1.0  # used in hess_psf for backward
        self.sigmasqinv = 1e-6

    def get_image_info(self):
        return (self.nx, self.ny, self.nmax, self.cell_rad,
                self.ra, self.dec, self.x0, self.y0,
                self.freq_out, self.time_out, self.wsumb)


    def set_image_data_products(self, model, dof=None, from_cache=False):
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
        else:
            residual, psfhat, wgt = image_data_products(
                                model,
                                self.xds.UVW.values,
                                self.freq,
                                self.xds.VIS.values,
                                self.xds.WEIGHT.values,
                                self.xds.MASK.values,
                                self.nx, self.ny,
                                self.nx_psf, self.ny_psf,
                                self.cell_rad, self.cell_rad,
                                dof=dof,
                                x0=self.x0, y0=self.y0,
                                nthreads=self.nvthreads,
                                epsilon=self.opts.epsilon,
                                do_wgridding=self.opts.do_wgridding,
                                double_accum=self.opts.double_accum,
                                timeid=0)

            self.wsumb = np.sum(wgt*self.xds.MASK.values)
            # cache vars affected by major cycle
            # we cache the weights in the
            data_vars = {
                'RESIDUAL': (('x','y'), residual),
                'PSFHAT': (('x_psf', 'y_psf'), np.abs(psfhat)),
                'WGT': (('row', 'chan'), wgt)
            }
            attrs = {
                'time_out': self.time_out,
                'wsumb': self.wsumb,

            }
            self.dds = xr.Dataset(data_vars, attrs=attrs)
            self.dds.to_zarr(self.cache_path, mode='a')

            # TODO - we still have two copies of the weights


        # return the per band resid since resid MFS required on runner
        return self.dds.RESIDUAL.values, self.wsumb

    def set_wsum(self, wsum):
        self.wsum = wsum  # wsum over all bands
        # we can only normalise once we have wsum from all bands
        self.dds['PSFHAT'] /= wsum
        self.dds['RESIDUAL'] /= wsum
        # are these updated on disk?


    def set_residual(self, x=None):
        if x is None:
            x = self.model

        if self.wsumb == 0:
            self.data = np.zeros_like(x)
            return np.zeros_like(x)

        residual = residual_from_vis(
                                x,
                                self.xds.UVW.values,
                                self.freq,
                                self.xds.VIS.values,
                                self.dds.WGT.values, # may have been reweighted
                                self.xds.MASK.values,
                                self.nx, self.ny,
                                self.cell_rad, self.cell_rad,
                                x0=self.x0, y0=self.y0,
                                nthreads=self.nvthreads,
                                epsilon=self.opts.epsilon,
                                do_wgridding=self.opts.do_wgridding,
                                double_accum=self.opts.double_accum,
                                timeid=0)

        # wsum doesn't change so we can normalise
        self.dds = self.dds.assign(
            **{'RESIDUAL': (('x','y'), residual/self.wsum)}
        )

        # create a dataset to write to disk
        # only update RESIDUAL not all vars
        data_vars = {
            'RESIDUAL': (('x','y'), residual/self.wsum),
        }
        dset = xr.Dataset(data_vars)
        dset.to_zarr(self.cache_path, mode='r+')

        # return residual since we need the MFS residual on the runner
        return residual


    def hess_psf(self, x, mode='forward'):
        if self.wsumb == 0:
            return np.zeros((self.nx, self.ny), dtype=self.real_type)
        self.xhat[...] = np.pad(x, self.padding, mode='constant')
        c2c(Fs(self.xhat, axes=(0,1)), out=self.xhat,
            forward=True, inorm=0, nthreads=self.nvthreads)
        if mode=='forward':
            self.xhat *= (self.dds.PSFHAT.values + self.sigmasqinv)
        elif mode=='backward':
            self.xhat /= (self.dds.PSFHAT.values + 1/self.sigma)
        c2c(self.xhat, out=self.xhat,
            forward=False, inorm=2, nthreads=self.nvthreads)
        # TODO - do we need the copy here?
        # c2c out must not overlap input
        return iFs(self.xhat, axes=(0,1))[self.unpad_x, self.unpad_y].copy().real


    def hess_wgt(self, x):
        convx = np.zeros_like(x)
        if self.wsumb == 0:
            return convx

        convx += _hessian_impl(x,
                               self.xds.UVW.values,
                               self.dds.WGT.values,
                               self.xds.MASK.values,
                               self.freq,
                               None, # self.beam
                               x0=0.0, y0=0.0,
                               cell=self.cell_rad,
                               do_wgridding=self.opts.do_wgridding,
                               epsilon=5e-4,  # we don't need much accuracy here
                               double_accum=self.opts.double_accum,
                               nthreads=self.nvthreads)

        return convx/self.wsum + self.sigmasqinv * x


    def backward_grad(self, x):
        '''
        The gradient function during the backward step
        '''
        # return -self.hess_psf(self.xtilde - x,
        #                       mode='forward')
        return -self.hess_wgt(self.xtilde - x)

    def psi_dot(self, x=None):
        '''
        signal to coeffs

        Input:
            x       - (nx, ny) input signal

        Output:
            alpha   - (nbasis, Nnmax) output coeffs
        '''
        if x is None:
            x = self.model
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
        self.vtilde = self.dual + self.sigma * self.psi_dot(self.model)
        return self.vtilde

    def cg_update(self):
        precond = lambda x: self.hess_psf(x,
                                          mode='backward')
        if self.hess_approx == 'wgt':
            x = pcg(
                A=self.hess_wgt,
                b=self.dds.RESIDUAL.values,
                x0=precond(self.dds.RESIDUAL.values),
                M=precond,
                tol=self.opts.cg_tol,
                maxit=self.opts.cg_maxit,
                minit=self.opts.cg_minit,
                verbosity=self.opts.cg_verbose,
                report_freq=self.opts.cg_report_freq,
                backtrack=self.opts.backtrack,
                return_resid=False
            )
        elif self.hess_approx == "psf":
            x = precond(self.dds.RESIDUAL.values)

        self.update = x
        self.xtilde = self.model + x

        return x

    def pd_update(self, ratio):
        # ratio - (nbasis, nmax)
        self.xp[...] = self.model[...]
        self.vp[...] = self.dual[...]

        self.dual[...] = self.vtilde * (1 - ratio)

        grad = self.backward_grad(self.xp)
        self.model[...] = self.xp - self.tau * (self.psi_hdot(2*self.dual - self.vp) + grad)

        if self.opts.positivity:
            self.model[self.model < 0] = 0.0

        self.vtilde[...] = self.dual + self.sigma * self.psi_dot(self.model)

        eps_num = np.sum((self.model-self.xp)**2)
        eps_den = np.sum(self.model**2)

        # vtilde - (nband, nbasis, nmax)
        return self.vtilde, eps_num, eps_den


    def init_random(self):
        self.b = np.random.randn(self.nx, self.ny)
        return np.sum(self.b**2)


    def pm_update(self, bnorm):
        self.bp[...] = self.b/bnorm
        if self.hess_approx == "wgt":
            self.b[...] = self.hess_wgt(self.bp)
        else:
            self.b[...] = self.hess_psf(self.bp, mode='forward')
        bsumsq = np.sum(self.b**2)
        beta_num = np.vdot(self.b, self.bp)
        beta_den = np.vdot(self.bp, self.bp)

        return bsumsq, beta_num, beta_den

    def give_model(self):
        return self.model, self.bandid


def image_data_products(model,
                        uvw,
                        freq,
                        vis,
                        wgt,
                        mask,
                        nx, ny,
                        nx_psf, ny_psf,
                        cellx, celly,
                        dof=None,
                        x0=0.0, y0=0.0,
                        nthreads=4,
                        epsilon=1e-7,
                        do_wgridding=True,
                        double_accum=True,
                        timeid=0):
    '''
    Function to compute image space data products in one go
        dirty
        psf
        psfhat
    '''
    if np.isinf(model).any() or np.isnan(model).any():
        raise ValueError('Model contains infs or nans')

    if model.any():
        # don't apply weights in this direction
        # residual_vis = vis.copy()
        residual_vis = dirty2vis(
                uvw=uvw,
                freq=freq,
                dirty=model,
                pixsize_x=cellx,
                pixsize_y=celly,
                center_x=x0,
                center_y=y0,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                flip_v=False,
                nthreads=nthreads,
                divide_by_n=False,
                sigma_min=1.1, sigma_max=3.0)
        # this should save us 1 vis sized array
        residual_vis *= -1 # negate model
        residual_vis += vis  # add vis
    else:
        residual_vis = vis

    if dof is not None:
        # apply mask to both
        residual_vis *= mask
        ressq = (residual_vis*residual_vis.conj()).real
        wcount = mask.sum()
        if wcount:
            ovar = ressq.sum()/wcount  # use 67% quantile?
            wgt = (dof + 1)/(dof + ressq/ovar)

    dirty = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=residual_vis,
        wgt=wgt,
        mask=mask,
        npix_x=nx, npix_y=ny,
        pixsize_x=cellx, pixsize_y=celly,
        center_x=x0, center_y=y0,
        epsilon=epsilon,
        flip_v=False,  # hardcoded for now
        do_wgridding=do_wgridding,
        divide_by_n=False,  # hardcoded for now
        nthreads=nthreads,
        sigma_min=1.1, sigma_max=3.0,
        double_precision_accumulation=double_accum)

    if x0 or y0:
        # -w is the wgridder convention
        n = np.sqrt(1 - x0**2 - y0**2)
        freqfactor = 2j*np.pi*freq[None, :]/lightspeed
        psf_vis = np.exp(freqfactor*(uvw[:, 0:1]*x0 +
                                     uvw[:, 1:2]*y0 -
                                     uvw[:, 2:]*(n-1)))
        if divide_by_n:
            psf_vis /= n

    else:
        nrow, _ = uvw.shape
        nchan = freq.size
        tmp = np.ones((1,), dtype=vis.dtype)
        # should be tiny
        psf_vis = np.broadcast_to(tmp, vis.shape)

    psf = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=psf_vis,
        wgt=wgt,
        mask=mask,
        npix_x=nx_psf, npix_y=ny_psf,
        pixsize_x=cellx, pixsize_y=celly,
        center_x=x0, center_y=y0,
        epsilon=epsilon,
        flip_v=False,  # hardcoded for now
        do_wgridding=do_wgridding,
        divide_by_n=False,  # hardcoded for now
        nthreads=nthreads,
        sigma_min=1.1, sigma_max=3.0,
        double_precision_accumulation=double_accum)

    # get FT of psf
    # we don't actually need the PSF anymore?
    psf = psf.astype(vis.dtype)
    c2c(Fs(psf, axes=(0, 1)), axes=(0, 1), out=psf,
        nthreads=nthreads,
        forward=True, inorm=0)

    return dirty, np.abs(psf), wgt

def residual_from_vis(
                model,
                uvw,
                freq,
                vis,
                wgt,
                mask,
                nx, ny,
                cellx, celly,
                x0=0.0, y0=0.0,
                nthreads=1,
                epsilon=1e-7,
                do_wgridding=True,
                double_accum=True,
                timeid=0):
    '''
    This is used for major cycles when we don't change the weights.
    '''

    if np.isinf(model).any() or np.isnan(model).any():
        raise ValueError('Model contains infs or nans')

    # don't apply weights in this direction
    # residual_vis = vis.copy()
    residual_vis = dirty2vis(
            uvw=uvw,
            freq=freq,
            dirty=model,
            pixsize_x=cellx,
            pixsize_y=celly,
            center_x=x0,
            center_y=y0,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            flip_v=False,
            nthreads=nthreads,
            divide_by_n=False,
            sigma_min=1.1, sigma_max=3.0)

    # this should save us 1 vis sized array
    # negate model
    residual_vis *= -1
    # add data
    residual_vis += vis

    residual = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=residual_vis,
        wgt=wgt,
        mask=mask,
        npix_x=nx, npix_y=ny,
        pixsize_x=cellx, pixsize_y=celly,
        center_x=x0, center_y=y0,
        epsilon=epsilon,
        flip_v=False,  # hardcoded for now
        do_wgridding=do_wgridding,
        divide_by_n=False,  # hardcoded for now
        nthreads=nthreads,
        sigma_min=1.1, sigma_max=3.0,
        double_precision_accumulation=double_accum)


    return residual
