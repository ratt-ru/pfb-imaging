import numpy as np
import xarray as xr
from distributed import get_client, worker_client, wait
from pfb.opt.pcg import pcg
from pfb.operators.hessian import _hessian_impl
from pfb.operators.psf import psf_convolve_slice
from pfb.utils.weighting import (_compute_counts,
                                 _counts_to_weights,
                                 _filter_extreme_counts)
from uuid import uuid4
import pywt
from pfb.wavelets.wavelets_jsk import  (get_buffer_size,
                                       dwt2d, idwt2d)
from ducc0.misc import make_noncritical
import concurrent.futures as cf
from africanus.constants import c as lightspeed
from ducc0.wgridder.experimental import vis2dirty, dirty2vis
from ducc0.fft import c2r, r2c, good_size
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

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

# this keeps track of dataset label
def counts_to_weights(counts, uvw, freq, nx, ny,
                      cell_size_x, cell_size_y,
                      robust, i):
    return i, _counts_to_weights(counts, uvw, freq, nx, ny,
                                 cell_size_x, cell_size_y,
                                 robust)

class band_actor(object):
    def __init__(self, ds_names, opts, bandid, cache_path):
        self.opts = opts
        self.bandid = bandid
        self.cache_path = cache_path + f'/band{bandid}'
        self.ds_names = ds_names

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
        wsum = 0.0
        for i, ds in enumerate(dsl):
            wgt = ds.WEIGHT.values
            mask = ds.MASK.values
            wsumt = (wgt*mask).sum()
            wsum += wsumt
            beam += ds.BEAM.values * wsumt
            ds = ds.drop_vars('BEAM')
            dsl[i] = ds.drop_dims(('l_beam', 'm_beam'))

        beam /= wsum

        # now there should only be a single one
        self.ds = xr.concat(dsl, dim='row')
        self.real_type = ds.WEIGHT.dtype
        self.complex_type = ds.VIS.dtype
        self.freq_out = np.mean(self.freq)
        self.time_out = np.mean(times_in)

        self.cell_rad = np.deg2rad(opts.cell/3600)
        self.nx = opts.nx
        self.ny = opts.ny  # also lastsize now
        self.nyo2 = ny//2 + 1
        self.nhthreads = opts.nthreads_dask
        self.nvthreads = opts.nvthreads

        # uv-cell counts
        self.counts = _compute_counts(self.ds.UVW.values,
                                      self.freq,
                                      self.ds.MASK.values,
                                      self.nx,
                                      self.ny,
                                      self.cell_rad,
                                      self.cell_rad,
                                      np.float64,  # same type as uvw
                                      self.ds.WEIGHT.values,
                                      ngrid=self.nvthreads)

        self.counts = np.sum(self.counts, axis=0)

        assert self.counts.shape[0] == self.nx
        assert self.counts.shape[1] == self.ny



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

        # pre-allocate tmp arrays required for convolution
        # make_noncritical for efficiency of FFT
        self.xout = make_noncritical(np.zeros((self.nx, self.ny),
                                              dtype=self.real_type))
        self.xhat = make_noncritical(np.zeros((self.nx_psf, self.nyo2),
                                              dtype=self.complex_type))
        self.xpad = make_noncritical(np.zeros((self.nx_psf, self.ny_psf),
                                              dtype=self.real_type))

    def get_image_info(self):
        return (self.nx, self.ny, self.nmax, self.cell_rad,
                self.ra[0], self.dec[0], self.x0[0], self.y0[0],
                self.freq_out, np.mean(self.times_out))


    def set_image_data_products(self, model, dual, dof=None, from_cache=False):
        '''
        This initialises the psf and residual.
        Also used to perform L2 reweighting so wsum can change.
        This is why we can't normalise by wsum.
        Returns the average per-band psf and resid so that we can
        compute the MFS images (and hence wsum, rms and rmax etc.)
        on the runner.
        '''
        # per band model and dual
        self.model = model
        self.dual = dual

        if from_cache:
            self.psfs = [1] * self.ntimes
            self.psfhats = [1] * self.ntimes
            self.resids = [1] * self.ntimes
            self.beams = [1] * self.ntimes
            for i in range(self.ntimes):
                cname = self.cache_path + f'_time{i}.zarr'
                ds = xr.open_zarr(cname, chunks=None)
                self.resids[i] = ds.RESID.values
                self.psfs[i] = ds.PSF.values
                self.psfhats[i] = ds.PSFHAT.values
                self.wgt[i] = ds.WGT.values
                self.beams[i] = ds.BEAM.values
            # we return the per band psf since we need the mfs psf
            # on the runner
            return np.sum(self.psfs, axis=0), np.sum(self.resids, axis=0)

        with cf.ThreadPoolExecutor(max_workers=self.nhthreads) as executor:
            futures = []
            for i, (uvw, imwgt, wgt, vis, mask, x0, y0) in enumerate(zip(
                                                    self.uvw, self.imwgt,
                                                    self.wgt, self.vis,
                                                    self.mask, self.x0,
                                                    self.y0)):

                fut = executor.submit(image_data_products,
                                      model,
                                      uvw,
                                      self.freq,
                                      vis,
                                      imwgt,
                                      wgt,
                                      mask,
                                      self.nx, self.ny,
                                      self.nx_psf, self.ny_psf,
                                      self.cell_rad, self.cell_rad,
                                      dof=dof,
                                      x0=x0, y0=y0,
                                      nthreads=self.nvthreads,
                                      epsilon=self.opts.epsilon,
                                      do_wgridding=self.opts.do_wgridding,
                                      double_accum=self.opts.double_accum,
                                      timeid=i)

                futures.append(fut)

            self.psfs = [1] * self.ntimes
            self.psfhats = [1] * self.ntimes
            self.resids = [1] * self.ntimes
            # self.beams = [1] * self.ntimes
            for fut in cf.as_completed(futures):
                i, residual, psf, psfhat, wgt = fut.result()
                self.resids[i] = residual
                self.psfs[i] = psf
                self.psfhats[i] = psfhat
                self.wgt[i] = wgt
                # cache vars affected by major cycle
                data_vars = {
                    'RESID': (('nx','ny'), residual),
                    'PSF': (('nx_psf','ny_psf'), psf),
                    'PSFHAT': (('nx_psf', 'nyo2'), psfhat),
                    'WGT': (('row', 'chan'), wgt)
                }
                attrs = {
                    'robustness': self.opts.robustness,
                    'time_out': self.times_out[i]

                }
                dset = xr.Dataset(data_vars, attrs=attrs)
                cname = self.cache_path + f'_time{i}.zarr'
                dset.to_zarr(cname, mode='a')

                # update natural weights if they have changed
                # we can do this here if not concatenating otherwise
                # do in separate loop
                # needed if we have done L2 reweighting and we are
                # resuming a run with a different robustness value
                if dof and not self.opts.concat:
                    ds = xr.open_zarr(self.ds_names[i])
                    ds['WEIGHT'] = wgt
                    ds.to_zarr(self.ds_names[i], mode='a')


        # TODO - parallel over time chunks
        # update natural weights if they have changed
        # we need a separate loop when concatenating
        # needed if we have done L2 reweighting and we are
        # resuming a run with a different robustness value
        if dof and self.opts.concat:
            wgt = self.wgt[0]  # only one of these
            rowi = 0
            for i, dsn in enumerate(self.ds_names):
                ds = xr.open_zarr(dsn, chunks=None)
                nrow = ds.WEIGHT.shape[0]
                ds['WEIGHT'] = (('row', 'chan'), wgt[rowi:rowi + nrow])
                ds.to_zarr(dsn, mode='a')
                rowi += nrow

        # we return the per band psf since we need the mfs psf
        # on the runner
        # TODO - we don't actually need to store the psfs on the worker
        # only psfhats
        return np.sum(self.psfs, axis=0), np.sum(self.resids, axis=0)


    def set_wsum_and_data(self, wsum):
        self.wsum = wsum
        if wsum > 0:
            resid = np.sum(self.resids, axis=0)
            self.data = resid/wsum + self.psf_conv(self.model)
            return 0
        else:
            self.data = np.zeros_like(self.model)
            return 1 # use to check?


    def set_residual(self, x=None):
        if x is None:
            x = self.model

        if self.wsumb == 0:
            self.data = np.zeros_like(x)
            return np.zeros_like(x)

        with cf.ThreadPoolExecutor(max_workers=self.nhthreads) as executor:
            futures = []
            for i, (uvw, imwgt, wgt, vis, mask, x0, y0) in enumerate(zip(
                                                    self.uvw, self.imwgt,
                                                    self.wgt, self.vis,
                                                    self.mask, self.x0,
                                                    self.y0)):

                fut = executor.submit(residual_from_vis,
                                      x,
                                      uvw,
                                      self.freq,
                                      vis,
                                      imwgt,
                                      wgt,
                                      mask,
                                      self.nx, self.ny,
                                      self.cell_rad, self.cell_rad,
                                      x0=x0, y0=y0,
                                      nthreads=self.nvthreads,
                                      epsilon=self.opts.epsilon,
                                      do_wgridding=self.opts.do_wgridding,
                                      double_accum=self.opts.double_accum,
                                      timeid=i)

                futures.append(fut)

            for fut in cf.as_completed(futures):
                i, residual = fut.result()
                self.resids[i] = residual
                # cache vars affected by major cycle
                data_vars = {
                    'RESID': (('nx','ny'), residual),
                }
                attrs = {
                    'time_out': self.times_out[i],

                }
                dset = xr.Dataset(data_vars, attrs=attrs)
                cname = self.cache_path + f'_time{i}.zarr'
                dset.to_zarr(cname, mode='r+')

        # we can do this here because wsum doesn't change
        resid = np.sum(self.resids, axis=0)
        self.resid = resid/self.wsum
        self.data = self.resid + self.psf_conv(x)

        # return residual since we need the MFS residual on the runner
        return resid

    def psf_conv(self, x):
        convx = np.zeros_like(x)
        if self.wsumb == 0:
            return convx

        for psfhat in self.psfhats:
            convx += psf_convolve_slice(self.xpad,
                                        self.xhat,
                                        self.xout,
                                        psfhat,
                                        self.lastsize,
                                        x,
                                        nthreads=self.nvthreads)

        return convx/self.wsum


    def hess_psf(self, x):
        return self.psf_conv(x) + self.sigmasq * x


    def hess_wgt(self, x):
        convx = np.zeros_like(x)
        if self.wsumb == 0:
            return convx

        for uvw, imwgt, mask, beam, x0, y0 in zip(self.uvw,
                                                  self.imwgt,
                                                  self.mask,
                                                  self.beam,
                                                  self.x0,
                                                  self.y0):
            convx += _hessian_impl(x,
                                   uvw,
                                   imwgt,
                                   mask,
                                   self.freq,
                                   beam,
                                   x0=x0,
                                   y0=y0,
                                   cell=self.cell_rad,
                                   do_wgridding=self.opts.do_wgridding,
                                   epsilon=self.opts.epsilon,
                                   double_accum=self.opts.double_accum,
                                   nthreads=self.nvthreads)

        return convx/self.wsum + self.sigmasq * x


    def almost_grad(self, x):
        return self.psf_conv(x) - self.data

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

    def init_pd_params(self, hessnorm, nu, gamma=1):
        self.hessnorm = hessnorm
        self.sigma = hessnorm/(2*gamma*nu)
        self.tau = 0.9 / (hessnorm / (2.0 * gamma) + self.sigma * nu**2)
        self.vtilde = self.dual + self.sigma * self.psi_dot(self.model)
        return self.vtilde

    def pd_update(self, ratio):
        # ratio - (nbasis, nmax)
        self.xp[...] = self.model[...]
        self.vp[...] = self.dual[...]

        self.dual[...] = self.vtilde * (1 - ratio)

        grad = self.almost_grad(self.xp)
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
        self.b[...] = self.psf_conv(self.bp)
        bsumsq = np.sum(self.b**2)
        beta_num = np.vdot(self.b, self.bp)
        beta_den = np.vdot(self.bp, self.bp)

        return bsumsq, beta_num, beta_den

    def give_model(self):
        return self.model, self.dual, self.bandid

    def cg_update(self, x, approx='psf'):
        if approx == 'psf':
            hess = self.hess_psf
        elif approx == 'wgt':
            hess = self.hess_wgt
        else:
            raise ValueError(f'Unknown approx method {approx}')

        x0 = np.zeros((nx, ny), dtype=float)
        update = pcg(hess,
                     self.resid,
                     x0,
                     tol=self.opts.cg_tol,
                     maxit=self.opts.cg_maxit,
                     minit=self.opts.cg_minit,
                     verbosity=0)

        return update


def image_data_products(model,
                        uvw,
                        freq,
                        vis,
                        imwgt,
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
        wgt=wgt*imwgt,
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
        # LB - what is wrong with this?
        # n = np.sqrt(1 - x0**2 - y0**2)
        # if convention.upper() == 'CASA':
        #     freqfactor = -2j*np.pi*freq[None, :]/lightspeed
        # else:
        #     freqfactor = 2j*np.pi*freq[None, :]/lightspeed
        # psf_vis = np.exp(freqfactor*(uvw[:, 0:1]*x0 +
        #                              uvw[:, 1:2]*y0 +
        #                              uvw[:, 2:]*(n-1)))
        # if divide_by_n:
        #     psf_vis /= n
        x = np.zeros((128,128), dtype=wgt.dtype)
        x[64,64] = 1.0
        psf_vis = dirty2vis(
            uvw=uvw,
            freq=freq,
            dirty=x,
            pixsize_x=cellx,
            pixsize_y=celly,
            center_x=x0,
            center_y=y0,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            nthreads=nthreads,
            divide_by_n=False,
            flip_v=False,  # hardcoded for now
            sigma_min=1.1, sigma_max=3.0)

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
        wgt=wgt*imwgt,
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
    psfhat = r2c(iFs(psf, axes=(0, 1)), axes=(0, 1),
                    nthreads=nthreads,
                    forward=True, inorm=0)

    return timeid, dirty, psf, psfhat, wgt

def residual_from_vis(
                model,
                uvw,
                freq,
                vis,
                imwgt,
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
        wgt=wgt*imwgt,
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


    return timeid, residual
