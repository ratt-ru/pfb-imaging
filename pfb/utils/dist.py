import numpy as np
import xarray as xr
from distributed import get_client, worker_client, wait
from pfb.utils.misc import fitcleanbeam
from pfb.operators.hessian import _hessian_impl
from pfb.operators.psf import psf_convolve_slice
from pfb.utils.weighting import (_compute_counts, _counts_to_weights,
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

class grad_actor(object):
    def __init__(self, ds_names, opts, bandid, cache_path):
        self.opts = opts
        self.bandid = bandid
        self.cache_path = cache_path + f'/band{bandid}'
        self.ds_names = ds_names

        dsl = []
        for dsn in ds_names:
            dsl.append(xr.open_zarr(dsn, chunks=None))

        # TODO - make sure these are consistent across datasets
        self.freq = dsl[0].chan.values
        self.freq_out = np.mean(self.freq)
        self.max_freq = dsl[0].max_freq
        self.uv_max = dsl[0].uv_max

        # beam = [ds.BEAM for ds in dsl]  # TODO
        wgt = [ds.WEIGHT for ds in dsl]
        vis = [ds.VIS for ds in dsl]
        mask = [ds.MASK for ds in dsl]
        uvw = [ds.UVW for ds in dsl]
        times_out = [ds.time_out for ds in dsl]

        # TODO - weighted sum of beam needs to be summed
        if opts.concat:
            self.wgt = [xr.concat(wgt, dim='row').values]
            self.vis = [xr.concat(vis, dim='row').values]
            self.mask = [xr.concat(mask, dim='row').values]
            self.uvw = [xr.concat(uvw, dim='row').values]
            self.times_out = [np.mean(times_out)]
        else:
            self.wgt = list(map(lambda d: d.values, wgt))
            self.vis = list(map(lambda d: d.values, vis))
            self.mask = list(map(lambda d: d.values, mask))
            self.uvw = list(map(lambda d: d.values, uvw))
            self.times_out = times_out

        self.real_type = self.wgt[0].dtype
        self.complex_type = self.vis[0].dtype
        self.ntimes = len(self.times_out)

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
        self.nyo2 = ny_psf//2 + 1
        self.nhthreads = opts.nthreads_dask
        self.nvthreads = opts.nvthreads
        self.lastsize = ny_psf

        # uv-cell counts
        with cf.ThreadPoolExecutor(max_workers=self.nhthreads) as executor:
            futures = []
            for uvw, mask in zip(self.uvw, self.mask):
                fut = executor.submit(_compute_counts,
                                      uvw,
                                      self.freq,
                                      mask,
                                      nx,
                                      ny,
                                      cell_rad,
                                      cell_rad,
                                      np.float64,  # same type as uvw
                                      ngrid=self.nvthreads)
                futures.append(fut)
            self.counts = []
            for fut in cf.as_completed(futures):
                self.counts.append(fut.result().sum(axis=0))

        # TODO - should we always sum all the counts in a band?
        self.counts = np.sum(self.counts, axis=0)

        # we can't compute imaging weights yet because we need
        # to communicate counts when using MF weighting

        # compute lm coordinates of target
        if opts.target is not None:
            if len(self.times_out) > 1:
                raise NotImplementedError("Off center targets are not yet "
                                          "supported for multiple output "
                                          "times.")
            tmp = opts.target.split(',')
            if len(tmp) == 1 and tmp[0] == opts.target:
                obs_time = self.times_out[0]
                self.ra, self.dec = get_coordinates(obs_time, target=opts.target)
            else:  # we assume a HH:MM:SS,DD:MM:SS format has been passed in
                from astropy import units as u
                from astropy.coordinates import SkyCoord
                c = SkyCoord(tmp[0], tmp[1], frame='fk5', unit=(u.hourangle, u.deg))
                ra = np.deg2rad(c.ra.value)
                dec = np.deg2rad(c.dec.value)

            tcoords=np.zeros((1,2))
            tcoords[0,0] = ra
            tcoords[0,1] = dec
            coords0 = np.array((ds.ra, ds.dec))
            lm0 = radec_to_lm(tcoords, coords0).squeeze()
            # LB - why the negative?
            self.x0 = [-lm0[0]]
            self.y0 = [-lm0[1]]
            self.ra = [ra]
            self.dec = [dec]
        else:
            self.x0 = [0.0] * self.ntimes
            self.y0 = [0.0] * self.ntimes
            self.ra = [ds.ra for ds in dsl]
            self.dec = [ds.dec for ds in dsl]

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


    def get_counts(self):
        # it doesn't seem to be possible to
        # access an actors attributes directly?
        return self.counts

    def get_image_info(self):
        return (self.nx, self.ny, self.nmax, self.cell_rad,
                self.ra[0], self.dec[0], self.x0[0], self.y0[0],
                self.freq_out, np.mean(self.times_out))

    def set_imaging_weights(self, counts=None):
        if counts is not None:
            self.counts = counts
        # TODO - this is more complicated if we do not always
        # sum all the counts in a band
        filter_level = self.opts.filter_counts_level
        if filter_level:
            self.counts = _filter_extreme_counts(self.counts,
                                                 level=filter_level)

        with cf.ThreadPoolExecutor(max_workers=self.nhthreads) as executor:
            futures = []
            for i, uvw in enumerate(self.uvw):
                fut = executor.submit(counts_to_weights,
                                      self.counts,
                                      uvw,
                                      self.freq,
                                      self.nx, self.ny,
                                      self.cell_rad,
                                      self.cell_rad,
                                      self.opts.robustness,
                                      i)
                futures.append(fut)

            # we keep imwgt and wgt separate for l2 reweighting purposes
            self.imwgt = [1] * len(self.uvw)
            self.wsums = [1] * len(self.uvw)
            for fut in cf.as_completed(futures):
                i, imwgt = fut.result()
                self.imwgt[i] = imwgt
                self.wsums[i] = (imwgt * self.wgt[i]).sum()

        # this is wsum for the band not in total
        self.wsumb = np.sum(self.wsums)

        return self.wsumb

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
            for i in range(self.ntimes):
                cname = self.cache_path + f'_time{i}.zarr'
                ds = xr.open_zarr(cname, chunks=None)
                self.resids[i] = ds.RESID.values
                self.psfs[i] = ds.PSF.values
                self.psfhats[i] = ds.PSFHAT.values
                self.wgt[i] = ds.WGT.values
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
        resid = np.sum(self.resids, axis=0)
        self.data = resid/wsum + self.psf_conv(self.model)
        return 1  # do we need to return something?


    def set_residual(self, x=None):
        if x is None:
            x = self.model

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
        self.data = resid/self.wsum + self.psf_conv(x)

        # return residual since we need the MFS residual on the runner
        return resid

    def psf_conv(self, x):
        convx = np.zeros_like(x)
        for psfhat in self.psfhats:
            convx += psf_convolve_slice(self.xpad,
                                        self.xhat,
                                        self.xout,
                                        psfhat,
                                        self.lastsize,
                                        x,
                                        nthreads=self.nvthreads)

        return convx/self.wsum


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

        if self.bandid == 0:
            from time import time
            ti = time()
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

        if self.bandid == 0:
            print('0 - ', time() - ti)

        # vtilde - (nband, nbasis, nmax)
        # convert to float32 for faster serialisation
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

    if model.any():
        # don't apply weights in this direction
        model_vis = dirty2vis(
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

        residual_vis = vis - model_vis
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
        psf_vis = np.ones((nrow, nchan), dtype=vis.dtype)

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

    # don't apply weights in this direction
    model_vis = dirty2vis(
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

    residual_vis = (vis - model_vis)

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
