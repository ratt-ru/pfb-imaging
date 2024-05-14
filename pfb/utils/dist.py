import numpy as np
import dask.array as da
from distributed import get_client, worker_client, wait
from pfb.utils.misc import fitcleanbeam
from pfb.operators.hessian import _hessian_impl
from pfb.operators.psf import psf_convolve_slice
from uuid import uuid4
import pywt
from pfb.wavelets.wavelets_jsk import  (get_buffer_size,
                                       dwt2d, idwt2d)
from ducc0.misc import make_noncritical


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
    def __init__(self, dds, opts, wsum, model, dual):
        # there should be only single versions of these for each band
        self.opts = opts
        self.cell_rad = dds[0].cell_rad
        self.nx, self.ny = model.shape
        self.wsum = wsum
        self.model = model
        self.dual = dual


        # there can be multiple of these
        self.psfhats = []
        self.dirtys = []
        self.resids = []
        self.beams = []
        self.weights = []
        self.uvws = []
        self.freqs = []
        self.masks = []
        self.wsumbs = []
        for ds in dds:
            self.psfhats.append(ds.PSFHAT.values/wsum)
            self.dirtys.append(ds.DIRTY.values/wsum)
            if model.any():
                self.resids.append(ds.RESIDUAL.values/wsum)
            else:
                self.resids.append(ds.DIRTY.values/wsum)
            self.beams.append(ds.BEAM.values)
            self.weights.append(ds.WEIGHT.values)
            self.uvws.append(ds.UVW.values)
            self.freqs.append(ds.FREQ.values)
            self.masks.append(ds.MASK.values)
            self.wsumbs.append(ds.WSUM.values)

        # TODO - we only need the sum of the dirty/resid images
        self.dirty = np.sum(self.dirtys, axis=0)
        self.resid = np.sum(self.resids, axis=0)
        if model.any():
            self.data = self.resid + self.psf_conv(model)
        else:
            self.data = self.dirty

        # set up psf convolve operators
        self.nthreads = opts.nvthreads
        self.lastsize = dds[0].PSF.shape[-1]

        # pre-allocate tmp arrays
        tmp = np.empty(self.dirty.shape,
                       dtype=self.dirty.dtype, order='C')
        self.xout = make_noncritical(tmp)
        tmp = np.empty(self.psfhats[0].shape,
                       dtype=self.psfhats[0].dtype,
                       order='C')
        self.xhat = make_noncritical(tmp)
        tmp = np.empty(ds.PSF.shape,
                       dtype=ds.PSF.dtype,
                       order='C')
        self.xpad = make_noncritical(tmp)

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
        self.b = np.zeros_like(model)
        self.bp = np.zeros_like(model)
        self.xp = np.zeros_like(model)
        self.vp = np.zeros_like(dual)
        self.vtilde = np.zeros_like(dual)


    def grad(self, x=None):
        self.resid = self.dirty.copy()

        if self.model.any():
            # TODO - do this in parallel
            for uvw, wgt, mask, freq, beam in zip(self.uvws, self.weights,
                                                  self.masks, self.freqs,
                                                  self.beams):
                self.resid -= _hessian_impl(x,
                                    uvw,
                                    wgt,
                                    mask,
                                    freq,
                                    beam,
                                    x0=0.0,
                                    y0=0.0,
                                    cell=self.cell_rad,
                                    do_wgridding=opts.do_wgridding,
                                    epsilon=opts.epsilon,
                                    double_accum=opts.double_accum,
                                    nthreads=opts.nvthreads)

        return -self.resid


    def psf_conv(self, x):
        convx = np.zeros_like(x)
        for psfhat in self.psfhats:
            convx += psf_convolve_slice(self.xpad,
                                        self.xhat,
                                        self.xout,
                                        psfhat,
                                        self.lastsize,
                                        x,
                                        nthreads=self.nthreads)

        return convx


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

        # comment below to eliminate the possibility of
        # wavelets/cf.futures being the culprit
        # run with bases=self only
        alpha[0, 0:self.buffer_size[self.bases[0]]] = x.ravel()

        # with cf.ThreadPoolExecutor(max_workers=self.nthreads) as executor:
        #     futures = []
        #     for i, wavelet in enumerate(self.bases):
        #         if wavelet=='self':
        #             alpha[i, 0:self.buffer_size[wavelet]] = x.ravel()
        #             continue

        #         f = executor.submit(dwt, x,
        #                             self.buffer[wavelet],
        #                             self.dec_lo[wavelet],
        #                             self.dec_hi[wavelet],
        #                             self.nlevel,
        #                             i)

        #         futures.append(f)

        #     for f in cf.as_completed(futures):
        #         buffer, i = f.result()
        #         alpha[i, 0:self.buffer_size[self.bases[i]]] = buffer

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

        # comment below to eliminate the possibility of
        # wavelets/cf.futures being the culprit
        # run with bases=self only
        nmax = self.buffer_size[self.bases[0]]
        x = alpha[0, 0:nmax].reshape(nx, ny)

        # with cf.ThreadPoolExecutor(max_workers=self.nthreads) as executor:
        #     futures = []
        #     for i, wavelet in enumerate(self.bases):
        #         nmax = self.buffer_size[wavelet]
        #         if wavelet=='self':
        #             x += alpha[i, 0:nmax].reshape(nx, ny)
        #             continue

        #         f = executor.submit(idwt,
        #                             alpha[i, 0:nmax],
        #                             self.rec_lo[wavelet],
        #                             self.rec_hi[wavelet],
        #                             self.nlevel,
        #                             self.nx,
        #                             self.ny,
        #                             i)


        #         futures.append(f)

        #     for f in cf.as_completed(futures):
        #         image, i = f.result()
        #         x += image

        return x

    def init_pd_params(self, hessnorm, nu, gamma=1):
        self.hessnorm = hessnorm
        self.sigma = hessnorm/(2*gamma*nu)
        self.tau = 0.9 / (hessnorm / (2.0 * gamma) + self.sigma * nu**2)
        self.vtilde = self.v + self.sigma * self.psi_dot(self.x)
        return self.vtilde

    def update_data(self):
        self.data = self.residual + self.psf_conv(self.x)

    def pd_update(self, ratio):
        self.xp[...] = self.model[...]
        self.vp[...] = self.dual[...]

        self.dual[...] = self.vtilde * (1 - ratio)

        grad = self.almost_grad(self.xp)
        self.model[...] = self.xp - self.tau * (self.psi_hdot(2*self.dual - self.vp) + grad)

        if self.positivity:
            self.model[self.model < 0] = 0.0

        self.vtilde[...] = self.dual + self.sigma * self.psi_dot(self.model)

        eps_num = np.sum((self.model-self.xp)**2)
        eps_den = np.sum(self.model**2)

        return self.vtilde, eps_num, eps_den


    def init_random(self):
        self.x = np.random.randn(self.nx, self.ny)
        return np.sum(self.x**2)


    def pm_update(self, bnorm):
        self.bp[...] = self.b/bnorm
        self.b[...] = self.psf_conv(self.bp)
        bsumsq = np.sum(self.b**2)
        beta_num = np.vdot(self.b, self.bp)
        beta_den = np.vdot(self.bp, self.bp)

        return bsumsq, beta_num, beta_den





