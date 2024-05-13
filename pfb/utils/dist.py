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

# submit on these
def accum_wsums(dds):
    return np.sum([ds.WSUM.values for ds in dds])


def get_cbeam_area(dds, wsum):
    psf_mfs = np.stack([ds.PSF.values for ds in dds]).sum(axis=0)/wsum
    # beam pars in pixel units
    GaussPar = fitcleanbeam(psf_mfs[None], level=0.5, pixsize=1.0)[0]
    return GaussPar[0]*GaussPar[1]*np.pi/4


def get_resids(ds, wsum, hessopts):
    dirty = ds.DIRTY.values
    nx, ny = dirty.shape
    if ds.MODEL.values.any():
        resid = dirty - _hessian_impl(ds.MODEL.values,
                                      ds.UVW.values,
                                      ds.WEIGHT.values,
                                      ds.MASK.values,
                                      ds.FREQ.values,
                                      ds.BEAM.values,
                                      x0=0.0,
                                      y0=0.0,
                                      **hessopts)
    else:
        resid = ds.DIRTY.values.copy()

    # dso = ds.assign(**{
    #     'RESIDUAL': (('x', 'y'), resid)
    # })

    return resid/wsum


def get_mfs_and_stats(resids):
    # resids should aready be normalised by wsum
    residual_mfs = np.sum(resids, axis=0)
    rms = np.std(residual_mfs)
    rmax = np.sqrt((residual_mfs**2).max())
    return residual_mfs, rms, rmax


class almost_grad(object):
    def __init__(self, model, psfo, residual):
        self.psfo = psfo
        if model.any():
            self.data = residual + self.psfo(model, 0.0)
        else:
            self.data = residual.copy()

    def __call__(self, model):
        return self.psfo(model, 0.0) - self.data

    def update_data(self, model, residual):
        self.data = residual + self.psfo(model, 0.0)





def get_epsb(xp, x):
    return np.sum((x-xp)**2), np.sum(x**2)

def get_eps(num, den):
    return np.sqrt(np.sum(num)/np.sum(den))


def psi_dot(psib, model):
    return psib.dot(model)


def l1reweight_func(psif, ddsf, rmsfactor, rms_comps, alpha=4):
    '''
    The logic here is that weights should remain the same for model
    components that are rmsfactor times larger than the rms.
    High SNR values should experience relatively small thresholding
    whereas small values should be strongly thresholded
    '''
    client = get_client()
    outvar = []
    for wname, ds in ddsf.items():
        outvar.append(client.submit(psi_dot,
                                    psif[wname],
                                    ds.MODEL.values,
                                    key='outvar-'+uuid4().hex,
                                    workers=wname))

    outvar = client.gather(outvar)
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
    def __init__(ddsl, opts, wsum):
        # there should be only single versions of these for each band
        self.opts = opts
        self.cell_rad = ddsl[0].cell_rad
        self.nx, self.ny = model.shape


        # there can be multiple of these
        self.psfhats = []
        self.dirtys = []
        self.beams = []
        self.weights = []
        self.uvws = []
        self.freqs = []
        self.masks = []
        self.wsumbs = []
        for ds in ddsl:
            self.psfhats.append(ds.PSFHAT.values.copy())
            self.dirtys.append(ds.DIRTY.values.copy())
            self.beams.append(ds.BEAM.values.copy())
            self.weights.append(ds.WEIGHT.values.copy())
            self.uvws.append(ds.UVW.values.copy())
            self.freqs.append(ds.FREQ.values.copy())
            self.masks.append(ds.MASK.values.copy())
            self.wsumbs.append(ds.WSUM.values.copy())

        # we only need to keep the sum of the dirty images
        self.dirty = np.sum(self.dirtys, axis=0)

        # set up psf convolve operators
        self.nthreads = opts.nvthreads
        self.lastsize = ddsl[0].PSF.shape[-1]
        self.psfhat = ds.PSFHAT.values
        self.beam = ds.BEAM.values
        self.wsumb = ds.WSUM.values[0]
        self.wsum = wsum

        # pre-allocate tmp arrays
        tmp = np.empty(self.dirty.shape,
                       dtype=self.dirty.dtype, order='C')
        self.xout = make_noncritical(tmp)
        tmp = np.empty(self.psfhats[0].shape,
                       dtype=self.psfhats[0].dtype,
                       order='C')
        self.xhat = make_noncritical(tmp)
        tmp = np.empty(ddsl[0].PSF.shape,
                       dtype=ddsl[0].PSF.dtype,
                       order='C')
        self.xpad = make_noncritical(tmp)

        # set up wavelet dictionaries
        self.bases = opts.bases
        self.nbasis = len(bases)
        self.nlevel = opts.nlevel
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

            self.buffer_size[wavelet] = get_buffer_size((nx, ny),
                                                        self.dec_lo[wavelet].size,
                                                        nlevel)
            self.buffer[wavelet] = np.zeros(self.buffer_size[wavelet],
                                            dtype=np.float64)

            self.nmax = np.maximum(self.nmax, self.buffer_size[wavelet])

        # # set primal dual params
        # self.sigma =


    def grad(self, x):
        resid = self.dirty.copy()

        # TODO - do this in parallel
        for uvw, wgt, mask, freq, beam in zip(self.uvws, self.weights,
                                              self.masks, self.freqs,
                                              self.beams):
            resid -= _hessian_impl(x,
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

        return -resid


    def psf_conv(self, x):
        convx = np.zeros_like(x)
        for psfhat in self.psfhats:
            convx += psf_convolve_slice(self.xpad,
                                        self.xhat,
                                        self.xout,
                                        psfat,
                                        self.lastsize,
                                        x,
                                        nthreads=self.nthreads)

        return convx


    def almost_grad(self, x):
        convx = self.psf_conv(x)
        return convx - self.data


    def psi_dot(self, x):
        '''
        signal to coeffs

        Input:
            x       - (nx, ny) input signal

        Output:
            alpha   - (nbasis, Nnmax) output coeffs
        '''
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

    def init_model(self, model=None, dual=None):
        if model is not None:
            self.x = model
            self.data = self.psf_conv(model) + self.residual
        else:
            self.x = np.zeros((nx, ny), dtype=self.dirty.dtype)
            self.data = self.residual
        if dual is not None:
            self.v = dual
        else:
            self.v = np.zeros((self.nbasis, self.nmax),
                              dtype=self.dirty.dtype)
        self.xp = np.zeros_like(self.x)
        self.vp = np.zeros_like(self.v)
        self.vtilde = np.zeros_like(self.v)


    def set_pd_params(self, hessnorm, sigma, nu, gamma=1):
        self.hessnorm = hessnorm
        self.sigma = hessnorm/(2*gamma*nu)
        self.tau = 0.9 / (hessnorm / (2.0 * gamma) + self.sigma * nu**2)

    def update_lam21(self, lam):
        self.lam = lam

    def init_vtilde(self):
        self.vtilde = self.v + self.sigma * self.psi_dot(self.x)
        return self.vtilde

    def pd_update(self, ratio):
        self.xp[...] = self.x[...]
        self.vp[...] = self.v[...]

        self.v[...] = self.vtilde * (1 - ratio)

        grad = self.almost_grad(self.xp)





