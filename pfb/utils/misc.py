import sys
from contextlib import nullcontext
import numpy as np
import numexpr as ne
from numba import njit, prange
from numba.extending import overload
import dask
import dask.array as da
from dask.distributed import performance_report
from dask.diagnostics import ProgressBar
from ducc0.fft import r2c, c2r, good_size
iFs = np.fft.ifftshift
Fs = np.fft.fftshift
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_from_storage_table as xds_from_table
from daskms.experimental.zarr import xds_from_zarr
from omegaconf import ListConfig
from skimage.morphology import label
from scipy.optimize import fmin_l_bfgs_b
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_triangular
import sympy as sm
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr
from africanus.constants import c as lightspeed
import jax.numpy as jnp
from jax import value_and_grad
import jax

JIT_OPTIONS = {
    "nogil": True,
    "cache": True
}

import pdb
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def compute_context(scheduler, output_filename, boring=True):
    if scheduler == "distributed":
        return performance_report(filename=output_filename + "_dask_report.html")
    else:
        if boring:
            return nullcontext()
        else:
            return ProgressBar()


def kron_matvec(A, b):
    D = len(A)
    N = b.size
    x = b

    for d in range(D):
        Gd = A[d].shape[0]
        NGd = N // Gd
        X = np.reshape(x, (Gd, NGd))
        Z = A[d].dot(X).T
        x = Z.ravel()
    return x.reshape(b.shape)

@njit(parallel=False, cache=True, nogil=True)
def kron_matvec2(A, b):
    D = len(A)
    N = b.size
    x = b

    for d in range(D):
        Gd = A[d].shape[0]
        NGd = N//Gd
        X = np.reshape(x, (Gd, NGd))
        Z = np.zeros((Gd, NGd), dtype=A[0].dtype)
        Ad = A[d]
        for i in range(Gd):
            for j in range(Gd):
                for k in range(NGd):
                    Z[j, k] += Ad[i, j] * X[i, k]
        x[:] = Z.T.ravel()
    return x


def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.), normalise=True, nsigma=5):
    S0, S1, PA = GaussPar
    Smaj = S0  #np.maximum(S0, S1)
    Smin = S1  #np.minimum(S0, S1)
    # print(f'using ex = {Smaj}, ey = {Smin}')
    A = np.array([[1. / Smin ** 2, 0],
                  [0, 1. / Smaj ** 2]])

    c, s, t = np.cos, np.sin, np.deg2rad(-PA)
    R = np.array([[c(t), -s(t)],
                  [s(t), c(t)]])
    A = np.dot(np.dot(R.T, A), R)
    sOut = xin.shape
    # only compute the result out to 5 * emaj
    extent = (nsigma * Smaj)**2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    idx, idy = np.where(xflat**2 + yflat**2 <= extent)
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    R = np.einsum('nb,bc,cn->n', x.T, A, x)
    # need to adjust for the fact that GaussPar corresponds to FWHM
    fwhm_conv = 2 * np.sqrt(2 * np.log(2))
    tmp = np.exp(-fwhm_conv * R)
    gausskern = np.zeros(xflat.shape, dtype=np.float64)
    gausskern[idx, idy] = tmp

    if normalise:
        gausskern /= np.sum(gausskern)
    return np.ascontiguousarray(gausskern.reshape(sOut),
                                dtype=np.float64)


def give_edges(p, q, nx, ny, nx_psf, ny_psf):
    nx0 = nx_psf//2
    ny0 = ny_psf//2

    # image overlap edges
    # left edge for x coordinate
    dxl = p - nx0
    xl = np.maximum(dxl, 0)

    # right edge for x coordinate
    dxu = p + nx0
    xu = np.minimum(dxu, nx)
    # left edge for y coordinate
    dyl = q - ny0
    yl = np.maximum(dyl, 0)
    # right edge for y coordinate
    dyu = q + ny0
    yu = np.minimum(dyu, ny)

    # PSF overlap edges
    xlpsf = np.maximum(nx0 - p , 0)
    xupsf = np.minimum(nx0 + nx - p, nx_psf)
    ylpsf = np.maximum(ny0 - q, 0)
    yupsf = np.minimum(ny0 + ny - q, ny_psf)

    return slice(xl, xu), slice(yl, yu), \
        slice(xlpsf, xupsf), slice(ylpsf, yupsf)


def get_padding_info(nx, ny, pfrac):
    npad_x = int(pfrac * nx)
    nfft = good_size(nx + npad_x, True)
    npad_xl = (nfft - nx) // 2
    npad_xr = nfft - nx - npad_xl

    npad_y = int(pfrac * ny)
    nfft = good_size(ny + npad_y, True)
    npad_yl = (nfft - ny) // 2
    npad_yr = nfft - ny - npad_yl
    padding = ((npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    return padding, unpad_x, unpad_y


def convolve2gaussres(image, xx, yy, gaussparf, nthreads=1, gausspari=None,
                      pfrac=0.5, norm_kernel=False):
    """
    Convolves the image to a specified resolution.

    Parameters
    ----------
    Image       - (nband, nx, ny) array to convolve
    xx/yy       - coordinates on the grid in the same units as gaussparf.
    gaussparf   - tuple containing Gaussian parameters of desired resolution
                  (emaj, emin, pa).
    gausspari   - initial resolution . By default it is assumed that the image
                  is a clean component image with no associated resolution.
                  If beampari is specified, it must be a tuple containing
                  gausspars for each imaging band in the same format.
    nthreads    - number of threads to use for the FFT's.
    pfrac       - padding used for the FFT based convolution.
                  Will pad by pfrac/2 on both sides of image
    """
    nband, nx, ny = image.shape
    if gausspari is not None and len(gausspari) != nband:
        raise ValueError('gausspari must be on length nband')
    padding, unpad_x, unpad_y = get_padding_info(nx, ny, pfrac)
    ax = (1, 2)  # axes over which to perform fft
    lastsize = ny + np.sum(padding[-1])

    padding = ((0,0),) + padding
    image = np.pad(image, padding, mode='constant')
    imhat = r2c(iFs(image, axes=ax), axes=ax, forward=True, nthreads=nthreads,
                inorm=0)

    if len(gaussparf) == 3:  # single final resolution
        gausskern = Gaussian2D(xx, yy, gaussparf, normalise=norm_kernel)
        gausskern = np.pad(gausskern, padding[1:], mode='constant')
        gausskernhat = r2c(iFs(gausskern, axes=(0,1)), axes=(0,1),
                           forward=True, nthreads=nthreads, inorm=0)
        gausskernhat = np.broadcast_to(gausskernhat[None], imhat.shape)
    else:
        assert len(gaussparf) == nband
        gausskernhat = np.zeros_like(imhat)
        for b in range(nband):
            gausskern = Gaussian2D(xx, yy, gaussparf[b], normalise=norm_kernel)
            gausskern = np.pad(gausskern, padding[1:], mode='constant')
            r2c(iFs(gausskern, axes=(0,1)), out=gausskernhat[b],
                axes=(0,1), forward=True, nthreads=nthreads, inorm=0)


    # convolve to desired resolution
    if gausspari is None:
        imhat *= gausskernhat
    else:
        for b in range(nband):
            thiskern = Gaussian2D(xx, yy, gausspari[b], normalise=norm_kernel)
            thiskern = np.pad(thiskern, padding[1:], mode='constant')
            thiskernhat = r2c(iFs(thiskern, axes=(0,1)), axes=(0,1), forward=True,
                              nthreads=nthreads, inorm=0)

            convkernhat = np.zeros_like(thiskernhat)
            msk = np.abs(thiskernhat) > 0.0
            convkernhat[msk] = gausskernhat[b, msk]/thiskernhat[msk]

            imhat[b] *= convkernhat

    image = Fs(c2r(imhat, axes=ax, forward=False, lastsize=lastsize, inorm=2,
                   nthreads=nthreads), axes=ax)[:, unpad_x, unpad_y]

    return image


@dask.delayed
def fetch_poltype(corr_type):
    corr_type = set(tuple(corr_type))
    if corr_type.issubset(set([9, 10, 11, 12])):
        return 'linear'
    elif corr_type.issubset(set([5, 6, 7, 8])):
        return 'circular'


def construct_mappings(ms_name,
                       gain_name=None,
                       ipi=None,
                       cpi=None,
                       freq_min=-np.inf,
                       freq_max=np.inf,
                       FIELD_IDs=None,
                       DDIDs=None,
                       SCANs=None):
    '''
    Construct dictionaries containing per MS, FIELD, DDID and SCAN
    time and frequency mappings.

    Input:
    ms_name     - list of ms names
    gain_name   - list of paths to gains, must be in same order as ms_names
    ipi         - integrations (i.e. unique times) per output image.
                  Defaults to one per scan.
    cpi         - Channels per image. Defaults to one per spw.

    The chan <-> band mapping is determined by:

    freqs           - dict[MS][IDT] frequencies chunked by band
    fbin_idx        - dict[MS][IDT] freq bin starting indices
    fbin_counts     - dict[MS][IDT] freq bin counts
    band_mapping    - dict[MS][IDT] output bands in dataset
    freq_out        - array of linearly spaced output frequencies
                      over entire frequency range.
                      freq_out[band_mapping[MS][IDT]] gives the
                      model frequencies in a dataset

    where IDT is constructed as FIELD#_DDID#_SCAN#.

    Similarly, the row <-> time mapping is determined by:

    utimes          - dict[MS][IDT] unique times per output image
    tbin_idx        - dict[MS][IDT] time bin starting indices
    tbin_counts     - dict[MS][IDT] time bin counts
    time_mapping    - dict[MS][IDT] utimes per dataset

    '''

    if not isinstance(ms_name, list) and not isinstance(ms_name, ListConfig):
        ms_name = [ms_name]

    if gain_name is not None:
        if not isinstance(gain_name, list) and not isinstance(gain_name, ListConfig):
            gain_name = [gain_name]
        assert len(ms_name) == len(gain_name)

    # collect times and frequencies per ms and ds
    freqs = {}
    chan_widths = {}
    times = {}
    gains = {}
    radecs = {}
    antpos = {}
    poltype = {}
    uv_maxs = []
    idts = {}
    for ims, ms in enumerate(ms_name):
        xds = xds_from_ms(ms, chunks={"row": -1}, columns=('TIME','UVW'),
                          group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])

        # subtables
        ddids = xds_from_table(ms + "::DATA_DESCRIPTION")[0]
        fields = xds_from_table(ms + "::FIELD")[0]
        feeds = xds_from_table(ms + "::FEED")[0]
        spws = xds_from_table(ms + "::SPECTRAL_WINDOW")[0]
        pols = xds_from_table(ms + "::POLARIZATION")[0]
        ants = xds_from_table(ms + "::ANTENNA")[0]

        antpos[ms] = ants.POSITION.data
        poltype[ms] = fetch_poltype(pols.CORR_TYPE.data.squeeze())
        unique_feeds = np.unique(feeds.POLARIZATION_TYPE.values)

        if np.all([feed in "XxYy" for feed in unique_feeds]):
            feed_type = "linear"
        elif np.all([feed in "LlRr" for feed in unique_feeds]):
            feed_type = "circular"
        else:
            raise ValueError("Unsupported feed type/configuration.")

        idts[ms] = []
        freqs[ms] = {}
        times[ms] = {}
        radecs[ms] = {}
        chan_widths[ms] = {}
        for ds in xds:
            fid = ds.FIELD_ID
            ddid = ds.DATA_DESC_ID
            scanid = ds.SCAN_NUMBER
            if (FIELD_IDs is not None) and (fid not in FIELD_IDs):
                continue
            if (DDIDs is not None) and (ddid not in DDIDs):
                continue
            if (SCANs is not None) and (scanid not in SCANs):
                continue

            idt = f"FIELD{fid}_DDID{ddid}_SCAN{scanid}"
            idts[ms].append(idt)
            radecs[ms][idt] = fields.PHASE_DIR.data[fid].squeeze()

            freqs[ms][idt] = spws.CHAN_FREQ.data[ddid]
            chan_widths[ms][idt] = spws.CHAN_WIDTH.data[ddid]
            times[ms][idt] = da.atleast_1d(ds.TIME.data.squeeze())
            uvw = ds.UVW.data
            u_max = abs(uvw[:, 0]).max()
            v_max = abs(uvw[:, 1]).max()
            uv_maxs.append(da.maximum(u_max, v_max))

    # Early compute to get metadata
    times, freqs, radecs, chan_widths, uv_maxs, antpos, poltype = \
        dask.compute(times, freqs,radecs, chan_widths,uv_maxs,
                     antpos, poltype)

    uv_max = max(uv_maxs)
    all_freqs = []
    all_times = []
    freq_mapping = {}
    row_mapping = {}
    time_mapping = {}
    utimes = {}
    ms_chunks = {}
    for ims, ms in enumerate(ms_name):
        freq_mapping[ms] = {}
        row_mapping[ms] = {}
        time_mapping[ms] = {}
        utimes[ms] = {}
        ms_chunks[ms] = []
        gains[ms] = {}

        if gain_name is not None:
            gds = xds_from_zarr(gain_name[ims])

        for idt in idts[ms]:
            ilo = idt.find('FIELD') + 5
            ihi = idt.find('_')
            fid = int(idt[ilo:ihi])
            ilo = idt.find('DDID') + 4
            ihi = idt.rfind('_')
            ddid = int(idt[ilo:ihi])
            ilo = idt.find('SCAN') + 4
            scanid = int(idt[ilo:])
            freq = freqs[ms][idt]
            nchan_in = freq.size
            idx = (freq>=freq_min) & (freq<=freq_max)
            if not idx.any():
                continue
            idx0 = np.argmax(idx) # returns index of first True element
            try:
                # returns zero if not idx.any()
                assert idx[idx0]
            except Exception as e:
                continue
            freq = freq[idx]
            nchan = freq.size
            if cpi in [-1, 0, None]:
                cpit = nchan
                cpi = nchan_in
            else:
                cpit = np.minimum(cpi, nchan)
                cpi = np.minimum(cpi, nchan_in)
            freq_mapping[ms][idt] = {}
            tmp = np.arange(idx0, idx0 + nchan, cpit)
            freq_mapping[ms][idt]['start_indices'] = tmp
            if cpit != nchan:
                tmp2 = np.append(tmp, [idx0 + nchan])
                freq_mapping[ms][idt]['counts'] = tmp2[1:] - tmp2[0:-1]
            else:
                freq_mapping[ms][idt]['counts'] = np.array((nchan,), dtype=int)

            time = times[ms][idt]
            utime = np.unique(time)
            utimes[ms][idt] = utime
            all_times.append(utime)

            ntime = utimes[ms][idt].size
            if ipi in [0, -1, None]:
                ipit = ntime
            else:
                ipit = np.minimum(ipi, ntime)
            row_chunks, ridx, rcounts = chunkify_rows(time, ipit,
                                                      daskify_idx=False)
            # these are for applying gains
            # essentially the number of rows per unique time
            row_mapping[ms][idt] = {}
            row_mapping[ms][idt]['start_indices'] = ridx
            row_mapping[ms][idt]['counts'] = rcounts

            freq_idx0 = freq_mapping[ms][idt]['start_indices'][0]
            if freq_idx0 != 0:
                freq_chunks = (freq_idx0,) + tuple(freq_mapping[ms][idt]['counts'])
            else:
                freq_chunks = tuple(freq_mapping[ms][idt]['counts'])
            freq_idxf = np.sum(freq_chunks)
            if freq_idxf != nchan_in:
                freq_chunkf = nchan_in - freq_idxf
                freq_chunks += (freq_chunkf,)

            try:
                assert np.sum(freq_chunks) == nchan_in
            except Exception as e:
                raise RuntimeError("Something went wrong constructing the "
                                   "frequency mapping. sum(fchunks != nchan)")

            ms_chunks[ms].append({'row': row_chunks,
                                  'chan': freq_chunks})

            time_mapping[ms][idt] = {}
            tmp = np.arange(0, ntime, ipit)
            time_mapping[ms][idt]['start_indices'] = tmp
            if ipit != ntime:
                tmp2 = np.append(tmp, [ntime])
                time_mapping[ms][idt]['counts'] = tmp2[1:] - tmp2[0:-1]
            else:
                time_mapping[ms][idt]['counts'] = np.array((ntime,))

            if gain_name is not None:
                freq0 = freq[0]
                freqf = freq[-1]
                gdsf = [dsg for dsg in gds if dsg.FIELD_ID == fid]
                gdsfd = [dsg for dsg in gdsf if dsg.DATA_DESC_ID == ddid]
                # gains may have been solved over scans
                if 'SCAN_NUMBER' in gdsfd[0].attrs:
                    gdsfds = [dsg for dsg in gdsfd if dsg.SCAN_NUMBER == scanid]
                    try:
                        assert len(gdsfds) == 1
                    except Exception as e:
                        raise RuntimeError(f"Gain datasets don't align for "
                                           f"ms {ms} at FIELD_ID = {fid}, "
                                           f"DATA_DESC_ID = {ddid}, "
                                           f"SCAN_NUMBER = {scanid}. "
                                           f"len(gds) = {len(gdsfds)}")
                    try:
                        assert (gdsfds[0].gain_time == utime).all()
                    except Exception as e:
                        raise ValueError(f'Mismatch between gain and MS '
                                            f'utimes for {ms} at {idt}')
                    
                    gains[ms][idt] = gdsfds[0]
                else:
                    try:
                        assert len(gdsfd) == 1
                    except Exception as e:
                        raise RuntimeError("Multiple gain datasets per "
                                           "FIELD and DDID but SCAN_NUMBER "
                                           "not in attributes")
                    t0 = utime[0]
                    tf = utime[-1]
                    gains[ms][idt] = gdsfd[0].sel(
                                        gain_time=slice(t0, tf))
            else:
                gains[ms][idt] = None

    return row_mapping, freq_mapping, time_mapping, \
           freqs, utimes, ms_chunks, gains, radecs, \
           chan_widths, uv_max, antpos, poltype


def restore_corrs(vis, ncorr):
    return da.blockwise(_restore_corrs, ('row', 'chan', 'corr'),
                        vis, ('row', 'chan'),
                        ncorr, None,
                        new_axes={"corr": ncorr},
                        dtype=vis.dtype)


def _restore_corrs(vis, ncorr):
    model_vis = np.zeros(vis.shape+(ncorr,), dtype=vis.dtype)
    model_vis[:, :, 0] = vis
    if model_vis.shape[-1] > 1:
        model_vis[:, :, -1] = vis
    return model_vis


@jax.jit
def psf_errorsq(x, data, xy):
    '''
    Returns sum of square error for best fit Gaussian to data.
    Note emaj must be larger than emin
    '''
    emaj, emin, pa = x
    A = jnp.array([[1. / emin ** 2, 0],
                    [0, 1. / emaj ** 2]])

    t = jnp.deg2rad(-pa)
    R = jnp.array([[jnp.cos(t), -jnp.sin(t)],
                    [jnp.sin(t), jnp.cos(t)]])
    B = jnp.dot(jnp.dot(R.T, A), R)
    Q = jnp.einsum('nb,bc,cn->n', xy.T, B, xy)
    # GaussPar should corresponds to FWHM
    fwhm_conv = 2 * jnp.sqrt(2 * jnp.log(2))
    model = jnp.exp(-fwhm_conv * Q)
    res = data - model
    return jnp.vdot(res, res)


def fitcleanbeam(psf: np.ndarray,
                 level: float = 0.5,
                 pixsize: float = 1.0,
                 extent: float = 15.0):
    """
    Find the Gaussian that approximates the PSF.
    First find the main lobe by identifying where PSF > level
    then fit Gaussian out to a radius of extent * max(x, y) where
    x and y are the coordinates where PSF > level.
    """
    nband, nx, ny = psf.shape

    # pixel coordinates
    x = np.arange(-nx / 2, nx / 2)
    y = np.arange(-ny / 2, ny / 2)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    Gausspars = []
    for v in range(nband):
        # make sure psf is normalised
        if not psf[v].any():
            Gausspars.append([np.nan, np.nan, np.nan])
            continue
        psfv = psf[v] / psf[v].max()
        # find regions where psf is above level
        mask = np.where(psfv > level, 1.0, 0)

        # label all islands and find center
        islands = label(mask)
        ncenter = islands[nx // 2, ny // 2]

        # get extend of main lobe
        x = xx[islands == ncenter]
        y = yy[islands == ncenter]
        xdiff = x.max() - x.min()
        ydiff = y.max() - y.min()
        rsq = np.abs(x).max()**2 + np.abs(y).max()**2
        rrsq = xx**2 + yy**2
        idxs = rrsq < extent * rsq

        # select psf main lobe
        psfv = psfv[idxs]
        x = xx[idxs]
        y = yy[idxs]
        xy = np.vstack((x, y))
        emaj0 = np.maximum(xdiff, ydiff)
        emin0 = np.minimum(xdiff, ydiff)
        dfunc = value_and_grad(psf_errorsq)
        p, f, d = fmin_l_bfgs_b(dfunc,
                                np.array((emaj0, emin0, 0.0)),
                                args=(psfv, xy),
                                bounds=((0, None), (0, None), (None, None)),
                                factr=1e11)
        Gausspars.append([p[0] * pixsize, p[1] * pixsize, p[2]])

    return Gausspars


# utility to produce a model image from fitted components
def model_from_comps(comps, freq, mask, band_mapping, ref_freq, fitted):
    '''
    comps - (order/nband, ncomps) array of fitted components or model vals
            per band if no fitting was done.

    freq - (nchan) array of output frequencies

    mask - (nx, ny) bool array indicating where model is non-zero.
           comps need to be aligned with this mask i.e.
           Ix, Iy = np.where(mask) are the xy locations of non-zero pixels
           model[:, Ix, Iy] = comps if no fitting was done

    band_mapping - tuple containg the indices of bands mapping to the output.
                   In the case of a single spectral window we usually have
                   band_mapping = np.arange(nband).

    ref_freq - reference frequency used during fitting i.e. we made the
                design matrix using freq/ref_freq as coordinate

    fitted - bool indicating if any fitting actually happened.
    '''
    return da.blockwise(_model_from_comps_wrapper, ('out', 'nx', 'ny'),
                        comps, ('com', 'pix'),
                        freq, ('chan',),
                        mask, ('nx', 'ny'),
                        band_mapping, ('out',),
                        ref_freq, None,
                        fitted, None,
                        align_arrays=False,
                        dtype=comps.dtype)


def _model_from_comps_wrapper(comps, freq, mask, band_mapping, ref_freq, fitted):
    return _model_from_comps(comps[0][0], freq[0], mask, band_mapping, ref_freq, fitted)


def _model_from_comps(comps, freq, mask, band_mapping, ref_freq, fitted):
    freqo = freq[band_mapping]
    nband = freqo.size
    nx, ny = mask.shape
    model = np.zeros((nband, nx, ny), dtype=comps.dtype)
    if fitted:
        order, npix = comps.shape
        nband = freqo.size
        nx, ny = mask.shape
        model = np.zeros((nband, nx, ny), dtype=comps.dtype)
        w = (freqo / ref_freq).reshape(nband, 1)
        Xdes = np.tile(w, order) ** np.arange(0, order)
        beta_rec = Xdes.dot(comps)
        model[:, mask] = beta_rec
    else:
        model[:, mask] = comps[band_mapping]

    return model


def init_mask(mask, model, output_type, log):
    if mask is None:
        print("No mask provided", file=log)
        mask = np.ones((mds.nx, mds.ny), dtype=output_type)
    elif mask.endswith('.fits'):
        try:
            mask = load_fits(mask, dtype=output_type).squeeze()
            assert mask.shape == (mds.nx, mds.ny)
            print('Using provided fits mask', file=log)
        except Exception as e:
            print(f"No mask found at {mask}", file=log)
            raise e
    elif mask.lower() == 'model':
        mask = np.any(model, axis=0)
        print('Using model to construct mask', file=log)
    else:
        raise ValueError(f'Unsupported masking option {mask}')
    return mask


def dds2cubes(dds, nband, apparent=False, dual=True,
              modelname='MODEL', residname='RESIDUAL'):
    real_type = dds[0].DIRTY.dtype
    complex_type = np.result_type(real_type, np.complex64)
    nx, ny = dds[0].DIRTY.shape
    dirty = [da.zeros((nx, ny), chunks=(-1, -1),
                      dtype=real_type) for _ in range(nband)]
    model = [da.zeros((nx, ny), chunks=(-1, -1),
                      dtype=real_type) for _ in range(nband)]
    if residname in dds[0]:
        residual = [da.zeros((nx, ny), chunks=(-1, -1),
                            dtype=real_type) for _ in range(nband)]
    else:
        residual = None
    wsums = [da.zeros(1, dtype=real_type) for _ in range(nband)]
    if 'PSF' in dds[0]:
        nx_psf, ny_psf = dds[0].PSF.shape
        psf = [da.zeros((nx_psf, ny_psf), chunks=(-1, -1),
                        dtype=real_type) for _ in range(nband)]
    else:
        psf = None
    if 'PSFHAT' in dds[0]:
        nx_psf, nyo2_psf = dds[0].PSFHAT.shape
        psfhat = [da.zeros((nx_psf, nyo2_psf), chunks=(-1, -1),
                            dtype=complex_type) for _ in range(nband)]
    else:
        psfhat = None
    mean_beam = [da.zeros((nx, ny), chunks=(-1, -1),
                            dtype=real_type) for _ in range(nband)]
    if dual and 'DUAL' in dds[0]:
        nbasis, nymax, nxmax = dds[0].DUAL.shape
        dual = [da.zeros((nbasis, nymax, nxmax), chunks=(-1, -1, -1),
                            dtype=real_type) for _ in range(nband)]
    else:
        dual = None
    for ds in dds:
        b = ds.bandid
        if apparent:
            dirty[b] += ds.DIRTY.data
            if residname in ds:
                residual[b] += getattr(ds, residname).data
        else:
            dirty[b] += ds.DIRTY.data * ds.BEAM.data
            if residname in ds:
                residual[b] += getattr(ds, residname).data * ds.BEAM.data
        if 'PSF' in ds:
            psf[b] += ds.PSF.data
        if 'PSFHAT' in ds:
            psfhat[b] += ds.PSFHAT.data
        if modelname in ds:
            model[b] = getattr(ds, modelname).data
        if dual and 'DUAL' in ds:
            dual[b] = ds.DUAL.data
        mean_beam[b] += ds.BEAM.data * ds.WSUM.data[0]
        wsums[b] += ds.WSUM.data[0]
    wsums = da.stack(wsums).reshape(nband)
    wsum = wsums.sum()
    dirty = da.stack(dirty)/wsum
    model = da.stack(model)
    if residname in ds:
        residual = da.stack(residual)/wsum
    if 'PSF' in ds:
        psf = da.stack(psf)/wsum
    if 'PSFHAT' in ds:
        psfhat = da.stack(psfhat)/wsum
    if dual and 'DUAL' in ds:
        dual = da.stack(dual)
    for b in range(nband):
        if wsums[b]:
            mean_beam[b] /= wsums[b]
    mean_beam = da.stack(mean_beam)
    dirty, model, residual, psf, psfhat, mean_beam, wsums, dual = dask.compute(
                                                                dirty,
                                                                model,
                                                                residual,
                                                                psf,
                                                                psfhat,
                                                                mean_beam,
                                                                wsums,
                                                                dual)
    return (dirty, model, residual, psf, psfhat,
            mean_beam, wsums, dual)


def chunkify_rows(time, utimes_per_chunk, daskify_idx=False):
    utimes, time_bin_counts = np.unique(time, return_counts=True)
    n_time = len(utimes)
    if utimes_per_chunk == 0 or utimes_per_chunk == -1:
        utimes_per_chunk = n_time
    row_chunks = [np.sum(time_bin_counts[i:i+utimes_per_chunk])
                  for i in range(0, n_time, utimes_per_chunk)]
    time_bin_indices = np.zeros(n_time, dtype=np.int32)
    time_bin_indices[1::] = np.cumsum(time_bin_counts)[0:-1]
    time_bin_indices = time_bin_indices.astype(np.int32)
    time_bin_counts = time_bin_counts.astype(np.int32)
    if daskify_idx:
        time_bin_indices = da.from_array(time_bin_indices, chunks=utimes_per_chunk)
        time_bin_counts = da.from_array(time_bin_counts, chunks=utimes_per_chunk)
    return tuple(row_chunks), time_bin_indices, time_bin_counts


def l1reweight_func(model,
                    psiH=None,
                    outvar=None,
                    rmsfactor=None,
                    rms_comps=None,
                    alpha=4):
    '''
    The logic here is that weights should remain the same for model
    components that are rmsfactor times larger than the rms.
    High SNR values should experience relatively small thresholding
    whereas small values should be strongly thresholded
    '''
    psiH(model, outvar)
    mcomps = np.abs(np.sum(outvar, axis=0))
    # rms_comps can be per basis
    if isinstance(rms_comps, np.ndarray):
        return (1 + rmsfactor)/(1 + mcomps**alpha/rms_comps[:, None, None]**alpha)
    else:
        return (1 + rmsfactor)/(1 + mcomps**alpha/rms_comps**alpha)


# TODO - this can be done in parallel by splitting the image into facets
def fit_image_cube(time, freq, image, wgt=None, nbasist=None, nbasisf=None,
                   method='poly', sigmasq=0):
    '''
    Fit the time and frequency axes of an image cube where

    image   - (ntime, nband, nx, ny) pixelated image
    wgt     - (ntime, nband) optional per time and frequency weights
    nbasist - number of time basis functions
    nbasisf - number of frequency basis functions
    method  - method to use for fitting

    If wgt is not supplied equal weights are assumed.
    If nbasist/f are not supplied we return the coefficients
    of a bilinear fit regardless of method.

    methods:
    poly    - fit a monomials in time and frequency


    returns:
    coeffs  - fitted coefficients
    locx    - x pixel values
    locy    - y pixel values
    expr    - a string representing the symbolic expression describing the fit
    params  - tuple of str, parameters to pass into function (excluding t and f)
    ref_time    - reference time
    ref_freq    - reference frequency


    The fit is performed in scaled coordinates (t=time/ref_time,f=freq/ref_freq)
    '''
    ntime = time.size
    nband = freq.size
    ref_time = time[0]
    ref_freq = freq[0]
    import sympy as sm
    from sympy.abc import a, t, f

    if nbasist is None:
        nbasist = ntime
    else:
        assert nbasist <= ntime
    if nbasisf is None:
        nbasisf = nband
    else:
        assert nbasisf <= nband

    mask = np.any(image, axis=(0,1))  # over t and f axes
    Ix, Iy = np.where(mask)
    ncomps = Ix.size

    # components excluding zeros
    beta = image[:, :, Ix, Iy].reshape(ntime*nband, ncomps)
    if wgt is not None:
        wgt = wgt.reshape(ntime*nband, 1)
    else:
        wgt = np.ones((ntime*nband, 1), dtype=float)

    # nothing to fit
    if ntime==1 and nband==1:
        coeffs = beta
        expr = a
        params = (a,)
    elif method=='poly':
        wt = time/ref_time
        tfunc = t/ref_time
        Xfit = np.tile(wt[:, None], (nband, nbasist))**np.arange(nbasist)
        params = sm.symbols(f't(0:{nbasist})')
        expr = sum(co*t**i for i, co in enumerate(params))
        # the costant offset will always be included since nbasist is at least one
        if nband > 1:
            wf = freq/ref_freq
            ffunc = f/ref_freq
            Xf = np.tile(wf[:, None], (ntime, nbasisf-1))**np.arange(1, nbasisf)
            Xfit = np.hstack((Xfit, Xf))
            paramsf = sm.symbols(f'f(1:{nbasisf})')
            expr += sum(co*f**(i+1) for i, co in enumerate(paramsf))
            params += paramsf

    elif method=='Legendre':
        # scale to lie between -1,1 for stability
        if ntime > 1:
            tmax = time.max()
            tmin = time.min()
            wt = (time - (tmax + tmin)/2)
            wtmax = wt.max()
            wt /= wtmax
            # function to convert time to interp domain
            tfunc = (t - (tmax + tmin)/2)/wtmax
        else:
            wt = time
            tfunc = t
        Xt = np.zeros((ntime, nbasist), dtype=float)
        params = sm.symbols(f't(0:{nbasist})')
        if nbasist > 1:
            expr = 0
            for i in range(nbasist):
                vals = np.polynomial.Legendre.basis(i)(wt)
                Xt[:, i] = vals
                expr += sm.polys.orthopolys.legendre_poly(i, t)*params[i]
        else:
            Xt[...] = 1.0
            expr = params[0]
        Xfit = np.tile(Xt, (nband, 1))
        paramsf = sm.symbols(f'f(1:{nbasisf})')
        if nband > 1:
            Xf = np.zeros((nband, nbasisf - 1))
            fmax = freq.max()
            fmin = freq.min()
            wf = freq - (fmax + fmin)/2
            wfmax = wf.max()
            wf /= wfmax
            ffunc = (f - (fmax + fmin)/2)/wfmax
            for i in range(1, nbasisf):
                vals = np.polynomial.Legendre.basis(i)(wf)
                Xf[:, i-1] = vals
                expr += sm.polys.orthopolys.legendre_poly(i, f)*paramsf[i-1]
            Xf = np.tile(Xf, (ntime, 1))
            Xfit = np.hstack((Xfit, Xf))
            params += paramsf
    else:
        raise NotImplementedError("Please help us!")

    dirty_coeffs = Xfit.T.dot(wgt*beta)
    hess_coeffs = Xfit.T.dot(wgt*Xfit)
    # to improve conditioning
    if sigmasq:
        hess_coeffs += sigmasq*np.eye(hess_coeffs.shape[0])
    coeffs = np.linalg.solve(hess_coeffs, dirty_coeffs)


    return coeffs, Ix, Iy, str(expr), list(map(str,params)), str(tfunc),str(ffunc)


def eval_coeffs_to_cube(time, freq, nx, ny, coeffs, Ix, Iy,
                        expr, paramf, texpr, fexpr):
    ntime = time.size
    nfreq = freq.size

    image = np.zeros((ntime, nfreq, nx, ny), dtype=float)
    params = sm.symbols(('t','f'))
    params += sm.symbols(tuple(paramf))
    symexpr = parse_expr(expr)
    modelf = lambdify(params, symexpr)
    texpr = parse_expr(texpr)
    tfunc = lambdify(params[0], texpr)
    fexpr = parse_expr(fexpr)
    ffunc = lambdify(params[1], fexpr)
    for i, tval in enumerate(time):
        for j, fval in enumerate(freq):
            image[i, j, Ix, Iy] = modelf(tfunc(tval), ffunc(fval), *coeffs)

    return image


def eval_coeffs_to_slice(time, freq, coeffs, Ix, Iy,
                         expr, paramf, texpr, fexpr,
                         nxi, nyi, cellxi, cellyi, x0i, y0i,
                         nxo, nyo, cellxo, cellyo, x0o, y0o):

    image_in = np.zeros((nxi, nyi), dtype=float)
    params = sm.symbols(('t','f'))
    params += sm.symbols(tuple(paramf))
    symexpr = parse_expr(expr)
    modelf = lambdify(params, symexpr)
    texpr = parse_expr(texpr)
    tfunc = lambdify(params[0], texpr)
    fexpr = parse_expr(fexpr)
    ffunc = lambdify(params[1], fexpr)
    image_in[Ix, Iy] = modelf(tfunc(time), ffunc(freq), *coeffs)

    pix_area_in = cellxi * cellyi
    pix_area_out = cellxo * cellyo
    area_ratio = pix_area_out/pix_area_in

    xin = (-(nxi//2) + np.arange(nxi))*cellxi + x0i
    yin = (-(nyi//2) + np.arange(nyi))*cellyi + y0i
    xo = (-(nxo//2) + np.arange(nxo))*cellxo + x0o
    yo = (-(nyo//2) + np.arange(nyo))*cellyo + y0o

    # how many pixels to pad by to extrapolate with zeros
    xldiff = xin.min() - xo.min()
    if xldiff > 0.0:
        npadxl = int(np.ceil(xldiff/cellxi))
    else:
        npadxl = 0
    yldiff = yin.min() - yo.min()
    if yldiff > 0.0:
        npadyl = int(np.ceil(yldiff/cellyi))
    else:
        npadyl = 0

    xudiff = xo.max() - xin.max()
    if xudiff > 0.0:
        npadxu = int(np.ceil(xudiff/cellxi))
    else:
        npadxu = 0
    yudiff = yo.max() - yin.max()
    if yudiff > 0.0:
        npadyu = int(np.ceil(yudiff/cellyi))
    else:
        npadyu = 0

    do_pad = npadxl > 0
    do_pad |= npadxu > 0
    do_pad |= npadyl > 0
    do_pad |= npadyu > 0
    if do_pad:
        image_in = np.pad(image_in,
                        ((npadxl, npadxu), (npadyl, npadyu)),
                        mode='constant')

        xin = (-(nxi//2+npadxl) + np.arange(nxi + npadxl + npadxu))*cellxi + x0i
        nxi = nxi + npadxl + npadxu
        yin = (-(nyi//2+npadyl) + np.arange(nyi + npadyl + npadyu))*cellyi + y0i
        nyi = nyi + npadyl + npadyu

    do_interp = cellxi != cellxo
    do_interp |= cellyi != cellyo
    do_interp |= x0i != x0o
    do_interp |= y0i != y0o
    do_interp |= nxi != nxo
    do_interp |= nyi != nyo
    if do_interp:
        interpo = RegularGridInterpolator((xin, yin), image_in,
                                          bounds_error=True, method='linear')
        xx, yy = np.meshgrid(xo, yo, indexing='ij')
        return interpo((xx, yy)) * area_ratio
    else:
        return image_in


def model_from_mds(mds_name):
    '''
    Evaluate component model at the original resolution
    '''
    mds = xr.open_zarr(mds_name, chunks=None)
    return eval_coeffs_to_cube(mds.times.values,
                               mds.freqs.values,
                               mds.npix_x, mds.npix_y,
                               mds.coefficients.values,
                               mds.location_x.values, mds.location_y.values,
                               mds.parametrisation,
                               mds.params.values,
                               mds.texpr, mds.fexpr)



@njit(nogil=True, cache=True)
def norm_diff(x, xp):
    return norm_diff_impl(x, xp)


def norm_diff_impl(x, xp):
    return NotImplementedError


# @overload(norm_diff_impl, jit_options={**JIT_OPTIONS, "parallel":True})
@overload(norm_diff_impl, jit_options={**JIT_OPTIONS})  # parallel reduction slower?
def nb_norm_diff_impl(x, xp):
    if x.ndim==3:
        def impl(x, xp):
            nband, nx, ny = x.shape
            num = 0.0
            den = 1e-12  # avoid div by zero
            for b in range(nband):
                for i in prange(nx):
                    for j in range(ny):
                        num += (x[b, i, j] - xp[b, i, j])**2
                        den += x[b, i, j]**2
            return np.sqrt(num/den)
    elif x.ndim==2:
        def impl(x, xp):
            nx, ny = x.shape
            num = 0.0
            den = 1e-12  # avoid div by zero
            for i in prange(nx):
                for j in range(ny):
                    num += (x[i, j] - xp[i, j])**2
                    den += x[i, j]**2
            return np.sqrt(num/den)
    else:
        raise ValueError("norm_diff is only implemented for 2D or 3D arrays")

    return impl



def remove_large_islands(x, max_island_size=100):
    islands = label(x.squeeze())
    num_islands = islands.max()
    for i in range(1,num_islands+1):
        msk = islands == i
        num_pix = np.sum(msk)
        if num_pix > max_island_size:
            x[msk] = 0.0
    return x


@njit(parallel=False, nogil=True, cache=True, inline='always')
def freqmul(A, x):
    nband, nx, ny = x.shape
    out = np.zeros_like(x)
    for k in range(nband):
        for l in range(nband):
            for i in range(nx):
                for j in range(ny):
                    out[k, i, j] += A[k, l] * x[l, i, j]
    return out


def setup_parametrisation(mode='id', minval=1e-5,
                          sigma=1.0, freq=None, lscale=1.0):
    '''
    Given a parametrisation x = f(s) return:

    func - operator that evaluates x
    finv - operator that evaluates s = f^{-1}(x)
    dfunc - operator that evaluates dx/ds at fixed x = x0
    dhfunc - operator that evaluates the adjoint of dfunc at x = x0
    '''
    nu = freq/np.mean(freq)
    nband = nu.size
    nudiffsq = (nu[:, None] - nu[None, :])**2
    K = sigma**2 * np.exp(-nudiffsq/(2*lscale**2))
    L = np.linalg.cholesky(K + 1e-10*np.eye(nband))
    LH = L.T
    if mode == 'id':
        def func(x):
            return freqmul(L, x)

        def finv(x):
            return solve_triangular(L, x, lower=True)

        def dfunc(x0, v):
            return freqmul(L, v)

        def dhfunc(x0, v):
            return freqmul(LH, v)
    elif mode == 'exp':
        def func(x):
            return np.exp(freqmul(L, x))

        def finv(x):
            tmp = solve_triangular(L, x, lower=True)
            return np.log(np.maximum(np.abs(tmp), minval))

        def dfunc(x0, v):
            return np.exp(freqmul(L, x0)) * freqmul(L, v)

        def dhfunc(x0, v):
            return freqmul(LH, v * np.exp(freqmul(L, x0)))

    else:
        raise ValueError(f"Unknown mode - {mode}")

    return func, finv, dfunc, dhfunc


def weight_from_sigma(sigma):
    weight = ne.evaluate('1.0/(s*s)',
                         local_dict={'s':sigma},
                         casting='same_kind')
    return weight


def combine_columns(x, y, dc, dc1, dc2):
    '''
    x   - dask array containing dc1
    y   - dask array containing dc2
    dc  - string that numexpr can evaluate
    dc1 - name of x
    dc2 - name of y
    '''
    ne.evaluate(dc,
                local_dict={dc1: x, dc2: y},
                out=x,
                casting='same_kind')
    return x


def set_image_size(
                uv_max,
                max_freq,
                field_of_view,
                super_resolution_factor,
                cell_size=None,
                nx=None, ny=None,
                psf_oversize=2.0):
    # max cell size
    cell_N = 1.0 / (2 * uv_max * max_freq / lightspeed)

    if cell_size is not None:
        cell_rad = cell_size * np.pi / 60 / 60 / 180
        if cell_N / cell_rad < 1:
            raise ValueError("Requested cell size too large. "
                             "Super resolution factor = ", cell_N / cell_rad)

    else:
        cell_rad = cell_N / super_resolution_factor
        cell_size = cell_rad * 60 * 60 * 180 / np.pi


    if nx is None:
        fov = field_of_view * 3600
        npix = int(fov / cell_size)
        npix = good_size(npix)
        while npix % 2:
            npix += 1
            npix = good_size(npix)
        nx = npix
        ny = npix
    else:
        nx = nx
        ny = ny if ny is not None else nx
        cell_deg = np.rad2deg(cell_rad)
        fovx = nx*cell_deg
        fovy = ny*cell_deg

    nx_psf = good_size(int(psf_oversize * nx))
    while nx_psf % 2:
        nx_psf += 1
        nx_psf = good_size(nx_psf)

    ny_psf = good_size(int(psf_oversize * ny))
    while ny_psf % 2:
        ny_psf += 1
        ny_psf = good_size(ny_psf)
    nyo2 = ny_psf//2 + 1

    return nx, ny, nx_psf, ny_psf, cell_N, cell_rad

def taperf(shape, taper_width):
    height, width = shape
    array = np.ones((height, width))

    # Create 1D taper for both dimensions
    taper_x = np.ones(width)
    taper_y = np.ones(height)

    # Apply cosine taper to the edges
    taper_x[:taper_width] = 0.5 * (1 + np.cos(np.linspace(1.1*np.pi, 2*np.pi, taper_width)))
    taper_x[-taper_width:] = 0.5 * (1 + np.cos(np.linspace(0, 0.9*np.pi, taper_width)))

    taper_y[:taper_width] = 0.5 * (1 + np.cos(np.linspace(1.1*np.pi, 2*np.pi, taper_width)))
    taper_y[-taper_width:] = 0.5 * (1 + np.cos(np.linspace(0, 0.9*np.pi, taper_width)))

    return np.outer(taper_y, taper_x)


@njit(parallel=True)
def parallel_standard_normal(shape):
    rows, cols = shape
    result = np.empty(shape, dtype=np.float64)

    for i in prange(rows):
        for j in range(cols):
            result[i, j] = np.random.standard_normal()

    return result


def taperf(shape, taper_width):
    tapers1d = ()
    for npix in shape:
        taper = np.ones(npix)
        taper[:taper_width] = 0.5 * (1 + np.cos(np.linspace(1.1*np.pi, 2*np.pi, taper_width)))
        taper[-taper_width:] = 0.5 * (1 + np.cos(np.linspace(0, 0.9*np.pi, taper_width)))
        tapers1d += (taper,)
    return np.outer(*tapers1d)


# def fft_interp(image, cellxi, cellyi, nxo, nyo,
#                cellxo, cellyo, shiftx, shifty):
#     '''
#     Use non-uniform fft to interpolate image in a flux conservative way

#     image   - input image
#     cellxi  - input x cell-size
#     cellyi  - input y cell-size
#     nxo     - number of x pixels in output
#     nyo     - number of y pixels in output
#     cellxo  - output x cell size
#     cellyo  - output y cell size
#     shiftx  - shift x coordinate by this amount
#     shifty  - shift y coordinate by this amount

#     All sizes are assumed to be in radians.
#         '''
#     import matplotlib.pyplot as plt
#     from scipy.fft import fftn, ifftn
#     Fs = np.fft.fftshift
#     iFs = np.fft.ifftshift
#     # basic
#     nx, ny = image.shape
#     imhat = Fs(fftn(image))

#     imabs = np.abs(imhat)
#     imphase = np.angle(imhat) - 1.0
#     # imphase = np.roll(imphase, nx//2, axis=0)
#     imshift = ifftn(iFs(imabs*np.exp(1j*imphase))).real

#     impad = np.pad(imhat, ((nx//2, nx//2), (ny//2, ny//2)), mode='constant')
#     imo = ifftn(iFs(impad)).real

#     print(np.sum(image) - np.sum(imo))

#     plt.figure(1)
#     plt.imshow(image/image.max(), vmin=0, vmax=1, interpolation=None)
#     plt.colorbar()
#     plt.figure(2)
#     plt.imshow(imo/imo.max(), vmin=0, vmax=1, interpolation=None)
#     plt.colorbar()
#     plt.figure(3)
#     plt.imshow(imshift/imshift.max() - image/image.max(), vmin=0, vmax=1, interpolation=None)
#     plt.colorbar()

#     plt.show()

    # # coordinates on input grid
    # nx, ny = image.shapeimhat
    # x = np.arange(-(nx//2), nx//2) * cellxi
    # y = np.arange(-(ny//2), ny//2) * cellyi
    # xx, yy = np.meshgrid(x, y, indexing='ij')

    # # frequencies on output grid
    # celluo = 1/(nxo*cellxo)
    # cellvo = 1/(nyo*cellyo)
    # uo = np.arange(-(nxo//2), nxo//2) * celluo/nxo
    # vo = np.arange(-(nyo//2), nyo//2) * cellvo/nyo

    # uu, vv = np.meshgrid(uo, vo, indexing='ij')
    # uv = np.vstack((uo, vo)).T


    # res1 = finufft.nufft2d3(xx.ravel(), yy.ravel(), image.ravel(), uu.ravel(), vv.ravel())



