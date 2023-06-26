import sys
import numpy as np
import numexpr as ne
from numba import jit, njit, prange
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
from scipy.optimize import curve_fit
from collections import namedtuple
from scipy.interpolate import RectBivariateSpline as rbs
from africanus.coordinates.coordinates import radec_to_lmn
import xarray as xr
import pdb
from quartical.utils.dask import Blocker


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
            from contextlib import nullcontext
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

@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
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


def to4d(data):
    if data.ndim == 4:
        return data
    elif data.ndim == 2:
        return data[None, None]
    elif data.ndim == 3:
        return data[None]
    elif data.ndim == 1:
        return data[None, None, None]
    else:
        raise ValueError("Only arrays with ndim <= 4 can be broadcast to 4D.")


def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.), normalise=True, nsigma=5):
    S0, S1, PA = GaussPar
    Smaj = S0  #np.maximum(S0, S1)
    Smin = S1  #np.minimum(S0, S1)
    print(f'using ex = {Smaj}, ey = {Smin}')
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
    ind = np.argwhere(xflat**2 + yflat**2 <= extent).squeeze()
    idx = ind[:, 0]
    idy = ind[:, 1]
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
    padding = ((0, 0), (npad_xl, npad_xr), (npad_yl, npad_yr))
    unpad_x = slice(npad_xl, -npad_xr)
    unpad_y = slice(npad_yl, -npad_yr)
    return padding, unpad_x, unpad_y


def convolve2gaussres(image, xx, yy, gaussparf, nthreads, gausspari=None,
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
    padding, unpad_x, unpad_y = get_padding_info(nx, ny, pfrac)
    ax = (1, 2)  # axes over which to perform fft
    lastsize = ny + np.sum(padding[-1])

    gausskern = Gaussian2D(xx, yy, gaussparf, normalise=norm_kernel)
    gausskern = np.pad(gausskern[None], padding, mode='constant')
    gausskernhat = r2c(iFs(gausskern, axes=ax), axes=ax, forward=True,
                       nthreads=nthreads, inorm=0)

    image = np.pad(image, padding, mode='constant')
    imhat = r2c(iFs(image, axes=ax), axes=ax, forward=True, nthreads=nthreads,
                inorm=0)

    # convolve to desired resolution
    if gausspari is None:
        imhat *= gausskernhat
    else:
        for i in range(nband):
            thiskern = Gaussian2D(xx, yy, gausspari[i], normalise=norm_kernel)
            thiskern = np.pad(thiskern[None], padding, mode='constant')
            thiskernhat = r2c(iFs(thiskern, axes=ax), axes=ax, forward=True,
                              nthreads=nthreads, inorm=0)

            convkernhat = np.where(np.abs(thiskernhat) > 0.0,
                                   gausskernhat / thiskernhat, 0.0)

            imhat[i] *= convkernhat[0]

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
                       freq_max=np.inf):
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
    gain_times = {}
    gain_freqs = {}
    gain_axes = {}
    gain_spec = {}
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
        spws = xds_from_table(ms + "::SPECTRAL_WINDOW")[0]
        pols = xds_from_table(ms + "::POLARIZATION")[0]
        ants = xds_from_table(ms + "::ANTENNA")[0]

        antpos[ms] = ants.POSITION.data
        poltype[ms] = fetch_poltype(pols.CORR_TYPE.data.squeeze())

        idts[ms] = []
        if gain_name is not None:
            gain = xds_from_zarr(gain_name[ims])
            gain_times[ms] = {}
            gain_freqs[ms] = {}
            gain_axes[ms] = {}
            gain_spec[ms] = {}

        freqs[ms] = {}
        times[ms] = {}
        radecs[ms] = {}
        chan_widths[ms] = {}
        for ids, ds in enumerate(xds):
            idt = f"FIELD{ds.FIELD_ID}_DDID{ds.DATA_DESC_ID}_SCAN{ds.SCAN_NUMBER}"
            idts[ms].append(idt)
            radecs[ms][idt] = fields.PHASE_DIR.data[ds.FIELD_ID].squeeze()

            freqs[ms][idt] = spws.CHAN_FREQ.data[ds.DATA_DESC_ID]
            chan_widths[ms][idt] = spws.CHAN_WIDTH.data[ds.DATA_DESC_ID]
            times[ms][idt] = da.atleast_1d(ds.TIME.data.squeeze())
            uvw = ds.UVW.data
            u_max = abs(uvw[:, 0].max())
            v_max = abs(uvw[:, 1]).max()
            uv_maxs.append(da.maximum(u_max, v_max))

            if gain_name is not None:
                gain_times[ms][idt] = gain[ids].gain_time.data
                gain_freqs[ms][idt] = gain[ids].gain_freq.data
                gain_axes[ms][idt] = gain[ids].GAIN_AXES
                gain_spec[ms][idt] = gain[ids].GAIN_SPEC


    # Early compute to get metadata
    times, freqs, gain_times, gain_freqs, gain_axes, gain_spec, radecs,\
        chan_widths, uv_maxs, antpos, poltype = dask.compute(times, freqs,\
        gain_times, gain_freqs, gain_axes, gain_spec, radecs, chan_widths,\
        uv_maxs, antpos, poltype)

    uv_max = max(uv_maxs)
    all_freqs = []
    all_times = []
    freq_mapping = {}
    row_mapping = {}
    time_mapping = {}
    utimes = {}
    ms_chunks = {}
    gain_chunks = {}
    for ms in ms_name:
        freq_mapping[ms] = {}
        row_mapping[ms] = {}
        time_mapping[ms] = {}
        utimes[ms] = {}
        ms_chunks[ms] = []
        gain_chunks[ms] = []
        for idt in idts[ms]:
            freq = freqs[ms][idt]
            if gain_name is not None:
                try:
                    assert (gain_freqs[ms][idt] == freq).all()
                except Exception as e:
                    raise ValueError(f'Mismatch between gain and MS '
                                     f'frequencies for {ms} at {idt}')

            idx = (freq>=freq_min) & (freq<=freq_max)
            if not idx.any():
                continue
            idx0 = idx.argmax()  # returns index of first True element
            freq = freq[idx]

            all_freqs.append(freq)
            nchan = freq.size
            if cpi in [-1, 0, None]:
                cpit = nchan
            else:
                cpit = np.minimum(cpi, nchan)
            freq_mapping[ms][idt] = {}
            tmp = np.arange(idx0, idx0 + nchan, cpit)
            freq_mapping[ms][idt]['start_indices'] = tmp
            if cpit != nchan:
                tmp2 = np.append(tmp, [nchan])
                freq_mapping[ms][idt]['counts'] = tmp2[1:] - tmp2[0:-1]
            else:
                freq_mapping[ms][idt]['counts'] = np.array((nchan,), dtype=int)

            time = times[ms][idt]
            utime = np.unique(time)
            if gain_name is not None:
                try:
                    assert (gain_times[ms][idt] == utime).all()
                except Exception as e:
                    raise ValueError(f'Mismatch between gain and MS '
                                     f'utimes for {ms} at {idt}')
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

            ms_chunks[ms].append({'row': row_chunks,
                                  'chan': tuple(freq_mapping[ms][idt]['counts'])})

            time_mapping[ms][idt] = {}
            tmp = np.arange(0, ntime, ipit)
            time_mapping[ms][idt]['start_indices'] = tmp
            if ipit != ntime:
                tmp2 = np.append(tmp, [ntime])
                time_mapping[ms][idt]['counts'] = tmp2[1:] - tmp2[0:-1]
            else:
                time_mapping[ms][idt]['counts'] = np.array((ntime,))

            # we may need to rechunk gains in time and freq to line up with MS
            if gain_name is not None:
                tmp_dict = {}
                for name, val in zip(gain_axes[ms][idt], gain_spec[ms][idt]):
                    if name == 'gain_time':
                        tmp_dict[name] = tuple(time_mapping[ms][idt]['counts'])
                    elif name == 'gain_freq':
                        tmp_dict[name] = tuple(freq_mapping[ms][idt]['counts'])
                    elif name == 'direction':
                        if len(val) > 1:
                            raise ValueError("DD gains not supported yet")
                        if val[0] > 1:
                            raise ValueError("DD gains not supported yet")
                        tmp_dict[name] = tuple(val)
                    else:
                        tmp_dict[name] = tuple(val)

                gain_chunks[ms].append(tmp_dict)

    return row_mapping, freq_mapping, time_mapping, \
           freqs, utimes, ms_chunks, gain_chunks, radecs, \
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

def fitcleanbeam(psf: np.ndarray,
                 level: float = 0.5,
                 pixsize: float = 1.0,
                 extent: float = 5.0):
    """
    Find the Gaussian that approximates the PSF.
    First find the main lobe by identifying where PSF > level
    then fit Gaussian out to a radius of extent * max(x, y) where
    x and y are the coordinates where PSF > level.
    """


    nband, nx, ny = psf.shape

    # coordinates
    x = np.arange(-nx / 2, nx / 2)
    y = np.arange(-ny / 2, ny / 2)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # model to fit
    def func(xy, emaj, emin, pa):
        Smin = np.minimum(emaj, emin)
        Smaj = np.maximum(emaj, emin)

        A = np.array([[1. / Smin ** 2, 0],
                      [0, 1. / Smaj ** 2]])

        c, s, t = np.cos, np.sin, np.deg2rad(-pa)
        R = np.array([[c(t), -s(t)],
                      [s(t), c(t)]])
        A = np.dot(np.dot(R.T, A), R)
        R = np.einsum('nb,bc,cn->n', xy.T, A, xy)
        # GaussPar should corresponds to FWHM
        fwhm_conv = 2 * np.sqrt(2 * np.log(2))
        return np.exp(-fwhm_conv * R)

    Gausspars = ()
    for v in range(nband):
        # make sure psf is normalised
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
        p, _ = curve_fit(func, xy, psfv, p0=(emaj0, emin0, 0.0))
        Gausspars += ((p[0] * pixsize, p[1] * pixsize, p[2]),)

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


def dds2cubes(dds, nband, apparent=False):
    real_type = dds[0].DIRTY.dtype
    complex_type = np.result_type(real_type, np.complex64)
    nx, ny = dds[0].DIRTY.shape
    dirty = [da.zeros((nx, ny), chunks=(-1, -1),
                      dtype=real_type) for _ in range(nband)]
    model = [da.zeros((nx, ny), chunks=(-1, -1),
                      dtype=real_type) for _ in range(nband)]
    if 'RESIDUAL' in dds[0]:
        residual = [da.zeros((nx, ny), chunks=(-1, -1),
                            dtype=real_type) for _ in range(nband)]
    else:
        residual = None
    wsums = [da.zeros(1, dtype=real_type) for _ in range(nband)]
    if 'PSF' in dds[0]:
        nx_psf, ny_psf = dds[0].PSF.shape
        nx_psf, nyo2_psf = dds[0].PSFHAT.shape
        psf = [da.zeros((nx_psf, ny_psf), chunks=(-1, -1),
                        dtype=real_type) for _ in range(nband)]
        psfhat = [da.zeros((nx_psf, nyo2_psf), chunks=(-1, -1),
                            dtype=complex_type) for _ in range(nband)]
    else:
        psf = None
        psfhat = None
    mean_beam = [da.zeros((nx, ny), chunks=(-1, -1),
                            dtype=real_type) for _ in range(nband)]
    if 'DUAL' in dds[0]:
        nbasis, nmax = dds[0].DUAL.shape
        dual = [da.zeros((nbasis, nmax), chunks=(-1, -1),
                            dtype=real_type) for _ in range(nband)]
    else:
        dual = None
    for ds in dds:
        b = ds.bandid
        if apparent:
            dirty[b] += ds.DIRTY.data
            if 'RESIDUAL' in ds:
                residual[b] += ds.RESIDUAL.data
        else:
            dirty[b] += ds.DIRTY.data * ds.BEAM.data
            if 'RESIDUAL' in ds:
                residual[b] += ds.RESIDUAL.data * ds.BEAM.data
        if 'PSF' in ds:
            psf[b] += ds.PSF.data
            psfhat[b] += ds.PSFHAT.data
        if 'MODEL' in ds:
            model[b] = ds.MODEL.data
        if 'DUAL' in ds:
            dual[b] = ds.DUAL.data
        mean_beam[b] += ds.BEAM.data * ds.WSUM.data[0]
        wsums[b] += ds.WSUM.data[0]
    wsums = da.stack(wsums).reshape(nband)
    wsum = wsums.sum()
    dirty = da.stack(dirty)/wsum
    model = da.stack(model)
    if 'RESIDUAL' in ds:
        residual = da.stack(residual)/wsum
    if 'PSF' in ds:
        psf = da.stack(psf)/wsum
        psfhat = da.stack(psfhat)/wsum
    if 'DUAL' in ds:
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
    return dirty, model, residual, psf, psfhat, mean_beam, wsums, dual


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


def rephase_vis(vis, uvw, radec_in, radec_out):
    return da.blockwise(_rephase_vis, 'rf',
                        vis, 'rf',
                        uvw, 'r3',
                        radec_in, None,
                        radec_out, None,
                        dtype=vis.dtype)

def _rephase_vis(vis, uvw, radec_in, radec_out):
    l_in, m_in, n_in = radec_to_lmn(radec_in)
    l_out, m_out, n_out = radec_to_lmn(radec_out)
    return vis * np.exp(1j*(uvw[:, 0]*(l_out-l_in) +
                            uvw[:, 1]*(m_out-m_in) +
                            uvw[:, 2]*(n_out-n_in)))


# TODO - concat functions should allow coarsening to values other than 1
def concat_row(xds):
    # TODO - how to compute average beam before we have access to grid?
    times_in = []
    freqs = []
    for ds in xds:
        times_in.append(ds.time_out)
        freqs.append(ds.freq_out)

    times_in = np.unique(times_in)
    freqs = np.unique(freqs)

    nband = freqs.size
    ntime_in = times_in.size

    if ntime_in == 1:  # no need to concatenate
        return xds

    # do merge manually because different variables require different
    # treatment anyway eg. the BEAM should be computed as a weighted sum
    xds_out = []
    for b in range(nband):
        xdsb = []
        times = []
        nu = freqs[b]
        for ds in xds:
            if ds.freq_out == nu:
                xdsb.append(ds)
                times.append(ds.time_out)

        wgt = [ds.WEIGHT for ds in xdsb]
        vis = [ds.VIS for ds in xdsb]
        mask = [ds.MASK for ds in xdsb]
        uvw = [ds.UVW for ds in xdsb]

        wgto = xr.concat(wgt, dim='row')
        viso = xr.concat(vis, dim='row')
        masko = xr.concat(mask, dim='row')
        uvwo = xr.concat(uvw, dim='row')

        xdso = xr.merge((wgto, viso, masko, uvwo))
        xdso['BEAM'] = xdsb[0].BEAM  # we need the grid to do this properly
        xdso['FREQ'] = xdsb[0].FREQ  # is this always going to be the case?

        xdso = xdso.chunk({'row':-1})

        xdso = xdso.assign_coords({
            'chan': (('chan',), xdsb[0].chan.data)
        })

        times = np.array(times)
        tout = np.round(np.mean(times), 5)  # avoid precision issues
        xdso = xdso.assign_attrs({
            'dec': xdsb[0].dec,  # always the case?
            'ra': xdsb[0].ra,    # always the case?
            'time_out': tout,
            'time_max': times.max(),
            'time_min': times.min(),
            'timeid': 0,
            'freq_out': nu,
            'freq_max': xdsb[0].freq_max,
            'freq_min': xdsb[0].freq_min,
        })
        xds_out.append(xdso)
    return xds_out


def concat_chan(xds, nband_out=1):
    times = []
    freqs_in = []
    freqs_min = []
    freqs_max = []
    all_freqs = []
    for ds in xds:
        times.append(ds.time_out)
        freqs_in.append(ds.freq_out)
        freqs_min.append(ds.freq_min)
        freqs_max.append(ds.freq_max)
        all_freqs.append(ds.chan)

    times = np.unique(times)
    freqs_in = np.unique(freqs_in)
    freqs_min = np.unique(freqs_min)
    freqs_max = np.unique(freqs_max)
    all_freqs = np.unique(all_freqs)

    nband_in = freqs_in.size
    ntime = times.size

    if nband_in == nband_out or nband_in == 1:  # no need to concatenate
        return xds

    # currently assuming linearly spaced frequencies
    freq_bins = np.linspace(freqs_min.min(), freqs_max.max(), nband_out+1)
    bin_centers = (freq_bins[1:] + freq_bins[0:-1])/2

    xds_out = []
    for t in range(ntime):
        time = times[t]
        for b in range(nband_out):
            xdst = []
            flow = freq_bins[b]
            fhigh = freq_bins[b+1]
            freqsb = all_freqs[all_freqs >= flow]
            freqsb = freqsb[freqsb < fhigh]
            for ds in xds:
                # ds overlaps output if either ds.freq_min or ds.freq_max lies in the bin
                low_in = ds.freq_min > flow and ds.freq_min < fhigh
                high_in = ds.freq_max > flow and ds.freq_max < fhigh
                if ds.time_out == time and (low_in or high_in):
                    xdst.append(ds)

            # LB - we should be able to avoid this stack operation by using Jon's *() magic
            wgt = da.stack([ds.WEIGHT.data for ds in xdst]).rechunk(-1, -1, -1)
            vis = da.stack([ds.VIS.data for ds in xdst]).rechunk(-1, -1, -1)
            mask = da.stack([ds.MASK.data for ds in xdst]).rechunk(-1, -1, -1)
            freq = da.stack([ds.FREQ.data for ds in xdst]).rechunk(-1, -1)

            nrow = xdst[0].row.size
            nchan = freqsb.size

            freqs_dask = da.from_array(freqsb, chunks=nchan)
            blocker = Blocker(sum_overlap, 's')
            blocker.add_input('vis', vis, 'src')
            blocker.add_input('wgt', wgt, 'src')
            blocker.add_input('mask', mask, 'src')
            blocker.add_input('freq', freq, 'sc')
            blocker.add_input('ufreq', freqs_dask, 'f')
            blocker.add_input('flow', flow, None)
            blocker.add_input('fhigh', fhigh, None)
            blocker.add_output('viso', 'rf', ((nrow,), (nchan,)), vis.dtype)
            blocker.add_output('wgto', 'rf', ((nrow,), (nchan,)), wgt.dtype)
            blocker.add_output('masko', 'rf', ((nrow,), (nchan,)), mask.dtype)

            out_dict = blocker.get_dask_outputs()

            data_vars = {
                'VIS': (('row', 'chan'), out_dict['viso']),
                'WEIGHT': (('row', 'chan'), out_dict['wgto']),
                'MASK': (('row', 'chan'), out_dict['masko']),
                'FREQ': (('chan',), freqs_dask),
                'UVW': (('row', 'three'), xdst[0].UVW.data), # should be the same across data sets
                'BEAM': (('scalar',), xdst[0].BEAM.data)  # need to pass in the grid to do this properly
            }

            coords = {
                'chan': (('chan',), freqsb)
            }

            fout = np.round(bin_centers[b], 5)  # avoid precision issues
            attrs = {
                'freq_out': fout,
                'bandid': b,
                'dec': xdst[0].dec,
                'ra': xdst[0].ra,
                'time_out': time
            }

            xdso = xr.Dataset(data_vars=data_vars,
                              attrs=attrs)

            xds_out.append(xdso)
    return xds_out


def sum_overlap(vis, wgt, mask, freq, ufreq, flow, fhigh):
    nds = vis.shape[0]

    # output grids
    nchan = ufreq.size
    nrow = vis.shape[1]
    viso = np.zeros((nrow, nchan), dtype=vis.dtype)
    wgto = np.zeros((nrow, nchan), dtype=wgt.dtype)
    masko = np.zeros((nrow, nchan), dtype=mask.dtype)

    # weighted sum at overlap
    for i in range(nds):
        nu = freq[i]
        _, idx0, idx1 = np.intersect1d(nu, ufreq, assume_unique=True, return_indices=True)
        viso[:, idx1] += vis[i][:, idx0] * wgt[i][:, idx0] * mask[i][:, idx0]
        wgto[:, idx1] += wgt[i][:, idx0] * mask[i][:, idx0]
        masko[:, idx1] += mask[i][:, idx0]

    # unmasked where at least one data point is unflagged
    masko = np.where(masko > 0, 1, 0)
    viso[masko.astype(bool)] = viso[masko.astype(bool)]/wgto[masko.astype(bool)]

    # blocker expects a dictionary as output
    out_dict = {}
    out_dict['viso'] = viso
    out_dict['wgto'] = wgto
    out_dict['masko'] = masko

    return out_dict


def l1reweight_func(psiH, outvar, rmsfactor, rms_comps, model):
    '''
    The logic here is that weights should remain the same for model
    components that are rmsfactor times larger than the rms.
    High SNR values should experience relatively small thresholding
    whereas small values should be strongly thresholded
    '''
    psiH(model, outvar)
    mcomps = np.sum(outvar, axis=0)
    # the **2 here results in more agressive reweighting
    return (1 + rmsfactor)/(1 + mcomps**4/rms_comps**4)

