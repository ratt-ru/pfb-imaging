import sys
import numpy as np
import numexpr as ne
from numba import jit, njit
import dask
import dask.array as da
from dask.distributed import performance_report
from dask.diagnostics import ProgressBar
from ducc0.fft import r2c, c2r, good_size
iFs = np.fft.ifftshift
Fs = np.fft.fftshift
from numba.core.extending import SentryLiteralArgs
import inspect
from daskms import xds_from_storage_ms as xds_from_ms
from daskms import xds_from_storage_table as xds_from_table
from daskms.experimental.zarr import xds_from_zarr
from omegaconf import ListConfig
from skimage.morphology import label
from scipy.optimize import curve_fit
from collections import namedtuple
from scipy.interpolate import RectBivariateSpline as rbs
from africanus.coordinates.coordinates import radec_to_lmn

def interp_cube(model, wsums, infreqs, outfreqs, ref_freq, spectral_poly_order):
    nband, nx, ny = model
    mask = np.any(model, axis=0)
    # components excluding zeros
    beta = model[:, mask]
    if spectral_poly_order > infreqs.size:
        raise ValueError("spectral-poly-order can't be larger than nband")

    # we are given frequencies at bin centers, convert to bin edges
    delta_freq = infreqs[1] - infreqs[0]
    wlow = (infreqs - delta_freq/2.0)/ref_freq
    whigh = (infreqs + delta_freq/2.0)/ref_freq
    wdiff = whigh - wlow

    # set design matrix for each component
    # look at Offringa and Smirnov 1706.06786
    Xfit = np.zeros([nband, order])
    for i in range(1, order+1):
        Xfit[:, i-1] = (whigh**i - wlow**i)/(i*wdiff)

    # we want to fit a function modeli = Xfit comps
    # where Xfit is the design matrix corresponding to an integrated
    # polynomial model. The normal equations tells us
    # comps = (Xfit.T wsums Xfit)**{-1} Xfit.T wsums modeli
    # (Xfit.T wsums Xfit) == hesscomps
    # Xfit.T wsums modeli == dirty_comps

    dirty_comps = Xfit.T.dot(wsums*beta)

    hess_comps = Xfit.T.dot(wsums*Xfit)

    comps = np.linalg.solve(hess_comps, dirty_comps)

    # now we want to evaluate the unintegrated polynomial coefficients
    # the corresponding design matrix is constructed for a polynomial of
    # the form
    # modeli = comps[0]*1 + comps[1] * w + comps[2] w**2 + ...
    # where w = outfreqs/ref_freq
    w = outfreqs/ref_freq
    # nchan = outfreqs
    # Xeval = np.zeros((nchan, order))
    # for c in range(nchan):
    #     Xeval[:, c] = w**c
    Xeval = np.tile(w, order)**np.arange(order)

    betaout = Xeval.dot(comps)

    modelout = np.zeros((nchan, nx, ny))
    modelout[:, mask] = betaout

    return modelout

def compute_context(scheduler, output_filename):
    if scheduler == "distributed":
        return performance_report(filename=output_filename + "_dask_report.html")
    else:
        return ProgressBar()

def estimate_data_size(nant, nhr, nsec, nchan, ncorr, nbytes):
    '''
    Estimates size of data in GB where:

    nant    - number of antennas
    nhr     - length of observation in hours
    nsec    - integration time in seconds
    nchan   - number of channels
    ncorr   - number of correlations
    nbytes  - bytes per item (eg. 8 for complex64)
    '''
    nbl = nant * (nant - 1) // 2
    ntime = nhr * 3600 // nsec
    return nbl * ntime * nchan * ncorr * nbytes / 1e9


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
    Smaj = np.maximum(S0, S1)
    Smin = np.minimum(S0, S1)
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

def chan_to_band_mapping(ms_name, nband=None,
                         group_by=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER']):
    '''
    Construct dictionaries containing per MS and SPW channel to band mapping.
    Currently assumes we are only imaging field 0 of the first MS.

    Input:
    ms_name     - list of ms names
    nband       - number of imaging bands
    group_by    - dataset grouping

    Output:
    freqs           - dict[MS][IDENTITY] chunked dask arrays of the freq to band mapping
    freq_bin_idx    - dict[MS][IDENTITY] chunked dask arrays of bin starting indices
    freq_bin_counts - dict[MS][IDENTITY] chunked dask arrays of counts in each bin
    freq_out        - frequencies of average
    band_mapping    - dict[MS][IDENTITY] identifying imaging bands going into degridder
    chan_chunks     - dict[MS][IDENTITY] specifies dask chunking scheme over channel

    where IDENTITY is constructed from the FIELD/DDID and SCAN ID's.
    '''

    if not isinstance(ms_name, list) and not isinstance(ms_name, ListConfig) :
        ms_name = [ms_name]

    # first pass through data to determine freq_mapping
    radec = None
    freqs = {}
    all_freqs = []
    for ims in ms_name:
        xds = xds_from_ms(ims, chunks={"row": -1}, columns=('TIME',),
                          group_cols=group_by)

        # subtables
        ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
        fields = xds_from_table(ims + "::FIELD")
        spws_table = xds_from_table(ims + "::SPECTRAL_WINDOW")
        pols = xds_from_table(ims + "::POLARIZATION")

        # subtable data
        ddids = dask.compute(ddids)[0]
        fields = dask.compute(fields)[0]
        spws_table = dask.compute(spws_table)[0]
        pols = dask.compute(pols)[0]

        freqs[ims] = {}
        for ds in xds:
            identity = f"FIELD{ds.FIELD_ID}_DDID{ds.DATA_DESC_ID}_SCAN{ds.SCAN_NUMBER}"
            field = fields[ds.FIELD_ID]

            # check fields match
            if radec is None:
                radec = field.PHASE_DIR.data.squeeze()

            if not np.array_equal(radec, field.PHASE_DIR.data.squeeze()):
                continue

            spw = spws_table[ds.DATA_DESC_ID]
            tmp_freq = spw.CHAN_FREQ.data.squeeze()
            freqs[ims][identity] = tmp_freq
            all_freqs.append(list([tmp_freq]))


    # freq mapping
    all_freqs = dask.compute(all_freqs)
    ufreqs = np.unique(all_freqs)  # sorted ascending
    nchan = ufreqs.size
    if nband in [None, -1]:
        nband = nchan
    else:
       nband = nband

    # bin edges
    fmin = ufreqs[0]
    fmax = ufreqs[-1]
    fbins = np.linspace(fmin, fmax, nband + 1)
    freq_out = np.zeros(nband)
    chan_count = 0
    for band in range(nband):
        indl = ufreqs >= fbins[band]
        # inclusive except for the last one
        if band == nband-1:
            indu = ufreqs <= fbins[band + 1]
        else:
            indu = ufreqs < fbins[band + 1]
        freq_out[band] = np.mean(ufreqs[indl&indu])
        chan_count += ufreqs[indl&indu].size

    if chan_count < nchan:
        raise RuntimeError("Something has gone wrong with the chan <-> band "
                           "mapping. This is probably a bug.")

    # chan <-> band mapping
    band_mapping = {}
    chan_chunks = {}
    freq_bin_idx = {}
    freq_bin_counts = {}
    for ims in freqs:
        freq_bin_idx[ims] = {}
        freq_bin_counts[ims] = {}
        band_mapping[ims] = {}
        chan_chunks[ims] = {}
        for idt in freqs[ims]:
            freq = np.atleast_1d(dask.compute(freqs[ims][idt])[0])
            band_map = np.zeros(freq.size, dtype=np.int32)
            for band in range(nband):
                indl = freq >= fbins[band]
                if band == nband-1:
                    indu = freq <= fbins[band + 1]
                else:
                    indu = freq < fbins[band + 1]
                band_map = np.where(indl & indu, band, band_map)
            # to dask arrays
            bands, bin_counts = np.unique(band_map, return_counts=True)
            band_mapping[ims][idt] = da.from_array(bands, chunks=1)
            chan_chunks[ims][idt] = tuple(bin_counts)
            freqs[ims][idt] = da.from_array(freq, chunks=tuple(bin_counts))
            bin_idx = np.append(np.array([0]), np.cumsum(bin_counts))[0:-1]
            freq_bin_idx[ims][idt] = da.from_array(bin_idx, chunks=1)
            freq_bin_counts[ims][idt] = da.from_array(bin_counts, chunks=1)

    return freqs, freq_bin_idx, freq_bin_counts, freq_out, band_mapping, chan_chunks

@dask.delayed
def fetch_poltype(corr_type):
    corr_type = set(tuple(corr_type))
    if corr_type.issubset(set([9, 10, 11, 12])):
        return 'linear'
    elif corr_type.issubset(set([5, 6, 7, 8])):
        return 'circular'


def construct_mappings(ms_name, gain_name=None, nband=None, ipi=None):
    '''
    Construct dictionaries containing per MS, FIELD, DDID and SCAN
    time and frequency mappings.

    Input:
    ms_name     - list of ms names
    nband       - number of imaging bands (defaults to a single band)
    ipi         - integrations (i.e. unique times) per output image.
                  Defaults to one per scan.

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
    idts = []
    for ims, ms in enumerate(ms_name):
        xds = xds_from_ms(ms, chunks={"row": -1}, columns=('TIME','UVW'),
                          group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])

        # subtables
        ddids = xds_from_table(ms + "::DATA_DESCRIPTION")
        fields = xds_from_table(ms + "::FIELD")
        spws = xds_from_table(ms + "::SPECTRAL_WINDOW")
        pols = xds_from_table(ms + "::POLARIZATION")
        ants = xds_from_table(ms + "::ANTENNA")

        antpos[ms] = ants[0].POSITION.data
        poltype[ms] = fetch_poltype(pols[0].CORR_TYPE.data.squeeze())


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
            idts.append(idt)
            field = fields[ds.FIELD_ID]
            radecs[ms][idt] = field.PHASE_DIR.data.squeeze()

            spw = spws[ds.DATA_DESC_ID]
            freqs[ms][idt] = da.atleast_1d(spw.CHAN_FREQ.data.squeeze())
            chan_widths[ms][idt] = da.atleast_1d(spw.CHAN_WIDTH.data.squeeze())
            times[ms][idt] = da.atleast_1d(ds.TIME.data.squeeze())
            uvw = ds.UVW.data
            u_max = abs(uvw[:, 0].max())
            v_max = abs(uvw[:, 1]).max()
            uv_maxs.append(da.maximum(u_max, v_max))

            if gain_name is not None:
                gain_times[ms][idt] = gain[ids].gain_t.data
                gain_freqs[ms][idt] = gain[ids].gain_f.data
                gain_axes[ms][idt] = gain[ids].GAIN_AXES
                gain_spec[ms][idt] = gain[ids].GAIN_SPEC


    # Early compute to get metadata
    times, freqs, gain_times, gain_freqs, gain_axes, gain_spec, radecs,\
        chan_widths, uv_maxs, antpos, poltype = dask.compute(times, freqs,\
        gain_times, gain_freqs, gain_axes, gain_spec, radecs, chan_widths,\
        uv_maxs, antpos, poltype)

    uv_max = max(uv_maxs)

    # Is this sufficient to make sure the MSs and gains are aligned?
    all_freqs = []
    utimes = {}
    ntimes_out = 0
    for ms in ms_name:
        utimes[ms] = {}
        for idt in idts:
            freq = freqs[ms][idt]
            if gain_name is not None:
                try:
                    assert (gain_freqs[ms][idt] == freq).all()
                except Exception as e:
                    raise ValueError(f'Mismatch between gain and MS '
                                     f'frequencies for {ms} at {idt}')
            all_freqs.append(freq)
            utime = np.unique(times[ms][idt])
            if gain_name is not None:
                try:
                    assert (gain_times[ms][idt] == utime).all()
                except Exception as e:
                    raise ValueError(f'Mismatch between gain and MS '
                                     f'utimes for {ms} at {idt}')  #WTF!!
            utimes[ms][idt] = utime
            if ipi in [0, -1, None]:
                ntimes_out += 1
            else:
                ntimes_out += np.ceil(utime.size/ipi)

    # freq mapping
    ufreqs = np.unique(all_freqs)  # sorted ascending
    nchan = ufreqs.size
    if nband is None:
        nband = 1
    else:
       nband = nband

    # should we use bin edges here? what about inhomogeneous channel widths?
    fmin = ufreqs[0]
    fmax = ufreqs[-1]
    fbins = np.linspace(fmin, fmax, nband + 1)

    freq_out = np.zeros(nband)
    chan_count = 0
    for band in range(nband):
        indl = ufreqs >= fbins[band]
        # inclusive except for the last one
        if band == nband-1:
            indu = ufreqs <= fbins[band + 1]
        else:
            indu = ufreqs < fbins[band + 1]
        freq_out[band] = np.mean(ufreqs[indl&indu])
        chan_count += ufreqs[indl&indu].size

    if chan_count < nchan:
        raise RuntimeError("Something has gone wrong with the chan <-> band "
                           "mapping. This is probably a bug.")

    band_mapping = {}
    fbin_idx = {}
    fbin_counts = {}
    for ms in ms_name:
        fbin_idx[ms] = {}
        fbin_counts[ms] = {}
        band_mapping[ms] = {}
        for idt in idts:
            freq = freqs[ms][idt]
            band_map = np.zeros(freq.size, dtype=np.int32)
            for band in range(nband):
                indl = freq >= fbins[band]
                if band == nband-1:
                    indu = freq <= fbins[band + 1]
                else:
                    indu = freq < fbins[band + 1]
                band_map = np.where(indl & indu, band, band_map)

            bands, bin_counts = np.unique(band_map, return_counts=True)
            band_mapping[ms][idt] = bands
            bin_idx = np.append(np.array([0]), np.cumsum(bin_counts))[0:-1]
            fbin_idx[ms][idt] = bin_idx
            fbin_counts[ms][idt] = bin_counts

    # This logic does not currently handle overlapping scans
    tbin_idx = {}
    tbin_counts = {}
    time_mapping = {}
    ms_chunks = {}
    gain_chunks = {}
    ti = 0
    for ims, ms in enumerate(ms_name):
        tbin_idx[ms] = {}
        tbin_counts[ms] = {}
        time_mapping[ms] = {}
        ms_chunks[ms] = []
        gain_chunks[ms] = []
        for idt in idts:
            time = times[ms][idt]
            # has to be here since scans not same length
            ntime = utimes[ms][idt].size
            if ipi in [0, -1, None]:
                ipit = ntime
            else:
                ipit = ipi
            row_chunks, tidx, tcounts = chunkify_rows(time, ipit,
                                                      daskify_idx=False)
            tbin_idx[ms][idt] = tidx
            tbin_counts[ms][idt] = tcounts
            time_mapping[ms][idt] = {}
            time_mapping[ms][idt]['low'] = np.arange(0, ntime, ipit)
            hmap = np.append(np.arange(ipit, ntime, ipit), ntime)
            time_mapping[ms][idt]['high'] = hmap
            time_mapping[ms][idt]['time_id'] = np.arange(ti, ti + hmap.size)
            ti += hmap.size

            ms_chunks[ms].append({'row': row_chunks,
                                  'chan': tuple(fbin_counts[ms][idt])})

            if gain_name is not None:
                tmp_dict = {}
                for name, val in zip(gain_axes[ms][idt], gain_spec[ms][idt]):
                    if name == 'gain_t':
                        ntimes = gain_times[ms][idt].size
                        nchunksm1 = ntimes//ipit
                        rem = ntimes - nchunksm1*ipit
                        tmp_dict[name] = (ipit,)*nchunksm1
                        if rem:
                            tmp_dict[name] += (rem,)
                    elif name == 'gain_f':
                        tmp_dict[name] = tuple(fbin_counts[ms][idt])
                    elif name == 'dir':
                        if len(val) > 1:
                            raise ValueError("DD gains not supported yet")
                        if val[0] > 1:
                            raise ValueError("DD gains not supported yet")
                        tmp_dict[name] = tuple(val)
                    else:
                        tmp_dict[name] = tuple(val)

                gain_chunks[ms].append(tmp_dict)

    return freqs, fbin_idx, fbin_counts, band_mapping, freq_out, \
        utimes, tbin_idx, tbin_counts, time_mapping, \
        ms_chunks, gain_chunks, radecs, \
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
                 pixsize: float = 1.0):
    """
    Find the Gaussian that approximates the main lobe of the PSF.
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

        # select psf main lobe
        psfv = psfv[islands == ncenter]
        x = xx[islands == ncenter]
        y = yy[islands == ncenter]
        xy = np.vstack((x, y))
        xdiff = x.max() - x.min()
        ydiff = y.max() - y.min()
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


def init_mask(mask, mds, output_type, log):
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
    elif mask.lower() == 'mds':
        try:
            mask = mds.MASK.values.astype(output_type)
            print('Using mask in mds', file=log)
        except:
            print(f"No mask in mds", file=log)
            raise e
    else:
        raise ValueError(f'Unsupported masking option {mask}')
    return mask


def coerce_literal(func, literals):
    func_locals = inspect.stack()[1].frame.f_locals  # One frame up.
    arg_types = [func_locals[k] for k in inspect.signature(func).parameters]
    SentryLiteralArgs(literals).for_function(func).bind(*arg_types)
    return


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
        mean_beam[b] += ds.BEAM.data * ds.WSUM.data[0]
        wsums[b] += ds.WSUM.data[0]
    wsums = da.stack(wsums).squeeze()
    wsum = wsums.sum()
    dirty = da.stack(dirty)/wsum
    model = da.stack(model)
    if 'RESIDUAL' in ds:
        residual = da.stack(residual)/wsum
    if 'PSF' in ds:
        psf = da.stack(psf)/wsum
        psfhat = da.stack(psfhat)/wsum
    for b in range(nband):
        if wsums[b]:
            mean_beam[b] /= wsums[b]

    dirty, model, residual, psf, psfhat, mean_beam, wsums = dask.compute(
                                                                dirty,
                                                                model,
                                                                residual,
                                                                psf,
                                                                psfhat,
                                                                mean_beam,
                                                                wsums)
    return dirty, model, residual, psf, psfhat, mean_beam, wsums


def interp_gain_grid(gdct, ant_names):
    nant = ant_names.size
    ncorr, ntime, nfreq = gdct[ant_names[0]].shape
    time = gdct['time']
    assert time.size==ntime
    freq = gdct['frequencies']
    assert freq.size==nfreq

    gain = np.zeros((ntime, nfreq, nant, 1, ncorr), dtype=np.complex128)

    # get axes in qcal order
    for p, name in enumerate(ant_names):
        gain[:, :, p, 0, :] = np.moveaxis(gdct[name], 0, -1)

    # fit spline to time and freq axes
    gobj_amp = np.zeros((nant, 1, ncorr), dtype=object)
    gobj_phase = np.zeros((nant, 1, ncorr), dtype=object)
    for p in range(nant):
        for c in range(ncorr):
            gobj_amp[p, 0, c] = rbs(time, freq, np.abs(gain[:, :, p, 0, c]))
            unwrapped_phase = np.unwrap(np.unwrap(np.angle(gain[:, :, p, 0, c]), axis=0), axis=1)
            gobj_phase[p, 0, c] = rbs(time, freq, unwrapped_phase)
    return gobj_amp, gobj_phase


def array2qcal_ds(gobj_amp, gobj_phase, time, freq, ant_names, fid, ddid, sid, fname):
    nant, ndir, ncorr = gobj_amp.shape
    ntime = time.size
    nchan = freq.size
    # gains not chunked on disk
    gain = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.complex128)
    for p in range(nant):
        for c in range(ncorr):
            gain[:, :, p, 0, c] = gobj_amp[p, 0, c](time, freq)*np.exp(1.0j*gobj_phase[p, 0, c](time, freq))
    gain = da.from_array(gain, chunks=(-1, -1, -1, -1, -1))
    gflags = da.zeros((ntime, nchan, nant, ndir), chunks=(-1, -1, -1, -1), dtype=np.int8)
    data_vars = {
        'gains':(('gain_t', 'gain_f', 'ant', 'dir', 'corr'), gain),
        'gain_flags':(('gain_t', 'gain_f', 'ant', 'dir'), gflags)
    }
    gain_spec_tup = namedtuple('gains_spec_tup', 'tchunk fchunk achunk dchunk cchunk')
    attrs = {
        'DATA_DESC_ID': int(ddid),
        'FIELD_ID': int(fid),
        'FIELD_NAME': fname,
        'GAIN_AXES': ('gain_t', 'gain_f', 'ant', 'dir', 'corr'),
        'GAIN_SPEC': gain_spec_tup(tchunk=(int(ntime),),
                                    fchunk=(int(nchan),),
                                    achunk=(int(nant),),
                                    dchunk=(int(1),),
                                    cchunk=(int(ncorr),)),
        'NAME': 'NET',
        'SCAN_NUMBER': int(sid),
        'TYPE': 'complex'
    }
    if ncorr==1:
        corrs = np.array(['XX'], dtype=object)
    elif ncorr==2:
        corrs = np.array(['XX', 'YY'], dtype=object)
    coords = {
        'gain_f': (('gain_f',), freq),
        'gain_t': (('gain_t',), time),
        'ant': (('ant'), ant_names),
        'corr': (('corr'), corrs),
        'dir': (('dir'), np.array([0], dtype=np.int32)),
        'f_chunk': (('f_chunk'), np.array([0], dtype=np.int32)),
        't_chunk': (('t_chunk'), np.array([0], dtype=np.int32))
    }
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


# not currently chunking over time
def accum_vis(data, flag, ant1, ant2, nant, ref_ant=-1):
    return da.blockwise(_accum_vis, 'afc',
                        data, 'rfc',
                        flag, 'rfc',
                        ant1, 'r',
                        ant2, 'r',
                        ref_ant, None,
                        new_axes={'a':nant},
                        dtype=np.complex128)


def _accum_vis(data, flag, ant1, ant2, ref_ant):
    return _accum_vis_impl(data[0], flag[0], ant1[0], ant2[0], ref_ant)

# @jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def _accum_vis_impl(data, flag, ant1, ant2, ref_ant):
    # select out reference antenna
    I = np.where((ant1==ref_ant) | (ant2==ref_ant))[0]
    data = data[I]
    flag = flag[I]
    ant1 = ant1[I]
    ant2 = ant2[I]

    # we can zero flagged data because they won't contribute to the FT
    data = np.where(flag, 0j, data)
    nrow, nchan, ncorr = data.shape
    nant = np.maximum(ant1.max(), ant2.max()) + 1
    if ref_ant == -1:
        ref_ant = nant-1
    ncorr = data.shape[-1]
    vis = np.zeros((nant, nchan, ncorr), dtype=np.complex128)
    counts = np.zeros((nant, nchan, ncorr), dtype=np.float64)
    for row in range(nrow):
        p = int(ant1[row])
        q = int(ant2[row])
        if p == ref_ant:
            vis[q] += data[row].astype(np.complex128).conj()
            counts[q] += flag[row].astype(np.float64)
        elif q == ref_ant:
            vis[p] += data[row].astype(np.complex128)
            counts[p] += flag[row].astype(np.float64)
    valid = counts > 0
    vis[valid] = vis[valid]/counts[valid]
    return vis

def estimate_delay(vis_ant, freq, min_delay):
    return da.blockwise(_estimate_delay, 'ac',
                        vis_ant, 'afc',
                        freq, 'f',
                        min_delay, None,
                        dtype=np.float64)


def _estimate_delay(vis_ant, freq, min_delay):
    return _estimate_delay_impl(vis_ant[0], freq[0], min_delay)

def _estimate_delay_impl(vis_ant, freq, min_delay):
    delta_freq = 1.0/min_delay
    nchan = freq.size
    fmax = freq.min() + delta_freq
    fexcess = fmax - freq.max()
    freq_cell = freq[1]-freq[0]
    if fexcess > 0:
        npad = np.int(np.ceil(fexcess/freq_cell))
        npix = (nchan + npad)
    else:
        npix = nchan
    while npix%2:
        npix = good_size(npix+1)
    npad = npix - nchan
    lag = np.fft.fftfreq(npix, freq_cell)
    lag = Fs(lag)
    dlag = lag[1] - lag[0]
    nant, _, ncorr = vis_ant.shape
    delays = np.zeros((nant, ncorr), dtype=np.float64)
    for p in range(nant):
        for c in range(ncorr):
            vis_fft = np.fft.fft(vis_ant[p, :, c], npix)
            pspec = np.abs(Fs(vis_fft))
            if not pspec.any():
                continue
            delay_idx = np.argmax(pspec)
            # fm1 = lag[delay_idx-1]
            # f0 = lag[delay_idx]
            # fp1 = lag[delay_idx+1]
            # delays[p,c] = 0.5*dlag*(fp1 - fm1)/(fm1 - 2*f0 + fp1)
            # print(p, c, lag[delay_idx])
            delays[p,c] = lag[delay_idx]
    return delays


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


def add_column(ms, col_name, like_col="DATA", like_type=None):
    if col_name not in ms.colnames():
        desc = ms.getcoldesc(like_col)
        desc['name'] = col_name
        desc['comment'] = desc['comment'].replace(" ", "_")  # got this from Cyril, not sure why
        dminfo = ms.getdminfo(like_col)
        dminfo["NAME"] =  "{}-{}".format(dminfo["NAME"], col_name)
        # if a different type is specified, insert that
        if like_type:
            desc['valueType'] = like_type
        ms.addcols(desc, dminfo)
    return ms


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


def lthreshold(x, sigma, kind='l1'):
    if kind=='l0':
        return np.where(np.abs(x) > sigma, x, 0) * np.sign(x)
    elif kind=='l1':
        absx = np.abs(x)
        return np.where(absx > sigma, absx - sigma, 0) * np.sign(x)


@njit(nogil=True, cache=True, inline='always')
def pad_and_shift(x, out):
    '''
    Pad x with zeros so as to have the same shape as out and perform
    ifftshift in place
    '''
    nxi, nyi = x.shape
    nxo, nyo = out.shape
    if nxi >= nxo or nyi >= nyo:
        raise ValueError('Output must be larger than input')
    padx = nxo-nxi
    pady = nyo-nyi
    out[...] = 0.0
    # first and last quadrants
    for i in range(nxi//2):
        for j in range(nyi//2):
            # first to last quadrant
            out[padx + nxi//2 + i, pady + nyi//2 + j] = x[i, j]
            # last to first quadrant
            out[i, j] = x[nxi//2+i, nyi//2+j]
            # third to second quadrant
            out[i, pady + nyi//2 + j] = x[nxi//2 + i, j]
            # second to third quadrant
            out[padx + nxi//2 + i, j] = x[i, nyi//2 + j]
    return out


@njit(nogil=True, cache=True, inline='always')
def unpad_and_unshift(x, out):
    '''
    fftshift x and unpad it into out
    '''
    nxi, nyi = x.shape
    nxo, nyo = out.shape
    if nxi < nxo or nyi < nyo:
        raise ValueError('Output must be smaller than input')
    out[...] = 0.0
    padx = nxo-nxi
    pady = nyo-nyi
    for i in range(nxo//2):
        for j in range(nyo//2):
            # first to last quadrant
            out[nxo//2+i, nyo//2+j] = x[i, j]
            # last to first quadrant
            out[i, j] = x[padx + nxo//2 + i, pady + nyo//2 + j]
            # third to second quadrant
            out[nxo//2 + i, j] = x[i, pady + nyo//2 + j]
            # second to third quadrant
            out[i, nyo//2 + j] = x[padx + nxo//2 + i, j]
    return out


@njit(nogil=True, cache=True, inline='always', parallel=True)
def pad_and_shift_cube(x, out):
    '''
    Pad x with zeros so as to have the same shape as out and perform
    ifftshift in place
    '''
    nband, nxi, nyi = x.shape
    nband, nxo, nyo = out.shape
    if nxi >= nxo or nyi >= nyo:
        raise ValueError('Output must be larger than input')
    padx = nxo-nxi
    pady = nyo-nyi
    out[...] = 0.0
    for b in prange(nband):
        for i in range(nxi//2):
            for j in range(nyi//2):
                # first to last quadrant
                out[b, padx + nxi//2 + i, pady + nyi//2 + j] = x[b, i, j]
                # last to first quadrant
                out[b, i, j] = x[b, nxi//2+i, nyi//2+j]
                # third to second quadrant
                out[b, i, pady + nyi//2 + j] = x[b, nxi//2 + i, j]
                # second to third quadrant
                out[b, padx + nxi//2 + i, j] = x[b, i, nyi//2 + j]
    return out


@njit(nogil=True, cache=True, inline='always', parallel=True)
def unpad_and_unshift_cube(x, out):
    '''
    fftshift x and unpad it into out
    '''
    nband, nxi, nyi = x.shape
    nband, nxo, nyo = out.shape
    if nxi < nxo or nyi < nyo:
        raise ValueError('Output must be smaller than input')
    out[...] = 0.0
    padx = nxo-nxi
    pady = nyo-nyi
    for b in prange(nband):
        for i in range(nxo//2):
            for j in range(nyo//2):
                # first to last quadrant
                out[b, nxo//2+i, nyo//2+j] = x[b, i, j]
                # last to first quadrant
                out[b, i, j] = x[b, padx + nxo//2 + i, pady + nyo//2 + j]
                # third to second quadrant
                out[b, nxo//2 + i, j] = x[b, i, pady + nyo//2 + j]
                # second to third quadrant
                out[b, i, nyo//2 + j] = x[b, padx + nxo//2 + i, j]
    return out
