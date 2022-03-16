import sys
import numpy as np
import numexpr as ne
from numba import jit, njit
import dask
import dask.array as da
from dask.distributed import performance_report
from dask.diagnostics import ProgressBar
from ducc0.fft import r2c, c2r
iFs = np.fft.ifftshift
Fs = np.fft.fftshift
from numba.core.extending import SentryLiteralArgs
import inspect


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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    from ducc0.fft import good_size
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
    from daskms import xds_from_ms
    from daskms import xds_from_table
    import dask
    import dask.array as da

    from omegaconf import ListConfig
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
    from skimage.morphology import label
    from scipy.optimize import curve_fit

    nband, nx, ny = psf.shape

    # # find extent required to capture main lobe
    # # saves on time to label islands
    # psf0 = psf[0]/psf[0].max()  # largest PSF at lowest freq
    # num_islands = 0
    # npix = np.minimum(nx, ny)
    # nbox = np.minimum(12, npix)  # 12 pixel minimum
    # if npix <= nbox:
    #     nbox = npix
    # else:
    #     while num_islands < 2:
    #         Ix = slice(nx//2 - nbox//2, nx//2 + nbox//2)
    #         Iy = slice(ny//2 - nbox//2, ny//2 + nbox//2)
    #         mask = np.where(psf0[Ix, Iy] > level, 1.0, 0)
    #         islands, num_islands = label(mask, return_num=True)
    #         if num_islands < 2:
    #             nbox *= 2  # double size and try again

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
        # find regions where psf is non-zero
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


def setup_image_data(dds, opts, rname, apparent=False, log=None):
    nband = opts.nband
    real_type = dds[0].DIRTY.dtype
    complex_type = np.result_type(real_type, np.complex64)
    nx, ny = dds[0].DIRTY.shape
    residual = [da.zeros((nx, ny), chunks=(-1, -1),
                         dtype=real_type) for _ in range(nband)]
    wsums = [da.zeros(1, dtype=real_type) for _ in range(nband)]
    if opts.mean_ds and opts.use_psf:
        nx_psf, ny_psf = dds[0].PSF.shape
        nx_psf, nyo2_psf = dds[0].PSFHAT.shape
        psf = [da.zeros((nx_psf, ny_psf), chunks=(-1, -1),
                        dtype=complex_type) for _ in range(nband)]
        psfhat = [da.zeros((nx_psf, nyo2_psf), chunks=(-1, -1),
                           dtype=complex_type) for _ in range(nband)]
        mean_beam = [da.zeros((nx, ny), chunks=(-1, -1),
                              dtype=real_type) for _ in range(nband)]
        for ds in dds:
            b = ds.bandid
            if apparent:
                # in case rname does not exist
                try:
                    residual[b] += ds.get(rname).data
                except:
                    print(f"Could not find {rname} in dds", file=log)
            else:
                try:
                    residual[b] += ds.get(rname).data * ds.BEAM.data
                except:
                    print(f"Cold not find {rname} in dds", file=log)
            psf[b] += ds.PSF.data
            psfhat[b] += ds.PSFHAT.data
            mean_beam[b] += ds.BEAM.data * ds.WSUM.data[0]
            wsums[b] += ds.WSUM.data[0]
        wsums = da.stack(wsums).squeeze()
        wsum = wsums.sum()
        residual = da.stack(residual)/wsum
        psf = da.stack(psf)/wsum
        psfhat = da.stack(psfhat)/wsum
        mean_beam = da.stack(mean_beam)/wsums[:, None, None]
        residual, psf, psfhat, mean_beam, wsum = dask.compute(residual, psf, psfhat,
                                                         mean_beam, wsum)
    else:
        for ds in dds:
            b = ds.bandid
            if apparent:
                try:
                    residual[b] += ds.get(rname).data
                except:
                    print(f"Cold not find {rname} in dds", file=log)
            else:
                try:
                    residual[b] += ds.get(rname).data * ds.BEAM.data
                except:
                    print(f"Cold not find {rname} in dds", file=log)
            wsums[b] += ds.WSUM.data[0]
        wsum = da.stack(wsums).sum()
        residual = da.stack(residual)/wsum
        residual, wsum = dask.compute(residual, wsum)
        psf = None
        psfhat = None
        mean_beam = None
    return residual, wsum, psf, psfhat, mean_beam
