import numpy as np
import dask.array as da
import sys


def fit_spi(image: np.ndarray,
            beam: np.ndarray,
            freqs: np.ndarray,
            weights: np.ndarray,
            threshold: float,
            nthreads: int,
            pb_min: float,
            padding_frac: float,
            ref_freq: float,
            dest=sys.stdout):

    try:
        assert model.ndim == 3
        assert beam.ndim == 3
        assert model.shape == beam.shape
        assert model.shape[0] > 1
        assert freqs.size == model.shape[0]
        assert weights.size == model.shape[0]
    except Exception as e:
        raise e

    # beam cut off
    image = np.where(beam_image > args.pb_min, image, 0)

    # get pixels above threshold
    minimage = np.amin(image, axis=0)
    maskindices = np.argwhere(minimage > threshold)
    nanindices = np.argwhere(minimage <= threshold)
    if not maskindices.size:
        raise ValueError("No components found above threshold. "
                         "Try lowering your threshold."
                         "Max of convolved model is %3.2e" % model.max())
    fitcube = image[:, maskindices[:, 0], maskindices[:, 1]].T
    beam_comps = beam[:, maskindices[:, 0], maskindices[:, 1]].T

    ncomps, _ = fitcube.shape
    fitcube = da.from_array(fitcube.astype(np.float64),
                            chunks=(ncomps//args.ncpu, nband))
    beam_comps = da.from_array(beam_comps.astype(np.float64),
                               chunks=(ncomps//args.ncpu, nband))
    weights = da.from_array(weights.astype(np.float64), chunks=(nband))
    freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

    print("Fitting %i components" % ncomps, file=dest)
    alpha, alpha_err, Iref, i0_err = fit_spi_components(fitcube, weights, freqsdask,
                                                        np.float64(ref_freq), beam=beam_comps).compute()
    print("Done. Writing output. \n", file=dest)

    alphamap = np.zeros(model[0].shape, dtype=model.dtype)
    alphamap[...] = np.nan
    alpha_err_map = np.zeros(model[0].shape, dtype=model.dtype)
    alpha_err_map[...] = np.nan
    i0map = np.zeros(model[0].shape, dtype=model.dtype)
    i0map[...] = np.nan
    i0_err_map = np.zeros(model[0].shape, dtype=model.dtype)
    i0_err_map[...] = np.nan
    alphamap[maskindices[:, 0], maskindices[:, 1]] = alpha
    alpha_err_map[maskindices[:, 0], maskindices[:, 1]] = alpha_err
    i0map[maskindices[:, 0], maskindices[:, 1]] = Iref
    i0_err_map[maskindices[:, 0], maskindices[:, 1]] = i0_err

    return alphamap, alpha_err_map, i0map, i0_err_map
