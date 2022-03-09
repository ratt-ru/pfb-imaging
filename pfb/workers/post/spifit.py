# flake8: noqa
from contextlib import ExitStack
import click
from omegaconf import OmegaConf
from pfb.workers.main import cli
import pyscilog
pyscilog.init('pfb')
log = pyscilog.get_logger('SPIFIT')

@cli.command()
@click.option('-image', '--image', required=True,
              help="Path to model or restored image cube.")
@click.option('-resid', "--residual", required=False,
              help="Path to residual image cube.")
@click.option('-o', '--output-filename', required=True,
              help="Path to output directory + prefix.")
@click.option('-pp', '--psf-pars', nargs=3, type=float,
              help="Beam parameters matching FWHM of restoring beam "
                   "specified as emaj emin pa."
                   "By default these are taken from the fits header "
                   "of the residual image.")
@click.option('--circ-psf/--no-circ-psf', default=False)
@click.option('-th', '--threshold', default=10, type=float, show_default=True,
              help="Multiple of the rms in the residual to threshold on."
                   "Only components above threshold*rms will be fit.")
@click.option('-maxdr', '--maxdr', default=100, type=float, show_default=True,
              help="Maximum dynamic range used to determine the "
                   "threshold above which components need to be fit. "
                   "Only used if residual is not passed in.")
@click.option('-bw', '--band-weights', type=float,
              help="Per bands weights to use during the fit")
@click.option('-pb-min', '--pb-min', type=float, default=0.15,
              help="Set image to zero where pb falls below this value")
@click.option('-products', '--products', default='aeikIcmrbd', type=str,
              help="Outputs to write. Letter correspond to: \n"
              "a - alpha map \n"
              "e - alpha error map \n"
              "i - I0 map \n"
              "k - I0 error map \n"
              "I - reconstructed cube form alpha and I0 \n"
              "c - restoring beam used for convolution \n"
              "m - convolved model \n"
              "r - convolved residual \n"
              "b - average primary beam image \n"
              "d - difference image \n"
              "Default is to write all of them")
@click.option('-pf', "--padding-frac", default=0.5, type=float,
              show_default=True, help="Padding factor for FFT's.")
@click.option('-dc', "--dont-convolve", is_flag=True,
              help="Do not convolve by the clean beam before fitting")
@click.option('-rf', '--ref-freq', type=float,
              help='Reference frequency where the I0 map is sought. '
              "Will overwrite in fits headers of output.")
@click.option('-otype', '--out-dtype', default='f4', type=str,
              help="Data type of output. Default is single precision")
@click.option('-acr', '--add-convolved-residuals', is_flag=True,
              help='Flag to add in the convolved residuals before '
              'fitting components')
@click.option('-bm', '--beam-model', default=None,
              help="Fits power beam model. It is assumed that the beam "
              "match the fits headers of --image. You can use the binterp "
              "worker to create compatible beam models")
@click.option('-ha', '--host-address',
              help='Address where the distributed client lives. '
              'Will use a local cluster if no address is provided')
@click.option('-nw', '--nworkers', type=int, default=1,
              help='Number of workers for the client.')
@click.option('-ntpw', '--nthreads-per-worker', type=int,
              help='Number of dask threads per worker.')
@click.option('-nvt', '--nvthreads', type=int,
              help="Total number of threads to use for vertical scaling (eg. gridder, fft's etc.)")
@click.option('-mem', '--mem-limit', type=int,
              help="Memory limit in GB. Default uses all available memory")
@click.option('-nthreads', '--nthreads', type=int,
              help="Total available threads. Default uses all available threads")
def spifit(**kw):
    """
    Spectral index fitter

    """
    opts = OmegaConf.create(kw)
    pyscilog.log_to_file(opts.output_filename + '.log')
    from glob import glob
    from omegaconf import ListConfig
    # image is either a string or a list of strings that we want to glob on
    if isinstance(opts.image, str):
        image = sorted(glob(opts.image))
    elif isinstance(opts.image, list) or isinstance(opts.image, ListConfig):
        image = []
        for i in len(opts.image):
            image.append(sorted(glob(opts.image[i])))

    # make sure it's not empty
    try:
        assert len(image) > 0
        opts.image = image
    except:
        raise ValueError(f"No image at {opts.image}")

    # same goes for the residual except that it may also be None
    if isinstance(opts.residual, str):
        residual = sorted(glob(opts.residual))
    elif isinstance(opts.residual, list) or isinstance(opts.residual, ListConfig):
        residual = []
        for i in len(opts.residual):
            residual.append(sorted(glob(opts.residual[i])))

    if opts.residual is not None:
        try:
            assert len(residual) > 0
            opts.residual = residual
        except:
            raise ValueError(f"No residual at {opts.residual}")
        # we also need the same number of residuals as images
        try:
            assert len(opts.image) == len(opts.residual)
        except:
            raise ValueError(f"Number of images and residuals need to "
                                "match")
    else:
        print("No residual passed in!", file=log)

    # and finally the beam model
    if isinstance(opts.beam_model, str):
        beam_model  = sorted(glob(opts.beam_model))
    elif isinstance(opts.beam_model, list) or isinstance(opts.beam_model, ListConfig):
        beam_model = []
        for i in len(opts.beam_model):
            beam_model.append(sorted(glob(opts.beam_model[i])))

    if opts.beam_model is not None:
        try:
            assert len(beam_model) > 0
            opts.beam_model = beam_model
        except:
            raise ValueError(f"No beam model at {opts.beam_model}")

        try:
            assert len(opts.image) == len(opts.beam_model)
        except:
            raise ValueError(f"Number of images and beam models need to "
                                "match")
    else:
        print("Not doing any form of primary beam correction", file=log)

    # LB - TODO: can we sort them along freq at this point already?

    OmegaConf.set_struct(opts, True)

    with ExitStack() as stack:
        from pfb import set_client
        opts = set_client(opts, stack, log)

        # TODO - prettier config printing
        print('Input Options:', file=log)
        for key in opts.keys():
            print('     %25s = %s' % (key, opts[key]), file=log)

        return _spifit(**opts)


def _spifit(**kw):
    opts = OmegaConf.create(kw)
    OmegaConf.set_struct(opts, True)

    import dask.array as da
    import numpy as np
    from astropy.io import fits
    from africanus.model.spi.dask import fit_spi_components
    from pfb.utils.fits import load_fits, save_fits, data_from_header, set_wcs
    from pfb.utils.misc import convolve2gaussres

    # get max gausspars
    gaussparf = None
    if opts.psf_pars is None:
        if opts.residual is None:
            ppsource = opts.image
        else:
            ppsource = opts.residual

        for image in ppsource:
            try:
                pphdr = fits.getheader(image)
            except Exception as e:
                raise e

            if 'BMAJ0' in pphdr.keys():
                emaj = pphdr['BMAJ0']
                emin = pphdr['BMIN0']
                pa = pphdr['BPA0']
                gausspars = [emaj, emin, pa]
                freq_idx0 = 0
            elif 'BMAJ1' in pphdr.keys():
                emaj = pphdr['BMAJ1']
                emin = pphdr['BMIN1']
                pa = pphdr['BPA1']
                gausspars = [emaj, emin, pa]
                freq_idx0 = 1
            elif 'BMAJ' in pphdr.keys():
                emaj = pphdr['BMAJ']
                emin = pphdr['BMIN']
                pa = pphdr['BPA']
                gausspars = [emaj, emin, pa]
                freq_idx0 = 0
            else:
                raise ValueError("No beam parameters found in residual."
                                "You will have to provide them manually.")

            if gaussparf is None:
                gaussparf = gausspars
            else:
                # we need to take the max in both directions
                gaussparf[0] = np.maximum(gaussparf[0], gausspars[0])
                gaussparf[1] = np.maximum(gaussparf[1], gausspars[1])
    else:
        freq_idx0 = 0  # assumption
        gaussparf = list(opts.psf_pars)

    if opts.circ_psf:
        e = np.maximum(gaussparf[0], gaussparf[1])
        gaussparf[0] = e
        gaussparf[1] = e
        gaussparf[2] = 0.0

    gaussparf = tuple(gaussparf)
    print("Using emaj = %3.2e, emin = %3.2e, PA = %3.2e \n" % gaussparf, file=log)

    # get required data products
    image_dict = {}
    for i in range(len(opts.image)):
        image_dict[i] = {}

        # load model image
        model = load_fits(opts.image[i], dtype=opts.out_dtype).squeeze()
        mhdr = fits.getheader(opts.image[i])

        if model.ndim < 3:
            model = model[None, :, :]

        l_coord, ref_l = data_from_header(mhdr, axis=1)
        l_coord -= ref_l
        m_coord, ref_m = data_from_header(mhdr, axis=2)
        m_coord -= ref_m
        if mhdr["CTYPE4"].lower() == 'freq':
            freq_axis = 4
            stokes_axis = 3
        elif mhdr["CTYPE3"].lower() == 'freq':
            freq_axis = 3
            stokes_axis = 4
        else:
            raise ValueError("Freq axis must be 3rd or 4th")

        freqs, ref_freq = data_from_header(mhdr, axis=freq_axis)

        image_dict[i]['freqs'] = freqs

        nband = freqs.size
        npix_l = l_coord.size
        npix_m = m_coord.size

        xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')

        # load beam
        if opts.beam_model is not None:
            bhdr = fits.getheader(opts.beam_model[i])
            l_coord_beam, ref_lb = data_from_header(bhdr, axis=1)
            l_coord_beam -= ref_lb
            if not np.array_equal(l_coord_beam, l_coord):
                raise ValueError("l coordinates of beam model do not match "
                                 "those of image. Use binterp to make "
                                 "compatible beam images")

            m_coord_beam, ref_mb = data_from_header(bhdr, axis=2)
            m_coord_beam -= ref_mb
            if not np.array_equal(m_coord_beam, m_coord):
                raise ValueError("m coordinates of beam model do not match "
                                 "those of image. Use binterp to make "
                                 "compatible beam images")
            freqs_beam, _ = data_from_header(bhdr, axis=freq_axis)
            if not np.array_equal(freqs, freqs_beam):
                raise ValueError("Freq coordinates of beam model do not match "
                                 "those of image. Use binterp to make "
                                 "compatible beam images")
            beam_image = load_fits(opts.beam_model[i],
                                   dtype=opts.out_dtype).squeeze()

            if beam_image.ndim < 3:
                beam_image = beam_image[None, :, :]

        else:
            beam_image = np.ones(model.shape, dtype=opts.out_dtype)

        image_dict[i]['beam'] = beam_image

        if not opts.dont_convolve:
            print("Convolving model %i"%i, file=log)
            # convolve model to desired resolution
            # get initial resolution of model
            gausspari = ()
            for b in range(nband):
                key = 'BMAJ' + str(b + freq_idx0)
                if key in mhdr.keys():
                    emaj = mhdr[key]
                    emin = mhdr[key]
                    pa = mhdr[key]
                    gausspari += ((emaj, emin, pa),)
                elif 'BMAJ' in mhdr.keys():
                    emaj = mhdr['BMAJ']
                    emin = mhdr['BMIN']
                    pa = mhdr['BPA']
                    gausspari += ((emaj, emin, pa),)
                else:
                    gausspari = None
                    break

            model, gausskern = convolve2gaussres(model, xx, yy, gaussparf,
                                                 opts.nthreads, gausspari,
                                                 opts.padding_frac,
                                                 norm_kernel=False if gausspari is None else True)

        # add in residuals and set threshold
        if opts.residual is not None:
            msg = "of residual do not match those of model"
            rhdr = fits.getheader(opts.residual[i])
            l_res, ref_lb = data_from_header(rhdr, axis=1)
            l_res -= ref_lb
            if not np.array_equal(l_res, l_coord):
                raise ValueError("l coordinates " + msg)

            m_res, ref_mb = data_from_header(rhdr, axis=2)
            m_res -= ref_mb
            if not np.array_equal(m_res, m_coord):
                raise ValueError("m coordinates " + msg)

            freqs_res, _ = data_from_header(rhdr, axis=freq_axis)
            if not np.array_equal(freqs, freqs_res):
                raise ValueError("Freqs " + msg)

            resid = load_fits(opts.residual[i],
                              dtype=opts.out_dtype).squeeze()
            if resid.ndim < 3:
                resid = resid[None, :, :]

            # convolve residual to same resolution as model
            gausspari = ()
            for b in range(nband):
                key = 'BMAJ' + str(b + freq_idx0)
                if key in rhdr.keys():
                    emaj = rhdr[key]
                    emin = rhdr[key]
                    pa = rhdr[key]
                    gausspari += ((emaj, emin, pa),)
                elif 'BMAJ' in rhdr.keys():
                    emaj = rhdr['BMAJ']
                    emin = rhdr['BMIN']
                    pa = rhdr['BPA']
                    gausspari += ((emaj, emin, pa),)
                else:
                    print("Can't find Gausspars in residual header, "
                          "unable to add residuals back in", file=log)
                    gausspari = None
                    break

            if gausspari is not None and opts.add_convolved_residuals:
                print("Convolving residuals %i"%i, file=log)
                resid, _ = convolve2gaussres(resid, xx, yy, gaussparf,
                                             opts.nthreads, gausspari,
                                             opts.padding_frac,
                                             norm_kernel=False)
                model += resid
                print("Convolved residuals added to convolved model %i"%i,
                      file=log)

            image_dict[i]['resid'] = resid

        else:
            image_dict[i]['resid'] = None

        image_dict[i]['model'] = model

    # concatenate images along frequency here
    freqs = []
    model = []
    beam_image = []
    resid = []
    for i in image_dict.keys():
        freqs.append(image_dict[i]['freqs'])
        model.append(image_dict[i]['model'])
        beam_image.append(image_dict[i]['beam'])
        resid.append(image_dict[i]['resid'])
    freqs = np.concatenate(freqs, axis=0)
    Isort = np.optsort(freqs)
    freqs = freqs[Isort]

    model = np.concatenate(model, axis=0)
    model = model[Isort]

    # create header
    cell_deg = mhdr['CDELT1']
    ra = np.deg2rad(mhdr['CRVAL1'])
    dec = np.deg2rad(mhdr['CRVAL2'])
    radec = [ra, dec]
    nband, nx, ny = model.shape
    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freqs)
    for i in range(1, nband+1):
        hdr['BMAJ' + str(i)] = gaussparf[0]
        hdr['BMIN' + str(i)] = gaussparf[1]
        hdr['BPA' + str(i)] = gaussparf[2]
    if opts.ref_freq is None:
        ref_freq = np.mean(freqs)
    else:
        ref_freq = opts.ref_freq
    hdr_mfs = set_wcs(cell_deg, cell_deg, nx, ny, radec, ref_freq)
    hdr_mfs['BMAJ'] = gaussparf[0]
    hdr_mfs['BMIN'] = gaussparf[1]
    hdr_mfs['BPA'] = gaussparf[2]

    # save convolved model
    if 'm' in opts.products:
        name = opts.output_filename + '.convolved_model.fits'
        save_fits(name, model, hdr, dtype=opts.out_dtype)
        print("Wrote convolved model to %s" % name, file=log)

    beam_image = np.concatenate(beam_image, axis=0)
    beam_image = beam_image[Isort]

    if 'b' in opts.products:
        name = opts.output_filename + '.power_beam.fits'
        save_fits(name, beam_image, hdr, dtype=opts.out_dtype)
        print("Wrote average power beam to %s" % name, file=log)

    if resid[0] is not None:
        resid = np.concatenate(resid, axis=0)
        resid = resid[Isort]

        if 'r' in opts.products:
            name = opts.output_filename + '.convolved_residual.fits'
            save_fits(name, resid, hdr, dtype=opts.out_dtype)
            print("Wrote convolved residuals to %s" % name, file=log)

        # get threshold
        counts = np.sum(resid != 0)
        rms = np.sqrt(np.sum(resid**2)/counts)
        rms_cube = np.std(resid.reshape(nband, npix_l*npix_m), axis=1).ravel()
        threshold = opts.threshold * rms
    else:
        print("No residual provided. Setting  threshold i.t.o dynamic range. "
              "Max dynamic range is %i " % opts.maxdr, file=log)
        threshold = model.max()/opts.maxdr
        rms_cube = None

    print("Threshold set to %f Jy. \n" % threshold, file=log)

    # beam cut off
    beam_min = np.amin(beam_image, axis=0)
    model = np.where(beam_min[None] > opts.pb_min, model, 0.0)

    # get pixels above threshold
    minimage = np.amin(model, axis=0)
    maskindices = np.argwhere(minimage > threshold)
    nanindices = np.argwhere(minimage <= threshold)
    if not maskindices.size:
        raise ValueError("No components found above threshold. "
                        "Try lowering your threshold."
                        "Max of convolved model is %3.2e" % model.max())
    fitcube = model[:, maskindices[:, 0], maskindices[:, 1]].T
    beam_comps = beam_image[:, maskindices[:, 0], maskindices[:, 1]].T

    # set weights for fit
    if rms_cube is not None:
        print("Using RMS in each imaging band to determine weights.", file=log)
        weights = np.where(rms_cube > 0, 1.0/rms_cube**2, 0.0)
        # normalise
        weights /= weights.max()
    else:
        if opts.band_weights is not None:
            weights = np.array(opts.band_weights)
            try:
                assert weights.size == nband
            except Exception as e:
                raise ValueError("Inconsistent weighst provided.")
            print("Using provided channel weights.", file=log)
        else:
            print("No residual or channel weights provided. Using equal weights.", file=log)
            weights = np.ones(nband, dtype=np.float64)

    ncomps, _ = fitcube.shape
    fitcube = da.from_array(fitcube.astype(np.float64),
                            chunks=(ncomps//opts.nthreads, nband))
    beam_comps = da.from_array(beam_comps.astype(np.float64),
                               chunks=(ncomps//opts.nthreads, nband))
    weights = da.from_array(weights.astype(np.float64), chunks=(nband))
    freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

    print("Fitting %i components" % ncomps, file=log)
    alpha, alpha_err, Iref, i0_err = fit_spi_components(fitcube, weights, freqsdask,
                                        np.float64(ref_freq), beam=beam_comps).compute()
    print("Done. Writing output.", file=log)

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

    if 'I' in opts.products:
        # get the reconstructed cube
        Irec_cube = i0map[None, :, :] * \
            (freqs[:, None, None]/ref_freq)**alphamap[None, :, :]
        name = opts.output_filename + '.Irec_cube.fits'
        save_fits(name, Irec_cube, hdr, dtype=opts.out_dtype)
        print("Wrote reconstructed cube to %s" % name, file=log)

    # save alpha map
    if 'a' in opts.products:
        name = opts.output_filename + '.alpha.fits'
        save_fits(name, alphamap, hdr_mfs, dtype=opts.out_dtype)
        print("Wrote alpha map to %s" % name, file=log)

    # save alpha error map
    if 'e' in opts.products:
        name = opts.output_filename + '.alpha_err.fits'
        save_fits(name, alpha_err_map, mhdr, dtype=opts.out_dtype)
        print("Wrote alpha error map to %s" % name, file=log)

    # save I0 map
    if 'i' in opts.products:
        name = opts.output_filename + '.I0.fits'
        save_fits(name, i0map, mhdr, dtype=opts.out_dtype)
        print("Wrote I0 map to %s" % name, file=log)

    # save I0 error map
    if 'k' in opts.products:
        name = opts.output_filename + '.I0_err.fits'
        save_fits(name, i0_err_map, mhdr, dtype=opts.out_dtype)
        print("Wrote I0 error map to %s" % name, file=log)

    if 'd' in opts.products:
        Irec_cube = i0map[None, :, :] * \
            (freqs[:, None, None]/ref_freq)**alphamap[None, :, :]
        diff = model - beam_image * Irec_cube
        diff_image = np.zeros(model.shape, dtype=model.dtype)
        diff_image[...] = np.nan
        diff_image[:, maskindices[:, 0], maskindices[:, 1]] = diff[:, maskindices[:, 0], maskindices[:, 1]]
        name = opts.output_filename + '.diff.fits'
        save_fits(name, diff_image, hdr, dtype=opts.out_dtype)
        print("Wrote diff map to %s" % name, file=log)


    print("All done here", file=log)
