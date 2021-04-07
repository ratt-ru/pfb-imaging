import click

import argparse
import dask
import dask.array as da
import numpy as np
from astropy.io import fits
from africanus.model.spi.dask import fit_spi_components
from pfb.utils import load_fits, save_fits, convolve2gaussres, data_from_header, set_header_info, str2bool
from scripts.power_beam_maker import interpolate_beam

def create_parser():
    p = argparse.ArgumentParser(description='Spectral index fitting tool.',
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-model', "--model", type=str)
    p.add_argument('-residual', "--residual", type=str)
    p.add_argument('-o', '--output-filename', type=str,
                   help="Path to output directory + prefix. \n"
                        "Placed next to input model if outfile not provided.")
    p.add_argument('-pp', '--psf-pars', default=None, nargs='+', type=float,
                   help="Beam parameters matching FWHM of restoring beam "
                        "specified as emaj emin pa. \n"
                        "By default these are taken from the fits header "
                        "of the residual image.")
    p.add_argument('-cp', "--circ-psf", type=str2bool, nargs='?', const=True, default=True,
                   help="Passing this flag will convolve with a circularised "
                   "beam instead of an elliptical one")
    p.add_argument('-th', '--threshold', default=10, type=float,
                   help="Multiple of the rms in the residual to threshold "
                        "on. \n"
                        "Only components above threshold*rms will be fit.")
    p.add_argument('-maxDR', '--maxDR', default=100, type=float,
                   help="Maximum dynamic range used to determine the "
                        "threshold above which components need to be fit. \n"
                        "Only used if residual is not passed in.")
    p.add_argument('-ncpu', '--ncpu', default=0, type=int,
                   help="Number of threads to use. \n"
                        "Default of zero means use all threads")
    p.add_argument('-pb-min', '--pb-min', type=float, default=0.15,
                   help="Set image to zero where pb falls below this value")
    p.add_argument('-products', '--products', default='aeikIcmrb', type=str,
                   help="Outputs to write. Letter correspond to: \n"
                   "a - alpha map \n"
                   "e - alpha error map \n"
                   "i - I0 map \n"
                   "k - I0 error map \n"
                   "I - reconstructed cube form alpha and I0 \n"
                   "c - restoring beam used for convolution \n"
                   "m - convolved model \n"
                   "r - convolved residual \n"
                   "b - average power beam \n"
                   "Default is to write all of them")
    p.add_argument('-pf', "--padding-frac", default=0.5, type=float,
                   help="Padding factor for FFT's.")
    p.add_argument('-dc', "--dont-convolve", type=str2bool, nargs='?', const=True, default=False,
                   help="Passing this flag bypasses the convolution "
                   "by the clean beam")
    p.add_argument('-cw', "--channel_weights", default=None, nargs='+', type=float,
                   help="Per-channel weights to use during fit to frequency axis. \n "
                   "Only has an effect if no residual is passed in (for now).")
    p.add_argument('-rf', '--ref-freq', default=None, type=np.float64,
                   help='Reference frequency where the I0 map is sought. \n'
                   "Will overwrite in fits headers of output.")
    p.add_argument('-otype', '--out_dtype', default='f4', type=str,
                   help="Data type of output. Default is single precision")
    p.add_argument('-acr', '--add-convolved-residuals', type=str2bool, nargs='?', const=True, default=True,
                   help='Flag to add in the convolved residuals before fitting components')
    p.add_argument('-ms', "--ms", nargs="+", type=str,
                   help="Mesurement sets used to make the image. \n"
                   "Used to get paralactic angles if doing primary beam correction")
    p.add_argument('-f', "--field", type=int, default=0,
                   help="Field ID")
    p.add_argument('-bm', '--beam-model', default=None, type=str,
                   help="Fits beam model to use. \n"
                        "It is assumed that the pattern is path_to_beam/"
                        "name_corr_re/im.fits. \n"
                        "Provide only the path up to name "
                        "e.g. /home/user/beams/meerkat_lband. \n"
                        "Patterns mathing corr are determined "
                        "automatically. \n"
                        "Only real and imaginary beam models currently "
                        "supported.")
    p.add_argument('-st', "--sparsify-time", type=int, default=10,
                   help="Used to select a subset of time ")
    p.add_argument('-ct', '--corr-type', type=str, default='linear',
                   help="Correlation typ i.e. linear or circular. ")
    p.add_argument('-band', "--band", type=str, default='l',
                   help="Band to use with JimBeam. L or UHF")
    return p

@click.command()
@click.option('-model', "--model", required=True,
              help="Path to model image cube.")
@click.option('-resid', "--residual", required=False,
              help="Path to residual image cube.")
@click.option('-o', '--output-filename',
              help="Path to output directory + prefix. \n"
              "Placed next to input model if outfile not provided.")
@click.option('-pp', '--psf-pars', nargs=3, type=float,
              help="Beam parameters matching FWHM of restoring beam "
                   "specified as emaj emin pa. \n"
                   "By default these are taken from the fits header "
                   "of the residual image."
def spi_fitter(model,
               residual,
               output_filename,
               psf_pars):
#     if args.psf_pars is None:
#         print("Attempting to take psf_pars from residual fits header")
#         try:
#             rhdr = fits.getheader(args.residual)
#         except KeyError:
#             raise RuntimeError("Either provide a residual with beam "
#                                "information or pass them in using --psf_pars "
#                                "argument")
#         if 'BMAJ1' in rhdr.keys():
#             emaj = rhdr['BMAJ1']
#             emin = rhdr['BMIN1']
#             pa = rhdr['BPA1']
#             gaussparf = (emaj, emin, pa)
#         elif 'BMAJ' in rhdr.keys():
#             emaj = rhdr['BMAJ']
#             emin = rhdr['BMIN']
#             pa = rhdr['BPA']
#             gaussparf = (emaj, emin, pa)
#     else:
#         gaussparf = tuple(args.psf_pars)

#     if args.circ_psf:
#         e = np.maximum(gaussparf[0], gaussparf[1])
#         gaussparf = list(gaussparf)
#         gaussparf[0] = e
#         gaussparf[1] = e
#         gaussparf[2] = 0.0
#         gaussparf = tuple(gaussparf)

#     print("Using emaj = %3.2e, emin = %3.2e, PA = %3.2e \n" % gaussparf)

#     # load model image
#     model = load_fits(args.model, dtype=args.out_dtype).squeeze()
#     orig_shape = model.shape
#     mhdr = fits.getheader(args.model)

#     l_coord, ref_l = data_from_header(mhdr, axis=1)
#     l_coord -= ref_l
#     m_coord, ref_m = data_from_header(mhdr, axis=2)
#     m_coord -= ref_m
#     if mhdr["CTYPE4"].lower() == 'freq':
#         freq_axis = 4
#     elif mhdr["CTYPE3"].lower() == 'freq':
#         freq_axis = 3
#     else:
#         raise ValueError("Freq axis must be 3rd or 4th")

#     mfs_shape = list(orig_shape)
#     mfs_shape[0] = 1
#     mfs_shape = tuple(mfs_shape)
#     freqs, ref_freq = data_from_header(mhdr, axis=freq_axis)

#     nband = freqs.size
#     if nband < 2:
#         raise ValueError("Can't produce alpha map from a single band image")
#     npix_l = l_coord.size
#     npix_m = m_coord.size

#     # update cube psf-pars
#     for i in range(1, nband+1):
#         mhdr['BMAJ' + str(i)] = gaussparf[0]
#         mhdr['BMIN' + str(i)] = gaussparf[1]
#         mhdr['BPA' + str(i)] = gaussparf[2]

#     if args.ref_freq is not None and args.ref_freq != ref_freq:
#         ref_freq = args.ref_freq
#         print('Provided reference frequency does not match that of fits file. Will overwrite.')

#     print("Cube frequencies:")
#     with np.printoptions(precision=2):
#         print(freqs)
#     print("Reference frequency is %3.2e Hz \n" % ref_freq)

#     # LB - new header for cubes if ref_freqs differ
#     new_hdr = set_header_info(mhdr, ref_freq, freq_axis, args, gaussparf)

#     # save next to model if no outfile is provided
#     if args.output_filename is None:
#         # strip .fits from model filename
#         tmp = args.model[::-1]
#         idx = tmp.find('.')
#         outfile = args.model[0:-(idx+1)]
#     else:
#         outfile = args.output_filename

#     xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')

#     # load beam
#     if args.beam_model is not None:
#         # we can pass in either a fits file with the already interpolated beam of we can interpolate from scratch
#         if args.beam_model[-5:] == '.fits':
#             bhdr = fits.getheader(args.beam_model)
#             l_coord_beam, ref_lb = data_from_header(bhdr, axis=1)
#             l_coord_beam -= ref_lb
#             if not np.array_equal(l_coord_beam, l_coord):
#                 raise ValueError("l coordinates of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

#             m_coord_beam, ref_mb = data_from_header(bhdr, axis=2)
#             m_coord_beam -= ref_mb
#             if not np.array_equal(m_coord_beam, m_coord):
#                 raise ValueError("m coordinates of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

#             freqs_beam, _ = data_from_header(bhdr, axis=freq_axis)
#             if not np.array_equal(freqs, freqs_beam):
#                 raise ValueError("Freqs of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

#             beam_image = load_fits(args.beam_model, dtype=args.out_dtype).squeeze()
#         elif args.beam_model == "JimBeam":
#             from katbeam import JimBeam
#             if args.band.lower() == 'l':
#                 beam = JimBeam('MKAT-AA-L-JIM-2020')
#             else:
#                 beam = JimBeam('MKAT-AA-UHF-JIM-2020')
#             beam_image = np.zeros(model.shape, dtype=args.out_dtype)
#             for v in range(freqs.size):
#                 beam_image[v] = beam.I(xx, yy, freqs[v])

#         else:
#             beam_image = interpolate_beam(xx, yy, freqs, args)

#         if 'b' in args.products:
#             name = outfile + '.power_beam.fits'
#             save_fits(name, beam_image, mhdr, dtype=args.out_dtype)
#             print("Wrote average power beam to %s \n" % name)

#     else:
#         beam_image = np.ones(model.shape, dtype=args.out_dtype)

#     # beam cut off
#     model = np.where(beam_image > args.pb_min, model, 0.0)

#     if not args.dont_convolve:
#         print("Convolving model")
#         # convolve model to desired resolution
#         model, gausskern = convolve2gaussres(model, xx, yy, gaussparf, args.ncpu, None, args.padding_frac)

#         # save clean beam
#         if 'c' in args.products:
#             name = outfile + '.clean_psf.fits'
#             save_fits(name, gausskern, new_hdr, dtype=args.out_dtype)
#             print("Wrote clean psf to %s \n" % name)

#         # save convolved model
#         if 'm' in args.products:
#             name = outfile + '.convolved_model.fits'
#             save_fits(name, model, new_hdr, dtype=args.out_dtype)
#             print("Wrote convolved model to %s \n" % name)


#     # add in residuals and set threshold
#     if args.residual is not None:
#         resid = load_fits(args.residual, dtype=args.out_dtype).squeeze()
#         rhdr = fits.getheader(args.residual)
#         l_res, ref_lb = data_from_header(rhdr, axis=1)
#         l_res -= ref_lb
#         if not np.array_equal(l_res, l_coord):
#             raise ValueError("l coordinates of residual do not match those of model")

#         m_res, ref_mb = data_from_header(rhdr, axis=2)
#         m_res -= ref_mb
#         if not np.array_equal(m_res, m_coord):
#             raise ValueError("m coordinates of residual do not match those of model")

#         freqs_res, _ = data_from_header(rhdr, axis=freq_axis)
#         if not np.array_equal(freqs, freqs_res):
#             raise ValueError("Freqs of residual do not match those of model")

#         # convolve residual to same resolution as model
#         gausspari = ()
#         for i in range(1,nband+1):
#             key = 'BMAJ' + str(i)
#             if key in rhdr.keys():
#                 emaj = rhdr[key]
#                 emin = rhdr['BMIN' + str(i)]
#                 pa = rhdr['BPA' + str(i)]
#                 gausspari += ((emaj, emin, pa),)
#             else:
#                 print("Can't find Gausspars in residual header, unable to add residuals back in")
#                 gausspari = None
#                 break

#         if gausspari is not None and args.add_convolved_residuals:
#             print("Convolving residuals")
#             resid, _ = convolve2gaussres(resid, xx, yy, gaussparf, args.ncpu, gausspari, args.padding_frac, norm_kernel=False)
#             model += resid
#             print("Convolved residuals added to convolved model")

#             if 'r' in args.products:
#                 name = outfile + '.convolved_residual.fits'
#                 save_fits(name, resid, rhdr)
#                 print("Wrote convolved residuals to %s" % name)



#         counts = np.sum(resid != 0)
#         rms = np.sqrt(np.sum(resid**2)/counts)
#         rms_cube = np.std(resid.reshape(nband, npix_l*npix_m), axis=1).ravel()
#         threshold = args.threshold * rms
#         print("Setting cutoff threshold as %i times the rms "
#             "of the residual " % args.threshold)
#         del resid
#     else:
#         print("No residual provided. Setting  threshold i.t.o dynamic range. "
#             "Max dynamic range is %i " % args.maxDR)
#         threshold = model.max()/args.maxDR
#         rms_cube = None

#     print("Threshold set to %f Jy. \n" % threshold)

#     # get pixels above threshold
#     minimage = np.amin(model, axis=0)
#     maskindices = np.argwhere(minimage > threshold)
#     nanindices = np.argwhere(minimage <= threshold)
#     if not maskindices.size:
#         raise ValueError("No components found above threshold. "
#                         "Try lowering your threshold."
#                         "Max of convolved model is %3.2e" % model.max())
#     fitcube = model[:, maskindices[:, 0], maskindices[:, 1]].T
#     beam_comps = beam_image[:, maskindices[:, 0], maskindices[:, 1]].T

#     # set weights for fit
#     if rms_cube is not None:
#         print("Using RMS in each imaging band to determine weights. \n")
#         weights = np.where(rms_cube > 0, 1.0/rms_cube**2, 0.0)
#         # normalise
#         weights /= weights.max()
#     else:
#         if args.channel_weights is not None:
#             weights = np.array(args.channel_weights)
#             print("Using provided channel weights \n")
#         else:
#             print("No residual or channel weights provided. Using equal weights. \n")
#             weights = np.ones(nband, dtype=np.float64)

#     ncomps, _ = fitcube.shape
#     fitcube = da.from_array(fitcube.astype(np.float64),
#                             chunks=(ncomps//args.ncpu, nband))
#     beam_comps = da.from_array(beam_comps.astype(np.float64),
#                                chunks=(ncomps//args.ncpu, nband))
#     weights = da.from_array(weights.astype(np.float64), chunks=(nband))
#     freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

#     print("Fitting %i components" % ncomps)
#     alpha, alpha_err, Iref, i0_err = fit_spi_components(fitcube, weights, freqsdask,
#                                         np.float64(ref_freq), beam=beam_comps).compute()
#     print("Done. Writing output. \n")

#     alphamap = np.zeros(model[0].shape, dtype=model.dtype)
#     alphamap[...] = np.nan
#     alpha_err_map = np.zeros(model[0].shape, dtype=model.dtype)
#     alpha_err_map[...] = np.nan
#     i0map = np.zeros(model[0].shape, dtype=model.dtype)
#     i0map[...] = np.nan
#     i0_err_map = np.zeros(model[0].shape, dtype=model.dtype)
#     i0_err_map[...] = np.nan
#     alphamap[maskindices[:, 0], maskindices[:, 1]] = alpha
#     alpha_err_map[maskindices[:, 0], maskindices[:, 1]] = alpha_err
#     i0map[maskindices[:, 0], maskindices[:, 1]] = Iref
#     i0_err_map[maskindices[:, 0], maskindices[:, 1]] = i0_err

#     if 'I' in args.products:
#         # get the reconstructed cube
#         Irec_cube = i0map[None, :, :] * \
#             (freqs[:, None, None]/ref_freq)**alphamap[None, :, :]
#         name = outfile + '.Irec_cube.fits'
#         save_fits(name, Irec_cube, mhdr, dtype=args.out_dtype)
#         print("Wrote reconstructed cube to %s" % name)

#     # save alpha map
#     if 'a' in args.products:
#         name = outfile + '.alpha.fits'
#         save_fits(name, alphamap, mhdr, dtype=args.out_dtype)
#         print("Wrote alpha map to %s" % name)

#     # save alpha error map
#     if 'e' in args.products:
#         name = outfile + '.alpha_err.fits'
#         save_fits(name, alpha_err_map, mhdr, dtype=args.out_dtype)
#         print("Wrote alpha error map to %s" % name)

#     # save I0 map
#     if 'i' in args.products:
#         name = outfile + '.I0.fits'
#         save_fits(name, i0map, mhdr, dtype=args.out_dtype)
#         print("Wrote I0 map to %s" % name)

#     # save I0 error map
#     if 'k' in args.products:
#         name = outfile + '.I0_err.fits'
#         save_fits(name, i0_err_map, mhdr, dtype=args.out_dtype)
#         print("Wrote I0 error map to %s" % name)

#     print(' \n ')

#     print("All done here")


# if __name__ == "__main__":
#     args = create_parser().parse_args()

#     if args.ncpu:
#         from multiprocessing.pool import ThreadPool
#         dask.config.set(pool=ThreadPool(args.ncpu))
#     else:
#         import multiprocessing
#         args.ncpu = multiprocessing.cpu_count()

#     print(' \n ')
#     GD = vars(args)
#     print('Input Options:')
#     for key in GD.keys():
#         print(key, ' = ', GD[key])

#     print(' \n ')

#     print("Using %i threads" % args.ncpu)

#     print(' \n ')

#     main(args)
