#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import argparse
import numpy as np
from astropy.io import fits
from pfb.utils import load_fits_contiguous, get_fits_freq_space_info, Gaussian2D
from ducc0.fft import r2c, c2r
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

def create_parser():
    p = argparse.ArgumentParser(description='Simple spectral index fitting'
                                            'tool.',
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-image', "--image", type=str, required=True)
    p.add_argument('-o', '--output-filename', type=str,
                   help="Path to output directory. \n"
                        "Placed next to input model if outfile not provided.")
    p.add_argument('-pp', '--psf-pars', default=None, nargs='+', type=float,
                   help="Beam parameters matching FWHM of restoring beam "
                        "specified as emaj emin pa. \n"
                        "By default these are taken from the fits header "
                        "of the residual image.")
    p.add_argument('-ncpu', '--ncpu', default=0, type=int,
                   help="Number of threads to use. \n"
                        "Default of zero means use all threads")
    p.add_argument('-cp', "--circ-psf", action="store_true",
                   help="Passing this flag will convolve with a circularised "
                   "beam instead of an elliptical one")
    p.add_argument('-bm', '--beam-model', default=None, type=str,
                   help="Fits beam model to use. \n"
                        "Use power_beam_maker to make power beam "
                        "corresponding to image. ")
    p.add_argument('-pb-min', '--pb-min', type=float, default=0.05,
                   help="Set image to zero where pb falls below this value")
    return p

def main(args):
    # read resolution of fits file
    hdr = fits.getheader(args.image)
    l_coord, m_coord, freqs, _, freq_axis = get_fits_freq_space_info(hdr)
    nchan = freqs.size
    psf_pars = {}
    for i in range(1,nchan+1):
        key = 'BMAJ' + str(i)
        if key in hdr.keys():
            emaj = hdr[key]
            emin = hdr['BMIN' + str(i)]
            pa = hdr['BPA' + str(i)]
            psf_pars[i] = (emaj, emin, pa)
    
    if len(psf_pars) == 0 and args.psf_pars is None:
        raise ValueError("No psf parameters in fits file and none passed in.")
    
    if len(psf_pars) == 0:
        print("No psf parameters in fits file. Convolving model to resolution specified by psf-pars.")
        beampars = tuple(args.psf_pars)
    else:
        if args.psf_pars is None:
            beampars = psf_pars[1]
        else:
            beampars = tuple(args.psf_pars)

    if args.circ_psf:
        e = (beampars[0] + beampars[1])/2.0
        beampars[0] = e
        beampars[1] = e
    
    print("Using emaj = %3.2e, emin = %3.2e, PA = %3.2e \n" % beampars)

    # update header
    for i in range(1, nchan+1):
        hdr['BMAJ' + str(i)] = beampars[0]
        hdr['BMIN' + str(i)] = beampars[1]
        hdr['BPA' + str(i)] = beampars[2]

    # coodinate grid
    xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij') 

    # get padding
    npix_l, npix_m = xx.shape
    pfrac = 0.2/2
    npad_l = int(pfrac*npix_l)
    npad_m = int(pfrac*npix_m)
    
    # get fast FFT sizes and update padding
    from scipy.fftpack import next_fast_len
    nfft = next_fast_len(npix_l + 2*npad_l)
    npad_ll = (nfft - npix_l)//2
    npad_lr = nfft - npix_l - npad_ll
    nfft = next_fast_len(npix_m + 2*npad_m)
    npad_ml = (nfft - npix_m)//2
    npad_mr = nfft - npix_m - npad_ml
    padding = ((0, 0), (npad_ll, npad_lr), (npad_ml, npad_mr))
    unpad_l = slice(npad_ll, -npad_lr)
    unpad_m = slice(npad_ml, -npad_mr)
    
    ax = (1, 2)  # axes over which to perform fft
    lastsize = npix_m + np.sum(padding[-1])

    # kernel of desired resolution
    gausskern = Gaussian2D(xx, yy, beampars)
    gausskern = np.pad(gausskern[None], padding, mode='constant')
    gausskernhat = r2c(iFs(gausskern, axes=ax), axes=ax, forward=True, nthreads=args.ncpu, inorm=0)

    # FT of image
    image = load_fits_contiguous(args.image)
    image = np.pad(image, padding, mode='constant')
    imhat = r2c(iFs(image, axes=ax), axes=ax, forward=True, nthreads=args.ncpu, inorm=0)

    # convolve to desired resolution
    if len(psf_pars) == 0:
        imhat *= gausskernhat
    else:
        for i in range(nchan):
            thiskern = Gaussian2D(xx, yy, psf_pars[i+1])
            thiskern = np.pad(thiskern[None], padding, mode='constant')
            thiskernhat = r2c(iFs(thiskern, axes=ax), axes=ax, forward=True, nthreads=args.ncpu, inorm=0)

            convkernhat = np.where(np.abs(thiskernhat)>0.0, gausskernhat/thiskernhat, 0.0)

            imhat[i] *= convkernhat[0]

    image = Fs(c2r(imhat, axes=ax, forward=False, lastsize=lastsize, inorm=2, nthreads=args.ncpu), axes=ax)[:, unpad_l, unpad_m]

    # load beam and correct
    if args.beam_model is not None:
        bhdr = fits.getheader(args.beam_model)
        l_coord_beam, m_coord_beam, freqs_beam, _, freq_axis = get_fits_freq_space_info(bhdr)
        if not np.array_equal(l_coord_beam, l_coord):
            raise ValueError("l coordinates of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

        if not np.array_equal(m_coord_beam, m_coord):
            raise ValueError("m coordinates of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

        if not np.array_equal(freqs, freqs_beam):
            raise ValueError("Freqs of beam model do not match those of image. Use power_beam_maker to interpolate to fits header.")

        beam_image = load_fits_contiguous(args.beam_model)

        image = np.where(beam_image >= args.pb_min, image/beam_image, 0.0)

    # save next to model if no outfile is provided
    if args.output_filename is None:
        # strip .fits from model filename 
        tmp = args.model[::-1]
        idx = tmp.find('.')
        outfile = args.model[0:-idx]
    else:
        outfile = args.output_filename

    

    # save clean beam
    hdu = fits.PrimaryHDU(header=hdr)
    hdu.data = gausskern.T[:, ::-1].astype(np.float32)
    name = outfile + '.clean_psf.fits'
    hdu.writeto(name, overwrite=True)
    print("Wrote clean psf to %s \n" % name)

    hdu = fits.PrimaryHDU(header=hdr)
    # save it
    if freq_axis == 3:
        hdu.data = np.transpose(image, axes=(0, 2, 1))[None, :, :, ::-1].astype(np.float32)
    elif freq_axis == 4:
        hdu.data = np.transpose(image, axes=(0, 2, 1))[:, None, :, ::-1].astype(np.float32)
    name = outfile + '.convolved.fits'
    hdu.writeto(name, overwrite=True)
    print("Wrote convolved model to %s \n" % name)

    print("All done here")

    

if __name__ == "__main__":
    args = create_parser().parse_args()

    if not args.ncpu:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    print(' \n ')
    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])

    print(' \n ')

    print("Using %i threads" % args.ncpu)

    print(' \n ')

    main(args)