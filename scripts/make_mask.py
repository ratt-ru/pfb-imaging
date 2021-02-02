#!/usr/bin/env python

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, median_filter, map_coordinates
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
from numba import njit, prange
from pfb.utils import data_from_header, save_fits, load_fits
from pfb.operators import PSI
import argparse


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="Restored image", type=str)
    p.add_argument("--region", default=None, help="Region based mask (optional)", type=str)
    p.add_argument("--sigma", default=1, help="Dilate mask sigma times", type=int)
    p.add_argument("--threshold", help="Absolute masking threshold to apply on reconstructed image", type=float)
    p.add_argument("--soft_threshold", default=1e-3, help="Soft thresholding for wavelet coefficients", type=float)
    p.add_argument("--outname", help="Mask out name", type=str)
    p.add_argument("--dtype", default='f8', help="single or double precision as f4 and f8 respectively", type=str)
    return p

@njit(fastmath=True, parallel=True)
def sliding_rms(image, box_size):
    nx, ny = image.shape
    rms = np.zeros((nx, ny), dtype=args.dtype)
    nstartx = box_size//2
    nendx = nx - box_size//2
    nstarty = box_size//2
    nendy = ny - box_size//2
    ind = np.arange(-(box_size//2), box_size//2)
    nnorm = box_size**2
    for i in prange(nstartx, nendx):
        for j in range(nstarty, nendy):
            for k in range(i-box_size//2, i + box_size//2):
                for l in range(j-box_size//2, j + box_size//2):
                    rms[i, j] += image[k, l]**2/nnorm
    return rms

@njit(fastmath=True, parallel=True)
def map_region(region, ox, oy, x, y):
    nx = x.size
    ny = y.size
    new_region = np.zeros((nx, ny), dtype=region.dtype)
    xinds = np.arange(ox.size)
    yinds = np.arange(oy.size)
    for i in prange(nx):
        for j in range(ny):
            tmpx = np.abs(ox - x[i])
            ix = xinds[tmpx == tmpx.min()][0]
            tmpy = np.abs(oy - y[j])
            iy = yinds[tmpy == tmpy.min()][0]
            new_region[i, j] = region[ix, iy]
            # print(new_region[i, j], region[ix, iy])
    return new_region

def main(args):
    image = load_fits(args.image, args.dtype).squeeze()
    hdr = fits.getheader(args.image)
    x_coords, _ = data_from_header(hdr, axis=1)
    y_coords, _ = data_from_header(hdr, axis=2)
    nx, ny = image.shape

    # print(x_coords)
    # print(y_coords)

    # quit()

    # check that cords lie in [0,360)
    if (x_coords < 0.0).any():
        x_coords += 360.0
    elif (x_coords >=360.0).any():
        x_coords -= 360
    if (y_coords < 0.0).any():
        y_coords += 360.0
    elif (y_coords >=360.0).any():
        y_coords -= 360

    if args.region is not None:
        region = fits.getdata(args.region).squeeze().astype(args.dtype)[::-1].T
        hdr = fits.getheader(args.region)
        old_x_coords = data_from_header(hdr, axis=1)
        old_y_coords = data_from_header(hdr, axis=2)

        if (old_x_coords < 0.0).any():
            old_x_coords += 360.0
        elif (old_x_coords >=360.0).any():
            old_x_coords -= 360
        if (old_y_coords < 0.0).any():
            old_y_coords += 360.0
        elif (old_y_coords >=360.0).any():
            old_y_coords -= 360

        if not (x_coords==old_x_coords) or not (y_coords==old_y_coords):
            print("Mapping region")
            new_region = map_region(region, old_x_coords, old_y_coords, x_coords, y_coords)
        else:
            print("Coordinates match")
            new_region = region
    else:
        print("Ended up here")
        new_region = np.ones_like(image)
    
    
    # apply the region file before wavelet decomposition (seems to improve effectiveness of thresholding)
    image *= new_region
    
    # wavelet decomposition + thresholding to get rid of spurious structure
    psi = PSI(imsize=(1, nx, ny), nlevels=2)  # TODO - adjust nlevels for image size

    alpha = psi.hdot(image[None])
    alpha = np.maximum(np.abs(alpha) - args.soft_threshold, 0.0) * np.sign(alpha)
    model = psi.dot(alpha)[0]
    
    # mask based on threshold and region
    mask = np.where(model * new_region > args.threshold, 1.0, 0.0)
    
    # dilate the mask a little by convolving with Gassian of width sigma
    if args.sigma:
        mask = binary_dilation(input=mask, iterations=args.sigma)

    plt.figure(1)
    plt.imshow(model, vmin=0.0, vmax=0.01*model.max())
    plt.colorbar()

    plt.figure(2)
    plt.imshow(image, vmin=0.0, vmax=0.01*model.max())
    plt.colorbar()

    plt.figure(3)
    plt.imshow(mask)
    plt.colorbar()
    
    if args.region is not None:
        plt.figure(4)
        plt.imshow(new_region)
        plt.colorbar()
    
    plt.show()

    
    hdr = fits.getheader(args.image)
    save_fits(args.outname, mask, hdr)


if __name__=="__main__":
    args = create_parser().parse_args()
    
    main(args)