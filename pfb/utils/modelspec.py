import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import sympy as sm
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr

specs = ['genesis', 'exodus']

def fit_image_cube(time, freq, image, wgt=None, nbasist=None, nbasisf=None,
                   method='poly', sigmasq=0):
    '''
    Fit the time and frequency axes of an image cube where

    time    - (ntime) time axis
    freq    - (nband) frequency axis
    image   - (ntime, nband, nx, ny) pixelated image
    wgt     - (ntime, nband) optional per time and frequency weights
    nbasist - number of time basis functions
    nbasisf - number of frequency basis functions
    method  - method to use for fitting (poly or Legendre)
    sigmasq - optional regularisation term to add to the Hessian
              to improve conditioning

    method:
    poly     - fit a monomials in time and frequency
    Legendre - fit a Legendre polynomial in time and frequency

    returns:
    coeffs  - fitted coefficients
    Ix, Iy  - pixel locations of non-zero pixels in the image
    expr    - a string representing the symbolic expression describing the fit
    params  - tuple of str, parameters to pass into function (excluding t and f)
    tfunc   - function which scales the time domain appropriately for method
    ffunc   - function which scales the frequency domain appropriately for method


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
        raise NotImplementedError(f"Method {method} not implemented")

    dirty_coeffs = Xfit.T.dot(wgt*beta)
    hess_coeffs = Xfit.T.dot(wgt*Xfit)
    # to improve conditioning
    if sigmasq:
        hess_coeffs += sigmasq*np.eye(hess_coeffs.shape[0])
    coeffs = np.linalg.solve(hess_coeffs, dirty_coeffs)

    return coeffs, Ix, Iy, str(expr), list(map(str,params)), str(tfunc),str(ffunc)


def fit_image_fscube(freq, image,
                     wgt=None, nbasisf=None,
                     method='Legendre', sigmasq=0):
    '''
    Fit the frequency axis of an image cube where

    freq    - (nband,) frequency axis
    image   - (nband, ncorr, nx, ny) pixelated image
    wgt     - (nband, ncorr) optional per time and frequency weights
    nbasisf - number of frequency basis functions
    method  - method to use for fitting (poly or Legendre)
    sigmasq - optional regularisation term to add to the Hessian
              to improve conditioning

    method:
    poly     - fit a monomials to frequency axis
    Legendre - fit a Legendre polynomial to frequency

    returns:
    coeffs  - (ncorr, nbasisf, ncomps) fitted coefficients
    Ix, Iy  - (ncomps,) pixel locations of non-zero pixels in the image
    expr    - a string representing the symbolic expression describing the fit
    params  - tuple of str, parameters to pass into function (excluding t and f)
    ffunc   - function which scales the frequency domain appropriately for method
    '''
    nband = freq.size
    ref_freq = freq[0]
    import sympy as sm
    from sympy.abc import f

    if nbasisf is None:
        nbasisf = nband
    else:
        assert nbasisf <= nband

    nband, ncorr, nx, ny = image.shape
    mask = np.any(image, axis=(0,1))  # over freq and corr axes
    Ix, Iy = np.where(mask)
    ncomps = Ix.size

    # components excluding zeros
    beta = image[:, :, Ix, Iy].reshape(nband, ncorr, ncomps)
    if wgt is not None:
        wgt = wgt.reshape(nband, ncorr, 1)
    else:
        wgt = np.ones((nband, ncorr, 1), dtype=float)

    params = sm.symbols(f'f(0:{nbasisf})')
    if nband==1:  # nothing to fit
        coeffs = beta
        expr = f
        params = (f,)
    elif method=='poly':
        wf = freq/ref_freq
        ffunc = f/ref_freq
        Xf = np.tile(wf[:, None], (1, nbasisf))**np.arange(nbasisf)
        expr = sum(co*f**i for i, co in enumerate(params))

    elif method=='Legendre':
        Xf = np.zeros((nband, nbasisf), dtype=float)
        fmax = freq.max()
        fmin = freq.min()
        wf = freq - (fmax + fmin)/2
        wfmax = wf.max()
        wf /= wfmax
        ffunc = (f - (fmax + fmin)/2)/wfmax
        Xf[:, 0] = 1.0
        expr = params[0]
        for i in range(1, nbasisf):
            vals = np.polynomial.Legendre.basis(i)(wf)
            Xf[:, i] = vals
            expr += sm.polys.orthopolys.legendre_poly(i, f)*params[i]
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    # fit each correlation separately
    coeffs = np.zeros((ncorr, nbasisf, ncomps), dtype=beta.dtype)
    for c in range(ncorr):
        dirty_coeffs = Xf.T.dot(wgt[:, c]*beta[:, c])
        hess_coeffs = Xf.T.dot(wgt[:, c]*Xf)
        # to improve conditioning
        if sigmasq:
            hess_coeffs += sigmasq*np.eye(hess_coeffs.shape[0])
        coeffs[c] = np.linalg.solve(hess_coeffs, dirty_coeffs)


    return coeffs, Ix, Iy, str(expr), list(map(str,params)), str(ffunc)


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


def model_from_mds(mds_name, freqs=None):
    '''
    Evaluate component model at the original resolution
    '''
    mds = xr.open_zarr(mds_name, chunks=None)
    if freqs is None:
        freqs = mds.freqs.values
    else:
        freqs = np.atleast_1d(freqs)
    return eval_coeffs_to_cube(mds.times.values,
                               freqs,
                               mds.npix_x, mds.npix_y,
                               mds.coefficients.values,
                               mds.location_x.values, mds.location_y.values,
                               mds.parametrisation,
                               mds.params.values,
                               mds.texpr, mds.fexpr)