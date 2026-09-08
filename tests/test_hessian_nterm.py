"""Which n-term convention keeps the PSF-convolution Hessian closest to the
exact (degrid/grid) operator? (wide-field mosaics, issue #281 follow-on)

Schemes (B = primary beam, n = sqrt(1 - l^2 - m^2), G = wgridder):
  a) status quo: divide_by_n=False everywhere, stored beam B. Self-consistent
     but the model comes out as I/n (flux bias ~ (n-1), 0.2% at a 5 deg fov).
  b) flip ducc divide_by_n=True everywhere, stored beam B. Physically correct
     model, but the 1/n lands INSIDE the gridded PSF as an image-plane
     envelope, which a convolution cannot represent: the Hessian approximation
     degrades with fov.
  c) fold: divide_by_n=False everywhere, stored beam Bn = B/n. The SAME
     physical operator as (b)'s exact one, but the diagonal n-factors ride in
     HessianTree's beam slots and are therefore captured exactly -- the
     approximation error is identical to (a)'s pure PSF-convolution baseline.
     Also immune to ducc's divide_by_n silently no-oping when
     do_wgridding=False.

Measured (nx=128, Gaussian uv coverage, w ~ 0, smooth test image, fov 10
deg): folded(c) == baseline(a) to 4 significant digits in every tested
geometry, while flipped-unfolded(b) is worse by 4-25% depending on how much
pure PSF-convolution error the geometry produces (the n-envelope error is a
fixed systematic on top of it). The exact-operator identity
B*(1/n)*GtWG(B x / n) == (B/n)*GtWG((B/n) x) holds to gridder epsilon.

Independent large error sources in the approximation, out of scope here but
measured along the way: the w-term (a convolution cannot represent it) and
the abs(PSFHAT) rectification (needed for a Hermitian-positive CG operator;
its cost grows with how slowly the PSF decays -- 13% on a pathological
uniform-to-Nyquist synthetic coverage, sub-percent on smooth coverage).
"""

import numpy as np
from ducc0.fft import r2c
from ducc0.wgridder.experimental import dirty2vis, vis2dirty

from pfb_imaging.operators.gridder import wgridder_conventions
from pfb_imaging.operators.hessian import HessianTree

ifftshift = np.fft.ifftshift

NX = NY = 128
FREQ = np.array([1.0e9])
LAM = 299792458.0 / FREQ[0]
NROW = 20000
FLIP_U, FLIP_V, FLIP_W, X0, Y0 = wgridder_conventions(0.0, 0.0)
EPS = 1e-10


def _setup(fov_deg, seed=42):
    rng = np.random.default_rng(seed)
    cell = np.deg2rad(fov_deg) / NX
    umax = 1.0 / (2 * cell)
    uvw = np.zeros((NROW, 3))
    # centrally-condensed coverage -> smooth decaying PSF (realistic array)
    uvw[:, 0] = np.clip(rng.standard_normal(NROW) * umax / 3.5, -umax, umax) * LAM
    uvw[:, 1] = np.clip(rng.standard_normal(NROW) * umax / 3.5, -umax, umax) * LAM
    uvw[:, 2] = rng.uniform(-0.01, 0.01, NROW) * LAM  # near-coplanar: isolate n
    wgt = rng.uniform(0.5, 1.5, (NROW, 1))
    x = (-(NX / 2) + np.arange(NX)) * cell
    y = (-(NY / 2) + np.arange(NY)) * cell
    yy, xx = np.meshgrid(y, x, indexing="ij")
    nlm = np.sqrt(1.0 - xx**2 - yy**2)
    sig = np.deg2rad(fov_deg) / 2 / 2.355
    beam = np.exp(-0.5 * (xx**2 + yy**2) / sig**2)
    yyp, xxp = np.mgrid[0:NY, 0:NX]
    blob = np.exp(-0.5 * ((xxp - 0.7 * NX) ** 2 + (yyp - 0.65 * NY) ** 2) / (NX / 20) ** 2)
    return cell, uvw, wgt, nlm, beam, blob


def _g_wg(x_img, cell, uvw, wgt, dbn):
    """Gt W G x via ducc ((Y, X) order), divide_by_n=dbn."""
    vis = dirty2vis(
        uvw=uvw,
        freq=FREQ,
        dirty=np.ascontiguousarray(x_img.T),
        pixsize_x=cell,
        pixsize_y=cell,
        center_x=X0,
        center_y=Y0,
        flip_u=FLIP_U,
        flip_v=FLIP_V,
        flip_w=FLIP_W,
        epsilon=EPS,
        do_wgridding=True,
        divide_by_n=dbn,
        nthreads=4,
    )
    out = np.zeros((NY, NX))
    vis2dirty(
        uvw=uvw,
        freq=FREQ,
        vis=vis,
        wgt=wgt,
        npix_x=NX,
        npix_y=NY,
        pixsize_x=cell,
        pixsize_y=cell,
        center_x=X0,
        center_y=Y0,
        flip_u=FLIP_U,
        flip_v=FLIP_V,
        flip_w=FLIP_W,
        epsilon=EPS,
        do_wgridding=True,
        divide_by_n=dbn,
        nthreads=4,
        double_precision_accumulation=True,
        dirty=out.T,
    )
    return out


def _psf(cell, uvw, wgt, dbn):
    nxp, nyp = 2 * NX, 2 * NY
    psf = np.zeros((nyp, nxp))
    vis2dirty(
        uvw=uvw,
        freq=FREQ,
        vis=np.ones((NROW, 1), dtype=np.complex128),
        wgt=wgt,
        npix_x=nxp,
        npix_y=nyp,
        pixsize_x=cell,
        pixsize_y=cell,
        center_x=X0,
        center_y=Y0,
        flip_u=FLIP_U,
        flip_v=FLIP_V,
        flip_w=FLIP_W,
        epsilon=EPS,
        do_wgridding=True,
        divide_by_n=dbn,
        nthreads=4,
        double_precision_accumulation=True,
        dirty=psf.T,
    )
    return psf


def _hess(beam_eff, psf, wsum):
    psfhat = np.abs(r2c(ifftshift(psf), axes=(0, 1), forward=True, inorm=0))
    parts = [{"psfhat": psfhat[None], "beam": beam_eff[None], "wsum": np.array([wsum])}]
    return HessianTree(parts, NX, NY, psf.shape[1], psf.shape[0], eta=0.0, wsum=np.array([wsum]), nthreads=4)


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_folding_n_into_beam_is_exact_and_optimal():
    """Folded B/n Hessian == pure-PSF baseline; unfolded True is worse at 10 deg."""
    cell, uvw, wgt, nlm, beam, blob = _setup(10.0)
    wsum = wgt.sum()
    beam_n = beam / nlm

    # the fold is an exact operator identity (ducc dbn=True is diag(1/n) on
    # both sides under do_wgridding=True)
    rng = np.random.default_rng(1)
    ximg = rng.standard_normal((NY, NX))
    e_true = beam * _g_wg(beam * ximg, cell, uvw, wgt, dbn=True)
    e_fold = beam_n * _g_wg(beam_n * ximg, cell, uvw, wgt, dbn=False)
    assert _rel(e_true, e_fold) < 1e-9

    psf_f = _psf(cell, uvw, wgt, dbn=False)
    psf_t = _psf(cell, uvw, wgt, dbn=True)

    # approximation errors on a smooth compact source at 0.7 fov radius
    ex_selfconsistent = beam * _g_wg(beam * blob, cell, uvw, wgt, dbn=False) / wsum
    ex_physical = beam_n * _g_wg(beam_n * blob, cell, uvw, wgt, dbn=False) / wsum
    err_baseline = _rel(_hess(beam, psf_f, wsum).dot(blob)[0], ex_selfconsistent)
    err_flipped = _rel(_hess(beam, psf_t, wsum).dot(blob)[0], ex_physical)
    err_folded = _rel(_hess(beam_n, psf_f, wsum).dot(blob)[0], ex_physical)

    # folding adds nothing on top of the pure PSF-convolution baseline
    assert err_folded < 1.01 * err_baseline, (err_folded, err_baseline)
    # the unfolded flip is systematically worse. The margin depends on how
    # much pure PSF-convolution error the geometry produces (measured ratios
    # 1.04-1.25 across row counts / grid sizes; 1.088 at this seed): assert a
    # conservative 2%, deterministic under the fixed seed.
    assert err_flipped > 1.02 * err_folded, (err_flipped, err_folded)
