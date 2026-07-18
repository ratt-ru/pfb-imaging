import dask.array as da
import numpy as np
from ducc0.fft import r2c
from ducc0.misc import resize_thread_pool
from ducc0.wgridder.experimental import dirty2vis, vis2dirty

from pfb_imaging.utils.misc import fitcleanbeam
from pfb_imaging.utils.weighting import counts_to_weights

ifftshift = np.fft.ifftshift
fftshift = np.fft.fftshift


def wgridder_conventions(l0, m0):
    """
    Returns

    flip_u, flip_v, flip_w, x0, y0

    according to the conventions documented here https://github.com/mreineck/ducc/issues/34

    Note that these conventions are stored as dataset attributes in order
    to call the operators acting on datasets with a consistent convention.
    """
    return False, True, False, -l0, -m0


def comps2vis(
    uvw,
    utime,
    freq,
    rbin_idx,
    rbin_cnts,
    tbin_idx,
    tbin_cnts,
    fbin_idx,
    fbin_cnts,
    region_mask,
    mds,
    modelf,
    tfunc,
    ffunc,
    epsilon=1e-7,
    nthreads=1,
    do_wgridding=True,
    divide_by_n=False,
    freq_min=-np.inf,
    freq_max=np.inf,
    product="I",
):
    # determine output type
    complex_type = da.result_type(mds.coefficients.dtype, np.complex64)
    ncorr_out = len(product)

    return da.blockwise(
        _comps2vis,
        "rfc",
        uvw,
        "r3",
        utime,
        "r",
        freq,
        "f",
        rbin_idx,
        "r",
        rbin_cnts,
        "r",
        tbin_idx,
        "r",
        tbin_cnts,
        "r",
        fbin_idx,
        "f",
        fbin_cnts,
        "f",
        region_mask,
        None,
        mds,
        None,
        modelf,
        None,
        tfunc,
        None,
        ffunc,
        None,
        epsilon,
        None,
        nthreads,
        None,
        do_wgridding,
        None,
        divide_by_n,
        None,
        freq_min,
        None,
        freq_max,
        None,
        product,
        None,
        new_axes={"c": ncorr_out},
        # it should be getting these from uvw and freq?
        adjust_chunks={"r": uvw.chunks[0]},
        dtype=complex_type,
        align_arrays=False,
    )


def _comps2vis(
    uvw,
    utime,
    freq,
    rbin_idx,
    rbin_cnts,
    tbin_idx,
    tbin_cnts,
    fbin_idx,
    fbin_cnts,
    region_mask,
    mds,
    modelf,
    tfunc,
    ffunc,
    epsilon=1e-7,
    nthreads=1,
    do_wgridding=True,
    divide_by_n=False,
    freq_min=-np.inf,
    freq_max=np.inf,
    product="I",
):
    return _comps2vis_impl(
        uvw[0],
        utime,
        freq,
        rbin_idx,
        rbin_cnts,
        tbin_idx,
        tbin_cnts,
        fbin_idx,
        fbin_cnts,
        region_mask,
        mds,
        modelf,
        tfunc,
        ffunc,
        epsilon=epsilon,
        nthreads=nthreads,
        do_wgridding=do_wgridding,
        divide_by_n=divide_by_n,
        freq_min=freq_min,
        freq_max=freq_max,
        product=product,
    )


def _comps2vis_impl(
    uvw,
    utime,
    freq,
    rbin_idx,
    rbin_cnts,
    tbin_idx,
    tbin_cnts,
    fbin_idx,
    fbin_cnts,
    region_mask,
    mds,
    modelf,
    tfunc,
    ffunc,
    epsilon=1e-7,
    nthreads=1,
    do_wgridding=True,
    divide_by_n=False,
    freq_min=-np.inf,
    freq_max=np.inf,
    product="I",
):
    # why is this necessary?
    resize_thread_pool(nthreads)

    # adjust for chunking
    # need a copy here if using multiple row chunks
    rbin_idx2 = rbin_idx - rbin_idx.min()
    tbin_idx2 = tbin_idx - tbin_idx.min()
    fbin_idx2 = fbin_idx - fbin_idx.min()

    ntime = tbin_idx.size
    nband = fbin_idx.size

    nrow = uvw.shape[0]
    nchan = freq.size
    nstokes_out = len(product)
    vis = np.zeros((nrow, nchan, nstokes_out), dtype=np.result_type(mds.coefficients.dtype, np.complex64))
    if not ((freq >= freq_min) & (freq <= freq_max)).any():
        return vis

    comps = mds.coefficients.values
    x_index = mds.location_x.values
    y_index = mds.location_y.values
    cellx = mds.cell_rad_x
    celly = mds.cell_rad_x
    nx = mds.npix_x
    ny = mds.npix_y
    # these are taken from dataset attrs to make sure they remain consistent
    x0 = mds.center_x
    y0 = mds.center_y
    flip_u = mds.flip_u
    flip_v = mds.flip_v
    flip_w = mds.flip_w
    for t in range(ntime):
        indt = slice(tbin_idx2[t], tbin_idx2[t] + tbin_cnts[t])
        # TODO - clean up this logic. row_mapping holds the number of rows per
        # utime and there are multiple utimes per row chunk
        indr = slice(rbin_idx2[indt][0], rbin_idx2[indt][-1] + rbin_cnts[indt][-1])
        for b in range(nband):
            indf = slice(fbin_idx2[b], fbin_idx2[b] + fbin_cnts[b])
            f = freq[indf]
            # don't degrid outside requested frequency range
            if not ((f >= freq_min) & (f <= freq_max)).any():
                continue
            # render components to image
            tout = tfunc(np.mean(utime[indt]))
            fout = ffunc(np.mean(freq[indf]))
            image = np.zeros((nx, ny), dtype=comps.dtype)
            image[x_index, y_index] = modelf(tout, fout, *comps[:, :])  # too magical?
            if np.any(region_mask):
                image = np.where(region_mask, image, 0.0)
                for c in range(nstokes_out):
                    vis[indr, indf, c] = dirty2vis(
                        uvw=uvw,
                        freq=f,
                        dirty=image,
                        pixsize_x=cellx,
                        pixsize_y=celly,
                        center_x=x0,
                        center_y=y0,
                        flip_u=flip_u,
                        flip_v=flip_v,
                        flip_w=flip_w,
                        epsilon=epsilon,
                        do_wgridding=do_wgridding,
                        divide_by_n=divide_by_n,
                        nthreads=nthreads,
                    )

    return vis


def grid_partition(
    part,
    counts,
    nx,
    ny,
    nx_psf,
    ny_psf,
    cell_rad,
    robustness=None,
    nx_pad=None,
    ny_pad=None,
    l0=0.0,
    m0=0.0,
    nthreads=1,
    epsilon=1e-7,
    do_wgridding=True,
    double_accum=True,
    do_psf=True,
):
    """Grid one data partition's image-space products with ducc0 (casacore-free).

    Applies imaging weights from a reduced counts grid, then grids the dirty
    image, PSF, its FT and the primary beam. This is the per-partition term of
    the sum-over-partitions Hessian; the caller sums the returned image-space
    products over a band's partitions.

    Args:
        part: Partition dataset with corr-first ``VIS``/``WEIGHT`` ``(corr,row,chan)``,
            ``MASK`` ``(row,chan)``, ``UVW`` ``(row,3)``, ``FREQ`` ``(chan,)`` and
            ``BEAM`` ``(corr, ny, nx)`` already on the output image grid
            (placed there in pass 1; see stokes_vis).
        counts: Reduced counts grid ``(ncorr, nx_pad, ny_pad)`` (already passed
            through ``filter_extreme_counts``/``box_sum_counts`` by the caller).
            Ignored when ``robustness`` is ``None``; copied before in-place use.
        nx, ny, nx_psf, ny_psf: Image and PSF sizes.
        cell_rad: Image cell size (rad).
        robustness: Briggs robustness; ``None`` (or ``> 2``) leaves natural weights.
        nx_pad, ny_pad: Padded uv-grid size matching ``counts``.
        l0, m0: Partition phase-centre offset (rad).
        nthreads, epsilon, do_wgridding, double_accum: gridder controls.
        do_psf: skip the PSF/PSFHAT/PSFPARSN products entirely when False
            (quicklook mode).

    Returns:
        dict with ``DIRTY`` ``(corr,ny,nx)``, ``BEAM`` ``(corr,ny,nx)``,
        ``WSUM`` ``(corr,)`` and the imaging ``WEIGHT`` ``(corr,row,chan)``.
        ``PSF`` ``(corr,ny_psf,nx_psf)``, ``PSFHAT`` ``(corr,ny_psf,nx_psf//2+1)``
        and ``PSFPARSN`` ``(corr,3)`` are present only when ``do_psf``.
        Image-space arrays are (Y, X)-ordered (wiki D19); ducc's x-major
        layout exists only behind transposed views.
    """
    resize_thread_pool(nthreads)
    flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(l0, m0)
    uvw = part.UVW.values
    vis = part.VIS.values
    wgt = part.WEIGHT.values.copy()
    mask = part.MASK.values
    freq = part.FREQ.values
    ncorr = part.corr.size

    # imaging weights from the reduced counts grid (counts_to_weights mutates
    # both weight and counts in place, so pass a copy of the shared counts)
    if robustness is not None:
        wgt = counts_to_weights(
            counts.copy(),
            uvw,
            freq,
            wgt,
            mask,
            nx_pad,
            ny_pad,
            cell_rad,
            cell_rad,
            robustness,
            usign=1.0 if flip_u else -1.0,
            vsign=1.0 if flip_v else -1.0,
        )

    wsum = wgt[:, mask.astype(bool)].sum(axis=-1)

    # the piece BEAM is already on the output image grid, placed there in
    # pass 1 (stokes_vis, #281): (corr, ny, nx), tangent point + target offset
    # included. Pieces of a partition share the field (and the rotation-
    # averaged beam is time-independent), so the first piece's beam stands in
    # for the partition; replace with a weighted mean when time-dependent
    # beams arrive.
    beam = np.require(part.BEAM.values, dtype=float)

    # ducc's x-major world exists only at the vis2dirty seams below via
    # zero-copy transposed views (the hci cfc96b9 pattern)
    dirty = np.zeros((ncorr, ny, nx), dtype=float)
    for c in range(ncorr):
        vis2dirty(
            uvw=uvw,
            freq=freq,
            vis=vis[c],
            wgt=wgt[c],
            mask=mask,
            npix_x=nx,
            npix_y=ny,
            pixsize_x=cell_rad,
            pixsize_y=cell_rad,
            center_x=x0,
            center_y=y0,
            epsilon=epsilon,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            do_wgridding=do_wgridding,
            divide_by_n=False,
            nthreads=nthreads,
            sigma_min=1.1,
            sigma_max=3.0,
            double_precision_accumulation=double_accum,
            dirty=dirty[c].T,
        )

    if do_psf:
        # PSF visibilities carry a phase ramp when the field is off image centre:
        # they must be the response of a delta function sitting at the image
        # centre (l0, m0), so that re-gridding with the same center_x/center_y
        # places its peak back at the PSF array's own centre. Predicting from an
        # actual unit-peak image via dirty2vis (rather than a hand-rolled trig
        # formula) guarantees the phase is the exact adjoint of the vis2dirty
        # call below for these same center_x/center_y/flip_* conventions.
        if x0 or y0:
            # match the weight dtype so single-precision runs get complex64 psf_vis
            # (ducc rejects mixed complex128 vis + float32 wgt); (Y, X) raster with
            # the ducc seam behind a zero-copy .T view, as everywhere else
            delta_im = np.zeros((ny, nx), dtype=wgt.dtype)
            delta_im[ny // 2, nx // 2] = 1.0
            psf_vis = dirty2vis(
                uvw=uvw,
                freq=freq,
                dirty=delta_im.T,
                pixsize_x=cell_rad,
                pixsize_y=cell_rad,
                center_x=x0,
                center_y=y0,
                epsilon=epsilon,
                flip_u=flip_u,
                flip_v=flip_v,
                flip_w=flip_w,
                do_wgridding=do_wgridding,
                divide_by_n=False,
                nthreads=nthreads,
            )
        else:
            psf_vis = np.broadcast_to(np.ones((1,), dtype=vis.dtype), (uvw.shape[0], freq.size))

        psf = np.zeros((ncorr, ny_psf, nx_psf), dtype=float)
        for c in range(ncorr):
            vis2dirty(
                uvw=uvw,
                freq=freq,
                vis=psf_vis,
                wgt=wgt[c],
                mask=mask,
                npix_x=nx_psf,
                npix_y=ny_psf,
                pixsize_x=cell_rad,
                pixsize_y=cell_rad,
                center_x=x0,
                center_y=y0,
                flip_u=flip_u,
                flip_v=flip_v,
                flip_w=flip_w,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                divide_by_n=False,
                nthreads=nthreads,
                sigma_min=1.1,
                sigma_max=3.0,
                double_precision_accumulation=double_accum,
                dirty=psf[c].T,
            )
        psfhat = r2c(ifftshift(psf, axes=(1, 2)), axes=(1, 2), nthreads=nthreads, forward=True, inorm=0)
        psfparsn = np.array(fitcleanbeam(psf, level=0.5, pixsize=1.0, yx_order=True))

    out = {
        "DIRTY": dirty,
        "BEAM": beam,
        "WSUM": wsum,
        "WEIGHT": wgt,
    }
    if do_psf:
        out["PSF"] = psf
        out["PSFHAT"] = psfhat
        out["PSFPARSN"] = psfparsn
    return out


def residual_from_partitions(
    dirty,
    parts,
    model,
    cell_rad,
    nthreads=1,
    epsilon=1e-7,
    do_wgridding=True,
    double_accum=True,
):
    """Recompute a band's residual by summing the exact degrid/grid over partitions.

    The DataTree analogue of ``compute_residual``: it reuses the per-partition
    gridding inputs already stored in pass 2 (``UVW``/``WEIGHT``/``MASK``/
    ``FREQ``/image-grid ``BEAM``) and never recomputes the PSF, so it is cheap
    enough to call once per major cycle. The beam is applied once on the degrid
    side, matching the once-attenuated convention of ``compute_residual`` and of
    the stored ``DIRTY``.

    Args:
        dirty: Band-node dirty image ``(corr, ny, nx)`` (sum over partitions,
            un-normalised by wsum, as produced in pass 2).
        parts: Iterable of partition datasets, each with corr-first ``WEIGHT``,
            ``MASK`` ``(row,chan)``, ``UVW`` ``(row,3)``, ``FREQ`` ``(chan,)`` and
            image-grid ``BEAM`` ``(corr,ny,nx)``, plus ``l0``/``m0`` attrs (0 if absent).
        model: Band model image ``(corr, ny, nx)``.
        cell_rad: Image cell size (rad).
        nthreads, epsilon, do_wgridding, double_accum: gridder controls.

    Returns:
        Residual image ``(corr, ny, nx)`` = ``dirty - Σ_p G_pᵀ W_p G_p (beam_p * model)``.
    """
    resize_thread_pool(nthreads)
    ncorr, ny, nx = dirty.shape
    convim = np.zeros_like(dirty)
    tmp = np.zeros((ny, nx), dtype=dirty.dtype)
    for part in parts:
        uvw = part.UVW.values
        wgt = part.WEIGHT.values
        mask = part.MASK.values
        freq = part.FREQ.values
        beam = part.BEAM.values
        l0 = part.attrs.get("l0", 0.0)
        m0 = part.attrs.get("m0", 0.0)
        flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(l0, m0)
        for c in range(ncorr):
            model_vis = dirty2vis(
                uvw=uvw,
                freq=freq,
                # (Y, X) product adapted to ducc's x-major input by a
                # zero-copy strided view
                dirty=(beam[c] * model[c]).T,
                pixsize_x=cell_rad,
                pixsize_y=cell_rad,
                center_x=x0,
                center_y=y0,
                flip_u=flip_u,
                flip_v=flip_v,
                flip_w=flip_w,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                nthreads=nthreads,
                divide_by_n=False,
                sigma_min=1.1,
                sigma_max=3.0,
            )
            vis2dirty(
                uvw=uvw,
                freq=freq,
                vis=model_vis,
                wgt=wgt[c],
                mask=mask,
                npix_x=nx,
                npix_y=ny,
                pixsize_x=cell_rad,
                pixsize_y=cell_rad,
                center_x=x0,
                center_y=y0,
                flip_u=flip_u,
                flip_v=flip_v,
                flip_w=flip_w,
                epsilon=epsilon,
                do_wgridding=do_wgridding,
                divide_by_n=False,
                nthreads=nthreads,
                sigma_min=1.1,
                sigma_max=3.0,
                double_precision_accumulation=double_accum,
                dirty=tmp.T,
            )
            convim[c] += tmp

    return dirty - convim
