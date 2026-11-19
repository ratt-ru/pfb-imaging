import numba
import numpy as np
import xarray as xr
from numba import literally, njit, prange
from numba.extending import overload, register_jitable
from rarg_numba_patterns import load_data
from scipy.constants import c as lightspeed
from scipy.ndimage import uniform_filter

from pfb_imaging.utils.stokes import stokes_expr_funcs

ifftshift = np.fft.ifftshift
fftshift = np.fft.fftshift

JIT_OPTIONS = {
    "nogil": True,
    "cache": True,
    "error_model": "numpy",
    # "fastmath": True  # causes weighting tests to fail
}


@njit(nogil=True, cache=True)
def _accumulate_counts(uvw, freq, mask, wgt, counts, seen, cell_size_x, cell_size_y, usign, vsign):
    """Nearest-cell sum of the weights into ``counts`` (folded, v >= 0 padded uv grid).

    Serial on purpose: per-thread grids plus their sum held nthreads + 1 padded
    grids (1.6 GiB at 3840^2 with 4 threads) and, measured on 1.3M-27M samples
    on a 6528^2 grid, were no faster -- zeroing and summing whole grids costs
    more than the scatter they parallelise (#339).

    Returns ``(cells, offgrid)``. With a non-empty ``seen`` (``nx * ny`` u1,
    zeroed) ``cells`` holds the flat ``u * ny + v`` cells it touched, in
    first-touch order, so a caller can read the occupied cells without scanning
    the grid (#365); with an empty one it is empty. ``offgrid`` is the float64
    per-correlation weight of the unflagged samples beyond the grid, which keep
    their natural weight in ``counts_to_weights`` (``piece_imaging_wsum``).
    """
    # factor of 2 because -umax/2 <= u < umax
    nx, ny = counts.shape[1], counts.shape[2]
    u_cell = 1 / (nx * cell_size_x)
    umax = np.abs(1 / cell_size_x / 2)
    v_cell = 1 / (ny * cell_size_y)
    vmax = np.abs(1 / cell_size_y / 2)

    track = seen.size > 0
    cells = np.empty(1024 if track else 0, dtype=np.int64)
    ncell = 0
    ncorr = wgt.shape[0]
    offgrid = np.zeros(ncorr, dtype=np.float64)
    nrow, nchan = mask.shape
    for r in range(nrow):
        uvw_row = uvw[r]
        wgt_row = wgt[:, r]
        mask_row = mask[r]
        for f in range(nchan):
            if not mask_row[f]:
                continue
            # current uv coords
            chan_normfreq = freq[f] / lightspeed
            u_tmp = uvw_row[0] * chan_normfreq * usign
            v_tmp = uvw_row[1] * chan_normfreq * vsign
            if v_tmp < 0:
                u_tmp = -u_tmp
                v_tmp = -v_tmp
            # pixel coordinates
            ug = (u_tmp + umax) / u_cell
            vg = (v_tmp + vmax) / v_cell
            # indices
            u_idx = np.int32(np.floor(ug))
            v_idx = np.int32(np.floor(vg))

            if (u_idx < 0) or (u_idx >= nx) or (v_idx < 0) or (v_idx >= ny):
                # out of bounds so continue.
                # raising an error means we can't grid at sub-Nyquist
                for c in range(ncorr):
                    offgrid[c] += wgt_row[c, f]
                continue

            if track:
                k = np.int64(u_idx) * ny + v_idx
                if not seen[k]:
                    seen[k] = 1
                    if ncell == cells.size:
                        grown = np.empty(2 * cells.size, dtype=np.int64)
                        grown[:ncell] = cells
                        cells = grown
                    cells[ncell] = k
                    ncell += 1

            # nearest neighbour
            counts[:, u_idx, v_idx] += wgt_row[:, f]

    return cells[:ncell], offgrid


@njit(nogil=True, cache=True)
def _compute_counts(uvw, freq, mask, wgt, nx, ny, cell_size_x, cell_size_y, dtype, usign=1.0, vsign=-1.0):
    """Dense ``(corr, nx, ny)`` counts grid; see ``_accumulate_counts``."""
    counts = np.zeros((wgt.shape[0], nx, ny), dtype=dtype)
    _accumulate_counts(
        uvw, freq, mask, wgt, counts, np.empty(0, dtype=np.uint8), cell_size_x, cell_size_y, usign, vsign
    )
    return counts


def compute_sparse_counts(
    uvw, freq, mask, wgt, nx, ny, cell_size_x, cell_size_y, dtype, usign=1.0, vsign=-1.0, accum_dtype=None
):
    """A piece's counts as its occupied cells, without scanning the dense grid (#365).

    Equal to ``sparse_counts(_compute_counts(...))`` when ``accum_dtype`` is
    ``dtype``. The grid and the touched mask are allocated by numpy, whose
    zeroed allocations are lazily mapped, so only the pages the piece's samples
    land on are ever touched; the occupied cells come from the scatter itself
    rather than ``np.flatnonzero`` over the whole padded grid, which #365
    measured at 6% of pass 1.

    Args:
        uvw, freq, mask, wgt, nx, ny, cell_size_x, cell_size_y, usign, vsign:
            as ``_compute_counts``.
        dtype: dtype of the stored ``COUNTS_VALUE`` (the data precision, D27).
        accum_dtype: dtype the cells are accumulated in (``--double-accum``
            gives float64); defaults to ``dtype``. Values are rounded to
            ``dtype`` once, after accumulation.

    Returns:
        ``(data_vars, wsum_offgrid)``: ``COUNTS_INDEX``/``COUNTS_VALUE`` as
        ``sparse_counts`` gives them, and the corr-summed natural weight of the
        unflagged samples beyond the grid (float).
    """
    ncorr = wgt.shape[0]
    counts = np.zeros((ncorr, nx, ny), dtype=dtype if accum_dtype is None else accum_dtype)
    seen = np.zeros(nx * ny, dtype=np.uint8)
    cells, offgrid = _accumulate_counts(uvw, freq, mask, wgt, counts, seen, cell_size_x, cell_size_y, usign, vsign)
    del seen
    cells.sort()
    values = counts.reshape(ncorr, -1)[:, cells].astype(dtype, copy=False)
    del counts
    index = np.arange(ncorr, dtype=np.int64)[:, None] * (nx * ny) + cells[None, :]
    # a cell touched only by zero weights is empty, as flatnonzero would have it
    keep = values != 0
    data_vars = {
        "COUNTS_INDEX": (("counts_cell",), index[keep], {"shape": [ncorr, nx, ny]}),
        "COUNTS_VALUE": (("counts_cell",), values[keep]),
    }
    return data_vars, float(offgrid.sum())


@njit(**JIT_OPTIONS)
def briggs_denominator(counts, robust):
    """Turn an applied counts grid into the per-cell Briggs divisor, in place.

    The imaging weight of a sample in cell ``k`` is ``w_nat / D[k]`` wherever
    ``D[k] > 0``. For ``robust > -2`` that is ``D = 1 + ssq * counts``; at
    ``robust <= -2`` (uniform) it is the counts themselves. The single source
    of the divisor for both ``counts_to_weights`` and the pre-pass-2 imaging
    wsum (``piece_imaging_wsum``), so the two cannot drift.

    Args:
        counts: Applied ``(corr, u, v)`` counts grid; overwritten.
        robust: Briggs robustness.

    Returns:
        ``counts``, now holding ``D``.
    """
    real_type = counts.dtype
    ncorr, nx, ny = counts.shape
    if robust > -2:
        numsqrt = 5 * 10 ** (-robust)
        avgwnum = np.zeros(ncorr, dtype=real_type)
        avgwden = np.zeros(ncorr, dtype=real_type)
        for c in range(ncorr):
            for i in range(nx):
                for j in range(ny):
                    cval = counts[c, i, j]
                    avgwnum[c] += cval * cval
                    avgwden[c] += cval
        ssq = numsqrt * numsqrt * avgwden / avgwnum
        counts *= ssq[:, None, None]
        counts += 1
    return counts


@njit(**JIT_OPTIONS)
def counts_to_weights(counts, uvw, freq, weight, mask, nx, ny, cell_size_x, cell_size_y, robust, usign=1.0, vsign=-1.0):
    # an all-zero grid (nothing gridded) leaves the natural weights
    if not counts.any():
        return weight

    # ufreq
    u_cell = 1 / (nx * cell_size_x)
    umax = np.abs(1 / cell_size_x / 2)

    # vfreq
    v_cell = 1 / (ny * cell_size_y)
    vmax = np.abs(1 / cell_size_y / 2)

    ncorr, nrow, nchan = weight.shape

    counts = briggs_denominator(counts, robust)

    for r in prange(nrow):
        uvw_row = uvw[r]
        wgt_row = weight[:, r]
        mask_row = mask[r]
        for f in range(nchan):
            if mask_row[f] == 0:
                continue
            # current uv coords
            chan_normfreq = freq[f] / lightspeed
            u_tmp = uvw_row[0] * chan_normfreq * usign
            v_tmp = uvw_row[1] * chan_normfreq * vsign
            if v_tmp < 0:
                u_tmp = -u_tmp
                v_tmp = -v_tmp
            # pixel coordinates
            ug = (u_tmp + umax) / u_cell
            vg = (v_tmp + vmax) / v_cell

            # indices
            u_idx = np.int32(np.floor(ug))
            v_idx = np.int32(np.floor(vg))

            if (u_idx < 0) or (u_idx >= nx) or (v_idx < 0) or (v_idx >= ny):
                # out of bounds so continue.
                # raising an error means we can't grid at sub-Nyquist
                continue

            # counts can be zero if there are zero weights
            for c in range(ncorr):
                if counts[c, u_idx, v_idx] > 0:
                    wgt_row[c, f] /= counts[c, u_idx, v_idx]

    return weight


# @njit(nogil=True, cache=True)
def filter_extreme_counts(counts, level=10.0):
    """
    Replaces extremely small counts by median to prevent
    upweighting nearly empty cells
    """
    # get the median counts value
    if not level:
        return counts
    ic, ix, iy = np.where(counts > 0)
    cnts = counts[ic, ix, iy]
    med = np.median(cnts)
    lowval = med / level
    cnts = np.maximum(cnts, lowval)
    counts[ic, ix, iy] = cnts
    return counts


def box_sum_counts(counts, npix_super):
    """Box-sum the counts grid for super-uniform weighting.

    Replaces each cell of ``counts`` with the sum over a
    ``(2*npix_super+1)^2`` window centred on that cell, with zero-padding at
    image edges. Returns ``counts`` unchanged when ``npix_super`` is ``None``
    or non-positive, which recovers standard uniform weighting.

    Args:
        counts: Array of shape ``(ncorr, nx, ny)``.
        npix_super: Half-size of the box in pixels. ``0`` => standard uniform.

    Returns:
        Array of shape ``(ncorr, nx, ny)`` with box-summed counts.
    """
    if npix_super is None or npix_super <= 0:
        return counts

    assert np.issubdtype(counts.dtype, np.floating), (
        f"box_sum_counts requires a floating-point counts array; got dtype={counts.dtype}"
    )
    size = 2 * npix_super + 1
    out = np.empty_like(counts)
    for c in range(counts.shape[0]):
        out[c] = uniform_filter(counts[c], size=size, mode="constant", cval=0.0) * (size * size)
    return out


@register_jitable(inline="always")
def _corr2_to_full_vis(v):
    # unsampled cross-hand visibilities are zero (legacy stokes_funcs convention)
    return v[0], 0j, 0j, v[1]


@register_jitable(inline="always")
def _corr2_to_full_wgt(w):
    # unsampled cross-hand weights are unity (legacy stokes_funcs convention)
    return w[0], 1.0, 1.0, w[1]


@register_jitable(inline="always")
def _corr4_passthrough(x):
    return x[0], x[1], x[2], x[3]


@njit(nogil=True, cache=True, parallel=False)
def weight_data(data, weight, flag, jones, ant1, ant2, pol, product, nc, wgt_mode):
    """Stokes visibilities and weights from correlations, on the MSv4 (time, baseline) grid.

    Everything is indexed the way xarray-ms opens a measurement set, so there is
    no row axis and no row -> time mapping: gain time ``t`` and channel ``c`` are
    the data's own.

    Args:
        data: ``(time, baseline, chan, corr)`` complex visibilities.
        weight: ``(time, baseline, chan, corr)`` weights.
        flag: ``(time, baseline, chan, corr)`` bool; a sample flagged in any
            correlation is skipped.
        jones: gains in QuartiCal's native order,
            ``(time, chan, ant, dir, corr)`` with ``corr`` 1 (scalar) or 2
            (diagonal), or ``(time, chan, ant, dir, 2, 2)`` full Jones. The time
            and channel axes are the data's, or length one (broadcast). There
            is exactly one direction.
        ant1: ``(baseline,)`` antenna indices into the jones antenna axis.
        ant2: as ``ant1``.
        pol: ``"linear"`` or ``"circular"``.
        product: Stokes product, a subset of ``"IQUV"``.
        nc: number of correlations as a string, ``"2"`` or ``"4"``.
        wgt_mode: ``"l2"`` or ``"minvar"``.

    Returns:
        ``(vis, wgt)``, each ``(stokes, time, baseline, chan)`` so that
        ``(stokes, row, chan)`` is a free reshape for ducc. Flagged samples are zero.

    Raises:
        ValueError: if a gain time/channel axis is neither length one nor the
            data's length, the jones has other than one direction, or its
            correlation axes are not 1, 2 or 2x2.
    """
    # the kernel indexes gain time/channel as 0 or the data's own index: guard its bounds
    if jones.shape[0] not in (1, data.shape[0]) or jones.shape[1] not in (1, data.shape[2]):
        raise ValueError("jones (time, chan) axes must each be length 1 or match the data's (time, chan)")
    # also catches a (time, chan, ant, 2, 2) full Jones with no direction axis,
    # which would otherwise be read as a two-direction diagonal one
    if jones.shape[3] != 1:
        raise ValueError("jones must have exactly one direction: (time, chan, ant, 1, corr) or (..., 1, 2, 2)")
    if jones.ndim == 5 and jones.shape[4] not in (1, 2):
        raise ValueError("a diagonal jones must have 1 or 2 correlations")
    if jones.ndim == 6 and (jones.shape[4] != 2 or jones.shape[5] != 2):
        raise ValueError("a full jones must be 2x2 in its last two axes")
    return _weight_data_impl(
        data, weight, flag, jones, ant1, ant2, literally(pol), literally(product), literally(nc), literally(wgt_mode)
    )


def _weight_data_impl(data, weight, flag, jones, ant1, ant2, pol, product, nc, wgt_mode):
    raise NotImplementedError


@overload(_weight_data_impl, prefer_literal=True, jit_options={**JIT_OPTIONS, "parallel": True})
def nb_weight_data_impl(data, weight, flag, jones, ant1, ant2, pol, product, nc, wgt_mode):
    try:
        vis_fns, wgt_fns = stokes_expr_funcs(
            product.literal_value, pol.literal_value, nc.literal_value, wgt_mode.literal_value, jones.ndim
        )
    except Exception as e:
        raise numba.core.errors.TypingError(f"Failed in overload resolution: {e}") from e

    ns = len(vis_fns)
    ncorr = int(nc.literal_value)
    if ncorr == 2:
        vext, wext = _corr2_to_full_vis, _corr2_to_full_wgt
    else:
        vext = _corr4_passthrough
        wext = _corr4_passthrough
    vf0, wf0 = vis_fns[0], wgt_fns[0]
    vf1, wf1 = (vis_fns[1], wgt_fns[1]) if ns > 1 else (vis_fns[0], wgt_fns[0])
    vf2, wf2 = (vis_fns[2], wgt_fns[2]) if ns > 2 else (vis_fns[0], wgt_fns[0])
    vf3, wf3 = (vis_fns[3], wgt_fns[3]) if ns > 3 else (vis_fns[0], wgt_fns[0])

    if jones.ndim == 5:  # scalar or diagonal

        def _impl(data, weight, flag, jones, ant1, ant2, pol, product, nc, wgt_mode):
            nt, nbl, nchan, _ = data.shape
            ngt = jones.shape[0]
            ngc = jones.shape[1]
            # a scalar gain is the same in both hands
            c1 = jones.shape[4] - 1
            vis = np.zeros((ns, nt, nbl, nchan), dtype=data.dtype)
            wgt = np.zeros((ns, nt, nbl, nchan), dtype=data.real.dtype)
            for t in prange(nt):
                tg = np.intp(t) if ngt > 1 else np.intp(0)
                for b in range(nbl):
                    p = int(ant1[b])
                    q = int(ant2[b])
                    for c in range(nchan):
                        if flag[t, b, c].any():
                            continue
                        cg = c if ngc > 1 else 0
                        jp00 = jones[tg, cg, p, 0, 0]
                        jp11 = jones[tg, cg, p, 0, c1]
                        jq00 = jones[tg, cg, q, 0, 0]
                        jq11 = jones[tg, cg, q, 0, c1]
                        v00, v01, v10, v11 = vext(load_data(data, (t, b, c), ncorr, 3))
                        w00, w01, w10, w11 = wext(load_data(weight, (t, b, c), ncorr, 3))
                        vis[0, t, b, c] = vf0(v00, v01, v10, v11, jp00, jp11, jq00, jq11)
                        wgt[0, t, b, c] = wf0(w00, w01, w10, w11, jp00, jp11, jq00, jq11)
                        if ns > 1:
                            vis[1, t, b, c] = vf1(v00, v01, v10, v11, jp00, jp11, jq00, jq11)
                            wgt[1, t, b, c] = wf1(w00, w01, w10, w11, jp00, jp11, jq00, jq11)
                        if ns > 2:
                            vis[2, t, b, c] = vf2(v00, v01, v10, v11, jp00, jp11, jq00, jq11)
                            wgt[2, t, b, c] = wf2(w00, w01, w10, w11, jp00, jp11, jq00, jq11)
                        if ns > 3:
                            vis[3, t, b, c] = vf3(v00, v01, v10, v11, jp00, jp11, jq00, jq11)
                            wgt[3, t, b, c] = wf3(w00, w01, w10, w11, jp00, jp11, jq00, jq11)
            return vis, wgt

    else:  # full 2x2

        def _impl(data, weight, flag, jones, ant1, ant2, pol, product, nc, wgt_mode):
            nt, nbl, nchan, _ = data.shape
            ngt = jones.shape[0]
            ngc = jones.shape[1]
            vis = np.zeros((ns, nt, nbl, nchan), dtype=data.dtype)
            wgt = np.zeros((ns, nt, nbl, nchan), dtype=data.real.dtype)
            for t in prange(nt):
                tg = np.intp(t) if ngt > 1 else np.intp(0)
                for b in range(nbl):
                    p = int(ant1[b])
                    q = int(ant2[b])
                    for c in range(nchan):
                        if flag[t, b, c].any():
                            continue
                        cg = c if ngc > 1 else 0
                        jp00 = jones[tg, cg, p, 0, 0, 0]
                        jp01 = jones[tg, cg, p, 0, 0, 1]
                        jp10 = jones[tg, cg, p, 0, 1, 0]
                        jp11 = jones[tg, cg, p, 0, 1, 1]
                        jq00 = jones[tg, cg, q, 0, 0, 0]
                        jq01 = jones[tg, cg, q, 0, 0, 1]
                        jq10 = jones[tg, cg, q, 0, 1, 0]
                        jq11 = jones[tg, cg, q, 0, 1, 1]
                        v00, v01, v10, v11 = vext(load_data(data, (t, b, c), ncorr, 3))
                        w00, w01, w10, w11 = wext(load_data(weight, (t, b, c), ncorr, 3))
                        vis[0, t, b, c] = vf0(v00, v01, v10, v11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                        wgt[0, t, b, c] = wf0(w00, w01, w10, w11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                        if ns > 1:
                            vis[1, t, b, c] = vf1(v00, v01, v10, v11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                            wgt[1, t, b, c] = wf1(w00, w01, w10, w11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                        if ns > 2:
                            vis[2, t, b, c] = vf2(v00, v01, v10, v11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                            wgt[2, t, b, c] = wf2(w00, w01, w10, w11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                        if ns > 3:
                            vis[3, t, b, c] = vf3(v00, v01, v10, v11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
                            wgt[3, t, b, c] = wf3(w00, w01, w10, w11, jp00, jp01, jp10, jp11, jq00, jq01, jq10, jq11)
            return vis, wgt

    return _impl


def sparse_counts(counts):
    """A piece's uv counts grid as its occupied cells, for the scratch store.

    A piece covers a sliver of the padded uv plane: on a 4608^2 MeerKAT image
    the grid is 491 MB of float64 with 0.05% of its cells occupied. Stored
    densely, every piece cost a 491 MB encode in pass 1 and a 491 MB decode
    and add in the driver's reduction.

    Args:
        counts: ``(corr, u, v)`` counts grid.

    Returns:
        Data variables ``COUNTS_INDEX`` (flat indices of the occupied cells,
        carrying the grid's ``shape`` as an attribute) and ``COUNTS_VALUE``.
    """
    flat = counts.reshape(-1)
    idx = np.flatnonzero(flat)
    return {
        "COUNTS_INDEX": (("counts_cell",), idx, {"shape": list(counts.shape)}),
        "COUNTS_VALUE": (("counts_cell",), flat[idx]),
    }


def piece_imaging_wsum(index, value, denom, wsum_offgrid):
    """A piece's imaging-weight sum, from its sparse counts and the group's Briggs divisor.

    The imaging weight is ``w_nat / D[cell]`` (``briggs_denominator``), so the
    samples a piece put in a cell sum to ``N[cell] / D[cell]``, where ``N`` is
    the piece's own counts. Samples beyond the padded uv grid are in neither
    ``N`` nor the divisor and keep their natural weight in
    ``counts_to_weights``; the scatter records their sum directly
    (``wsum_offgrid``). Deriving it as ``wsum_nat - sum(N)`` instead cancelled
    catastrophically with float32 counts: 1.3e-3 relative error at 2e8
    samples on a 256^2 grid at robust -2, against 2e-9 recorded directly.

    Args:
        index: The piece's ``COUNTS_INDEX`` (flat cell indices).
        value: The piece's ``COUNTS_VALUE``.
        denom: The group's divisor ``D`` ``(corr, u, v)``, or ``None`` for an
            all-zero group grid (natural weights throughout).
        wsum_offgrid: The piece's corr-summed natural weight beyond the grid.

    Returns:
        The piece's corr-summed imaging-weight sum.
    """
    value = value.astype(np.float64)
    if denom is None:
        return float(value.sum()) + float(wsum_offgrid)
    d = denom.reshape(-1)[index].astype(np.float64)
    inside = float(np.sum(np.divide(value, d, out=value.copy(), where=d > 0)))
    return inside + float(wsum_offgrid)


def write_group_counts(store, sources, robust, filter_level=5.0, npix_super=0):
    """Reduce pass-1 piece COUNTS into one applied grid per weighting group, on disk.

    Streams: each piece's occupied cells (``sparse_counts``) are read and added
    into the group's dense accumulator, so the caller holds one grid whatever
    the number of pieces. The tree
    is opened with ``cache=False`` because xarray's default memoises every
    ``.values`` read on an open tree, which kept every piece's grid alive in the
    imager driver (#339). The applied grid (``filter_extreme_counts`` then
    ``box_sum_counts``) is written once per group, so pass-2 workers read it from
    the store instead of receiving a copy per task through the Ray object store.

    With the applied grid in hand, a second (sparse) read of the group's pieces
    gives each piece's imaging-weight sum (``piece_imaging_wsum``). That is the
    reduction that needs a full scan of the data, done here so the driver knows
    every image's effective frequency before pass 2 (wiki D28).

    Args:
        store: scratch store URL holding the pass-1 pieces.
        sources: weighting-group key -> scratch node paths of the group's pieces.
        robust: Briggs robustness the grids will be applied with.
        filter_level: ``filter_extreme_counts`` level.
        npix_super: ``box_sum_counts`` half-width (0 = standard uniform).

    Returns:
        ``(groups, wsum_img)``: weighting-group key -> node path
        (``counts/group####``) of its applied ``COUNTS`` grid in ``store``, and
        piece node path -> corr-summed imaging-weight sum.
    """
    dt = xr.open_datatree(store, engine="zarr", chunks=None, cache=False)
    out = {}
    wsum_img = {}
    try:
        for gi, key in enumerate(sorted(sources)):
            acc = None
            for path in sources[key]:
                ds = dt[path].ds
                index = ds.COUNTS_INDEX
                if acc is None:
                    acc = np.zeros(index.attrs["shape"], dtype=ds.COUNTS_VALUE.dtype)
                # cells are unique within a piece, so the fancy-indexed add is
                # exact and sums each cell in the same order as a dense add
                acc.reshape(-1)[index.values] += ds.COUNTS_VALUE.values
            acc = box_sum_counts(filter_extreme_counts(acc, level=filter_level), npix_super)
            group = f"counts/group{gi:04d}"
            xr.Dataset({"COUNTS": (("corr", "u", "v"), acc)}).to_zarr(store, group=group, mode="w", consolidated=False)
            out[key] = group
            # counts_to_weights leaves an all-zero grid's weights natural
            denom = briggs_denominator(acc, robust) if acc.any() else None
            del acc
            for path in sources[key]:
                ds = dt[path].ds
                wsum_img[path] = piece_imaging_wsum(
                    ds.COUNTS_INDEX.values, ds.COUNTS_VALUE.values, denom, ds.attrs["wsum_offgrid"]
                )
            del denom
    finally:
        dt.close()
    return out, wsum_img


def as_contiguous_readonly_view(a):
    """
    C-contiguous readonly view of ``a`` (copies only if non-contiguous).

    weight_data never mutates its inputs, and numba specialises on each
    array's layout and writeable flag: normalising here means one compiled
    signature per dtype configuration instead of one per readonly/layout
    permutation of the eight array arguments (issue #273).
    """
    a = np.ascontiguousarray(a)
    v = a.view()
    v.flags.writeable = False
    return v
