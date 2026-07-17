import numpy as np
from africanus.rime import parallactic_angles
from africanus.rime.fast_beam_cubes import beam_cube_dde
from astropy.wcs import WCS
from katbeam import JimBeam
from reproject import reproject_interp
from scipy.interpolate import RegularGridInterpolator

from pfb_imaging.utils.stokes import jones_to_mueller, mueller_to_stokes


def interp_beam(freq, nx, ny, cell_deg, btype, utime=None, ant_pos=None, phase_dir=None):
    """
    A function that returns an object array containing a function
    returning beam values given (l,m) coordinates at a single frequency.
    Frequency mapped to imaging band extenally. Result is meant to be
    passed into eval_beam below.
    """
    if isinstance(freq, np.ndarray):
        assert freq.size == 1, "Only single frequency interpolation currently supported"
        freq = freq[0]
    if btype is None:
        l_coord = (-(nx // 2) + np.arange(nx)) * cell_deg
        m_coord = (-(ny // 2) + np.arange(ny)) * cell_deg
        return np.ones((nx, ny), dtype=float), l_coord, m_coord
    elif btype.endswith(".npz"):
        # these are expected to be in the format given here
        # https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/index.html
        dct = np.load(btype)
        beam = dct["abeam"]
        l_coord = np.deg2rad(dct["ldeg"])
        m_coord = np.deg2rad(dct["mdeg"])
        ll, mm = np.meshgrid(l_coord, m_coord, indexing="ij")
        bfreqs = dct["freq"]
        beam_amp = (beam[0, :, :, :] * beam[0, :, :, :].conj() + beam[-1, :, :, :] * beam[-1, :, :, :].conj()) / 2.0
        beam_amp = np.transpose(beam_amp, (1, 2, 0))[:, :, :, None, None].real
    else:
        btype = btype.lower()
        btype = btype.replace("-", "_")
        l_coord = (-(nx // 2) + np.arange(nx)) * cell_deg
        m_coord = (-(ny // 2) + np.arange(ny)) * cell_deg
        ll, mm = np.meshgrid(l_coord, m_coord, indexing="ij")
        if btype in ["kbl", "kb_l", "katbeam_l"]:
            # katbeam L band
            beam_amp = JimBeam("MKAT-AA-L-JIM-2020").I(ll.flatten(), mm.flatten(), freqMHz=freq / 1e6)
        elif btype in ["kbuhf", "kb_uhf", "katbeam_uhf"]:
            # katbeam L band
            beam_amp = JimBeam("MKAT-AA-UHF-JIM-2020").I(ll.flatten(), mm.flatten(), freqMHz=freq / 1e6)
        else:
            raise ValueError(f"Unknown beam model {btype}")
        beam_amp = beam_amp.reshape(nx, ny)[:, :, None, None, None]
        bfreqs = np.array((freq,))

    if utime is None:
        return beam_amp.squeeze(), l_coord, m_coord

    parangles = parallactic_angles(utime, ant_pos, phase_dir, backend="astropy")
    # mean over antanna nant -> 1
    parangles = np.mean(parangles, axis=1, keepdims=True)
    nant = 1
    # beam_cube_dde requirements
    nband = 1
    ntimes = utime.size
    ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
    point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)
    beam_extents = np.array([[l_coord.min(), l_coord.max()], [m_coord.min(), m_coord.max()]])
    lm = np.vstack((ll.flatten(), mm.flatten())).T
    beam_image = beam_cube_dde(
        np.ascontiguousarray(beam_amp), beam_extents, bfreqs, lm, parangles, point_errs, ant_scale, np.array((freq,))
    ).squeeze()

    return beam_image.squeeze(), l_coord, m_coord


def eval_beam(beam_image, l_in, m_in, l_out, m_out):
    if l_out.ndim == 2:
        ll = l_out
        mm = m_out
    elif l_out.ndim == 1:
        ll, mm = np.meshgrid(l_out, m_out, indexing="ij")
    else:
        msg = "Only 1 or 2D coordinates supported for beam evaluation"
        raise ValueError(msg)

    if (beam_image == 1.0).all():
        return np.ones_like(ll)
    else:  # this gets expensive
        beamo = RegularGridInterpolator((l_in, m_in), beam_image, bounds_error=False, method="linear", fill_value=1.0)
        return beamo((ll, mm))


def reproject_and_interp_beam(
    beam, time, antpos, radec0, radecf, cell_deg_in, cell_deg_out, nxo, nyo, poltype, product, weight=None, nthreads=1
):
    """
    beam    - (2, 2, nx, ny)
    time    - nrow
    antpos  - (nant, 3)
    radec0  - original pointing direction
    radecf  - direction to project to
    """
    # parangles = parallactic_angles(utime, antpos, np.array(radec0))
    # # use the mean over antenna
    # parangles = np.mean(parangles, axis=-1, keepdims=False)
    _, _, nxi, nyi = beam.shape
    # beamo = np.zeros((ntime, 2, 2, nxi, nyi), dtype=beam.dtype)
    # for i, parang in enumerate(parangles):
    #     # spatial rotation (assuming position angle is paralactic angle i.e. linear North-South,
    #     East-West interferometer)
    #     beamo[i] = rotate(beam, parang, axes=(-2, -1), reshape=False, order=1, mode='nearest')
    #     # feed rotation
    #     beamo[i] = rotate(beamo[i], parang, axes=(0, 1), reshape=False, order=1, mode='nearest')

    # # compute the weighted sum over time
    # if weight is not None and ntime > 1:
    #     wsumt = np.zeros((ntime, 2, 2))
    #     for i, t in enumerate(utime):
    #         sel = time==t
    #         wsumt[i] = weight[sel].sum(axis=(0, 1)).reshape(2, 2)
    #     beamo = np.sum(wsumt[:, :, :, None, None] * beamo, axis=0)
    #     beamo /= np.sum(wsumt, axis=0)[:, :, None, None]
    # else:
    #     beamo = np.mean(beamo, axis=0)

    # jones to Mueller
    beamo = beam
    beamo = jones_to_mueller(beamo, beamo)

    # Mueller to Stokes
    beamo = mueller_to_stokes(beamo, poltype=poltype)

    # select required products
    i = ()
    if "I" in product:
        i += (0,)
    if "Q" in product:
        i += (1,)
    if "U" in product:
        i += (2,)
    if "V" in product:
        i += (3,)
    beamo = beamo[i, ...]

    # reproject onto target field
    wcs_ref = WCS(naxis=2)
    wcs_ref.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs_ref.wcs.cdelt = np.array((cell_deg_in, cell_deg_in))
    wcs_ref.wcs.cunit = ["deg", "deg"]
    wcs_ref.wcs.crval = np.array((radec0[0] * 180.0 / np.pi, radec0[1] * 180.0 / np.pi))
    wcs_ref.wcs.crpix = [1 + nxi // 2, 1 + nyi // 2]
    wcs_ref.array_shape = [nxi, nyi]

    # header for target field
    wcs_target = WCS(naxis=2)
    wcs_target.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs_target.wcs.cdelt = np.array((cell_deg_out, cell_deg_out))
    wcs_target.wcs.cunit = ["deg", "deg"]
    wcs_target.wcs.crval = np.array((radecf[0] * 180.0 / np.pi, radecf[1] * 180.0 / np.pi))
    wcs_target.wcs.crpix = [nxo // 2, nyo // 2]
    wcs_target.array_shape = [nxo, nyo]

    pbeam = np.zeros((len(product), nxo, nyo), dtype=beamo.dtype)
    pmask = np.zeros((len(product), nxo, nyo), dtype=beamo.dtype)
    for i in range(len(product)):
        pbeam[i], pmask[i] = reproject_interp(
            (beamo[i], wcs_ref), wcs_target, shape_out=(nxo, nyo)
        )  # , block_size='auto', parallel=nthreads

    # set beam to zero where it is not defined
    pmask = pmask.astype(bool)
    pbeam[~pmask] = 0.0
    return pbeam


def reproject_and_interp_scat_beam(
    beam,
    l_beam,
    m_beam,
    radec0,
    radecf,
    cell_deg_out,
    nxo,
    nyo,
    product,
    l0=0.0,
    m0=0.0,
):
    """Reproject rotation-averaged beam maps onto the output image grid.

    Axis conventions (measured; see docs/wiki/image-and-beam-orientation.md):
    reproject binds a numpy array to a celestial WCS as (row, col) =
    (Dec-like, RA-like), so both the input maps and the returned maps are
    (Y, X)-ordered. For RA---SIN the intermediate coordinate
    (pixel - crpix) * cdelt is l itself, so signed cdelt taken from the actual
    l/m coordinates describes either grid direction; the target WCS is the hci
    output header verbatim (CDELT1 = -cell, CRPIX = 1 + n//2).

    Args:
        beam: (nstokes, m_beam.size, l_beam.size) beam maps in (Y, X) order,
            as returned by BeamWizard.get_rotation_averaged_beam, centred on
            the pointing direction radec0.
        l_beam: 1D l coordinates of the beam columns (deg, East positive).
        m_beam: 1D m coordinates of the beam rows (deg, North positive).
        radec0: original pointing direction (rad).
        radecf: output image phase centre to project to (rad).
        cell_deg_out: output cell size (deg).
        nxo: number of output X pixels.
        nyo: number of output Y pixels.
        product: Stokes product string.
        l0/m0: image-centre offset from the tangent point in radians
            (--target). Same facet convention as ``utils.fits.set_wcs``:
            CRVAL stays the tangent point and CRPIX shifts so the centre
            pixel lands on the target direction.

    Returns:
        (nstokes, nyo, nxo) beam maps in cube (Y, X) order, zeroed where the
        input maps have no coverage.
    """
    nstokes = beam.shape[0]
    assert nstokes == len(product), "Number of Stokes products in beam does not match length of product string"
    assert beam.shape[1:] == (m_beam.size, l_beam.size), "Beam maps must be (Y, X)-ordered on the l/m grid"

    dl = l_beam[1] - l_beam[0]
    dm = m_beam[1] - m_beam[0]

    # WCS describing the beam grid around the pointing direction
    wcs_ref = WCS(naxis=2)
    wcs_ref.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs_ref.wcs.cdelt = np.array((dl, dm))
    wcs_ref.wcs.cunit = ["deg", "deg"]
    wcs_ref.wcs.crval = np.array((np.rad2deg(radec0[0]), np.rad2deg(radec0[1])))
    # 1-based crpix of the zero-offset pixel
    wcs_ref.wcs.crpix = [1.0 - l_beam[0] / dl, 1.0 - m_beam[0] / dm]

    # target WCS = the hci output header (core/hci.py scaffold construction)
    wcs_target = WCS(naxis=2)
    wcs_target.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs_target.wcs.cdelt = np.array((-cell_deg_out, cell_deg_out))
    wcs_target.wcs.cunit = ["deg", "deg"]
    wcs_target.wcs.crval = np.array((np.rad2deg(radecf[0]), np.rad2deg(radecf[1])))
    # facet convention (see utils.fits.set_wcs): RA axis has cdelt = -cell, so
    # centring the image on the target shifts crpix by +l0/cell; Dec by -m0/cell
    wcs_target.wcs.crpix = [
        1 + nxo // 2 + np.rad2deg(l0) / cell_deg_out,
        1 + nyo // 2 - np.rad2deg(m0) / cell_deg_out,
    ]

    pbeam = np.zeros((nstokes, nyo, nxo), dtype=beam.dtype)
    for i in range(nstokes):
        pbeam[i], pmask = reproject_interp((beam[i], wcs_ref), wcs_target, shape_out=(nyo, nxo))
        # set beam to zero where it is not defined
        pbeam[i][~pmask.astype(bool)] = 0.0
    return pbeam
