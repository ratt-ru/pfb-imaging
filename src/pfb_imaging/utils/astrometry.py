import numpy as np
from pyrap.measures import measures
from pyrap.quanta import quantity
from astropy.time import Time
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, EarthLocation, AltAz
from astropy.coordinates import get_body_barycentric, get_body, get_moon
from scipy.constants import c as lightspeed
from pfb_imaging.operators.gridder import wgridder_conventions
from africanus.coordinates import radec_to_lmn


# Based on
# https://github.com/tart-telescope/tart2ms/blob/master/tart2ms/fixvis.py
def synthesize_uvw(station_ECEF, time, ant1, ant2,
                   phase_ref,
                   stopctr_units=["rad", "rad"], stopctr_epoch="j2000",
                   time_TZ="UTC", time_unit="s",
                   posframe="ITRF", posunits=["m", "m", "m"]):
    """
    Synthesizes new UVW coordinates based on time according to
    NRAO CASA convention (same as in fixvis). 

    Note - this should work with missing rows as long as 
    
    inputs:
        station_ECEF: ITRF station coordinates read from MS::ANTENNA
        time: time column, preferably time centroid
        ant1: ANTENNA_1 index
        ant2: ANTENNA_2 index
        phase_ref: phase reference centre in radians
    
    returns uvw coordinates w.r.t. phase_ref
    """
    assert time.size == ant1.size
    assert ant1.size == ant2.size
    unique_time = np.unique(time)

    # ant to index mapping
    uants, iants = np.unique(np.concatenate((ant1, ant2)), return_index=True)
    # np.unique sorts the outputs, we need to unsort them
    # for the ant to index mapping to remain consistent
    isort = np.argsort(iants)
    uants = uants[isort]
    ants_dct = {uant: i for i, uant in enumerate(uants)}
    nrow = ant1.size
    uvw_new = np.zeros((nrow, 3))

    dm = measures()
    epoch = dm.epoch(time_TZ, quantity(unique_time[0], time_unit))
    refdir = dm.direction(stopctr_epoch,
                          quantity(phase_ref[0], stopctr_units[0]),
                          quantity(phase_ref[1], stopctr_units[1]))
    obs = dm.position(posframe,
                      quantity(station_ECEF[0, 0], posunits[0]),
                      quantity(station_ECEF[0, 1], posunits[1]),
                      quantity(station_ECEF[0, 2], posunits[2]))

    #setup local horizon coordinate frame with antenna 0 as reference position
    dm.do_frame(obs)
    dm.do_frame(refdir)
    dm.do_frame(epoch)
    for t in unique_time:
        epoch = dm.epoch(time_TZ, quantity(t, "s"))
        dm.do_frame(epoch)

        station_uv = np.zeros_like(station_ECEF)
        for iapos, apos in enumerate(station_ECEF):
            compuvw = dm.to_uvw(dm.baseline(posframe,
                                            quantity([apos[0], station_ECEF[0, 0]], posunits[0]),
                                            quantity([apos[1], station_ECEF[0, 1]], posunits[1]),
                                            quantity([apos[2], station_ECEF[0, 2]], posunits[2])))
            station_uv[iapos] = compuvw["xyz"].get_value()[0:3]
        
        rows = np.where(time == t)[0]

        for row in rows:
            a1 = ant1[row]
            a2 = ant2[row]
            bla1 = ants_dct[a1]
            bla2 = ants_dct[a2]
            uvw_new[row] = station_uv[bla1] - station_uv[bla2]

    return uvw_new


def rephase(vis, uvw, freq, radec_new, radec_ref, phasesign=-1):
    '''
    vis         - (nrow, nchan, ncorr) visibilities to rephase
    uvw         - (nrow, 3) visibility space coordinates
    freq        - (nchan) frequencies in Hz
    radec_new   - new phase center in radians
    radec_ref   - old phase center in radians
    phasesign   - opposite phase sign to im -> vis direction 
    '''
    ra = radec_new[0]
    dec = radec_new[1]
    ra0 = radec_ref[0]
    dec0 = radec_ref[1]
    dra = (ra - ra0)
    cos_dec = np.cos(dec)
    sin_dec = np.sin(dec)
    sin_dra = np.sin(dra)
    cos_dra = np.cos(dra)
    cos_dec0 = np.cos(dec0)
    sin_dec0 = np.sin(dec0)
    ll = cos_dec * sin_dra
    mm = sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_dra
    nn = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_dra - 1.0

    nrow, _, _ = vis.shape
    uvw_freq = np.zeros((nrow, freq.size, 3))
    uvw_freq[:, :, 0] = uvw[:, 0:1] * freq[None, :]/lightspeed
    uvw_freq[:, :, 1] = uvw[:, 1:2] * freq[None, :]/lightspeed
    uvw_freq[:, :, 2] = uvw[:, 2:] * freq[None, :]/lightspeed

    # rephasing conventions should match wgridder
    flip_u, flip_v, flip_w, _, _ = wgridder_conventions(0, 0)
    usign = -1 if flip_u else 1
    vsign = -1 if flip_v else 1
    wsign = -1 if flip_w else 1
    # ll *= usign
    # mm *= vsign
    x = np.exp(phasesign * 2.0j * np.pi * (uvw_freq[:, :, 0] * ll * usign +
                                           uvw_freq[:, :, 1] * mm * vsign -
                                           uvw_freq[:, :, 2] * nn * wsign))
    
    return vis[:, :, :] * x[:, :, None]


def format_coords(ra0,dec0):
    c = SkyCoord(ra0*units.deg,dec0*units.deg,frame='fk5')
    hms = str(c.ra.to_string(units.hour))
    dms = str(c.dec)
    return hms,dms

# Pillaged from
# https://github.com/ratt-ru/solarkat/blob/main/solarkat-pipeline/find_sun_stimela.py
# obs_lat and obs_lon hardcoded for MeerKAT
def get_coordinates(obs_time,
                    obs_lat=-30.71323598930457,
                    obs_lon=21.443001467965008,
                    target='Sun'):
    '''
    Give location of object given telescope location (defaults to MeerKAT)
    and time of observation.

    Inputs:

    obs_time - should be weighted mean of TIME from MS
    obs_lat  - telescope lattitude in degrees
    obs_lon  - telescope longitude in degrees
    '''
    loc = EarthLocation.from_geodetic(obs_lat,obs_lon) #,obs_height,ellipsoid)
    t = Time(obs_time/86400.0,format='mjd')  # factor converts Jsecs to Jdays 24 * 60**2
    with solar_system_ephemeris.set('builtin'):
        sun = get_body(target, t, loc)
        sun_ra = sun.ra.value
        sun_dec = sun.dec.value
    # sun_hms=format_coords(sun_ra,sun_dec)
    # print(sun_hms[0],sun_hms[1])
    return np.deg2rad(sun_ra), np.deg2rad(sun_dec)


def create_cross_product_matrix(k):
    """
    Create the cross-product matrix (skew-symmetric matrix) for vector k.
    
    For a vector k = [kx, ky, kz], the cross-product matrix K is:
    K * v = k × v for any vector v
    
    Parameters:
    -----------
    k : array (3,)
        Vector to create cross-product matrix from
    
    Returns:
    --------
    K : array (3, 3)
        Cross-product matrix
    """
    return np.array([[0, -k[2], k[1]],
                     [k[2], 0, -k[0]],
                     [-k[1], k[0], 0]])


def create_rotation_matrix_rodrigues(s0, s1):
    """
    Create rotation matrix using Rodrigues' formula to transform UVW coordinates 
    from old to new phase center.
    
    Uses the hybrid approach: Rodrigues' formula to compute the matrix once,
    then apply matrix multiplication for efficiency with multiple baselines.
    
    Parameters:
    -----------
    ra0, dec0 : float
        Original phase center (radians)
    ra1, dec1 : float
        New phase center (radians)
    
    Returns:
    --------
    R : ndarray (3, 3)
        Rotation matrix
    """
    # Calculate rotation axis (perpendicular to both directions)
    k = np.cross(s0, s1)
    k_norm = np.linalg.norm(k)
    
    # Handle special case: phase centers are identical or opposite
    if k_norm < 1e-10:
        if np.dot(s0, s1) > 0:
            # Same direction - identity matrix
            return np.eye(3)
        else:
            # Opposite directions - 180 degree rotation
            # Need to find any perpendicular axis
            # Use the axis most perpendicular to s0
            if abs(s0[0]) < 0.9:
                k = np.cross(s0, np.array([1, 0, 0]))
            else:
                k = np.cross(s0, np.array([0, 1, 0]))
            k = k / np.linalg.norm(k)
            # 180 degree rotation matrix using Rodrigues
            K = create_cross_product_matrix(k)
            return np.eye(3) + 2.0 * K @ K
    
    # Normalize rotation axis
    k = k / k_norm
    
    # Calculate rotation angle
    cos_theta = np.dot(s0, s1)
    # Clamp to avoid numerical issues with arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = k_norm  # |s0 × s1| = sin(θ) for unit vectors
    
    # Build rotation matrix using Rodrigues' formula:
    # R = I + sin(θ) * K + (1 - cos(θ)) * K²
    # where K is the cross-product matrix of k
    
    K = create_cross_product_matrix(k)
    K2 = K @ K
    
    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * K2
    
    return R

def change_phase_dir(vis, uvw, freq, radec_new, radec_ref, phasesign=-1):
    # Direction cosines for new phase center
    l, m, n = radec_to_lmn(radec_new[None, :], radec_ref)[0]
    s1 = np.array([l, m, n])

    # Direction cosines for original phase center
    s0 = np.array([0, 0, 1])

    # Difference in direction cosines
    dl = s1[0]
    dm = -s1[1]
    dn = s1[2] - 1
    
    # Phase shift
    nrow, nchan, _ = vis.shape
    phase = np.zeros((nrow, nchan), dtype=uvw.dtype)
    flip_u, flip_v, flip_w = wgridder_conventions(0,0)[0:3]
    usign = -1 if flip_u else 1
    phase += usign * dl * uvw[:, 0:1] * freq[None, :]/lightspeed
    vsign = -1 if flip_v else 1
    phase += vsign * dm * uvw[:, 1:2] * freq[None, :]/lightspeed
    wsign = -1 if flip_w else 1
    phase -= wsign * dn * uvw[:, 2:3] * freq[None, :]/lightspeed
    vis *= np.exp(phasesign*2j*np.pi*phase)[:, :, None]

    # # get rotation matrix
    # R = create_rotation_matrix_rodrigues(s0, s1)

    # # rotate uvw's (einsum is slower)
    # uvw_new = (R @ uvw.T).T

    return vis, uvw
    

def uvw_rotate(uvw, ra0, dec0, ra, dec):
    """
        Compute the following 3x3 coordinate transformation matrix:
        Z_rot(facet_new_rotation) * \\
        T(new_phase_centre_ra,new_phase_centre_dec) * \\
        transpose(T(old_phase_centre_ra,
                    old_phase_centre_dec)) * \\
        transpose(Z_rot(facet_old_rotation))
        where:
                            |      cRA             -sRA            0       |
        T (RA,D) =          |      -sDsRA          -sDcRA          cD      |
                            |      cDsRA           cDcRA           sD      |
        This is the similar to the one in Thompson, A. R.; Moran, J. M.;
        and Swenson, G. W., Jr. Interferometry and Synthesis
        in Radio Astronomy, New York: Wiley, ch. 4, but in a
        lefthanded system.
        We're not transforming between a coordinate system with w pointing
        towards the pole and one with w pointing towards the reference
        centre here, so the last rotation matrix is ignored!
        This transformation will let the image be tangent to the celestial
        sphere at the new delay centre
    """
    d_ra = ra - ra0
    c_d_ra = np.cos(d_ra)
    s_d_ra = np.sin(d_ra)
    c_new_dec = np.cos(dec)
    c_old_dec = np.cos(dec0)
    s_new_dec = np.sin(dec)
    s_old_dec = np.sin(dec0)
    mat_11 = c_d_ra
    mat_12 = s_old_dec * s_d_ra
    mat_13 = -c_old_dec * s_d_ra
    mat_21 = -s_new_dec * s_d_ra
    mat_22 = s_new_dec * s_old_dec * c_d_ra + c_new_dec * c_old_dec
    mat_23 = -c_old_dec * s_new_dec * c_d_ra + c_new_dec * s_old_dec
    mat_31 = c_new_dec * s_d_ra
    mat_32 = -c_new_dec * s_old_dec * c_d_ra + s_new_dec * c_old_dec
    mat_33 = c_new_dec * c_old_dec * c_d_ra + s_new_dec * s_old_dec
    uvw[0] = mat_11 * uvw[0] + mat_12 * uvw[1] + mat_13 * uvw[3]
    uvw[1] = mat_21 * uvw[0] + mat_22 * uvw[1] + mat_23 * uvw[3]
    uvw[2] = mat_31 * uvw[0] + mat_32 * uvw[1] + mat_33 * uvw[3]
    return uvw