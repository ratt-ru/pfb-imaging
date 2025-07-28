'''
Shamelessly pillaged from
https://github.com/tart-telescope/tart2ms/blob/master/tart2ms/fixvis.py
'''

from pyrap.tables import table as tbl
import numpy as np
from pyrap.measures import measures
from pyrap.quanta import quantity
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, EarthLocation, AltAz
from astropy.coordinates import get_body_barycentric, get_body, get_moon
from scipy.constants import c as lightspeed

def synthesize_uvw(station_ECEF, time, a1, a2,
                   phase_ref,
                   stopctr_units=["rad", "rad"], stopctr_epoch="j2000",
                   time_TZ="UTC", time_unit="s",
                   posframe="ITRF", posunits=["m", "m", "m"]):
    """
    Synthesizes new UVW coordinates based on time according to
    NRAO CASA convention (same as in fixvis)
    
    inputs:
        station_ECEF: ITRF station coordinates read from MS::ANTENNA
        time: time column, preferably time centroid
        a1: ANTENNA_1 index
        a2: ANTENNA_2 index
        phase_ref: phase reference centre in radians
    
    returns uvw coordinates w.r.t. phase_ref
    """
    assert time.size == a1.size
    assert a1.size == a2.size
    # assume sorted in time
    tdiff = time[1:] - time[0:-1]
    if (tdiff<0).any():
        raise NotImplementedError("Times must be sorted for UVW computation")
    
    na = na = np.maximum(a1.max(), a2.max()) + 1
    nbl = na * (na - 1) // 2 + na
    unique_time = np.unique(time)
    ntime = unique_time.size
    # keep a full uvw array for all antennae - including those
    # dropped by previous calibration and CASA splitting
    nrow = a1.size
    uvw_new = np.zeros((nrow, 3))

    # padded_uvw = np.zeros((ntime * nbl, 3), dtype=np.float64)
    # antindices = np.stack(np.triu_indices(na, 0),
    #                       axis=1)
    # padded_time = unique_time.repeat(nbl)
    # padded_a1 = np.tile(antindices[:, 0], (1, ntime)).ravel()
    # padded_a2 = np.tile(antindices[:, 1], (1, ntime)).ravel()

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
    for ti, t in enumerate(unique_time):
        epoch = dm.epoch("UT1", quantity(t, "s"))
        dm.do_frame(epoch)

        station_uv = np.zeros_like(station_ECEF)
        for iapos, apos in enumerate(station_ECEF):
            compuvw = dm.to_uvw(dm.baseline(posframe,
                                            quantity([apos[0], station_ECEF[0, 0]], posunits[0]),
                                            quantity([apos[1], station_ECEF[0, 1]], posunits[1]),
                                            quantity([apos[2], station_ECEF[0, 2]], posunits[2])))
            station_uv[iapos] = compuvw["xyz"].get_value()[0:3]
        
        idx = np.where(time == t)[0]
        ant1 = a1[idx]
        ant2 = a2[idx]

        for row in idx:
            bla1 = a1[row]
            bla2 = a2[row]
            uvw_new[row] = station_uv[bla1] - station_uv[bla2]

        # for bl in range(nbl):
        #     blants = antindices[bl]
        #     bla1 = blants[0]
        #     bla2 = blants[1]
        #     # same as in CASA convention (Convention for UVW calculations in CASA, Rau 2013)
        #     padded_uvw[ti*nbl + bl, :] = station_uv[bla1] - station_uv[bla2]

    return uvw_new


# Pirated from
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
    def format_coords(ra0,dec0):
        c = SkyCoord(ra0*u.deg,dec0*u.deg,frame='fk5')
        hms = str(c.ra.to_string(u.hour))
        dms = str(c.dec)
        return hms,dms


    loc = EarthLocation.from_geodetic(obs_lat,obs_lon) #,obs_height,ellipsoid)
    t = Time(obs_time/86400.0,format='mjd')  # factor converts Jsecs to Jdays 24 * 60**2
    with solar_system_ephemeris.set('builtin'):
        sun = get_body(target, t, loc)
        sun_ra = sun.ra.value
        sun_dec = sun.dec.value
    sun_hms=format_coords(sun_ra,sun_dec)
    # print(sun_hms[0],sun_hms[1])
    return np.deg2rad(sun_ra), np.deg2rad(sun_dec)


def rephase(vis, uvw, freq, radec_new, radec_ref, phasesign=-1):
    ra = radec_new[0]
    dec = radec_new[1]
    ra0 = radec_ref[0]
    dec0 = radec_ref[1]
    d_ra = (ra - ra0)
    d_dec = dec
    d_decp = dec0
    c_d_dec = np.cos(d_dec)
    s_d_dec = np.sin(d_dec)
    s_d_ra = np.sin(d_ra)
    c_d_ra = np.cos(d_ra)
    c_d_decp = np.cos(d_decp)
    s_d_decp = np.sin(d_decp)
    ll = c_d_dec * s_d_ra
    mm = (s_d_dec * c_d_decp - c_d_dec * s_d_decp * c_d_ra)
    nn = s_d_dec * s_d_decp + c_d_dec * c_d_decp * c_d_ra - 1.0

    nrow, _, _ = vis.shape
    uvw_freq = np.zeros((nrow, freq.size, 3))
    uvw_freq[:, :, 0] = uvw[:, 0:1] * freq[None, :]/lightspeed
    uvw_freq[:, :, 1] = uvw[:, 1:2] * freq[None, :]/lightspeed
    uvw_freq[:, :, 2] = uvw[:, 2:] * freq[None, :]/lightspeed

    x = np.exp(phasesign * 2.0j * np.pi * (uvw_freq[:, :, 0] * ll +
                                            uvw_freq[:, :, 1] * mm +
                                            uvw_freq[:, :, 2] * nn))
    
    return vis[:, :, :] * x[:, :, None]

def dense2sparse_uvw(a1, a2, time, padded_uvw):
    """
    Copy a dense uvw matrix onto a sparse uvw matrix
        a1: sparse antenna 1 index
        a2: sparse antenna 2 index
        time: sparse time
        ddid: sparse data discriptor index
        padded_uvw: a dense ddid-less uvw matrix
                    returned by synthesize_uvw of shape
                    (ntime * nbl, 3), fastest varying
                    by baseline, including auto correlations
    """
    assert time.size == a1.size
    assert a1.size == a2.size
    na = np.maximum(a1.max(), a2.max()) + 1
    nbl = na * (na - 1) // 2 + na
    unique_time = np.unique(time)
    new_uvw = np.zeros((a1.size, 3), dtype=padded_uvw.dtype)
    outbl = baseline_index(a1, a2, na) - 1
    for outrow in range(a1.size):
        lookupt = np.argwhere(unique_time == time[outrow])[0][0]
        # print(padded_uvw.shape, lookupt * nbl, outbl[outrow], outrow)
        # from time import sleep
        # sleep(1)
        # note: uvw same for all ddid (in m)
        new_uvw[outrow][:] = padded_uvw[lookupt * nbl + outbl[outrow], :]

    return new_uvw

def baseline_index(a1, a2, no_antennae):
    """
    Computes unique index of a baseline given antenna 1 and antenna 2
    (zero indexed) as input. The arrays may or may not contain
    auto-correlations.

    There is a quadratic series expression relating a1 and a2
    to a unique baseline index(can be found by the double difference
    method)

    Let slow_varying_index be S = min(a1, a2). The goal is to find
    the number of fast varying terms. As the slow
    varying terms increase these get fewer and fewer, because
    we only consider unique baselines and not the conjugate
    baselines)
    B = (-S ^ 2 + 2 * S *  # Ant + S) / 2 + diff between the
    slowest and fastest varying antenna

    :param a1: array of ANTENNA_1 ids
    :param a2: array of ANTENNA_2 ids
    :param no_antennae: number of antennae in the array
    :return: array of baseline ids

    Note: na must be strictly greater than max of 0-indexed
          ANTENNA_1 and ANTENNA_2
    """
    if a1.shape != a2.shape:
        raise ValueError("a1 and a2 must have the same shape!")

    slow_index = np.min(np.array([a1, a2]), axis=0)

    return (slow_index * (-slow_index + (2 * no_antennae + 1))) // 2 + \
        np.abs(a1 - a2)