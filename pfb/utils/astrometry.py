import numpy as np
from pyrap.measures import measures
from pyrap.quanta import quantity
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import solar_system_ephemeris, EarthLocation, AltAz
from astropy.coordinates import get_body_barycentric, get_body, get_moon
from scipy.constants import c as lightspeed


# Based on
# https://github.com/tart-telescope/tart2ms/blob/master/tart2ms/fixvis.py
def synthesize_uvw(station_ECEF, time, ant1, ant2,
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
        ant1: ANTENNA_1 index
        ant2: ANTENNA_2 index
        phase_ref: phase reference centre in radians
    
    returns uvw coordinates w.r.t. phase_ref
    """
    assert time.size == ant1.size
    assert ant1.size == ant2.size
    # assume sorted in time
    tdiff = time[1:] - time[0:-1]
    if (tdiff<0).any():
        raise NotImplementedError("Times must be sorted for UVW computation")
    unique_time = np.unique(time)

    # ant to index mapping (this assumes antennas are sorted)
    uants = np.unique(np.concatenate((ant1, ant2)))
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
        epoch = dm.epoch("UT1", quantity(t, "s"))
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