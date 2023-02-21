'''
Shameless pillaged from
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

def synthesize_uvw(station_ECEF, time, a1, a2,
                   phase_ref,
                   stopctr_units=["rad", "rad"], stopctr_epoch="j2000",
                   time_TZ="UTC", time_unit="s",
                   posframe="ITRF", posunits=["m", "m", "m"]):
    """
    Synthesizes new UVW coordinates based on time according to
    NRAO CASA convention (same as in fixvis)
    station_ECEF: ITRF station coordinates read from MS::ANTENNA
    time: time column, preferably time centroid
    a1: ANTENNA_1 index
    a2: ANTENNA_2 index
    phase_ref: phase reference centre in radians
    returns dictionary of dense uvw coordinates and indices:
        {
         "UVW": shape (nbl * ntime, 3),
         "TIME_CENTROID": shape (nbl * ntime,),
         "ANTENNA_1": shape (nbl * ntime,),
         "ANTENNA_2": shape (nbl * ntime,)
        }
    Note: input and output antenna indexes may not have the same
          order or be flipped in 1 to 2 index
    Note: This operation CANNOT be applied blockwise due
          to a casacore.measures threadsafety issue
    """
    assert time.size == a1.size
    assert a1.size == a2.size

    ants = np.concatenate((a1, a2))
    unique_ants = np.arange(np.max(ants) + 1)
    unique_time = np.unique(time)
    na = unique_ants.size
    nbl = na * (na - 1) // 2 + na
    ntime = unique_time.size
    # keep a full uvw array for all antennae - including those
    # dropped by previous calibration and CASA splitting
    padded_uvw = np.zeros((ntime * nbl, 3), dtype=np.float64)
    antindices = np.stack(np.triu_indices(na, 0),
                          axis=1)
    padded_time = unique_time.repeat(nbl)
    padded_a1 = np.tile(antindices[:, 0], (1, ntime)).ravel()
    padded_a2 = np.tile(antindices[:, 1], (1, ntime)).ravel()

    dm = measures()
    epoch = dm.epoch(time_TZ, quantity(time[0], time_unit))
    refdir = dm.direction(stopctr_epoch,
                          quantity(phase_ref[0, 0], stopctr_units[0]),
                          quantity(phase_ref[0, 1], stopctr_units[1]))
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
        for bl in range(nbl):
            blants = antindices[bl]
            bla1 = blants[0]
            bla2 = blants[1]
            # same as in CASA convention (Convention for UVW calculations in CASA, Rau 2013)
            padded_uvw[ti*nbl + bl, :] = station_uv[bla1] - station_uv[bla2]

    # # hack to remove auto-correlations
    # if a1.size != padded_a1.size:
    #     I = np.where(padded_a1 != padded_a2)
    #     padded_uvw = padded_uvw[I]
    #     padded_time = padded_time[I]
    #     padded_a1 = padded_a1[I]
    #     padded_a2 = padded_a2[I]

    return dict(zip(["UVW", "TIME_CENTROID", "ANTENNA1", "ANTENNA2"],
                    [padded_uvw, padded_time, padded_a1, padded_a2]))


# Pirated from
# https://github.com/ratt-ru/solarkat/blob/main/solarkat-pipeline/find_sun_stimela.py
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
    t = Time(obs_time/86400.0,format='mjd')  # where is this factor from?
    with solar_system_ephemeris.set('builtin'):
        sun = get_body(target, t, loc)
        sun_ra = sun.ra.value
        sun_dec = sun.dec.value
    sun_hms=format_coords(sun_ra,sun_dec)
    # print(sun_hms[0],sun_hms[1])
    # import pdb; pdb.set_trace()
    return np.deg2rad(sun_ra), np.deg2rad(sun_dec)
