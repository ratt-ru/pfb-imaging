import numpy as np
from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord, get_body, solar_system_ephemeris
from astropy.time import Time
from pyrap.measures import measures
from pyrap.quanta import quantity


# Based on
# https://github.com/tart-telescope/tart2ms/blob/master/tart2ms/fixvis.py
def synthesize_uvw(
    station_ecef,
    time,
    ant1,
    ant2,
    phase_ref,
    stopctr_units=["rad", "rad"],
    stopctr_epoch="j2000",
    time_tz="UTC",
    time_unit="s",
    posframe="ITRF",
    posunits=["m", "m", "m"],
):
    """
    Synthesizes new UVW coordinates based on time according to
    NRAO CASA convention (same as in fixvis).

    Note - this should work with missing rows as long as

    inputs:
        station_ecef: ITRF station coordinates read from MS::ANTENNA
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
    epoch = dm.epoch(time_tz, quantity(unique_time[0], time_unit))
    refdir = dm.direction(
        stopctr_epoch, quantity(phase_ref[0], stopctr_units[0]), quantity(phase_ref[1], stopctr_units[1])
    )
    obs = dm.position(
        posframe,
        quantity(station_ecef[0, 0], posunits[0]),
        quantity(station_ecef[0, 1], posunits[1]),
        quantity(station_ecef[0, 2], posunits[2]),
    )

    # setup local horizon coordinate frame with antenna 0 as reference position
    dm.do_frame(obs)
    dm.do_frame(refdir)
    dm.do_frame(epoch)
    for t in unique_time:
        epoch = dm.epoch(time_tz, quantity(t, "s"))
        dm.do_frame(epoch)

        station_uv = np.zeros_like(station_ecef)
        for iapos, apos in enumerate(station_ecef):
            compuvw = dm.to_uvw(
                dm.baseline(
                    posframe,
                    quantity([apos[0], station_ecef[0, 0]], posunits[0]),
                    quantity([apos[1], station_ecef[0, 1]], posunits[1]),
                    quantity([apos[2], station_ecef[0, 2]], posunits[2]),
                )
            )
            station_uv[iapos] = compuvw["xyz"].get_value()[0:3]

        rows = np.where(time == t)[0]

        for row in rows:
            a1 = ant1[row]
            a2 = ant2[row]
            bla1 = ants_dct[a1]
            bla2 = ants_dct[a2]
            uvw_new[row] = station_uv[bla1] - station_uv[bla2]

    return uvw_new


# Pillaged from
# https://github.com/ratt-ru/solarkat/blob/main/solarkat-pipeline/find_sun_stimela.py
# obs_lat and obs_lon hardcoded for MeerKAT
def resolve_target_radec(target, obs_time):
    """Resolve a ``--target`` string to the image-centre (ra, dec) in radians.

    Two forms are accepted, and they differ in a way that matters to any caller
    building a shared coordinate axis:

    * ``"HH:MM:SS,DD:MM:SS"`` — a fixed position, identical for every image.
    * a bare object name (e.g. ``"Sun"``) — a solar-system body, resolved at
      ``obs_time``, so it **moves** from one output image to the next.

    Args:
        target: the target string.
        obs_time: weighted mean MS TIME of the image, in MJD seconds. Used only
            for the ephemeris form.

    Returns:
        ``(ra, dec)`` in radians, and ``True`` if the position is
        time-dependent (the ephemeris form).
    """
    parts = target.split(",")
    if len(parts) == 1:
        ra, dec = get_coordinates(obs_time, target=target)
        return ra, dec, True
    coord = SkyCoord(parts[0], parts[1], frame="fk5", unit=(units.hourangle, units.deg))
    # deg -> rad here and rad -> deg on the way out, so every consumer of this
    # reference goes through the same arithmetic (wiki design-decisions D36)
    return np.deg2rad(coord.ra.value), np.deg2rad(coord.dec.value), False


def get_coordinates(obs_time, obs_lat=-30.71323598930457, obs_lon=21.443001467965008, target="Sun"):
    """
    Give location of object given telescope location (defaults to MeerKAT)
    and time of observation.

    Inputs:

    obs_time - should be weighted mean of TIME from MS
    obs_lat  - telescope lattitude in degrees
    obs_lon  - telescope longitude in degrees
    """
    loc = EarthLocation.from_geodetic(obs_lat, obs_lon)  # ,obs_height,ellipsoid)
    t = Time(obs_time / 86400.0, format="mjd")  # factor converts Jsecs to Jdays 24 * 60**2
    with solar_system_ephemeris.set("builtin"):
        sun = get_body(target, t, loc)
        sun_ra = sun.ra.value
        sun_dec = sun.dec.value
    return np.deg2rad(sun_ra), np.deg2rad(sun_dec)
