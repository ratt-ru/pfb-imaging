import numpy as np
from astropy.time import Time

from pfb_imaging.utils.misc import to_unix_time


def test_casacore_vs_astropy(ms_name, ms_meta):
    """
    Test casacore.quanta.quantity against astropy.time.Time unix time conversion
    """
    from casacore.quanta import quantity

    xds = ms_meta.xds
    time = xds.TIME.values
    utime = np.unique(time)

    for t in utime:
        # casacore conversion
        ctime = quantity(f"{t}s").to_unix_time()
        # astropy conversion
        atime = Time(t / 86400.0, format="mjd", scale="utc").unix
        assert np.abs(ctime - atime) < 1e-6
        # manual version
        mtime = to_unix_time(t)
        assert np.abs(ctime - mtime) < 1e-6
