import numpy as np
import ray

from pfb_imaging.utils.misc import convolve2gaussres
from pfb_imaging.utils.naming import xds_from_list


@ray.remote
def rrestore_image(*args, **kwargs):
    return restore_image(*args, **kwargs)


def restore_image(
    ds_name,
    model_name,
    residual_name,
    gaussparf,  # final desired resolution
    nthreads=1,
):
    """
    Writes restored image to dds.
    Final resolution is specified by gaussparf.
    The intrinsic resolution is stored in PSFPARSN in the dds.
    The dataset will contain a new variable PSFPARSF with the final resolution in pixel units,
    and the restored image will be in a variable called IMAGE.
    Only the restored image's resolution will be updated.

    Args:
        ds_name: str or list of str
            Name(s) of dataset(s) to read from and write to.
        model_name: str
            Name of variable in dataset containing model image.
        residual_name: str
            Name of variable in dataset containing residual image.
        gaussparf: (ncorr, 3) array of floats
            Desired final resolution (emaj, emin, pa) in units (pixels, pixels, radians).
        nthreads: int
            Number of threads to use for convolution.

    Returns:
        gaussparf: (ncorr, 3) array of floats
            Final resolution in pixel units.
        bandid: int
            Band ID of dataset.
        timeid: int
            Time ID of dataset.
    """
    # convolve residual
    if not isinstance(ds_name, list):
        ds_name = [ds_name]
    drop_all_but = [model_name, residual_name, "PSFPARSN", "WSUM"]
    dds = xds_from_list(ds_name, nthreads=nthreads, drop_all_but=drop_all_but)
    if len(dds) > 1:
        raise RuntimeError("Some thing went wrong. Done per band so should return a single dataset")
    ds = dds[0]
    if not isinstance(gaussparf, np.ndarray):
        gaussparf = np.array(gaussparf)
        if len(gaussparf.shape) == 1:
            gaussparf = np.tile(gaussparf, (ds.corr.size, 1))
        assert gaussparf.shape == (ds.corr.size, 3), "gaussparf should have shape (ncorr, 3)"
    if model_name not in ds:
        raise ValueError(f"Could not find {model_name} in dds")
    if residual_name not in ds:
        raise ValueError(f"Could not find {residual_name} in dds")
    wsum = ds.WSUM.values
    model = ds.get(model_name).values
    _, nx, ny = model.shape
    l_coord = -(nx // 2) + np.arange(nx)
    m_coord = -(ny // 2) + np.arange(ny)
    xx, yy = np.meshgrid(l_coord, m_coord, indexing="ij")
    residual = ds.get(residual_name).values / wsum[:, None, None]

    # units are (pixel, pixel, radians)
    gausspari = ds.PSFPARSN.values

    if np.allclose(gaussparf, gausspari):  # passed in in pixel units
        # no convolution required
        rconv = residual
    else:
        # convolve with ratio of Gaussians
        rconv = convolve2gaussres(
            residual, xx, yy, gaussparf, nthreads=nthreads, gausspari=gausspari, pfrac=0.2, norm_kernel=False
        )
    # convolve with gaussparf to get final resolution
    # TODO - this assumes the model's intrinsic resolution is zero, which is not really true
    mconv = convolve2gaussres(model, xx, yy, gaussparf, nthreads=nthreads, pfrac=0.2, norm_kernel=False)

    mconv += rconv

    ds = ds.assign({"IMAGE": (("corr", "x", "y"), mconv), "PSFPARSF": (("corr", "bpar"), gaussparf)})
    # only write updates
    ds = ds[["IMAGE", "PSFPARSF"]]
    ds.to_zarr(ds_name[0], mode="a")
    return gaussparf, ds.bandid, ds.timeid
