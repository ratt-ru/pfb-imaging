import numpy as np
from pfb.utils.naming import xds_from_list, xds_from_url
from daskms.fsspec_store import DaskMSStore
from omegaconf import OmegaConf
from pfb.utils.dist import band_actor
from pfb.opt.primal_dual import get_ratio
import time

if __name__=='__main__':
    # xds_name = '/scratch/bester/stage7_combined_bda_I.xds'
    xds_name = '/home/landman/testing/pfb/out/data_I.xds'
    xds = xds_from_url(xds_name)
    xds_store = DaskMSStore(xds_name)
    xds_list = xds_store.fs.glob(f'{xds_store.url}/*')


    ds_list = []
    uv_max = 0.0
    max_freq = 0.0
    for ds_name, ds in zip(xds_list, xds):
        idx = ds_name.find('band') + 4
        bid = ds_name[idx:idx+4]
        uv_max = np.maximum(uv_max, ds.uv_max)
        max_freq = np.maximum(max_freq, ds.max_freq)
        # import ipdb; ipdb.set_trace()
        if bid == '0001':
            ds_list.append(ds_name)

    from pfb.parser.schemas import schema
    init_args = {}
    for key in schema.spotless["inputs"].keys():
        init_args[key.replace("-", "_")] = schema.spotless["inputs"][key]["default"]
    opts = OmegaConf.create(init_args)
    opts['nthreads'] = 1
    opts['field_of_view'] = 2.0

    # import ipdb; ipdb.set_trace()
    actor = band_actor(ds_list,
                       opts,
                       1,
                       '/home/landman/testing/pfb/out/',
                       uv_max,
                       max_freq)

    nx, ny, nymax, nxmax, cell_rad, ra, dec, x0, y0, freq_out, time_out = actor.get_image_info()

    print(f"Image size set to ({nx}, {ny})")

    bases = tuple(opts.bases.split(','))
    nbasis = len(bases)

    model = np.zeros((1, nx, ny))

    residual, wsum = actor.set_image_data_products(model[0],0,from_cache=False)
    residual /= wsum
    rms = np.std(residual)

    hess_norm = 100.0
    l1weight = np.ones((nbasis, nymax, nxmax))
    ratio = np.zeros(l1weight.shape, dtype=l1weight.dtype)

    actor.set_wsum(wsum)

    gamma=1
    nu = nbasis
    lam = rms
    sigma = hess_norm / (2.0 * gamma) / nu

    update = actor.cg_update()

    vtilde, _ = actor.init_pd_params(hess_norm, nbasis, gamma=gamma)
    vtilde = vtilde[None]
    get_ratio(vtilde, l1weight, sigma, rms, ratio)

    ti = time.time()
    vtilde, eps_num, eps_den, bandid = actor.pd_update(ratio)
    print(time.time() - ti)
