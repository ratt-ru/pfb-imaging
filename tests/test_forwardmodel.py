
# import pytest
# from pathlib import Path
# from xarray import Dataset
# from collections import namedtuple
# import dask
# dask.config.set(**{'array.slicing.split_large_chunks': False})
# from multiprocessing.pool import ThreadPool
# dask.config.set(pool=ThreadPool(64))
# import dask.array as da
# from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
# pmp = pytest.mark.parametrize

# # @pmp('beam_model', (None, 'kbl',))
# # @pmp('do_gains', (False, True))
# def test_forwardmodel(gain_name, ms_name):
#     '''
#     This test is based on the observation that imaging all of corrected
#     data at natural weighting should be the same applying the gains per
#     band and time on the fly and computing the sum afterwards.
#     '''
#     import numpy as np
#     np.random.seed(420)
#     from numpy.testing import assert_allclose
#     from daskms import xds_from_storage_ms as xds_from_ms
#     from daskms import xds_from_storage_table as xds_from_table
#     from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
#     from africanus.constants import c as lightspeed
#     from ducc0.fft import good_size
#     from ducc0.wgridder.experimental import vis2dirty
#     from africanus.calibration.utils.dask import corrupt_vis, correct_vis
#     from africanus.calibration.utils import chunkify_rows
#     import xarray as xr

#     test_dir = Path(ms_name).resolve().parent
#     xds = xds_from_ms(ms_name)
#     xds = xr.concat(xds, 'row')
#     time = xds.TIME.values
#     utime = np.unique(time)
#     ntime = utime.size
#     utpc = 50
#     row_chunks, tbin_idx, tbin_counts = chunkify_rows(time, utpc)
#     tbin_idx = da.from_array(tbin_idx, chunks=utpc)
#     tbin_counts = da.from_array(tbin_counts, chunks=utpc)

#     spw = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0]
#     freq = spw.CHAN_FREQ.values.squeeze()
#     nchan = freq.size
#     cchunk = 256
#     chan_chunks = (cchunk,)*int(np.ceil(nchan/256))


#     xds = xds.chunk({'row': row_chunks, 'chan': chan_chunks})
#     xds = xds[{'corr': slice(0, 4, 3)}]
#     nant = np.maximum(xds.ANTENNA1.values.max(), xds.ANTENNA2.values.max()) + 1

#     uvw = xds.UVW.values
#     nrow = uvw.shape[0]
#     u_max = abs(uvw[:, 0]).max()
#     v_max = abs(uvw[:, 1]).max()
#     uv_max = np.maximum(u_max, v_max)

#     # image size
#     cell_N = 1.0 / (2 * uv_max * freq.max() / lightspeed)

#     srf = 2.0
#     cell_rad = cell_N / srf
#     cell_deg = cell_rad * 180 / np.pi
#     cell_size = cell_deg * 3600
#     print("Cell size set to %5.5e arcseconds" % cell_size)


#     # the test will fail in intrinsic if sources fall near beam sidelobes
#     fov = 1.0
#     npix = good_size(int(fov / cell_deg))
#     while npix % 2:
#         npix += 1
#         npix = good_size(npix)

#     nx = npix
#     ny = npix
#     nband = 8
#     print("Image size set to (%i, %i, %i)" % (nband, nx, ny))

#     # first make full image
#     gds = xds_from_zarr(f"{gain_name.rstrip('/')}::GK-net",
#                         chunks={'gain_time': utpc, 'gain_freq': cchunk})
#     gds = xr.concat(gds, 'gain_time')
#     print('Loading data')
#     jones = da.swapaxes(gds.gains.data, 1, 2)
#     data = xds.DATA.data.astype(np.complex128)
#     wgt = xds.WEIGHT_SPECTRUM.data.astype(np.complex128)
#     flag = xds.FLAG.data
#     ant1 = xds.ANTENNA1.data
#     ant2 = xds.ANTENNA2.data

#     ncorr = 2
#     # corrupt twice for cwgt
#     wgt = wgt.reshape(nrow, nchan, 1, ncorr)

#     wgt = corrupt_vis(tbin_idx,
#                       tbin_counts,
#                       ant1,
#                       ant2,
#                       jones*jones.conj(),
#                       wgt).real

#     # correct data
#     data = correct_vis(tbin_idx,
#                        tbin_counts,
#                        ant1,
#                        ant2,
#                        jones,
#                        data,
#                        flag).reshape(nrow, nchan, ncorr)

#     # take weighted sum of correlations to get Stokes I vis
#     # data = da.map_blocks(lambda x, y: x[:, :, 0] * y[:, :, 0] + x[:, :, -1] * y[:, :, -1],
#     #                      data, wgt, chunks=data.chunks, dtype=data.dtype)
#     # wgt = wgt.map_blocks(lambda x: x[:, :, 0] + x[:, :, -1], chunks=wgt.chunks, dtype=wgt.dtype)
#     data = wgt[:, :, 0] * data[:, :, 0] + wgt[:, :, -1] * data[:, :, -1]
#     wgt = wgt[:, :, 0] + wgt[:, :, -1]


#     print("Correcting vis data products")
#     data, wgt = dask.compute(data, wgt)  #, optimize_graph=True)

#     msk = wgt > 0
#     data[msk] = data[msk]/wgt[msk]
#     flag = flag.reshape(nrow, nchan, ncorr)
#     mask = (~(flag[:, :, 0] | flag[:, :, -1])).astype(np.uint8).compute()

#     wsum0 = wgt[mask.astype(bool)].sum()

#     # image Stokes I data
#     print("Making dirty image")
#     dirty0 = vis2dirty(uvw=xds.UVW.values,
#                        freq=freq,
#                        vis=data,
#                        wgt=wgt,
#                        mask=mask,
#                        npix_x=nx, npix_y=ny,
#                        pixsize_x=cell_rad, pixsize_y=cell_rad,
#                        epsilon=1e-7,
#                        do_wgridding=True,
#                        divide_by_n=False,
#                        nthreads=64)

#     print("Making psf")
#     psf0 = vis2dirty(uvw=xds.UVW.values,
#                     freq=freq,
#                     vis=wgt.astype(np.complex128),
#                     mask=mask,
#                     npix_x=nx, npix_y=ny,
#                     pixsize_x=cell_rad, pixsize_y=cell_rad,
#                     epsilon=1e-7,
#                     do_wgridding=True,
#                     divide_by_n=False,
#                     nthreads=64)

#     # sanity check
#     assert np.allclose(psf0.max()/wsum0, 1.0, rtol=1e-7, atol=1e-7)

#     # next compute via pfb workers
#     suffix = "main"
#     outname = str(test_dir / 'test')
#     # set defaults from schema
#     from pfb.parser.schemas import schema
#     init_args = {}
#     for key in schema.init["inputs"].keys():
#         init_args[key] = schema.init["inputs"][key]["default"]
#     # overwrite defaults
#     init_args["ms"] = str(ms_name)
#     init_args["output_filename"] = outname
#     init_args["data_column"] = "DATA"
#     init_args["flag_column"] = 'FLAG'
#     init_args["weight_column"] = 'WEIGHT_SPECTRUM'
#     init_args["gain_table"] = gain_name
#     init_args['gain_term'] = 'GK-net'
#     init_args["max_field_of_view"] = fov
#     init_args["overwrite"] = True
#     init_args["integrations_per_image"] = 50
#     init_args["channels_per_image"] = 128
#     init_args["nvthreads"] = 8
#     from pfb.workers.init import _init
#     xds = _init(**init_args)

#     # grid data to produce dirty image
#     grid_args = {}
#     for key in schema.grid["inputs"].keys():
#         grid_args[key] = schema.grid["inputs"][key]["default"]
#     # overwrite defaults
#     grid_args["output_filename"] = outname
#     grid_args["suffix"] = suffix
#     grid_args["nband"] = 8
#     grid_args["field_of_view"] = fov
#     grid_args["super_resolution_factor"] = srf
#     grid_args["fits_mfs"] = True
#     grid_args["psf"] = True
#     grid_args["psf_oversize"] = 1.0
#     grid_args["residual"] = False
#     grid_args["nthreads"] = 8  # has to be set when calling _grid
#     grid_args["nvthreads"] = 64
#     grid_args["overwrite"] = True
#     grid_args["robustness"] = None
#     grid_args["do_wgridding"] = True
#     from pfb.workers.grid import _grid
#     dds = _grid(xdsi=xds, **grid_args)

#     dds = dask.compute(dds)[0]
#     nx_psf, ny_psf = dds[0].x_psf.size, dds[0].y_psf.size
#     dirty = np.zeros((nx, ny), dtype=np.float64)
#     psf = np.zeros((nx_psf, ny_psf), dtype=np.float64)
#     wsum = 0.0
#     for ds in dds:
#         dirty += ds.DIRTY.values
#         psf += ds.PSF.values
#         wsum += ds.WSUM.values

#     try:
#         assert np.allclose(1+np.abs(dirty0)/wsum, 1+np.abs(dirty)/wsum)
#         assert np.allclose(1+np.abs(psf0)/wsum, 1+np.abs(psf)/wsum)
#     except:
#         import ipdb; ipdb.set_trace()


# ms_name = '/scratch/bester/ms2_target_scan2.zarr'
# gain_name = '/home/bester/projects/ESO137/output/ms2_gains_scan2.qc'
# test_forwardmodel(gain_name, ms_name)
