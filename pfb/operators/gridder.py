
import numpy as np
import dask
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset
import zarr
from africanus.gridding.wgridder.dask import dirty as vis2im
from africanus.gridding.wgridder.dask import model as im2vis
from africanus.gridding.wgridder.dask import residual as im2residim
from africanus.model.coherency.dask import convert
from pprint import pprint

class Gridder(object):
    def __init__(self, ms_name, nx, ny, cell_size, nband=None, nthreads=8, do_wstacking=1, Stokes='I',
                 row_chunks=100000, optimise_chunks=True, epsilon=1e-5,
                 data_column='CORRECTED_DATA', weight_column='WEIGHT_SPECTRUM', model_column="MODEL_DATA", flag_column='FLAG'):
        if Stokes != 'I':
            raise NotImplementedError("Only Stokes I currently supported")
        self.nx = nx
        self.ny = ny
        self.cell = cell_size * np.pi/60/60/180
        self.nthreads = nthreads
        self.do_wstacking = do_wstacking
        self.epsilon = epsilon

        self.data_column = data_column
        self.weight_column = weight_column
        self.model_column = model_column
        self.flag_column = flag_column
        if isinstance(ms_name, list):
            self.ms = ms_name
        else:
            self.ms = [ms_name]

        # first pass through data to determine freq_mapping
        self.radec = None
        self.freq = {}
        all_freqs = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks={"row":-1},
                              columns=('TIME'))

            # subtables
            ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
            fields = xds_from_table(ims + "::FIELD", group_cols="__row__")
            spws = xds_from_table(ims + "::SPECTRAL_WINDOW", group_cols="__row__")
            pols = xds_from_table(ims + "::POLARIZATION", group_cols="__row__")

            # subtable data
            ddids = dask.compute(ddids)[0]
            fields = dask.compute(fields)[0]
            spws = dask.compute(spws)[0]
            pols = dask.compute(pols)[0]
            

            self.freq[ims] = {}
            for ds in xds:
                field = fields[ds.FIELD_ID]
                radec = field.PHASE_DIR.data.squeeze()

                # check fields match
                if self.radec is None:
                    self.radec = radec

                if not np.array_equal(radec, self.radec):
                    continue

                
                spw = spws[ds.DATA_DESC_ID]
                tmp_freq = spw.CHAN_FREQ.data.squeeze()
                self.freq[ims][ds.DATA_DESC_ID] = tmp_freq
                all_freqs.append(list(tmp_freq))


        # freq mapping
        all_freqs = dask.compute(all_freqs)
        ufreqs = np.unique(all_freqs)  # returns ascending sorted
        self.nchan = ufreqs.size
        if nband is None:
            self.nband = self.nchan
        else:
            self.nband = nband
        
        # bin edges
        fmin = ufreqs[0]
        fmax = ufreqs[-1]
        fbins = np.linspace(fmin, fmax, self.nband+1)
        self.freq_out = np.zeros(self.nband)
        for band in range(self.nband):
            indl = ufreqs >= fbins[band]
            indu = ufreqs < fbins[band + 1] + 1e-6
            self.freq_out[band] = np.mean(ufreqs[indl & indu])
        
        # chan <-> band mapping
        self.band_mapping = {}
        self.chunks = {}
        self.freq_bin_idx = {}
        self.freq_bin_counts = {}
        for ims in self.freq:
            self.freq_bin_idx[ims] = {}
            self.freq_bin_counts[ims] = {}
            self.band_mapping[ims] = {}
            self.chunks[ims] = []
            for spw in self.freq[ims]:
                freq = dask.compute(self.freq[ims][spw])[0]
                band_map = np.zeros(freq.size, dtype=np.int32)
                for band in range(self.nband):
                    indl = freq >= fbins[band]
                    indu = freq < fbins[band + 1] + 1e-6
                    band_map = np.where(indl & indu, band, band_map)
                # to dask arrays
                bands, bin_counts = np.unique(band_map, return_counts=True)
                self.band_mapping[ims][spw] = tuple(bands)
                self.chunks[ims].append({'row':(-1,), 'chan':tuple(bin_counts)})
                self.freq[ims][spw] = da.from_array(freq, chunks=tuple(bin_counts))
                bin_idx = np.append(np.array([0]), np.cumsum(bin_counts))[0:-1]
                self.freq_bin_idx[ims][spw] = da.from_array(bin_idx, chunks=1)
                self.freq_bin_counts[ims][spw] = da.from_array(bin_counts, chunks=1)

        # optimise chunking by concatenating measurement sets and pre-computing corr to Stokes
        # if optimise_chunks:
        #     for 
        

    def make_residual(self, x):
        print("Making residual")
        x = da.from_array(x.astype(np.float32), chunks=(1, self.nx, self.ny))
        residuals = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks=self.chunks[ims],
                              columns=(self.data_column, self.weight_column, self.flag_column, 'UVW'))

            # subtables
            ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
            fields = xds_from_table(ims + "::FIELD", group_cols="__row__")
            spws = xds_from_table(ims + "::SPECTRAL_WINDOW", group_cols="__row__")
            pols = xds_from_table(ims + "::POLARIZATION", group_cols="__row__")

            # subtable data
            ddids = dask.compute(ddids)[0]
            fields = dask.compute(fields)[0]
            spws = dask.compute(spws)[0]
            pols = dask.compute(pols)[0]

            for ds in xds:
                field = fields[ds.FIELD_ID]
                radec = field.PHASE_DIR.data.squeeze()
                if not np.array_equal(radec, self.radec):
                    continue

                spw = ds.DATA_DESC_ID  # this is not correct, need to use spw

                # print("Processing ms %s, field %i, spw %i"%(ims, ds.FIELD_ID, ds.DATA_DESC_ID))
                freq_bin_idx = self.freq_bin_idx[ims][spw]
                freq_bin_counts = self.freq_bin_counts[ims][spw]
                freq = self.freq[ims][spw]

                uvw = ds.UVW.data

                data = getattr(ds, self.data_column).data
                dataxx = data[:, :, 0]
                datayy = data[:, :, -1]

                weights = getattr(ds, self.weight_column).data
                if len(weights.shape) < 3:
                    weights = da.broadcast_to(weights[:, None, :], data.shape, chunks=data.chunks)
                weightsxx = weights[:, :, 0]
                weightsyy = weights[:, :, -1]

                # weighted sum corr to Stokes I
                weights = weightsxx + weightsyy
                data = (weightsxx * dataxx + weightsyy * datayy)
                data = da.where(weights, data/weights, 0.0j)
                
                # only keep data where both corrs are unflagged
                flag = getattr(ds, self.flag_column).data
                flagxx = flag[:, :, 0]
                flagyy = flag[:, :, -1]
                flag = ~ (flagxx | flagyy)  # ducc0 convention

                bands = self.band_mapping[ims][spw]
                model = x[list(bands), :, :]
                residual = im2residim(uvw, freq, model, data, freq_bin_idx, freq_bin_counts,
                                      self.cell, weights=weights, flag=flag.astype(np.uint8),
                                      nthreads=self.nthreads, epsilon=self.epsilon, do_wstacking=self.do_wstacking)

                residuals.append(residual)
        
        
        residuals = dask.compute(residuals, scheduler='single-threaded')[0]
        
        return accumulate_dirty(residuals, self.nband, self.band_mapping).astype(np.float64)

    def make_dirty(self):
        print("Making dirty")
        dirty = da.zeros((self.nband, self.nx, self.ny), 
                         dtype=np.float32,
                         chunks=(1, self.nx, self.ny))
        dirties = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks=self.chunks[ims],
                              columns=(self.data_column, self.weight_column, self.flag_column, 'UVW'))

            # subtables
            ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
            fields = xds_from_table(ims + "::FIELD", group_cols="__row__")
            spws = xds_from_table(ims + "::SPECTRAL_WINDOW", group_cols="__row__")
            pols = xds_from_table(ims + "::POLARIZATION", group_cols="__row__")

            # subtable data
            ddids = dask.compute(ddids)[0]
            fields = dask.compute(fields)[0]
            spws = dask.compute(spws)[0]
            pols = dask.compute(pols)[0]

            # pprint(self.chunks[ims])

            for ds in xds:
                field = fields[ds.FIELD_ID]
                radec = field.PHASE_DIR.data.squeeze()
                if not np.array_equal(radec, self.radec):
                    continue

                spw = ds.DATA_DESC_ID  # this is not correct, need to use spw
                
                # print("Processing ms %s, field %i, spw %i"%(ims, ds.FIELD_ID, ds.DATA_DESC_ID))
                freq_bin_idx = self.freq_bin_idx[ims][spw]
                freq_bin_counts = self.freq_bin_counts[ims][spw]
                freq = self.freq[ims][spw]
                freq_chunk = freq_bin_counts[0].compute()

                uvw = ds.UVW.data

                data = getattr(ds, self.data_column).data
                dataxx = data[:, :, 0]
                datayy = data[:, :, -1]

                weights = getattr(ds, self.weight_column).data
                if len(weights.shape) < 3:
                    weights = da.broadcast_to(weights[:, None, :], data.shape, chunks=data.chunks)
                weightsxx = weights[:, :, 0]
                weightsyy = weights[:, :, -1]

                # weighted sum corr to Stokes I
                weights = weightsxx + weightsyy
                data = (weightsxx * dataxx + weightsyy * datayy)
                data = da.where(weights, data/weights, 0.0j)
                
                # only keep data where both corrs are unflagged
                flag = getattr(ds, self.flag_column).data
                flagxx = flag[:, :, 0]
                flagyy = flag[:, :, -1]
                flag = ~ (flagxx | flagyy)  # ducc0 convention

                dirty = vis2im(uvw, freq, data, freq_bin_idx, freq_bin_counts,
                               self.nx, self.ny, self.cell, weights=weights,
                               flag=flag.astype(np.uint8), nthreads=self.nthreads,
                               epsilon=self.epsilon, do_wstacking=self.do_wstacking)

                
                dirties.append(dirty)
        
        dirties = dask.compute(dirties, scheduler='single-threaded')[0]
        
        return accumulate_dirty(dirties, self.nband, self.band_mapping).astype(np.float64)

    def make_psf(self):
        print("Making PSF")
        psfs = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks=self.chunks[ims],
                              columns=(self.data_column, self.weight_column, self.flag_column, 'UVW'))

            # subtables
            ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
            fields = xds_from_table(ims + "::FIELD", group_cols="__row__")
            spws = xds_from_table(ims + "::SPECTRAL_WINDOW", group_cols="__row__")
            pols = xds_from_table(ims + "::POLARIZATION", group_cols="__row__")

            # subtable data
            ddids = dask.compute(ddids)[0]
            fields = dask.compute(fields)[0]
            spws = dask.compute(spws)[0]
            pols = dask.compute(pols)[0]

            for ds in xds:
                field = fields[ds.FIELD_ID]
                radec = field.PHASE_DIR.data.squeeze()
                if not np.array_equal(radec, self.radec):
                    continue

                spw = ds.DATA_DESC_ID  # this is not correct, need to use spw
                
                # print("Processing ms %s, field %i, spw %i"%(ims, ds.FIELD_ID, ds.DATA_DESC_ID))
                freq_bin_idx = self.freq_bin_idx[ims][spw]
                freq_bin_counts = self.freq_bin_counts[ims][spw]
                freq = self.freq[ims][spw]

                uvw = ds.UVW.data

                flag = getattr(ds, self.flag_column).data

                weights = getattr(ds, self.weight_column).data
                if len(weights.shape) < 3:
                    weights = da.broadcast_to(weights[:, None, :], flag.shape, chunks=flag.chunks)
                weightsxx = weights[:, :, 0]
                weightsyy = weights[:, :, -1]

                # weighted sum corr to Stokes I
                weights = weightsxx + weightsyy
                data = weights.astype(np.complex64)
                
                # only keep data where both corrs are unflagged
                flagxx = flag[:, :, 0]
                flagyy = flag[:, :, -1]
                flag = ~ (flagxx | flagyy)  # ducc0 convention


                psf = vis2im(uvw, freq, data, freq_bin_idx, freq_bin_counts,
                             2*self.nx, 2*self.ny, self.cell, flag=flag.astype(np.uint8),
                             nthreads=self.nthreads, epsilon=self.epsilon, do_wstacking=self.do_wstacking)

                psfs.append(psf)

        psfs = dask.compute(psfs)[0]
                
        return accumulate_dirty(psfs, self.nband, self.band_mapping).astype(np.float64)

    def write_model(self, x):
        print("Writing model data")
        x = da.from_array(x.astype(np.float32), chunks=(1, self.nx, self.ny))
        writes  = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks=self.chunks[ims],
                              columns=(self.model_column, 'UVW'))

            # subtables
            ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
            fields = xds_from_table(ims + "::FIELD", group_cols="__row__")
            spws = xds_from_table(ims + "::SPECTRAL_WINDOW", group_cols="__row__")
            pols = xds_from_table(ims + "::POLARIZATION", group_cols="__row__")

            # subtable data
            ddids = dask.compute(ddids)[0]
            fields = dask.compute(fields)[0]
            spws = dask.compute(spws)[0]
            pols = dask.compute(pols)[0]

            out_data = []
            for ds in xds:
                field = fields[ds.FIELD_ID]
                radec = field.PHASE_DIR.data.squeeze()
                if not np.array_equal(radec, self.radec):
                    continue

                # print("Processing ms %s, field %i, spw %i"%(ims, ds.FIELD_ID, ds.DATA_DESC_ID))
                spw = ds.DATA_DESC_ID  # this is not correct, need to use spw

                freq_bin_idx = self.freq_bin_idx[ims][spw]
                freq_bin_counts = self.freq_bin_counts[ims][spw]
                freq = self.freq[ims][spw]

                uvw = ds.UVW.data

                model_vis = getattr(ds, self.model_column).data

                bands = self.band_mapping[ims][spw]
                model = x[list(bands), :, :]
                vis = im2vis(uvw, freq, model, freq_bin_idx, freq_bin_counts,
                             self.cell, nthreads=self.nthreads, epsilon=self.epsilon,
                             do_wstacking=self.do_wstacking)

                model_vis = populate_model(vis, model_vis)
                
                out_ds = ds.assign(**{self.model_column: (("row", "chan", "corr"),
                                                          model_vis)})
                out_data.append(out_ds)
            writes.append(xds_to_table(out_data, ims, columns=[self.model_column]))
        dask.compute(writes, scheduler='single-threaded')

                

def populate_model(vis, model_vis):
    return da.blockwise(_populate_model, ('row', 'chan', 'corr'),
                        vis, ('row', 'chan'),
                        model_vis, ('row', 'chan', 'corr'),
                        dtype=model_vis.dtype)

def _populate_model(vis, model_vis):
    model_vis[:, :, 0] = vis
    if model_vis.shape[-1] > 1:
        model_vis[:, :, -1] = vis
    return model_vis

def accumulate_dirty(dirties, nband, band_mapping):
    _, nx, ny = dirties[0].shape
    dirty = np.zeros((nband, nx, ny), dtype=dirties[0].dtype)
    d = 0
    for ims in band_mapping:
        for spw in band_mapping[ims]:
            for b, band in enumerate(band_mapping[ims][spw]):
                dirty[band] += dirties[d][b]
            d += 1
    return dirty

def corr_to_stokes(data, weights):
    return da.blockwise(_corr_to_stokes_wrapper, ('row', 'chan'),
                        data, ('row', 'chan', 'corr'),
                        weights, ('row', 'chan', 'corr'),
                        dtype=data.dtype)

def _corr_to_stokes_wrapper(data, weights):
    return _corr_to_stokes(data[0], weights[0])

def _corr_to_stokes(data, weights):
    if data.shape[-1] > 1:
        dataxx = data[:, :, 0]
        datayy = data[:, :, -1]

        weightsxx = weights[:, :, 0]
        weightsyy = weights[:, :, -1]

        data = dataxx * weightsxx + datayy * weightsyy
        weights = weightsxx + weightsyy
        data = da.where(weights > 0, data/weights, 0.0j)
    else:
        data = data[:, :, 0]
        weights = weights[:, :, 0]
    return data


def sum_weights(weights):
    return da.blockwise(_sum_weights_wrapper, ('row', 'chan'),
                        weights, ('row', 'chan', 'corr'),
                        dtype=weights.dtype)

def _sum_weights_wrapper(weights):
    return _sum_weights(weights[0])

def _sum_weights(weights):
    if weights.shape[-1] > 1:
        return weights[:, :, 0] + weights[:, :, -1]
    else:
        return weights[:, :, 0]