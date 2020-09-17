
import numpy as np
import dask
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset
import zarr
from africanus.gridding.wgridder.dask import dirty as vis2im
from africanus.gridding.wgridder.dask import model as im2vis
from africanus.gridding.wgridder.dask import residual as im2residim
from africanus.model.coherency.dask import convert


class Gridder(object):
    def __init__(self, ms_name, nx, ny, cell_size, nband=None, ncpu=8, do_wstacking=1, Stokes='I', row_chunks=100000,
                 data_column='DATA', weight_column='WEIGHT', model_column="MODEL_DATA", flag_column='FLAG'):
        if Stokes != 'I':
            raise NotImplementedError("Only Stokes I currently supported")
        self.nx = nx
        self.ny = ny
        self.cell = cell_size * np.pi/60/60/180
        self.nthreads = ncpu

        self.data_column = data_column
        self.weight_column = weight_column
        self.model_column = model_column
        self.flag_column = flag_column
        if isinstance(ms_name, list):
            self.ms = ms_name
        else:
            self.ms = [ms_name]

        # first pass through data to determine freq_mapping chunking
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
        self.bands_da = da.from_array(np.arange(self.nband), chunks=1)
        # bin edges
        fmin = ufreqs[0]
        fmax = ufreqs[-1]
        fbins = np.linspace(fmin, fmax, self.nband+1)
        # chan <-> band mapping
        self.band_mapping = {}
        self.chunks = {}
        for ims in self.freq:
            self.band_mapping[ims] = {}
            self.chunks[ims] = []
            for spw in self.freq[ims]:
                freq = dask.compute(self.freq[ims][spw])[0]
                band_map = np.zeros(freq.size, dtype=np.int32)
                for band in range(self.nband):
                    indl = freq >= fbins[band]
                    indu = freq < fbins[band + 1] + 1e-6
                    band_map = np.where(indl & indu, band, band_map)
                    # bin_idx.append(I[0])
                    # bin_counts.append(I.size)
                # to dask arrays
                bands, counts = np.unique(band_map, return_counts=True)
                self.band_mapping[ims][spw] = tuple(bands)
                self.chunks[ims].append({'row':(-1,), 'chan':tuple(counts)})
                
                # self.freq[ims][spw] = da.from_array(freq, chunks=bin_counts)
                # self.freq_bin_idx[ims][spw] = da.from_array(bin_idx, chunks=1)
                # self.freq_bin_counts[ims][spw] = da.from_array(bin_counts, chunks=1)

        # if optimise_chunks:
        # ....

    def make_residual(self, x):
        print("Making residual")
        residual = da.zeros(x.shape, dtype=np.float32, chunks=(1, self.nx, self.ny))
        x = da.from_array(x.astype(np.float32), chunks=(1, self.nx, self.ny))
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks={"row":-1, "chan": self.chan_chunks},
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

                print("Processing field %i"%ds.FIELD_ID)
                ddid = ds.DATA_DESC_ID  # this is not correct, need to use spw
                
                freq_bin_idx = self.freq_bin_idx[ims][ddid]
                freq_bin_counts = self.freq_bin_counts[ims][ddid]
                freq = self.freq[ims][ddid]

                uvw = ds.UVW.data

                data = getattr(ds, self.data_column).data
                dataxx = data[:, :, 0].rechunk((-1, freq_bin_counts))
                datayy = data[:, :, -1].rechunk((-1, freq_bin_counts))

                weights = getattr(ds, self.weight_column).data
                if len(weights.shape) < 3:
                    weights = da.broadcast_to(weights, data.shape, chunks=(-1, freq_bin_counts))
                weightsxx = weights[:, :, 0]
                weightsyy = weights[:, :, -1]

                # weighted sum corr to Stokes I
                weights = weightsxx + weightsyy
                data = (weightsxx * dataxx + weightsyy * datayy)
                ind = weights > 0
                data[ind] = data[ind]/weights[ind]
                
                # only keep data where both corrs are unflagged
                flag = getattr(ds, self.flag_column).data
                flagxx = flag[:, :, 0].rechunk((-1, freq_bin_counts))
                flagyy = flag[:, :, -1].rechunk((-1, freq_bin_counts))
                flag = flagxx | flagyy

                residual += im2residim(uvw, freq, x, data, self.freq_bin_idx, self.freq_bin_counts,
                                       self.cell, weights=weights, flag=flag, nthreads=self.nthreads)
                
        return residual.compute(scheduler='single-threaded')

    def make_dirty(self):
        print("Making dirty")
        dirty = da.zeros((self.nband, self.nx, self.ny), 
                         dtype=np.float32,
                         chunks=(1, self.nx, self.ny))
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

                print("Processing field %i"%ds.FIELD_ID)
                spw = ds.DATA_DESC_ID  # this is not correct, need to use spw
                
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
                    weights = da.broadcast_to(weights[:, None, :], data.shape, chunks=(-1, freq_chunk, -1))
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

                this_dirty = vis2im(uvw, freq, data, freq_bin_idx, freq_bin_counts,
                                    self.nx, self.ny, self.cell, weights=weights,
                                    flag=flag.astype(np.uint8), nthreads=self.nthreads)

                dirty = accumulate_dirty(dirty, this_dirty, self.bands_da, self.band_mapping[ims][spw])
                
                
        return dirty.compute(scheduler='single-threaded')

    def make_psf(self):
        print("Making PSF")
        psf = da.zeros((self.nband, 2*self.nx, 2*self.ny), 
                       dtype=np.float32,
                       chunks=(1, 2*self.nx, 2*self.ny))
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks={"row":-1, "chan": -1},
                              columns=(self.weight_column, self.flag_column, 'UVW'))

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

                print("Processing field %i"%ds.FIELD_ID)
                ddid = ds.DATA_DESC_ID  # this is not correct, need to use spw
                
                freq_bin_idx = self.freq_bin_idx[ims][ddid]
                freq_bin_counts = self.freq_bin_counts[ims][ddid]
                freq = self.freq[ims][ddid]
                freq_chunk = freq_bin_counts[0].compute()

                uvw = ds.UVW.data

                flag = getattr(ds, self.flag_column).data

                weights = getattr(ds, self.weight_column).data
                if len(weights.shape) < 3:
                    weights = da.broadcast_to(weights[:, None, :], flag.shape, chunks=(-1, freq_chunk, -1))
                weightsxx = weights[:, :, 0]
                weightsyy = weights[:, :, -1]

                # weighted sum corr to Stokes I
                weights = weightsxx + weightsyy
                data = weights.astype(np.complex64)
                
                # only keep data where both corrs are unflagged
                flagxx = flag[:, :, 0].rechunk((-1, freq_chunk))
                flagyy = flag[:, :, -1].rechunk((-1, freq_chunk))
                flag = ~ (flagxx | flagyy)  # ducc0 convention

                psf += vis2im(uvw, freq, data, freq_bin_idx, freq_bin_counts,
                              2*self.nx, 2*self.ny, self.cell, flag=flag,
                              nthreads=self.nthreads)
                
        return psf.compute(scheduler='single-threaded')

    def write_model(self, x):
        x = da.from_array(x.astype(np.float32), chunks=(1, self.nx, self.ny))
        writes  = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks={"row":-1, "chan": -1},
                              columns=(self.model_column, self.flag_column, 'UVW'))
            
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

                print("Processing field %i"%ds.FIELD_ID)
                ddid = ds.DATA_DESC_ID  # this is not correct, need to use spw

                freq_bin_idx = self.freq_bin_idx[ims][ddid]
                freq_bin_counts = self.freq_bin_counts[ims][ddid]
                freq = self.freq[ims][ddid]
                freq_chunk = freq_bin_counts[0].compute()

                uvw = ds.UVW.data
                flag = getattr(ds, self.flag_column).data.rechunk(-1, freq_chunk, -1)

                visxx = im2vis(uvw, freq, x, freq_bin_idx, freq_bin_counts,
                               self.cell, nthreads=self.nthreads)

                # convert Stokes to corr
                zero = da.zeros(visxx.shape, dtype=visxx.dtype, chunks=(-1, freq_chunk))
                model = da.stack((visxx, zero, zero, visxx), axis=2)
                
                out_ds = ds.assign(**{self.model_column: (("row", "chan", "corr"),
                                                          model)})
                out_data.append(out_ds)
            writes.append(xds_to_table(out_data, ims, columns=[self.model_column]))
        dask.compute(writes, scheduler='single-threaded')

                

# def make_stokes_I(vis, weight, flag):
#     visxx = vis[:, :, 0]
#     visyy = vis[:, :, -1]

def accumulate_dirty(dirty, this_dirty, bands, subbands):
    return da.blockwise(_accumulate_dirty, ('band', 'nx', 'ny'),
                        dirty, ('band', 'nx', 'ny'),
                        this_dirty, ('subband', 'nx', 'ny'),
                        bands, ('band',),
                        subbands, None,
                        dtype=dirty_da.dtype).compute(scheduler='single-threaded')

def _accumulate_dirty(dirty, this_dirty, bands, subbands):
    if bands in subbands:
        dirty += this_dirty[subbands.index(bands)]
    return dirty


# # check freqs match
# if not np.testing.assert_array_equal(self.freq, spw.CHAN_FREQ)
#     continue

# # get data and convert to Stokes I
# data = getattr(ds, self.data_column).data
# weight = getattr(ds, self.weight_column).data
# if len(weight.shape)<3:  # tile over frequency if no WEIGHT_SPECTRUM
#     weight = da.tile(weight[:, None, :], (1, self.freq.size, 1))
# uvw = ds.UVW.data
# flag = da.logical_or(ds.FLAG.data, ds.FLAG_ROW.data[:, None, None])

# if len(data.shape) > and data.shape[-1] > 1:
#     data = (weight[:, :, 0] * data[:, :, 0] + weight[:, :, -1] * data[:, :, -1])
#     weight = (weight[:, :, 0] + weight[:, :, -1])
#     # normalise by sum of weights
#     weight = da.where(flag, 0.0, weight)
#     data = da.where(weight != 0.0, data/weight, 0.0j)
#     flag = flag[:, :, 0] | flag[:, :, -1]  # only keep data where both correlations are unflagged
# else:
#     data = data.squeeze()
#     weight = weight.squeeze()
#     flag = flag.squeeze()

# data_vars = {
#     'FIELD_ID':(('row',), da.full_like(ds.TIME.data, field_id)),
#     'DATA_DESC_ID':(('row',), da.full_like(ds.TIME.data, ddid_id)),
#     args.data_out_column:(('row', 'chan'), data),
#     'WEIGHT':(('row', 'chan'), weight),
#     args.weight_out_column:(('row', 'chan'), da.sqrt(weight)),
#     'UVW':(('row', 'uvw'), uvw)
# }

# out_ds = Dataset(data_vars)

# out_datasets.append(out_ds) 