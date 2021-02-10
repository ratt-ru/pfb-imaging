
import numpy as np
import dask
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset
import zarr
from africanus.gridding.wgridder.dask import dirty as vis2im
from africanus.gridding.wgridder.dask import model as im2vis
from africanus.gridding.wgridder.dask import residual as im2residim
from africanus.gridding.wgridder.dask import hessian
from africanus.model.coherency.dask import convert
from ducc0.fft import good_size

class Gridder(object):
    def __init__(self, ms_name, nx, ny, cell_size, nband=None, nthreads=8, do_wstacking=1, Stokes='I',
                 row_chunks=100000, chan_chunks=32, optimise_chunks=True, epsilon=1e-5, psf_oversize=2.0,
                 data_column='CORRECTED_DATA', weight_column='WEIGHT_SPECTRUM', mueller_column=None,
                 model_column="MODEL_DATA", flag_column='FLAG', imaging_weight_column=None, mask=None, real_type='f4'):
        '''
        A note on chunking - currently row_chunks and chan_chunks are only used for the
        compute_weights() and write_component_model() methods. All other methods assume
        that the data for a single imaging band per ms and spw fit into memory. 
        '''
        if Stokes != 'I':
            raise NotImplementedError("Only Stokes I currently supported")
        self.nx = nx
        self.ny = ny
        self.cell = cell_size * np.pi/60/60/180
        self.nthreads = nthreads
        self.do_wstacking = do_wstacking
        self.epsilon = epsilon
        self.row_chunks = row_chunks
        self.chan_chunks = chan_chunks
        self.psf_oversize = psf_oversize
        self.nx_psf = good_size(int(self.psf_oversize * self.nx))
        self.ny_psf = good_size(int(self.psf_oversize * self.ny))
        self.real_type = real_type

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
                all_freqs.append(list([tmp_freq]))


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
                freq = np.atleast_1d(dask.compute(self.freq[ims][spw])[0])
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


        self.data_column = data_column
        self.weight_column = weight_column
        self.model_column = model_column
        self.flag_column = flag_column

        self.imaging_weight_column = imaging_weight_column
        self.mueller_column = mueller_column

        self.conv_weight_column = 'CONV_WEIGHT_SPECTRUM'

        self.columns = (self.data_column, self.weight_column, self.model_column,
                        self.flag_column, 'UVW')

        if self.imaging_weight_column is not None:
            self.columns += (self.imaging_weight_column,)

        if self.mueller_column is not None:
            self.columns += (self.mueller_column,)

        # self.conv_weight_column)


        if mask is not None:
            self.mask = mask
        else:
            self.mask = lambda x: x

        # optimise chunking by concatenating measurement sets and pre-computing corr to Stokes
        # if optimise_chunks:
        #     for 

    def compute_weights(self):
        from pfb.utils.weighting import compute_uniform_counts, counts_to_weights
        # compute counts
        counts = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks=self.chunks[ims],
                              columns=('UVW'))

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

                freq_bin_idx = self.freq_bin_idx[ims][spw]
                freq_bin_counts = self.freq_bin_counts[ims][spw]
                freq = self.freq[ims][spw]

                uvw = ds.UVW.data

                count = compute_uniform_counts(uvw, freq, freq_bin_idx, freq_bin_counts, 2*self.nx, 2*self.ny, self.cell, self.cell, self.real_type)

                counts.append(count) 
            
        counts = dask.compute(counts)[0]
        counts = accumulate_dirty(counts, self.nband, self.band_mapping)

        # return counts
        
        counts = da.from_array(counts, chunks=(1, -1, -1))

        # convert counts to uniform weights
        writes = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks={'row':(self.row_chunks,), 'chan':(self.chan_chunks,)},
                              columns=self.columns)

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

                spw = ds.DATA_DESC_ID  # this is not correct, need to use spw

                freq_bin_idx = self.freq_bin_idx[ims][spw]
                freq_bin_counts = self.freq_bin_counts[ims][spw]
                freq = self.freq[ims][spw]

                uvw = ds.UVW.data

                weights = counts_to_weights(counts, uvw, freq, freq_bin_idx, freq_bin_counts, 2*self.nx, 2*self.ny, self.cell, self.cell, np.float32)

                # hack to get shape and chunking info
                data = getattr(ds, self.data_column).data
                weights = da.broadcast_to(weights[:, :, None], data.shape, chunks=data.chunks)
                out_ds = ds.assign(**{"IMAGING_WEIGHT_SPECTRUM": (("row", "chan", "corr"),
                                   weights)})
                out_data.append(out_ds)
            writes.append(xds_to_table(out_data, ims, columns=["IMAGING_WEIGHT_SPECTRUM"]))
        dask.compute(writes)

    def make_residual(self, x):
        # Note deprecated (does not support Jones terms) 
        print("Making residual")
        x = da.from_array(self.mask(x).astype(self.real_type), chunks=(1, self.nx, self.ny), name=False)
        residuals = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks=self.chunks[ims],
                              columns=self.columns)

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
                    

                if self.imaging_weight_column is not None:
                    imaging_weights = getattr(ds, self.imaging_weight_column).data
                    if len(imaging_weights.shape) < 3:
                        imaging_weights = da.broadcast_to(imaging_weights[:, None, :], data.shape, chunks=data.chunks)

                    weightsxx = imaging_weights[:, :, 0] * weights[:, :, 0]
                    weightsyy = imaging_weights[:, :, -1] * weights[:, :, -1]
                else:
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
                                      nthreads=self.nthreads, epsilon=self.epsilon,
                                      do_wstacking=self.do_wstacking, double_accum=True)

                residuals.append(residual)
        
        
        residuals = dask.compute(residuals, scheduler='single-threaded')[0]
        
        return self.mask(accumulate_dirty(residuals, self.nband, self.band_mapping).astype(self.real_type))

    def make_dirty(self):
        print("Making dirty")
        dirty = da.zeros((self.nband, self.nx, self.ny), 
                         dtype=np.float32,
                         chunks=(1, self.nx, self.ny), name=False)
        dirties = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks=self.chunks[ims],
                              columns=self.columns)

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
                
                if self.imaging_weight_column is not None:
                    imaging_weights = getattr(ds, self.imaging_weight_column).data
                    if len(imaging_weights.shape) < 3:
                        imaging_weights = da.broadcast_to(imaging_weights[:, None, :], data.shape, chunks=data.chunks)

                    weightsxx = imaging_weights[:, :, 0] * weights[:, :, 0]
                    weightsyy = imaging_weights[:, :, -1] * weights[:, :, -1]
                else:
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
                               epsilon=self.epsilon, do_wstacking=self.do_wstacking,
                               double_accum=True)

                
                dirties.append(dirty)
        
        dirties = dask.compute(dirties, scheduler='single-threaded')[0]
        
        return self.mask(accumulate_dirty(dirties, self.nband, self.band_mapping).astype(self.real_type))

    def make_psf(self):
        print("Making PSF")
        psfs = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks=self.chunks[ims],
                              columns=self.columns)

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
                
                freq_bin_idx = self.freq_bin_idx[ims][spw]
                freq_bin_counts = self.freq_bin_counts[ims][spw]
                freq = self.freq[ims][spw]

                uvw = ds.UVW.data

                flag = getattr(ds, self.flag_column).data

                weights = getattr(ds, self.weight_column).data
                if len(weights.shape) < 3:
                    weights = da.broadcast_to(weights[:, None, :], flag.shape, chunks=flag.chunks)
                
                if self.imaging_weight_column is not None:
                    imaging_weights = getattr(ds, self.imaging_weight_column).data
                    if len(imaging_weights.shape) < 3:
                        imaging_weights = da.broadcast_to(imaging_weights[:, None, :], data.shape, chunks=data.chunks)

                    weightsxx = imaging_weights[:, :, 0] * weights[:, :, 0]
                    weightsyy = imaging_weights[:, :, -1] * weights[:, :, -1]
                else:
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
                             self.nx_psf, self.ny_psf, self.cell, flag=flag.astype(np.uint8),
                             nthreads=self.nthreads, epsilon=self.epsilon,
                             do_wstacking=self.do_wstacking, double_accum=True)

                psfs.append(psf)

        psfs = dask.compute(psfs)[0]

        # LB - this assumes that the beam is normalised to 1 at the center        
        return accumulate_dirty(psfs, self.nband, self.band_mapping).astype(self.real_type)

    def convolve(self, x):
        print("Applying Hessian")
        x = da.from_array(self.mask(x).astype(self.real_type), chunks=(1, self.nx, self.ny), name=False)
        convolvedims = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks=self.chunks[ims],
                              columns=self.columns)

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

                spw = ds.DATA_DESC_ID  # is this correct?

                freq_bin_idx = self.freq_bin_idx[ims][spw]
                freq_bin_counts = self.freq_bin_counts[ims][spw]
                freq = self.freq[ims][spw]

                uvw = ds.UVW.data

                # only keep data where both corrs are unflagged
                flag = getattr(ds, self.flag_column).data

                weights = getattr(ds, self.weight_column).data
                if len(weights.shape) < 3:
                    weights = da.broadcast_to(weights[:, None, :], flag.shape, chunks=flag.chunks)
                    

                if self.imaging_weight_column is not None:
                    imaging_weights = getattr(ds, self.imaging_weight_column).data
                    if len(imaging_weights.shape) < 3:
                        imaging_weights = da.broadcast_to(imaging_weights[:, None, :], flag.shape, chunks=flag.chunks)

                    weightsxx = imaging_weights[:, :, 0] * weights[:, :, 0]
                    weightsyy = imaging_weights[:, :, -1] * weights[:, :, -1]
                else:
                    weightsxx = weights[:, :, 0]
                    weightsyy = weights[:, :, -1]

                # weighted sum corr to Stokes I
                weights = weightsxx + weightsyy
                
                flagxx = flag[:, :, 0]
                flagyy = flag[:, :, -1]
                flag = ~ (flagxx | flagyy)  # ducc0 convention

                bands = self.band_mapping[ims][spw]
                model = x[list(bands), :, :]
                convolvedim = hessian(uvw, freq, model, freq_bin_idx, freq_bin_counts,
                                      self.cell, weights=weights, flag=flag.astype(np.uint8),
                                      nthreads=self.nthreads, epsilon=self.epsilon,
                                      do_wstacking=self.do_wstacking, double_accum=True)

                convolvedims.append(convolvedim)
        
        
        convolvedims = dask.compute(convolvedims, scheduler='single-threaded')[0]
        
        return self.mask(accumulate_dirty(convolvedims, self.nband, self.band_mapping).astype(self.real_type))

    def write_model(self, x):
        print("Writing model data")
        x = da.from_array(self.mask(x).astype(np.float32), chunks=(1, self.nx, self.ny))
        writes  = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks=self.chunks[ims],
                              columns=('MODEL_DATA', 'UVW'))

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

                spw = ds.DATA_DESC_ID  # this is not correct, need to use spw

                freq_bin_idx = self.freq_bin_idx[ims][spw]
                freq_bin_counts = self.freq_bin_counts[ims][spw]
                freq = self.freq[ims][spw]

                uvw = ds.UVW.data

                model_vis = getattr(ds, 'MODEL_DATA').data

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

    def write_component_model(self, comps, ref_freq, mask, row_chunks, chan_chunks):
        raise NotImplementedError("Need to fix beamed masking before this will work")
        print("Writing model data at full freq resolution")
        order, npix = comps.shape
        comps = da.from_array(comps, chunks=(-1, -1))
        mask = da.from_array(mask.squeeze(), chunks=(-1, -1))
        writes  = []
        for ims in self.ms:
            xds = xds_from_ms(ims, group_cols=('FIELD_ID', 'DATA_DESC_ID'),
                              chunks={'row':(row_chunks,), 'chan':(chan_chunks,)},
                              columns=('MODEL_DATA', 'UVW'))

            # subtables
            ddids = xds_from_table(ims + "::DATA_DESCRIPTION")
            fields = xds_from_table(ims + "::FIELD", group_cols="__row__")
            spws = xds_from_table(ims + "::SPECTRAL_WINDOW", group_cols="__row__")
            pols = xds_from_table(ims + "::POLARIZATION", group_cols="__row__")

            # subtable data
            ddids = dask.compute(ddids)[0]
            fields = dask.compute(fields)[0]
            pols = dask.compute(pols)[0]

            out_data = []
            for ds in xds:
                field = fields[ds.FIELD_ID]
                radec = field.PHASE_DIR.data.squeeze()
                if not np.array_equal(radec, self.radec):
                    continue

                spw = spws[ds.DATA_DESC_ID]
                freq = spw.CHAN_FREQ.data.squeeze()
                freq_bin_idx = da.arange(0, freq.size, 1, chunks=freq.chunks, dtype=np.int64)
                freq_bin_counts = da.ones(freq.size, chunks=freq.chunks, dtype=np.int64)

                uvw = ds.UVW.data

                model_vis = getattr(ds, 'MODEL_DATA').data

                model = model_from_comps(comps, freq, mask, ref_freq)
                
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
        data = np.where(weights > 0, data/weights, 0.0j)
    else:
        data = data[:, :, 0]
        weights = weights[:, :, 0]
    return data


def model_from_comps(comps, freq, mask, ref_freq):
    return da.blockwise(_model_from_comps_wrapper, ('chan', 'nx', 'ny'),
                        comps, ('com', 'pix'),
                        freq, ('chan',),
                        mask, ('nx', 'ny'),
                        ref_freq, None,
                        dtype=comps.dtype)

def _model_from_comps_wrapper(comps, freq, mask, ref_freq):
    return _model_from_comps(comps[0][0], freq, mask, ref_freq)

def _model_from_comps(comps, freq, mask, ref_freq):
    order, npix = comps.shape
    nband = freq.size
    nx, ny = mask.shape
    model = np.zeros((nband, nx, ny), dtype=comps.dtype)
    w = (freq / ref_freq).reshape(freq.size, 1)
    Xdes = np.tile(w, order) ** np.arange(0, order)
    beta_rec = Xdes.dot(comps)
    Idx = np.argwhere(mask).squeeze()
    for i, xy in enumerate(Idx):
        ix = xy[0]
        iy = xy[1]
        model[:, ix, iy] = beta_rec[:, i]
    return model


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