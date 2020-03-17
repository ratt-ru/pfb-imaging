import numpy as np
import dask.array as da
from daskms import xds_from_table
import nifty_gridder as ng
from pypocketfft import r2c, c2r
from utils import freqmul
from africanus.gps.kernels import exponential_squared as expsq
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

class Gridder(object):
    def __init__(self, uvw, freq, sqrtW, args):
        self.wgt = sqrtW
        self.uvw = uvw
        self.nrow = uvw.shape[0]
        self.nx = args.nx
        self.ny = args.ny
        self.cell = args.cell_size * np.pi/60/60/180
        self.freq = freq
        self.precision = args.precision
        self.nthreads = args.ncpu
        self.do_wstacking = args.do_wstacking
        self.flags = np.where(self.wgt==0, 1, 0)

        # freq mapping
        self.nchan = freq.size
        if args.channels_out is None or args.channels_out == 0:
            args.channels_out = self.nchan
        step = self.nchan//args.channels_out
        freq_mapping = np.arange(0, self.nchan, step)
        self.freq_mapping = np.append(freq_mapping, self.nchan)
        self.nband = self.freq_mapping.size - 1
        self.freq_out = np.zeros(self.nband)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            self.freq_out[i] = np.mean(self.freq[Ilow:Ihigh])

    def dot(self, x):
        model_data = np.zeros((self.nrow, self.nchan), dtype=np.complex128)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            model_data[:, Ilow:Ihigh] = ng.dirty2ms(uvw=self.uvw, freq=self.freq[Ilow:Ihigh], dirty=x[i], wgt=self.wgt[:, Ilow:Ihigh],
                                                    pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                                    nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return model_data

    def hdot(self, x):
        image = np.zeros((self.nband, self.nx, self.ny), dtype=np.float64)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            image[i] = ng.ms2dirty(uvw=self.uvw, freq=self.freq[Ilow:Ihigh], ms=x[:, Ilow:Ihigh], wgt=self.wgt[:, Ilow:Ihigh],
                                   npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                   nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return image

    def make_psf(self):
        psf_array = np.zeros((self.nband, 2*self.nx, 2*self.ny))
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            psf_array[i] = ng.ms2dirty(uvw=self.uvw, freq=self.freq[Ilow:Ihigh], 
                                       ms=self.wgt[:, Ilow:Ihigh].astype(np.complex128), wgt=self.wgt[:, Ilow:Ihigh],
                                       npix_x=2*self.nx, npix_y=2*self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                       epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking)
        return psf_array

    def convolve(self, x):
        return self.hdot(self.dot(x))


class OutMemGridder(object):
    def __init__(self, table_name, freq, args):
        self.nx = args.nx
        self.ny = args.ny
        self.cell = args.cell_size * np.pi/60/60/180
        self.precision = args.precision
        self.nthreads = args.ncpu
        self.do_wstacking = args.do_wstacking

        # freq mapping
        self.freq = freq
        self.nchan = freq.size
        if args.channels_out is None or args.channels_out == 0:
            args.channels_out = self.nchan
        step = self.nchan//args.channels_out
        freq_mapping = np.arange(0, self.nchan, step)
        self.freq_mapping = np.append(freq_mapping, self.nchan)
        self.nband = self.freq_mapping.size - 1
        self.freq_out = np.zeros(self.nband)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            self.freq_out[i] = np.mean(self.freq[Ilow:Ihigh])

        self.chan_chunks = self.freq_mapping[1] - self.freq_mapping[0]

        # meta info for xds_from_table
        self.table_name = table_name
        self.schema = {
            "DATA": {'dims': ('chan',)},
            "WEIGHT": {'dims': ('chan', )},
            "UVW": {'dims': ('uvw',)},
        }
        
    def make_residual(self, x, v_dof=None):
        tbl = xds_from_table(self.table_name, chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)[0]
        data = tbl.DATA.data
        weights = tbl.WEIGHT.data
        uvw = tbl.UVW.data.compute()
        residual = np.zeros(x.shape, dtype=x.dtype)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            weighti = weights.blocks[:, i].compute().astype(np.float64)
            datai = data.blocks[:, i].compute().astype(np.complex128)
            residual_vis = datai - ng.dirty2ms(uvw=uvw, freq=self.freq[Ilow:Ihigh], dirty=x[i], wgt=weighti,
                                               pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                               nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
            # # recompute weights for band
            # if v_dof is not None:
            #     _, robust_reweight

            # make residual image
            residual[i] = ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=residual_vis, wgt=weighti,
                                      npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                      epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return residual

    def make_dirty(self):
        tbl = xds_from_table(self.table_name, chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)[0]
        data = tbl.DATA.data
        weights = tbl.WEIGHT.data
        uvw = tbl.UVW.data.compute()
        dirty = np.zeros((self.nband, self.nx, self.ny), dtype=np.float64)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            weighti = weights.blocks[:, i].compute().astype(np.float64)
            datai = data.blocks[:, i].compute().astype(np.complex128)
            dirty[i] = ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=datai, wgt=weighti,
                                   npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                   epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return dirty

    def make_psf(self):
        tbl = xds_from_table(self.table_name, chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)[0]
        weights = tbl.WEIGHT.data
        uvw = tbl.UVW.data.compute()
        psf_array = np.zeros((self.nband, 2*self.nx, 2*self.ny))
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            weighti = weights.blocks[:, i].compute().astype(np.float64)
            psf_array[i] = ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=weighti.astype(np.complex128), wgt=weighti,
                                       npix_x=2*self.nx, npix_y=2*self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                       epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking)
        return psf_array


class PSF(object):
    def __init__(self, psf, nthreads):
        self.nthreads = nthreads
        self.nband, nx_psf, ny_psf = psf.shape
        nx = nx_psf//2
        ny = ny_psf//2
        npad_x = (nx_psf - nx)//2
        npad_y = (ny_psf - ny)//2
        self.padding = ((0,0), (npad_x, npad_x), (npad_y, npad_y))
        self.ax = (1,2)
        self.unpad_x = slice(npad_x, -npad_x)
        self.unpad_y = slice(npad_y, -npad_y)
        self.lastsize = ny + np.sum(self.padding[-1])
        self.psf = psf
        psf_pad = iFs(psf, axes=self.ax)
        self.psfhat = r2c(psf_pad, axes=self.ax, forward=True, nthreads=nthreads, inorm=0)

    def convolve(self, x):
        xhat = iFs(np.pad(x, self.padding, mode='constant'), axes=self.ax)
        xhat = r2c(xhat, axes=self.ax, nthreads=self.nthreads, forward=True, inorm=0)
        xhat = c2r(xhat * self.psfhat, axes=self.ax, forward=False, lastsize=self.lastsize, inorm=2, nthreads=self.nthreads)
        return Fs(xhat, axes=self.ax)[:, self.unpad_x, self.unpad_y] 

class Prior(object):
    def __init__(self, freq, sigma0, l, nx, ny, nthreads=8):
        self.nthreads = nthreads
        self.nx = nx
        self.ny = ny
        self.nband = freq.size
        self.freq = freq/np.mean(freq)
        self.Kv = expsq(self.freq, self.freq, sigma0, l)
        self.x0 = np.zeros((self.nband, self.nx, self.ny), dtype=freq.dtype)

        self.Kvinv = np.linalg.inv(self.Kv + 1e-12*np.eye(self.nband))

        self.L = np.linalg.cholesky(self.Kv + 1e-12*np.eye(self.nband))
        self.LH = self.L.T
        
    def idot(self, x):
        return freqmul(self.Kvinv, x.reshape(self.nband, self.nx*self.ny)).reshape(self.nband, self.nx, self.ny)
    
    def dot(self, x):
        return freqmul(self.Kv, x.reshape(self.nband, self.nx*self.ny)).reshape(self.nband, self.nx, self.ny)

    def sqrtdot(self, x):
        return freqmul(self.L, x.reshape(self.nband, self.nx*self.ny)).reshape(self.nband, self.nx, self.ny)

    def sqrthdot(self, x):
        return freqmul(self.LH, x.reshape(self.nband, self.nx*self.ny)).reshape(self.nband, self.nx, self.ny)