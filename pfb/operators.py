import numpy as np
import dask.array as da
from daskms import xds_from_table
import nifty_gridder as ng
from pypocketfft import r2c, c2r
from pfb.utils import freqmul
from africanus.gps.kernels import exponential_squared as expsq
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

class Gridder(object):
    def __init__(self, uvw, freq, sqrtW, nx, ny, cell_size, nband=None, precision=1e-7, ncpu=8, do_wstacking=1):
        self.wgt = sqrtW
        self.uvw = uvw
        self.nrow = uvw.shape[0]
        self.nx = nx
        self.ny = ny
        self.cell = cell_size * np.pi/60/60/180
        self.freq = freq
        self.precision = precision
        self.nthreads = ncpu
        self.do_wstacking = do_wstacking
        self.flags = np.where(self.wgt==0, 1, 0)

        # freq mapping
        self.nchan = freq.size
        if nband is None or nband == 0:
            self.nband = self.nchan
        else:
            self.nband = nband
        step = self.nchan//self.nband
        freq_mapping = np.arange(0, self.nchan, step)
        self.freq_mapping = np.append(freq_mapping, self.nchan)
        self.freq_out = np.zeros(self.nband)
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            self.freq_out[i] = np.mean(self.freq[Ilow:Ihigh])  # weighted mean?

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
    def __init__(self, table_name, nx, ny, cell_size, freq, nband=None, field=0, precision=1e-7, ncpu=8, do_wstacking=1):
        if precision > 1e-6:
            self.real_type = np.float32
            self.complex_type = np.complex64
        else:
            self.real_type = np.float64
            self.complex_type=np.complex128

        self.nx = nx
        self.ny = ny
        self.cell = cell_size * np.pi/60/60/180
        if isinstance(field, list):
            self.field = field
        else:
            self.field = [field]
        self.precision = precision
        self.nthreads = ncpu
        self.do_wstacking = do_wstacking

        # freq mapping
        self.freq = freq
        self.nchan = freq.size
        if nband is None:
            self.nband = self.nchan
        else:
            self.nband = nband
        step = self.nchan//self.nband
        freq_mapping = np.arange(0, self.nchan, step)
        self.freq_mapping = np.append(freq_mapping, self.nchan)
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
        print("Making residual")
        residual = np.zeros(x.shape, dtype=x.dtype)
        xds = xds_from_table(self.table_name, group_cols=('FIELD_ID'), chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)
        for ds in xds:
            if ds.FIELD_ID not in list(self.field):
                continue
            print("Processing field %i"%ds.FIELD_ID)
            data = ds.DATA.data
            weights = ds.WEIGHT.data
            uvw = ds.UVW.data.compute().astype(self.real_type)
            
            for i in range(self.nband):
                Ilow = self.freq_mapping[i]
                Ihigh = self.freq_mapping[i+1]
                weighti = weights.blocks[:, i].compute().astype(self.real_type)
                datai = data.blocks[:, i].compute().astype(self.complex_type)

                # TODO - load and apply interpolated fits beam patterns for field

                # get residual vis
                residual_vis = weighti * datai - ng.dirty2ms(uvw=uvw, freq=self.freq[Ilow:Ihigh], dirty=x[i], wgt=weighti,
                                                             pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                                             nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)

                # make residual image
                residual[i] += ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=residual_vis, wgt=weighti,
                                           npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                           epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return residual

    def make_dirty(self):
        print("Making dirty")
        dirty = np.zeros((self.nband, self.nx, self.ny), dtype=self.real_type)
        xds = xds_from_table(self.table_name, group_cols=('FIELD_ID'), chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)
        for ds in xds:
            if ds.FIELD_ID not in list(self.field):
                continue
            print("Processing field %i"%ds.FIELD_ID)
            data = ds.DATA.data
            weights = ds.WEIGHT.data
            uvw = ds.UVW.data.compute().astype(self.real_type)
        
            for i in range(self.nband):
                Ilow = self.freq_mapping[i]
                Ihigh = self.freq_mapping[i+1]
                weighti = weights.blocks[:, i].compute().astype(self.real_type)
                datai = data.blocks[:, i].compute().astype(self.complex_type)

                # TODO - load and apply interpolated fits beam patterns for field

                dirty[i] += ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=weighti*datai, wgt=weighti,
                                        npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell,
                                        epsilon=self.precision, nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return dirty

    def make_psf(self):
        print("Making PSF")
        psf_array = np.zeros((self.nband, 2*self.nx, 2*self.ny))
        xds = xds_from_table(self.table_name, group_cols=('FIELD_ID'), chunks={"row":-1, "chan": self.chan_chunks}, table_schema=self.schema)
        for ds in xds:
            if ds.FIELD_ID not in list(self.field):
                continue
            print("Processing field %i"%ds.FIELD_ID)
            weights = ds.WEIGHT.data
            uvw = ds.UVW.data.compute().astype(self.real_type)
        
            for i in range(self.nband):
                Ilow = self.freq_mapping[i]
                Ihigh = self.freq_mapping[i+1]
                weighti = weights.blocks[:, i].compute().astype(self.real_type)
                psf_array[i] += ng.ms2dirty(uvw=uvw, freq=self.freq[Ilow:Ihigh], ms=weighti.astype(self.complex_type), wgt=weighti,
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

class PSI(object):
    def __init__(self, nx, ny,
                 nlevels=2,
                 basis=['self', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8']):
        """
        Sets up operators to move between wavelet coefficients
        in each basis and the image x. 
        
        Parameters
        ----------
        nx - number of pixels in x-dimension
        ny - number of pixels in y-dimension
        nlevels - The level of the decomposition. Default=2
        basis - List holding basis names. 
                Default is delta + first 8 DB wavelets

        Returns
        =======
        Psi - list of operators performing coeff to image where
            each entry corresponds to one of the basis elements.
        Psi_t - list of operators performing image to coeff where
                each entry corresponds to one of the basis elements.
        """
        self.P = len(basis)
        self.sqrtP = np.sqrt(self.P)

        # add dictionary entry for each basis element
        Psi = {}
        for i in range(P):
            Psi[i] = lambda x, b=basis[i]: Psi_func(x, b)

        # add dictionary entry for each basis element
        Psi_t = {}
        for i in range(P):
            Psi_t[i] = lambda x, b=basis[i]: Psi_t_func(x, nlevels, b)      

    def dot(alpha, base):
        """
        Takes array of coefficients to image. 
        The input does not have the form expected by pywt
        so we have to reshape it. Comes in as a flat vector
        arranged as

        [cAn, cHn, cVn, cDn, ..., cH1, cV1, cD1]

        where each entry is a flattened array and n denotes the 
        level of the decomposition. This has to be restructured as

        [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)]

        where entries are (nx//2**level + nx%2, ny//2**level + ny%2)
        arrays. 
        """ 
        if base == 'self':
            return alpha.reshape(nx, ny)/sqrtP
        else:
            # stack array back into expected shape
            indx = nx//2**nlevels + nx%2
            indy = ny//2**nlevels + ny%2
            n = indx * indy
            alpha_rec = [alpha[0:n].reshape(indx, indy)]
            ind = n
            for i in range(nlevels):
                indx = nx//2**(nlevels-i) + nx%2
                indy = ny//2**(nlevels-i) + ny%2
                n = indx * indy
                tpl = ()
                for j in range(3):
                    tpl += (alpha[ind:ind+n].reshape(indx, indy),)
                    ind += n
                alpha_rec.append(tpl)
            # return reconstructed image from coeff
            return pywt.waverec2(alpha_rec, base, mode='periodization')/sqrtP

    def hdot(self, x, level, base):
        """
        This implements the adjoint of Psi_func. 
        """
        if base == 'self':
            # just flatten image, no need to stack in this case
            return x.ravel()/sqrtP
        else:
            # decompose
            alpha = pywt.wavedec2(x, base, mode='periodization', level=level)
            # stack decomp into vector
            tmp = [alpha[0].ravel()]
            for item in alpha[1::]:
                for j in range(len(item)):
                    tmp.append(item[j].ravel())
            return np.concatenate(tmp)/sqrtP

