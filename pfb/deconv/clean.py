# import numpy as np
# from pfb.operators import PSF

# def grad_func(x, dirty, psfo):
#     return psfo.convolve(x) - dirty

# def clean(psf, model, residual, mask, sig_21, nthreads=0, maxit=10, 
#           gamma=0.1, peak_factor=0.85):

#     nband, nx, ny = residual.shape
    
#     # PSF operator
#     psfo = PSF(psf, nthreads)

#     # init dirty
#     if model.any():
#         dirty = residual + psfo.convolve(model)
#     else:
#         dirty = residual

#     # residual
#     residual_mfs = np.sum(residual, axis=0)
#     rms = np.std(residual_mfs)
#     rmax = np.abs(residual_mfs).max()



    
#     return model, residual