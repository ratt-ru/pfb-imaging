import numpy as np
import pywt

def set_Psi(nx, ny,
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
    # construct Psi
    P = len(basis)
    sqrtP = np.sqrt(P)

    def Psi_func(alpha, base):
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
        arrays. Replicates the combined sararec2(vec2coeff(x)) of
        original SARA class.

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

    # add dictionary entry for each basis element
    Psi = {}
    for i in range(P):
        Psi[i] = lambda x, b=basis[i]: Psi_func(x, b)
        
    # Construct Psi_t
    def Psi_t_func(x, level, base):
        """
        This implements the adjoint of Psi_func. Replicates the
        combination of coef2vec(saradec2(x)) of original SARA class.
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

    # add dictionary entry for each basis element
    Psi_t = {}
    for i in range(P):
        Psi_t[i] = lambda x, b=basis[i]: Psi_t_func(x, nlevels, b)        
    
    return Psi, Psi_t