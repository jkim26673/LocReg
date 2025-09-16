# %BLUR Test problem: digital image deblurring.
# %
# % function [A,b,x] = blur(N,band,sigma)
# %
# % The matrix A is an N*N-by-N*N symmetric, doubly block Toeplitz matrix that
# % models blurring of an N-by-N image by a Gaussian point spread function.
# % It is stored in sparse matrix format.
# %
# % In each Toeplitz block, only matrix elements within a distance band-1
# % from the diagonal are nonzero (i.e., band is the half-bandwidth).
# % If band is not specified, band = 3 is used.
# %
# % The parameter sigma controls the width of the Gaussian point spread
# % function and thus the amount of smoothing (the larger the sigma, the wider
# % the function and the more ill posed the problem).  If sigma is not
# % specified, sigma = 0.7 is used.
# %
# % The vector x is a columnwise stacked version of a simple test image, while
# % b holds a columnwise stacked version of the blurrred image; i.e, b = A*x.

# % Per Christian Hansen, IMM, 11/11/97.

from scipy.linalg import toeplitz
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import kron, spdiags, triu
import math

def blur(N =50,sigma = 5, band = 5, nargin = 1, nargout = 3):
    if (nargin < 2):
        band = 5
    band = min(band,N)
    if (nargin < 3):
        sigma = 5
    z_gaussian = np.exp(-((np.arange(band) ** 2) / (2 * sigma ** 2)))
    z = np.concatenate((z_gaussian, np.zeros(N - band)))
    A = toeplitz(z)
    A = csc_matrix(A)
    A = (1/(2*np.pi* (sigma**2))) * kron(A,A)

    def normal_round(n):
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)
    
    if (nargout > 1):
        x = np.zeros((N,N))
        N2 =  normal_round(N/2)
        N3 =  normal_round(N/3)
        N6 =  normal_round(N/6)
        N12 = normal_round(N/12)

        T = np.zeros((N6,N3))
        for i in np.arange(1,N6 + 1):
            for j in np.arange(1,N3 + 1):
                if ((i/N6)**2 + (j/N3)**2 < 1):
                    T[i-1,j-1] = 1
        
        T = np.concatenate((np.fliplr(T), T), axis=1)
        T = np.concatenate((T[::-1,:], T), axis=0)
        
        x[(2+1)-1:2+2*N6, (N3-1+1)-1:N3-1+2*N3] = T

        T = np.zeros((N6,N3))
        for i in np.arange(1,N6 + 1):
            for j in np.arange(1,N3 + 1):
                if ((i/N6)**2 + (j/N3)**2 < 0.6):
                    T[i-1,j-1] = 1
        
        T = np.concatenate((np.fliplr(T), T), axis=1)
        T = np.concatenate((T[::-1,:], T), axis=0)

        x[N6+1-1: N6+2*N6, N3-1+1-1: N3-1+2*N3] += 2 * T

        f = np.where(x == 3)
        x[f] = 2 * np.ones_like(x[f])

        T = np.triu(np.ones((N3,N3)))
        mT,nT = T.shape
        x[N3 + N12 +1-1 : N3 + N12 + nT, 1+1-1 : 1 + mT] = 3 * T

        T = np.zeros((2*N6 + 1, 2*N6 + 1))
        mT, nT = T.shape
        T[N6,:nT] = 1
        T[:mT, N6] = 1
        
        # new_shape = (max(N2+N12,N2+N12+mT),max(N2,N2+nT))
        # x.resize(new_shape)
        x[(N2+N12): (N2+N12+mT), N2:(N2+nT)] = 4*T
        x = x[:N,:N].flatten(order = 'F')
        x = x.reshape(N**2, 1)
        # test = x[:N,:N].reshape(N**2, 1)
        b = A @ x
    
    return A,b.flatten(),x.flatten()








