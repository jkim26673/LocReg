import numpy as np
from pylops import Identity
# from trips.utilities.reg_param.gcv import *
from collections.abc import Iterable

# !/usr/bin/env python
"""
Builds functions for generalized cross validation
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford"
__affiliations__ = 'MIT and Tufts University, University of Bath, Arizona State University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com;"


from pylops import LinearOperator
# from trips.utilities.operators_old import *
from scipy import linalg as la
# from scipy.sparse._arrays import _sparray
from scipy import sparse

import numpy as np

from pylops import Identity, LinearOperator

"""
Utility functions.
"""
from venv import create
import numpy as np
import pylops
from scipy.ndimage import convolve
from scipy import sparse
import scipy
from scipy import linalg as la

"""regularization operators (derivatives)"""
## First derivative operator 1D
def gen_first_derivative_operator(n):
    D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None)
    L = sparse.identity(n)-D
    Lx = L[0:-1, :]
    return Lx
## First derivative operator 2D
def gen_first_derivative_operator_2D(nx, ny):
    D_x = gen_first_derivative_operator(nx)
    D_y = gen_first_derivative_operator(ny)
    IDx = sparse.kron( sparse.identity(nx), D_x)
    DyI = sparse.kron(D_y, sparse.identity(ny))
    L = sparse.vstack((IDx, DyI))
    return L

## Space time detivative operator
def gen_spacetime_derivative_operator(nx, ny, nt):
    D_spatial = gen_first_derivative_operator_2D(nx,ny)
    Lt = gen_first_derivative_operator(nt)
    ITLs = sparse.kron(sparse.identity(nt), D_spatial)
    LTIN = sparse.kron(Lt, sparse.identity(nx**2))
    L =  sparse.vstack((ITLs, LTIN))
    return L    

## Framelet operator
"""Framelet operators"""

def construct_H(l,n):

    e = np.ones((n,))
    # build H_0
    H_0 = sparse.spdiags(e, -1-l+1, n, n) + sparse.spdiags(2*e, 0, n, n) + sparse.spdiags(e, 1+l-1, n, n)
    H_0 = H_0.tocsr()
    for jj in range(0,l):
        H_0[jj, l-jj-1] += 1
        H_0[-jj-1, -l+jj] += 1
    H_0 /= 4
    # build H_1

    H_1 = sparse.spdiags(-e, -1-l+1, n, n) + sparse.spdiags(e, 1+l-1, n, n)
    H_1 = H_1.tocsr()

    for jj in range(0,l):
        H_1[jj, l-jj-1] -= 1
        H_1[-jj-1, -l+jj] += 1

    H_1 *= np.sqrt(2)/4

    # build H_2

    H_2 = sparse.spdiags(-e, -1-l+1, n, n) + sparse.spdiags(2*e, 0, n, n) + sparse.spdiags(-e, 1+l-1, n, n)
    H_2 = H_2.tocsr()

    for jj in range(0,l):
        H_2[jj, l-jj-1] -= 1
        H_2[-jj-1, -l+jj] -= 1

    H_2 /= 4

    return (H_0, H_1, H_2)


def create_analysis_operator_rec(n, level, l, w):

    if level == l:
        return sparse.vstack( construct_H(level, n) )

    else:
        (H_0, H_1, H_2) = construct_H(level, n)
        W_1 = create_analysis_operator_rec(n, level+1, l, H_0)

        return sparse.vstack( (W_1, H_1, H_2) ) * w


def create_analysis_operator(n, l):

    return create_analysis_operator_rec(n, 1, l, 1)


def create_framelet_operator(n,m,l):

    W_n = create_analysis_operator(n, l)
    W_m = create_analysis_operator(m, l)

    proj_forward = lambda x: (W_n @ (x.reshape(n,m, order='F') @ W_m.H)).reshape(-1,1, order='F')

    proj_backward = lambda x: (W_n.H @ (x.reshape( n*(2*l+1) , m*(2*l+1), order='F' ) @ W_m)).reshape(-1,1, order='F')

    W = pylops.FunctionOperator(proj_forward, proj_backward, n*(2*l+1) * m*(2*l+1), n*m)

    return W

""" 
Other operators
"""

def operator_qr(A):

    """
    Handles QR decomposition for an operator A of any form: dense or sparse array, or a pylops LinearOperator.
    """

    if isinstance(A, LinearOperator):
        return la.qr(A.todense(), mode='economic')
    else:
        return la.qr(A, mode='economic')
    

def operator_svd(A):

    """
    Handles QR decomposition for an operator A of any form: dense or sparse array, or a pylops LinearOperator.
    """

    if isinstance(A, LinearOperator):
        return la.svd(A.todense(), full_matrices=False)
    else:
        return la.svd(A, full_matrices=False)


def soft_thresh(x, mu):
    #y = np.sign(x)*np.max([np.abs(x)-mu], 0)
    y = np.abs(x) - mu
    y[y < 0] = 0
    y = y * np.sign(x)
    return y

def generate_noise(shape, noise_level, dist='normal'):
    """
    Produces noise at the desired noise level.
    """

    if dist == 'normal':
        noise = np.random.randn(shape)
    elif dist == 'poisson':
        noise = np.random.poisson
    e = noise_level * noise / la.norm(noise)


def is_identity(A):
    """
    Checks whether the operator A is identity.
    """

    if isinstance(A, Identity): # check if A is a pylops identity operator
        return True

    elif (not isinstance(A, LinearOperator)) and ( A.shape[0] == A.shape[1] ) and ( np.allclose(A, np.eye(A.shape[0])) ): # check if A is an array resembling the identity matrix
        return True
    
    # elif isinstance(A, _sparray) and ( A.shape[0] == A.shape[1] ) and ( A - sparse.eye(A.shape[0]) ).sum() < 10**(-6):
    #     return print("This is important")
    
    else:
        return False

def check_noise_type(noise_type):
    if noise_type in ['g', 'p', 'l', 'gaussian', 'Gaussian', 'Poisson', 'poisson', 'Laplace', 'laplace']:
        valid = True
    else:
        valid = False
    if not valid:
       raise TypeError('You must enter a valid name for the noise. For Gaussian noise input g or Gaussian or gaussian. For Poisson noise input p or Poisson or poisson. For Laplace noise input l or laplace or laplace.')

def check_noise_level(noise_level):
    valid  = False
    if (isinstance(noise_level, float) or isinstance(noise_level, int)):
        if int(noise_level) > 0 or int(noise_level) == 0:
            valid = True
    if not valid:
        raise TypeError('You must enter a valid noise level! Choose 0 for 0 %, 1 for 1%, or other valid values acordingly.')

def check_Regparam(Regparam = 1):
    valid = False
    case1 = False
    # if str(Regparam).isnumeric():
    if (isinstance(Regparam, float) or isinstance(Regparam, int)):
        if int(Regparam) > 0:
            valid = True
        else:
            valid = False
            case1 = True
    elif Regparam in ['gcv', 'GCV', 'Gcv', 'DP', 'dp', 'Dp', 'Discrepancy Principle', 'Discrepancy principle', 'discrepancy principle']:
        valid = True
    if not valid and case1 == True:
        raise TypeError("You must specify a valid regularization parameter. Input a positive number!")
    elif not valid:
        raise TypeError("You must specify a valid regularization parameter. For Generalized Cross Validation type 'gcv'. For 'Discrepancy Principle type 'dp'.")

def check_Positivescalar(value):
    if int(value) > 0:
        valid = True
    else:
        valid = False

def check_operator_type(A):
    aa = str(type(A))
    if 'array' in aa:
        A = A
    # elif 'sparse' in aa:
    else:
        A = A.todense()
    return A

# def check_imagesize_toreshape(existingimage, chooseimage, old_size, newsize):
#     path_package = '/Users/mirjetapasha/Documents/Research_Projects/TRIPS_June25/multiparameter_package'
#     if (old_size[0] != newsize[0] or old_size[1] != newsize[1]):
#         Deblur.plot_rec(existingimage.reshape((shape), order = 'F'), save_imgs = False)
#         temp_im = Image.open(path_package + '/demos/data/images/'+chooseimage+'_'+str(newsize[0])+'.jpg')
#         image_new =  np.array(temp_im.resize((newsize[0], newsize[1])))
#         spio.savemat(path_package + '/demos/data/images/'+chooseimage+'_'+str(newsize[0])+'.mat', mdict={'x_true': image_new})
#     return image_new


def get_input_image_size(image):
    imshape = image.shape
    if imshape[1] == 1:
        nx = int(np.sqrt(imshape[0]))
        ny = int(np.sqrt(imshape[0]))
    else:
        nx = imshape[0]
        ny = imshape[1]
    newshape = (nx, ny)
    return newshape


def check_if_vector(im, nx, ny):
    if im.shape[1] == 1:
        im_vec = im
    else:
        im_vec = im.reshape((nx*ny, 1)) 
    return im_vec

def image_to_new_size(image, n):
    X, Y = np.meshgrid(np.linspace(1, image.shape[1], n[0]), np.linspace(1, image.shape[0], n[1]))
    im = interp2linear(image, X, Y, extrapval=np.nan)
    return im

def interp2linear(z, xi, yi, extrapval=np.nan):
    """
    This function is obtained from this github repository: https://github.com/serge-m/pyinterp2 to be used for automatically reshaping the images
    __author__ = 'Sergey Matyunin'
    Linear interpolation equivalent to interp2(z, xi, yi,'linear') in MATLAB
    @param z: function defined on square lattice [0..width(z))X[0..height(z))
    @param xi: matrix of x coordinates where interpolation is required
    @param yi: matrix of y coordinates where interpolation is required
    @param extrapval: value for out of range positions. default is numpy.nan
    @return: interpolated values in [xi,yi] points
    @raise Exception:
    """
    x = xi.copy()
    y = yi.copy()
    nrows, ncols = z.shape
    if nrows < 2 or ncols < 2:
        raise Exception("z shape is too small")
    if not x.shape == y.shape:
        raise Exception("sizes of X indexes and Y-indexes must match")
    # find x values out of range
    x_bad = ( (x < 0) | (x > ncols - 1))
    if x_bad.any():
        x[x_bad] = 0
    # find y values out of range
    y_bad = ((y < 0) | (y > nrows - 1))
    if y_bad.any():
        y[y_bad] = 0
    # linear indexing. z must be in 'C' order
    ndx = np.floor(y) * ncols + np.floor(x)
    ndx = ndx.astype('int32')
    # fix parameters on x border
    d = (x == ncols - 1)
    x = (x - np.floor(x))
    if d.any():
        x[d] += 1
        ndx[d] -= 1
    # fix parameters on y border
    d = (y == nrows - 1)
    y = (y - np.floor(y))
    if d.any():
        y[d] += 1
        ndx[d] -= ncols
    # interpolate
    one_minus_t = 1 - y
    z = z.ravel()
    f = (z[ndx] * one_minus_t + z[ndx + ncols] * y ) * (1 - x) + (
        z[ndx + 1] * one_minus_t + z[ndx + ncols + 1] * y) * x
    # Set out of range positions to extrapval
    if x_bad.any():
        f[x_bad] = extrapval
    if y_bad.any():
        f[y_bad] = extrapval
    return f

import numpy as np 
from scipy.optimize import newton, minimize, nnls
import scipy.linalg as la
import scipy.optimize as op
from pylops import Identity, LinearOperator
# from ..utilities.utils import operator_qr, operator_svd, is_identity

"""
Generalized crossvalidation
"""

def gcv_numerator(reg_param, Q_A, R_A, R_L, b, **kwargs):

    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'

    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)
    
    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    # the inverse term:

    # print("(R_A_2 + reg_param * R_L_2).shape", (R_A_2 + reg_param * R_L_2).shape)
    # print('done pt 1')
    # print("(R_A.T).shape", (R_A.T).shape)
    # print("Q_A.T.shape", Q_A.T.shape)
    # print("b.shape",b.shape)
    # print('done pt 2')
    A1 = R_A_2 + reg_param * R_L_2
    # print("(R_A.T)", R_A.T)
    # print("(Q_A.T)", Q_A.T)
    # print("b",b)
    # Q_A = Q_A.matvec(Q_A.H(np.eye(Q_A.shape[0])))
    Q_A = np.eye(Q_A.shape[0])
    # print("Q_A.T.shape after mod", Q_A.T)
    # test1 = (R_A.T @ Q_A.T)
    # print("test1", test1.shape)
    # d = test1 @ b
    # print("d",d)
    # inverted = la.solve(A1, d)
    # inverted = nnls( ( R_A_2 + reg_param * R_L_2), (R_A.T @ Q_A.T @ b), maxiter = 400)[0]
    inverted = np.linalg.solve(( R_A_2 + reg_param * R_L_2), (R_A.T @ Q_A.T @ b))
    # inverted = np.linalg.lstsq(( R_A_2 + reg_param * R_L_2), (R_A.T @ Q_A.T @ b), rcond = None)[0]
    # return np.sqrt((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)
    if variant == 'modified':
        return ((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)
    # elif variant == "NNLS":
    #     return 
    else:
        return (np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2

def gcv_denominator(reg_param, R_A, R_L, b, **kwargs):

    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'
    # print(variant)
    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)

    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    inverted = la.solve( ( R_A_2 + reg_param * R_L_2), R_A.T )

    if variant == 'modified':
       m = kwargs['fullsize'] # probably this can be b.size -- NOT FOR HYBRID SOLVERS!
       # trace_term = (m - R_A.shape[1]) - np.trace(R_A @ inverted) # b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities 
       trace_term = m - np.trace(R_A @ inverted) # b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities 
    else:
        # in this way works even if we revert to the fully projected pb (call with Q_A.T@b)
        # trace_term = b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities
        trace_term = R_A.shape[1] - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities
    
    return trace_term**2


def generalized_crossvalidation(Q_A, R_A, R_L, b, **kwargs):

    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 10**(-12)

    if 'gcvtype' in kwargs:
        gcvtype = kwargs['gcvtype']
    else:
        gcvtype = 'tikhonov'
    
    # function to minimize
    # if gcvtype == 'tikhonov':    
    gcv_func = lambda reg_param: gcv_numerator(reg_param, Q_A, R_A, R_L, b) / gcv_denominator(reg_param, R_A, R_L, b, **kwargs)
    lambdah = op.fminbound(func = gcv_func, x1 = 1e-09, x2 = 1e2, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=0) ## should there be tol here?
    # elif gcvtype == 'tsvd':
    #     m = Q_A.shape[0]
    #     n = R_L.shape[1]
    #     gcv_vals = []
    #     bhat = Q_A.T@b
    #     f = np.ones((m,1))
    #     for i in range(n):
    #         f[n-(i+1),] = 0
    #         fvar = np.concatenate((1 - f[:n,], f[n:,]))
    #         coeff = (fvar*bhat)**2
    #         gcv_numerator = np.sum(coeff)
    #         gcv_denominator = (m - (n-(i+1)))**2
    #         gcv_vals.append(gcv_numerator/gcv_denominator)   
    #     lambdah = n - (gcv_vals.index(min(gcv_vals))+1)
    # elif gcvtype == 'tgsvd':
    #     m = Q_A.shape[0]
    #     n = R_L.shape[1]
    #     p = R_L.shape[0]
    #     gcv_vals = []
    #     bhat = Q_A.T@b
    #     f = np.ones((m,1))
    #     for i in range(n):
    #         f[i,] = 0
    #         fvar = np.concatenate((1 - f[:n,], f[n:,]))
    #         coeff = (fvar*bhat)**2
    #         gcv_numerator = np.sum(coeff)
    #         gcv_denominator = (m - (n-(i+1)) - (n-p))**2
    #         gcv_vals.append(gcv_numerator/gcv_denominator)   
    #     lambdah = gcv_vals.index(min(gcv_vals))
    
    return lambdah

"""
Definition of functions for Discrepancy principle
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, and Connor Sanderford"
__affiliations__ = 'MIT and Tufts University, University of Bath, Arizona State University,'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "mpasha@mit.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com;"

import numpy as np 
import scipy.linalg as la
# import operator_qr, operator_svd, is_identity
import warnings

def discrepancy_principle(Q, A, L, b, delta = None, eta = 1.01, **kwargs):

    if not ( isinstance(delta, float) or isinstance(delta, int)):
        raise Exception("""A value for the noise level delta was not provided and the discrepancy principle cannot be applied. 
                    Please supply a value of delta based on the estimated noise level of the problem, or choose the regularization parameter according to gcv.""")
    
    explicitProj = kwargs['explicitProj'] if ('explicitProj' in kwargs) else False

    if 'dptype' in kwargs:
        dptype = kwargs['dptype']
    else:
        dptype = 'tikhonov'

    if dptype == 'tikhonov':  
        bfull = b
        b = Q.T@b
        if is_identity(L):
            Anew = A
            bnew = b
        else:
            UL, SL, VL = la.svd(L)
            if L.shape[0] >= L.shape[1] and SL[-1] != 0:
                Anew = A@(VL.T@np.diag((SL)**(-1)))
                bnew = b
            elif L.shape[0] >= L.shape[1] and SL[-1] == 0:
                zeroind = np.where(SL == 0)
                W = VL[zeroind,:].reshape((-1,1))
                AW = A@W
                Q_AW, R_AW = np.linalg.qr(AW, mode='reduced')
                Q_LT, R_LT = np.linalg.qr(L.T, mode='reduced')
                LAwpinv = (np.eye(L.shape[1]) - (W@np.linalg.inv(R_AW)@Q_AW.T@A))@Q_LT@np.linalg.inv(R_LT.T)
                Anew = A@LAwpinv
                xnull = W@np.linalg.inv(R_AW)@Q_AW.T@b
                bnew = b - A@xnull
            elif (L.shape[0] < L.shape[1]):
                W = VL[L.shape[0]-L.shape[1]:,:].T
                AW = A@W
                Q_AW, R_AW = np.linalg.qr(AW, mode='reduced')
                Q_LT, R_LT = np.linalg.qr(L.T, mode='reduced')
                LAwpinv = (np.eye(L.shape[1]) - (W@np.linalg.inv(R_AW)@Q_AW.T@A))@Q_LT@np.linalg.inv(R_LT.T)
                Anew = A@LAwpinv
                xnull = W@np.linalg.inv(R_AW)@Q_AW.T@b
                bnew = b - A@xnull

        U, S, V = la.svd(Anew)
        singular_values = S**2
        bhat = U.T @ bnew
        if Anew.shape[0] > Anew.shape[1]:
            singular_values = np.append(singular_values.reshape((-1,1)), np.zeros((Anew.shape[0]-Anew.shape[1],1)))
            if explicitProj:
                testzero = la.norm(bhat[Anew.shape[1]-Anew.shape[0]:,:])**2 + la.norm(bfull - Q@b)**2 - (eta*delta)**2 # this is OK but need reorthogonalization
            else:
                testzero = la.norm(bhat[Anew.shape[1]-Anew.shape[0]:,:])**2 - (eta*delta)**2
        else:
            testzero = la.norm(bfull - Q@b)**2 - (eta*delta)**2
        singular_values.shape = (singular_values.shape[0], 1)
    
        beta = 1e-8
        iterations = 0

        if testzero < 0:
            while (iterations < 30) or ((iterations <= 100) and (np.abs(alpha) < 10**(-16))):
                zbeta = (((singular_values*beta + 1)**(-1))*bhat.reshape((-1,1))).reshape((-1,1))
                if explicitProj:
                    f = la.norm(zbeta)**2 + la.norm(bfull - Q@b)**2 - (eta*delta)**2 # this is OK but need reorthogonalization
                else:
                    f = la.norm(zbeta)**2 - (eta*delta)**2
                wbeta = (((singular_values*beta + 1)**(-1))*zbeta).reshape((-1,1))
                f_prime = 2/beta*zbeta.T@(wbeta - zbeta)

                beta_new = beta - f/f_prime

                if abs(beta_new - beta) < 10**(-12)* beta:
                    break

                beta = beta_new
                alpha = 1/beta_new[0,0]

                iterations += 1
        else:
            alpha = 0
    elif dptype == 'tsvd':
        m = Q.shape[0]
        n = L.shape[1]
        f = np.ones((m,1))
        bhat = Q.T@b
        alpha = n
        for i in range(n):
            f[n-(i+1),] = 0
            fvar = np.concatenate((1 - f[:n,], f[n:,]))
            coeff = (fvar*bhat)**2
            dp_val = np.sum(coeff) - (eta*delta)**2
            if dp_val < 0:
                alpha = n - (i+1)
            else:
                break
    elif dptype == 'tgsvd':
        m = Q.shape[0]
        n = L.shape[1]
        f = np.ones((m,1))
        bhat = Q.T@b
        alpha = n
        coeff = np.square(bhat)
        for i in range(n):
            coeff[n-(i+1),] = 0
            dp_val = np.sum(coeff) - (eta*delta)**2
            if dp_val >= 0:
                alpha = i
            else:
                break

    return alpha

def Tikhonov(A, b, L, x_true, regparam = 'gcv', **kwargs):
    A = A.todense() if isinstance(A, LinearOperator) else A
    L = L.todense() if isinstance(L, LinearOperator) else L
    if regparam in ['gcv', 'GCV', 'Gcv']:
        lambdah = generalized_crossvalidation(Identity(A.shape[0]), A, L, b) #, variant = 'modified', fullsize = A.shape[0], **kwargs)
    elif regparam in ['DP', 'dp', 'Dp', 'Discrepancy Principle', 'Discrepancy principle', 'discrepancy principle']:
        lambdah = discrepancy_principle(Identity(A.shape[0]), A, L, b, **kwargs) # find ideal lambdas by discrepancy principle
    else:
        lambdah = regparam
    xTikh = np.linalg.solve(A.T@A + lambdah*L.T@L, A.T@b)
    #comopare with tikhonov_vec.py...
    return xTikh, lambdah