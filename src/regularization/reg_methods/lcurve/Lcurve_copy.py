import numpy as np
from src.regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm
from src.regularization.subfunc.l_curve_corner import l_curve_corner

def Lcurve(dat_noisy, A, Lambda):
    #Takes in lambda**2
    nLambda = len(Lambda)
    m = A.shape[1]
    n = A.shape[0]
    
    X = np.zeros((m, nLambda))
    rho = np.zeros(nLambda)
    eta = np.zeros(nLambda)

    for kk in range(nLambda):
        X[:, kk], rho[kk], eta[kk] = nonnegtik_hnorm(A, dat_noisy, Lambda[kk], '0', nargin = 4)
    
    # X, mdl_err, xnorm = nonnegtik_hnorm_batch(A, dat_noisy, Lambda, nm='0')

    # #Check if Lambda is a 2D array, if not, reshape it to a column array
    # if Lambda.ndim == 1:
    #     Lambda = Lambda.reshape(-1,1)
    
    #Josh's addition after assuming that nonnegtik_knorm takes in lambda**2
    # Lambda = np.sqrt(Lambda)
    # Lcurve selection method
    reg_c, index, kappa = l_curve_corner(rho, eta, np.fliplr(Lambda))

    #assume reg_c must be reg_c**2; Josh's addition
    # reg_c = reg_c**2

    f_rec_LC, _, _ = nonnegtik_hnorm(A, dat_noisy, reg_c, '0', nargin =4)

    return f_rec_LC, reg_c