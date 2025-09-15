import numpy as np
from regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm
from regularization.subfunc.l_corner import l_corner
def Lcurve(dat_noisy, A, Lambda):
    nLambda = len(Lambda)
    m = A.shape[1]
    n = A.shape[0]
    X = np.zeros((m, nLambda))
    rho = np.zeros(nLambda)
    eta = np.zeros(nLambda)
    
    for kk in range(nLambda):
        X[:, kk], rho[kk], eta[kk] = nonnegtik_hnorm(A, dat_noisy, Lambda[kk], '0')
    
    # Lcurve selection method
    reg_c, rho_c, eta_c = l_corner(rho, eta, np.flipud(Lambda))
    f_rec_LC = nonnegtik_hnorm(A, dat_noisy, reg_c, '0')
    
    return f_rec_LC, reg_c
