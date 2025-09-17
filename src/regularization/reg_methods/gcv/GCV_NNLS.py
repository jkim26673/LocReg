import numpy as np
from src.regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm

def GCV_NNLS(dat_noisy, A, Lambda):
    # This code implements GCV for regularized NNLS method

    nLambda = len(Lambda)
    m = A.shape[1]
    n = A.shape[0]
    X = np.zeros((m, nLambda))
    rho = np.zeros(nLambda)

    for kk in range(len(Lambda)):
        X[:, kk], rho[kk], trash = nonnegtik_hnorm(A, dat_noisy, Lambda[kk], '0', nargin = 4)

    # Lambda = np.sqrt(Lambda)
    # GCV selection method
    GCV = np.zeros(nLambda)
    for ij in range(nLambda):
        # A(lambda) = trace(I - A*(A'*A + sqrt(Lambda(j))*I)^(-1)*A')
        # as denominator
        Alambda = np.trace(np.eye(n) - A @ np.linalg.pinv(A.T @ A + Lambda[ij] * np.eye(m)) @ A.T)
        # gcv(lambda) = n * RSS/denom^2;
        GCV[ij] = n * rho[ij] / Alambda**2


    id_GCV = GCV == np.min(GCV)
    f_rec_GCV = X[:, id_GCV]
    lambda_val = Lambda[id_GCV]

    return f_rec_GCV, lambda_val

