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

import numpy as np

# def GCV_NNLS(dat_noisy, A, Lambda):
#     # Get dimensions
#     nLambda = len(Lambda)
#     m = A.shape[1]
#     n = A.shape[0]

#     # Pre-allocate arrays
#     X = np.zeros((m, nLambda))
#     rho = np.zeros(nLambda)
    
#     # Compute solutions and residuals for each lambda
#     # X, mdl_err, xnorm = nonnegtik_hnorm_batch(A, dat_noisy, Lambda, nm='0')
#     X, mdl_err, xnorm = nonnegtik_hnorm_batch(A, dat_noisy, Lambda, nm='0')
    
#     # Calculate Alambda for all Lambda values
#     I_n = np.eye(n)
#     I_m = np.eye(m)
#     A_T_A = A.T @ A
    
#     # Compute the matrix to invert for each Lambda
#     Lambda_matrix = Lambda[:, np.newaxis] * I_m
#     A_T_A_plus_Lambda = A_T_A[:, :, np.newaxis] + Lambda_matrix
#     A_T_A_plus_Lambda_inv = np.linalg.pinv(A_T_A_plus_Lambda)
    
#     # Compute Alambda
#     term = A @ A_T_A_plus_Lambda_inv @ A.T
#     Alambda = np.trace(I_n - term, axis1=0, axis2=1)
    
#     # Compute GCV values
#     GCV = n * rho / (Alambda ** 2)

#     # Find index of minimum GCV
#     id_GCV = np.argmin(GCV)
    
#     # Select the best lambda and corresponding solution
#     f_rec_GCV = X[:, id_GCV]
#     lambda_val = Lambda[id_GCV]

#     return f_rec_GCV, lambda_val


# def GCV_NNLS_mod(dat_noisy, A, Lambda):
#     # This code implements GCV for regularized NNLS method

#     nLambda = len(Lambda)
#     m = A.shape[1]
#     n = A.shape[0]
#     X = np.zeros((m, nLambda))
#     rho = np.zeros(nLambda)

#     for kk in range(len(Lambda)):
#         X[:, kk], rho[kk], trash = nonnegtik_hnorm(A, dat_noisy, Lambda[kk], '0', nargin = 4)

#     # GCV selection method
#     GCV = np.zeros(nLambda)
#     for ij in range(nLambda):
#         # A(lambda) = trace(I - A*(A'*A + sqrt(Lambda(j))*I)^(-1)*A')
#         # as denominator
#         Alambda = np.trace(np.eye(n) - A @ np.linalg.pinv(A.T @ A + Lambda[ij] * np.eye(m)) @ A.T)
#         # gcv(lambda) = n * RSS/denom^2;
#         GCV[ij] = n * rho[ij] / Alambda**2

#     id_GCV = GCV == np.min(GCV)
#     f_rec_GCV = X[:, id_GCV]
#     lambda_val = Lambda[id_GCV]

#     return f_rec_GCV, lambda_val
