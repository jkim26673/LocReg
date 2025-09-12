import numpy as np

# %TIKHONOV Tikhonov regularization.
# %
# % [x_lambda,rho,eta] = tikhonov(U,s,V,b,lambda,x_0)
# % [x_lambda,rho,eta] = tikhonov(U,sm,X,b,lambda,x_0) ,  sm = [sigma,mu]
# %
# % Computes the Tikhonov regularized solution x_lambda, given the SVD or
# % GSVD as computed via csvd or cgsvd, respectively.  If the SVD is used,
# % i.e. if U, s, and V are specified, then standard-form regularization
# % is applied:
# %    min { || A x - b ||^2 + lambda^2 || x - x_0 ||^2 } .
# % If, on the other hand, the GSVD is used, i.e. if U, sm, and X are
# % specified, then general-form regularization is applied:
# %    min { || A x - b ||^2 + lambda^2 || L (x - x_0) ||^2 } .
# %
# % If an initial estimate x_0 is not specified, then x_0 = 0 is used.
# %
# % Note that x_0 cannot be used if A is underdetermined and L ~= I.
# %
# % If lambda is a vector, then x_lambda is a matrix such that
# %    x_lambda = [ x_lambda(1), x_lambda(2), ... ] .
# %
# % The solution norm (standard-form case) or seminorm (general-form
# % case) and the residual norm are returned in eta and rho.

# % Per Christian Hansen, DTU Compute, April 14, 2003.

# % Reference: A. N. Tikhonov & V. Y. Arsenin, "Solutions of Ill-Posed
# % Problems", Wiley, 1977.

# Initialization.

def tikhonov_vec(U, s, V, data_noisy, lamb_da, x_0, nargin):
    m = U.shape[0]
    n = V.shape[0]
    s = s.reshape(-1,1)
    p, ps = s.shape
    beta = U[:, :p].T @ data_noisy
    zeta = s[:, 0] * beta
    s = s.flatten()
    ll = 1
    x_lambda = np.zeros((n,ll))
    # print("V.shape", V.shape)
    # print("p shape", p)
    # print("x_0 shape", x_0)

    #Treat each lambda separately
    if ps == 1:
        #The standard-form case.
        if nargin == 6:
            omega = V.T @ x_0
        for i in range(ll):
            if nargin == 5:
                #value = (V[:,:p] @ (zeta / (s ** 2 + lamb_da ** 2)))
                x_lambda[:, i] = (V[:,:p] @ (zeta / (s ** 2 + lamb_da ** 2)))
                x_lambda= x_lambda.flatten()
                # print("This before the loop")
                # print("x_lambda:", x_lambda.shape)
            else:
                x_lambda[:, i] = (V[:, :p] @ ((zeta + (lamb_da ** 2) @ omega) / (s ** 2 + lamb_da ** 2)))
                x_lambda= x_lambda.flatten()
        # x_lambda = x_lambda.flatten()
        # print("This after the loop")
        # print("x_lambda:", x_lambda.shape)
    return x_lambda, lamb_da