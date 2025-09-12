#TIKHONOV Tikhonov regularization.

# [x_lambda,rho,eta] = tikhonov(U,s,V,b,lambda,x_0)
# [x_lambda,rho,eta] = tikhonov(U,sm,X,b,lambda,x_0) ,  sm = [sigma,mu]

# Computes the Tikhonov regularized solution x_lambda, given the SVD or
# GSVD as computed via csvd or cgsvd, respectively.  If the SVD is used,
# i.e. if U, s, and V are specified, then standard-form regularization
# is applied:
#    min { || A x - b ||^2 + lambda^2 || x - x_0 ||^2 } .
# If, on the other hand, the GSVD is used, i.e. if U, sm, and X are
# specified, then general-form regularization is applied:
#   min { || A x - b ||^2 + lambda^2 || L (x - x_0) ||^2 } .

#If an initial estimate x_0 is not specified, then x_0 = 0 is used.

# Note that x_0 cannot be used if A is underdetermined and L ~= I.

# If lambda is a vector, then x_lambda is a matrix such that
#    x_lambda = [ x_lambda(1), x_lambda(2), ... ] .

#The solution norm (standard-form case) or seminorm (general-form
#case) and the residual norm are returned in eta and rho.

#Per Christian Hansen, DTU Compute, April 14, 2003.

#Reference: A. N. Tikhonov & V. Y. Arsenin, "Solutions of Ill-Posed
#Problems", Wiley, 1977.
import numpy as np
from numpy.linalg import norm

def tikhonov(U, s, V, b, lambda_val, x_0=None, nargin=None, nargout=None):
    if (np.min(lambda_val) < 0):
        raise ValueError ("Illegal regularization parameter lambda")
    m = U.shape[0]
    n = V.shape[0]
    if s.ndim == 1:
        s = s.reshape(-1,1)
    p, ps = s.shape
    lambda_val = [lambda_val]
    beta = np.dot(U[:, :p].T, b)
    zeta = s[:, 0] * beta
    ll = len(lambda_val)
    x_lambda = np.zeros((n, ll))
    rho = np.zeros((ll,1))
    eta = np.zeros((ll,1))
    s = s.flatten()
    if (ps == 1):
        if (nargin == 6):
            omega = V.T @ x_0
            # check if @ or *
        for i in range(ll):
            if (nargin == 5):
                x_lambda[:,i] = V[:,:p] @ (zeta/(s**2 + lambda_val[i]**2))
                rho[i] = lambda_val[i]**2 * norm(beta/(s**2 + lambda_val[i]**2))
            else:
                x_lambda[:,i] = V[:,:p] @ ((zeta + lambda_val[i]**2 @ omega / (s**2 + lambda_val[i]**2)))
                rho[i] = lambda_val[i]**2 + norm((beta - s * omega)/ (s **2 + lambda_val[i]**2))
            eta[i] = norm(x_lambda[:,i])
        if (nargout > 1 and U.shape[0] > p):
            rho = np.sqrt(rho**2 + norm(b - U[:,:n] @ np.vstack((beta, (U[:, p:n].T @ b)))) ** 2)
    elif (m >= n):
        gamma2 = (s[:,0] / s[:,1])**2
        if (nargin == 6):
            omega = np.linalg.solve(V,x_0)
            omega = omega[:p]
        if (p ==n):
            x0 = np.zeros(n)
        else:
            x0 = V[:,p:n]
        for i in range(ll):
            if (nargin==5):
                xi = zeta / (s[:,0]**2 + lambda_val[i]**2 @ s[:,1]**2)
                x_lambda[:,i] = V[:,:p] @ xi + x0
                rho[i] = lambda_val[i]**2 * norm(beta/(gamma2 + lambda_val[i]**2))
            else:
                xi = (zeta + lambda_val[i]**2 @ (s[:,1] ** 2) * omega) / (s[:,0] ** 2 + lambda_val[i]**2 @ s[:,1]**2)
                x_lambda[:,i] = V[:,:p] @ xi + x0
                rho[i] = lambda_val[i]**2 * norm((beta-s[:,0] * omega) / (gamma2 + lambda_val[i]**2))
            eta[i] = norm(s[:,1] * xi)
        if (nargout > 1 and U.shape[0] > p):
            rho = np.sqrt(rho**2 + norm(b - U[:,:n] @ np.vstack((beta, U[:,p:n].T @ b)))**2)
    else:
        gamma2 = (s[:, 0] / s[:, 1])**2
        if nargin == 6:
            raise ValueError('x_0 not allowed')
        if p == m:
            x0 = np.zeros(n)
        else:
            x0 = V[:, p:m] @ U[:, p:m].T @ b

        for i in range(ll):
            xi = zeta / (s[:, 0]**2 + lambda_val[i]**2 @ s[:, 1]**2)
            x_lambda[:, i] = V[:, :p]@ xi + x0
            rho[i] = lambda_val[i]**2 * norm(beta / (gamma2 + lambda_val[i]**2))
            eta[i] = norm(s[:, 1] * xi)
    x_lambda = x_lambda.flatten()
    return x_lambda, rho, eta

