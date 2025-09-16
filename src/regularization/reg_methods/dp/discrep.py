# %DISCREP Discrepancy principle criterion for choosing the reg. parameter.
# %
# % [x_delta,lambda] = discrep(U,s,V,b,delta,x_0)
# % [x_delta,lambda] = discrep(U,sm,X,b,delta,x_0)  ,  sm = [sigma,mu]
# %
# % Least squares minimization with a quadratic inequality constraint:
# %    min || x - x_0 ||       subject to   || A x - b || <= delta
# %    min || L (x - x_0) ||   subject to   || A x - b || <= delta
# % where x_0 is an initial guess of the solution, and delta is a
# % positive constant.  Requires either the compact SVD of A saved as
# % U, s, and V, or part of the GSVD of (A,L) saved as U, sm, and X.
# % The regularization parameter lambda is also returned.
# %
# % If delta is a vector, then x_delta is a matrix such that
# %    x_delta = [ x_delta(1), x_delta(2), ... ] .
# %
# % If x_0 is not specified, x_0 = 0 is used.

# % Reference: V. A. Morozov, "Methods for Solving Incorrectly Posed
# % Problems", Springer, 1984; Chapter 26.

# % Per Christian Hansen, IMM, August 6, 2007.
import numpy as np
from numpy.linalg import norm
from src.regularization.subfunc.newton import newton

def discrep(U,s,V,b,delta,x_0, nargin):
    m = U.shape[0]
    n = V.shape[0]
    p,ps = s.reshape(-1,1).shape
    if s.ndim == 2:
        p = s.shape[1]
    # print("p shape", p)

    delta = [delta]
    ld = len(delta)
    x_delta = np.zeros((n,ld))
    lamb_da = np.zeros(ld)
    rho = np.zeros(p)

    if (min(delta) < 0):
        raise ValueError ("Illegal inequality constraint delta")
    
    if (nargin == 5):
        x_0 = np.zeros(n)
    if (ps == 1):
        omega = V.T @ x_0
    else:
        #check
        omega = np.linalg.solve(V, x_0)
    
    beta = U.T @ b
    if (ps == 1):
        delta_0 = norm(b - U @ beta)
        rho[p-1] = delta_0 ** 2
        for i in range(p, 1, -1):
            rho[i-2] = rho[i-1] + (beta[i-1] - s[i-1] * omega[i-1])**2
    else:
        delta_0 = norm (b - U @ beta)
        rho[0] = delta_0 ** 2
        for i in range(p-1):
            #not sure about this
            rho[i+1] = rho[i] + (beta[i] - s[i] * omega[i])**2
    
    if (min(delta) < delta_0):
        raise ValueError("Irrelevant delta < || (I - U*U'')*b ||")
    
    if (ps == 1):
        s2 = s**2
        for k in range(ld):
            if (delta[k]**2 >= norm(beta-s*omega)**2 + delta_0**2):
                x_delta[:,k] = x_0
            else:
                dummy = np.min(np.abs(rho - delta[k]**2))
                kmin = np.argmin(np.abs(rho - delta[k]**2))
                lambda_0 = s[kmin]
                lamb_da[k] = newton(lambda_0, delta[k], s, beta,omega, delta_0)
                e = s/ (s2 + lamb_da[k]**2)
                f = s * e
                x_delta[:,k] = V[:,:p] @ (e * beta + (1-f) * omega)
    elif (m >=n):
        omega = omega[:p]
        gamma = s[:,0]/s[:,1]
        x_u = V[:,p:n] @ beta[p:n]
        for k in range(ld):
            if (delta[k]**2 >= norm(beta[:p] - s[:,0] * omega)**2 + delta_0**2):
                x_delta[:,k] = V @ np.concatenate((omega, U[:,p:n].T @ b))
            else:
                dummy = np.min(np.abs(rho - delta[k]**2))
                kmin = np.argmin(np.abs(rho - delta[k]**2))
                lambda_0 = gamma[kmin]
                lamb_da[k] = newton(lambda_0, delta[k], s, beta[:p], omega, delta_0)
                e = gamma/(gamma**2 + lamb_da[k]**2)
                f = gamma * e
                x_delta[:,k] = V[:,:p] @ (e * beta[:p]/s[:,1] + (1-f)*s[:,1]*omega) + x_u
    else:
        omega = omega[:p]
        gamma = s[:,0]/s[:,1]
        x_u = V[:,p:m] @ beta[p:m]
        for k in range(ld):
            if (delta[k]**2 >= norm(beta[:p] - s[:,0] *omega)**2 + delta_0**2):
                x_delta[:,k] = V @ np.concatenate((omega, U[:, p:m].conj().T @ b))
            else:
                dummy = np.min(np.abs(rho - delta[k]**2))
                kmin = np.argmin(np.abs(rho - delta[k]**2))
                lambda_0 = gamma[kmin]
                lamb_da[k] = newton(lambda_0, delta[k], s, beta[:p], omega, delta_0)
                e = gamma/(gamma **2 + lamb_da[k]**2)
                f =gamma * e
                x_delta[:,k] = V[:, :p] @ (e * beta[:p]/s[:,1] + (1-f)*s[:,1]*omega) + x_u
    
    return x_delta, lamb_da
