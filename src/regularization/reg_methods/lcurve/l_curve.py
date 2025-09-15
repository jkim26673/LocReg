# %L_CURVE Plot the L-curve and find its "corner".
# %
# % [reg_corner,rho,eta,reg_param] =
# %                  l_curve(U,s,b,method)
# %                  l_curve(U,sm,b,method)  ,  sm = [sigma,mu]
# %                  l_curve(U,s,b,method,L,V)
# %
# % Plots the L-shaped curve of eta, the solution norm || x || or
# % semi-norm || L x ||, as a function of rho, the residual norm
# % || A x - b ||, for the following methods:
# %    method = 'Tikh'  : Tikhonov regularization   (solid line )
# %    method = 'tsvd'  : truncated SVD or GSVD     (o markers  )
# %    method = 'dsvd'  : damped SVD or GSVD        (dotted line)
# %    method = 'mtsvd' : modified TSVD             (x markers  )
# % The corresponding reg. parameters are returned in reg_param.  If no
# % method is specified then 'Tikh' is default.  For other methods use plot_lc.
# %
# % Note that 'Tikh', 'tsvd' and 'dsvd' require either U and s (standard-
# % form regularization) computed by the function csvd, or U and sm (general-
# % form regularization) computed by the function cgsvd, while 'mtvsd'
# % requires U and s as well as L and V computed by the function csvd.
# %
# % If any output arguments are specified, then the corner of the L-curve
# % is identified and the corresponding reg. parameter reg_corner is
# % returned.  Use routine l_corner if an upper bound on eta is required.

# % Reference: P. C. Hansen & D. P. O'Leary, "The use of the L-curve in
# % the regularization of discrete ill-posed problems",  SIAM J. Sci.
# % Comput. 14 (1993), pp. 1487-1503.

# % Per Christian Hansen, DTU Compute, October 27, 2010.

# % Set defaults.
import numpy as np
from regularization.subfunc.l_corner import l_corner
from regularization.subfunc.l_curve_corner import l_curve_corner
from numpy.linalg import norm

def l_curve(U,sm,b,method,L,V, nargin, nargout):
    if (nargin == 3):
        method = 'Tikh'
    npoints = 200
    smin_ratio = 16 * np.finfo(float).eps
    m,n = U.shape
    p,ps = sm.reshape(-1,1).shape
    if (nargout > 0):
        locate = 1
    else:
        locate = 0
    beta = U.T @ b
    #Python:
    #beta = array([-0.88619825, -0.77405807, -0.39090139, -0.14894758,  0.13425634,
    #   -0.02487103,  0.05534861, -0.0076849 , -0.03377109, -0.00150965])
    #Matlab:array([-0.88619825, -0.77405807, -0.39090139, -0.14894758,  0.13425634,
    #   -0.02487103,  0.05534861, -0.0076849 , -0.03377109, 0.000647345383])
    #Last value is different
    #
    
    beta2 = (norm(b)**2) - (norm(beta)**2)
    if (ps == 1):
        s = sm
        beta = beta[:p]
    else:
        #This part dsmoesn't work in the matlab code version
        #Error: index in position 2 exceeds array bounds (must not exceed 1).
        s = sm[p-1:1:-1,0] / sm[p-1:1:-1,1]
        beta = beta[p-1:1:-1]
    xi = beta[:p]/s
    xi[np.isinf(xi)] = 0

    if method.startswith('Tikh') or method.startswith('tikh'):
        eta = np.zeros(npoints)
        rho = np.zeros(npoints)
        reg_param = np.zeros(npoints)
        s2 = s**2
        reg_param[-1] = max([s[p-1], s[0] * smin_ratio])
        ratio = (s[0] / reg_param[npoints-1]) ** (1 / (npoints - 1))
        for i in range(npoints - 1, 0, -1):
            reg_param[i-1] = ratio * reg_param[i]
        for i in range(npoints):
            f = s2 / (s2 + (reg_param[i]) ** 2)
            #test = np.append(test,f)
            eta[i] = norm(f * xi)
            rho[i] = norm((1 - f) * beta[:p])
        if m > n and beta2 > 0:
            rho = np.sqrt(rho**2 + beta2)
        marker = '-'
        txt = 'Tikh.'
    elif method.startswith('tsvd') or method.startswith('tgsv'):
        eta = np.zeros(p)
        rho = np.zeros(p)
        eta[0] = np.abs(xi[0]) ** 2
        for k in range(1, p):
            eta[k] = eta[k - 1] + np.abs(xi[k])** 2
        eta = np.sqrt(eta)
        if m > n:
            if beta2 > 0:
                rho[p - 1] = beta2
            else:
                rho[p - 1] = (np.finfo(float).eps) ** 2
        else:
            rho[p - 1] = np.finfo(float).eps ** 2
        for k in range(p - 2, -1, -1):
            rho[k] = rho[k + 1] + np.abs(beta[k + 1]) ** 2
        rho = np.sqrt(rho)
        reg_param = np.arange(1, p+1). T
        marker = 'o'
        if ps == 1:
            U = U[:, :p]
            txt = 'TSVD'
        else:
            U = U[:, :p]
            txt = 'TGSVD'
    elif method.startswith('dsvd') or method.startswith('dgsv'):
        eta = np.zeros(npoints)
        rho = np.zeros(npoints)
        reg_param = np.zeros(npoints)
        reg_param[-1] = np.max([s[p - 1], s[0] * smin_ratio])
        ratio = (s[0] / reg_param[-1]) ** (1 / (npoints - 1))
        for i in range(npoints - 2, -1, -1):
            reg_param[i] = ratio * reg_param[i + 1]
        for i in range(npoints):
            f = s / (s + reg_param[i])
            eta[i] = np.linalg.norm(f * xi)
            rho[i] = np.linalg.norm((1 - f) * beta[:p])
        if m > n and beta2 > 0:
            rho = np.sqrt(rho ** 2 + beta2)
        marker = ':'
        if ps == 1:
            txt = 'DSVD'
        else:
            txt = 'DGSVD'
    elif method.startswith('mtsv'):
        if nargin != 6:
            raise ValueError('The matrices L and V must also be specified')

        p, n = L.shape
        rho = np.zeros(p)
        eta = np.zeros(p)

        Q, R = np.linalg.qr(L @ (V[:,n - 1:n-p-1:-1]), mode='reduced')
        
        for i in range(p):
            k = n - p + (i+1)
            Lxk = L @ (V[:, :k]) @ (xi[:k])
            zk = np.linalg.solve(R[:n - k, :n - k], (Q[:, :n - k].conj().T @ Lxk))
            zk = zk[n - k - 1::-1]
            eta[i] = np.linalg.norm((Q[:, n - k :p].conj().T) @ Lxk)
            #eta = np.copy(eta)
            if i < p - 1:
                rho[i] = np.linalg.norm(beta[k:n] + s[k:n] * zk)
                #rho = rho.copy()
            else:
                rho[i] = np.finfo(float).eps
                #rho = rho.copy()

        if m > n and beta2 > 0:
            rho = np.sqrt(rho ** 2 + beta2)

        reg_param = np.arange(n - p + 1, n + 1).conj().T
        txt = 'MTSVD'
        U = U[:, reg_param - 1]
        sm = sm[reg_param - 1]
        marker = 'x'
        ps = 2  # General form regularization.
    else:
        raise ValueError("Illegal method")

    if (locate):
        reg_corner,ireg_corner, kappa = l_curve_corner(rho,eta,reg_param)
    
    return reg_corner, rho, eta, reg_param
