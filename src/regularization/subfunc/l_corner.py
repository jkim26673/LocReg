import numpy as np
from scipy.optimize import minimize_scalar
from src.regularization.subfunc.lcfun import lcfun 
from src.regularization.subfunc.fnder import fnder
from src.regularization.subfunc.corner import corner
from src.regularization.subfunc.sp2pp import sp2pp
from src.regularization.subfunc.spleval import spleval
from src.regularization.subfunc.spmak import spmak
from src.regularization.subfunc.ppbrk import ppbrk

def l_corner(rho, eta, reg_param, nargin, U=None, s=None, b=None, method=None, M=None):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    # Ensure that rho and eta are column vectors.
    rho = rho.flatten()
    eta = eta.flatten()

    # Set default regularization method.
    if (nargin <= 3):
        method = 'none'
        if nargin == 2:
            reg_param = np.arange(1, len(rho) + 1).T
    else:
        if nargin == 6:
            method = 'Tikh'

    # Set this logical variable to 1 (true) if the corner algorithm
    # should always be used, even if the Spline Toolbox is available.
    alwayscorner = False

    # Set threshold for skipping very small singular values in the
    # analysis of a discrete L-curve.
    s_thr = np.finfo(float).eps  # Neglect singular values less than s_thr.

    # Set default parameters for treatment of discrete L-curve.
    deg = 2  # Degree of local smoothing polynomial.
    q = 2  # Half-width of local smoothing interval.
    order = 4  # Order of fitting 2-D spline curve.

    # Initialization.
    if len(rho) < order:
        raise ValueError('Too few data points for L-curve analysis')

    if nargin > 3:
        p, ps = s.reshape(-1,1).shape
        m, n = U.shape

        beta = U.T @ b
        b0 = b - (U @ beta)

        if ps == 2:
            s = s[::-1, 0] / s[::-1, 1]
            beta = beta[::-1]

        xi = beta / s

        if m > n:  # Take care of the least-squares residual.
            beta = np.concatenate((beta, np.linalg.norm(b0)))

    # Restrict the analysis of the L-curve according to M (if specified).
    if nargin == 8:
        index = np.where(eta < M)[0]
        rho = rho[index]
        eta = eta[index]
        reg_param = reg_param[index]

    if method.startswith('Tikh') or method.startswith('tikh'):
        # The L-curve is differentiable; computation of curvature in
        # log-log scale is easy.

        # Compute g = - curvature of L-curve.
        g = lcfun(reg_param, s, beta, xi, nargin = 4)

        # Locate the corner. If the curvature is negative everywhere,
        # then define the leftmost point of the L-curve as the corner.
        gi = np.argmin(g)
        reg_c = fminbound(lambda x: lcfun(x, s, beta, xi, nargin = 4), reg_param[min(gi + 2, len(g)) - 1],
                          reg_param[max(gi, 1) - 1])
        kappa_max = -lcfun(reg_c, s, beta, xi, nargin = 4)

        if kappa_max < 0:
            lr = len(rho)
            reg_c = reg_param[lr]
            rho_c = rho[lr]
            eta_c = eta[lr]
        else:
            f = (s ** 2) / (s ** 2 + reg_c ** 2)
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1 - f) * beta[:len(f)])

            if m > n:
                rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0) ** 2)

    elif method.startswith('tsvd') or method.startswith('tgsv') or \
            method.startswith('mtsv') or method.startswith('none'):
        # Use the adaptive pruning algorithm to find the corner if the
        # Spline Toolbox is not available.
        if not alwayscorner:
            # error('The Spline Toolbox in not available so l_corner cannot be used')
            reg_c = corner(rho, eta)
            rho_c = rho[reg_c]
            eta_c = eta[reg_c]
            return reg_c, rho_c, eta_c
        # try:
        # # Check if the corner function is available
        #     reg_c = corner(rho, eta)
        #     rho_c = rho[reg_c]
        #     eta_c = eta[reg_c]
        #     return
        # except NameError:
        #     # If corner function is not available, or alwayscorner is True, perform other operations
        #     pass

        # Otherwise use local smoothing followed by fitting a 2-D spline curve
        # to the smoothed discrete L-curve. Restrict the analysis of the L-curve
        # according to s_thr.
        if nargin > 3:
            if total_num_args == 8:  # In case the bound M is in action.
                s = s[index]

            index = np.where(s > s_thr)[0]
            rho = rho[index]
            eta = eta[index]
            reg_param = reg_param[index]

        # Convert to logarithms.
        lr = len(rho)
        lrho = np.log(rho)
        leta = np.log(eta)
        slrho = lrho.copy()
        sleta = leta.copy()

        # For all interior points k = q+1:length(rho)-q-1 on the discrete
        # L-curve, perform local smoothing with a polynomial of degree deg
        # to the points k-q:k+q.
        v = np.arange(-q, q + 1)
        A = np.column_stack([np.ones(len(v))] + [np.power(v, j) for j in range(1, deg + 1)])

        for k in range(q, lr - q - 1):
            cr = np.linalg.lstsq(A, lrho[k + v], rcond=None)[0]
            slrho[k] = cr[0]

            ce = np.linalg.lstsq(A, leta[k + v], rcond=None)[0]
            sleta[k] = ce[0]

        # Fit a 2-D spline curve to the smoothed discrete L-curve.
        lrp1 = lr + order
        lr_range = np.arange(1, lrp1)

        sp = spmak(lr_range, np.column_stack([slrho, sleta]).T)
        pp = ppbrk(sp2pp(sp), [4, lrp1])

        # Extract abscissa and ordinate splines and differentiate them.
        # Compute as many function values as default in spleval.
        P = spleval(pp)
        dpp = fnder(pp)
        D = spleval(dpp)
        ddpp = fnder(pp, 2)
        DD = spleval(ddpp)

        ppx = P[0]
        ppy = P[1]
        dppx = D[0]
        dppy = D[1]
        ddppx = DD[0]
        ddppy = DD[1]

        # Compute the corner of the discretized spline curve via max. curvature.
        # No need to refine this corner since the final regularization
        # parameter is discrete anyway.
        # Define curvature = 0 where both dppx and dppy are zero.
        k1 = dppx * ddppy - ddppx * dppy
        k2 = np.power(dppx, 2) + np.power(dppy, 2)
        I_nz = np.where(k2 != 0)[0]
        kappa = np.zeros(len(dppx))
        kappa[I_nz] = -k1[I_nz] / k2[I_nz]

        ikmax = np.argmax(kappa)
        x_corner = ppx[ikmax]
        y_corner = ppy[ikmax]

        # Locate the point on the discrete L-curve which is closest to the
        # corner of the spline curve. Prefer a point below and to the
        # left of the corner. If the curvature is negative everywhere,
        # then define the leftmost point of the L-curve as the corner.
        if kappa_max < 0:
            reg_c = reg_param[lr]
            rho_c = rho[lr]
            eta_c = eta[lr]
        else:
            index = np.where((lrho < x_corner) & (leta < y_corner))[0]
            if len(index) != 0:
                rpi = index[np.argmin(np.power(lrho[index] - x_corner, 2) + np.power(leta[index] - y_corner, 2))]
            else:
                rpi = np.argmin(np.power(lrho - x_corner, 2) + np.power(leta - y_corner, 2))

            reg_c = reg_param[rpi]
            rho_c = rho[rpi]
            eta_c = eta[rpi]

    elif method.startswith('dsvd') or method.startswith('dgsv'):
        # The L-curve is differentiable; computation of curvature in
        # log-log scale is easy.

        # Compute g = - curvature of L-curve.
        g = lcfun(reg_param, s, beta, xi, 1)

        # Locate the corner. If the curvature is negative everywhere,
        # then define the leftmost point of the L-curve as the corner.
        gi = np.argmin(g)
        reg_c = fminbound(lambda x: lcfun(x, s, beta, xi, 1), reg_param[min(gi + 1, len(g))],
                          reg_param[max(gi - 1, 1)], xtol=1e-6, full_output=False)
        kappa_max = -lcfun(reg_c, s, beta, xi, 1)

        if kappa_max < 0:
            lr = len(rho)
            reg_c = reg_param[lr]
            rho_c = rho[lr]
            eta_c = eta[lr]
        else:
            f = s / (s + reg_c)
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1 - f) * beta[:len(f)])

            if m > n:
                rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0) ** 2)

    else:
        raise ValueError('Illegal method')

    return reg_c, rho_c, eta_c

import numpy as np
from scipy.optimize import fminbound

def l_corner(rho, eta, reg_param, U=None, s=None, b=None, method='none', M=None):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    # Ensure that rho and eta are column vectors.
    rho = rho.reshape(-1, 1)
    eta = eta.reshape(-1, 1)

    num_args = len(args)
    num_named_args = len(kwargs)
    total_num_args = num_args + num_named_args

    # Set default regularization method.
    if total_num_args <= 3:
        method = 'none'
        if total_num_args == 2:
            reg_param = np.arange(1, len(rho) + 1).reshape(-1, 1)
    else:
        if total_num_args == 6:
            method = 'Tikh'

    # Set this logical variable to True if the corner algorithm
    # should always be used, even if the Spline Toolbox is available.
    alwayscorner = False

    # Set threshold for skipping very small singular values in the
    # analysis of a discrete L-curve.
    s_thr = np.finfo(float).eps  # Neglect singular values less than s_thr.

    # Set default parameters for treatment of discrete L-curve.
    deg = 2  # Degree of local smoothing polynomial.
    q = 2    # Half-width of local smoothing interval.
    order = 4  # Order of fitting 2-D spline curve.

    # Initialization.
    if len(rho) < order:
        raise ValueError("Too few data points for L-curve analysis")

    if total_num_args > 3:
        m, n = U.shape
        p, ps = s.shape
        beta = U.T @ b
        b0 = b - U @ beta

        if ps == 2:
            s = s[p-1::-1, 0] / s[p-1::-1, 1] #CHECK HERE TO SEE IF CORRECT
            beta = beta[p-1::-1] #CHECK HERE TO SEE IF CORRECT

        xi = beta / s

        if m > n:  # Take care of the least-squares residual.
            beta = np.append(beta, np.linalg.norm(b0))

    # Restrict the analysis of the L-curve according to M (if specified).
    if total_num_args == 8:
        index = np.where(eta < M)[0]
        rho = rho[index]
        eta = eta[index]
        reg_param = reg_param[index]

    if method.startswith('Tikh') or method.startswith('tikh'):
        # The L-curve is differentiable; computation of curvature in
        # log-log scale is easy.

        # Compute g = - curvature of L-curve.
        g = lcfun(reg_param, s, beta, xi)

        # Locate the corner.  If the curvature is negative everywhere,
        # then define the leftmost point of the L-curve as the corner.
        gi = np.argmin(g)
        reg_c = fminbound(lcfun, reg_param[min(gi + 1, len(gi)), reg_param[max(gi - 1, 1)]],
                          args=(s, beta, xi))  # Minimizer.
        kappa_max = -lcfun(reg_c, s, beta, xi)  # Maximum curvature.

        if kappa_max < 0:
            lr = len(rho)
            reg_c = reg_param[lr-1] #CHECK
            rho_c = rho[lr-1] #CHECK
            eta_c = eta[lr-1] #CHECK
        else:
            f = (s ** 2) / (s ** 2 + reg_c ** 2)
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1 - f) * beta[:len(f)])
            if m > n:
                rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0) ** 2)
    elif method.startswith('tsvd') or method.startswith('tgsv') or method.startswith('mtsv') or method.startswith('none'):
        # Use the adaptive pruning algorithm to find the corner if the
        # Spline Toolbox is not available.
        try:
        # Check if the corner function is available
            reg_c = corner(rho, eta)
            rho_c = rho[reg_c]
            eta_c = eta[reg_c]
            return
        except NameError:
            # If corner function is not available, or alwayscorner is True, perform other operations
            pass
        # Otherwise use local smoothing followed by fitting a 2-D spline curve
        # to the smoothed discrete L-curve. Restrict the analysis of the L-curve
        # according to s_thr.
        if total_num_args > 3:
            if total_num_args == 8:  # In case the bound M is in action.
                s = s[index]
            index = np.where(s > s_thr)[0]
            rho = rho[index]
            eta = eta[index]
            reg_param = reg_param[index]

        # Convert to logarithms.
        lr = len(rho)
        lrho = np.log(rho)
        leta = np.log(eta)
        slrho = lrho.copy()
        sleta = leta.copy()

        # For all interior points k = q+1:length(rho)-q-1 on the discrete
        # L-curve, perform local smoothing with a polynomial of degree deg
        # to the points k-q:k+q.
        v = np.arange(-q, q + 1)
        A = np.zeros((2 * q + 1, deg + 1))
        A[:, 0] = 1.0
        for j in range(1, deg + 1):
            A[:, j] = A[:, j - 1] * v

        for k in range(q, lr - q):
            cr = np.linalg.lstsq(A, lrho[k + v], rcond=None)[0]
            slrho[k] = cr[0]
            ce = np.linalg.lstsq(A, leta[k + v], rcond=None)[0]
            sleta[k] = ce[0]

        # Fit a 2-D spline curve to the smoothed discrete L-curve.
        lr += order
        sp = (np.arange(1, lr + 1), np.column_stack([slrho, sleta]))
        pp = sp2pp(spcol(sp, [4, lr + 1]))

        # Extract abscissa and ordinate splines and differentiate them.
        # Compute as many function values as default in spleval.
        P = spleval(pp)
        dpp = fnder(pp)
        D = spleval(dpp)
        ddpp = fnder(pp, 2)
        DD = spleval(ddpp)

        ppx = P[0, :]
        ppy = P[1, :]
        dppx = D[0, :]
        dppy = D[1, :]
        ddppx = DD[0, :]
        ddppy = DD[1, :]

        # Compute the corner of the discretized spline curve via max. curvature.
        # No need to refine this corner, since the final regularization
        # parameter is discrete anyway.
        # Define curvature = 0 where both dppx and dppy are zero.
        k1 = dppx * ddppy - ddppx * dppy
        k2 = (dppx ** 2 + dppy ** 2) ** (1.5)
        I_nz = np.where(k2 != 0)[0]
        kappa = np.zeros(len(dppx))
        kappa[I_nz] = -k1[I_nz] / k2[I_nz]
        kmax, ikmax = np.max(kappa), np.argmax(kappa)
        x_corner, y_corner = ppx[ikmax], ppy[ikmax]

        # Locate the point on the discrete L-curve which is closest to the
        # corner of the spline curve. Prefer a point below and to the
        # left of the corner. If the curvature is negative everywhere,
        # then define the leftmost point of the L-curve as the corner.
        if kmax < 0:
            reg_c = reg_param[lr - 1]
            rho_c = rho[lr - 1]
            eta_c = eta[lr - 1]
        else:
            index = np.where((lrho < x_corner) & (leta < y_corner))[0]
            if len(index) > 0:
                rpi = index[np.argmin((lrho[index] - x_corner) ** 2 + (leta[index] - y_corner) ** 2)]
            else:
                rpi = np.argmin((lrho - x_corner) ** 2 + (leta - y_corner) ** 2)

            reg_c = reg_param[rpi]
            rho_c = rho[rpi]
            eta_c = eta[rpi]

    elif method.lower().startswith('dsvd') or method.lower().startswith('dgsv'):
        # The L-curve is differentiable; computation of curvature in
        # log-log scale is easy.

        # Compute g = - curvature of L-curve.
        g = lcfun(reg_param, s, beta, xi, 1)

        # Locate the corner.  If the curvature is negative everywhere,
        # then define the leftmost point of the L-curve as the corner.
        gi = np.argmin(g)
        reg_c = fminbound(lcfun, reg_param[min(gi + 1, len(g)): max(gi - 1, 1)],
                          args=(s, beta, xi, 1))  # Minimizer.
        kappa_max = -lcfun(reg_c, s, beta, xi, 1)  # Maximum curvature.

        if kappa_max < 0:
            lr = len(rho)
            reg_c = reg_param[lr - 1]
            rho_c = rho[lr - 1]
            eta_c = eta[lr - 1]
        else:
            f = s / (s + reg_c)
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1 - f) * beta[:len(f)])
            if m > n:
                rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0) ** 2)
    else:
        raise ValueError("Illegal method")

    return reg_c, rho_c, eta_c
