import numpy as np
from numpy.linalg import norm
from scipy.optimize import fminbound, minimize_scalar
from regu.gcvfun import gcvfun
import matplotlib.pyplot as plt

def gcv(U, s, b, method, nargin, nargout):
    if (nargin == 3):
        method = 'Tikh'
    npoints = 200
    smin_ratio = 16 * np.finfo(float).eps
    m, n = U.shape
    p, ps = s.reshape(-1,1).shape
    beta = U.T @ b
    beta2 = norm(b)**2 - norm(beta)**2
    if ps == 2:
        s= s[p-1:1:-1, 0]/ s[p-1:1:-1, 1]
        beta = beta[p-1:1:-1]
    if (nargout > 0):
        find_min = 1
    else:
        find_min = 0
    if method.startswith('Tikh'):
        reg_param = np.zeros(npoints)
        G = np.zeros(npoints)
        s2 =s ** 2
        reg_param[-1] = max(s[p - 1], s[0] * smin_ratio)
        ratio = (s[0] / reg_param[npoints-1]) ** (1 / (npoints - 1))
        for i in range(npoints - 1, 0, -1):
            reg_param[i-1] = ratio * reg_param[i]
        delta0 = 0
        if m > n and beta2 > 0:
            delta0 = beta2
        for i in range(npoints):
            G[i] = gcvfun(reg_param[i], s2, beta[:p], delta0, m - n, dsvd = None, nargin = 5)
        if find_min:
            minG = np.min(G)
            minGi = np.argmin(G)
            # reg_min = minimize_scalar(
            #     lambda x: gcvfun(x, s2, beta[0:p], delta0, m - n, dvsd = None, nargin = 5),
            #     bounds=(reg_param[min(minGi+1,npoints)], reg_param[max(minGi-1,1)]),
            #     method='bounded',
            #     options={'disp': False}).x
            reg_min = fminbound(lambda x: gcvfun(x, s2, beta[:p], delta0, m - n, dsvd = None, nargin = 5), 
                                reg_param[min(minGi + 2, npoints) - 1],
                                reg_param[max(minGi, 1) - 1], xtol= 1e-3)
            # if reg_min < 1e-3:
            #     reg_min = 1e-2
            #For /Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/2D_multi_reg_MRR_0116.py
            if reg_min < 1e-4:
                reg_min = 1e-3
            # fig = plt.figure()
            # ax = fig.add_subplot(2, 1, 1)
            # ax.plot(reg_min, minG, color='blue', lw=2)
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            # ax.set_title(f"GCV function, minimum at lambda = {reg_min}")
            # ax.show()
            # plt.close(fig)
            # reg_param[min(minGi + 2, npoints) - 1],
            #                     reg_param[max(minGi - 2, 1) - 1])
            minG = gcvfun(reg_min, s2, beta[:p], delta0, m - n, dsvd = None, nargin = 5)
    elif method.startswith('tsvd') or method.startswith('tgsv'):
        rho2 = np.zeros(p - 1)
        rho2[p - 2] = np.abs(beta[p - 1]) ** 2
        if m > n and beta2 > 0:
            rho2[p - 2] += beta2
        for k in range(p - 3, -1, -1):
            rho2[k] = rho2[k + 1] + np.abs(beta[k + 1]) ** 2
        G = np.zeros(p-1)
        for k in range(p-1):
            G[k] = rho2[k]/((m - (k+1) + (n-p))**2)
        reg_param = np.arange(1, p).conj().T
        if find_min:
            minG = np.min(G)
            reg_min = np.argmin(G)

    elif method.startswith('dsvd') or method.startswith('dgsv'):
        reg_param[npoints - 1] = max(s[p - 1], s[0] * smin_ratio)
        ratio = (s[0] / reg_param[npoints - 1]) ** (1 / (npoints - 1))
        for i in range(npoints - 2, -1, -1):
            reg_param[i] = ratio * reg_param[i + 1]
        delta0 = 0
        if m > n and beta2 > 0:
            delta0 = beta2
        for i in range(npoints):
            G[i] = gcvfun(reg_param[i], s, beta[:p], delta0, m - n, 1, nargin = 6)
        if find_min:
            minGi = np.argmin(G)
            # reg_min = minimize_scalar(
            #     lambda x: gcvfun(x, s2, beta[0:p], delta0, m - n, dvsd = None, nargin = 5),
            #     bounds=(reg_param[min(minGi+1,npoints)], reg_param[max(minGi-1,1)]),
            #     method='bounded',
            #     options={'disp': False}).x
            reg_min = fminbound(lambda x: gcvfun(x, s, beta[:p], delta0, m - n, 1, nargin = 6), 
                                reg_param[min(minGi + 1, npoints-1)],
                                reg_param[max(minGi - 1, 0)])
            minG = gcvfun(reg_min, s, beta[:p], delta0, m - n, 1, nargin = 6)

    elif method.startswith('mtsv') or method.startswith('ttls'):
        raise ValueError("The MTSVD and TTLS methods are not supported")

    else:
        raise ValueError("Illegal method")
    
    if find_min:
        return reg_min, G, reg_param
    else:
        return G, reg_param
