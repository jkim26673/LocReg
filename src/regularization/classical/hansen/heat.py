# Purpose:
# Test problem: inverse heat equation.
# Synopsis:
# [A,b,x] = heat (n,kappa)
# Description:
# The inverse heat equation used here is a Volterra integral equation of the first kind
# with [0, 1] as integration interval. The kernel is K(s, t) = k(s− t) with
# k(t) = t−3/2
# 2 kappa√π
# exp−
# 1
# 4 kappa2 t
# .
# Here, the parameter kappa controls the ill-conditioning of the matrix A:
# kappa = 5 gives a well-conditioned matrix,
# kappa = 1 gives an ill-conditioned matrix.
# The default is kappa = 1.
# Algorithm:
# The integral equation is discretized by means of simple collocation and the mid-
# point rule with n points, cf. [1,2]. An exact solution x is constructed, and then the
# right-hand side b is produced as b= A x.
# References:
# 1. A. S. Carasso, Determining surface temperatures from interior observations,
# SIAM J. Appl. Math. 42 (1982), 558–574.
# 2. L. Eld´en, The numerical solution of a non-characteristic Cauchy problem for
# a parabolic equation; in P. Deuflhart & E. Hairer (Eds.), Numerical Treatment
# of Inverse Problems in Diﬀerential and Integral Equations, Birkh¨auser, Boston,
# 1983.
# % Per Christian Hansen, IMM, 11/11/97.

import numpy as np
from scipy.linalg import toeplitz

def heat(n, kappa, nargin, nargout):

    if (nargin==1):
        kappa = 1

    h = 1/n
    t = np.arange(h/2, 1 + h/2, h)
    c = h / (2 * kappa * np.sqrt(np.pi))
    d = 1 / (4 * kappa**2)

    # Compute the matrix A.
    k = c * t**(-1.5) * np.exp(-d / t)
    r = np.zeros(len(t))
    r[0] = k[0]
    A = toeplitz(k, r)

    # Compute the vectors x and b.
    if (nargout > 1):
        x = np.zeros(n)
        for i in np.arange(1,n // 2 + 1):
            ti = i * 20 / n
            if ti < 2:
                x[i-1] = 0.75 * ti**2 / 4
            elif ti < 3:
                x[i-1] = 0.75 + (ti - 2) * (3 - ti)
            else:
                x[i-1] = 0.75 * np.exp(-(ti - 3) * 2)
        
        x[n // 2:n+1] = np.zeros(n//2)
        b = A @ x

    return A, b, x