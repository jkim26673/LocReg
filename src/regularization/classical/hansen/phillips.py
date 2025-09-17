# %PHILLIPS Test problem: Phillips' "famous" problem.
# %
# % [A,b,x] = phillips(n)
# %
# % Discretization of the `famous' first-kind Fredholm integral
# % equation deviced by D. L. Phillips.  Define the function
# %    phi(x) = | 1 + cos(x*pi/3) ,  |x| <  3 .
# %             | 0               ,  |x| >= 3
# % Then the kernel K, the solution f, and the right-hand side
# % g are given by:
# %    K(s,t) = phi(s-t) ,
# %    f(t)   = phi(t) ,
# %    g(s)   = (6-|s|)*(1+.5*cos(s*pi/3)) + 9/(2*pi)*sin(|s|*pi/3) .
# % Both integration intervals are [-6,6].
# %
# % The order n must be a multiple of 4.

# % Reference: D. L. Phillips, "A technique for the numerical solution
# % of certain integral equations of the first kind", J. ACM 9
# % (1962), 84-97.

# % Discretized by Galerkin method with orthonormal box functions.

# % Per Christian Hansen, IMM, 09/17/92.

import math
import sympy as sp
import numpy as np
from scipy.linalg import toeplitz

def scos(x): return sp.N(sp.cos(x))


def phillips(n, nargout):
    if n % 4 != 0:
        raise ValueError("The order n must be a multiple of 4")

    h = 12 / n
    n4 = n // 4
    r1 = np.zeros(n)
    c = np.cos((np.arange(-1,n4 + 1)) * 4 * np.pi / n)
    r1[:n4] = h + 9 / (h * np.pi**2) * (2 * c[1:n4+1] - c[:n4] - c[2:n4+2])
    r1[n4] = h / 2 + 9 / (h * np.pi**2) * (np.cos(4 * np.pi / n) - 1)
    A = toeplitz(r1)

    if nargout > 1:
        b = np.zeros(n)
        c = np.pi / 3
        for i in np.arange(n // 2 + 1, n + 1):
            t1 = -6 + i * h
            t2 = t1 - h
            b[i-1] = (
                t1 * (6 - abs(t1) / 2)
                + ((3 - abs(t1) / 2) * np.sin(c * t1) - 2 / c * (np.cos(c * t1) - 1)) / c
                - t2 * (6 - abs(t2) / 2)
                - ((3 - abs(t2) / 2) * np.sin(c * t2) - 2 / c * (np.cos(c * t2) - 1)) / c )
            b[n - i] = b[i-1]
        b /= np.sqrt(h)

    if (nargout == 3):
        x = np.zeros(n)
        val = np.arange(0, 3 + 10 * np.spacing(1), h)
        diff_val = np.around(np.sin(val.conj().T * c), decimals = 40)
        x[2*n4 : 3 * n4] = (h + np.diff(diff_val)/c)/np.sqrt(h)
        # x[n4 : 2 * n4] = x[3 * n4 - 1 : 2*n4 + 1 : -1]
        x[n4:2*n4] = x[3*n4-1: 2*n4-1:-1]

    
    return A, b, x

