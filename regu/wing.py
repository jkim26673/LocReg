# %WING Test problem with a discontinuous solution.
# %
# % [A,b,x] = wing(n,t1,t2)
# %
# % Discretization of a first kind Fredholm integral eqaution with
# % kernel K and right-hand side g given by
# %    K(s,t) = t*exp(-s*t^2)                       0 < s,t < 1
# %    g(s)   = (exp(-s*t1^2) - exp(-s*t2^2)/(2*s)  0 < s   < 1
# % and with the solution f given by
# %    f(t) = | 1  for  t1 < t < t2
# %           | 0  elsewhere.
# %
# % Here, t1 and t2 are constants satisfying t1 < t2.  If they are
# % not speficied, the values t1 = 1/3 and t2 = 2/3 are used.

# % Reference: G. M. Wing, "A Primer on Integral Equations of the
# % First Kind", SIAM, 1991; p. 109.

# % Discretized by Galerkin method with orthonormal box functions;
# % both integrations are done by the midpoint rule.

# % Per Christian Hansen, IMM, 09/17/92.

import numpy as np

def wing(n,  nargin, nargout, t1=None, t2=None):
    if nargin == 1:
        t1 = 1/3
        t2 = 2/3
    else:
        if (t1>t2):
            raise ValueError("t1 must be smaller than t2")

    A = np.zeros((n, n))
    h = 1/n
    sti = ((np.arange(1, n+1) - 0.5) * h)

    for i in np.arange(1,n+1):
        A[i-1, :] = h * sti * np.exp(-sti[i-1] * sti**2)

    if (nargout > 1):
        b = np.sqrt(h) * 0.5 * (np.exp(-sti * t1**2) - np.exp(-sti * t2**2)) / sti

    if (nargout == 3):
        x = np.zeros(n)
        I = np.where((t1 < sti) & (sti < t2))
        x[I] = np.sqrt(h) * np.ones(len(I))

    return A, b, x

