import numpy as np
import math
# %GRAVITY Test problem: 1-D gravity surveying model problem
# %
# % [A,b,x] = gravity(n,example,a,b,d)
# %
# % Discretization of a 1-D model problem in gravity surveying, in which
# % a mass distribution f(t) is located at depth d, while the vertical
# % component of the gravity field g(s) is measured at the surface.
# %
# % The resulting problem is a first-kind Fredholm integral equation
# % with kernel
# %    K(s,t) = d*(d^2 + (s-t)^2)^(-3/2) .
# % The following three examples are implemented (example = 1 is default):
# %    1: f(t) = sin(pi*t) + 0.5*sin(2*pi*t),
# %    2: f(t) = piecewise linear function,
# %    3: f(t) = piecewise constant function.
# % The problem is discretized by means of the midpoint quadrature rule
# % with n points, leading to the matrix A and the vector x.  Then the
# % right-hand side is computed as b = A*x.
# %
# % The t integration interval is fixed to [0,1], while the s integration
# % interval [a,b] can be specified by the user. The default interval is
# % [0,1], leading to a symmetric Toeplitz matrix.
# %
# % The parameter d is the depth at which the magnetic deposit is located,
# % and the default value is d = 0.25. The larger the d, the faster the
# % decay of the singular values.

# % Reference: G. M. Wing and J. D. Zahrt, "A Primer on Integral Equations
# % of the First Kind", SIAM, Philadelphia, 1991; p. 17.

# % Per Christian Hansen, IMM, November 18, 2001.

def gravity(n, nargin, example=1, a=0, b=1, d=0.25):
    if (nargin < 2):
        example = 1
    if (nargin < 4):
        a = 0
        b = 1
    if (nargin < 5):
        d=0.25
    if example is None:
        example = 1
    if a is None:
        a = 0
    if b is None:
        b = 1

    dt = 1/n
    ds = (b-a)/n
    t = dt * (np.arange(1, n+1) - 0.5)
    s = a + ds * (np.arange(1, n+1) - 0.5)
    T, S = np.meshgrid(t, s)
    A = dt * d * np.ones((n, n)) / (d**2 + (S - T)**2)**(3/2)
    def normal_round(n):
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)
    nt = normal_round(n/3)
    nn = normal_round(n*7/8)
    x = np.ones(n)
    if example == 1:
        x = np.sin(np.pi * t) + 0.5 * np.sin(2 * np.pi * t)
    elif example == 2:
        x[:nt] = (2/nt) * np.arange(1, nt+1)
        x[nt:nn] = ((2*nn-nt) - np.arange(nt+1, nn+1)) / (nn-nt)
        x[nn:] = (n - np.arange(nn+1, n+1)) / (n-nn)
    elif example == 3:
        x[:nt] = 2 * np.ones(nt)
    else:
        raise ValueError('Illegal value of example')

    b = A @ x
    return A, b, x

