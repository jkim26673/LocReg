# %SHAW Test problem: one-dimensional image restoration model.
# %
# % [A,b,x] = shaw(n)
# %
# % Discretization of a first kind Fredholm integral equation with
# % [-pi/2,pi/2] as both integration intervals.  The kernel K and
# % the solution f, which are given by
# %    K(s,t) = (cos(s) + cos(t))*(sin(u)/u)^2
# %    u = pi*(sin(s) + sin(t))
# %    f(t) = a1*exp(-c1*(t - t1)^2) + a2*exp(-c2*(t - t2)^2) ,
# % are discretized by simple quadrature to produce A and x.
# % Then the discrete right-hand b side is produced as b = A*x.
# %
# % The order n must be even.

# % Reference: C. B. Shaw, Jr., "Improvements of the resolution of
# % an instrument by numerical solution of an integral equation",
# % J. Math. Anal. Appl. 37 (1972), 83-112.

# % Per Christian Hansen, IMM, 08/20/91.

import numpy as np

def shaw(n, nargout):
    if n % 2 != 0:
        raise ValueError("The order n must be even")

    h = np.pi / n
    A = np.zeros((n, n))

    co = np.cos(-np.pi/2 + (np.arange(0.5, n + 1 - 0.5)) * h)
    psi = np.pi * np.sin(-np.pi/2 + (np.arange(0.5, n +1 - 0.5)) * h)

    for i in np.arange(1,n // 2 + 1):
        for j in np.arange(i, n - i + 1):
            ss = psi[i-1] + psi[j-1]
            A[i-1, j-1] = ((co[i-1] + co[j-1]) * np.sin(ss) / ss) ** 2
            A[n - j-1 + 1, n - i-1 + 1] = A[i-1, j-1]
        A[i-1, n - i-1 + 1] = (2 * co[i-1]) ** 2

    A = A + np.triu(A, 1).T
    A = A * h

    a1, c1, t1 = 2, 6, 0.8
    a2, c2, t2 = 1, 2, -0.5

    if nargout > 1:
        x = (
            a1 * np.exp(-c1 * (-np.pi/2 + (np.arange(0.5, n + 1 - 0.5)) * h - t1) ** 2)
            + a2 * np.exp(-c2 * (-np.pi/2 + (np.arange(0.5, n + 1 - 0.5)) * h - t2) ** 2)
        )
    b = A @ x

    return A, b, x

