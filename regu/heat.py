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