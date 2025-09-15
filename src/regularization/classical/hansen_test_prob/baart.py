import numpy as np
import math

def baart(n, nargout = 3):
    if (n % 2) != 0:
        raise ValueError("The order n must be even")
    hs = np.pi/(2 * n)
    hs_test = np.pi/(2 * n)
    ht = np.pi/n
    c = 1/(3 * np.sqrt(2))
    A = np.zeros((n,n))
    #Problem here

    #Diff Here
    ihs = np.arange(n+1) * hs
    n1 = n + 1

    #Diff Here
    nh = n/2
    f3 = np.exp(ihs[1:n1]) - np.exp(ihs[:n])

    for j in range(1, n+1):
        f1 = f3
        co2 = math.cos(((j-1) + 0.5) * ht)
        co3 = math.cos(j * ht)
        f2 = (np.exp(ihs[1:n1] * co2) - np.exp(ihs[:n] * co2)) / co2
        if (j == nh):
            f3 = hs * np.ones(n)
        else:
            f3 = (np.exp(ihs[1:n1] * co3) - np.exp(ihs[:n] * co3))/ co3
        A[:,j-1] = c * (f1 + 4*f2 + f3)

    if (nargout > 1):
        si = []
        si[:2*n] = np.arange(0.5, n + 0.5, 0.5).T * hs
        si = np.array(si)
        si = np.sinh(si)/si
        b = np.zeros(n)
        b[0] = 1 + 4*si[0] + si[1]
        indices1 = np.arange(2, 2 * n, 2) - 1
        indices2 = np.arange(3, 2 * n + 1, 2) - 1
        indices3 = np.arange(4, 2 * n + 2, 2) - 1
        b[1:] = si[indices1] + 4 * si[indices2] + si[indices3]
        b = b * np.sqrt(hs)/3
    if (nargout == 3):
         x = -np.diff(np.cos(np.arange(n+1).conj().T * ht))/np.sqrt(ht)
    
    return A, b, x