import numpy as np
from scipy.interpolate import PPoly

def fnder(pp, m=1):
    """
    Differentiate a piecewise polynomial (pp) object.
    
    Parameters:
        pp (PPoly): Piecewise polynomial object.
        m (int): Order of differentiation (default is 1).
    
    Returns:
        PPoly: Differentiated piecewise polynomial object.
    """
    breaks = pp.x
    coeffs = pp.c
    k = pp.k
    d = pp.c.shape[1]
    
    for _ in range(m):
        new_coeffs = np.zeros((coeffs.shape[0] - 1, d))
        
        for i in range(d):
            new_coeffs[:, i] = np.diff(coeffs[:, i]) * (k - np.arange(k)) / (breaks[1:] - breaks[:-1])
        
        coeffs = new_coeffs
        k -= 1
    
    return PPoly(coeffs, breaks, extrapolate=False)
