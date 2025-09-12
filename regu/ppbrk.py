import numpy as np

def ppbrk(f):
    """
    Breaks up the pp-form spline into its components.
    
    Parameters:
        f (ndarray): Array representing the pp-form spline.
    
    Returns:
        tuple: Tuple containing the breaks, coefficients, order, number of dimensions,
               and length of the coefficients array.
    """
    breaks = f[0, :]
    coefs = f[1:, :]
    l, k, d = coefs.shape
    return breaks, coefs, l, k, d
