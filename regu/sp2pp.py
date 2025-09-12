import numpy as np
from scipy.interpolate import PPoly

def sp2pp(sp):
    """
    Convert a spline in B-form (sp) to piecewise polynomial (pp) form.
    
    Parameters:
        sp (ndarray): Array representing the spline in B-form.
    
    Returns:
        PPoly: Piecewise polynomial object representing the spline in pp-form.
    """
    breaks = sp[0, :]
    coefs = sp[1:, :]
    k = sp.shape[0] - 1
    d = sp.shape[1] - 1
    pp = PPoly.from_spline((breaks, coefs), extrapolate=False)
    return pp
