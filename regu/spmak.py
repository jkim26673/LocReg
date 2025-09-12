import numpy as np
from scipy.interpolate import BSpline

def spmak(t, c, k):
    """
    Construct a B-spline object using the given data.

    Parameters:
        t (ndarray): Array of knots.
        c (ndarray): Array of coefficients.
        k (int): Degree of the spline.

    Returns:
        BSpline: B-spline object.

    """
    t = np.asarray(t)
    c = np.asarray(c)
    k = int(k)

    # Check data shapes
    if t.ndim != 1 or c.ndim != 1 or t.shape[0] != c.shape[0]:
        raise ValueError("Input arrays t and c must have the same shape and be 1-D.")

    # Check degree
    if k < 0:
        raise ValueError("Degree k must be a non-negative integer.")

    # Create B-spline object
    return BSpline(t, c, k, extrapolate=False)
