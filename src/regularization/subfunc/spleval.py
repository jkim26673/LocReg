import numpy as np

def spleval(f, npoints=300):
    """
    Evaluation of a spline or spline curve.

    Parameters:
        f (BSpline): B-spline object.
        npoints (int): Number of points to compute on the spline curve.

    Returns:
        ndarray: Array of points on the spline curve.

    """
    breaks, coefs, l, k, d = f.t, f.c, f.k, f.degree, f.dim

    x = np.linspace(breaks[0], breaks[l], npoints)
    v = f(x)

    if d == 1:
        points = np.vstack((x, v))
    else:
        points = v

    return points
