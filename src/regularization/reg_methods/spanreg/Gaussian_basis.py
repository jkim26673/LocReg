import numpy as np
from scipy.stats import norm

def Gaussian_basis(x, cmin, cmax, Nc, sigma_min, sigma_max):
    """
    Create a set of Gaussian basis functions represented by a matrix, whose columns are of size nx (or n(T2))
    and the rows are of size sum(Nc). Each column is one Gaussian distribution.

    Parameters:
    x (numpy array): Input data (a 1D numpy array) where the basis functions will be evaluated.
    cmin (float): Minimum value for the centers of the Gaussian basis functions.
    cmax (float): Maximum value for the centers of the Gaussian basis functions.
    Nc (np.array): List containing the number of centers for each sigma (standard deviation).
    sigma_min (float): Minimum value for the standard deviation of the Gaussian basis functions.
    sigma_max (float): Maximum value for the standard deviation of the Gaussian basis functions.

    Returns:
    LGBs (numpy array): Matrix containing the Gaussian basis functions.
    """
    
    nsigma = len(Nc)
    sigma = np.linspace(sigma_min, sigma_max, nsigma)  # Set of standard deviations

    c = [None] * nsigma

    for k in range(nsigma):
        c[k] = np.linspace(cmin + 3 * sigma[k], cmax - 3 * sigma[k], Nc[k])
    
    c = np.array(c, dtype=object)

    x = np.array(x)
    x = x.reshape(len(x),1)
    a = x.shape[0]
    b = x.shape[1]

    if a < b:
        x = x.T
    
    LGBs = np.empty((x.shape[0], 0))
    for j in range(nsigma):
        for i in range(len(c[j])):
            gaus = norm.pdf(x, loc=c[j][i], scale=sigma[j])
            LGBs = np.hstack((LGBs, gaus))
    return LGBs
