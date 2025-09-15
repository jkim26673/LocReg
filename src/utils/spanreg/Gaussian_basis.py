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
            # LGBs = np.array(LGBs).T
    #LGBs = np.vstack(LGBs)
    #print("LGBs shape: ", LGBs.shape)
    return LGBs
    # nsigma = len(Nc)
    # sigma = np.linspace(sigma_min, sigma_max, nsigma)  # Set of standard deviations

    # c = []
    # for k in range(nsigma):
    #     c.append(np.linspace(cmin + 3 * sigma[k], cmax - 3 * sigma[k], Nc[k]))  # Set of centers for each sigma

    # x = x.reshape(-1, 1) if x.shape[0] < x.shape[1] else x  # Convert x to Nx1 vector if needed
    # LGBs = []  # Initialize empty list for Gaussian basis functions

    # for j in range(nsigma):
    #     for i in range(len(c[j])):
    #         gaus = norm.pdf(x, loc=c[j][i], scale=sigma[j])  # Gaussian probability density function
    #         LGBs.append(gaus.flatten())  # Append Gaussian function to LGBs

    # LGBs = np.array(LGBs).T
    # return LGBs


# def Gaussian_basis(x, cmin, cmax, Nc, sigma_min, sigma_max):
#     nsigma = len(Nc)
#     sigma = np.linspace(sigma_min, sigma_max, nsigma)  # Set of standard deviations

#     c = [np.linspace(cmin + 3 * s, cmax - 3 * s, n) for s, n in zip(sigma, Nc)]  # Set of centers for each sigma
    
#     # c = []  # Create an empty list to store the set of centers for each sigma

#     # for k in range(nsigma):
#     #     centers_k = np.linspace(cmin + 3 * sigma[k], cmax - 3 * sigma[k], Nc[k])
#     #     c.append(centers_k)  # Append the centers for the current sigma to the list

#     a, b = x.shape
#     if x.shape[0] < x.shape[1]:
#         x = x.T  # Transpose x to ensure it's a column vector

#     LGBs = np.array([])
#     for j in range(nsigma):
#         for i in range(len(c[j])):
#             gaus = norm.pdf(x, c[j][i], sigma[j])
#             LGBs.append(gaus)

#     LGBs = np.array(LGBs).T
#     return LGBs

#test = Gaussian_basis(T2, cmin, cmax, Nc, sigma_min, sigma_max)