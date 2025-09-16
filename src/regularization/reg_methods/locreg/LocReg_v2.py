import numpy as np
from regularization.subfunc.find_corner import find_corner
from kneed import KneeLocator
from regularization.subfunc.csvd import csvd

def LocReg_v2(data_noisy, G, lambda_ini):
    # Singular Value Decomposition
    U, s, Vt = csvd(G, tst = None, nargin = 1, nargout = 3)
    # V = Vt.T
    UD_fix = U.T @ data_noisy
    W = Vt @ np.diag(UD_fix)

    # Calculate GWpart
    GWpart = G @ W

    # Initialize variables
    x_ini = np.zeros(len(s))
    for i in range(len(s)):
        x_ini[i] = s[i] / (s[i] ** 2 + lambda_ini ** 2)

    num_columns = G.shape[1]
    x_rec = np.zeros(num_columns)
    projection_temp = np.zeros(len(data_noisy))
    projection_norms = np.zeros(num_columns)

    for i in range(num_columns):
        data_noisy = data_noisy - projection_temp
        u = data_noisy
        x_rec[i] = np.dot(u, GWpart[:, i]) / (np.dot(GWpart[:, i], GWpart[:, i]))
        # #run the below code for heat_prob
        # x_rec[i] = np.dot(u, GWpart[:, i]) / (np.dot(GWpart[:, i], GWpart[:, i]) + 1e-4)
        projection_temp = x_rec[i] * GWpart[:, i]
        projection_norms[i] = np.linalg.norm(projection_temp)

    #Selection criteria of the columns in the projection_norms
    scale = range(1, len(projection_norms)+1)
    # Choose num_of_var_lambdas
    # kn = KneeLocator(scale, projection_norms, curve='convex', direction='decreasing')
    kn = KneeLocator(scale, projection_norms, curve='convex', direction='decreasing', interp_method = 'polynomial')
    ireg_corner_kn = kn.knee
    if ireg_corner_kn == None:
        ireg_corner_kn = 3
    ireg_corner_f = find_corner(projection_norms)

    x_locreg_k = x_rec
    x_locreg_f = x_rec

    x_locreg_k[1+ireg_corner_kn:] = x_ini[1+ireg_corner_kn:]
    x_locreg_f[1+ireg_corner_f:] = x_ini[1+ireg_corner_f:]


    f_rec_locreg_k = W @ x_locreg_k
    f_rec_locreg_f = W @ x_locreg_f


    # Calculate lambda_locreg
    lambda_locreg_k = s / (np.linalg.solve(W, f_rec_locreg_k)) - s ** 2
    lambda_locreg_f = s / (np.linalg.solve(W, f_rec_locreg_f)) - s ** 2
    # p1 = W @ (s / (s**2 + lambda_approx))

    return f_rec_locreg_k, lambda_locreg_k,f_rec_locreg_f, lambda_locreg_f
