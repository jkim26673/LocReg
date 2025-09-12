from Utilities_functions.lsqnonneg import lsqnonneg
from scipy.optimize import nnls
import numpy as np

def LocReg_NEW_NNLS(data_noisy, G, x0_ini, maxiter):
    # x0_LS = lsqnonneg(G, data_noisy)[0]
    # x0_LS_nnls = nnls(G, data_noisy)[0]
    estimated_noise = lsqnonneg(G, data_noisy)[2]
    #estimated_noise = nnls(G, data_noisy)[1]
    ep = 1e-2
    nT2 = G.shape[1]
    prev_x = x0_ini

    estimated_noise_std = np.std(estimated_noise)
    #track = []
    cur_iter = 1

    while True:
        lambda_val = (np.linalg.norm(G.T @ data_noisy, 2)/ (np.linalg.norm(G.T @ G @ G.T, 2)))/(np.abs(x0_ini) + ep)

        LHS = G.T @ G + np.diag(lambda_val)
        RHS = G.T @ data_noisy + (G.T @ G * ep) @ np.ones(nT2) + ep*lambda_val
        #RHS = G.T @ data_noisy
        try:
            x0_ini = nnls(LHS,RHS, maxiter = 10000)[0]
        except RuntimeWarning as e:
            print("x0_ini error calculation")
            pass
        x0_ini = x0_ini - ep
        x0_ini[x0_ini < 0] = 0
        #curr_noise = np.dot(G, x0_ini) - data_noisy
        curr_noise = (G @ x0_ini) - data_noisy

        delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
        prev = np.linalg.norm(delta_p)
        LHS_temp = LHS.copy()
        while True:
            x0_ini = x0_ini - delta_p
            x0_ini[x0_ini < 0] = 0
            curr_noise = G @ x0_ini - data_noisy
            try:
                delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
            except RuntimeWarning as e:
                print("error with delta_p calculation")
                pass
            if np.abs(np.linalg.norm(delta_p) / prev - 1) < 1e-2:
                break
            prev = np.linalg.norm(delta_p)

        x0_ini = x0_ini - delta_p
        x0_ini[x0_ini < 0] = 0

        #track.append(np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x))
        if (np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x)) < 1e-2 or cur_iter >= maxiter:
            break

        # ax.plot(T2, x0_ini)
        # plt.draw()

        ep = ep / 1.2
        if ep <= 1e-4:
            ep = 1e-4
        cur_iter = cur_iter + 1
        prev_x = x0_ini

    f_rec_logreg = x0_ini
    lambda_locreg = lambda_val

    return f_rec_logreg, lambda_locreg