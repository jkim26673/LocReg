import numpy as np
import matplotlib.pyplot as plt
from Utilities_functions.tikhonov_vec import tikhonov_vec

def LocReg_unconstrainedB(data_noisy, G, x0_ini, lambda_ini, ep1, ep2, ep3):
    U, s, V = np.linalg.svd(G)
    prev_x = 2 * x0_ini
    lambda_ini = lambda_ini * np.ones(G.shape[1])
    SD_noise = np.std(data_noisy - G.dot(x0_ini))

    while True:
        # plt.figure(1)
        # plt.plot(x0_ini)
        # plt.draw()
        # plt.hold(True)

        dlambda1 = SD_noise / (x0_ini + ep1)
        dlambda1 = 1 * np.mean(np.abs(lambda_ini)) * (dlambda1 - np.min(dlambda1)) / (np.max(dlambda1) - np.min(dlambda1))
        lambda1 = lambda_ini + dlambda1

        x0_ini = tikhonov_vec(U, s, V, data_noisy, lambda1)
        curr_noise = data_noisy - G.dot(x0_ini)

        delta_p = tikhonov_vec(U, s, V, curr_noise, lambda1)

        # Commented out the inner while loop because it appears to be unused

        x0_ini = x0_ini + delta_p

        if np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x) < ep3:
            f_rec_logreg = x0_ini
            lambda_locreg = lambda1
            break

        ep1 = ep1 / 1.1
        prev_x = x0_ini

    return f_rec_logreg, lambda_locreg