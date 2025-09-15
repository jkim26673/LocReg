import numpy as np
import sys
sys.path.append('.')
import matplotlib.pyplot as plt
from regu.baart import baart
from regu.foxgood import foxgood
from regu.heat import heat
from regu.gravity import gravity
from regu.csvd import csvd
from regu.l_curve import l_curve
from regu.tikhonov import tikhonov
from regu.gcv import gcv
from regu.discrep import discrep
from regu.csvd import csvd
from regu.find_corner import find_corner

if __name__ == '__main__':
    # Initialize parameters
    n = 300
    nT2 = n
    T2 = np.linspace(-np.pi/2, np.pi/2, n)
    TE = T2
    # G, data_noiseless, g = heat(n, kappa = 1, nargin = 1, nargout=3)
    G, data_noiseless, g = baart(n)
    # G, data_noiseless, g = foxgood(n)
    # G, data_noiseless, g = gravity(n, nargin=1)

    U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
    SNR = 30
    SD_noise = 1 / SNR * np.max(np.abs(data_noiseless))
    noise = np.random.normal(0, SD_noise, size=data_noiseless.shape)
    data_noisy = data_noiseless + noise
    # Lambda_vec = np.logspace(-5, 5, 40)
    # nLambda = len(Lambda_vec)
    UD_fix = U.T @ data_noisy
    W = V @ np.diag(UD_fix)

    delta = np.linalg.norm(noise) * 1.05
    f_DP, lambda_DP = discrep(U, s, V, data_noisy, delta, x_0= None, nargin = 5)
    s_DP = np.zeros(len(s))
    # s_DP_filt = np.zeros(len(s))

    for i in range(len(s)):
        # s_DP[i] = s[i] / (s[i] ** 2 + lambda_DP ** 2)
        s_DP[i] = s[i] / (s[i] ** 2 + lambda_DP ** 2)

    # f_rec_DP_filt = W @ s_DP_filt
    f_rec_DP = W @ s_DP

    x_GT = np.linalg.solve(W, g)

    # x_0 = np.zeros(G.shape[1])
    # ep = 1e-5
    # x = np.zeros(G.shape[1])
    # r = G @ x - data_noisy
    # while r > ep:
    #     p0 = r
    #     k = 0

    # r = G @ 

    # M = U @ np.diag(s_DP**2) @ U.T
    # T = M @ s_DP

    # for i in range(2, 11):
    #     print(np.linalg.cond(G @ np.diag(T) @ W[:, :i]))

    # GWpart = G @ np.diag(T) @ W
    GWpart = G @ W

    # lambda_vec = np.zeros(G.shape[1])

    x_rec = np.zeros(G.shape[1])
    projection_temp = np.zeros(len(data_noisy))
    projection_norms = np.zeros(G.shape[1])

    for i in range(G.shape[1]):
        data_noisy = data_noisy - projection_temp
        u = data_noisy
        x_rec[i] = np.dot(u, GWpart[:, i]) / np.dot(GWpart[:, i], GWpart[:, i])
        projection_temp = x_rec[i] * GWpart[:, i]
        projection_norms[i] = np.linalg.norm(projection_temp)
    # x = range(1, len(projection_norms)+1)

    # noise = np.random.normal(0,5, data_noiseless.shape)
    # y = projection_norms + noise

    # from kneed import KneeLocator
    # kn = KneeLocator(x, projection_norms, curve='convex', direction='decreasing')
    # print(kn.knee)

    ireg_cornerf = find_corner(projection_norms)
    print(ireg_cornerf)


    # import matplotlib.pyplot as plt
    # plt.xlabel('number of clusters k')
    # plt.ylabel('projection')
    # plt.plot(x, projection_norms, 'bx-')
    # plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    # plt.vlines(ireg_cornerf, plt.ylim()[0], plt.ylim()[1], linestyles='solid')
    # plt.show()
    # ireg_corner = 2
    # x_locreg_filt = x_rec
    x_locreg_zero = x_rec
    x_test = x_rec

    x_locreg_zero[1 + ireg_cornerf:] = s_DP[1 + ireg_cornerf:]
    # print("x_locreg_zero", x_locreg_zero)
    x_test[:10] = x_GT[:10]
 
    # x_locreg_filt[1 + ireg_cornerf:] = s_DP_filt[1 + ireg_cornerf:]
    # print("x_locreg_filt", x_locreg_filt)

    # x_locreg = x_DP
    from numpy.linalg import norm
    err_DP = norm(g - W @ s_DP)
    # err_DP_filt = norm(g - W @ s_DP_filt)

    err_LR_zero = norm(g - W @ x_locreg_zero)
    err_LR_oracle = norm(g - W @ x_test)

    # err_LR_filt = norm(g - W @ x_locreg_filt)

    # Plot the results
    plt.figure()
    plt.plot(g, label = "gt")
    # plt.plot(W @ x_GT, linewidth=2, label= 'W @ gt')
    plt.plot(W @ s_DP, label = 'DP')
    # plt.plot(W @ s_DP_filt, label = 'DP_filt')

    # # plt.plot(f_DP)
    # plt.plot(f_rec_DP)
    # plt.plot(W @ x_locreg_filt, label = 'LR filt')
    # plt.plot(W @ x_locreg_zero, label = 'LR')
    plt.plot(W @ x_test, label = "10oracle")
    # plt.legend(['f_DP', 'f_rec_dp'])
    # plt.legend(['GT', 'DP no filter factor', 'DP with filter factor', 'LR with filter factor', 'LR no filter factor'])
    # plt.suptitle(f"Comparing Reconstructions of Projection Method With and Without Filter Factors\n" + f"err_DP_no_filt: {err_DP}\n" + f"err_DP_filt: {err_DP_filt}\n"
    #             +f"err_LR_no_filt: {err_LR_zero}\n" +
    #             f"err_LR_filt: {err_LR_filt}", fontsize = 8)
    plt.legend(['GT', "DP", "oracleLR"])
    plt.suptitle(f"Comparing Reconstructions of Projection Method\n" + f"err_DP_no_filt: {err_DP}\n" 
                + f"err_LR_oracle10: {err_LR_oracle}\n", fontsize = 8)
    plt.legend(fontsize='small', loc='best')
    plt.xlabel('T2', fontsize=20, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
    plt.show()
    print("Plot Complete")
    plt.savefig("subspacealgo.png")


