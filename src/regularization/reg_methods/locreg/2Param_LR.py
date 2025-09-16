import sys
sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
from regularization.subfunc.csvd import csvd
import numpy as np
import matplotlib.pyplot as plt
from regularization.reg_methods.nnls.tikhonov_vec import tikhonov_vec
from numpy.linalg import norm
from scipy.special import airy,gamma,itairy, gammaln, entr,tklmbda,expit

def Multi_Param_LR(data_noisy, G, L_1, L_2, alpha_1, alpha_2, ep, c, delta):
    #initialization
    norm_dat = norm(data_noisy)**2
    k = 1
    #Step 2
    while True:
        x_delta = np.linalg.inv(alpha_1 * L_1.T @ L_1 + alpha_2 * L_2.T @ L_2 + G.T @ G) @ G.T @ data_noisy
        print(f"alpha_1 for {k} iteration:", alpha_1)
        print(f"alpha_2 for {k} iteration:", alpha_2)
        if norm(G @ x_delta - data_noisy) >= c*delta:
            print(f"iteration {k}")
            print(f"norm(G @ x_delta - data_noisy) for {k} iteration:", norm(G @ x_delta - data_noisy))
            print(f"c*delta for {k} iteration:", c*delta)
            F = norm(G @ x_delta - data_noisy)**2 + alpha_1 * norm(L_1 @ x_delta)**2 + alpha_2 * norm(L_2 @ x_delta)**2
            F_1 = norm(L_1 @ x_delta)**2
            F_2 = norm(L_2 @ x_delta)**2
            F_last = norm(x_delta)**2
            C_1 = -F_1 *(alpha_1)**2 
            C_2 = -F_2 *(alpha_2)**2 
            D = - (norm_dat - F - (alpha_1 * F_1 + alpha_2 * F_2))**2 / F_last
            # D = -(norm(G @ x_delta)**2 + alpha_2 * norm(L_2 @ x_delta)**2)**2 / norm(L_2 @ x_delta)**2
            T = (norm_dat - F - (alpha_1 * F_1 + alpha_2 * F_2)) / F_last
        else:
            print("")
            print("Step 2 is satisfied")
            break
        #Step 3
        alpha_1_new = (2 * C_1) / (c**2 * delta**2 - norm_dat -(2*C_2/alpha_2) - D/T)
        alpha_2_new = (2 * C_2) / (c**2 * delta**2 - norm_dat -(2*C_1/alpha_1) - D/T)
        print(f"alpha_1_new for {k} iteration:", alpha_1_new)
        print(f"alpha_2_new for {k} iteration:", alpha_2_new)
        #Step 4
        if np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) >= ep and alpha_1 > ep and alpha_2 > ep:
            print("np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2)", np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2))
            alpha_1 = alpha_1_new
            alpha_2 = alpha_2_new
        else:
            print("")
            print("Step 4 is satisfied")
            break
        k = k + 1
    print(f"Total of {k} iterations")
    print("")
    return alpha_1,alpha_2, x_delta
