import sys
sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
from regu.csvd import csvd
import numpy as np
import matplotlib.pyplot as plt
from Utilities_functions.tikhonov_vec import tikhonov_vec
from numpy.linalg import norm
from scipy.special import airy,gamma,itairy, gammaln, entr,tklmbda,expit

def Multi_Param_LR(data_noisy, G, g,  B_1, B_2, alpha_1, alpha_2, beta, omega, ep, c, delta):
    #initialization
    beta = 0
    norm_dat = norm(data_noisy)**2
    k = 1
    #Step 2
    while True:
        x_delta = np.linalg.inv(G.T @ G + alpha_1 * B_1.T @ B_1 + alpha_2 * B_2.T @ B_2 + beta * np.eye(data_noisy.shape[0])) @ (G.T @ data_noisy)
        print("Step 2:")
        print(f"norm(G @ x_delta - data_noisy) for {k} iteration:", norm(G @ x_delta - data_noisy))
        print(f"c*delta for {k} iteration:", c * delta)
        print("")
        # print(f"alpha_1 for {k} iteration:", alpha_1)
        # print(f"alpha_2 for {k} iteration:", alpha_2)
        if norm(G @ x_delta - data_noisy) < c*delta and k != 1:
            final_alpha_1 = alpha_1
            final_alpha_2 = alpha_2
            final_beta = beta
            final_x_delta = x_delta
            print("")
            print("Step 2 is satisfied")
            print("")
            break
        else:
            F = norm(G @ x_delta - data_noisy)**2 + alpha_1 * norm(B_1 @ x_delta)**2 + alpha_2 * norm(B_2 @ x_delta)**2 + beta * norm(x_delta)**2
            F_1 = norm(B_1 @ x_delta)**2
            F_2 = norm(B_2 @ x_delta)**2
            F_last = norm(x_delta)**2
            C_1 = -F_1 *(alpha_1)**2 
            print("C_1:", C_1)
            C_2 = -F_2 *(alpha_2)**2 
            print("C_2:", C_2)
            D = - (norm_dat - F - (alpha_1 * F_1 + alpha_2 * F_2))**2 / F_last
            T = (norm_dat - F - (alpha_1 * F_1 + alpha_2 * F_2)) / (F_last - beta)
        #Step 3
        if alpha_1 == 0 and alpha_2 == 0:
            alpha_1_new = 0
            alpha_2_new = 0
        else:
            alpha_1_new = (2 * C_1) / (c**2 * delta**2 - norm_dat -(2*C_2/alpha_2) - D/(T + beta) - ((beta * D) / (beta + T)**2))
            print("alpha_1_new:", alpha_1_new)
            alpha_2_new = (2 * C_2) / (c**2 * delta**2 - norm_dat -(2*C_1/alpha_1) - D/(T + beta) - ((beta * D) / (beta + T)**2))
            print("alpha_2_new:", alpha_2_new)

        if alpha_1 == 0 and alpha_2 == 0:
            beta_new = omega * ((2*D /(c**2 * delta**2 - norm_dat + ((T * D) / (beta + T)**2))) - T)
            print("beta_new:", beta_new)
        else: 
            beta_new = omega * ((2*D /(c**2 * delta**2 - norm_dat - (2*C_2/alpha_2) - (2*C_1/alpha_1) + ((T * D) / (beta + T)**2))) - T)
        #Step 4
        print("Step 4:")
        print(f"(np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)) for {k} iteration:", 
              (np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)))
        print("ep value:", ep)
        print("")
        if k == 1:
            alpha_1 = alpha_1_new
            alpha_2 = alpha_2_new
            beta = beta_new
            print("Iteration is one, so we update values")
            print("")
        elif ((np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)) >= ep and alpha_1 > ep and alpha_2 > ep and beta > ep):
            alpha_1 = alpha_1_new
            alpha_2 = alpha_2_new
            beta = beta_new
        else:
            final_alpha_1 = alpha_1
            final_alpha_2 = alpha_2
            final_beta = beta
            final_x_delta = x_delta
            print("")
            print(f"(np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)) for {k} iteration:", 
                  (np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)))
            print("")
            print("Step 4 is satisfied")
            print("")
            break
        k = k + 1
    print(f"Total of {k} iterations")
    print("")
    rel_error = norm(g - final_x_delta)/norm(g)
    return final_alpha_1,final_alpha_2,final_beta, final_x_delta, rel_error

# def Multi_Param_LR(data_noisy, G, g,  B_1, B_2, alpha_1, alpha_2, beta, omega, ep, c, delta):
#     #initialization
#     norm_dat = norm(data_noisy)**2
#     k = 1
#     #Step 2
#     while True:
#         x_delta = np.linalg.inv(G.T @ G + alpha_1 * B_1.T @ B_1 + alpha_2 * B_2.T @ B_2 + beta * np.eye(data_noisy.shape[0])) @ (G.T @ data_noisy)
#         print(f"norm(G @ x_delta - data_noisy) for {k} iteration:", norm(G @ x_delta - data_noisy))
#         print(f"c*delta for {k} iteration:", c * delta)
#         # print(f"alpha_1 for {k} iteration:", alpha_1)
#         # print(f"alpha_2 for {k} iteration:", alpha_2)
#         if norm(G @ x_delta - data_noisy) < c*delta:
#             final_alpha_1 = alpha_1
#             final_alpha_2 = alpha_2
#             final_beta = beta
#             final_x_delta = x_delta
#             print("")
#             print("Step 2 is satisfied")
#             break
#         else:
#             F = norm(G @ x_delta - data_noisy)**2 + alpha_1 * norm(B_1 @ x_delta)**2 + alpha_2 * norm(B_2 @ x_delta)**2 + beta * norm(x_delta)**2
#             F_1 = norm(B_1 @ x_delta)**2
#             F_2 = norm(B_2 @ x_delta)**2
#             F_last = norm(x_delta)**2
#             C_1 = -F_1 *(alpha_1)**2 
#             C_2 = -F_2 *(alpha_2)**2 
#             D = - (norm_dat - F - (alpha_1 * F_1 + alpha_2 * F_2))**2 / F_last
#             T = (norm_dat - F - (alpha_1 * F_1 + alpha_2 * F_2)) / (F_last - beta)
#         #Step 3
#         alpha_1_new = (2 * C_1) / (c**2 * delta**2 - norm_dat -(2*C_2/alpha_2) - D/(T + beta) - ((beta * D) / (beta + T)**2))
#         alpha_2_new = (2 * C_2) / (c**2 * delta**2 - norm_dat -(2*C_1/alpha_1) - D/(T + beta) - ((beta * D) / (beta + T)**2))
#         beta_new = omega * ((2*D /(c**2 * delta**2 - norm_dat - (2*C_2/alpha_2) - (2*C_1/alpha_1) + ((T * D) / (beta + T)**2))) - T)
#         #Step 4
#         print(f"(np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)) for {k} iteration:", 
#               (np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)))
#         if ((np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)) >= ep and alpha_1 > ep and alpha_2 > ep and beta > ep):
#             alpha_1 = alpha_1_new
#             alpha_2 = alpha_2_new
#             beta = beta_new
#         else:
#             final_alpha_1 = alpha_1
#             final_alpha_2 = alpha_2
#             final_beta = beta
#             final_x_delta = x_delta
#             print("")
#             print(f"(np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)) for {k} iteration:", 
#                   (np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)))
#             print("Step 4 is satisfied")
#             break
#         k = k + 1
#     print(f"Total of {k} iterations")
#     print("")
#     rel_error = norm(g - final_x_delta)/norm(g)
#     return final_alpha_1,final_alpha_2,final_beta, final_x_delta, rel_error




# def Multi_Param_LR(data_noisy, G, L_1, L_2, alpha_1, alpha_2, ep, c, delta):
#     #initialization
#     norm_dat = norm(data_noisy)**2
#     k = 1
#     #Step 2
#     while True:
#         x_delta = np.linalg.inv(alpha_1 * L_1.T @ L_1 + alpha_2 * L_2.T @ L_2 + G.T @ G) @ G.T @ data_noisy
#         print(f"alpha_1 for {k} iteration:", alpha_1)
#         print(f"alpha_2 for {k} iteration:", alpha_2)
#         if norm(G @ x_delta - data_noisy) >= c*delta:
#             F = norm(G @ x_delta - data_noisy)**2 + alpha_1 * norm(L_1 @ x_delta)**2 + alpha_2 * norm(L_2 @ x_delta)**2
#             F_1 = norm(L_1 @ x_delta)**2
#             F_2 = norm(L_2 @ x_delta)**2
#             F_last = norm(x_delta)**2
#             C_1 = -F_1 *(alpha_1)**2 
#             C_2 = -F_2 *(alpha_2)**2 
#             D = - (norm_dat - F - (alpha_1 * F_1 + alpha_2 * F_2))**2 / F_last
#             T = (norm_dat - F - (alpha_1 * F_1 + alpha_2 * F_2)) / F_last
#         else:
#             print("")
#             print("Step 2 is satisfied")
#             break
#         #Step 3
        #Step 3
        # if alpha_1 == 0 and alpha_2 == 0:
        #     # alpha_1_new = (2 * C_1) / (c**2 * delta**2 - norm_dat  - D/(T + beta) - ((beta * D) / (beta + T)**2))
        #     # alpha_2_new = (2 * C_2) / (c**2 * delta**2 - norm_dat  - D/(T + beta) - ((beta * D) / (beta + T)**2))
        #     alpha_1_new = 0
        #     alpha_2_new = 0
        #     beta_new = omega * ((2*D /(c**2 * delta**2 - norm_dat + ((T * D) / (beta + T)**2))) - T)
        # elif alpha_1 == 0 and alpha_2 != 0:
        #     # alpha_1_new = (2 * C_1) / (c**2 * delta**2 - norm_dat -(2*C_2/alpha_2) - D/(T + beta) - ((beta * D) / (beta + T)**2))
        #     alpha_1_new = 0
        #     alpha_2_new = (2 * C_2) / (c**2 * delta**2 - norm_dat  - D/(T + beta) - ((beta * D) / (beta + T)**2))
        #     beta_new = omega * ((2*D /(c**2 * delta**2 - norm_dat - (2*C_2/alpha_2) + ((T * D) / (beta + T)**2))) - T)
        # elif alpha_1 != 0 and alpha_2 == 0:
        #     alpha_1_new = (2 * C_1) / (c**2 * delta**2 - norm_dat - D/(T + beta) - ((beta * D) / (beta + T)**2))
        #     alpha_2_new = 0
        #     # alpha_2_new = (2 * C_2) / (c**2 * delta**2 - norm_dat -(2*C_1/alpha_1) - D/(T + beta) - ((beta * D) / (beta + T)**2))
        #     beta_new = omega * ((2*D /(c**2 * delta**2 - norm_dat - (2*C_1/alpha_1) + ((T * D) / (beta + T)**2))) - T)
        # else:
#         alpha_1_new = (2 * C_1) / (c**2 * delta**2 - norm_dat -(2*C_2/alpha_2) - D/T)
#         alpha_2_new = (2 * C_2) / (c**2 * delta**2 - norm_dat -(2*C_1/alpha_1) - D/T)

#         #Step 4
#         if np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) >= ep and alpha_1 > ep and alpha_2 > ep:
#             alpha_1 = alpha_1_new
#             alpha_2 = alpha_2_new
#         else:
#             print("")
#             print("Step 4 is satisfied")
#             break
#         k = k + 1
#     print(f"Total of {k} iterations")
#     print("")
#     return alpha_1,alpha_2, x_delta



# def Multi_Param_LR(data_noisy, G, g,  L_1, L_2, alpha_1, alpha_2, beta, omega, ep, c, delta):
#     #initialization
#     norm_dat = norm(data_noisy)**2
#     k = 1
#     alpha_1_new = alpha_1
#     #Step 4
#     print(f"(np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)) for {k} iteration:", (np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)))
#     if (np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)) >= ep and alpha_1 > ep and alpha_2 > ep and beta > ep:
#         alpha_1 = alpha_1_new
#         alpha_2 = alpha_2_new
#         beta = beta_new
        
#         #Step 2
#         x_delta = np.linalg.inv(G.T @ G + alpha_1 * L_1.T @ L_1 + alpha_2 * L_2.T @ L_2 + beta * np.eye(L_1.shape[0])) @ G.T @ data_noisy
#         print(f"norm(G @ x_delta - data_noisy) for {k} iteration:", norm(G @ x_delta - data_noisy))
#         print(f"c*delta for {k} iteration:", c*delta)
#         # print(f"alpha_1 for {k} iteration:", alpha_1)
#         # print(f"alpha_2 for {k} iteration:", alpha_2)
#         if norm(G @ x_delta - data_noisy) < c*delta:
#             final_alpha_1 = alpha_1
#             final_alpha_2 = alpha_2
#             final_beta = beta
#             final_x_delta = x_delta
#             print("")
#             print("Step 2 is satisfied")
#         else:
#             F = norm(G @ x_delta - data_noisy)**2 + alpha_1 * norm(L_1 @ x_delta)**2 + alpha_2 * norm(L_2 @ x_delta)**2 + beta * norm(x_delta)**2
#             F_1 = norm(L_1 @ x_delta)**2
#             F_2 = norm(L_2 @ x_delta)**2
#             F_last = norm(x_delta)**2
#             C_1 = -F_1 *(alpha_1)**2 
#             C_2 = -F_2 *(alpha_2)**2 
#             D = - (norm_dat - F - (alpha_1 * F_1 + alpha_2 * F_2))**2 / F_last
#             T = (norm_dat - F - (alpha_1 * F_1 + alpha_2 * F_2)) / (F_last - beta)

#         #Step 3
#         if alpha_1 == 0 and alpha_2 == 0:
#             # alpha_1_new = (2 * C_1) / (c**2 * delta**2 - norm_dat  - D/(T + beta) - ((beta * D) / (beta + T)**2))
#             # alpha_2_new = (2 * C_2) / (c**2 * delta**2 - norm_dat  - D/(T + beta) - ((beta * D) / (beta + T)**2))
#             alpha_1_new = 0
#             alpha_2_new = 0
#             beta_new = omega * ((2*D /(c**2 * delta**2 - norm_dat + ((T * D) / (beta + T)**2))) - T)
#         elif alpha_1 == 0 and alpha_2 != 0:
#             # alpha_1_new = (2 * C_1) / (c**2 * delta**2 - norm_dat -(2*C_2/alpha_2) - D/(T + beta) - ((beta * D) / (beta + T)**2))
#             alpha_1_new = 0
#             alpha_2_new = (2 * C_2) / (c**2 * delta**2 - norm_dat  - D/(T + beta) - ((beta * D) / (beta + T)**2))
#             beta_new = omega * ((2*D /(c**2 * delta**2 - norm_dat - (2*C_2/alpha_2) + ((T * D) / (beta + T)**2))) - T)
#         elif alpha_1 != 0 and alpha_2 == 0:
#             alpha_1_new = (2 * C_1) / (c**2 * delta**2 - norm_dat - D/(T + beta) - ((beta * D) / (beta + T)**2))
#             alpha_2_new = 0
#             # alpha_2_new = (2 * C_2) / (c**2 * delta**2 - norm_dat -(2*C_1/alpha_1) - D/(T + beta) - ((beta * D) / (beta + T)**2))
#             beta_new = omega * ((2*D /(c**2 * delta**2 - norm_dat - (2*C_1/alpha_1) + ((T * D) / (beta + T)**2))) - T)
#         else:
#             alpha_1_new = (2 * C_1) / (c**2 * delta**2 - norm_dat -(2*C_2/alpha_2) - D/(T + beta) - ((beta * D) / (beta + T)**2))
#             alpha_2_new = (2 * C_2) / (c**2 * delta**2 - norm_dat -(2*C_1/alpha_1) - D/(T + beta) - ((beta * D) / (beta + T)**2))
#             beta_new = omega * ((2*D /(c**2 * delta**2 - norm_dat - (2*C_2/alpha_2) - (2*C_1/alpha_1) + ((T * D) / (beta + T)**2))) - T)
#     else:
#         print("")
#         print(f"(np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)) for {k} iteration:", (np.abs(alpha_1_new - alpha_1) + np.abs(alpha_2_new - alpha_2) + np.abs(beta_new - beta)))
#         final_alpha_1 = alpha_1
#         final_alpha_2 = alpha_2
#         final_beta = beta
#         final_x_delta = x_delta
#         print("")
#         print("Step 4 is satisfied")
#     k = k + 1
#     print(f"Total of {k} iterations")
#     print("")
#     rel_error = norm(g - final_x_delta)/norm(g)
#     return final_alpha_1,final_alpha_2,final_beta, final_x_delta, rel_error
