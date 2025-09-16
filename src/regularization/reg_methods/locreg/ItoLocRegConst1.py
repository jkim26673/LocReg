# #Ito LocReg Constrainted
# import sys
# import os
# parent = os.path.dirname(os.path.abspath(''))
# sys.path.append(parent)
# # sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
# from Utilities_functions.lsqnonneg import lsqnonneg
# from scipy.optimize import nnls
# import numpy as np
# # import cvxpy as cp
# # import mosek
# # import cvxpy as cp
# from scipy.signal import savgol_filter
# from Utilities_functions.tikhonov_vec import tikhonov_vec
# from regu.csvd import csvd
# import pylops
# from scipy.ndimage import convolve
# from scipy import sparse
# import scipy
# from scipy import linalg as la
# from Utilities_functions.pasha_gcv import Tikhonov
from utils.load_imports.loading import *
### N Parameter Ito problem
def LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter):
    #Initialize the MRR Problem
    TE = np.arange(1,512,4).T
    #Generate the T2 values
    T2 = np.arange(1,201).T
    dT2 = T2[1] - T2[0] 
    #Generate G_matrix
    G = np.zeros((len(TE),len(T2)))
    #For every column in each row, fill in the e^(-TE(i))
    for i in range(len(TE)):
        for j in range(len(T2)):
            G[i,j] = np.exp(-TE[i]/T2[j]) * dT2
    nTE = len(TE)
    nT2 = len(T2)

    def first_deriv(n):
        D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None)
        L = sparse.identity(n)-D
        Lx = L[0:-1, :]
        Lx = Lx.toarray()
        zero_row = np.zeros((1,n))
        zero_row[0,-1] = 1
        Lx = np.vstack([Lx, zero_row])
        return Lx

    def minimize(lam_vector):
        machine_eps = np.finfo(float).eps
        A = (G.T @ G + first_deriv(G.shape[1]).T @ np.diag((lam_vector)) @ first_deriv(G.shape[1]))
        b = (G.T @ data_noisy)
        sol = np.linalg.solve(A,b)
        sol = np.array(sol)
        return sol, A, b


    def minimize(lam_vector):
        machine_eps = np.finfo(float).eps
        eps = 1e-2
        A = (G.T @ G + np.diag(lam_vector))
        ep4 = np.ones(G.shape[1]) * eps
        b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * (lam_vector)
        sol = nnls(A, b, maxiter=1000)[0]
        sol = sol - eps
        sol = np.array(sol)
        sol[sol < 0] = 0
        return sol, A, b

    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2)**2

    def fixed_point_algo(gamma, lam_vec, check):
        """
        gamma: gamma val
        lam_vec: vector of lambdas
        """
        nT2 = G.shape[1]
        # lam_curr = np.sqrt(lam_vec)
        lam_curr = lam_vec
        k = 1

        ep = 1e-2
        # ep = 1e-3
        ep_min = 1e-2
        # epscond = False
        # ini_f_rec = minimize(lam_curr, ep_min, epscond)
        f_old = np.ones(G.shape[1])

        c_arr = []
        lam_arr = []
        sol_arr = []
        # fig, axs = plt.subplots(3, 1, figsize=(6, 6))
        # fig,ax = plt.subplots(5,1,figsize = (12,8))
        # # # Show the initial plot
        # plt.tight_layout()
        # plt.ion()  # Turn on interactive mode
        
        first_f_rec, _, _ = minimize(lam_curr)
        #Uncomment the code below to run the Fixed Point Algo (FPA) using while loop until convergence

        while True:
            #Initial L-Curve solution
            try:
                # curr_f_rec = minimize(lam_curr, ep_min, epscond)
                curr_f_rec, LHS, RHS = minimize(lam_curr)
                if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
                    print(f"curr_f_rec is None after minimization for iteration {k}")
                else:
                    pass
            except Exception as e:
                print("An error occurred during minimization:", e)

            curr_noise = (G @ curr_f_rec) - data_noisy
            delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
            prev = np.linalg.norm(delta_p)
            LHS_temp = LHS.copy()
            while True:
                curr_f_rec = curr_f_rec - delta_p
                curr_f_rec[curr_f_rec < 0] = 0
                curr_noise = G @ curr_f_rec - data_noisy
                try:
                    delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
                except RuntimeWarning as e:
                    print("error with delta_p calculation")
                    pass
                if np.abs(np.linalg.norm(delta_p) / prev - 1) < 1e-2:
                    break
                prev = np.linalg.norm(delta_p)

            curr_f_rec = curr_f_rec - delta_p
            curr_f_rec[curr_f_rec < 0] = 0

            #track.append(np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x))
            # if (np.linalg.norm(psi_lam/phi_new)) < 1e-4:
            #     print("condition passed")
            #     print("np.linalg.norm(psi_lam/ phi_new)", np.linalg.norm(psi_lam/ phi_new))
            #     psi_lam = psi_lam + ep_min
            # psi_lam = list(psi_lam)


            #Get new solution with new lambda vector

            if check == True:
                axs[0].plot(T2,g, color = "black", label = "ground truth")
                axs[0].plot(T2, curr_f_rec, label = "reconstruction")
                axs[1].semilogy(T2, lam_curr, label = "lambdas")
                # Redraw the plot
                plt.draw()
                plt.tight_layout()
                plt.pause(0.01)

            machine_eps = np.finfo(float).eps

            #Update lambda: then check
            #New Lambda find the new residual and the new penalty
            phi_new = np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2
            # print("phi_new",phi_new)
            # lam_new = []
            # for i in range(len(curr_f_rec)):
            #     nu_i = (phi_new + sum(curr_f_rec[j] for j in range(len(curr_f_rec)) if j != i)) / (curr_f_rec[i] + 1e-3)
            #     lam_new.append(nu_i)
            # add_port = [phi_new + lam_curr[i] * curr_f_rec[i] for i in range(len(curr_f_rec))]
            
            psi_lam = [curr_f_rec[i] for i in range(len(curr_f_rec))]

            #define scaling factor;
            c = 1/gamma
            # c = ((gamma**gamma)/((1+gamma)**(1+gamma)))/gamma
            # c1 =((gamma**gamma)/((1+gamma)**(1+gamma)))/(gamma + ((gamma**gamma)/((1+gamma)**(1+gamma))))
            # c = 1/(1 + gamma)
            c1 = 1
            # print("gamma", gamma)
            # c = np.std(data_noisy - - np.dot(G,first_f_rec))
            c_arr.append(c)

            psi_lam = np.array(psi_lam)
            # lam_new = np.array(lam_new)
            # lam_new = c * lam_new
            # print("psi_lam",np.median(psi_lam))
            
            ep_2 = 1e-2
            # ep_min = 1e-5
            # print("np.linalg.norm(psi_lam):", np.linalg.norm(psi_lam))
            #STEP 4
            #redefine new lam
            # plt.plot(lam_curr)
            # plt.show()
            # lam_new = c * (phi_new / (psi_lam + ((lam_curr > 1e2) & (lam_curr < 1e-3)) * ep_2) + (lam_curr > 1e3) * ep_min)
            # lam_new = c * (phi_new / (psi_lam + machine_eps))
            # lam_new = c * (phi_new / (psi_lam + (psi_lam < 0.5 * np.median(curr_f_rec)) * machine_eps))
            lam_new = c * (phi_new / (psi_lam + ep_min))
            # print("5 * np.min(curr_f_rec)",5 * np.min(curr_f_rec))
            # lam_new2 = c * (phi_new / (psi_lam + 1e-5))
            # curr_f_rec3 = minimize(lam_new)
            # curr_f_rec2 = minimize(lam_new2)
            # print("np.median(first_f_rec)",np.median(first_f_rec))
            # # print("np.where(psi_lam > 1.5 * np.median(first_f_rec))", np.where(psi_lam < np.median(first_f_rec)))
            # # lam_new = c * (phi_new / (psi_lam + (psi_lam < machine_eps) * ep_min))
            # ax[0].plot(lam_curr)
            # ax[1].plot(curr_f_rec3)
            # ax[2].plot(lam_new)
            # ax[3].plot(lam_new2)
            # ax[4].plot(curr_f_rec2)
            # # ax[2].set_ylim(0, 
            # # upper_limit_2)
            # # ax[2].set_ylim(0, 25)
            # # ax[3].set_ylim(0, 10) 
            # # ax[2].set_ylim(0, upper_limit_2)
            # ax[0].legend(["lam_curr"])
            # ax[1].legend(["curr_f_rec_lam_new"])
            # ax[2].legend(["lam_global_reg"])
            # ax[3].legend(["lam_peaks_reg"])
            # ax[4].legend(["curr_f_rec_lam_peaks_reg"])

            # plt.draw()
            # plt.tight_layout()
            # plt.pause(0.0000001)            # if np.median(psi_lam) < 1e-4:
            #     lam_new = c * (phi_new / (psi_lam + ep_min))
            # else:
            #     lam_new = c * (phi_new / (psi_lam))
            # print("Lam_new.shape", lam_new.shape)
            # cs = c * np.ones(len(psi_lam))

            #If doesnt converge; update f

            #Step4: Check stopping criteria based on relative change of regularization parameter eta
            #or the  inverse solution
            #update criteria of lambda
            # if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) + (np.linalg.norm(lam_new-lam_curr)/np.linalg.norm(lam_curr)) < ep or k >= maxiter:
            if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) < ep or k >= maxiter:
                # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old))
                # print("ep value: ", ep)
                if k >= maxiter:
                    print("max hit")
                else:
                    pass
                c_arr_fin = np.array(c_arr)
                lam_arr_fin = np.array(lam_arr)
                sol_arr_fin = np.array(sol_arr)
                # plt.ioff()
                # plt.show()
                # print(f"Total of {k} iterations")
                return curr_f_rec, lam_curr, c_arr_fin, lam_arr_fin, sol_arr_fin
            else:
                ep_min = ep_min / 1.2
                # print("ep_min",ep_min)
                if ep_min <= 1e-4:
                    ep_min = 1e-4
                # print(f"Finished Iteration {k}")
                lam_curr = lam_new
                f_old = curr_f_rec
                k = k + 1
                lam_arr.append(lam_new)
                sol_arr.append(curr_f_rec)


        #Running the FPA iteration by iteration
        # testiter = 5
        # for k in range(testiter):
        #     try:
        #         # curr_f_rec = minimize(lam_curr, ep_min, epscond)
        #         curr_f_rec = minimize(lam_curr)
        #         if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
        #             print(f"curr_f_rec is None after minimization for iteration {k}")
        #         else:
        #             pass
        #     except Exception as e:
        #         print("An error occurred during minimization:", e)
        #
        #     # Get new solution with new lambda vector
        #
        #     # Update lambda: then check
        #     # New Lambda find the new residual and the new penalty
        #     phi_new = np.linalg.norm(data_noisy - np.dot(G, curr_f_rec), 2)**2
        #     psi_lam = [curr_f_rec[i] * lam_curr[i] for i in range(len(lam_curr))]
        #
        #
        #     # define scaling factor;
        #     c = 1 / (1 + gamma)
        #     c = ((gamma**gamma)/((1+gamma)**(1+gamma)))
        #     # c = np.std(data_noisy - - np.dot(G,curr_f_rec))/gamma
        #     # c = ((gamma**gamma)/((1+gamma)**(1+gamma)))/gamma
        #
        #     c_arr.append(c)
        #
        #     # STEP 4
        #     # redefine new lam
        #     lam_new = c * (phi_new / psi_lam)
        #
        #     #Make terms into arrays
        #     psi_lam = np.array(psi_lam)
        #     lam_new = np.array(lam_new)
        #     machine_eps = np.finfo(float).eps
        #
        #     #Try Yvonne's idea
        #     if np.any(psi_lam/phi_new) < machine_eps:
        #         print("condition satisfied")
        #
        #     # If doesnt converge; update f
        #
        #     #Plot iteration by iteration
        #     if check == True:
        #         axs[0].plot(T2, g, color="black", label="ground truth")
        #         axs[0].plot(T2, curr_f_rec, label="reconstruction")
        #         axs[1].semilogy(T2, lam_curr, label="lambdas")
        #         axs[2].semilogy(T2, lam_new, label="new lambda")
        #         # axs[3].semilogy(T2, test, label="lambda_new")
        #         # axs[4].semilogy(T2, np.array(psi_lam2), label="lambda_new * reconstruction")
        #
        #         # Redraw the plot
        #         plt.draw()
        #         # axs[0].legend()
        #         # axs[1].legend()
        #         # axs[2].legend()
        #         # axs[3].legend()
        #         # axs[4].legend()
        #         plt.tight_layout()
        #         plt.pause(0.001)
        #     else:
        #         pass
        #
        #     # Step4: Check stopping criteria based on relative change of regularization parameter eta
        #     # or the  inverse solution
        #     # update criteria of lambda
        #     if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) + (np.linalg.norm(lam_new - lam_curr) / np.linalg.norm(lam_curr)) < ep or k == maxiter-1 or  k >= maxiter:
        #         # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old))
        #         # print("ep value: ", ep)
        #         c_arr_fin = np.array(c_arr)
        #         lam_arr_fin = np.array(lam_arr)
        #         sol_arr_fin = np.array(sol_arr)
        #         plt.ioff()
        #         plt.show()
        #         print(f"Total of {k} iterations")
        #         return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
        #     else:
        #         # ep_min = ep_min / 1.2
        #         # if ep_min <= 1e-4:
        #         #     ep_min = 1e-4
        #         # print(f"Finished Iteration {k}")
        #         k = k + 1
        #         lam_curr = lam_new
        #         f_old = curr_f_rec
        #         lam_arr.append(lam_new)
        #         sol_arr.append(curr_f_rec)

    #MAIN CODE FOR ITO LR:

    #Step 1: Initialize gamma and lambda as lam_vec
    lam_vec = lam_ini * np.ones(G.shape[1])
    lam_vec = np.sqrt(lam_vec)
    #Step 2:Run FPA until convergence
    check = False 
    best_f_rec1, fin_lam1, c_arr_fin1, lam_arr_fin, sol_arr_fin = fixed_point_algo(gamma_init, lam_vec, check = False)
    # print("first FPA is done")
    # fin_lam1 = np.sqrt(fin_lam1)

    #Step 3: Calculate new noise level (phi_resid)
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    
    #Step 4: Calculate and update new gamma:
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G,zero_vec, data_noisy)

    #If residual is L2:
    gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25

    #If residual is L1:
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.5
 
    #Step 4: Perform fixed point algo with new gamma value
    # check = True ; if want to print iteration by ieration
    fin_lam1 = np.sqrt(fin_lam1)
    best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin  = fixed_point_algo(gamma_new, fin_lam1, check = False)
    fin_lam2 = np.sqrt(fin_lam2)
    return best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin


def LocReg_Ito_C(data_noisy, G, lam_ini, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback):
    lam_first = lam_ini * np.ones(G.shape[1])
    # U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
    def first_deriv(n):
        D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None)
        L = sparse.identity(n)-D
        Lx = L[0:-1, :]
        Lx = Lx.toarray()
        zero_row = np.zeros((1,n))
        zero_row[0,-1] = 1
        Lx = np.vstack([Lx, zero_row])
        return Lx

    def minimize(lam_vector):
        machine_eps = np.finfo(float).eps
        A = (G.T @ G + first_deriv(G.shape[1]).T @ np.diag((lam_vector)) @ first_deriv(G.shape[1]))
        b = (G.T @ data_noisy)
        sol = nnls(A, b, maxiter=1000)[0]
        sol = np.array(sol)
        return sol, A, b

    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2)**2

    def fixed_point_algo(gamma, lam_vec, eps1, ep_min, eps_cut, eps_floor, check):
        nT2 = G.shape[1]
        lam_first = lam_vec
        lam_curr = lam_vec
        k = 1
        ep = eps1
        ep_min = 1e-2
        f_old = np.ones(G.shape[1])
        lam_arr = []
        first_f_rec, _ , _ = minimize(lam_curr)
        while True:
            #Minimization
            try:
                # curr_f_rec = minimize(lam_curr, ep_min, epscond)
                curr_f_rec, LHS, RHS = minimize(lam_curr)
                if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
                    print(f"curr_f_rec is None after minimization for iteration {k}")
                else:
                    pass
            except Exception as e:
                print("An error occurred during minimization:", e)
            # if np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2 > 10 * np.linalg.norm(data_noisy - np.dot(G,first_f_rec), 2)**2:
            #     curr_f_rec = first_f_rec
            #     lam_new = lam_first
            #     return curr_f_rec, lam_new
            #Feedback
            if feedback == True:
                curr_noise = (G @ curr_f_rec) - data_noisy
                delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                prev = np.linalg.norm(delta_p)
                LHS_temp = LHS.copy()
                iteration = 1
                while True and iteration < 20:
                    curr_f_rec = curr_f_rec - delta_p
                    curr_f_rec[curr_f_rec < 0] = 0
                    curr_noise = G @ curr_f_rec - data_noisy
                    try:
                        delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
                    except RuntimeWarning as e:
                        print("error with delta_p calculation")
                        pass
                    if np.abs(np.linalg.norm(delta_p) / (prev) - 1) < 5*1e-2:
                        break
                    prev = np.linalg.norm(delta_p)
                    iteration += 1
                curr_f_rec = curr_f_rec - delta_p
                # print(curr_f_rec)
            else:
                pass
            #Update lambda: then check
            #New Lambda find the new residual and the new penalty
            phi_new = np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2
            psi_lam  = np.abs(first_deriv(G.shape[1]) @ curr_f_rec)
            c = 0.05

            # c_arr.append(c)
            psi_lam = np.array(psi_lam)
            # print("psi_lam",np.median(psi_lam))

            machine_eps = np.finfo(float).eps
            lam_new = c * (phi_new**(1 - exp) / (np.abs(psi_lam) + ep_min))
            #If doesnt converge; update f

            #Step4: Check stopping criteria based on relative change of regularization parameter eta
            #or the  inverse solution
            #update criteria of lambda
            # if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) + (np.linalg.norm(lam_new-lam_curr)/np.linalg.norm(lam_curr)) < ep or k >= maxiter:
            if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) < ep or k >= maxiter:
                # c_arr_fin = np.array(c_arr)
                # lam_arr_fin = np.array(lam_arr)
                # sol_arr_fin = np.array(sol_arr)
                if k >= maxiter:
                    print("max hit")
                else:
                    pass
                return curr_f_rec, lam_new
            else:
                ep_min = ep_min/eps_cut
                if ep_min <= eps_floor:
                    ep_min = eps_floor
                lam_curr = lam_new
                f_old = curr_f_rec
                k = k + 1
                lam_arr.append(lam_new)
                # sol_arr.append(curr_f_rec)

    #MAIN CODE FOR ITO LR:
    #Step 1: Initialize gamma and lambda as lam_vec
    lam_vec = lam_ini * np.ones(G.shape[1])
    lam_vec = (lam_vec)**2
    best_f_rec2, fin_lam2  = fixed_point_algo(gamma_init, lam_vec,eps1, ep_min, eps_cut, eps_floor, check = False)
    # fin_lam2 = np.sqrt(fin_lam2)
    return best_f_rec2, fin_lam2


### N Parameter Unconstrained Ito problem
def LocReg_Ito_C_2(data_noisy, G, lam_ini, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback):
    lam_first = lam_ini * np.ones(G.shape[1])
    U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
    def first_deriv(n):
        D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None)
        L = sparse.identity(n)-D
        Lx = L[0:-1, :]
        Lx = Lx.toarray()
        return Lx
    
    def minimize(lam_vector):
        machine_eps = np.finfo(float).eps
        A = (G.T @ G + np.diag(lam_vector))
        b = (G.T @ data_noisy)
        sol = nnls(A, b, maxiter=1000)[0]
        sol = np.array(sol)
        return sol, A, b

    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2)**2

    def fixed_point_algo(gamma, lam_vec, eps1, ep_min, eps_cut, eps_floor, check):
        """
        gamma: gamma val
        lam_vec: vector of lambdas
        """
        nT2 = G.shape[1]
        lam_curr = lam_vec
        k = 1
        ep = eps1
        ep_min = 1e-2
        f_old = np.ones(G.shape[1])
        c_arr = []
        lam_arr = []
        sol_arr = []
        first_f_rec, _ = tikhonov_vec(U, s, V, data_noisy, lam_curr, x_0 = None, nargin = 5)

        #Uncomment the code below to run the Fixed Point Algo (FPA) using while loop until convergence
        while True:
            #Minimization
            try:
                curr_f_rec, _ = tikhonov_vec(U, s, V, data_noisy, lam_curr, x_0 = None, nargin = 5)
                LHS = G.T @ G + np.diag(lam_curr**2)
                RHS = G.T @ data_noisy
                if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
                    print(f"curr_f_rec is None after minimization for iteration {k}")
                else:
                    pass
            except Exception as e:
                print("An error occurred during minimization:", e)

            #Feedback
            # if feedback == True:
            #     curr_noise = (G @ curr_f_rec) - data_noisy
            #     delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
            #     prev = np.linalg.norm(delta_p)
            #     LHS_temp = LHS.copy()
            #     while True:
            #         curr_f_rec = curr_f_rec - delta_p
            #         # curr_f_rec[curr_f_rec < 0] = 0
            #         curr_noise = G @ curr_f_rec - data_noisy
            #         try:
            #             delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
            #         except RuntimeWarning as e:
            #             print("error with delta_p calculation")
            #             pass
            #         # print("np.abs(np.linalg.norm(delta_p) / (prev + 1e-3) - 1)", np.abs(np.linalg.norm(delta_p) / (prev) - 1))
            #         if np.abs(np.linalg.norm(delta_p) / (prev ) - 1) < 1e-2:
            #             break
            #         prev = np.linalg.norm(delta_p)
            #     curr_f_rec = curr_f_rec - delta_p
            # else:
            #     pass
            if feedback == True:
                curr_noise = (G @ curr_f_rec) - data_noisy
                delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                prev = np.linalg.norm(delta_p)
                LHS_temp = LHS.copy()
                iteration = 1
                while True and iteration < 20:
                    curr_f_rec = curr_f_rec - delta_p
                    curr_f_rec[curr_f_rec < 0] = 0
                    curr_noise = G @ curr_f_rec - data_noisy
                    try:
                        delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
                    except RuntimeWarning as e:
                        print("error with delta_p calculation")
                        pass
                    if np.abs(np.linalg.norm(delta_p) / (prev) - 1) < 5*1e-2:
                        break
                    prev = np.linalg.norm(delta_p)
                    iteration += 1
                curr_f_rec = curr_f_rec - delta_p
                # print(curr_f_rec)
            else:
                pass
            

            #Update lambda: then check
            #New Lambda find the new residual and the new penalty
            phi_new = np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2

            psi_lam = np.abs(curr_f_rec)

            # c = 1/(gamma)
            c = 0.05

            c_arr.append(c)

            psi_lam = np.array(psi_lam)

            #STEP 4
            #redefine new lam
            machine_eps = np.finfo(float).eps

            lam_new = c * (phi_new**(1 - exp) / (np.abs(psi_lam) + ep_min))

            #Step4: Check stopping criteria based on relative change of regularization parameter eta
            #or the  inverse solution
            #update criteria of lambda
            # if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) + (np.linalg.norm(lam_new-lam_curr)/np.linalg.norm(lam_curr)) < ep or k >= maxiter:
            if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) < ep or k >= maxiter:
                # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old))
                # print("ep value: ", ep)
                c_arr_fin = np.array(c_arr)
                lam_arr_fin = np.array(lam_arr)
                sol_arr_fin = np.array(sol_arr)
                if k >= maxiter:
                    print("max hit")
                else:
                    pass

                return curr_f_rec, lam_new
            else:
                ep_min = ep_min/eps_cut
                if ep_min <= eps_floor:
                    ep_min = eps_floor

                lam_curr = lam_new
                f_old = curr_f_rec
                k = k + 1
                lam_arr.append(lam_new)
                sol_arr.append(curr_f_rec)

    #MAIN CODE FOR ITO LR:

    #Step 1: Initialize gamma and lambda as lam_vec
    lam_vec = lam_ini * np.ones(G.shape[1])
    best_f_rec2, fin_lam2  = fixed_point_algo(gamma_init, lam_vec,eps1, ep_min, eps_cut, eps_floor, check = False)
    return best_f_rec2, fin_lam2


def LocReg_Ito_C_3(data_noisy, G, lam_ini, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback):
    lam_first = lam_ini * np.ones(G.shape[1])
    U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
    def first_deriv(n):
        D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None)
        L = sparse.identity(n)-D
        Lx = L[0:-1, :]
        Lx = Lx.toarray()
        return Lx

    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2)**2

    def fixed_point_algo(gamma, lam_vec, eps1, ep_min, eps_cut, eps_floor, check):
        """
        gamma: gamma val
        lam_vec: vector of lambdas
        """
        nT2 = G.shape[1]
        # lam_curr = np.sqrt(lam_vec)
        lam_curr = lam_vec
        k = 1
        ep = eps1
        ep_min = 1e-2
        f_old = np.ones(G.shape[1])
        c_arr = []
        lam_arr = []
        sol_arr = []
        first_f_rec, _ = tikhonov_vec(U, s, V, data_noisy, lam_curr, x_0 = None, nargin = 5)
        #Uncomment the code below to run the Fixed Point Algo (FPA) using while loop until convergence
        while True:
            #Minimization
            try:
                curr_f_rec, _ = tikhonov_vec(U, s, V, data_noisy, lam_curr, x_0 = None, nargin = 5)
                LHS = G.T @ G + np.diag(lam_curr)
                RHS = G.T @ data_noisy
                if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
                    print(f"curr_f_rec is None after minimization for iteration {k}")
                else:
                    pass
            except Exception as e:
                print("An error occurred during minimization:", e)
            #Feedback
            # if feedback == True:
            #     curr_noise = (G @ curr_f_rec) - data_noisy
            #     delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
            #     prev = np.linalg.norm(delta_p)
            #     LHS_temp = LHS.copy()
            #     while True:
            #         curr_f_rec = curr_f_rec - delta_p
            #         # curr_f_rec[curr_f_rec < 0] = 0
            #         curr_noise = G @ curr_f_rec - data_noisy
            #         try:
            #             delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
            #         except RuntimeWarning as e:
            #             print("error with delta_p calculation")
            #             pass
            #         if np.abs(np.linalg.norm(delta_p) / (prev) - 1) < 1e-2:
            #             break
            #         prev = np.linalg.norm(delta_p)
            #     curr_f_rec = curr_f_rec - delta_p
            # else:
            #     pass
            if feedback == True:
                curr_noise = (G @ curr_f_rec) - data_noisy
                delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                prev = np.linalg.norm(delta_p)
                LHS_temp = LHS.copy()
                iteration = 1
                while True and iteration < 20:
                    curr_f_rec = curr_f_rec - delta_p
                    curr_f_rec[curr_f_rec < 0] = 0
                    curr_noise = G @ curr_f_rec - data_noisy
                    try:
                        delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
                    except RuntimeWarning as e:
                        print("error with delta_p calculation")
                        pass
                    if np.abs(np.linalg.norm(delta_p) / (prev) - 1) < 5*1e-2:
                        break
                    prev = np.linalg.norm(delta_p)
                    iteration += 1
                curr_f_rec = curr_f_rec - delta_p
                # print(curr_f_rec)
            else:
                pass
            #Update lambda: then check
            #New Lambda find the new residual and the new penalty
            phi_new = np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2
            psi_lam = curr_f_rec
            c = 1/(gamma)
            c1 = 1
            c_arr.append(c)
            psi_lam = np.array(psi_lam)
            #STEP 4
            #redefine new lam
            machine_eps = np.finfo(float).eps
            lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))
            #Step4: Check stopping criteria based on relative change of regularization parameter eta
            #or the  inverse solution
            #update criteria of lambda
            if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) < ep or k >= maxiter:
                c_arr_fin = np.array(c_arr)
                lam_arr_fin = np.array(lam_arr)
                sol_arr_fin = np.array(sol_arr)
                if k >= maxiter:
                    print("max hit")
                else:
                    pass
                return curr_f_rec, lam_new
            else:
                # ep_min = ep_min / 5
                ep_min = ep_min/eps_cut
                # print("ep_min",ep_min)
                if ep_min <= eps_floor:
                    ep_min = eps_floor
                lam_curr = lam_new
                f_old = curr_f_rec
                k = k + 1
                lam_arr.append(lam_new)
                sol_arr.append(curr_f_rec)

    #MAIN CODE FOR ITO LR:

    #Step 1: Initialize gamma and lambda as lam_vec
    lam_vec = lam_ini * np.ones(G.shape[1])
    check = False 
    best_f_rec1, fin_lam1 = fixed_point_algo(gamma_init, lam_vec, eps1, ep_min, eps_cut, eps_floor, check = False)
    #Step 3: Calculate new noise level (phi_resid)
    
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    #Step 4: Calculate and update new gamma:
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G,zero_vec, data_noisy)
    # If residual is L2:
    gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25
    best_f_rec2, fin_lam2  = fixed_point_algo(gamma_new, fin_lam1,eps1, ep_min, eps_cut, eps_floor, check = False)
    return best_f_rec2, fin_lam2, gamma_new

def LocReg_Ito_C_4(data_noisy, G, lam_ini, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback):
    lam_first = lam_ini * np.ones(G.shape[1])
    U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
    def first_deriv(n):
        D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None)
        L = sparse.identity(n)-D
        Lx = L[0:-1, :]
        Lx = Lx.toarray()
        zero_row = np.zeros((1,n))
        zero_row[0,-1] = 1
        Lx = np.vstack([Lx, zero_row])
        return Lx
    
    def minimize(lam_vector):
        machine_eps = np.finfo(float).eps
        A = (G.T @ G + first_deriv(G.shape[1]).T @ np.diag(lam_vector) @ first_deriv(G.shape[1]))
        b = (G.T @ data_noisy)
        sol = nnls(A, b, maxiter=1000)[0]
        sol = np.array(sol)
        return sol, A, b

    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2)**2

    def fixed_point_algo(gamma, lam_vec, eps1, ep_min, eps_cut, eps_floor, check):
        """
        gamma: gamma val
        lam_vec: vector of lambdas
        """
        nT2 = G.shape[1]
        lam_curr = lam_vec
        k = 1
        ep = eps1
        ep_min = 1e-2
        f_old = np.ones(G.shape[1])
        c_arr = []
        lam_arr = []
        sol_arr = []
        first_f_rec, _ , _ = minimize(lam_curr)
        #Uncomment the code below to run the Fixed Point Algo (FPA) using while loop until convergence
        while True:
            #Minimization
            try:
                curr_f_rec, LHS, RHS = minimize(lam_curr)
                if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
                    print(f"curr_f_rec is None after minimization for iteration {k}")
                else:
                    pass
            except Exception as e:
                print("An error occurred during minimization:", e)
            #Feedback
            # if feedback == True:
            #     curr_noise = (G @ curr_f_rec) - data_noisy
            #     delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
            #     prev = np.linalg.norm(delta_p)
            #     LHS_temp = LHS.copy()
            #     while True:
            #         curr_f_rec = curr_f_rec - delta_p
            #         # curr_f_rec[curr_f_rec < 0] = 0
            #         curr_noise = G @ curr_f_rec - data_noisy
            #         try:
            #             delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
            #         except RuntimeWarning as e:
            #             print("error with delta_p calculation")
            #             pass
            #         if np.abs(np.linalg.norm(delta_p) / (prev) - 1) < 1e-1:
            #             break
            #         prev = np.linalg.norm(delta_p)
            #     curr_f_rec = curr_f_rec - delta_p
            # else:
            #     pass
            if feedback == True:
                curr_noise = (G @ curr_f_rec) - data_noisy
                delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                prev = np.linalg.norm(delta_p)
                LHS_temp = LHS.copy()
                iteration = 1
                while True and iteration < 20:
                    curr_f_rec = curr_f_rec - delta_p
                    curr_f_rec[curr_f_rec < 0] = 0
                    curr_noise = G @ curr_f_rec - data_noisy
                    try:
                        delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
                    except RuntimeWarning as e:
                        print("error with delta_p calculation")
                        pass
                    if np.abs(np.linalg.norm(delta_p) / (prev) - 1) < 5*1e-2:
                        break
                    prev = np.linalg.norm(delta_p)
                    iteration += 1
                curr_f_rec = curr_f_rec - delta_p
                # print(curr_f_rec)
            else:
                pass
            #Update lambda: then check
            #New Lambda find the new residual and the new penalty
            phi_new = np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2
            psi_lam = [curr_f_rec[i] for i in range(len(curr_f_rec))]
            c = 1/(gamma)
            c_arr.append(c)
            psi_lam = np.array(psi_lam)
            machine_eps = np.finfo(float).eps
            lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))

            #Step4: Check stopping criteria based on relative change of regularization parameter eta
            #or the  inverse solution
            #update criteria of lambda
            if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) < ep or k >= maxiter:
                c_arr_fin = np.array(c_arr)
                lam_arr_fin = np.array(lam_arr)
                sol_arr_fin = np.array(sol_arr)
                if k >= maxiter:
                    print("max hit")
                else:
                    pass
                return curr_f_rec, lam_new
            else:
                ep_min = ep_min/eps_cut
                if ep_min <= eps_floor:
                    ep_min = eps_floor
                lam_curr = lam_new
                f_old = curr_f_rec
                k = k + 1
                lam_arr.append(lam_new)
                sol_arr.append(curr_f_rec)
    #MAIN CODE FOR ITO LR:

    #Step 1: Initialize gamma and lambda as lam_vec
    lam_vec = lam_ini * np.ones(G.shape[1])
    lam_vec = (lam_vec)**2
    #Step 2:Run FPA until convergence
    check = False 
    best_f_rec1, fin_lam1 = fixed_point_algo(gamma_init, lam_vec, eps1, ep_min, eps_cut, eps_floor, check = False)
    #Step 3: Calculate new noise level (phi_resid)
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    #Step 4: Calculate and update new gamma:
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G,zero_vec, data_noisy)
    # If residual is L2:
    gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25
    fin_lam1 = (fin_lam1)**2
    best_f_rec2, fin_lam2  = fixed_point_algo(gamma_new, fin_lam1,eps1, ep_min, eps_cut, eps_floor, check = False)
    # fin_lam2 = np.sqrt(fin_lam2)
    return best_f_rec2, fin_lam2,gamma_new