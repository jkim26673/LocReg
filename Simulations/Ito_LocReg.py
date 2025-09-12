#!/usr/bin/env python3
import sys
# sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
sys.path.append(".")
# from Utilities_functions.lsqnonneg import lsqnonneg
from scipy.optimize import nnls
import numpy as np
import cvxpy as cp
import os
import mosek
import cvxpy as cp
from scipy.signal import savgol_filter
from Utilities_functions.tikhonov_vec import tikhonov_vec
from regu.csvd import csvd
import pylops
from scipy.ndimage import convolve
from scipy import sparse
import scipy
from scipy import linalg as la
from Utilities_functions.pasha_gcv import Tikhonov
from regu.nonnegtik_hnorm import nonnegtik_hnorm

mosek_lic_path = "/Users/steveh/Downloads/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_lic_path
# % Select interior-point optimizer... (integer parameter)
# % ... without basis identification (integer parameter)
# param.MSK_IPAR_INTPNT_BASIS = 'MSK_BI_NEVER';
# % Set relative gap tolerance (double parameter)
# param.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1.0e-7;

### N Parameter Unconstrained Ito problem
def LocReg_Ito_UC(data_noisy, G, lam_ini, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback):
    lam_first = lam_ini * np.ones(G.shape[1])
    # U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
    def first_deriv(n):
        D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None)
        L = sparse.identity(n)-D
        Lx = L[0:-1, :]
        Lx = Lx.toarray()
        return Lx
    
    def minimize(lam_vector):
        machine_eps = np.finfo(float).eps

        # eps = 1e-2
        # eps = 1e-3
        # print(np.diag(lam_vector**2).shape)
        # print(first_deriv(G.shape[1]).shape)

        A = (G.T @ G + first_deriv(G.shape[1]).T  @ np.diag(lam_vector)) @ first_deriv(G.shape[1])
        # ep4 = np.ones(G.shape[1]) * eps
        # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vector
        # A = (G.T @ G + np.diag(lam_vector))
        # ep4 = np.ones(G.shape[1]) * eps
        # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vector
        b = (G.T @ data_noisy)
        # sol = nnls(A, b, maxiter=1000)[0]
        sol = np.linalg.solve(A,b)
        # sol = np.linalg.solve(A,b)
        # print("sol",sol)
        # if sol == np.zeros(len(sol)):
        
        # sol = np.linalg.solve(A,b)
        # sol = sol - eps
        # print(np.any(sol < 0))
        # machine_eps = np.finfo(float).eps
        # print(type(sol))
        sol = np.array(sol)
        # sol[sol < 0] = 0
        # sol[sol < 0] = machine_eps
        return sol, A, b

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

        # ep = 1e-2
        # ep = 1e-1
        ep = eps1
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
        
        first_f_rec, _ , _ = minimize(lam_curr)
        # first_f_rec, _ = tikhonov_vec(U, s, V, data_noisy, lam_curr, x_0 = None, nargin = 5)

        #Uncomment the code below to run the Fixed Point Algo (FPA) using while loop until convergence
        while True:
            #Minimization
            try:
                # curr_f_rec = minimize(lam_curr, ep_min, epscond)
                curr_f_rec, LHS, RHS = minimize(lam_curr)
                # curr_f_rec, _ = tikhonov_vec(U, s, V, data_noisy, lam_curr, x_0 = None, nargin = 5)
                # LHS = (G.T @ G + first_deriv(G.shape[1]).T @ first_deriv(G.shape[1]) @ np.diag(lam_vector))
                # RHS = G.T @ data_noisy
                if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
                    print(f"curr_f_rec is None after minimization for iteration {k}")
                else:
                    pass
            except Exception as e:
                print("An error occurred during minimization:", e)

            #Feedback
            if feedback == True:
                curr_noise = (G @ curr_f_rec) - data_noisy
                delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                prev = np.linalg.norm(delta_p)
                LHS_temp = LHS.copy()
                while True:
                    curr_f_rec = curr_f_rec - delta_p
                    # curr_f_rec[curr_f_rec < 0] = 0
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
            else:
                pass
            # curr_f_rec[curr_f_rec < 0] = 0

            # if (np.linalg.norm(psi_lam/phi_new)) < 1e-4:
            #     print("condition passed")
            #     print("np.linalg.norm(psi_lam/ phi_new)", np.linalg.norm(psi_lam/ phi_new))
            #     psi_lam = psi_lam + ep_min
            # psi_lam = list(psi_lam)

            #Get new solution with new lambda vector

            # if check == True:
            #     axs[0].plot(T2,g, color = "black", label = "ground truth")
            #     axs[0].plot(T2, curr_f_rec, label = "reconstruction")
            #     axs[1].semilogy(T2, lam_curr, label = "lambdas")
            #     # Redraw the plot
            #     plt.draw()
            #     plt.tight_layout()
            #     plt.pause(0.01)


            #Update lambda: then check
            #New Lambda find the new residual and the new penalty
            phi_new = np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2
            # print("phi_new",phi_new)
            psi_lam = [curr_f_rec[i] for i in range(len(curr_f_rec))]

            #define scaling factor;
            # c = 1/(gamma)
            c = 0.05
            # c = ((gamma**gamma)/((1+gamma)**(1+gamma)))
            # c1 =((gamma**gamma)/((1+gamma)**(1+gamma)))/(gamma + ((gamma**gamma)/((1+gamma)**(1+gamma))))
            # c = 1/(1 + gamma)
            c1 = 1
            # print("gamma", gamma)
            # c = np.std(data_noisy - - np.dot(G,first_f_rec))
            c_arr.append(c)

            psi_lam = np.array(psi_lam)
            # print("psi_lam",np.median(psi_lam))
            
            # ep_min = 1e-10
            # print("np.linalg.norm(psi_lam):", np.linalg.norm(psi_lam))
            #STEP 4
            #redefine new lam
            machine_eps = np.finfo(float).eps
            # plt.plot(lam_curr)
            # plt.show()
            # lam_new = c * (phi_new / (psi_lam + ((lam_curr > 1e2) & (lam_curr < 1e-3)) * ep_2) + (lam_curr > 1e3) * ep_min)
            # lam_new = c * (phi_new / (psi_lam + machine_eps))
            # lam_new = c * (phi_new / (psi_lam + (psi_lam < 0.5 * np.median(curr_f_rec)) * machine_eps))
            # ep_min = 1e-2
            lam_new = c * (phi_new**(1 - exp) / (np.abs(psi_lam) + ep_min))
            # lam_new = c * (phi_new**(1-1/2) / (np.abs(psi_lam) + ep_min))

            # lam_new = []
            # for i in range(len(psi_lam)):
            #     # Exclude the i-th element
            #     psi_lam_excluded = np.delete(psi_lam, i)
            #     lam_curr_excluded = np.delete(lam_curr, i)
            #     # Calculate val
            #     val = c * (phi_new + np.sum(lam_curr_excluded * psi_lam_excluded)) / (np.abs(psi_lam[i]) + ep_min)
            #     # Append to lam_new
            #     lam_new.append(val)
            # lam_new = np.array(lam_new)
            # print("lam_new",lam_new)
            # lam_new = c * lam_new
            # lam_new = savgol_filter(lam_new, window_length=11, polyorder=8)

            # lam_new = np.abs(lam_new)
            # lam_new[lam_new < 0] = 
            # if np.any(lam_new < 0):
            #     print(np.where(lam_new < 0))
            #     print(lam_new[np.where(lam_new < 0)])
            #     lam_new = np.abs(lam_new)

            # dlambda1 = 1 * np.median(lam_first) * (dlambda1 - np.median(lam_first)) / (np.max(dlambda1) - np.min(dlambda1) + ep1)

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
                c_arr_fin = np.array(c_arr)
                lam_arr_fin = np.array(lam_arr)
                sol_arr_fin = np.array(sol_arr)
                if k >= maxiter:
                    print("max hit")
                else:
                    pass
                # plt.ioff()
                # plt.show()
                # print(f"Total of {k} iterations")
                return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
            else:
                # ep_min = ep_min / 5
                ep_min = ep_min/eps_cut
                # print("ep_min",ep_min)
                if ep_min <= eps_floor:
                    ep_min = eps_floor
                # if ep_min <= 1e-3:
                    # ep_min = 1e-3
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
    # lam_vec = np.sqrt(lam_vec)
    #Step 2:Run FPA until convergence
    # check = False 
    # best_f_rec1, fin_lam1, c_arr_fin1, lam_arr_fin, sol_arr_fin = fixed_point_algo(gamma_init, lam_vec, eps1, ep_min, eps_cut, eps_floor, check = False)
    # # print("first FPA is done")
    # #Step 3: Calculate new noise level (phi_resid)
    # new_resid = phi_resid(G, best_f_rec1, data_noisy)
    
    # #Step 4: Calculate and update new gamma:
    # zero_vec = np.zeros(len(best_f_rec1))
    # zero_resid = phi_resid(G,zero_vec, data_noisy)

    #If residual is L2:
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25

    # If residual is L1:
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.5
 
    # Step 4: Perform fixed point algo with new gamma value
    # check = True ; if want to print iteration by ieration
    # fin_lam1 = np.sqrt(fin_lam1)
    # lam_vec = np.sqrt(lam_vec)
    best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin  = fixed_point_algo(gamma_init, lam_vec,eps1, ep_min, eps_cut, eps_floor, check = False)
    # print("min", min(fin_lam2))
    # print("max", max(fin_lam2))
    f_rec_final = best_f_rec2
    y_rec_temp = G @ f_rec_final
    y_ratio = np.max(data_noisy)/np.max(y_rec_temp)
    f_rec_final = y_ratio * f_rec_final
    best_f_rec2 = f_rec_final
    # fin_lam2 = np.sqrt(fin_lam2)
    return best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin



### N Parameter Unconstrained Ito problem
def LocReg_Ito_UC_2(data_noisy, G, lam_ini, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback):
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

        # eps = 1e-2
        # eps = 1e-3
        # print(np.diag(lam_vector**2).shape)
        # print(first_deriv(G.shape[1]).shape)

        # A = (G.T @ G + first_deriv(G.shape[1]).T @ first_deriv(G.shape[1]) @ np.diag(lam_vector))
        A = (G.T @ G + np.diag(lam_vector))
        # ep4 = np.ones(G.shape[1]) * eps
        # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vector
        # A = (G.T @ G + np.diag(lam_vector))
        # ep4 = np.ones(G.shape[1]) * eps
        # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vector
        b = (G.T @ data_noisy)
        # sol = nnls(A, b, maxiter=1000)[0]
        sol = np.linalg.solve(A,b)
        # sol = np.linalg.solve(A,b)
        # print("sol",sol)
        # if sol == np.zeros(len(sol)):
        
        # sol = np.linalg.solve(A,b)
        # sol = sol - eps
        # print(np.any(sol < 0))
        # machine_eps = np.finfo(float).eps
        # print(type(sol))
        sol = np.array(sol)
        # sol[sol < 0] = 0
        # sol[sol < 0] = machine_eps
        return sol, A, b

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

        # ep = 1e-2
        # ep = 1e-1
        ep = eps1
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
        
        # first_f_rec, _ , _ = minimize(lam_curr)
        first_f_rec, _ = tikhonov_vec(U, s, V, data_noisy, lam_curr, x_0 = None, nargin = 5)

        #Uncomment the code below to run the Fixed Point Algo (FPA) using while loop until convergence
        while True:
            #Minimization
            try:
                # curr_f_rec = minimize(lam_curr, ep_min, epscond)
                # curr_f_rec, LHS, RHS = minimize(lam_curr)
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
            if feedback == True:
                curr_noise = (G @ curr_f_rec) - data_noisy
                delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                prev = np.linalg.norm(delta_p)
                LHS_temp = LHS.copy()
                while True:
                    curr_f_rec = curr_f_rec - delta_p
                    # curr_f_rec[curr_f_rec < 0] = 0
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
            else:
                pass
            # curr_f_rec[curr_f_rec < 0] = 0

            # if (np.linalg.norm(psi_lam/phi_new)) < 1e-4:
            #     print("condition passed")
            #     print("np.linalg.norm(psi_lam/ phi_new)", np.linalg.norm(psi_lam/ phi_new))
            #     psi_lam = psi_lam + ep_min
            # psi_lam = list(psi_lam)

            #Get new solution with new lambda vector

            # if check == True:
            #     axs[0].plot(T2,g, color = "black", label = "ground truth")
            #     axs[0].plot(T2, curr_f_rec, label = "reconstruction")
            #     axs[1].semilogy(T2, lam_curr, label = "lambdas")
            #     # Redraw the plot
            #     plt.draw()
            #     plt.tight_layout()
            #     plt.pause(0.01)


            #Update lambda: then check
            #New Lambda find the new residual and the new penalty
            phi_new = np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2
            # print("phi_new",phi_new)
            psi_lam = [curr_f_rec[i] for i in range(len(curr_f_rec))]

            #define scaling factor;
            # c = 1/(gamma)
            c = 0.05
            # c = ((gamma**gamma)/((1+gamma)**(1+gamma)))
            # c1 =((gamma**gamma)/((1+gamma)**(1+gamma)))/(gamma + ((gamma**gamma)/((1+gamma)**(1+gamma))))
            # c = 1/(1 + gamma)
            c1 = 1
            # print("gamma", gamma)
            # c = np.std(data_noisy - - np.dot(G,first_f_rec))
            c_arr.append(c)

            psi_lam = np.array(psi_lam)
            # print("psi_lam",np.median(psi_lam))
            
            # ep_min = 1e-10
            # print("np.linalg.norm(psi_lam):", np.linalg.norm(psi_lam))
            #STEP 4
            #redefine new lam
            machine_eps = np.finfo(float).eps
            # plt.plot(lam_curr)
            # plt.show()
            # lam_new = c * (phi_new / (psi_lam + ((lam_curr > 1e2) & (lam_curr < 1e-3)) * ep_2) + (lam_curr > 1e3) * ep_min)
            # lam_new = c * (phi_new / (psi_lam + machine_eps))
            # lam_new = c * (phi_new / (psi_lam + (psi_lam < 0.5 * np.median(curr_f_rec)) * machine_eps))
            # ep_min = 1e-2
            lam_new = c * (phi_new**(1 - exp) / (np.abs(psi_lam) + ep_min))
            # lam_new = c * (phi_new**(1-1/2) / (np.abs(psi_lam) + ep_min))

            # lam_new = []
            # for i in range(len(psi_lam)):
            #     # Exclude the i-th element
            #     psi_lam_excluded = np.delete(psi_lam, i)
            #     lam_curr_excluded = np.delete(lam_curr, i)
            #     # Calculate val
            #     val = c * (phi_new + np.sum(lam_curr_excluded * psi_lam_excluded)) / (np.abs(psi_lam[i]) + ep_min)
            #     # Append to lam_new
            #     lam_new.append(val)
            # lam_new = np.array(lam_new)
            # print("lam_new",lam_new)
            # lam_new = c * lam_new
            # lam_new = savgol_filter(lam_new, window_length=11, polyorder=8)

            # lam_new = np.abs(lam_new)
            # lam_new[lam_new < 0] = 
            # if np.any(lam_new < 0):
            #     print(np.where(lam_new < 0))
            #     print(lam_new[np.where(lam_new < 0)])
            #     lam_new = np.abs(lam_new)

            # dlambda1 = 1 * np.median(lam_first) * (dlambda1 - np.median(lam_first)) / (np.max(dlambda1) - np.min(dlambda1) + ep1)

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
                c_arr_fin = np.array(c_arr)
                lam_arr_fin = np.array(lam_arr)
                sol_arr_fin = np.array(sol_arr)
                if k >= maxiter:
                    print("max hit")
                else:
                    pass
                # plt.ioff()
                # plt.show()
                # print(f"Total of {k} iterations")
                return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
            else:
                # ep_min = ep_min / 5
                ep_min = ep_min/eps_cut
                # print("ep_min",ep_min)
                if ep_min <= eps_floor:
                    ep_min = eps_floor
                # if ep_min <= 1e-3:
                    # ep_min = 1e-3
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
    # lam_vec = np.sqrt(lam_vec)
    #Step 2:Run FPA until convergence
    # check = False 
    # best_f_rec1, fin_lam1, c_arr_fin1, lam_arr_fin, sol_arr_fin = fixed_point_algo(gamma_init, lam_vec, eps1, ep_min, eps_cut, eps_floor, check = False)
    # # print("first FPA is done")
    # #Step 3: Calculate new noise level (phi_resid)
    # new_resid = phi_resid(G, best_f_rec1, data_noisy)
    
    # #Step 4: Calculate and update new gamma:
    # zero_vec = np.zeros(len(best_f_rec1))
    # zero_resid = phi_resid(G,zero_vec, data_noisy)

    #If residual is L2:
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25

    # If residual is L1:
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.5
 
    # Step 4: Perform fixed point algo with new gamma value
    # check = True ; if want to print iteration by ieration
    # fin_lam1 = np.sqrt(fin_lam1)
    # lam_vec = np.sqrt(lam_vec)
    best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin  = fixed_point_algo(gamma_init, lam_vec,eps1, ep_min, eps_cut, eps_floor, check = False)
    # print("min", min(fin_lam2))
    # print("max", max(fin_lam2))
    f_rec_final = best_f_rec2
    y_rec_temp = G @ f_rec_final
    y_ratio = np.max(data_noisy)/np.max(y_rec_temp)
    f_rec_final = y_ratio * f_rec_final
    best_f_rec2 = f_rec_final
    # fin_lam2 = np.sqrt(fin_lam2)
    return best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin


def LocReg_Ito_UC_3(data_noisy, G, lam_ini, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback):
    lam_first = lam_ini * np.ones(G.shape[1])
    U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
    def first_deriv(n):
        D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None)
        L = sparse.identity(n)-D
        Lx = L[0:-1, :]
        Lx = Lx.toarray()
        return Lx
    
    # def minimize(lam_vector):
    #     machine_eps = np.finfo(float).eps

    #     # eps = 1e-2
    #     # eps = 1e-3
    #     # print(np.diag(lam_vector**2).shape)
    #     # print(first_deriv(G.shape[1]).shape)

    #     # A = (G.T @ G + first_deriv(G.shape[1]).T @ first_deriv(G.shape[1]) @ np.diag(lam_vector))
    #     A = (G.T @ G + np.diag(lam_vector))
    #     # ep4 = np.ones(G.shape[1]) * eps
    #     # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vector
    #     # A = (G.T @ G + np.diag(lam_vector))
    #     # ep4 = np.ones(G.shape[1]) * eps
    #     # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vector
    #     b = (G.T @ data_noisy)
    #     # sol = nnls(A, b, maxiter=1000)[0]
    #     sol = np.linalg.solve(A,b)
    #     # sol = np.linalg.solve(A,b)
    #     # print("sol",sol)
    #     # if sol == np.zeros(len(sol)):
        
    #     # sol = np.linalg.solve(A,b)
    #     # sol = sol - eps
    #     # print(np.any(sol < 0))
    #     # machine_eps = np.finfo(float).eps
    #     # print(type(sol))
    #     sol = np.array(sol)
    #     # sol[sol < 0] = 0
    #     # sol[sol < 0] = machine_eps
    #     return sol, A, b

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

        # ep = 1e-2
        # ep = 1e-1
        ep = eps1
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
        
        # first_f_rec, _ , _ = minimize(lam_curr)
        first_f_rec, _ = tikhonov_vec(U, s, V, data_noisy, lam_curr, x_0 = None, nargin = 5)

        #Uncomment the code below to run the Fixed Point Algo (FPA) using while loop until convergence
        while True:
            #Minimization
            try:
                # curr_f_rec = minimize(lam_curr, ep_min, epscond)
                # curr_f_rec, LHS, RHS = minimize(lam_curr)
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
            if feedback == True:
                curr_noise = (G @ curr_f_rec) - data_noisy
                delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                prev = np.linalg.norm(delta_p)
                LHS_temp = LHS.copy()
                while True:
                    curr_f_rec = curr_f_rec - delta_p
                    # curr_f_rec[curr_f_rec < 0] = 0
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
            else:
                pass
            # curr_f_rec[curr_f_rec < 0] = 0

            # if (np.linalg.norm(psi_lam/phi_new)) < 1e-4:
            #     print("condition passed")
            #     print("np.linalg.norm(psi_lam/ phi_new)", np.linalg.norm(psi_lam/ phi_new))
            #     psi_lam = psi_lam + ep_min
            # psi_lam = list(psi_lam)

            #Get new solution with new lambda vector

            # if check == True:
            #     axs[0].plot(T2,g, color = "black", label = "ground truth")
            #     axs[0].plot(T2, curr_f_rec, label = "reconstruction")
            #     axs[1].semilogy(T2, lam_curr, label = "lambdas")
            #     # Redraw the plot
            #     plt.draw()
            #     plt.tight_layout()
            #     plt.pause(0.01)


            #Update lambda: then check
            #New Lambda find the new residual and the new penalty
            phi_new = np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2
            # print("phi_new",phi_new)
            psi_lam = [curr_f_rec[i] for i in range(len(curr_f_rec))]

            #define scaling factor;
            c = 1/(gamma)
            # c = 0.05
            # c = ((gamma**gamma)/((1+gamma)**(1+gamma)))
            # c1 =((gamma**gamma)/((1+gamma)**(1+gamma)))/(gamma + ((gamma**gamma)/((1+gamma)**(1+gamma))))
            # c = 1/(1 + gamma)
            c1 = 1
            # print("gamma", gamma)
            # c = np.std(data_noisy - - np.dot(G,first_f_rec))
            c_arr.append(c)

            psi_lam = np.array(psi_lam)
            # print("psi_lam",np.median(psi_lam))
            
            # ep_min = 1e-10
            # print("np.linalg.norm(psi_lam):", np.linalg.norm(psi_lam))
            #STEP 4
            #redefine new lam
            machine_eps = np.finfo(float).eps
            # plt.plot(lam_curr)
            # plt.show()
            # lam_new = c * (phi_new / (psi_lam + ((lam_curr > 1e2) & (lam_curr < 1e-3)) * ep_2) + (lam_curr > 1e3) * ep_min)
            # lam_new = c * (phi_new / (psi_lam + machine_eps))
            # lam_new = c * (phi_new / (psi_lam + (psi_lam < 0.5 * np.median(curr_f_rec)) * machine_eps))
            # ep_min = 1e-2
            # lam_new = c * (phi_new**(1 - exp) / (np.abs(psi_lam) + ep_min))
            # lam_new = c * (phi_new**(1-1/2) / (np.abs(psi_lam) + ep_min))
            # lam_new = c * (phi_new**(1 - exp) / (np.abs(psi_lam) + ep_min))
            lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))

            # lam_new = []
            # for i in range(len(psi_lam)):
            #     # Exclude the i-th element
            #     psi_lam_excluded = np.delete(psi_lam, i)
            #     lam_curr_excluded = np.delete(lam_curr, i)
            #     # Calculate val
            #     val = c * (phi_new + np.sum(lam_curr_excluded * psi_lam_excluded)) / (np.abs(psi_lam[i]) + ep_min)
            #     # Append to lam_new
            #     lam_new.append(val)
            # lam_new = np.array(lam_new)
            # print("lam_new",lam_new)
            # lam_new = c * lam_new
            # lam_new = savgol_filter(lam_new, window_length=11, polyorder=8)

            # lam_new = np.abs(lam_new)
            # lam_new[lam_new < 0] = 
            # if np.any(lam_new < 0):
            #     print(np.where(lam_new < 0))
            #     print(lam_new[np.where(lam_new < 0)])
            #     lam_new = np.abs(lam_new)

            # dlambda1 = 1 * np.median(lam_first) * (dlambda1 - np.median(lam_first)) / (np.max(dlambda1) - np.min(dlambda1) + ep1)

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
                c_arr_fin = np.array(c_arr)
                lam_arr_fin = np.array(lam_arr)
                sol_arr_fin = np.array(sol_arr)
                if k >= maxiter:
                    print("max hit")
                else:
                    pass
                # plt.ioff()
                # plt.show()
                # print(f"Total of {k} iterations")
                return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
            else:
                # ep_min = ep_min / 5
                ep_min = ep_min/eps_cut
                # print("ep_min",ep_min)
                if ep_min <= eps_floor:
                    ep_min = eps_floor
                # if ep_min <= 1e-3:
                    # ep_min = 1e-3
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
    # lam_vec = np.sqrt(lam_vec)
    #Step 2:Run FPA until convergence
    check = False 
    best_f_rec1, fin_lam1, c_arr_fin1, lam_arr_fin, sol_arr_fin = fixed_point_algo(gamma_init, lam_vec, eps1, ep_min, eps_cut, eps_floor, check = False)
    # print("first FPA is done")
    #Step 3: Calculate new noise level (phi_resid)
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    
    #Step 4: Calculate and update new gamma:
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G,zero_vec, data_noisy)

    # If residual is L2:
    gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25

    # If residual is L1:
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.5
 
    # Step 4: Perform fixed point algo with new gamma value
    # check = True ; if want to print iteration by ieration
    # fin_lam1 = np.sqrt(fin_lam1)
    # lam_vec = np.sqrt(lam_vec)
    # fin_lam1 =np.sqrt(fin_lam1)
    best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin  = fixed_point_algo(gamma_new, fin_lam1,eps1, ep_min, eps_cut, eps_floor, check = False)
    # print("min", min(fin_lam2))
    # print("max", max(fin_lam2))
    f_rec_final = best_f_rec2
    y_rec_temp = G @ f_rec_final
    y_ratio = np.max(data_noisy)/np.max(y_rec_temp)
    f_rec_final = y_ratio * f_rec_final
    best_f_rec2 = f_rec_final
    # fin_lam2 = np.sqrt(fin_lam2)
    return best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin


def LocReg_Ito_UC_4(data_noisy, G, lam_ini, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback):
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

        # eps = 1e-2
        # eps = 1e-3
        # print(np.diag(lam_vector**2).shape)
        # print(first_deriv(G.shape[1]).shape)

        A = (G.T @ G + first_deriv(G.shape[1]).T @ first_deriv(G.shape[1]) @ np.diag(lam_vector))
        # A = (G.T @ G + np.diag(lam_vector))
        # ep4 = np.ones(G.shape[1]) * eps
        # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vector
        # A = (G.T @ G + np.diag(lam_vector))
        # ep4 = np.ones(G.shape[1]) * eps
        # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vector
        b = (G.T @ data_noisy)
        # sol = nnls(A, b, maxiter=1000)[0]
        sol = np.linalg.solve(A,b)
        # sol = np.linalg.solve(A,b)
        # print("sol",sol)
        # if sol == np.zeros(len(sol)):
        
        # sol = np.linalg.solve(A,b)
        # sol = sol - eps
        # print(np.any(sol < 0))
        # machine_eps = np.finfo(float).eps
        # print(type(sol))
        sol = np.array(sol)
        # sol[sol < 0] = 0
        # sol[sol < 0] = machine_eps
        return sol, A, b

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

        # ep = 1e-2
        # ep = 1e-1
        ep = eps1
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
        
        first_f_rec, _ , _ = minimize(lam_curr)
        # first_f_rec, _ = tikhonov_vec(U, s, V, data_noisy, lam_curr, x_0 = None, nargin = 5)

        #Uncomment the code below to run the Fixed Point Algo (FPA) using while loop until convergence
        while True:
            #Minimization
            try:
                # curr_f_rec = minimize(lam_curr, ep_min, epscond)
                curr_f_rec, LHS, RHS = minimize(lam_curr)
                # curr_f_rec, _ = tikhonov_vec(U, s, V, data_noisy, lam_curr, x_0 = None, nargin = 5)
                # LHS = (G.T @ G + first_deriv(G.shape[1]).T @ first_deriv(G.shape[1]) @ np.diag(lam_vector))
                # RHS = G.T @ data_noisy
                if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
                    print(f"curr_f_rec is None after minimization for iteration {k}")
                else:
                    pass
            except Exception as e:
                print("An error occurred during minimization:", e)

            #Feedback
            if feedback == True:
                curr_noise = (G @ curr_f_rec) - data_noisy
                delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                prev = np.linalg.norm(delta_p)
                LHS_temp = LHS.copy()
                while True:
                    curr_f_rec = curr_f_rec - delta_p
                    # curr_f_rec[curr_f_rec < 0] = 0
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
            else:
                pass
            # curr_f_rec[curr_f_rec < 0] = 0

            # if (np.linalg.norm(psi_lam/phi_new)) < 1e-4:
            #     print("condition passed")
            #     print("np.linalg.norm(psi_lam/ phi_new)", np.linalg.norm(psi_lam/ phi_new))
            #     psi_lam = psi_lam + ep_min
            # psi_lam = list(psi_lam)

            #Get new solution with new lambda vector

            # if check == True:
            #     axs[0].plot(T2,g, color = "black", label = "ground truth")
            #     axs[0].plot(T2, curr_f_rec, label = "reconstruction")
            #     axs[1].semilogy(T2, lam_curr, label = "lambdas")
            #     # Redraw the plot
            #     plt.draw()
            #     plt.tight_layout()
            #     plt.pause(0.01)


            #Update lambda: then check
            #New Lambda find the new residual and the new penalty
            phi_new = np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2
            # print("phi_new",phi_new)
            psi_lam = [curr_f_rec[i] for i in range(len(curr_f_rec))]

            #define scaling factor;
            c = 1/(gamma)
            # c = 0.05
            # c = ((gamma**gamma)/((1+gamma)**(1+gamma)))
            # c1 =((gamma**gamma)/((1+gamma)**(1+gamma)))/(gamma + ((gamma**gamma)/((1+gamma)**(1+gamma))))
            # c = 1/(1 + gamma)
            c1 = 1
            # print("gamma", gamma)
            # c = np.std(data_noisy - - np.dot(G,first_f_rec))
            c_arr.append(c)

            psi_lam = np.array(psi_lam)
            # print("psi_lam",np.median(psi_lam))
            
            # ep_min = 1e-10
            # print("np.linalg.norm(psi_lam):", np.linalg.norm(psi_lam))
            #STEP 4
            #redefine new lam
            machine_eps = np.finfo(float).eps
            # plt.plot(lam_curr)
            # plt.show()
            # lam_new = c * (phi_new / (psi_lam + ((lam_curr > 1e2) & (lam_curr < 1e-3)) * ep_2) + (lam_curr > 1e3) * ep_min)
            # lam_new = c * (phi_new / (psi_lam + machine_eps))
            # lam_new = c * (phi_new / (psi_lam + (psi_lam < 0.5 * np.median(curr_f_rec)) * machine_eps))
            # ep_min = 1e-2
            # lam_new = c * (phi_new**(1 - exp) / (np.abs(psi_lam) + ep_min))
            # lam_new = c * (phi_new**(1-1/2) / (np.abs(psi_lam) + ep_min))
            # lam_new = c * (phi_new**(1 - exp) / (np.abs(psi_lam) + ep_min))
            lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))

            # lam_new = []
            # for i in range(len(psi_lam)):
            #     # Exclude the i-th element
            #     psi_lam_excluded = np.delete(psi_lam, i)
            #     lam_curr_excluded = np.delete(lam_curr, i)
            #     # Calculate val
            #     val = c * (phi_new + np.sum(lam_curr_excluded * psi_lam_excluded)) / (np.abs(psi_lam[i]) + ep_min)
            #     # Append to lam_new
            #     lam_new.append(val)
            # lam_new = np.array(lam_new)
            # print("lam_new",lam_new)
            # lam_new = c * lam_new
            # lam_new = savgol_filter(lam_new, window_length=11, polyorder=8)

            # lam_new = np.abs(lam_new)
            # lam_new[lam_new < 0] = 
            # if np.any(lam_new < 0):
            #     print(np.where(lam_new < 0))
            #     print(lam_new[np.where(lam_new < 0)])
            #     lam_new = np.abs(lam_new)

            # dlambda1 = 1 * np.median(lam_first) * (dlambda1 - np.median(lam_first)) / (np.max(dlambda1) - np.min(dlambda1) + ep1)

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
                c_arr_fin = np.array(c_arr)
                lam_arr_fin = np.array(lam_arr)
                sol_arr_fin = np.array(sol_arr)
                if k >= maxiter:
                    print("max hit")
                else:
                    pass
                # plt.ioff()
                # plt.show()
                # print(f"Total of {k} iterations")
                return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
            else:
                # ep_min = ep_min / 5
                ep_min = ep_min/eps_cut
                # print("ep_min",ep_min)
                if ep_min <= eps_floor:
                    ep_min = eps_floor
                # if ep_min <= 1e-3:
                    # ep_min = 1e-3
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
    # lam_vec = np.sqrt(lam_vec)
    #Step 2:Run FPA until convergence
    check = False 
    best_f_rec1, fin_lam1, c_arr_fin1, lam_arr_fin, sol_arr_fin = fixed_point_algo(gamma_init, lam_vec, eps1, ep_min, eps_cut, eps_floor, check = False)
    # print("first FPA is done")
    #Step 3: Calculate new noise level (phi_resid)
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    
    #Step 4: Calculate and update new gamma:
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G,zero_vec, data_noisy)

    # If residual is L2:
    gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25

    # If residual is L1:
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.5
 
    # Step 4: Perform fixed point algo with new gamma value
    # check = True ; if want to print iteration by ieration
    # fin_lam1 = np.sqrt(fin_lam1)
    # lam_vec = np.sqrt(lam_vec)
    # fin_lam1 =np.sqrt(fin_lam1)
    best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin  = fixed_point_algo(gamma_new, fin_lam1,eps1, ep_min, eps_cut, eps_floor, check = False)
    # print("min", min(fin_lam2))
    # print("max", max(fin_lam2))
    f_rec_final = best_f_rec2
    y_rec_temp = G @ f_rec_final
    y_ratio = np.max(data_noisy)/np.max(y_rec_temp)
    f_rec_final = y_ratio * f_rec_final
    best_f_rec2 = f_rec_final
    # fin_lam2 = np.sqrt(fin_lam2)
    return best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin



### N Parameter Ito problem
import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, csc_matrix
from regu.nonnegtik_hnorm import nonnegtik_hnormLR
import numpy as np
import cvxpy as cp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# #working version of mod for publication; don't delete:

# def LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter):
#     def minimize(lam_vec):
#         eps = 1e-2
#         A = G.T @ G + np.diag(lam_vec)        
#         ep4 = np.ones(G.shape[1]) * eps
#         b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
#         # sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
#         y = cp.Variable(G.shape[1])
#         cost = cp.norm(A @ y - b, 'fro')**2
#         constraints = [y >= 0]
#         problem = cp.Problem(cp.Minimize(cost), constraints)
#         problem.solve(solver=cp.MOSEK, mosek_params={
#             'MSK_IPAR_INTPNT_MAX_ITERATIONS ': '100',
#             'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'  # Turn on Mixed Integer Optimization if needed
#               # Turn on detailed logging
#             }, verbose = False)
        
#         #Change tolerance to 10-3; MSK_IPAR_INTPNT_MAX_ITERATIONS increase to 1000; see if total time changes

#         # print("Solver used:", problem.solver_stats.solver_name)
#         # print("Solver version:", problem.solver_stats.solver_version)
#         sol = y.value
#         # if sol is not None:
#         #     sol = np.maximum(sol - eps, 0)
#         sol = np.maximum(sol - eps, 0)
#         return sol, A, b

#     def phi_resid(G, param_vec, data_noisy):
#         return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

#     def fixed_point_algo(gamma, lam_vec, choice_val):
#         lam_curr = lam_vec
#         ep = 1e-2
#         ep_min = 1e-2
#         f_old = np.ones(G.shape[1])
#         first_f_rec, _, _ = minimize(lam_curr)
#         val = 5e-2
#         k = 1
#         while True:
#             try:
#                 curr_f_rec, LHS, _ = minimize(lam_curr)
#                 if curr_f_rec is None:
#                     print(f"curr_f_rec is None after minimization for iteration {k}")
#                     continue
#             except Exception as e:
#                 print("An error occurred during minimization:", e)
#                 continue
#             # print("finished minimization")
#             curr_noise = G @ curr_f_rec - data_noisy
#             LHS_sparse = csr_matrix(LHS)
#             delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#             # delta_p = np.linalg.inv(LHS) @ G.T @ curr_noise
#             # L = np.linalg.cholesky(LHS)
#             # delta_p = scipy.linalg.cho_solve((L,True), G.T @ curr_noise)
#             prev = np.linalg.norm(delta_p)
#             iterationval = 1
#             while iterationval < 20:
#                 curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
#                 curr_noise = G @ curr_f_rec - data_noisy
#                 try:
#                     delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                     # delta_p = np.linalg.inv(LHS) @ G.T @ curr_noise
#                     # delta_p = scipy.linalg.cho_solve((L,True), G.T @ curr_noise)
#                 except RuntimeWarning:
#                     print("Error with delta_p calculation")
#                 val /= 1.03
#                 val = max(val, 8e-3)
#                 if np.abs(np.linalg.norm(delta_p) / prev - 1) < val:
#                     break
#                 prev = np.linalg.norm(delta_p)
#                 # print("prev",  prev)
#                 # iteration += 1
#                 iterationval+=1
            
#             # print("finished feedback")
#             curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
#             phi_new = phi_resid(G, curr_f_rec, data_noisy)
#             psi_lam = np.array(curr_f_rec)
#             c = 1 / gamma
#             lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))

#             if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < ep:
#                 return curr_f_rec, lam_curr, val
#             else:
#                 ep_min = ep_min / 1.2
#                 if ep_min <= 1e-4:
#                     ep_min = 1e-4
#                 lam_curr = lam_new
#                 f_old = curr_f_rec
#                 k += 1
#                 # return curr_f_rec, lam_curr, val

#     lam_vec = lam_ini * np.ones(G.shape[1])
#     choice_val = 9e-3
#     best_f_rec1, fin_lam1, _ = fixed_point_algo(gamma_init, lam_vec, choice_val)
#     # print("finished first step locreg")
#     new_resid = phi_resid(G, best_f_rec1, data_noisy)
#     zero_vec = np.zeros(len(best_f_rec1))
#     zero_resid = phi_resid(G, zero_vec, data_noisy)
#     gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

#     new_choice2 = 5e-3
#     best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2)
#     # print("finished second step locreg")
#     x_normalized = best_f_rec2
#     return x_normalized, fin_lam2,_,_,_

#BEST SO FAR
def LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter):
    def minimize(lam_vec):
            # Fallback to nonnegtik_hnorm
        eps = 1e-2
        A = G.T @ G + np.diag(lam_vec)        
        ep4 = np.ones(G.shape[1]) * eps
        b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
        # sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
        y = cp.Variable(G.shape[1])
        cost = cp.norm(A @ y - b, 'fro')**2
        constraints = [y >= 0]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.MOSEK, mosek_params={
            'MSK_IPAR_INTPNT_MAX_ITERATIONS ': '100',
            'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
            # ,"MSK_DPAR_BASIS_REL_TOL_S": '1e-4'  # Turn on Mixed Integer Optimization if needed
            # Turn on detailed logging
            }, verbose = False)
        
        #Change tolerance to 10-3; MSK_IPAR_INTPNT_MAX_ITERATIONS increase to 1000; see if total time changes

        # print("Solver used:", problem.solver_stats.solver_name)
        # print("Solver version:", problem.solver_stats.solver_version)
        sol = y.value
        # if sol is not None:
        #     sol = np.maximum(sol - eps, 0)
        sol = np.maximum(sol - eps, 0)
        # print("sol", sol)
        if sol is None or np.any([x is None for x in sol]):
            print("Solution contains None values, switching to nonnegtik_hnorm")
            sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)

        return sol, A, b

    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

    def fixed_point_algo(gamma, lam_vec, choice_val):
        lam_curr = lam_vec
        ep = 1e-2
        # ep = 1e-2
        ep_min = 1e-2
        f_old = np.ones(G.shape[1])
        first_f_rec, _, _ = minimize(lam_curr)
        # gammastd = np.std(G @ first_f_rec - data_noisy)
        # val = 5e-2
        val = choice_val
        k = 1
        while True and k <= maxiter:
            try:
                curr_f_rec, LHS, _ = minimize(lam_curr)
                if curr_f_rec is None or np.any([x is None for x in curr_f_rec]):
                    print(f"curr_f_rec returns a None after minimization for iteration {k}")
                    continue
            except Exception as e:
                print("An error occurred during minimization:", e)
                continue
            # print("finished minimization")
            curr_noise = G @ curr_f_rec - data_noisy
            # LHS_sparse = csr_matrix(LHS)
            # delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
            # delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
            L = np.linalg.cholesky(LHS)
            delta_p = scipy.linalg.cho_solve((L,True), G.T @ curr_noise)
            # print("delta_p", delta_p)
            prev = np.linalg.norm(delta_p)
            
            iterationval = 1
            # while iterationval < 300:
            # while iterationval < 550:
            while iterationval < 200:
            # while True:
                curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
                curr_noise = G @ curr_f_rec - data_noisy
                try:
                    # delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
                    # delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                    delta_p = scipy.linalg.cho_solve((L,True), G.T @ curr_noise)
                except RuntimeWarning:
                    print("Error with delta_p calculation")
                # val /= 1.03
                # val = max(val, 8e-3)
                if np.abs(np.linalg.norm(delta_p) / prev - 1) < 1e-5:
                    print("reached tol of 1e-10")
                    break
                elif iterationval == 200:
                    print("max iteration feedback reached")
                else:
                    pass
                prev = np.linalg.norm(delta_p)
                # iteration += 1
                iterationval+=1
                # if iterationval == 100:
                #     print("max feedback iteration reached")
            # print("finished feedback")
            curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
            phi_new = phi_resid(G, curr_f_rec, data_noisy)
            psi_lam = np.array(curr_f_rec)
            # c = 1 / gammastd
            c = 1/gamma
            lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))
            if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < ep or k == maxiter:
                # print("Converged")
                if k == maxiter:
                    print("Maximum Iteration Reached")
                return curr_f_rec, lam_curr, k
            else:
                ep_min = ep_min / 1.2
                if ep_min <= 1e-4:
                    ep_min = 1e-4
                lam_curr = lam_new
                f_old = curr_f_rec
                k += 1
                # return curr_f_rec, lam_curr, val

    #Main Code
    lam_vec = lam_ini * np.ones(G.shape[1])
    # choice_val = 9e-3
    choice_val = 1e-5
    try:
        best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, choice_val)
    except Exception as e:
        print("Error in locreg")
        print("lam_vec", lam_vec)
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G, zero_vec, data_noisy)
    gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

    # new_choice2 = 5e-3
    new_choice2 = 1e-5
    best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2)
    # print("finished second step locreg")
    x_normalized = best_f_rec2
    return x_normalized, fin_lam2, best_f_rec1, fin_lam1, iternum

from scipy.sparse.linalg import splu


def is_symmetric(A):
    return np.allclose(A, A.T)
from scipy.linalg import cholesky, LinAlgError

def is_positive_definite(A):
    try:
        cholesky(A)
        return True
    except LinAlgError:
        return False

def is_sparse(A, threshold=0.1):
    num_nonzero = np.count_nonzero(A)
    total_elements = A.size
    sparsity = num_nonzero / total_elements
    return sparsity < threshold

def condition_number(A):
    return np.linalg.cond(A)

def matrix_rank(A):
    return np.linalg.matrix_rank(A)

def eigenvalues(A):
    return np.linalg.eigvals(A)

from numba import jit
from scipy.linalg import solve, cho_solve

from scipy.sparse.linalg import cg, splu
from Simulations.sparse_solver import solvesparse
from scipy.sparse.linalg import gmres  # GMRES
from concurrent.futures import ProcessPoolExecutor, as_completed
#9/13/24
# At the start of the fixed_point_algo function

def LocReg_Ito_modsparse(data_noisy, G, lam_ini, gamma_init, maxiter):
    def minimize(lam_vec):
            # Fallback to nonnegtik_hnorm
        eps = 1e-2
        A = G.T @ G + np.diag(lam_vec)        
        ep4 = np.ones(G.shape[1]) * eps
        b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
        # sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
        y = cp.Variable(G.shape[1])
        cost = cp.norm(A @ y - b, 'fro')**2
        constraints = [y >= 0]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.MOSEK, mosek_params={
            'MSK_IPAR_INTPNT_MAX_ITERATIONS ': '100',
            'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
            # ,"MSK_DPAR_BASIS_REL_TOL_S": '1e-4'  # Turn on Mixed Integer Optimization if needed
            # Turn on detailed logging
            }, verbose = False)
        
        #Change tolerance to 10-3; MSK_IPAR_INTPNT_MAX_ITERATIONS increase to 1000; see if total time changes

        # print("Solver used:", problem.solver_stats.solver_name)
        # print("Solver version:", problem.solver_stats.solver_version)
        sol = y.value
        # if sol is not None:
        #     sol = np.maximum(sol - eps, 0)
        sol = np.maximum(sol - eps, 0)
        # print("sol", sol)
        if sol is None or np.any([x is None for x in sol]):
            print("Solution contains None values, switching to nonnegtik_hnorm")
            sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)

        return sol, A, b

    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

    def fixed_point_algo(gamma, lam_vec, choice_val):
        lam_curr = lam_vec
        # ep = 1e-4
        ep = 1e-2
        ep_min = 1e-2
        f_old = np.ones(G.shape[1])
        first_f_rec, _, _ = minimize(lam_curr)
        # gammastd = np.std(G @ first_f_rec - data_noisy)
        # val = 5e-2
        val = choice_val
        k = 1
        while True and k <= maxiter:
            try:
                curr_f_rec, LHS, _ = minimize(lam_curr)
                if curr_f_rec is None or np.any([x is None for x in curr_f_rec]):
                    print(f"curr_f_rec returns a None after minimization for iteration {k}")
                    continue
            except Exception as e:
                print("An error occurred during minimization:", e)
                continue
            # print("finished minimization")

            # print("Is symmetric:", is_symmetric(LHS))
            # print("Is positive definite:", is_positive_definite(LHS))
            # print("Is sparse:", is_sparse(LHS))
            # print("Condition number:", condition_number(LHS))
            # print("Rank of the matrix:", matrix_rank(LHS))
            # print("Eigenvalues:", eigenvalues(LHS))

            #This gives me the current residual first residential
            curr_noise = G @ curr_f_rec - data_noisy
            LHS_sparse = csr_matrix(LHS)
            # L = np.linalg.cholesky(LHS)
            # delta_p = scipy.linalg.cho_solve((L,True), G.T @ curr_noise)
            # delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
            delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
            # delta_p = solvesparse(LHS, G.T @ curr_noise)
            # lu = splu(LHS_sparse)
            # delta_p = lu.solve(G.T @ curr_noise)


            # delta_p = numbasolve(LHS, G.T @ curr_noise)

            # delta_p = scipy.linalg.solve(LHS, G.T @ curr_noise)
            # delta_p, exit_code = gmres(LHS, G.T @ curr_noise)

            # print("delta_p",delta_p)
            # delta_p, exit_code = cg(LHS_sparse, G.T @ curr_noise)
            # delta_p = np.linalg.solve(LHS, G.T @ curr_noise)

            #delta p is the estimate for that gradient/maximum change
            prev = np.linalg.norm(delta_p)
            
            iterationval = 1
            while iterationval < 300:
            # while iterationval < 1000:
            # while True:
                #retrieve the ideal f_rec by subtracting the delta_p estimate
                curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

                #Then get the current_noise
                curr_noise = G @ curr_f_rec - data_noisy
                
                try:
                    # delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
                    # delta_p = solvesparse(LHS, G.T @ curr_noise)
                    # LHS_sparse = csc_matrix(LHS)
                    delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
                    # lu = splu(LHS_sparse)
                    # delta_p = lu.solve(G.T @ curr_noise)
                    # delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                    # delta_p = scipy.linalg.cho_solve((L,True), G.T @ curr_noise)
                    # delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
                    # delta_p = solvesparse(LHS, G.T @ curr_noise)
                    # delta_p = numbasolve(LHS, G.T @ curr_noise)
                    # delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                    # delta_p = scipy.linalg.solve(LHS, G.T @ curr_noise)
                    # delta_p, exit_code = gmres(LHS, G.T @ curr_noise)
                    # delta_p, exit_code = cg(LHS_sparse, G.T @ curr_noise)
                    # delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
                    # x, exit_code = cg(A_sparse, b)

                except RuntimeWarning:
                    print("Error with delta_p calculation")
                # val /= 1.03
                # val = max(val, 8e-3)
                if np.abs(np.linalg.norm(delta_p) / prev ) < val:
                    break
                prev = np.linalg.norm(delta_p)
                # iteration += 1
                iterationval+=1
                # if iterationval == 100:
                #     print("max feedback iteration reached")
            # print("finished feedback")
            curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

            phi_new = phi_resid(G, curr_f_rec, data_noisy)
            psi_lam = np.array(curr_f_rec)
            # c = 1 / gammastd
            c = 1/gamma
            lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))

            if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < ep or k == maxiter:
                # print("Converged")
                if k == maxiter:
                    print("Maximum Iteration Reached")
                return curr_f_rec, lam_curr, k
            else:
                ep_min = ep_min / 1.2
                if ep_min <= 1e-4:
                    ep_min = 1e-4
                lam_curr = lam_new
                f_old = curr_f_rec
                k += 1
                # return curr_f_rec, lam_curr, val

    #Main Code
    lam_vec = lam_ini * np.ones(G.shape[1])
    # choice_val = 9e-3
    # choice_val = 1e-5
    choice_val = 1e-5
    try:
        best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, choice_val)
    except Exception as e:
        print("Error in locreg")
        print("lam_vec", lam_vec)
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G, zero_vec, data_noisy)
    gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

    new_choice2 = 1e-5
    # new_choice2 = 1e-5
    # choice_val = 1e-6
    best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2)
    # print("finished second step locreg")
    x_normalized = best_f_rec2
    return x_normalized, fin_lam2, best_f_rec1, fin_lam1, iternum


# def LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter):
#     def minimize(lam_vec):
#             # Fallback to nonnegtik_hnorm
#         eps = 1e-2
#         A = G.T @ G + np.diag(lam_vec)        
#         ep4 = np.ones(G.shape[1]) * eps
#         b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
#         # sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
#         y = cp.Variable(G.shape[1])
#         cost = cp.norm(A @ y - b, 'fro')**2
#         constraints = [y >= 0]
#         problem = cp.Problem(cp.Minimize(cost), constraints)
#         problem.solve(solver=cp.MOSEK, mosek_params={
#             'MSK_IPAR_INTPNT_MAX_ITERATIONS ': '100',
#             'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
#             # ,"MSK_DPAR_BASIS_REL_TOL_S": '1e-4'  # Turn on Mixed Integer Optimization if needed
#             # Turn on detailed logging
#             }, verbose = False)
        
#         #Change tolerance to 10-3; MSK_IPAR_INTPNT_MAX_ITERATIONS increase to 1000; see if total time changes

#         # print("Solver used:", problem.solver_stats.solver_name)
#         # print("Solver version:", problem.solver_stats.solver_version)
#         sol = y.value
#         # if sol is not None:
#         #     sol = np.maximum(sol - eps, 0)
#         sol = np.maximum(sol - eps, 0)
#         # print("sol", sol)
#         if sol is None or np.any([x is None for x in sol]):
#             print("Solution contains None values, switching to nonnegtik_hnorm")
#             sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)

#         return sol, A, b

#     def phi_resid(G, param_vec, data_noisy):
#         return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

#     def fixed_point_algo(gamma, lam_vec, choice_val):
#         lam_curr = lam_vec
#         # ep = 1e-4
#         ep = 1e-2
#         ep_min = 1e-2
#         f_old = np.ones(G.shape[1])
#         first_f_rec, _, _ = minimize(lam_curr)
#         # gammastd = np.std(G @ first_f_rec - data_noisy)
#         # val = 5e-2
#         val = choice_val
#         k = 1
#         while True and k <= maxiter:
#             try:
#                 curr_f_rec, LHS, _ = minimize(lam_curr)
#                 if curr_f_rec is None or np.any([x is None for x in curr_f_rec]):
#                     print(f"curr_f_rec returns a None after minimization for iteration {k}")
#                     continue
#             except Exception as e:
#                 print("An error occurred during minimization:", e)
#                 continue
#             # print("finished minimization")

#             # print("Is symmetric:", is_symmetric(LHS))
#             # print("Is positive definite:", is_positive_definite(LHS))
#             # print("Is sparse:", is_sparse(LHS))
#             # print("Condition number:", condition_number(LHS))
#             # print("Rank of the matrix:", matrix_rank(LHS))
#             # print("Eigenvalues:", eigenvalues(LHS))

#             if k % 1 == 0:
#                 #This gives me the current residual first residential
#                 curr_noise = G @ curr_f_rec - data_noisy
#                 LHS_sparse = csc_matrix(LHS)
#                 # delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                 lu = splu(LHS_sparse)
#                 delta_p = lu.solve(G.T @ curr_noise)
#                 # delta_p = solvesparse(LHS, G.T @ curr_noise)
#                 # delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                 # delta_p = solvesparse(LHS, G.T @ curr_noise)  
# #
#                 #delta p is the estimate for that gradient/maximum change
#                 prev = np.linalg.norm(delta_p)
                
#                 iterationval = 1
#                 while iterationval < 300:
#                 # while iterationval < 1000:
#                 # while True:
#                     #retrieve the ideal f_rec by subtracting the delta_p estimate
#                     curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

#                     #Then get the current_noise
#                     curr_noise = G @ curr_f_rec - data_noisy
                    
#                     try:
#                         # LHS_sparse = csr_matrix(LHS)
#                         delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                         # lu = splu(LHS_sparse)
#                         # delta_p = lu.solve(G.T @ curr_noise)
#                         # delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                         # delta_p = solvesparse(LHS, G.T @ curr_noise)
                        
#                     except RuntimeWarning:
#                         print("Error with delta_p calculation")
#                     # val /= 1.03
#                     # val = max(val, 8e-3)
#                     if np.abs(np.linalg.norm(delta_p) / prev - 1) < val:
#                         break
#                     prev = np.linalg.norm(delta_p)
#                     # iteration += 1
#                     iterationval+=1
#                     # if iterationval == 100:
#                     #     print("max feedback iteration reached")
#                 # print("finished feedback")
#                 curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

#             phi_new = phi_resid(G, curr_f_rec, data_noisy)
#             psi_lam = np.array(curr_f_rec)
#             # c = 1 / gammastd
#             c = 1/gamma
#             lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))

#             if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < ep or k == maxiter:
#                 # print("Converged")
#                 if k == maxiter:
#                     print("Maximum Iteration Reached")
#                 return curr_f_rec, lam_curr, k
#             else:
#                 ep_min = ep_min / 1.2
#                 if ep_min <= 1e-4:
#                     ep_min = 1e-4
#                 lam_curr = lam_new
#                 f_old = curr_f_rec
#                 k += 1
#                 # return curr_f_rec, lam_curr, val

#     #Main Code
#     lam_vec = lam_ini * np.ones(G.shape[1])
#     # choice_val = 9e-3
#     # choice_val = 1e-5
#     choice_val = 1e-5
#     try:
#         best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, choice_val)
#     except Exception as e:
#         print("Error in locreg")
#         print("lam_vec", lam_vec)
#     new_resid = phi_resid(G, best_f_rec1, data_noisy)
#     zero_vec = np.zeros(len(best_f_rec1))
#     zero_resid = phi_resid(G, zero_vec, data_noisy)
#     gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

#     new_choice2 = 1e-5
#     # new_choice2 = 1e-5
#     # choice_val = 1e-6
#     best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2)
#     # print("finished second step locreg")
#     x_normalized = best_f_rec2
#     return x_normalized, fin_lam2, best_f_rec1, fin_lam1, iternum


# def LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter):
#     def minimize(lam_vec):
#             # Fallback to nonnegtik_hnorm
#         eps = 1e-2
#         A = G.T @ G + np.diag(lam_vec)        
#         ep4 = np.ones(G.shape[1]) * eps
#         b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
#         # sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
#         y = cp.Variable(G.shape[1])
#         cost = cp.norm(A @ y - b, 'fro')**2
#         constraints = [y >= 0]
#         problem = cp.Problem(cp.Minimize(cost), constraints)
#         problem.solve(solver=cp.MOSEK, mosek_params={
#             'MSK_IPAR_INTPNT_MAX_ITERATIONS ': '100',
#             'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
#             # ,"MSK_DPAR_BASIS_REL_TOL_S": '1e-4'  # Turn on Mixed Integer Optimization if needed
#             # Turn on detailed logging
#             }, verbose = False)
        
#         #Change tolerance to 10-3; MSK_IPAR_INTPNT_MAX_ITERATIONS increase to 1000; see if total time changes

#         # print("Solver used:", problem.solver_stats.solver_name)
#         # print("Solver version:", problem.solver_stats.solver_version)
#         sol = y.value
#         # if sol is not None:
#         #     sol = np.maximum(sol - eps, 0)
#         sol = np.maximum(sol - eps, 0)
#         # print("sol", sol)
#         if sol is None or np.any([x is None for x in sol]):
#             print("Solution contains None values, switching to nonnegtik_hnorm")
#             sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)

#         return sol, A, b

#     def phi_resid(G, param_vec, data_noisy):
#         return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

#     def fixed_point_algo(gamma, lam_vec, choice_val):
#         lam_curr = lam_vec
#         # ep = 1e-4
#         ep = 1e-2
#         ep_min = 1e-2
#         f_old = np.ones(G.shape[1])
#         first_f_rec, _, _ = minimize(lam_curr)
#         # gammastd = np.std(G @ first_f_rec - data_noisy)
#         # val = 5e-2
#         val = choice_val
#         k = 1
#         while True and k <= maxiter:
#             try:
#                 curr_f_rec, LHS, _ = minimize(lam_curr)
#                 if curr_f_rec is None or np.any([x is None for x in curr_f_rec]):
#                     print(f"curr_f_rec returns a None after minimization for iteration {k}")
#                     continue
#             except Exception as e:
#                 print("An error occurred during minimization:", e)
#                 continue
#             # print("finished minimization")

#             # print("Is symmetric:", is_symmetric(LHS))
#             # print("Is positive definite:", is_positive_definite(LHS))
#             # print("Is sparse:", is_sparse(LHS))
#             # print("Condition number:", condition_number(LHS))
#             # print("Rank of the matrix:", matrix_rank(LHS))
#             # print("Eigenvalues:", eigenvalues(LHS))

#             if k % 1 == 0:
#                 #This gives me the current residual first residential
#                 curr_noise = G @ curr_f_rec - data_noisy
#                 LHS_sparse = csr_matrix(LHS)
#                 delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                 # delta_p = solvesparse(LHS, G.T @ curr_noise)

                
#                 #delta p is the estimate for that gradient/maximum change
#                 prev = np.linalg.norm(delta_p)
                
#                 iterationval = 1
#                 while iterationval < 300:
#                 # while iterationval < 1000:
#                 # while True:
#                     #retrieve the ideal f_rec by subtracting the delta_p estimate
#                     curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

#                     #Then get the current_noise
#                     curr_noise = G @ curr_f_rec - data_noisy
                    
#                     try:
#                         delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                         # delta_p = solvesparse(LHS, G.T @ curr_noise)
                        
#                     except RuntimeWarning:
#                         print("Error with delta_p calculation")
#                     # val /= 1.03
#                     # val = max(val, 8e-3)
#                     if np.abs(np.linalg.norm(delta_p) / prev - 1) < val:
#                         break
#                     prev = np.linalg.norm(delta_p)
#                     # iteration += 1
#                     iterationval+=1
#                     # if iterationval == 100:
#                     #     print("max feedback iteration reached")
#                 # print("finished feedback")
#                 curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

#             phi_new = phi_resid(G, curr_f_rec, data_noisy)
#             psi_lam = np.array(curr_f_rec)
#             # c = 1 / gammastd
#             c = 1/gamma
#             lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))

#             if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < ep or k == maxiter:
#                 # print("Converged")
#                 if k == maxiter:
#                     print("Maximum Iteration Reached")
#                 return curr_f_rec, lam_curr, k
#             else:
#                 ep_min = ep_min / 1.2
#                 if ep_min <= 1e-4:
#                     ep_min = 1e-4
#                 lam_curr = lam_new
#                 f_old = curr_f_rec
#                 k += 1
#                 # return curr_f_rec, lam_curr, val

#     #Main Code
#     lam_vec = lam_ini * np.ones(G.shape[1])
#     # choice_val = 9e-3
#     # choice_val = 1e-5
#     choice_val = 1e-5
#     try:
#         best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, choice_val)
#     except Exception as e:
#         print("Error in locreg")
#         print("lam_vec", lam_vec)
#     new_resid = phi_resid(G, best_f_rec1, data_noisy)
#     zero_vec = np.zeros(len(best_f_rec1))
#     zero_resid = phi_resid(G, zero_vec, data_noisy)
#     gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

#     new_choice2 = 1e-5
#     # new_choice2 = 1e-5
#     # choice_val = 1e-6
#     best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2)
#     # print("finished second step locreg")
#     x_normalized = best_f_rec2
#     return x_normalized, fin_lam2, best_f_rec1, fin_lam1, iternum



# def LocReg_Ito_mod(data_noisy, G, lam_ini, f_rec_ini, gamma_init, maxiter):
#     def minimize(lam_vec):
#             # Fallback to nonnegtik_hnorm
#         eps = 1e-2
#         A = G.T @ G + np.diag(lam_vec)        
#         ep4 = np.ones(G.shape[1]) * eps
#         b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
#         # sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
#         y = cp.Variable(G.shape[1])
#         cost = cp.norm(A @ y - b, 'fro')**2
#         constraints = [y >= 0]
#         problem = cp.Problem(cp.Minimize(cost), constraints)
#         problem.solve(solver=cp.MOSEK, mosek_params={
#             'MSK_IPAR_INTPNT_MAX_ITERATIONS ': '100',
#             'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
#             # ,"MSK_DPAR_BASIS_REL_TOL_S": '1e-4'  # Turn on Mixed Integer Optimization if needed
#             # Turn on detailed logging
#             }, verbose = False)
        
#         #Change tolerance to 10-3; MSK_IPAR_INTPNT_MAX_ITERATIONS increase to 1000; see if total time changes

#         # print("Solver used:", problem.solver_stats.solver_name)
#         # print("Solver version:", problem.solver_stats.solver_version)
#         sol = y.value
#         # if sol is not None:
#         #     sol = np.maximum(sol - eps, 0)
#         sol = np.maximum(sol - eps, 0)
#         # print("sol", sol)
#         if sol is None or np.any([x is None for x in sol]):
#             print("Solution contains None values, switching to nonnegtik_hnorm")
#             sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)

#         return sol, A, b

#     def phi_resid(G, param_vec, data_noisy):
#         return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

#     def fixed_point_algo(gamma, lam_vec, choice_val):
#         lam_curr = lam_vec
#         # ep = 1e-4
#         ep = 1e-2
#         ep_min = 1e-2
#         f_old = np.ones(G.shape[1])
#         first_f_rec, _, _ = minimize(lam_curr)
#         # gammastd = np.std(G @ first_f_rec - data_noisy)
#         # val = 5e-2
#         val = choice_val
#         k = 1
#         while True and k <= maxiter:
#             try:
#                 curr_f_rec, LHS, _ = minimize(lam_curr)
#                 if curr_f_rec is None or np.any([x is None for x in curr_f_rec]):
#                     print(f"curr_f_rec returns a None after minimization for iteration {k}")
#                     continue
#             except Exception as e:
#                 print("An error occurred during minimization:", e)
#                 continue
#             # print("finished minimization")

#             if k % 1 == 0:
#                 #This gives me the current residual first residential
#                 curr_noise = G @ curr_f_rec - data_noisy
#                 LHS_sparse = csr_matrix(LHS)

#                 #G.T @ residual = gradient
#                 delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                 #delta p is the estimate for that gradient/maximum change
#                 prev = np.linalg.norm(delta_p)
                
#                 iterationval = 1
#                 while iterationval < 300:
#                 # while iterationval < 1000:
#                 # while True:
#                     #retrieve the ideal f_rec by subtracting the delta_p estimate
#                     curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

#                     #Then get the current_noise
#                     curr_noise = G @ curr_f_rec - data_noisy
#                     try:
#                         delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                     except RuntimeWarning:
#                         print("Error with delta_p calculation")
#                     # val /= 1.03
#                     # val = max(val, 8e-3)
#                     if np.abs(np.linalg.norm(delta_p) / prev - 1) < val:
#                         break
#                     prev = np.linalg.norm(delta_p)
#                     # iteration += 1
#                     iterationval+=1
#                     # if iterationval == 100:
#                     #     print("max feedback iteration reached")
#                 # print("finished feedback")
#                 curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

#             phi_new = phi_resid(G, curr_f_rec, data_noisy)
#             psi_lam = np.array(curr_f_rec)
#             # c = 1 / gammastd
#             c = 1/gamma
#             lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))

#             if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < ep or k == maxiter:
#                 # print("Converged")
#                 if k == maxiter:
#                     print("Maximum Iteration Reached")
#                 return curr_f_rec, lam_curr, k
#             else:
#                 ep_min = ep_min / 1.2
#                 if ep_min <= 1e-4:
#                     ep_min = 1e-4
#                 lam_curr = lam_new
#                 f_old = curr_f_rec
#                 k += 1
#                 # return curr_f_rec, lam_curr, val

#     #Main Code
#     lam_vec = lam_ini * np.ones(G.shape[1])
#     # choice_val = 9e-3
#     # choice_val = 1e-5
#     choice_val = 1e-5
#     # try:
#     #     best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, choice_val)
#     # except Exception as e:
#     #     print("Error in locreg")
#     #     print("lam_vec", lam_vec)
#     best_f_rec1 = f_rec_ini
#     fin_lam1 = lam_vec
#     new_resid = phi_resid(G, best_f_rec1, data_noisy)
#     zero_vec = np.zeros(len(best_f_rec1))
#     zero_resid = phi_resid(G, zero_vec, data_noisy)
#     gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

#     new_choice2 = 1e-5
#     # new_choice2 = 1e-5
#     # choice_val = 1e-6
#     best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2)
#     # print("finished second step locreg")
#     x_normalized = best_f_rec2
#     return x_normalized, fin_lam2, best_f_rec1, fin_lam1



# def LocReg_Ito_mod(data_noisy, G, lam_ini, f_rec_ini, gamma_init, maxiter):
#     def minimize(lam_vec):
#             # Fallback to nonnegtik_hnorm
#         eps = 1e-2
#         A = G.T @ G + np.diag(lam_vec)        
#         ep4 = np.ones(G.shape[1]) * eps
#         b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
#         # sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
#         y = cp.Variable(G.shape[1])
#         cost = cp.norm(A @ y - b, 'fro')**2
#         constraints = [y >= 0]
#         problem = cp.Problem(cp.Minimize(cost), constraints)
#         problem.solve(solver=cp.MOSEK, mosek_params={
#             'MSK_IPAR_INTPNT_MAX_ITERATIONS ': '100',
#             'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
#             # ,"MSK_DPAR_BASIS_REL_TOL_S": '1e-4'  # Turn on Mixed Integer Optimization if needed
#             # Turn on detailed logging
#             }, verbose = False)
        
#         #Change tolerance to 10-3; MSK_IPAR_INTPNT_MAX_ITERATIONS increase to 1000; see if total time changes

#         # print("Solver used:", problem.solver_stats.solver_name)
#         # print("Solver version:", problem.solver_stats.solver_version)
#         sol = y.value
#         # if sol is not None:
#         #     sol = np.maximum(sol - eps, 0)
#         sol = np.maximum(sol - eps, 0)
#         # print("sol", sol)
#         if sol is None or np.any([x is None for x in sol]):
#             print("Solution contains None values, switching to nonnegtik_hnorm")
#             sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)

#         return sol, A, b

#     def phi_resid(G, param_vec, data_noisy):
#         return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

#     def fixed_point_algo(gamma, lam_vec, choice_val):
#         lam_curr = lam_vec
#         # ep = 1e-4
#         ep = 1e-2
#         ep_min = 1e-2
#         f_old = np.ones(G.shape[1])
#         first_f_rec, _, _ = minimize(lam_curr)
#         # gammastd = np.std(G @ first_f_rec - data_noisy)
#         # val = 5e-2
#         val = choice_val
#         k = 1
#         while True and k <= maxiter:
#             try:
#                 curr_f_rec, LHS, _ = minimize(lam_curr)
#                 if curr_f_rec is None or np.any([x is None for x in curr_f_rec]):
#                     print(f"curr_f_rec returns a None after minimization for iteration {k}")
#                     continue
#             except Exception as e:
#                 print("An error occurred during minimization:", e)
#                 continue
#             # print("finished minimization")

#             if k % 1 == 0:
#                 #This gives me the current residual first residential
#                 curr_noise = G @ curr_f_rec - data_noisy
#                 LHS_sparse = csr_matrix(LHS)

#                 #G.T @ residual = gradient
#                 delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                 #delta p is the estimate for that gradient/maximum change
#                 prev = np.linalg.norm(delta_p)
                
#                 iterationval = 1
#                 while iterationval < 300:
#                 # while iterationval < 1000:
#                 # while True:
#                     #retrieve the ideal f_rec by subtracting the delta_p estimate
#                     curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

#                     #Then get the current_noise
#                     curr_noise = G @ curr_f_rec - data_noisy
#                     try:
#                         delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                     except RuntimeWarning:
#                         print("Error with delta_p calculation")
#                     # val /= 1.03
#                     # val = max(val, 8e-3)
#                     if np.abs(np.linalg.norm(delta_p) / prev - 1) < val:
#                         break
#                     prev = np.linalg.norm(delta_p)
#                     # iteration += 1
#                     iterationval+=1
#                     # if iterationval == 100:
#                     #     print("max feedback iteration reached")
#                 # print("finished feedback")
#                 curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

#             phi_new = phi_resid(G, curr_f_rec, data_noisy)
#             psi_lam = np.array(curr_f_rec)
#             # c = 1 / gammastd
#             c = 1/gamma
#             lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))


#             if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < ep or k == maxiter:
#                 # print("Converged")
#                 if k == maxiter:
#                     print("Maximum Iteration Reached")
#                 return curr_f_rec, lam_curr, k
#             else:
#                 ep_min = ep_min / 1.2
#                 if ep_min <= 1e-4:
#                     ep_min = 1e-4
#                 lam_curr = lam_new
#                 f_old = curr_f_rec
#                 k += 1
#                 # return curr_f_rec, lam_curr, val

#     #Main Code
#     lam_vec = lam_ini * np.ones(G.shape[1])
#     # # choice_val = 9e-3
#     # choice_val = 1e-5
#     # try:
#     #     best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, choice_val)
#     # except Exception as e:
#     #     print("Error in locreg")
#     #     print("lam_vec", lam_vec)
#     fin_lam1 = lam_vec
#     best_f_rec1 = f_rec_ini
#     new_resid = phi_resid(G, best_f_rec1, data_noisy)
#     zero_vec = np.zeros(len(best_f_rec1))
#     zero_resid = phi_resid(G, zero_vec, data_noisy)
#     gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

#     # new_choice2 = 5e-3
#     new_choice2 = 1e-5
#     best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2)
#     # print("finished second step locreg")
#     x_normalized = best_f_rec2
#     return x_normalized, fin_lam2, best_f_rec1, fin_lam1


def LocReg_Ito_classical(data_noisy, G, lam_ini, gamma_init, maxiter):
    def minimize(lam_vec):
            # Fallback to nonnegtik_hnorm
        eps = 1e-2
        A = G.T @ G + np.diag(lam_vec)        
        ep4 = np.ones(G.shape[1]) * eps
        b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
        # sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
        y = cp.Variable(G.shape[1])
        cost = cp.norm(A @ y - b, 'fro')**2
        # constraints = [y >= 0]
        # problem = cp.Problem(cp.Minimize(cost), constraints)
        problem = cp.Problem(cp.Minimize(cost))
        problem.solve(solver=cp.MOSEK, mosek_params={
            'MSK_IPAR_INTPNT_MAX_ITERATIONS ': '100',
            'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
            # ,"MSK_DPAR_BASIS_REL_TOL_S": '1e-4'  # Turn on Mixed Integer Optimization if needed
            # Turn on detailed logging
            }, verbose = False)
        
        #Change tolerance to 10-3; MSK_IPAR_INTPNT_MAX_ITERATIONS increase to 1000; see if total time changes

        # print("Solver used:", problem.solver_stats.solver_name)
        # print("Solver version:", problem.solver_stats.solver_version)
        sol = y.value
        # if sol is not None:
        #     sol = np.maximum(sol - eps, 0)
        sol = sol-eps
        # sol = np.maximum(sol - eps, 0)
        # print("sol", sol)
        if sol is None or np.any([x is None for x in sol]):
            print("Solution contains None values,break the program")
        #     # sol, rho, trash = tikhonov_vec(G, data_noisy, lam_vec, '0', nargin=4)
        #     exp, _ = tikhonov_vec(U, s, V, data_noisy, Alpha_vec[j], x_0 = None, nargin = 5)
        return sol, A, b

    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

    def fixed_point_algo(gamma, lam_vec, choice_val):
        lam_curr = lam_vec
        # ep = 1e-4
        ep = 1e-2
        ep_min = 1e-2
        f_old = np.ones(G.shape[1])
        first_f_rec, _, _ = minimize(lam_curr)
        # gammastd = np.std(G @ first_f_rec - data_noisy)
        # val = 5e-2
        val = choice_val
        k = 1
        while True and k <= maxiter:
            try:
                curr_f_rec, LHS, _ = minimize(lam_curr)
                if curr_f_rec is None or np.any([x is None for x in curr_f_rec]):
                    print(f"curr_f_rec returns a None after minimization for iteration {k}")
                    continue
            except Exception as e:
                print("An error occurred during minimization:", e)
                continue
            # print("finished minimization")

            if k % 1 == 0:
                curr_noise = G @ curr_f_rec - data_noisy
                LHS_sparse = csr_matrix(LHS)
                delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
                prev = np.linalg.norm(delta_p)
                
                iterationval = 1
                while iterationval < 300:
                # while iterationval < 1000:
                # while True:
                    curr_f_rec = curr_f_rec - delta_p
                    curr_noise = G @ curr_f_rec - data_noisy
                    try:
                        delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
                    except RuntimeWarning:
                        print("Error with delta_p calculation")
                    # val /= 1.03
                    # val = max(val, 8e-3)
                    if np.abs(np.linalg.norm(delta_p) / prev - 1) < val:
                        break
                    prev = np.linalg.norm(delta_p)
                    # iteration += 1
                    iterationval+=1
                    # if iterationval == 100:
                    #     print("max feedback iteration reached")
                # print("finished feedback")
                curr_f_rec = curr_f_rec - delta_p

            phi_new = phi_resid(G, curr_f_rec, data_noisy)
            psi_lam = np.array(curr_f_rec)
            # c = 1 / gammastd
            c = 1/gamma
            lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))

            if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < ep or k == maxiter:
                # print("Converged")
                if k == maxiter:
                    print("Maximum Iteration Reached")
                return curr_f_rec, lam_curr, k
            else:
                ep_min = ep_min / 1.2
                if ep_min <= 1e-4:
                    ep_min = 1e-4
                lam_curr = lam_new
                f_old = curr_f_rec
                k += 1
                # return curr_f_rec, lam_curr, val

    lam_vec = lam_ini * np.ones(G.shape[1])
    # choice_val = 9e-3
    choice_val = 1e-5
    try:
        best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, choice_val)
    except Exception as e:
        print("Error in locreg")
        print("lam_vec", lam_vec)
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G, zero_vec, data_noisy)
    gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

    # new_choice2 = 5e-3
    new_choice2 = 1e-5
    best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2)
    # print("finished second step locreg")
    x_normalized = best_f_rec2
    return x_normalized, fin_lam2, best_f_rec1, fin_lam1, iternum


# def LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter):
#     # def minimize(lam_vector):
#     #     eps = 1e-2
#     #     A = (G.T @ G + np.diag((lam_vector)))
#     #     b = (G.T @ data_noisy)
#     #     # ep4 = np.ones(G.shape[1]) * eps
#     #     # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * (lam_vector)
#     #     # sol, rho, trash = nonnegtik_hnorm(G, data_noisy, (lam_vector), '0', nargin=4)
#     #     # sol = sol - eps
#     #     # sol = np.max(sol,0)
#     #     return sol, A, b
#     def minimize(lam_vec):
#         A = (G.T @ G + np.diag(lam_vec))
#         # b = G.T @ data_noisy
#         eps = 1e-2
#         ep4 = np.ones(A.shape[1]) * eps
#         b = (G.T @ data_noisy) + ((G.T @ G) @ ep4) + ep4 * (lam_vec)
#         y = cp.Variable(G.shape[1])
#         cost = cp.norm(A @ y - b, 2)**2
#         constraints = [y >= 0]
#         problem = cp.Problem(cp.Minimize(cost), constraints)
#         problem.solve(solver=cp.MOSEK, verbose=False)
#         sol = y.value
#         sol = sol - eps
#         sol = np.maximum(sol,0)
#         return sol, A, b


#     def phi_resid(G, param_vec, data_noisy):
#         return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

#     def fixed_point_algo(gamma, lam_vec, choice_val):
#         lam_curr = lam_vec
#         ep = 1e-2
#         ep_min = 1e-2
#         f_old = np.ones(G.shape[1])
#         first_f_rec, _, _ = minimize(lam_curr)
#         val = 5e-2
#         k = 1
#         while True:
#             try:
#                 curr_f_rec, LHS, _ = minimize(lam_curr)
#                 if curr_f_rec is None:
#                     print(f"curr_f_rec is None after minimization for iteration {k}")
#                     continue
#             except Exception as e:
#                 print("An error occurred during minimization:", e)
#                 continue

#             curr_noise = (G @ curr_f_rec) - data_noisy
#             LHS_sparse = csr_matrix(LHS)
#             delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#             prev = np.linalg.norm(delta_p)
#             iteration = 1

#             while True:
#                 curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
#                 curr_noise = G @ curr_f_rec - data_noisy
#                 try:
#                     delta_p = spsolve(LHS_sparse, G.T @ curr_noise)
#                 except RuntimeWarning as e:
#                     print("Error with delta_p calculation")
#                     pass
#                 val /= 1.03
#                 val = max(val, 8e-3)
#                 if np.abs(np.linalg.norm(delta_p) / prev - 1) < val:
#                     break
#                 prev = np.linalg.norm(delta_p)
#                 iteration += 1
#             curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

#             phi_new = np.linalg.norm(data_noisy - np.dot(G, curr_f_rec), 2) ** 2
#             psi_lam = np.array(curr_f_rec)
#             c = 1 / gamma
#             lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))

#             if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < ep or k >= maxiter:
#                 return curr_f_rec, lam_curr, val
#             else:
#                 # ep_min = max(ep_min / 1.2, 1e-4)
#                 ep_min = ep_min / 1.2
#                 if ep_min <= 1e-4:
#                     ep_min = 1e-4
#                 lam_curr = lam_new
#                 f_old = curr_f_rec
#                 k = k + 1

#     lam_vec = lam_ini * np.ones(G.shape[1])
#     choice_val = 9e-3
#     best_f_rec1, fin_lam1, _ = fixed_point_algo(gamma_init, lam_vec, choice_val)

#     new_resid = phi_resid(G, best_f_rec1, data_noisy)
#     zero_vec = np.zeros(len(best_f_rec1))
#     zero_resid = phi_resid(G, zero_vec, data_noisy)
#     # gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25
#     gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25

#     new_choice2 = 5e-3
#     best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2)

#     # sum_x = np.sum(best_f_rec2)
#     # x_normalized = best_f_rec2 / sum_x
#     x_normalized = best_f_rec2
#     return x_normalized, fin_lam2

# def LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter):
    # def minimize(lam_vector):
    #     A = (G.T @ G + np.diag(lam_vector))
    #     b = (G.T @ data_noisy)
    #     sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vector, '0', nargin = 4)
    #     return sol, A, b

    # def phi_resid(G, param_vec, data_noisy):
    #     return np.linalg.norm(G @ param_vec - data_noisy, 2)**2

    # def fixed_point_algo(gamma, lam_vec, choice_val, check):
    #     nT2 = G.shape[1]
    #     lam_curr = lam_vec
    #     k = 1
    #     ep = 1e-2
    #     ep_min = 1e-2
    #     f_old = np.ones(G.shape[1])
    #     first_f_rec, _, _ = minimize(lam_curr)
    #     val = 5*1e-2
    #     while True:
    #         #Initial L-Curve solution
    #         try:
    #             # curr_f_rec = minimize(lam_curr, ep_min, epscond)
    #             curr_f_rec, LHS, _ = minimize(lam_curr)
    #             if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
    #                 print(f"curr_f_rec is None after minimization for iteration {k}")
    #             else:
    #                 pass
    #         except Exception as e:
    #             print("An error occurred during minimization:", e)

    #         curr_noise = (G @ curr_f_rec) - data_noisy
    #         delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
    #         prev = np.linalg.norm(delta_p)
    #         LHS_temp = LHS.copy()
    #         iteration = 1 
    #         while iteration < 20:
    #             # curr_f_rec = curr_f_rec - delta_p
    #             # curr_f_rec[curr_f_rec < 0] = 0
    #             curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
    #             curr_noise = G @ curr_f_rec - data_noisy
    #             try:
    #                 delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
    #             except RuntimeWarning as e:
    #                 print("error with delta_p calculation")
    #                 pass
    #             # print("initial val", val)
    #             val /= 1.03
    #             val = max(val, 8e-3)
    #             # print("updated val", val)
    #             if np.abs(np.linalg.norm(delta_p) / prev - 1) < val or iteration > 20:
    #                 new_choice_val = val
    #                 break
    #             prev = np.linalg.norm(delta_p)
    #             iteration += 1

    #         # curr_f_rec = curr_f_rec - delta_p
    #         # curr_f_rec[curr_f_rec < 0] = 0
    #         curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

    #         #Get new solution with new lambda vector
    #         machine_eps = np.finfo(float).eps
    #         #Update lambda: then check
    #         #New Lambda find the new residual and the new penalty
    #         phi_new = np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2
    #         psi_lam = [curr_f_rec[i] for i in range(len(curr_f_rec))]
    #         #define scaling factor;
    #         c = 1/gamma
    #         psi_lam = np.array(psi_lam)
    #         ep_2 = 1e-2
    #         lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))
    #         #If doesnt converge; update f

    #         #Step4: Check stopping criteria based on relative change of regularization parameter eta
    #         #or the  inverse solution
    #         #update criteria of lambda
    #         if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) < ep or k >= maxiter:
    #             if k >= maxiter:
    #                 print("max hit")
    #             else:
    #                 pass
    #             return curr_f_rec, lam_curr, new_choice_val
    #         else:
    #             ep_min = ep_min / 1.2
    #             # print("ep_min",ep_min)
    #             if ep_min <= 1e-4:
    #                 ep_min = 1e-4
    #             # print(f"Finished Iteration {k}")
    #             lam_curr = lam_new
    #             f_old = curr_f_rec
    #             k = k + 1


    # #MAIN CODE FOR ITO LR:
    # #Step 1: Initialize gamma and lambda as lam_vec
    # lam_vec = lam_ini * np.ones(G.shape[1])
    # #Step 2:Run FPA until convergence
    # check = False 
    # choice_val = 9e-3
    # best_f_rec1, fin_lam1, new_choice1 = fixed_point_algo(gamma_init, lam_vec, choice_val,  check = False)
    # #Step 3: Calculate new noise level (phi_resid)
    # new_resid = phi_resid(G, best_f_rec1, data_noisy)
    # #Step 4: Calculate and update new gamma:
    # zero_vec = np.zeros(len(best_f_rec1))
    # zero_resid = phi_resid(G,zero_vec, data_noisy)
    # #If residual is L2:
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25
    # #Step 4: Perform fixed point algo with new gamma value
    # new_choice2 = 5e-3
    # best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2, check = True)
    # f_rec_final = best_f_rec2
    # #normalization:
    # sum_x = np.sum(f_rec_final)
    # # Normalize the vector
    # x_normalized = f_rec_final / sum_x
    # f_rec_final = x_normalized
    # best_f_rec2 = f_rec_final
    # return best_f_rec2, fin_lam2

# def LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter):
#     #Initialize the MRR Problem
#     # TE = np.arange(1,512,4).T
#     # #Generate the T2 values
#     # T2 = np.arange(1,201).T
#     # dT2 = T2[1] - T2[0] 
#     # #Generate G_matrix
#     # G = np.zeros((len(TE),len(T2)))
#     # #For every column in each row, fill in the e^(-TE(i))
#     # for i in range(len(TE)):
#     #     for j in range(len(T2)):
#     #         G[i,j] = np.exp(-TE[i]/T2[j]) * dT2
#     # nTE = len(TE)
#     # nT2 = len(T2)

#     #sigma1 >= 3; cover as many T2 points 
#     # sigma1 = 10
#     # mu1 = 80
#     # sigma2 = 3
#     # mu2 = 100
#     # import matplotlib.pyplot as plt

#     # #Create ground truth
#     # g = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
#     # g = g + (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
#     # g = g/2

#     #Generalize, add G and data_noisy in the input; take outside of the func.
#     #make sure lam_vec is squared??
#     # def minimize(lam_vec):
#     #     A = (G.T @ G + np.diag(lam_vec))
#     #     b = G.T @ data_noisy
#     #     y = cp.Variable(G.shape[1])
#     #     cost = cp.norm(A @ y - b, 2)**2
#     #     constraints = [y >= 0]
#     #     problem = cp.Problem(cp.Minimize(cost), constraints)
#     #     try:
#     #         problem.solve(solver=cp.MOSEK, verbose=False)
#     #         #you can try a different solver if you don't have the license for MOSEK doesn't work (should be free for one year)
#     #     except Exception as e:
#     #         print(e)
#     #     # reconst = y.value
#     #     # reconst,_ = nnls(A,b, maxiter = 10000)
#     #     sol = y.value
#     #     # machine_eps = np.finfo(float).eps
#     #     # sol = np.array(sol)
#     #     # sol[sol < 0] = machine_eps
#     #     # reconst,_ = nnls(A,b, maxiter = 10000)
#     #     return sol

#     def minimize(lam_vector):
#         machine_eps = np.finfo(float).eps
#         # ep = machine_eps
#         # A = (g_mat.T @ g_mat + np.diag(lam_vector))
#         # b = g_mat.T @ data_noisy + (g_mat.T @ g_mat * ep) + ep*lam_vector
#         # b = g_mat.T @ data_noisy
#         # ep = machine_eps
#         # L = np.eye(G.shape[1])
#         # xTikh = np.linalg.solve(A.T@A + lambdah*L.T@L, A.T@b)

#         eps = 1e-2
#         # A = (G.T @ G + np.diag(lam_vector))
#         A = (G.T @ G + np.diag(lam_vector))
#         # b = G.T @ data_noisy + (G.T @ G * eps) @ np.ones(nT2) + eps*lam_vector
#         # ep4 = np.ones(G.shape[1]) * eps
#         # print("(G.T @ data_noisy).shape", (G.T @ data_noisy).shape)
#         # print("(G.T @ G @ ep4).shape", (G.T @ G @ ep4).shape)
#         # print("ep4 * (lam_vector).shape", ep4 * (lam_vector))
#         # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * (lam_vector)
#         b = (G.T @ data_noisy)
#         # sol = np.linalg.solve(A, b)
#         sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vector, '0', nargin = 4)
#         # try:
#         #     # Solve the problem using nnls
#         #     sol = nnls(A, b, maxiter=1000)[0]
#         #     sol = sol - eps
            
#         #     # Ensure non-negative solution
#         #     sol = np.array(sol)
#         #     sol[sol < 0] = 0
            
#         # except Exception as e:
#         #     # Handle exceptions if nnls fails
#         #     y = cp.Variable(G.shape[1])
#         #     cost = cp.norm(A @ y - b, 2)**2
#         #     constraints = [y >= 0]
#         #     problem = cp.Problem(cp.Minimize(cost), constraints)
#         #     try:
#         #         problem.solve(solver=cp.MOSEK, verbose=False)
#         #         #you can try a different solver if you don't have the license for MOSEK doesn't work (should be free for one year)
#         #     except Exception as e:
#         #         print(e)
#         #     # reconst = y.value
#         #     # reconst,_ = nnls(A,b, maxiter = 10000)
#         #     sol = y.value
#         #     print(f"An error occurred during nnls optimization, using MOSEK instead: {e}")
#         #     sol = sol - eps
#         #     # Ensure non-negative solution
#         #     sol = np.array(sol)
#         #     sol[sol < 0] = 0
#             # Alternative statement: Set sol to T2
#         # machine_eps = np.finfo(float).eps
#         # print(type(sol))
#         # sol[sol < 0] = machine_eps
#         return sol, A, b

#     def phi_resid(G, param_vec, data_noisy):
#         return np.linalg.norm(G @ param_vec - data_noisy, 2)**2

#     # import matplotlib.pyplot as plt
#     # import matplotlib
#     # matplotlib.use('TkAgg')  # Use TkAgg backend
#     # import matplotlib.pyplot as plt
#     def fixed_point_algo(gamma, lam_vec, choice_val, check):
#         """
#         gamma: gamma val
#         lam_vec: vector of lambdas
#         """
#         nT2 = G.shape[1]
#         # lam_curr = np.sqrt(lam_vec)
#         lam_curr = lam_vec
#         k = 1

#         ep = 1e-2
#         # ep = 1e-3
#         ep_min = 1e-2
#         # epscond = False
#         # ini_f_rec = minimize(lam_curr, ep_min, epscond)
#         f_old = np.ones(G.shape[1])

#         c_arr = []
#         lam_arr = []
#         sol_arr = []
#         # fig, axs = plt.subplots(3, 1, figsize=(6, 6))
#         # fig,ax = plt.subplots(5,1,figsize = (12,8))
#         # # # Show the initial plot
#         # plt.tight_layout()
#         # plt.ion()  # Turn on interactive mode
        
#         first_f_rec, _, _ = minimize(lam_curr)
#         #Uncomment the code below to run the Fixed Point Algo (FPA) using while loop until convergence
#         # val = 5 * 1e-2
#         val = 5*1e-2
#         # plt.ion()
#         while True:
#             #Initial L-Curve solution
#             try:
#                 # curr_f_rec = minimize(lam_curr, ep_min, epscond)
#                 curr_f_rec, LHS, RHS = minimize(lam_curr)
#                 if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
#                     print(f"curr_f_rec is None after minimization for iteration {k}")
#                 else:
#                     pass
#             except Exception as e:
#                 print("An error occurred during minimization:", e)

#             curr_noise = (G @ curr_f_rec) - data_noisy
#             delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
#             prev = np.linalg.norm(delta_p)
#             LHS_temp = LHS.copy()
#             iteration = 1 
#             while True or iteration < 20:
#                 curr_f_rec = curr_f_rec - delta_p
#                 curr_f_rec[curr_f_rec < 0] = 0
#                 curr_noise = G @ curr_f_rec - data_noisy
#                 try:
#                     delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
#                 except RuntimeWarning as e:
#                     print("error with delta_p calculation")
#                     pass
#                 # print("initial val", val)
#                 val = val/1.03
#                 if val < 8 * 1e-3:
#                     val = 8 * 1e-3
#                 # print("updated val", val)
#                 if np.abs(np.linalg.norm(delta_p) / prev - 1) < val or iteration > 20:
#                     new_choice_val = val
#                     break
#                 prev = np.linalg.norm(delta_p)
#                 iteration += 1

#             curr_f_rec = curr_f_rec - delta_p
#             curr_f_rec[curr_f_rec < 0] = 0

#             #track.append(np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x))
#             # if (np.linalg.norm(psi_lam/phi_new)) < 1e-4:
#             #     print("condition passed")
#             #     print("np.linalg.norm(psi_lam/ phi_new)", np.linalg.norm(psi_lam/ phi_new))
#             #     psi_lam = psi_lam + ep_min
#             # psi_lam = list(psi_lam)


#             #Get new solution with new lambda vector
#             if check == True:
#                 pass
#                 # fig, axs = plt.subplots(2, 1, figsize=(12, 12))
#                 # # axs[0].plot(T2,g, color = "black", label = "ground truth")
#                 # axs[0].plot(T2, curr_f_rec, label = "reconstruction")
#                 # axs[1].semilogy(T2, lam_curr, label = "lambdas")
#                 # # Redraw the plot
#                 # # plt.draw()
#                 # plt.tight_layout()
#                 # plt.pause(0.01)
#                 # plt.show()
#             # plt.ioff()
#             # plt.close()
#             machine_eps = np.finfo(float).eps

#             #Update lambda: then check
#             #New Lambda find the new residual and the new penalty
#             phi_new = np.linalg.norm(data_noisy - np.dot(G,curr_f_rec), 2)**2
#             # print("phi_new",phi_new)
#             # lam_new = []
#             # for i in range(len(curr_f_rec)):
#             #     nu_i = (phi_new + sum(curr_f_rec[j] for j in range(len(curr_f_rec)) if j != i)) / (curr_f_rec[i] + 1e-3)
#             #     lam_new.append(nu_i)
#             # add_port = [phi_new + lam_curr[i] * curr_f_rec[i] for i in range(len(curr_f_rec))]
            
#             psi_lam = [curr_f_rec[i] for i in range(len(curr_f_rec))]

#             #define scaling factor;
#             c = 1/gamma
#             # c = ((gamma**gamma)/((1+gamma)**(1+gamma)))/gamma
#             # c1 =((gamma**gamma)/((1+gamma)**(1+gamma)))/(gamma + ((gamma**gamma)/((1+gamma)**(1+gamma))))
#             # c = 1/(1 + gamma)
#             c1 = 1
#             # print("gamma", gamma)
#             # c = np.std(data_noisy - - np.dot(G,first_f_rec))
#             c_arr.append(c)

#             psi_lam = np.array(psi_lam)
#             # lam_new = np.array(lam_new)
#             # lam_new = c * lam_new
#             # print("psi_lam",np.median(psi_lam))
            
#             ep_2 = 1e-2
#             # ep_min = 1e-5
#             # print("np.linalg.norm(psi_lam):", np.linalg.norm(psi_lam))
#             #STEP 4
#             #redefine new lam
#             # plt.plot(lam_curr)
#             # plt.show()
#             # lam_new = c * (phi_new / (psi_lam + ((lam_curr > 1e2) & (lam_curr < 1e-3)) * ep_2) + (lam_curr > 1e3) * ep_min)
#             # lam_new = c * (phi_new / (psi_lam + machine_eps))
#             # lam_new = c * (phi_new / (psi_lam + (psi_lam < 0.5 * np.median(curr_f_rec)) * machine_eps))
#             lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))
#             # print("5 * np.min(curr_f_rec)",5 * np.min(curr_f_rec))
#             # lam_new2 = c * (phi_new / (psi_lam + 1e-5))
#             # curr_f_rec3 = minimize(lam_new)
#             # curr_f_rec2 = minimize(lam_new2)
#             # print("np.median(first_f_rec)",np.median(first_f_rec))
#             # # print("np.where(psi_lam > 1.5 * np.median(first_f_rec))", np.where(psi_lam < np.median(first_f_rec)))
#             # # lam_new = c * (phi_new / (psi_lam + (psi_lam < machine_eps) * ep_min))
#             # ax[0].plot(lam_curr)
#             # ax[1].plot(curr_f_rec3)
#             # ax[2].plot(lam_new)
#             # ax[3].plot(lam_new2)
#             # ax[4].plot(curr_f_rec2)
#             # # ax[2].set_ylim(0, 
#             # # upper_limit_2)
#             # # ax[2].set_ylim(0, 25)
#             # # ax[3].set_ylim(0, 10) 
#             # # ax[2].set_ylim(0, upper_limit_2)
#             # ax[0].legend(["lam_curr"])
#             # ax[1].legend(["curr_f_rec_lam_new"])
#             # ax[2].legend(["lam_global_reg"])
#             # ax[3].legend(["lam_peaks_reg"])
#             # ax[4].legend(["curr_f_rec_lam_peaks_reg"])

#             # plt.draw()
#             # plt.tight_layout()
#             # plt.pause(0.0000001)            # if np.median(psi_lam) < 1e-4:
#             #     lam_new = c * (phi_new / (psi_lam + ep_min))
#             # else:
#             #     lam_new = c * (phi_new / (psi_lam))
#             # print("Lam_new.shape", lam_new.shape)
#             # cs = c * np.ones(len(psi_lam))

#             #If doesnt converge; update f

#             #Step4: Check stopping criteria based on relative change of regularization parameter eta
#             #or the  inverse solution
#             #update criteria of lambda
#             # if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) + (np.linalg.norm(lam_new-lam_curr)/np.linalg.norm(lam_curr)) < ep or k >= maxiter:
#             if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) < ep or k >= maxiter:
#                 # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old))
#                 # print("ep value: ", ep)
#                 if k >= maxiter:
#                     print("max hit")
#                 else:
#                     pass
#                 c_arr_fin = np.array(c_arr)
#                 lam_arr_fin = np.array(lam_arr)
#                 sol_arr_fin = np.array(sol_arr)
#                 # plt.ioff()
#                 # plt.show()
#                 # print(f"Total of {k} iterations")
#                 return curr_f_rec, lam_curr, new_choice_val
#             else:
#                 ep_min = ep_min / 1.2
#                 # print("ep_min",ep_min)
#                 if ep_min <= 1e-4:
#                     ep_min = 1e-4
#                 # print(f"Finished Iteration {k}")
#                 lam_curr = lam_new
#                 f_old = curr_f_rec
#                 k = k + 1
#                 lam_arr.append(lam_new)
#                 sol_arr.append(curr_f_rec)


#         #Running the FPA iteration by iteration
#         # testiter = 5
#         # for k in range(testiter):
#         #     try:
#         #         # curr_f_rec = minimize(lam_curr, ep_min, epscond)
#         #         curr_f_rec = minimize(lam_curr)
#         #         if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
#         #             print(f"curr_f_rec is None after minimization for iteration {k}")
#         #         else:
#         #             pass
#         #     except Exception as e:
#         #         print("An error occurred during minimization:", e)
#         #
#         #     # Get new solution with new lambda vector
#         #
#         #     # Update lambda: then check
#         #     # New Lambda find the new residual and the new penalty
#         #     phi_new = np.linalg.norm(data_noisy - np.dot(G, curr_f_rec), 2)**2
#         #     psi_lam = [curr_f_rec[i] * lam_curr[i] for i in range(len(lam_curr))]
#         #
#         #
#         #     # define scaling factor;
#         #     c = 1 / (1 + gamma)
#         #     c = ((gamma**gamma)/((1+gamma)**(1+gamma)))
#         #     # c = np.std(data_noisy - - np.dot(G,curr_f_rec))/gamma
#         #     # c = ((gamma**gamma)/((1+gamma)**(1+gamma)))/gamma
#         #
#         #     c_arr.append(c)
#         #
#         #     # STEP 4
#         #     # redefine new lam
#         #     lam_new = c * (phi_new / psi_lam)
#         #
#         #     #Make terms into arrays
#         #     psi_lam = np.array(psi_lam)
#         #     lam_new = np.array(lam_new)
#         #     machine_eps = np.finfo(float).eps
#         #
#         #     #Try Yvonne's idea
#         #     if np.any(psi_lam/phi_new) < machine_eps:
#         #         print("condition satisfied")
#         #
#         #     # If doesnt converge; update f
#         #
#         #     #Plot iteration by iteration
#         #     if check == True:
#         #         axs[0].plot(T2, g, color="black", label="ground truth")
#         #         axs[0].plot(T2, curr_f_rec, label="reconstruction")
#         #         axs[1].semilogy(T2, lam_curr, label="lambdas")
#         #         axs[2].semilogy(T2, lam_new, label="new lambda")
#         #         # axs[3].semilogy(T2, test, label="lambda_new")
#         #         # axs[4].semilogy(T2, np.array(psi_lam2), label="lambda_new * reconstruction")
#         #
#         #         # Redraw the plot
#         #         plt.draw()
#         #         # axs[0].legend()
#         #         # axs[1].legend()
#         #         # axs[2].legend()
#         #         # axs[3].legend()
#         #         # axs[4].legend()
#         #         plt.tight_layout()
#         #         plt.pause(0.001)
#         #     else:
#         #         pass
#         #
#         #     # Step4: Check stopping criteria based on relative change of regularization parameter eta
#         #     # or the  inverse solution
#         #     # update criteria of lambda
#         #     if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) + (np.linalg.norm(lam_new - lam_curr) / np.linalg.norm(lam_curr)) < ep or k == maxiter-1 or  k >= maxiter:
#         #         # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old))
#         #         # print("ep value: ", ep)
#         #         c_arr_fin = np.array(c_arr)
#         #         lam_arr_fin = np.array(lam_arr)
#         #         sol_arr_fin = np.array(sol_arr)
#         #         plt.ioff()
#         #         plt.show()
#         #         print(f"Total of {k} iterations")
#         #         return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
#         #     else:
#         #         # ep_min = ep_min / 1.2
#         #         # if ep_min <= 1e-4:
#         #         #     ep_min = 1e-4
#         #         # print(f"Finished Iteration {k}")
#         #         k = k + 1
#         #         lam_curr = lam_new
#         #         f_old = curr_f_rec
#         #         lam_arr.append(lam_new)
#         #         sol_arr.append(curr_f_rec)

#     #MAIN CODE FOR ITO LR:

#     #Step 1: Initialize gamma and lambda as lam_vec
#     lam_vec = lam_ini * np.ones(G.shape[1])
#     # lam_vec = np.square(lam_vec)
#     #Step 2:Run FPA until convergence
#     check = False 
#     choice_val = 9e-3
#     best_f_rec1, fin_lam1, new_choice1 = fixed_point_algo(gamma_init, lam_vec, choice_val,  check = False)
#     # print("first FPA is done")
#     # fin_lam1 = np.sqrt(fin_lam1)

#     #Step 3: Calculate new noise level (phi_resid)
#     new_resid = phi_resid(G, best_f_rec1, data_noisy)
    
#     #Step 4: Calculate and update new gamma:
#     zero_vec = np.zeros(len(best_f_rec1))
#     zero_resid = phi_resid(G,zero_vec, data_noisy)

#     #If residual is L2:
#     gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25
#     # print("gamma_new", gamma_new)
#     #If residual is L1:
#     # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.5
 
#     #Step 4: Perform fixed point algo with new gamma value
#     # check = True ; if want to print iteration by ieration
#     # fin_lam1 = np.sqrt(fin_lam1)
#     new_choice2 = 5e-3
#     best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2, check = True)

#     # if normset == True:
#     f_rec_final = best_f_rec2
#     #normalization:
#     # y_rec_temp = G @ f_rec_final
#     # y_ratio = np.max(data_noisy)/np.max(y_rec_temp)
#     # f_rec_final = y_ratio * f_rec_final
#     sum_x = np.sum(f_rec_final)
#     # Normalize the vector
#     x_normalized = f_rec_final / sum_x
#     f_rec_final = x_normalized
#     best_f_rec2 = f_rec_final
#     # best_f_rec2 = f_rec_final
#     # else:
#     #     pass
#     # fin_lam2 = np.sqrt(fin_lam2)
#     return best_f_rec2, fin_lam2


def LocReg_Ito_class(data_noisy, G, lam_ini, gamma_init, maxiter):
    #Initialize the MRR Problem
    # TE = np.arange(1,512,4).T
    # #Generate the T2 values
    # T2 = np.arange(1,201).T
    # dT2 = T2[1] - T2[0] 
    # #Generate G_matrix
    # G = np.zeros((len(TE),len(T2)))
    # #For every column in each row, fill in the e^(-TE(i))
    # for i in range(len(TE)):
    #     for j in range(len(T2)):
    #         G[i,j] = np.exp(-TE[i]/T2[j]) * dT2
    # nTE = len(TE)
    # nT2 = len(T2)

    #sigma1 >= 3; cover as many T2 points 
    # sigma1 = 10
    # mu1 = 80
    # sigma2 = 3
    # mu2 = 100
    # import matplotlib.pyplot as plt

    # #Create ground truth
    # g = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
    # g = g + (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
    # g = g/2

    #Generalize, add G and data_noisy in the input; take outside of the func.
    #make sure lam_vec is squared??
    # def minimize(lam_vec):
    #     A = (G.T @ G + np.diag(lam_vec))
    #     b = G.T @ data_noisy
    #     y = cp.Variable(G.shape[1])
    #     cost = cp.norm(A @ y - b, 2)**2
    #     constraints = [y >= 0]
    #     problem = cp.Problem(cp.Minimize(cost), constraints)
    #     try:
    #         problem.solve(solver=cp.MOSEK, verbose=False)
    #         #you can try a different solver if you don't have the license for MOSEK doesn't work (should be free for one year)
    #     except Exception as e:
    #         print(e)
    #     # reconst = y.value
    #     # reconst,_ = nnls(A,b, maxiter = 10000)
    #     sol = y.value
    #     # machine_eps = np.finfo(float).eps
    #     # sol = np.array(sol)
    #     # sol[sol < 0] = machine_eps
    #     # reconst,_ = nnls(A,b, maxiter = 10000)
    #     return sol

    def minimize(lam_vector):
        machine_eps = np.finfo(float).eps
        # ep = machine_eps
        # A = (g_mat.T @ g_mat + np.diag(lam_vector))
        # b = g_mat.T @ data_noisy + (g_mat.T @ g_mat * ep) + ep*lam_vector
        # b = g_mat.T @ data_noisy
        # ep = machine_eps
        # L = np.eye(G.shape[1])
        # xTikh = np.linalg.solve(A.T@A + lambdah*L.T@L, A.T@b)

        eps = 1e-2
        # A = (G.T @ G + np.diag(lam_vector))
        A = (G.T @ G + np.diag(lam_vector))

        # b = G.T @ data_noisy + (G.T @ G * eps) @ np.ones(nT2) + eps*lam_vector
        ep4 = np.ones(G.shape[1]) * eps
        # print("(G.T @ data_noisy).shape", (G.T @ data_noisy).shape)
        # print("(G.T @ G @ ep4).shape", (G.T @ G @ ep4).shape)
        # print("ep4 * (lam_vector).shape", ep4 * (lam_vector))
        b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * (lam_vector)
        # b = (G.T @ data_noisy)
        # sol = np.linalg.solve(A, b)
        try:
            # Solve the problem using nnls
            # sol = nnls(A, b, maxiter=10000)[0]
            sol = np.linalg.solve(A,b)
            sol = sol - eps
            
            # Ensure non-negative solution
            sol = np.array(sol)
            # sol[sol < 0] = 0
            
        except Exception as e:
            # Handle exceptions if nnls fails
            y = cp.Variable(G.shape[1])
            cost = cp.norm(A @ y - b, 2)**2
            # constraints = [y >= 0]
            constraints = [None]
            problem = cp.Problem(cp.Minimize(cost), constraints)
            try:
                problem.solve(solver=cp.MOSEK, verbose=False)
                #you can try a different solver if you don't have the license for MOSEK doesn't work (should be free for one year)
            except Exception as e:
                print(e)
            # reconst = y.value
            # reconst,_ = nnls(A,b, maxiter = 10000)
            sol = y.value
            print(f"An error occurred during nnls optimization, using MOSEK instead: {e}")
            sol = sol - eps
            # Ensure non-negative solution
            sol = np.array(sol)
            sol[sol < 0] = 0
            # Alternative statement: Set sol to T2
        # machine_eps = np.finfo(float).eps
        # print(type(sol))
        # sol[sol < 0] = machine_eps
        return sol, A, b

    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2)**2

    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')  # Use TkAgg backend
    # import matplotlib.pyplot as plt
    def fixed_point_algo(gamma, lam_vec, choice_val, check):
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
        # val = 5 * 1e-2
        val = 5*1e-2
        # plt.ion()
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
                # curr_f_rec[curr_f_rec < 0] = 0
                curr_noise = G @ curr_f_rec - data_noisy
                try:
                    delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
                except RuntimeWarning as e:
                    print("error with delta_p calculation")
                    pass
                # print("initial val", val)
                val = val/1.03
                if val < 8 * 1e-3:
                    val = 8 * 1e-3
                # print("updated val", val)
                if np.abs(np.linalg.norm(delta_p) / prev - 1) < val:
                    new_choice_val = val
                    break
                prev = np.linalg.norm(delta_p)

            curr_f_rec = curr_f_rec - delta_p
            # curr_f_rec[curr_f_rec < 0] = 0

            #track.append(np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x))
            # if (np.linalg.norm(psi_lam/phi_new)) < 1e-4:
            #     print("condition passed")
            #     print("np.linalg.norm(psi_lam/ phi_new)", np.linalg.norm(psi_lam/ phi_new))
            #     psi_lam = psi_lam + ep_min
            # psi_lam = list(psi_lam)


            #Get new solution with new lambda vector
            if check == True:
                pass
                # fig, axs = plt.subplots(2, 1, figsize=(12, 12))
                # # axs[0].plot(T2,g, color = "black", label = "ground truth")
                # axs[0].plot(T2, curr_f_rec, label = "reconstruction")
                # axs[1].semilogy(T2, lam_curr, label = "lambdas")
                # # Redraw the plot
                # # plt.draw()
                # plt.tight_layout()
                # plt.pause(0.01)
                # plt.show()
            # plt.ioff()
            # plt.close()
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
            lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))
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
                return curr_f_rec, lam_curr, new_choice_val
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

    #MAIN CODE FOR ITO LR:

    #Step 1: Initialize gamma and lambda as lam_vec
    lam_vec = lam_ini * np.ones(G.shape[1])
    # lam_vec = np.square(lam_vec)
    #Step 2:Run FPA until convergence
    check = False 
    choice_val = 9e-3
    best_f_rec1, fin_lam1, new_choice1 = fixed_point_algo(gamma_init, lam_vec, choice_val,  check = False)
    # print("first FPA is done")
    # fin_lam1 = np.sqrt(fin_lam1)

    #Step 3: Calculate new noise level (phi_resid)
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    
    #Step 4: Calculate and update new gamma:
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G,zero_vec, data_noisy)

    #If residual is L2:
    gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25

    # print("gamma_new", gamma_new)
    #If residual is L1:
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.5
 
    #Step 4: Perform fixed point algo with new gamma value
    # check = True ; if want to print iteration by ieration
    # fin_lam1 = np.sqrt(fin_lam1)
    new_choice2 = 5e-3
    best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2, check = True)
    # fin_lam2 = np.sqrt(fin_lam2)
    return best_f_rec2, fin_lam2


### 2 Parameter Ito problem
def Ito_LocReg(data_noisy, G, lam_ini, gamma, param_num, B_mats, maxiter):
    # def minimize(eta1, eta2):
    #     A = (G.T @ G + eta1 * B_1.T @ B_1 + eta2 * B_2.T @ B_2)
    #     b = G.T @ data_noisy
    #     y = cp.Variable(nT2)
    #     cost = cp.norm(A @ y - b, 2)**2
    #     # constraints = [x >= 0]
    #     # problem = cp.Problem(cp.Minimize(cost), constraints)
    #     # constraints = [x >= 0]
    #     constraints = [y >= 0]
    #     problem = cp.Problem(cp.Minimize(cost), constraints)
    #     # problem = cp.Problem(cp.Minimize(cost))
    #     # exp,_ = fnnls(A,b)
    #     problem.solve(solver=cp.MOSEK, verbose=False)
    #     reconst = y.value
    #     return reconst

    def minimize(eta1, eta2):
        A = (G.T @ G + eta1 * B_1.T @ B_1 + eta2 * B_2.T @ B_2)
        b = G.T @ data_noisy
        reconst,_ = nnls(A,b, maxiter = 1000)
        return reconst
    # x0_LS = lsqnonneg(G, data_noisy)[0]
    # x0_LS_nnls = nnls(G, data_noisy)[0]
    # estimated_noise = lsqnonneg(G, data_noisy)[2] 
    nT2 = G.shape[1]
    eta_ini = lam_ini * np.ones(param_num)
    B_1 = B_mats[0]
    B_2 = B_mats[1]

    # print("B_1: ", B_1)
    # print("B_2: ", B_2)

    eta_curr = eta_ini
    ep = 1e-3
    k = 1
    curr_eta1 = eta_curr[0]
    # print("ini_eta1: ", curr_eta1)
    curr_eta2 = eta_curr[1]
    # print("ini_eta2: ", curr_eta2)
    ini_f_rec = minimize(curr_eta1, curr_eta2)
    c = np.std(data_noisy - np.dot(G,ini_f_rec))
    while True:
        # print(f"Printing Iteration {k}")
        #Step2: Minimization
        curr_f_rec = minimize(curr_eta1, curr_eta2)
        # x,_ = nnls(A, b, maxiter) 
        # x = x - ep
        # x[x < 0] = 0
        # u_new = x
        # print("curr_f_rec: ", curr_f_rec)
        # curr_f_rec = np.linalg.inv(G.T @ G + curr_eta1 * B_1.T @ B_1 + curr_eta2 * B_2.T @ B_2) @ (G.T @ data_noisy)
        u_new = curr_f_rec
        #Step 3: Update
        phi_new = 0.5 * np.linalg.norm(data_noisy - np.dot(G,u_new))**2
        psi_1 = 0.5 * np.linalg.norm(B_1 @ u_new)**2
        # print("psi_1", psi_1)
        psi_2 = 0.5 * np.linalg.norm(B_2 @ u_new)**2
        # print("psi_2", psi_2)

        #Run 1

        #Run 2
        # eta_new_1 = (1/ gamma) * ((phi_new )/psi_1)
        # eta_new_2 = (1/ gamma) * ((phi_new )/psi_2)
        
        #Run 3
        eta_new_1 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_1)
        eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_2)

        # #Run 4
        # eta_new_1 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new + curr_eta2 * psi_2)/psi_1)
        # eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new + curr_eta1 * psi_1)/psi_2)

        #Run 5
        # eta_new_1 = (c) * ((phi_new + curr_eta2 * psi_2)/psi_1)
        # eta_new_2 = (c) * ((phi_new + curr_eta1 * psi_1)/psi_2)

        # eta_new_1 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_1)
        # eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_2)
        

        #Get new solution
        reconst = minimize(eta_new_1, eta_new_2)
        # new_f_rec = np.linalg.inv(G.T @ G + eta_new_1 * B_1.T @ B_1 + eta_new_2 * B_2.T @ B_2) @ (G.T @ data_noisy)
        new_f_rec = reconst
        # print("new_f_rec: ", new_f_rec)

        #Step4: Check stopping criteria based on relative change of regularization parameter eta 
        #or the  inverse solution 
        # if np.abs(eta_new_1-curr_eta1)/curr_eta1 < ep and np.abs(eta_new_2-curr_eta2)/curr_eta2 < ep:
        #     print("(eta_new_1-curr_eta1)/curr_eta1: ", np.abs(eta_new_1-curr_eta1)/curr_eta1)
        #     print("(eta_new_2-curr_eta2)/curr_eta2: ", np.abs(eta_new_2-curr_eta2)/curr_eta2)
        #     print("ep value: ", ep)
        #     break
        if (np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)) < ep or k >= maxiter:
            # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)))
            # print("ep value: ", ep)
            break
        else:
            # print(f"Finished Iteration {k}")
            # print("curr_eta1", curr_eta1)
            # print("curr_eta2", curr_eta2)
            # print("eta_new_1", eta_new_1)
            # print("eta_new_2", eta_new_2)
            # print("(eta_new_1-curr_eta1)/curr_eta1: ",  np.abs(eta_new_1-curr_eta1)/curr_eta1)
            # print("(eta_new_2-curr_eta2)/curr_eta2: ",np.abs(eta_new_2-curr_eta2)/curr_eta2)
            curr_eta1 = eta_new_1
            curr_eta2 = eta_new_2
            k = k + 1

    #Return new eta values
    eta1_fin = eta_new_1
    eta2_fin = eta_new_2
    fin_etas = np.array([eta1_fin, eta2_fin])
    best_f_rec =  minimize(eta1_fin, eta2_fin)
    # print(f"Total of {k} iterations")
    return best_f_rec, fin_etas

    # phi = data_noisy - np.dot(G,x)

    # x, _ = nnls(G, data_noisy,maxiter)
    # phi = data_noisy - np.dot(G,x)
    # ep = 1e-2
    # nT2 = G.shape[1]
    # prev_x = x0_ini

    # estimated_noise_std = np.std(estimated_noise)
    # #track = []
    # cur_iter = 1

    # while True:
    #     lambda_val = estimated_noise_std / (np.abs(x0_ini) + ep)
    #     LHS = G.T @ G + np.diag(lambda_val)
    #     RHS = G.T @ data_noisy + (G.T @ G * ep) @ np.ones(nT2) + ep*lambda_val
    #     #RHS = G.T @ data_noisy
    #     try:
    #         x0_ini = nnls(LHS,RHS, maxiter = 10000)[0]
    #     except RuntimeWarning as e:
    #         print("x0_ini error calculation")
    #         pass
    #     x0_ini = x0_ini - ep
    #     x0_ini[x0_ini < 0] = 0
    #     #curr_noise = np.dot(G, x0_ini) - data_noisy
    #     curr_noise = (G @ x0_ini) - data_noisy

    #     delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
    #     prev = np.linalg.norm(delta_p)
    #     LHS_temp = LHS.copy()
    #     while True:
    #         x0_ini = x0_ini - delta_p
    #         x0_ini[x0_ini < 0] = 0
    #         curr_noise = G @ x0_ini - data_noisy
    #         try:
    #             delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
    #         except RuntimeWarning as e:
    #             print("error with delta_p calculation")
    #             pass
    #         if np.abs(np.linalg.norm(delta_p) / prev - 1) < 1e-2:
    #             break
    #         prev = np.linalg.norm(delta_p)

    #     x0_ini = x0_ini - delta_p
    #     x0_ini[x0_ini < 0] = 0

    #     #track.append(np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x))
    #     if (np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x)) < 1e-2 or cur_iter >= maxiter:
    #         break

    #     # ax.plot(T2, x0_ini)
    #     # plt.draw()

    #     ep = ep / 1.2
    #     if ep <= 1e-4:
    #         ep = 1e-4
    #     cur_iter = cur_iter + 1
    #     prev_x = x0_ini

    # f_rec_logreg = x0_ini
    # lambda_locreg = lambda_val

    # return f_rec_logreg, lambda_locreg

# Ito_LocReg(data_noisy, G, lam_ini, gamma, param_num, B_mats, maxiter)
# def LocReg_Ito_mod(data_noisy, G, lam_ini, gamma, maxiter):
#     def minimize(lam_vec):
#         A = (G.T @ G + np.diag(lam_vec))
#         b = G.T @ data_noisy
#         y = cp.Variable(nT2)
#         cost = cp.norm(A @ y - b, 2)**2
#         # constraints = [x >= 0]
#         # problem = cp.Problem(cp.Minimize(cost), constraints)
#         # constraints = [x >= 0]
#         # constraints = [y >= 0]
#         # constraints = [np.diag(lam) @ y ] 
#         # constraints = [cp.diag(lam) @ y == cp.diag(lam)[0] * np.ones_like(y)]
#         # constraints = [cp.norm(lam_vec[0] * y[0])**2 <= cp.sum([cp.norm(lam_vec[i] * y[i])**2 for i in range(1, len(lam_vec))])]
#         # constraints = [cp.norm(lam_vec[0] * y[0])**2 - cp.sum([cp.norm(lam_vec[i] * y[i])**2 for i in range(1, len(lam_vec))]) <= 0]
#         # constraints = [cp.norm(lam_vec[0] * y[0])**2 <= 5]
#         # lam_vec2 = cp.Variable(lam_vec)
#         # lam_vec2 = cp.Variable(nT2)
#         # constraints = [cp.sum([-cp.square(cp.norm(lam_vec[i] * y[i], 2)) for i in range(len(lam_vec))]) <= 0]
#         # constraints = [cp.sum([-cp.square(cp.norm(lam_vec[i] * y[i], 2)) for i in range(len(lam_vec))]) <= 0]
#         # constraints = [cp.sum(-cp.square(cp.norm(y[i], 2))) <= 0 for i in range(len(lam_vec))]

#         # constraints = [cp.sum([cp.square(cp.norm(lam_vec[i] * y[i], 2)) for i in range(1, len(lam_vec))]) <= cp.square(cp.norm(lam_vec[0] * y[0]))]
#         # constraints = [cp.sum([cp.square(cp.norm(lam_vec[i] * y[i], 2)) for i in range(1, len(lam_vec))]) >= cp.square(cp.norm(lam_vec[0] * y[0]))]
#         # constraints = [cp.sum([cp.square(cp.norm(lam_vec[i] * y[i], 2)) for i in range(1, len(lam_vec))]) >= cp.square(cp.norm(lam_vec[0] * y[0]))]
#         # constraints = [cp.square(cp.norm(lam_vec[0] * y[0])) /
#         #                          cp.sum(cp.hstack([cp.square(cp.norm(lam_vec[i] * y[i], 2)) for i in range(1, len(lam_vec))])) <= 1]
#         #                         #  cp.norm(cp.hstack([lam_vec[0] * y[0] - lam_vec[i] * y[i]) for  i in range(1, len(lam_vec)]),2)]
#         # constraints = [cp.quad_over_lin(cp.square(cp.norm(lam_vec[0] * y[0])), cp.sum([cp.square(cp.norm(lam_vec[i] * y[i], 2)) for i in range(1, len(lam_vec))])) <= 1, cp.quad_over_lin(cp.square(cp.norm(lam_vec[0] * y[0])), cp.sum([cp.square(cp.norm(lam_vec[i] * y[i], 2)) for i in range(1, len(lam_vec))])) >= 0]
#         # part1 = [cp.square(cp.norm(lam_vec[0] * y[0])) >= 0]
#         # part2 =  [(cp.sum(cp.hstack([cp.square(cp.norm(lam_vec[i] * y[i], 2)) for i in range(1, len(lam_vec))])))]
        
#         # print("curvature of part1:", part1.curvature)
#         # print("curvature of part2:", part2.curvature)
#         constraints = [y >= 0]
#         # constraints = [cp.square(cp.norm(lam_vec[i] * y[i], 2)) == cp.square(cp.norm(lam_vec[0] * y[0], 2)) for i in range(1, len(lam_vec))]
#         # cp.hstack([])
#         # constraints = [
#         #     cp.square(cp.norm(lam_vec[i] * y[i])) + cp.square(cp.norm(lam_vec[0] * y[0])) >= 0 for i in range(1, len(lam_vec))
#         # ]
#         # constraints = [
#         #     cp.sum(cp.square(cp.norm(lam_vec[i] * y[i])))  >= 0 for i in range(len(lam_vec))
#         # ]
#         # cost1 = cp.sum(cp.norm)
#         # print("y[0]",y[0])
#         # val = np.linalg.norm(lam_vec[0] * y[0], 2)**2
#         # val = cp.norm(lam_vec[0] * y[0], 2)**2

#         # constraints = [
#         #     sum(cp.square(cp.norm(lam_vec[i] * y[i], 2))) >= 0 for i in range(len(lam_vec))
#         # ]
#         # # constraints.append(y >= 0)  # Append non-negativity constraint for y
#         # additional_constraints = [
#         #     sum(cp.square(cp.norm(lam_vec[i] * y[i], 2))) <= 0 for i in range(len(lam_vec))
#         # ]
#         # constraints.extend(additional_constraints) 

#         problem = cp.Problem(cp.Minimize(cost), constraints)
#         print(f"combined is DCP: {problem.is_dcp()}")
#         print("combined is done")
#         # print("starting prob 2 ")
#         # problem2 = cp.Problem(cp.Minimize(cost), part1)
#         # print(f"part1 is DCP: {problem2.is_dcp()}")
#         # problem3 = cp.Problem(cp.Minimize(cost), part2)
#         # print(f"part2 is DCP: {problem3.is_dcp()}")
#         # problem = cp.Problem(cp.Minimize(cost))
#         # exp,_ = fnnls(A,b)
#         try:
#             problem.solve(solver=cp.MOSEK, verbose=False)
#         except Exception as e:
#             print(e)
#         reconst = y.value
#         print("reconst:", reconst)
#         return reconst

#     def diag_mat(lam, i):
#         diag = np.zeros_like(lam)
#         diag[i] = lam[i]
#         return np.diag(diag)
#     # def minimize(eta1, eta2):
#     #     A = (G.T @ G + eta1 * B_1.T @ B_1 + eta2 * B_2.T @ B_2)
#     #     b = G.T @ data_noisy
#     #     y = cp.Variable(nT2)
#     #     cost = cp.norm(A @ y - b, 2)**2
#     #     # constraints = [x >= 0]
#     #     # problem = cp.Problem(cp.Minimize(cost), constraints)
#     #     # constraints = [x >= 0]
#     #     constraints = [y >= 0]
#     #     problem = cp.Problem(cp.Minimize(cost), constraints)
#     #     # problem = cp.Problem(cp.Minimize(cost))
#     #     # exp,_ = fnnls(A,b)
#     #     problem.solve(solver=cp.MOSEK, verbose=False)
#     #     reconst = y.value
#     #     return reconst
#     # x0_LS = lsqnonneg(G, data_noisy)[0]
#     # x0_LS_nnls = nnls(G, data_noisy)[0]
#     # estimated_noise = lsqnonneg(G, data_noisy)[2] 
#     nT2 = G.shape[1]
#     lam_vec = lam_ini * np.ones(nT2)

#     lam_curr = lam_vec
#     ep = 1e-3
#     k = 1
#     # curr_eta1 = eta_curr[0]
#     # print("ini_eta1: ", curr_eta1)
#     # curr_eta2 = eta_curr[1]
#     # print("ini_eta2: ", curr_eta2)
#     ini_f_rec = minimize(lam_curr)
#     print(ini_f_rec)
#     c = np.std(data_noisy - np.dot(G,ini_f_rec))
#     while True:
#         print(f"Printing Iteration {k}")
#         #Step2: Minimization
#         curr_f_rec = minimize(lam_curr)
#         # x,_ = nnls(A, b, maxiter) 
#         # x = x - ep
#         # x[x < 0] = 0
#         # u_new = x
#         # print("curr_f_rec: ", curr_f_rec)
#         # curr_f_rec = np.linalg.inv(G.T @ G + curr_eta1 * B_1.T @ B_1 + curr_eta2 * B_2.T @ B_2) @ (G.T @ data_noisy)
#         u_new = curr_f_rec
#         #Step 3: Update
#         phi_new = np.linalg.norm(data_noisy - np.dot(G,u_new))**2
#         # psi_1 = 0.5 * np.linalg.norm(B_1 @ u_new)**2
#         # print("psi_1", psi_1)
#         # psi_2 = 0.5 * np.linalg.norm(B_2 @ u_new)**2
#         # print("psi_2", psi_2)
#         # psi_lam = 0.5 * np.linalg.norm(np.diag(lam_curr) @ u_new)**2
#         #Run 1
#         psi_lam = [np.linalg.norm(diag_mat(lam_curr, i) @ u_new)**2 for i in range(len(lam_curr))]

#         #Run 2
#         # eta_new_1 = (1/ gamma) * ((phi_new )/psi_1)
#         # eta_new_2 = (1/ gamma) * ((phi_new )/psi_2)

#         #Run 3
#         lam_new =  (np.std(data_noisy - np.dot(G,u_new))) * (phi_new / psi_lam)
#         # lam_new =  (c) * (phi_new / psi_lam)

#         # eta_new_1 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_1)
#         # eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_2)

#         # #Run 4
#         # eta_new_1 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new + curr_eta2 * psi_2)/psi_1)
#         # eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new + curr_eta1 * psi_1)/psi_2)

#         #Run 5
#         # eta_new_1 = (c) * ((phi_new + curr_eta2 * psi_2)/psi_1)
#         # eta_new_2 = (c) * ((phi_new + curr_eta1 * psi_1)/psi_2)

#         # eta_new_1 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_1)
#         # eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_2)
#         #Get new solution
#         reconst = minimize(lam_new)
#         # new_f_rec = np.linalg.inv(G.T @ G + eta_new_1 * B_1.T @ B_1 + eta_new_2 * B_2.T @ B_2) @ (G.T @ data_noisy)
#         new_f_rec = reconst
#         # print("new_f_rec: ", new_f_rec)

#         #Step4: Check stopping criteria based on relative change of regularization parameter eta 
#         #or the  inverse solution 
#         # if np.abs(eta_new_1-curr_eta1)/curr_eta1 < ep and np.abs(eta_new_2-curr_eta2)/curr_eta2 < ep:
#         #     print("(eta_new_1-curr_eta1)/curr_eta1: ", np.abs(eta_new_1-curr_eta1)/curr_eta1)
#         #     print("(eta_new_2-curr_eta2)/curr_eta2: ", np.abs(eta_new_2-curr_eta2)/curr_eta2)
#         #     print("ep value: ", ep)
#         #     break
#         if (np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)) < ep or k >= maxiter:
#             print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)))
#             print("ep value: ", ep)
#             break
#         else:
#             print(f"Finished Iteration {k}")
#             print("curr_lam", lam_curr)
#             print("new_lam", lam_new)
#             # print("(eta_new_1-curr_eta1)/curr_eta1: ",  np.abs(eta_new_1-curr_eta1)/curr_eta1)
#             # print("(eta_new_2-curr_eta2)/curr_eta2: ",np.abs(eta_new_2-curr_eta2)/curr_eta2)
#             curr_lam = lam_new
#             k = k + 1

#     #Return new eta values
#     fin_lam = lam_new
#     # fin_etas = np.array([eta1_fin, eta2_fin])
#     best_f_rec =  minimize(fin_lam)
#     print(f"Total of {k} iterations")
#     return best_f_rec, fin_lam














def LocReg(data_noisy, G, x0_ini, maxiter):
    # x0_LS = lsqnonneg(G, data_noisy)[0]
    # x0_LS_nnls = nnls(G, data_noisy)[0]
    # estimated_noise = lsqnonneg(G, data_noisy)[2] 

    x, _ = nnls(G, data_noisy,maxiter)
    estimated_noise = data_noisy - np.dot(G,x)
    ep = 1e-2
    nT2 = G.shape[1]
    prev_x = x0_ini

    estimated_noise_std = np.std(estimated_noise)
    #track = []
    cur_iter = 1


    while True:
        lambda_val = estimated_noise_std / (np.abs(x0_ini) + ep)
        # lambda_val = np.linalg.norm(estimated_noise)**2 
        LHS = G.T @ G + np.diag(lambda_val)
        RHS = G.T @ data_noisy + (G.T @ G * ep) + ep*lambda_val
        #RHS = G.T @ data_noisy
        try:
            x0_ini = nnls(LHS,RHS, maxiter)[0]
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




def blur_ito(data_noisy, G, lam_ini, gamma, maxiter):

    def minimize(eta):
        x = cp.Variable(G.shape[1])
        # Define the objective function φ(x)
        phi_x = G @ x - data_noisy
        # Define the constraint function ψ(x)
        psi_x = np.linalg.norm(x, 1) + (1e-3/2)*np.linalg.norm(x,2)**2
        # Define the optimization problem
        objective = cp.Minimize(phi_x + eta * psi_x)
        constraints = [x >= 0]  # Adding the constraint Ax <= b
        problem = cp.Problem(objective, constraints)
        problem.solve()
        optimal_x = x.value
        # reconst,_ = nnls(A,b, maxiter = 1000)
        return optimal_x
  
    nT2 = G.shape[1]
    eta_ini = lam_ini 
    eta_curr = eta_ini
    ep = 1e-3
    k = 1
    ini_f_rec = minimize(eta_curr)
    c = np.std(data_noisy - np.dot(G,ini_f_rec))
    while True:
        print(f"Printing Iteration {k}")
        #Step2: Minimization
        curr_f_rec = minimize(eta_curr)
        u_new = curr_f_rec
        #Step 3: Update
        phi_new = 0.5 * np.linalg.norm(data_noisy - np.dot(G,u_new))**2
        # psi_1 = 0.5 * np.linalg.norm(B_1 @ u_new)**2
        # print("psi_1", psi_1)
        # psi_2 = 0.5 * np.linalg.norm(B_2 @ u_new)**2
        # print("psi_2", psi_2)
        psi = np.linalg.norm(u_new, 1) + (1e-3/2)*np.linalg.norm(u_new,2)**2

        #Run 1

        #Run 2
        # eta_new_1 = (1/ gamma) * ((phi_new )/psi_1)
        # eta_new_2 = (1/ gamma) * ((phi_new )/psi_2)
        
        #Run 3
        eta_new = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi)
        # eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_2)

        # #Run 4
        # eta_new_1 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new + curr_eta2 * psi_2)/psi_1)
        # eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new + curr_eta1 * psi_1)/psi_2)

        #Run 5
        # eta_new_1 = (c) * ((phi_new + curr_eta2 * psi_2)/psi_1)
        # eta_new_2 = (c) * ((phi_new + curr_eta1 * psi_1)/psi_2)

        # eta_new_1 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_1)
        # eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_2)
        

        #Get new solution
        reconst = minimize(eta_new)
        # new_f_rec = np.linalg.inv(G.T @ G + eta_new_1 * B_1.T @ B_1 + eta_new_2 * B_2.T @ B_2) @ (G.T @ data_noisy)
        new_f_rec = reconst
        # print("new_f_rec: ", new_f_rec)

        #Step4: Check stopping criteria based on relative change of regularization parameter eta 
        #or the  inverse solution 
        # if np.abs(eta_new_1-curr_eta1)/curr_eta1 < ep and np.abs(eta_new_2-curr_eta2)/curr_eta2 < ep:
        #     print("(eta_new_1-curr_eta1)/curr_eta1: ", np.abs(eta_new_1-curr_eta1)/curr_eta1)
        #     print("(eta_new_2-curr_eta2)/curr_eta2: ", np.abs(eta_new_2-curr_eta2)/curr_eta2)
        #     print("ep value: ", ep)
        #     break
        if (np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)) < ep or k >= maxiter:
            print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)))
            print("ep value: ", ep)
            break
        else:
            curr_eta = eta_new
            k = k + 1

    #Return new eta values
    eta1_fin = eta_new_1
    eta2_fin = eta_new_2
    fin_etas = np.array([eta1_fin, eta2_fin])
    best_f_rec =  minimize(eta1_fin, eta2_fin)
    print(f"Total of {k} iterations")
    return best_f_rec, fin_etas


def blur_ito(data_noisy, G, lam_ini, gamma_init, maxiter):
    import matplotlib.pyplot as plt

    def minimize(eta):
        x = cp.Variable(G.shape[1])
        # Define the objective function φ(x)
        phi_x = cp.sum_squares(G @ x - data_noisy)  # Sum of squares instead of vector
        # Define the constraint function ψ(x)
        psi_x = cp.norm(x, 1) + (1e-3/2) * cp.norm(x, 2)**2
        # Define the optimization problem
        objective = cp.Minimize(phi_x + eta * psi_x)
        constraints = [x >= 0]  # Adding the constraint Ax <= b
        problem = cp.Problem(objective, constraints)
        problem.solve()
        optimal_x = x.value
        return optimal_x

    def diag_mat(lam, i):
        diag = np.zeros_like(lam)
        diag[i] = lam[i]
        return np.diag(diag)
    
    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy)**2
    

    def fixed_point_algo(gamma, lam_vec, check):
        """
        gamma: gamma val
        lam_vec: vector of lambdas
        """
        nT2 = G.shape[1]
        # lam_curr = np.sqrt(lam_vec)
        lam_curr = lam_vec
        k = 1

        ep = 1e-3
        ep_min = ep
        # epscond = False
        # ini_f_rec = minimize(lam_curr, ep_min, epscond)
        f_old = np.ones(G.shape[1])

        c_arr = []
        lam_arr = []
        sol_arr = []

        # fig, axs = plt.subplots(3, 1, figsize=(6, 6))

        # # Show the initial plot
        # plt.tight_layout()
        # plt.ion()  # Turn on interactive mode

        # testiter = maxiter
        # for k in range(testiter):
        while True:
            # Initial L-Curve solution
            try:
                # curr_f_rec = minimize(lam_curr, ep_min, epscond)
                curr_f_rec = minimize(lam_curr)
                if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
                    print(f"curr_f_rec is None after minimization for iteration {k}")
                else:
                    pass
            except Exception as e:
                print("An error occurred during minimization:", e)

            # if (np.linalg.norm(psi_lam/phi_new)) < 1e-4:
            #     print("condition passed")
            #     print("np.linalg.norm(psi_lam/ phi_new)", np.linalg.norm(psi_lam/ phi_new))
            #     psi_lam = psi_lam + ep_min
            # psi_lam = list(psi_lam)

            # Get new solution with new lambda vector

            # Update lambda: then check
            # New Lambda find the new residual and the new penalty
            print("curr_f_rec", curr_f_rec)
            print("G",G)
            phi_new = np.linalg.norm(data_noisy - np.dot(G, curr_f_rec), 2)**2
            psi_lam = np.linalg.norm(curr_f_rec, 1) + (1e-3/2) * np.linalg.norm(curr_f_rec,2)**2

            # psi_lam = [curr_f_rec[i] * lam_curr[i] for i in range(len(lam_curr))]

            # print("gamma:",gamma)
            c = 1 / (gamma)
            # c = np.std(data_noisy - - np.dot(G,curr_f_rec))/gamma
            # c = ((gamma**gamma)/((1+gamma)**(1+gamma)))/gamma

            c_arr.append(c)

            # STEP 4
            # redefine new lam
            lam_new = c * (phi_new / psi_lam)
            # lam_new = [c * (phi_new + (curr_f_rec[j] * lam_curr[j]))/psi_lam[i] for i in range(len(psi_lam)), j != i]
            # arr_2 = [sum([curr_f_rec[j] * lam_curr[j] for j in range(len(curr_f_rec)) if j != i]) for i in range(len(curr_f_rec))]

            # lam_new = [c * (phi_new + (curr_f_rec[j] * lam_curr[j])) / psi_lam[i]
            #            for i in range(len(psi_lam))
            #            for j in range(len(lam_curr))
            #            if j != i]

            # lam_new = [c * (phi_new + arr_2[j]) / psi_lam[i]
            #            for i in range(len(psi_lam))
            #            for j in range(len(arr_2))
            #            if j != i]

            lam_test = (phi_new / psi_lam)

            # psi_lam = np.array(psi_lam)
            # lam_new = np.array(lam_new)
            machine_eps = np.finfo(float).eps

            if np.any(psi_lam/phi_new) < machine_eps:
                print("condition satisfied")

            # test = lam_test
            # psi_lam2 = [test[i] * curr_f_rec[i] for i in range(len(lam_new))]
            # psi_lam2 = np.array(psi_lam2)
            # print("Lam_new.shape", lam_new.shape)
            # cs = c * np.ones(len(psi_lam))
            # print("np.linalg.norm(phi_new/psi_lam):", np.linalg.norm(phi_new / psi_lam))

            # If doesnt converge; update f
            # print("check",check)
            if check == True:
                # axs[0].plot(T2, g, color="black", label="ground truth")
                # axs[0].plot(T2, curr_f_rec, label="reconstruction")
                # axs[1].semilogy(T2, lam_curr, label="lambdas")
                # axs[2].semilogy(T2, lam_new, label="new lambda")
                # axs[3].semilogy(T2, test, label="lambda_new")
                # axs[4].semilogy(T2, np.array(psi_lam2), label="lambda_new * reconstruction")

                # Redraw the plot
                plt.draw()
                # axs[0].legend()
                # axs[1].legend()
                # axs[2].legend()
                # axs[3].legend()
                # axs[4].legend()
                plt.tight_layout()
                plt.pause(0.001)
            else:
                pass

            # Step4: Check stopping criteria based on relative change of regularization parameter eta
            # or the  inverse solution
            # update criteria of lambda
            if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) + (np.linalg.norm(lam_new - lam_curr) / np.linalg.norm(lam_curr)) < ep or k >= maxiter:
                # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old))
                # print("ep value: ", ep)
                c_arr_fin = np.array(c_arr)
                lam_arr_fin = np.array(lam_arr)
                sol_arr_fin = np.array(sol_arr)
                plt.ioff()
                plt.show()
                print(f"Total of {k} iterations")
                return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
            else:
                # ep_min = ep_min / 1.2
                # if ep_min <= 1e-4:
                #     ep_min = 1e-4
                # print(f"Finished Iteration {k}")
                k = k + 1
                lam_curr = lam_new
                f_old = curr_f_rec
                lam_arr.append(lam_new)
                sol_arr.append(curr_f_rec)
    #
    #Step 1:
    check = False

    # lam_vec = lam_ini * np.ones(G.shape[1])
    lam_vec = lam_ini 
    best_f_rec1, fin_lam1, c_arr_fin1, lam_arr_fin, sol_arr_fin = fixed_point_algo(gamma_init, lam_vec, check = False)

    #Step 2: Calculate new phi
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    
    #Step 3: Calculate and update new gamma:
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G,zero_vec, data_noisy)
    gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.5
 
    #Step 4: Perform fixed point algo with new gamma value
    
    best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin  = fixed_point_algo(gamma_new, fin_lam1, check = False)
    print("final lams:", fin_lam2)
    print("final gamma:", gamma_new)
    return best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin





def grav_ito(data_noisy, G, lam_ini, gamma_init, maxiter):
    import matplotlib.pyplot as plt
    def h1_seminorm_psi_x(x):
        # Compute the gradient along the axis
        grad_x = np.gradient(x)
        # Compute the squared gradient
        grad_sq_norm = np.square(grad_x)
        # Compute the square root of the sum of squared gradient
        h1_norm = np.sqrt(np.sum(grad_sq_norm))
        return h1_norm

    def minimize(eta):
        x = cp.Variable(G.shape[1])
        # Define the objective function φ(x)
        phi_x = cp.sum_squares(G @ x - data_noisy)  # Sum of squares instead of vector
        # Define the constraint function ψ(x)
        psi_x = h1_seminorm_psi_x(x)
        # Define the optimization problem
        objective = cp.Minimize(phi_x + eta * psi_x)
        constraints = [x >= 0]  # Adding the constraint Ax <= b
        problem = cp.Problem(objective, constraints)
        problem.solve()
        optimal_x = x.value
        return optimal_x

    def diag_mat(lam, i):
        diag = np.zeros_like(lam)
        diag[i] = lam[i]
        return np.diag(diag)
    
    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy)**2
    

    def fixed_point_algo(gamma, lam_vec, check):
        """
        gamma: gamma val
        lam_vec: vector of lambdas
        """
        nT2 = G.shape[1]
        # lam_curr = np.sqrt(lam_vec)
        lam_curr = lam_vec
        k = 1

        ep = 1e-3
        ep_min = ep
        # epscond = False
        # ini_f_rec = minimize(lam_curr, ep_min, epscond)
        f_old = np.ones(G.shape[1])

        c_arr = []
        lam_arr = []
        sol_arr = []

        # fig, axs = plt.subplots(3, 1, figsize=(6, 6))

        # # Show the initial plot
        # plt.tight_layout()
        # plt.ion()  # Turn on interactive mode

        # testiter = maxiter
        # for k in range(testiter):
        while True:
            # Initial L-Curve solution
            try:
                # curr_f_rec = minimize(lam_curr, ep_min, epscond)
                curr_f_rec = minimize(lam_curr)
                if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
                    print(f"curr_f_rec is None after minimization for iteration {k}")
                else:
                    pass
            except Exception as e:
                print("An error occurred during minimization:", e)

            # if (np.linalg.norm(psi_lam/phi_new)) < 1e-4:
            #     print("condition passed")
            #     print("np.linalg.norm(psi_lam/ phi_new)", np.linalg.norm(psi_lam/ phi_new))
            #     psi_lam = psi_lam + ep_min
            # psi_lam = list(psi_lam)

            # Get new solution with new lambda vector

            # Update lambda: then check
            # New Lambda find the new residual and the new penalty
            # print("curr_f_rec", curr_f_rec)
            # print("G",G)
            phi_new = np.linalg.norm(data_noisy - np.dot(G, curr_f_rec), 2)**2
            # psi_lam = np.linalg.norm(curr_f_rec, 1) + (1e-3/2) * np.linalg.norm(curr_f_rec,2)**2
            psi_lam = h1_seminorm_psi_x(curr_f_rec)
            # print("psi_lam:", psi_lam)
            # psi_lam = [curr_f_rec[i] * lam_curr[i] for i in range(len(lam_curr))]

            # print("gamma:",gamma)
            # c = 1 / (gamma)
            # c = np.std(data_noisy - - np.dot(G,curr_f_rec))/gamma
            c = ((gamma**gamma)/((1+gamma)**(1+gamma)))

            c_arr.append(c)

            # STEP 4
            #  redefine new lam
            lam_new = c * (phi_new / psi_lam)
            # lam_new = [c * (phi_new + (curr_f_rec[j] * lam_curr[j]))/psi_lam[i] for i in range(len(psi_lam)), j != i]
            # arr_2 = [sum([curr_f_rec[j] * lam_curr[j] for j in range(len(curr_f_rec)) if j != i]) for i in range(len(curr_f_rec))]

            # lam_new = [c * (phi_new + (curr_f_rec[j] * lam_curr[j])) / psi_lam[i]
            #            for i in range(len(psi_lam))
            #            for j in range(len(lam_curr))
            #            if j != i]

            # lam_new = [c * (phi_new + arr_2[j]) / psi_lam[i]
            #            for i in range(len(psi_lam))
            #            for j in range(len(arr_2))
            #            if j != i]

            lam_test = (phi_new / psi_lam)

            # psi_lam = np.array(psi_lam)
            # lam_new = np.array(lam_new)
            machine_eps = np.finfo(float).eps

            if np.any(psi_lam/phi_new) < machine_eps:
                print("condition satisfied")

            # test = lam_test
            # psi_lam2 = [test[i] * curr_f_rec[i] for i in range(len(lam_new))]
            # psi_lam2 = np.array(psi_lam2)
            # print("Lam_new.shape", lam_new.shape)
            # cs = c * np.ones(len(psi_lam))
            # print("np.linalg.norm(phi_new/psi_lam):", np.linalg.norm(phi_new / psi_lam))

            # If doesnt converge; update f
            # print("check",check)
            if check == True:
                # axs[0].plot(T2, g, color="black", label="ground truth")
                # axs[0].plot(T2, curr_f_rec, label="reconstruction")
                # axs[1].semilogy(T2, lam_curr, label="lambdas")
                # axs[2].semilogy(T2, lam_new, label="new lambda")
                # axs[3].semilogy(T2, test, label="lambda_new")
                # axs[4].semilogy(T2, np.array(psi_lam2), label="lambda_new * reconstruction")

                # Redraw the plot
                plt.draw()
                # axs[0].legend()
                # axs[1].legend()
                # axs[2].legend()
                # axs[3].legend()
                # axs[4].legend()
                plt.tight_layout()
                plt.pause(0.001)
            else:
                pass

            # Step4: Check stopping criteria based on relative change of regularization parameter eta
            # or the  inverse solution
            # update criteria of lambda
            if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) + (np.linalg.norm(lam_new - lam_curr) / np.linalg.norm(lam_curr)) < ep or k >= maxiter:
                # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old))
                # print("ep value: ", ep)
                c_arr_fin = np.array(c_arr)
                lam_arr_fin = np.array(lam_arr)
                sol_arr_fin = np.array(sol_arr)
                plt.ioff()
                plt.show()
                print(f"Total of {k} iterations")
                return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
            else:
                # ep_min = ep_min / 1.2
                # if ep_min <= 1e-4:
                #     ep_min = 1e-4
                # print(f"Finished Iteration {k}")
                k = k + 1
                lam_curr = lam_new
                f_old = curr_f_rec
                lam_arr.append(lam_new)
                sol_arr.append(curr_f_rec)
    #
    #Step 1:
    check = False

    # lam_vec = lam_ini * np.ones(G.shape[1])
    lam_vec = lam_ini 
    best_f_rec1, fin_lam1, c_arr_fin1, lam_arr_fin, sol_arr_fin = fixed_point_algo(gamma_init, lam_vec, check = False)

    #Step 2: Calculate new phi
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    
    #Step 3: Calculate and update new gamma:
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G,zero_vec, data_noisy)
    gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.25
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.5
 
    #Step 4: Perform fixed point algo with new gamma value
    
    best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin  = fixed_point_algo(gamma_new, fin_lam1, check = False)
    # print("final lams:", fin_lam2)
    print("final gamma:", gamma_new)
    return best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin,sol_arr_fin






def LocReg_Ito_mod2(data_noisy, G, x0_ini, maxiter):
    # x0_LS = lsqnonneg(G, data_noisy)[0]
    # x0_LS_nnls = nnls(G, data_noisy)[0]
    # estimated_noise = lsqnonneg(G, data_noisy)[2] 
    def minimize(lam):
        A = (G.T @ G + np.diag(lam))
        b = G.T @ data_noisy
        y = cp.Variable(nT2)
        cost = cp.norm(A @ y - b, 2)**2
        # constraints = [x >= 0]
        # problem = cp.Problem(cp.Minimize(cost), constraints)
        # constraints = [x >= 0]
        # constraints = [y >= 0]
        # constraints = [np.diag(lam) @ y ] 
        # constraints = [cp.diag(lam) @ y == cp.diag(lam)[0] * np.ones_like(y)]
        constraints = [lam[0] * y[0] == cp.sum([lam[i] * y[i] for i in range(1, len(lam))])]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        # problem = cp.Problem(cp.Minimize(cost))
        # exp,_ = fnnls(A,b)
        problem.solve(solver=cp.MOSEK, verbose=False)
        reconst = y.value
        return reconst

    x, _ = nnls(G, data_noisy,maxiter)
    estimated_noise = data_noisy - np.dot(G,x)
    ep = 1e-2
    nT2 = G.shape[1]
    prev_x = x0_ini

    estimated_noise_std = np.std(estimated_noise)
    #track = []
    cur_iter = 1


    while True:
        lambda_val = estimated_noise_std / (np.abs(x0_ini) + ep)
        LHS = G.T @ G + np.diag(lambda_val)
        RHS = G.T @ data_noisy + (G.T @ G * ep) + ep*lambda_val
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

        # ep = ep / 1.7
        # if ep <= 1e-9:
        #     ep = 1e-9
        cur_iter = cur_iter + 1
        prev_x = x0_ini

    f_rec_logreg = x0_ini
    lambda_locreg = lambda_val

    return f_rec_logreg, lambda_locreg


#Ignore code here:

# lam_new = [c * (phi_new + (curr_f_rec[j] * lam_curr[j]))/psi_lam[i] for i in range(len(psi_lam)), j != i]
# arr_2 = [sum([curr_f_rec[j] * lam_curr[j] for j in range(len(curr_f_rec)) if j != i]) for i in range(len(curr_f_rec))]

# lam_new = [c * (phi_new + (curr_f_rec[j] * lam_curr[j])) / psi_lam[i]
#            for i in range(len(psi_lam))
#            for j in range(len(lam_curr))
#            if j != i]

# lam_new = [c * (phi_new + arr_2[j]) / psi_lam[i]
#            for i in range(len(psi_lam))
#            for j in range(len(arr_2))
#            if j != i]


# def diag_mat(lam, i):
#     diag = np.zeros_like(lam)
#     diag[i] = lam[i]
#     return np.diag(diag)


# print("phi_new", phi_new)
# print("psi_lam", np.linalg.norm(psi_lam, 2))
# print("psi_lam_filt", np.linalg.norm(psi_lam[21:], 2))
# lam_sqr = lam_curr**2
# psi_lam = [lam_sqr[i] * curr_f_rec[i] for i in range(len(lam_curr))]


    # #check new reconstr
    # try:
    #     # reconst = minimize(lam_new, ep_min, epscond)
    #     reconst = minimize(lam_new)
    #     if reconst is None or any(elem is None for elem in reconst):
    #         print(f"new_f_rec is None after minimization for iteration {k}")
    #     else:
    #         pass
    # except Exception as e:
    #     print("An error occurred during minimization:", e)
    
    # new_f_rec = reconst
    # #Return new eta values
    # fin_lam = lam_new
    # # best_f_rec =  minimize(fin_lam, ep_min, epscond)
    # best_f_rec =  minimize(fin_lam)


    # return best_f_rec, fin_lam, c_arr_fin, lam_arr_fin, sol_arr_fin


# if (np.linalg.norm(psi_lam/phi_new)) < 1e-4:
#     print("condition passed")
#     print("np.linalg.norm(psi_lam/ phi_new)", np.linalg.norm(psi_lam/ phi_new))
#     psi_lam = psi_lam + ep_min
# psi_lam = list(psi_lam)


# test = lam_test
# psi_lam2 = [test[i] * curr_f_rec[i] for i in range(len(lam_new))]
# psi_lam2 = np.array(psi_lam2)
# print("Lam_new.shape", lam_new.shape)
# cs = c * np.ones(len(psi_lam))
# print("np.linalg.norm(phi_new/psi_lam):", np.linalg.norm(phi_new / psi_lam))


def test_Ito(data_noisy, G, lam_ini, gamma, param_num, B_mats, maxiter):


    def minimize(eta1, eta2):
        A = (G.T @ G + eta1 * B_1.T @ B_1 + eta2 * B_2.T @ B_2)
        b = G.T @ data_noisy
        reconst,_ = nnls(A,b, maxiter = 1000)
        return reconst
    # x0_LS = lsqnonneg(G, data_noisy)[0]
    # x0_LS_nnls = nnls(G, data_noisy)[0]
    # estimated_noise = lsqnonneg(G, data_noisy)[2] 
    nT2 = G.shape[1]
    eta_ini = lam_ini * np.ones(param_num)
    B_1 = B_mats[0]
    B_2 = B_mats[1]

    print("B_1: ", B_1)
    print("B_2: ", B_2)

    eta_curr = eta_ini
    ep = 1e-3
    k = 1
    curr_eta1 = eta_curr[0]
    print("ini_eta1: ", curr_eta1)
    curr_eta2 = eta_curr[1]
    print("ini_eta2: ", curr_eta2)
    ini_f_rec = minimize(curr_eta1, curr_eta2)
    c = np.std(data_noisy - np.dot(G,ini_f_rec))
    while True:
        print(f"Printing Iteration {k}")
        #Step2: Minimization
        curr_f_rec = minimize(curr_eta1, curr_eta2)
        # x,_ = nnls(A, b, maxiter) 
        # x = x - ep
        # x[x < 0] = 0
        # u_new = x
        # print("curr_f_rec: ", curr_f_rec)
        # curr_f_rec = np.linalg.inv(G.T @ G + curr_eta1 * B_1.T @ B_1 + curr_eta2 * B_2.T @ B_2) @ (G.T @ data_noisy)
        u_new = curr_f_rec
        #Step 3: Update
        phi_new = 0.5 * np.linalg.norm(data_noisy - np.dot(G,u_new))**2
        psi_1 = 0.5 * np.linalg.norm(B_1 @ u_new)**2
        print("psi_1", psi_1)
        psi_2 = 0.5 * np.linalg.norm(B_2 @ u_new)**2
        print("psi_2", psi_2)

        #Run 1

        #Run 2
        # eta_new_1 = (1/ gamma) * ((phi_new )/psi_1)
        # eta_new_2 = (1/ gamma) * ((phi_new )/psi_2)
        
        #Run 3
        eta_new_1 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_1)
        eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_2)

        # #Run 4
        # eta_new_1 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new + curr_eta2 * psi_2)/psi_1)
        # eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new + curr_eta1 * psi_1)/psi_2)

        #Run 5
        # eta_new_1 = (c) * ((phi_new + curr_eta2 * psi_2)/psi_1)
        # eta_new_2 = (c) * ((phi_new + curr_eta1 * psi_1)/psi_2)

        # eta_new_1 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_1)
        # eta_new_2 = (np.std(data_noisy - np.dot(G,u_new))) * ((phi_new)/psi_2)
        

        #Get new solution
        reconst = minimize(eta_new_1, eta_new_2)
        # new_f_rec = np.linalg.inv(G.T @ G + eta_new_1 * B_1.T @ B_1 + eta_new_2 * B_2.T @ B_2) @ (G.T @ data_noisy)
        new_f_rec = reconst
        # print("new_f_rec: ", new_f_rec)

        #Step4: Check stopping criteria based on relative change of regularization parameter eta 
        #or the  inverse solution 
        # if np.abs(eta_new_1-curr_eta1)/curr_eta1 < ep and np.abs(eta_new_2-curr_eta2)/curr_eta2 < ep:
        #     print("(eta_new_1-curr_eta1)/curr_eta1: ", np.abs(eta_new_1-curr_eta1)/curr_eta1)
        #     print("(eta_new_2-curr_eta2)/curr_eta2: ", np.abs(eta_new_2-curr_eta2)/curr_eta2)
        #     print("ep value: ", ep)
        #     break
        if (np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)) < ep or k >= maxiter:
            print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)))
            print("ep value: ", ep)
            break
        else:
            print(f"Finished Iteration {k}")
            print("curr_eta1", curr_eta1)
            print("curr_eta2", curr_eta2)
            print("eta_new_1", eta_new_1)
            print("eta_new_2", eta_new_2)
            # print("(eta_new_1-curr_eta1)/curr_eta1: ",  np.abs(eta_new_1-curr_eta1)/curr_eta1)
            # print("(eta_new_2-curr_eta2)/curr_eta2: ",np.abs(eta_new_2-curr_eta2)/curr_eta2)
            curr_eta1 = eta_new_1
            curr_eta2 = eta_new_2
            k = k + 1

    #Return new eta values
    eta1_fin = eta_new_1
    eta2_fin = eta_new_2
    fin_etas = np.array([eta1_fin, eta2_fin])
    best_f_rec =  minimize(eta1_fin, eta2_fin)
    print(f"Total of {k} iterations")
    return best_f_rec, fin_etas