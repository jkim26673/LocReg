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

mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
def LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter):
    def minimize(lam_vec):
            # Fallback to nonnegtik_hnorm
        # try:
        #     eps = 1e-2
        #     # First attempt: Call nonnegtik_hnorm (in case of initial failure with MOSEK)
        #     sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
        #     sol = np.maximum(sol, 0)
        #     A = G.T @ G + np.diag(lam_vec)
        #     ep4 = np.ones(G.shape[1]) * eps
        #     b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
        # except Exception as e:
        #     print(f"nonnegtickn failed, using MOSEK: {e}")
        #     # This block will now only be used if MOSEK fails entirely
        #     eps = 1e-2
        #     # eps = 0
        #     A = G.T @ G + np.diag(lam_vec)        
        #     ep4 = np.ones(G.shape[1]) * eps
        #     b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)

        #     # Second attempt: Use MOSEK solver
        #     y = cp.Variable(G.shape[1])
        #     cost = cp.norm(A @ y - b, 'fro')**2
        #     constraints = [y >= 0]
        #     problem = cp.Problem(cp.Minimize(cost), constraints)

        #     problem.solve(solver=cp.MOSEK)
            
        #     sol = y.value
        #     sol = np.maximum(sol - eps, 0)

        try:
            eps = 1e-2
            # eps = 0
            A = G.T @ G + np.diag((lam_vec))        
            ep4 = np.ones(G.shape[1]) * eps
            b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
            # sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
            y = cp.Variable(G.shape[1])
            cost = cp.norm(A @ y - b, 'fro')**2
            constraints = [y >= 0]
            problem = cp.Problem(cp.Minimize(cost), constraints)
            # problem.solve(solver=cp.MOSEK, mosek_params={
            #     # 'MSK_IPAR_INTPNT_MAX_ITERATIONS ': '100'
            #     # 'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
            #     # ,"MSK_DPAR_BASIS_REL_TOL_S": '1e-4'  # Turn on Mixed Integer Optimization if needed
            #     # Turn on detailed logging
            #     }, verbose = True)
            problem.solve(solver = cp.MOSEK)
            
            #Change tolerance to 10-3; MSK_IPAR_INTPNT_MAX_ITERATIONS increase to 1000; see if total time changes

            # print("Solver used:", problem.solver_stats.solver_name)
            # print("Solver version:", problem.solver_stats.solver_version)
            sol = y.value
            # if sol is not None:
            #     sol = np.maximum(sol - eps, 0)
            sol = np.maximum(sol - eps, 0)
        # print("sol", sol)
        except Exception as e:
            print(f"MOSEK failed {e}")
            # if sol is None or np.any([x is None for x in sol]):
            #     print("Solution contains None values, switching to nonnegtik_hnorm")
            sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
            sol = np.maximum(sol,0)
        return sol, A, b

    def phi_resid(G, param_vec, data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

    def fixed_point_algo(gamma, lam_vec,tol):
        lam_curr = lam_vec
        ep = 1e-2
        # ep = 1e-2
        ep_min = 1e-2
        f_old = np.ones(G.shape[1])
        # first_f_rec, _, _ = minimize(lam_curr)
        # gammastd = np.std(G @ first_f_rec - data_noisy)
        # val = 5e-2
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
            # while iterationval < 550: testing and 1e-5; sufficient
            # while iterationval < 600: testing and 1e-4 2nd one; no good peak resolution
            # iterations < 700, 1e-5, similar to 550, worse overteim
            #best iterations 200, 1e-5
            #10-14-24 runs iteration val < 200, 1e-2
            #10-15-24 runs iteration val < 180, 1e-5
            #10-17-24 : iteration 180, 1e-3
            while iterationval < 200: 
                curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
                curr_noise = G @ curr_f_rec - data_noisy
                try:
                    delta_p = scipy.linalg.cho_solve((L,True), G.T @ curr_noise)
                except RuntimeWarning:
                    print("Error with delta_p calculation")
                # val /= 1.03
                # val = max(val, 8e-3)
                # print("np.abs(np.linalg.norm(delta_p) / prev)",  (np.abs((np.linalg.norm(delta_p) - prev) / prev)) )
                # if np.abs((np.linalg.norm(delta_p) / prev) - 1) < 1e-5:
                if np.abs((np.linalg.norm(delta_p) / prev) - 1) < 1e-3:
                    # print("reached tol of 1e-2")
                    break
                else:
                    pass
                prev = np.linalg.norm(delta_p)
                iterationval+=1
            # print("finished feedback")
            curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
            # print("iterationval", iterationval)
            phi_new = phi_resid(G, curr_f_rec, data_noisy)
            psi_lam = np.array(curr_f_rec)
            # c = gammastd
            c = 1/gamma
            # c = np.linalg.norm(curr_noise)
            lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))
            if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < tol or k == maxiter:
                # print("Converged")
                if k == maxiter:
                    print("Maximum Iteration Reached")
                return curr_f_rec, lam_curr, k
            else:
                #10-17-24 : iteration 180, 1e-3
                # ep_min = ep_min / 1.7
                ep_min = ep_min / 2
                # if ep_min <= 8e-5:
                #     ep_min = 8e-5
                if ep_min <= 1e-5:
                    ep_min = 1e-5
                # ep_min = ep_min / 1.2
                # if ep_min <= 1e-4:
                #     ep_min = 1e-4
                lam_curr = lam_new
                f_old = curr_f_rec
                k += 1
                # return curr_f_rec, lam_curr, val

    #Main Code
    lam_vec = lam_ini * np.ones(G.shape[1])
    # choice_val = 9e-3
    try:
        best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, tol=1e-2)
        # fin_lam1 = np.sqrt(fin_lam1)
    except Exception as e:
        print("Error in locreg")
        print("lam_vec", lam_vec)
    # fin_lam1 = np.sqrt(fin_lam1)
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G, zero_vec, data_noisy)
    gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

    # new_choice2 = 5e-3
    best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, tol=1e-3)
    # fin_lam2 = np.sqrt(fin_lam2)
    # print("finished second step locreg")
    x_normalized = best_f_rec2
    return x_normalized, fin_lam2, best_f_rec1, fin_lam1, iternum

# filepath = "/home/kimjosy/LocReg_Regularization-1/data/debugfigures/brainLR"
# import matplotlib.pyplot as plt
# plt.figure()
# plt.savefig(f"{filepath}/ "")

# filepath = "/home/kimjosy/LocReg_Regularization-1/data/debugfigures/brainLR"
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(best_f_rec1, label = "FPA step 1")
# plt.savefig(f"{filepath}/FPA1gammainitshifttol")

# plt.figure()
# plt.plot(fin_lam1, label = "FPA lam vec step 1")
# plt.savefig(f"{filepath}/FPA1gammainitshifttol_lam1")

# plt.figure()
# plt.plot(best_f_rec2, label = "FPA step 2")
# plt.savefig(f"{filepath}/FPA2gammainitshifttol")

# plt.figure()
# plt.plot(fin_lam2, label = "FPA2 lambda2")
# plt.savefig(f"{filepath}/FPA2_lam2_gammainitshifttol")