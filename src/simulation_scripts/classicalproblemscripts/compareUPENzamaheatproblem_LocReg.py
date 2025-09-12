import sys
import os
print("Setting system path")
sys.path.append(".")  # Replace this path with the actual path to the parent directory of Utilities_functions
import numpy as np
from scipy.stats import norm as normsci
from scipy.linalg import norm as linalg_norm
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import pickle
from Utilities_functions.discrep_L2 import discrep_L2
from Utilities_functions.GCV_NNLS import GCV_NNLS
from Utilities_functions.Lcurve import Lcurve
import pandas as pd
import cvxpy as cp
from scipy.linalg import svd
from regu.csvd import csvd
from regu.discrep import discrep
from Simulations.LRalgo import LocReg_Ito_mod, LocReg_Ito_mod_deriv, LocReg_Ito_mod_deriv2, LocReg_Ito_mod_feedmod, LocReg_Ito_mod_deriv_feedmod, LocReg_Ito_mod_deriv2_feedbackmod, LocReg_Ito_mod_deriv2_mimic
from Utilities_functions.pasha_gcv import Tikhonov
from regu.l_curve import l_curve
from tqdm import tqdm
from Utilities_functions.tikhonov_vec import tikhonov_vec
import mosek
import seaborn as sns
from regu.nonnegtik_hnorm import nonnegtik_hnorm
import multiprocess as mp
from multiprocessing import Pool, freeze_support
from multiprocessing import set_start_method
import functools
from datetime import date
import random
import cProfile
import pstats
from Simulations.resolutionanalysis import find_min_between_peaks, check_resolution
import logging
import time
from scipy.stats import wasserstein_distance
import matplotlib.ticker as ticker  # Add this import
from Simulations.upencode import upen_param_setup, upen_setup
from Simulations.upenzama import UPEN_Zama
from regu.csvd import csvd
from regu.heat import heat
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\ZamaUPEN\1D_test', nargout=0)
err_type = "WassScore"
file_path_final = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\classicalproblems\upenzamacomparison"
def minimize_OP(Alpha_vec, data_noisy, G, nT2, g):
    OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
    OP_rhos = np.zeros((len(Alpha_vec)))
    for j in (range(len(Alpha_vec))):
        try:
            # Fallback to nonnegtik_hnorm
            sol, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j], '0', nargin=4)
            if np.all(sol == 0):
                print(f"Solution from nonnegtik_hnorm is a zero vector, switching to CVXPY solver.")
                raise ValueError("Zero vector detected, switching to CVXPY.")
        except Exception as e:
            print(f"Error in nonnegtik_hnorm: {e}")

            # If both methods fail or if the solution was a zero vector, solve using cvxpy
            lam_vec = Alpha_vec[j] * np.ones(G.shape[1])
            A = (G.T @ G + np.diag(lam_vec))
            eps = 1e-2
            ep4 = np.ones(A.shape[1]) * eps
            b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vec
            y = cp.Variable(G.shape[1])
            cost = cp.norm(A @ y - b, 2)**2
            constraints = [y >= 0]
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.MOSEK, verbose=False)
            sol = y.value
            sol = sol - eps
            sol = np.maximum(sol, 0)
        # except Exception as e:
        #     print(f"Error in nonnegtik_hnorm: {e}")
        #     # If both methods fail, solve using cvxpy
        #     lam_vec = Alpha_vec[j] * np.ones(G.shape[1])
        #     A = (G.T @ G + np.diag(lam_vec))
        #     eps = 1e-2
        #     ep4 = np.ones(A.shape[1]) * eps
        #     b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vec
        #     y = cp.Variable(G.shape[1])
        #     cost = cp.norm(A @ y - b, 2)**2
        #     constraints = [y >= 0]
        #     problem = cp.Problem(cp.Minimize(cost), constraints)
        #     problem.solve(solver=cp.MOSEK, verbose=False)
        #     sol = y.value
        #     sol = sol - eps
        #     sol = np.maximum(sol, 0)
        #     OP_x_lc_vec[:, j] = sol
        #     # OP_rhos[j] = np.linalg.norm(OP_x_lc_vec[:,j] - g, 2)**2
        #     OP_rhos[j] = np.linalg.norm(OP_x_lc_vec[:,j] - g, 2)**2
        #     print("OP_x_lc_vec[:,j]", OP_x_lc_vec[:,j])
        #     print("g", g)
        OP_x_lc_vec[:, j] = sol
        # Calculate the error (rho)
        if err_type == "WassScore":
                    # err_LC = wass_error(IdealModel_weighted, f_rec_LC)
            OP_rhos[j] = wass_error(g, OP_x_lc_vec[:, j])
        else:
            pass
        # OP_rhos[j] = np.linalg.norm(OP_x_lc_vec[:, j] - g, 2)
        # print("OP_rhos[j]", OP_rhos[j])
    # OP_log_err_norm = np.log10(OP_rhos)
    # print("OP_log_err_norm", OP_log_err_norm)
    # print("np.argmin(OP_log_err_norm)", np.argmin(OP_log_err_norm))
    # min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)
    # print("min OP_rhos",min(OP_rhos))
    min_rhos = min(OP_rhos)
    min_index = np.argmin(OP_rhos)
    # print("min_index", min_index)
    # min_x = Alpha_vec[min_index[0]]
    min_x = Alpha_vec[min_index]
    # print("min_lambda", min_x)
    OP_min_alpha1 = min_x
    # OP_min_alpha1_ind = min_index[0]
    OP_min_alpha1_ind = min_index
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    return f_rec_OP_grid, OP_min_alpha1, min_rhos , min_index


def wass_error(IdealModel,reconstr):
    true_norm = linalg_norm(IdealModel)
    #check the absolute errors pattern vs SNRs.
    # err = wasserstein_distance(IdealModel,reconstr)/true_norm
    err = wasserstein_distance(IdealModel,reconstr)
    return err

from datetime import datetime
today_date = datetime.now().date().strftime("%Y-%m-%d").replace("-", "_")
kernel = "heatZamaT1"
nametag = f"{today_date}_{kernel}_"

#Gaussian blur kernel
iter_sim = 0
noise_level = 0.1
result = eng.T1(matlab.double(100),matlab.double(noise_level), nargout=5)
A = np.array(result[0])
dat_noisy = np.array(result[1]).flatten()
xex = np.array(result[2]).flatten()
noise_norm = np.array(result[3]).flatten()
# noise_norm
b_exact = np.array(result[4]).flatten()
# print("type(np.array(b_exact))", type(np.array(b_exact)))
Kmax = 500
beta_0 = 1e-7
tol_lam=1e-5
# beta_0 = 1e-3
# tol_lam=1.e-5  
        # Lambda=logspace(-3,0,200); 
Lambda=np.logspace(-5,1,200)
SNR = 100

def runs(numNRs):
    df_final = pd.DataFrame()
    for iter_sim in tqdm(range(numNRs)):
        SD_noise = np.max(np.abs(b_exact))/SNR
        noise = np.random.normal(0,SD_noise, len(b_exact))
        noise_norm = np.linalg.norm(noise)
        dat_noisy = b_exact + noise
        f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
        f_rec_GCV = f_rec_GCV[:, 0]
        lambda_GCV = np.squeeze(lambda_GCV)
        print("GCV done")

        f_rec_oracle, lambda_oracle, min_rhos , min_index = minimize_OP(Lambda, dat_noisy, A, len(xex), xex)
        print("Oracle done")

        LRIto_ini_lam = lambda_GCV
        f_rec_ini = f_rec_GCV

        maxiter = 500
        gamma_init = 0.5
        # maxiter = 50
        # LRIto_ini_lam = lambda_GCV
        # f_rec_ini = f_rec_GCV
        # f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
        # print("LocReg done")

        # LRIto_ini_lam = lambda_GCV
        # f_rec_ini = f_rec_GCV
        # f_rec_LocReg_LC1, lambda_locreg_LC1, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
        # print("LocReg 1st Deriv done")

        LRIto_ini_lam = lambda_GCV
        f_rec_ini = f_rec_GCV
        f_rec_LocReg_LC2, lambda_locreg_LC2, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv2(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
        print("LocReg 2nd Deriv done")

        # LRIto_ini_lam = lambda_GCV
        # f_rec_ini = f_rec_GCV
        # f_rec_LocReg_nofeed, lambda_locreg_nofeed, test_frec1_nofeed, test_lam1_nofeed, numiterate_nofeed = LocReg_Ito_mod_feedmod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, feedbackmod=False)
        # print("LocReg Nofeed done")

        # LRIto_ini_lam = lambda_GCV
        # f_rec_ini = f_rec_GCV
        # f_rec_LocRegD1_nofeed, lambda_locregD1_nofeed, test_frec1_nofeed, test_lam1_nofeed, numiterate_nofeed = LocReg_Ito_mod_deriv_feedmod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, feedbackmod=False)
        # print("LocReg 1st Deriv Nofeed done")

        LRIto_ini_lam = lambda_GCV
        f_rec_ini = f_rec_GCV
        f_rec_LocRegD2_nofeed, lambda_locregD2_nofeed, test_frec1_nofeed, test_lam1_nofeed, numiterate_nofeed = LocReg_Ito_mod_deriv2_feedbackmod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, feedbackmod=False)
        print("LocReg 2nd Deriv Nofeed done")

        # LRIto_ini_lam = lambda_GCV
        # f_rec_ini = f_rec_GCV
        # f_rec_LocReg_mimic, lambda_locreg_mimic = LocReg_Ito_mod_deriv2_mimic(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, feedbackmod=False)
        # print("LocReg 2nd Deriv Nofeed Mimic done")
        # result = upen_param_setup(TE, T2, A, dat_noisy)
        # upen_sol, _ ,_ , upen_lams= upen_setup(result, dat_noisy, LRIto_ini_lam, True)
        # noise_norm = np.linalg.norm(noise)
        try:
            # Call the function and get the desired outputs
            upen_zama_sol, upen_zama_lams = UPEN_Zama(A, dat_noisy, xex, noise_norm, beta_0, Kmax, tol_lam)
            # print("upen_zama_sol", upen_zama_sol)
            # print("upen_zama_lams", upen_zama_lams)
        except Exception as e:
            # Catch any exceptions and print an error message
            print("An error occurred while running the UPEN_Zama function:")
            print(e)

        #normalization
        # sum_x = np.sum(f_rec_LocReg_LC) * dT
        # f_rec_LocReg_LC = f_rec_LocReg_LC / sum_x
        # sum_oracle = np.sum(f_rec_oracle) * dT
        # f_rec_oracle = f_rec_oracle / sum_oracle
        # sum_GCV = np.sum(f_rec_GCV) * dT
        # f_rec_GCV = f_rec_GCV / sum_GCV
        # sum_LC = np.sum(f_rec_LC) * dT
        # f_rec_LC = f_rec_LC / sum_LC
        # sum_DP = np.sum(f_rec_DP) * dT
        # f_rec_DP = f_rec_DP / sum_DP
        # T2 = xex
        # sum_x1 = np.trapz(f_rec_LocReg_LC1, T2)
        # f_rec_LocReg_LC1 = f_rec_LocReg_LC1 / sum_x1

        # sum_x2 = np.trapz(f_rec_LocReg_LC2, T2)
        # f_rec_LocReg_LC2 = f_rec_LocReg_LC2 / sum_x2

        # sum_x = np.trapz(f_rec_LocReg_LC, T2)
        # f_rec_LocReg_LC = f_rec_LocReg_LC / sum_x

        # sum_oracle = np.trapz(f_rec_oracle, T2)
        # f_rec_oracle = f_rec_oracle / sum_oracle
        # sum_GCV = np.trapz(f_rec_GCV, T2)
        # f_rec_GCV = f_rec_GCV / sum_GCV
        # sum_LC = np.trapz(f_rec_LC, T2)
        # f_rec_LC = f_rec_LC / sum_LC
        # sum_DP = np.trapz(f_rec_DP, T2)
        # f_rec_DP = f_rec_DP / sum_DP
        # upen_sum = np.trapz(upen_zama_sol, T2)
        # upen_zama_sol = upen_zama_sol / upen_sum

        # if np.isclose(np.sum(f_rec_LocReg_LC) * dT, 1.0):
        #     pass
        # else:
        #     print("(np.sum(f_rec_LocReg_LC) * dT", (np.sum(f_rec_LocReg_LC) * dT))
        #     print("f_rec_LocReg_LC is not normalized.")

        # if np.isclose(np.sum(f_rec_oracle) * dT, 1.0):
        #     pass
        # else:
        #     print("(np.sum(f_rec_oracle) * dT", (np.sum(f_rec_oracle) * dT))
        #     print("f_rec_oracle is not normalized.")

        # if np.isclose(np.sum(f_rec_GCV) * dT, 1.0):
        #     pass
        # else:
        #     print("(np.sum(f_rec_GCV) * dT", (np.sum(f_rec_GCV) * dT))
        #     print("f_rec_GCV is not normalized.")

        # if np.isclose(np.sum(f_rec_LC) * dT, 1.0):
        #     pass
        # else:
        #     print("(np.sum(f_rec_LC) * dT", (np.sum(f_rec_LC) * dT))
        #     print("f_rec_LC is not normalized.")

        # if np.isclose(np.sum(f_rec_DP) * dT, 1.0):
        #     pass
        # else:
        #     print("(np.sum(f_rec_DP) * dT", (np.sum(f_rec_DP) * dT))
        #     print("f_rec_DP is not normalized.")
        # Flatten results
        f_rec_GCV = f_rec_GCV.flatten()
        # f_rec_LocReg_LC = f_rec_LocReg_LC.flatten()
        # f_rec_LocReg_LC1 = f_rec_LocReg_LC1.flatten()
        f_rec_LocReg_LC2 = f_rec_LocReg_LC2.flatten()
        # f_rec_LocReg_D1_nofeed = f_rec_LocRegD1_nofeed.flatten()
        f_rec_LocReg_D2_nofeed = f_rec_LocRegD2_nofeed.flatten()
        # f_rec_LocReg_nofeed = f_rec_LocReg_nofeed.flatten()
        f_rec_oracle = f_rec_oracle.flatten()
        upen_zama_sol = upen_zama_sol.flatten()
        # f_rec_LocReg_mimic = f_rec_LocReg_mimic.flatten()

        # Calculate Relative L2 Error
        IdealModel_weighted = xex
        if err_type == "WassScore":
            err_GCV = wass_error(IdealModel_weighted, f_rec_GCV)
            err_oracle = wass_error( IdealModel_weighted, f_rec_oracle)
            # err_LR = wass_error(IdealModel_weighted, f_rec_LocReg_LC)
            # err_LR1 = wass_error(IdealModel_weighted, f_rec_LocReg_LC1)
            err_LR2 = wass_error(IdealModel_weighted, f_rec_LocReg_LC2)
            err_upen = wass_error(IdealModel_weighted, upen_zama_sol)
            # err_LR_nofeed = wass_error(IdealModel_weighted, f_rec_LocReg_nofeed)
            # err_LR1_nofeed = wass_error(IdealModel_weighted, f_rec_LocReg_D1_nofeed)
            err_LR2_nofeed = wass_error(IdealModel_weighted, f_rec_LocReg_D2_nofeed)
            # err_LR2_mimic = wass_error(IdealModel_weighted, f_rec_LocReg_mimic)
        realresult = 1
        N = len(xex)
        T2 = np.arange(1, N+1)
        TE = np.arange(1, len(A @ b_exact) + 1)
        lam_ini_val = LRIto_ini_lam

        # Assuming you are inside a loop or block where you want to check these conditions
        if not (err_oracle <= err_GCV):
            print("err oracle", err_oracle)
            print("oracle", f_rec_oracle)
            print("err_GCV", err_GCV)
            print("lambda_oracle", lambda_oracle)
            print("min_rhos", min_rhos)
            print("min_index", min_index)
            realresult = 0
            print("Oracle Error should not be larger than other single parameter methods")

        def plot(iter_sim):
            plt.figure(figsize=(17, 8))
            # Plotting the first subplot
            # plt.subplot(1, 3, 1) 
            # plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
            # plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} (Error: {"{:.2e}".format(err_LR)})')
            # plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle (Error: {"{:.2e}".format(err_oracle)})')
            # plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {"{:.2e}".format(err_DP)})')
            # plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {"{:.2e}".format(err_GCV)})')
            # plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Error: {"{:.2e}".format(err_LC)})')
            # errors = {'LocReg': err_LR, 'Oracle': err_oracle, 'GCV': err_GCV, "UPEN_Zama": err_upen,
            #             'LocReg_1st_Der': err_LR1, 'LocReg_2nd_Der': err_LR2,'LocReg_NF': err_LR_nofeed, 'LocReg_1st_Der_NF': err_LR1_nofeed, 'LocReg_2nd_Der_NF': err_LR2_nofeed}
            
            errors = {'Oracle': err_oracle, 'GCV': err_GCV, "UPEN_Zama": err_upen,
                    'LocReg_2nd_Der': err_LR2, 'LocReg_2nd_Der_NF': err_LR2_nofeed}

            min_method = min(errors, key=errors.get)
            # Modify the plot labels to include a star next to the method with the lowest error
            plt.subplot(1, 3, 1)
            plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
            # locreg_label = f'LocReg GCV (Error: {"{:.2e}".format(err_LR)})'
            # if min_method == 'LocReg':
            #     locreg_label += ' *'
            # plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=locreg_label)

            # locreg_label1 = f'LocReg 1st Deriv GCV (Error: {"{:.2e}".format(err_LR1)})'
            # if min_method == 'LocReg_1st_Der':
            #     locreg_label1 += ' *'
            # plt.plot(T2, f_rec_LocReg_LC1, linestyle=':', linewidth=3, color='green', label=locreg_label1)

            locreg2_label = f'LocReg 2nd Deriv GCV (Error: {"{:.2e}".format(err_LR2)})'
            if min_method == 'LocReg_2nd_Der':
                locreg2_label += ' *'
            plt.plot(T2, f_rec_LocReg_LC2, linestyle=':', linewidth=3, color='cyan', label=locreg2_label)

            oracle_label = f'Oracle (Error: {"{:.2e}".format(err_oracle)})'
            if min_method == 'Oracle':
                oracle_label += ' *'
            plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=oracle_label)
            # # dp_label = f'DP (Error: {"{:.2e}".format(err_DP)})'
            # # if min_method == 'DP':
            # #     dp_label += ' *'
            # # plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=dp_label)
            gcv_label = f'GCV (Error: {"{:.2e}".format(err_GCV)})'
            if min_method == 'GCV':
                gcv_label += ' *'
            plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='orange', label=gcv_label)
            # plt.legend(fontsize=10, loc='best')

            # locregNF_label = f'LocReg No Feedback (Error: {"{:.2e}".format(err_LR_nofeed)})'
            # if min_method == 'LocReg_NF':
            #     locregNF_label += ' *'
            # plt.plot(T2, f_rec_LocReg_nofeed, linestyle=':', linewidth=3, color='yellow', label=locregNF_label)

            # locregNF_label1 = f'LocReg 1st Deriv No Feedback (Error: {"{:.2e}".format(err_LR1_nofeed)})'
            # if min_method == 'LocReg_1st_Der_NF':
            #     locregNF_label1 += ' *'
            # plt.plot(T2, f_rec_LocReg_D1_nofeed, linestyle=':', linewidth=3, color='pink', label=locregNF_label1)

            locreg2NF_label = f'LocReg 2nd Deriv No Feedback (Error: {"{:.2e}".format(err_LR2_nofeed)})'
            if min_method == 'LocReg_2nd_Der_NF':
                locreg2NF_label += ' *'
            plt.plot(T2, f_rec_LocReg_D2_nofeed, linestyle=':', linewidth=3, color='blue', label=locreg2NF_label)

            # lc_label = f'LCurve (Error: {"{:.2e}".format(err_LC)})'
            # if min_method == 'LC':
            #     lc_label += ' *'
            # plt.plot(T2, f_rec_LC, linestyle='--', linewidth=3, color='purple', label=lc_label)
            # upen_label = f'UPEN (Error: {"{:.2e}".format(err_upen)})'
            # if min_method == 'UPEN':
            #     upen_label += ' *'
            # plt.plot(T2, upen_zama_sol, linestyle='--', linewidth=3, color='purple', label=upen_label)

            upen_zama_label = f'UPEN_Zama (Error: {"{:.2e}".format(err_upen)})'
            if min_method == 'UPEN_Zama':
                upen_zama_label += ' *'
            plt.plot(T2, upen_zama_sol, linestyle='--', linewidth=3, color='purple', label=upen_zama_label)

            plt.legend(fontsize=10, loc='best')
            plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
            plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
            ymax = np.max(IdealModel_weighted) * 1.15
            plt.ylim(0, ymax)

            # Plotting the second subplot
            plt.subplot(1, 3, 2)
            plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
            # plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label= f'LocReg GCV')
            # plt.plot(TE, A @ f_rec_LocReg_LC1, linestyle=':', linewidth=3, color='green', label= f'LocReg 1stDer GCV')
            plt.plot(TE, A @ f_rec_LocReg_LC2, linestyle=':', linewidth=3, color='cyan', label= f'LocReg 2ndDer GCV')
            plt.plot(TE, A @ f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle')
            # plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
            plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='orange', label='GCV')
            # plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
            # plt.plot(TE, A @ upen_zama_sol, linestyle='-.', linewidth=3, color='purple', label='UPEN')
            plt.plot(TE, A @ upen_zama_sol, linestyle='-.', linewidth=3, color='purple', label='UPEN Zama')
            # plt.plot(TE, A @ f_rec_LocReg_nofeed, linestyle='-.', linewidth=3, color='yellow', label='LocReg_NF')
            # plt.plot(TE, A @ f_rec_LocReg_D1_nofeed, linestyle='-.', linewidth=3, color='pink', label='LocReg_1st_Der_NF')
            plt.plot(TE, A @ f_rec_LocReg_D2_nofeed, linestyle='-.', linewidth=3, color='blue', label='LocReg_2nd_Der_NF')

            plt.legend(fontsize=10, loc='best')
            plt.xlabel('TE', fontsize=20, fontweight='bold')
            plt.ylabel('Intensity', fontsize=20, fontweight='bold')
            
            plt.subplot(1, 3, 3)
            # plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
            plt.semilogy(T2, (lambda_GCV) * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
            # plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
            # plt.semilogy(T2, (lambda_locreg_LC) * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg GCV')
            # plt.semilogy(T2, (lambda_locreg_LC1) * np.ones(len(T2)), linestyle=':', linewidth=3, color='green', label=f'LocReg 1stDeriv GCV')
            plt.semilogy(T2, (lambda_locreg_LC2) * np.ones(len(T2)), linestyle=':', linewidth=3, color='cyan', label=f'LocReg 2ndDeriv GCV')
            plt.semilogy(T2, (lambda_oracle) * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')
            # plt.semilogy(T2, upen_zama_lams * np.ones(len(T2)), linestyle='-.', linewidth=3, color='purple', label='UPEN')
            plt.semilogy(T2, (upen_zama_lams) * np.ones(len(T2)), linestyle='-.', linewidth=3, color='purple', label='UPEN Zama')
            # plt.semilogy(T2, (lambda_locreg_nofeed) * np.ones(len(T2)), linestyle='-.', linewidth=3, color='yellow', label='LocReg_NF')
            # plt.semilogy(T2, (lambda_locregD1_nofeed) * np.ones(len(T2)), linestyle='-.', linewidth=3, color='pink', label='LocReg_1st_Der_NF')
            plt.semilogy(T2, (lambda_locregD2_nofeed) * np.ones(len(T2)), linestyle='-.', linewidth=3, color='blue', label='LocReg_2nd_Der_NF')

            # plt.plot(T2, (lambda_GCV) * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
            # plt.plot(T2, (lambda_locreg_LC) * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')
            # plt.plot(T2, (lambda_oracle) * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')
            # plt.plot(T2, upen_lams * np.ones(len(T2)), linestyle='-.', linewidth=3, color='purple', label='UPEN')
            plt.legend(fontsize=10, loc='best')
            plt.xlabel('T2', fontsize=20, fontweight='bold')
            plt.ylabel('Lambda', fontsize=20, fontweight='bold')
            plt.tight_layout()
            # plt.ylim(bottom=0, top=2)
            plt.savefig(os.path.join(file_path_final, f"{nametag}_Simulation{iter_sim}.png"))
            plt.close() 

        # Plot a set of 25 reconstructions for simulation 0
        if iter_sim < 5:
            plot(iter_sim)
            print(f"Finished Plots for iteration {iter_sim}")
        else:
            pass
        # Create a new row as a DataFrame with the necessary values
        new_row = pd.DataFrame({"n_sim": iter_sim,
            "f_rec_LocReg_D2_nofeed": [f_rec_LocReg_D2_nofeed],
            "f_rec_GCV": [f_rec_GCV],
            "f_rec_oracle": [f_rec_oracle],
            "f_rec_upen_zama": [upen_zama_sol],
            "f_rec_LocReg_D2_feed": [f_rec_LocReg_LC2],
            "LR_D2_lam": [lambda_locreg_LC2],
            "GCV_lam": [lambda_GCV],
            "oracle_lam": [lambda_oracle],
            "upen_lam": [upen_zama_lams],
            "LR_D2_nofeed_lam": [lambda_locregD2_nofeed],
            "err_LR2D": [err_LR2],
            "err_GCV": [err_GCV],
            "err_oracle": [err_oracle],
            "err_upen": [err_upen],
            "err_LR2D_nofeed": [err_LR2_nofeed]
        })
        df_final = pd.concat([df_final, new_row], ignore_index=True)
    return df_final

nsim = 50
df_final = runs(nsim)

df_final

import os
import pandas as pd
from pathlib import Path
import datetime

save_dir = Path(r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\classicalproblems\upenzamacomparison")
save_dir.mkdir(parents=True, exist_ok=True)

# Generate a unique filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"upenzamacomparison_{timestamp}.pkl"

# Full file path
file_path = save_dir / file_name

# Save the DataFrame with compression (gzip in this case)
df_final.to_pickle(file_path)
import pickle
import pandas as pd

# Path to your pickle file
# pickle_file_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\classicalproblems\upenzamacomparison\upenzamacomparison_20250326_160516.pkl"


# Check the first few rows of the DataFrame to understand its structure (optional)
df = df_final
# Compute the average values of the specified error columns
avg_err_LR2 = df['err_LR2D'].mean() if 'err_LR2D' in df.columns else None
avg_err_GCV = df['err_GCV'].mean() if 'err_GCV' in df.columns else None
avg_err_oracle = df['err_oracle'].mean() if 'err_oracle' in df.columns else None
avg_err_upen = df['err_upen'].mean() if 'err_upen' in df.columns else None
avg_err_LR2_nofeed = df['err_LR2D_nofeed'].mean() if 'err_LR2D_nofeed' in df.columns else None

# Print the averages
print(f"Average err_LR2D: {avg_err_LR2}")
print(f"Average err_GCV: {avg_err_GCV}")
print(f"Average err_oracle: {avg_err_oracle}")
print(f"Average err_upen: {avg_err_upen}")
print(f"Average err_LR2D_nofeed: {avg_err_LR2_nofeed}")
print("length of df", len(df))