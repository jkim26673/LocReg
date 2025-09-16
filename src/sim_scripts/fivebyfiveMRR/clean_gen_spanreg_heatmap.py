# import sys
# import os
# import random
# import time
# import logging
# from datetime import date
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import cProfile
# import pstats
# from scipy.stats import norm as normsci, wasserstein_distance
# from scipy.linalg import norm as linalg_norm, svd
# from scipy.optimize import nnls
# import matplotlib.ticker as ticker
# import cvxpy as cp
# import mosek
# import multiprocess as mp
# from multiprocessing import Pool, freeze_support, set_start_method
# from tqdm import tqdm
# import functools
# sys.path.append(".")  # Make sure this points to the parent directory of Utilities_functions
# from Utilities_functions.discrep_L2 import discrep_L2
# from Utilities_functions.GCV_NNLS import GCV_NNLS
# from Utilities_functions.Lcurve import Lcurve
# from Utilities_functions.pasha_gcv import Tikhonov
# from Utilities_functions.tikhonov_vec import tikhonov_vec
# from regu.csvd import csvd
# from regu.discrep import discrep
# from regu.l_curve import l_curve
# from regu.nonnegtik_hnorm import nonnegtik_hnorm
# from Simulations.LRalgo import LocReg_Ito_mod, LocReg_Ito_mod_deriv, LocReg_Ito_mod_deriv2
# from Simulations.resolutionanalysis import find_min_between_peaks, check_resolution
# from Simulations.upencode import upen_param_setup, upen_setup
# from Simulations.upenzama import UPEN_Zama, UPEN_Zama0th, UPEN_Zama1st
# # Configure logging
# logging.basicConfig(filename='my_script.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# print("setting license path")
# mosek_license_path = r"C:\Users\kimjosy\Downloads\mosek\mosek.lic"
# os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# # Adjusting the file paths for compatibility with Windows
# # Get the parent directory
# parent = os.path.dirname(os.path.abspath(''))
# sys.path.append(parent)
# # Get the current working directory
# cwd = os.getcwd()
# # Adjust the base file and create the correct path
# base_file = 'LocReg_Regularization-1'
# cwd_temp = os.getcwd()
# cwd_cut = os.path.join(cwd_temp.split(base_file, 1)[0], base_file)
# # Define simulation save folder
# pat_tag = "MRR"
# series_tag = "SpanRegFig"
# simulation_save_folder = os.path.join("SimulationSets", pat_tag, series_tag)
# # Full path for saving simulation data
# cwd_full = os.path.join(cwd_cut, simulation_save_folder)
# # Number of simulations and SNR
# n_sim = 10
# SNR_value = 1000
# # Hyperparameters and Global Parameters
# npeaks = 2
# nsigma = 5
# f_coef = np.ones(npeaks)
# rps = np.linspace(1.1, 4, nsigma).T
# nrps = len(rps)
# # Show Plots
# show = 1
# # Parallel processing
# parallel = False
# # Error metric
# err_type = "WassScore"
# # Resolution peak analysis
# peak_test_true = True
# if peak_test_true:
#     peak_test = []
# # Shifting distribution by a constant beta value
# testing = False
# shift_beta = False
# if shift_beta:
#     beta_list = np.linspace(-100, 100, 1000)
# Kmax = 500
# beta_0 = 1e-7
# tol_lam = 1e-5
# # LocReg hyperparameters
# eps1 = 1e-2
# ep_min = 1e-2
# eps_cut = 1.2
# eps_floor = 1e-4
# exp = 0.5
# feedback = True
# lam_ini_val = "GCV"
# # Distribution type for the simulation
# dist_type = f"narrowL_broadR_parallel_nsim{n_sim}_SNR_{SNR_value}_errtype_{err_type}_compare1st2ndDeriv_UPEN"
# # dist_type = f"broadL_narrowR_parallel_nsim{n_sim}_SNR_{SNR_value}_errtype_{err_type}_compare1st2ndDeriv_UPEN"
# # Initial gamma value
# gamma_init = 0.5
# # Load Data File (adjusted for Windows path)
# file_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\Simulations\num_of_basis_functions\lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_6_basis2_40110lmbda_min-6lmbda_max008Oct24.pkl"
# # Load the Gaussian data
# Gaus_info = np.load(file_path, allow_pickle=True)
# TEtest = Gaus_info["TE"]
# T2test = Gaus_info["T2"]
# A = Gaus_info["A"]
# n, m = Gaus_info['A'].shape
# # Define the regularization parameter range
# # Lambda=np.logspace(-5,1,200)
# reg_param_lb = -5
# reg_param_ub = 0
# N_reg = 16
# Lambda = np.logspace(reg_param_lb, reg_param_ub, N_reg).reshape(-1, 1)
# # Date information for the file name
# date = date.today()
# day = date.strftime('%d')
# month = date.strftime('%B')[0:3]
# year = date.strftime('%y')
# # Data folder path and creation (adjusted for Windows path)
# data_path = os.path.join("SimulationsSets", "MRR", "SpanRegFig")
# add_tag = ""
# data_head = "est_table"
# data_tag = (f"{data_head}_SNR{SNR_value}_iter{n_sim}_lamini_{lam_ini_val}_dist_{dist_type}_{add_tag}{day}{month}{year}")
# # Combine the full path for the data folder
# data_folder = os.path.join(os.getcwd(), data_path)
# os.makedirs(data_folder, exist_ok=True)
# # Number of tasks to execute
# target_iterator = [(a, b, c) for a in range(n_sim) for b in range(nsigma) for c in range(nrps)]
# print("target_iterator", len(target_iterator))
# # Get the number of CPUs available
# num_cpus_avail = os.cpu_count()
# ####Predefine noise path
# preset_noise = False
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-22_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Wass. Score/est_table_SNR1000_iter10_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Wass. Score_22Oct24noise_arr.npy"
# noisy_data = np.zeros((n_sim, nsigma, nrps, n))
# noiseless_data = np.zeros((n_sim, nsigma, nrps, n))
# std_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-22_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Wass. Score/est_table_SNR1000_iter10_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Wass. Score_22Oct24stdnoise_data.npy"
# stdnoise_data = np.zeros((n_sim, nsigma, nrps, n))
# if preset_noise == True:
#     noise_arr = np.load(noise_file_path, allow_pickle=True)
#     stdnoise_data = np.load(std_file_path, allow_pickle=True)
# else:
#     noise_arr = np.zeros((n_sim, nsigma, nrps, n))
#     stdnoise_data = np.zeros((n_sim, nsigma, nrps, n))

# def create_result_folder(string, SNR, lam_ini_val, dist_type):
#     # Get current date in the format 'dd_Mmm_yy'
#     # Create the folder name using os.path.join to ensure cross-platform compatibility
#     folder_name = os.path.join(cwd_full, f"{string}_{month}{day}{year}_nsim{n_sim}")
    
#     # Check if the folder exists, if not, create it
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name, exist_ok=True)
    
#     return folder_name

# #run for 5x5;
# def minimize_OP(Alpha_vec, data_noisy, G, nT2, g):
#     OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
#     OP_rhos = np.zeros((len(Alpha_vec)))
#     for j in (range(len(Alpha_vec))):
#         try:
#             # Fallback to nonnegtik_hnorm
#             sol, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j], '0', nargin=4)
#             if np.all(sol == 0):
#                 print(f"Solution from nonnegtik_hnorm is a zero vector, switching to CVXPY solver.")
#                 raise ValueError("Zero vector detected, switching to CVXPY.")
#         except Exception as e:
#             print(f"Error in nonnegtik_hnorm: {e}")

#             # If both methods fail or if the solution was a zero vector, solve using cvxpy
#             lam_vec = Alpha_vec[j] * np.ones(G.shape[1])
#             A = (G.T @ G + np.diag(lam_vec))
#             eps = 1e-2
#             ep4 = np.ones(A.shape[1]) * eps
#             b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vec
#             y = cp.Variable(G.shape[1])
#             cost = cp.norm(A @ y - b, 2)**2
#             constraints = [y >= 0]
#             problem = cp.Problem(cp.Minimize(cost), constraints)
#             problem.solve(solver=cp.MOSEK, verbose=False)
#             sol = y.value
#             sol = sol - eps
#             sol = np.maximum(sol, 0)
#         OP_x_lc_vec[:, j] = sol
#         # Calculate the error (rho)
#         if err_type == "WassScore":
#                     # err_LC = wass_error(IdealModel_weighted, f_rec_LC)
#             OP_rhos[j] = wass_error(g, OP_x_lc_vec[:, j])
#         else:
#             OP_rhos[j] = l2_error(g, OP_x_lc_vec[:, j])
#     min_rhos = min(OP_rhos)
#     min_index = np.argmin(OP_rhos)
#     min_x = Alpha_vec[min_index][0]
#     OP_min_alpha1 = min_x
#     OP_min_alpha1_ind = min_index
#     f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
#     return f_rec_OP_grid, OP_min_alpha1, min_rhos , min_index

# def calc_T2mu(rps):
#     mps = rps / 2
#     nrps = len(rps)
#     T2_left = 40 * np.ones(nrps)
#     T2_mid = T2_left * mps
#     T2_right = T2_left * rps
#     T2mu = np.column_stack((T2_left, T2_right))
#     return T2mu

# def calc_sigma_i(iter_i, diff_sigma):
#     sigma_i = diff_sigma[iter_i, :]
#     return sigma_i

# def calc_rps_val(iter_j, rps):
#     rps_val = rps[iter_j]
#     return rps_val

# def calc_diff_sigma(nsigma):
#     unif_sigma = np.linspace(2, 5, nsigma).T
#     diff_sigma = np.column_stack((unif_sigma, 3 *unif_sigma))
#     # diff_sigma = np.column_stack((3 *unif_sigma, unif_sigma))

#     return unif_sigma, diff_sigma

# def load_Gaus(Gaus_info):
#     n, m = Gaus_info['A'].shape
#     # T2 = Gaus_info['T2'].flatten()
#     T2 = np.linspace(10,200,m)
#     TE = Gaus_info['TE'].flatten()
#     A = Gaus_info['A']
#     # Lambda = Gaus_info['Lambda'].reshape(-1,1)
#     # Lambda = np.append(0, Lambda)
#     # Lambda = np.append(0, np.logspace(-6,-1,20)).reshape(-1,1)
#     SNR = SNR_value
#     return T2, TE, A, m,  SNR

# def calc_dat_noisy(A, TE, IdealModel_weighted, SNR, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     dat_noiseless = A @ IdealModel_weighted  # Compute noiseless data
#     SD_noise = np.max(np.abs(dat_noiseless)) / SNR  # Standard deviation of noise
#     noise = np.random.normal(0, SD_noise, size=dat_noiseless.shape)  # Add noise
#     dat_noisy = dat_noiseless + noise
#     return dat_noisy, noise, SD_noise

# def get_IdealModel_weighted(iter_j, m, npeaks, T2, T2mu, sigma_i):
#     p = np.zeros((npeaks, m))
#     T2mu_sim = T2mu[iter_j, :]
#     p = np.array([normsci.pdf(T2, mu, sigma) for mu, sigma in zip(T2mu_sim, sigma_i)])
#     IdealModel_weighted = (p.T @ f_coef) / npeaks
#     return IdealModel_weighted

# def l2_error(IdealModel,reconstr):
#     true_norm = linalg_norm(IdealModel)
#     err = linalg_norm(IdealModel - reconstr) / true_norm
#     return err

# def wass_error(IdealModel,reconstr):
#     true_norm = linalg_norm(IdealModel)
#     #check the absolute errors pattern vs SNRs.
#     # err = wasserstein_distance(IdealModel,reconstr)/true_norm
#     err = wasserstein_distance(IdealModel,reconstr)
#     return err

# def generate_estimates(i_param_combo, seed=None):
#  def plot(iter_sim, iter_sigma, iter_rps):
#     plt.figure(figsize=(17, 8))
#     errors = {'LocReg': err_LR, 'Oracle': err_oracle, 'DP': err_DP, 'GCV': err_GCV, 'L-curve': err_LC, "UPEN_Zama": err_upen, 'LocReg_1st_Der': err_LR1, 'LocReg_2nd_Der': err_LR2}
#     min_method = min(errors, key=errors.get)
#     plt.subplot(1, 3, 1)
#     plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
#     locreg_label = f'LocReg (Error: {err_LR:.2e})' + (' *' if min_method == 'LocReg' else '')
#     plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=locreg_label)
#     locreg_label1 = f'LocReg 1st Derivative (Error: {err_LR1:.2e})' + (' *' if min_method == 'LocReg_1st_Der' else '')
#     plt.plot(T2, f_rec_LocReg_LC1, linestyle=':', linewidth=3, color='green', label=locreg_label1)
#     locreg2_label = f'LocReg 2nd Derivative (Error: {err_LR2:.2e})' + (' *' if min_method == 'LocReg_2nd_Der' else '')
#     plt.plot(T2, f_rec_LocReg_LC2, linestyle=':', linewidth=3, color='orange', label=locreg2_label)
#     oracle_label = f'Oracle (Error: {err_oracle:.2e})' + (' *' if min_method == 'Oracle' else '')
#     plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=oracle_label)
#     gcv_label = f'GCV (Error: {err_GCV:.2e})' + (' *' if min_method == 'GCV' else '')
#     plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=gcv_label)
#     upen_zama_label = f'UPEN (Error: {err_upen:.2e})' + (' *' if min_method == 'UPEN_Zama' else '')
#     plt.plot(T2, upen_zama_sol, linestyle='--', linewidth=3, color='purple', label=upen_zama_label)
#     plt.legend(fontsize=10, loc='best')
#     plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
#     plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
#     plt.ylim(0, np.max(IdealModel_weighted) * 1.15)
#     plt.subplot(1, 3, 2)
#     plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
#     plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label='LocReg')
#     plt.plot(TE, A @ f_rec_LocReg_LC1, linestyle=':', linewidth=3, color='green', label='LocReg 1st Derivative')
#     plt.plot(TE, A @ f_rec_LocReg_LC2, linestyle=':', linewidth=3, color='orange', label='LocReg 2nd Derivative')
#     plt.plot(TE, A @ f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle')
#     plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
#     plt.plot(TE, A @ upen_zama_sol, linestyle='-.', linewidth=3, color='purple', label='UPEN')
#     plt.legend(fontsize=10, loc='best')
#     plt.xlabel('TE', fontsize=20, fontweight='bold')
#     plt.ylabel('Intensity', fontsize=20, fontweight='bold')
#     plt.subplot(1, 3, 3)
#     plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
#     plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label='LocReg')
#     plt.semilogy(T2, lambda_locreg_LC1 * np.ones(len(T2)), linestyle=':', linewidth=3, color='green', label='LocReg 1st Derivative')
#     plt.semilogy(T2, lambda_locreg_LC2 * np.ones(len(T2)), linestyle=':', linewidth=3, color='orange', label='LocReg 2nd Derivative')
#     plt.semilogy(T2, lambda_oracle * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')
#     plt.semilogy(T2, upen_zama_lams * np.ones(len(T2)), linestyle='-.', linewidth=3, color='purple', label='UPEN')
#     plt.legend(fontsize=10, loc='best')
#     plt.xlabel('T2', fontsize=20, fontweight='bold')
#     plt.ylabel('Lambda', fontsize=20, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(os.path.join(file_path_final, f"Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}.png"))
#     plt.close()
# # Identify which sigma and rps for a given n_sim iteration
#     if parallel == True:
#         iter_sim, iter_sigma, iter_rps = target_iterator[i_param_combo]
#     else:
#         iter_sim, iter_sigma, iter_rps = i_param_combo
#     sigma_i = diff_sigma[iter_sigma, :]
#     rps_val = calc_rps_val(iter_rps, rps)
#     IdealModel_weighted = get_IdealModel_weighted(iter_rps, m, npeaks, T2, T2mu, sigma_i)
#     dT = T2[-1]/m
#     if preset_noise == False:
#         dat_noisy,noise, stdnoise = calc_dat_noisy(A, TE, IdealModel_weighted, SNR, seed)
#     else:
#         dat_noiseless = A @ IdealModel_weighted
#         try:
#             noise = noise_arr[iter_sim, iter_sigma, iter_rps,:]
#         except IndexError:
#             print("Preset Noise Array Doesn't Match Current Number of Simulations")
#         dat_noisy = dat_noiseless + np.ravel(noise)
#         noisy_data[iter_sim, iter_sigma, iter_rps,:] = dat_noisy
#         noiseless_data[iter_sim, iter_sigma, iter_rps,:] = dat_noiseless
#         stdnoise = stdnoise_data[iter_sim,iter_sigma,iter_rps,:]
#         stdnoise = stdnoise[0]
#     f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda, stdnoise)
#     f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
#     f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
#     f_rec_GCV = f_rec_GCV[:, 0]
#     lambda_GCV = np.squeeze(lambda_GCV)
#     f_rec_oracle, lambda_oracle, min_rhos , min_index = minimize_OP(Lambda, dat_noisy, A, len(T2), IdealModel_weighted)
#     if lam_ini_val == "LCurve" or lam_ini_val == "L-Curve":
#         LRIto_ini_lam = lambda_LC
#         f_rec_ini = f_rec_LC
#     elif lam_ini_val == "GCV" or lam_ini_val == "gcv":
#         LRIto_ini_lam = lambda_GCV
#         f_rec_ini = f_rec_GCV
#     elif lam_ini_val == "DP" or lam_ini_val == "dp":
#         LRIto_ini_lam = lambda_DP
#         f_rec_ini = f_rec_DP
#     maxiter = 50
#     LRIto_ini_lam = lambda_GCV
#     f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
#     f_rec_LocReg_LC1, lambda_locreg_LC1, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
#     f_rec_LocReg_LC2, lambda_locreg_LC2, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv2(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
#     noise_norm = np.linalg.norm(noise)
#     try:
#         upen_zama_sol, upen_zama_lams = UPEN_Zama(A, dat_noisy, IdealModel_weighted, noise_norm, beta_0, Kmax, tol_lam)
#     except Exception as e:
#         print("An error occurred while running the UPEN_Zama function:")
#         print(e)
#     try:
#         upen_zama_sol1D, upen_zama_lams1D = UPEN_Zama1st(A, dat_noisy, IdealModel_weighted, noise_norm, beta_0, Kmax, tol_lam)
#     except Exception as e:
#         print("An error occurred while running the UPEN_1st Derivative function:")
#         print(e)
#     try:
#         upen_zama_sol0D, upen_zama_lams0D = UPEN_Zama0th(A, dat_noisy, IdealModel_weighted, noise_norm, beta_0, Kmax, tol_lam)
#     except Exception as e:
#         print("An error occurred while running the UPEN 0th Derivative function:")
#         print(e)
#     sum_x2 = np.trapz(f_rec_LocReg_LC2, T2)
#     f_rec_LocReg_LC2 = f_rec_LocReg_LC2 / sum_x2
#     sum_x = np.trapz(f_rec_LocReg_LC, T2)
#     f_rec_LocReg_LC = f_rec_LocReg_LC / sum_x
#     sum_x1 = np.trapz(f_rec_LocReg_LC1, T2)
#     f_rec_LocReg_LC1 = f_rec_LocReg_LC1 / sum_x1
#     sum_oracle = np.trapz(f_rec_oracle, T2)
#     f_rec_oracle = f_rec_oracle / sum_oracle
#     sum_GCV = np.trapz(f_rec_GCV, T2)
#     f_rec_GCV = f_rec_GCV / sum_GCV
#     sum_LC = np.trapz(f_rec_LC, T2)
#     f_rec_LC = f_rec_LC / sum_LC
#     sum_DP = np.trapz(f_rec_DP, T2)
#     f_rec_DP = f_rec_DP / sum_DP
#     upen_sum = np.trapz(upen_zama_sol, T2)
#     upen_zama_sol = upen_zama_sol / upen_sum
#     upen1D_sum = np.trapz(upen_zama_sol1D, T2)
#     upen_zama_sol1D = upen_zama_sol1D / upen1D_sum
#     upen0D_sum = np.trapz(upen_zama_sol0D, T2)
#     upen_zama_sol0D = upen_zama_sol0D / upen0D_sum
#     f_rec_GCV = f_rec_GCV.flatten()
#     f_rec_DP = f_rec_DP.flatten()
#     f_rec_LC = f_rec_LC.flatten()
#     f_rec_LocReg_LC = f_rec_LocReg_LC.flatten()
#     f_rec_LocReg_LC1 = f_rec_LocReg_LC1.flatten()
#     f_rec_LocReg_LC2 = f_rec_LocReg_LC2.flatten()
#     f_rec_oracle = f_rec_oracle.flatten()
#     upen_zama_sol = upen_zama_sol.flatten()
#     upen_zama_sol1D = upen_zama_sol1D.flatten()
#     upen_zama_sol0D = upen_zama_sol0D.flatten()
#     if err_type == "WassScore":
#         err_LC = wass_error(IdealModel_weighted, f_rec_LC)
#         err_DP = wass_error(IdealModel_weighted, f_rec_DP)
#         err_GCV = wass_error(IdealModel_weighted, f_rec_GCV)
#         err_oracle = wass_error(IdealModel_weighted, f_rec_oracle)
#         err_LR = wass_error(IdealModel_weighted, f_rec_LocReg_LC)
#         err_LR1 = wass_error(IdealModel_weighted, f_rec_LocReg_LC1)
#         err_LR2 = wass_error(IdealModel_weighted, f_rec_LocReg_LC2)
#         err_upen = wass_error(IdealModel_weighted, upen_zama_sol)
#         err_upen1D = wass_error(IdealModel_weighted, upen_zama_sol1D)
#         err_upen0D = wass_error(IdealModel_weighted, upen_zama_sol0D)
#         bias_LC = np.mean((f_rec_LC - IdealModel_weighted))
#         bias_DP = np.mean((f_rec_DP - IdealModel_weighted))
#         bias_GCV = np.mean((f_rec_GCV - IdealModel_weighted))
#         bias_oracle = np.mean((f_rec_oracle - IdealModel_weighted))
#         bias_LR = np.mean((f_rec_LocReg_LC - IdealModel_weighted))
#         bias_LR1 = np.mean((f_rec_LocReg_LC1 - IdealModel_weighted))
#         bias_LR2 = np.mean((f_rec_LocReg_LC2 - IdealModel_weighted))
#         bias_upen = np.mean((upen_zama_sol - IdealModel_weighted))
#         bias_upen1D = np.mean((upen_zama_sol1D - IdealModel_weighted))
#         bias_upen0D = np.mean((upen_zama_sol0D - IdealModel_weighted))
#         var_LC = np.var(f_rec_LC - IdealModel_weighted)
#         var_DP = np.var(f_rec_DP - IdealModel_weighted)
#         var_GCV = np.var(f_rec_GCV - IdealModel_weighted)
#         var_oracle = np.var(f_rec_oracle - IdealModel_weighted)
#         var_LR = np.var(f_rec_LocReg_LC - IdealModel_weighted)
#         var_LR1 = np.var(f_rec_LocReg_LC1 - IdealModel_weighted)
#         var_LR2 = np.var(f_rec_LocReg_LC2 - IdealModel_weighted)
#         var_upen = np.var(upen_zama_sol - IdealModel_weighted)
#         var_upen1D = np.var(upen_zama_sol1D - IdealModel_weighted)
#         var_upen0D = np.var(upen_zama_sol0D - IdealModel_weighted)
#         MSE_LC = np.mean((f_rec_LC - IdealModel_weighted)**2)
#         MSE_DP = np.mean((f_rec_DP - IdealModel_weighted)**2)
#         MSE_GCV = np.mean((f_rec_GCV - IdealModel_weighted)**2)
#         MSE_oracle = np.mean((f_rec_oracle - IdealModel_weighted)**2)
#         MSE_LR = np.mean((f_rec_LocReg_LC - IdealModel_weighted)**2)
#         MSE_LR1 = np.mean((f_rec_LocReg_LC1 - IdealModel_weighted)**2)
#         MSE_LR2 = np.mean((f_rec_LocReg_LC2 - IdealModel_weighted)**2)
#         MSE_upen = np.mean((upen_zama_sol - IdealModel_weighted)**2)
#         MSE_upen1D = np.mean((upen_zama_sol1D - IdealModel_weighted)**2)
#         MSE_upen0D = np.mean((upen_zama_sol0D - IdealModel_weighted)**2)
#     else:
#         err_LC = l2_error(IdealModel_weighted, f_rec_LC)
#         err_DP = l2_error(IdealModel_weighted, f_rec_DP)
#         err_GCV = l2_error(IdealModel_weighted, f_rec_GCV)
#         err_oracle = l2_error(IdealModel_weighted, f_rec_oracle)
#         err_LR = l2_error(IdealModel_weighted, f_rec_LocReg_LC)
#         err_LR1 = l2_error(IdealModel_weighted, f_rec_LocReg_LC1)
#         err_LR2 = l2_error(IdealModel_weighted, f_rec_LocReg_LC2)
#         err_upen = l2_error(IdealModel_weighted, upen_zama_sol)
#     realresult = 1
#     if not(err_oracle <= err_LC and err_oracle <= err_GCV and err_oracle <= err_DP):
#         print("err oracle", err_oracle)
#         print("oracle", f_rec_oracle)
#         print("err_LC", err_LC)
#         print("err_GCV", err_GCV)
#         print("err_DP", err_DP)
#         print("lambda_oracle", lambda_oracle)
#         print("min_rhos", min_rhos)
#         print("min_index", min_index)
#         realresult = 0
#         print("Oracle Error should not be larger than other single parameter methods")
#     if iter_sim == 0:
#         plot(iter_sim, iter_sigma, iter_rps)
#         print(f"Finished Plots for iteration {iter_sim} sigma {iter_sigma} rps {iter_rps}")
#     if realresult == 1:
#         feature_df = pd.DataFrame(columns=["NR",'Sigma','RPS_val','err_DP',"err_LC","err_LR","err_GCV","err_oracle",
#                                         "LR_vect","oracle_vect","DP_vect","LC_vect","GCV_vect",
#                                         "MSE_LC","Bias_LC","Var_LC","MSE_DP","Bias_DP","Var_DP",
#                                         "MSE_GCV","Bias_GCV","Var_GCV","MSE_oracle","Bias_oracle",
#                                         "Var_oracle","MSE_LR","Bias_LR","Var_LR","MSE_LR1","Bias_LR1",
#                                         "Var_LR1","MSE_LR2","Bias_LR2","Var_LR2","MSE_upen","Bias_upen",
#                                         "Var_upen","DP_lam","LC_lam","GCV_lam","upen_lam","oracle_lam",
#                                         "LR_lam","LR_lam_1stDer","LR_lam_2ndDer"])
#         feature_df["NR"]=[iter_sim];feature_df["Sigma"]=[sigma_i];feature_df["RPS_val"]=[rps_val];feature_df["GT"]=[IdealModel_weighted]
#         feature_df["err_DP"]=[err_DP];feature_df["err_LC"]=[err_LC];feature_df["err_LR"]=[err_LR];feature_df["err_LR_1stDer"]=[err_LR1];feature_df["err_LR_2ndDer"]=[err_LR2]
#         feature_df["err_GCV"]=[err_GCV];feature_df["err_oracle"]=[err_oracle];feature_df["err_upen"]=[err_upen];feature_df["err_upen0D"]=[err_upen0D];feature_df["err_upen1D"]=[err_upen1D]
#         feature_df["LR_vect"]=[f_rec_LocReg_LC];feature_df["LR_vect_1stDer"]=[f_rec_LocReg_LC1];feature_df["LR_vect_2ndDer"]=[f_rec_LocReg_LC2]
#         feature_df["oracle_vect"]=[f_rec_oracle];feature_df["DP_vect"]=[f_rec_DP];feature_df["LC_vect"]=[f_rec_LC];feature_df["GCV_vect"]=[f_rec_GCV]
#         feature_df["upen_vect"]=[upen_zama_sol];feature_df["upen_vect1D"]=[upen_zama_sol1D];feature_df["upen_vect0D"]=[upen_zama_sol0D]
#         feature_df["DP_lam"]=[lambda_DP];feature_df["LC_lam"]=[lambda_LC];feature_df["GCV_lam"]=[lambda_GCV];feature_df["upen_lam"]=[upen_zama_lams]
#         feature_df["upen_lam1D"]=[upen_zama_lams1D];feature_df["upen_lam0D"]=[upen_zama_lams0D];feature_df["oracle_lam"]=[lambda_oracle]
#         feature_df["LR_lam"]=[lambda_locreg_LC];feature_df["LR_lam_1stDer"]=[lambda_locreg_LC1];feature_df["LR_lam_2ndDer"]=[lambda_locreg_LC2]
#         feature_df["MSE_LC"]=[MSE_LC];feature_df["Bias_LC"]=[bias_LC];feature_df["Var_LC"]=[var_LC];feature_df["MSE_DP"]=[MSE_DP];feature_df["Bias_DP"]=[bias_DP]
#         feature_df["Var_DP"]=[var_DP];feature_df["MSE_GCV"]=[MSE_GCV];feature_df["Bias_GCV"]=[bias_GCV];feature_df["Var_GCV"]=[var_GCV];feature_df["MSE_oracle"]=[MSE_oracle]
#         feature_df["Bias_oracle"]=[bias_oracle];feature_df["Var_oracle"]=[var_oracle];feature_df["MSE_LR"]=[MSE_LR];feature_df["Bias_LR"]=[bias_LR];feature_df["Var_LR"]=[var_LR]
#         feature_df["MSE_LR1"]=[MSE_LR1];feature_df["Bias_LR1"]=[bias_LR1];feature_df["Var_LR1"]=[var_LR1];feature_df["MSE_LR2"]=[MSE_LR2];feature_df["Bias_LR2"]=[bias_LR2]
#         feature_df["Var_LR2"]=[var_LR2];feature_df["MSE_upen"]=[MSE_upen];feature_df["Bias_upen"]=[bias_upen];feature_df["Var_upen"]=[var_upen]
#         feature_df["MSE_upen0D"]=[MSE_upen0D];feature_df["Bias_upen0D"]=[bias_upen0D];feature_df["Var_upen0D"]=[var_upen0D];feature_df["MSE_upen1D"]=[MSE_upen1D]
#         feature_df["Bias_upen1D"]=[bias_upen1D];feature_df["Var_upen1D"]=[var_upen1D]
#     else:
#         print("Skipped because not a good noise realization where Oracle is not the lowest value")
#         feature_df = pd.DataFrame(columns=["NR",'Sigma','RPS_val','err_DP',"err_LC","err_LR","err_GCV","err_oracle",
#                                         "LR_vect","oracle_vect","DP_vect","LC_vect","GCV_vect",
#                                         "MSE_LC","Bias_LC","Var_LC","MSE_DP","Bias_DP","Var_DP",
#                                         "MSE_GCV","Bias_GCV","Var_GCV","MSE_oracle","Bias_oracle",
#                                         "Var_oracle","MSE_LR","Bias_LR","Var_LR","MSE_LR1","Bias_LR1",
#                                         "Var_LR1","MSE_LR2","Bias_LR2","Var_LR2","MSE_upen","Bias_upen",
#                                         "Var_upen","DP_lam","LC_lam","GCV_lam","upen_lam","oracle_lam",
#                                         "LR_lam","LR_lam_1stDer","LR_lam_2ndDer"])
#         feature_df["NR"]=[iter_sim];feature_df["Sigma"]=[sigma_i];feature_df["GT"]=[IdealModel_weighted];feature_df["RPS_val"]=[rps_val]
#         feature_df["err_DP"]=[None];feature_df["err_LC"]=[None];feature_df["err_LR"]=[None];feature_df["err_GCV"]=[None];feature_df["err_oracle"]=[None]
#         feature_df["err_upen"]=[None];feature_df["LR_vect"]=[None];feature_df["oracle_vect"]=[None];feature_df["DP_vect"]=[None];feature_df["LC_vect"]=[None]
#         feature_df["GCV_vect"]=[None];feature_df["upen_vect"]=[None];feature_df["DP_lam"]=[lambda_DP];feature_df["LC_lam"]=[lambda_LC];feature_df["GCV_lam"]=[lambda_GCV]
#         feature_df["upen_lam"]=[upen_zama_lams];feature_df["upen1D_lam"]=[upen_zama_lams1D];feature_df["upen0D_lam"]=[upen_zama_lams0D];feature_df["oracle_lam"]=[lambda_oracle]
#         feature_df["LR_lam"]=[lambda_locreg_LC];feature_df["LR_lam_1stDer"]=[lambda_locreg_LC1];feature_df["LR_lam_2ndDer"]=[lambda_locreg_LC2]
#     return feature_df, iter_sim, iter_sigma, iter_rps, noise, stdnoise

# def compare_heatmap():
#     fig, axs = plt.subplots(3, 2, sharey=True, figsize=(18, 22))
#     plt.subplots_adjust(wspace=0.3, hspace=0.4)
#     tick_labels = [
#         ['LocReg2ndDeriv is better', 'Neutral', 'GCV is better'],
#         ['LocReg2ndDeriv is better', 'Neutral', 'DP is better'],
#         ['LocReg2ndDeriv is better', 'Neutral', 'L-Curve is better'],
#         ['LocReg2ndDeriv is better', 'Neutral', 'Oracle is better'],
#         ['LocReg2ndDeriv is better', 'Neutral', 'UPEN is better']
#     ]
#     axs = axs.flatten()
#     x_ticksval, y_ticksval = rps, unif_sigma
#     if err_type == "WassScore":
#         vmax1 = np.max([np.max(errs_LC), np.max(errs_GCV), np.max(errs_DP), np.max(errs_oracle), np.max(errs_upen), np.max(errs_LR2)])
#         vmin1 = -vmax1
#     else:
#         vmin1, vmax1 = -0.5, 0.5
#     fmt1 = ".2e" if err_type == "WassScore" else ".3e"

#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         im = sns.heatmap(data, cmap='jet', ax=ax, cbar=True, vmin=vmin1, vmax=vmax1, annot=True, fmt=fmt1,
#                          annot_kws={"size": 12, "weight": "bold"}, linewidths=0.5, linecolor='black',
#                          cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8}, xticklabels=1, yticklabels=1)
#         ax.set_xlabel('Peak Separation', fontsize=20)
#         ax.set_ylabel('Peak Width', fontsize=20)
#         ax.set_title(title, fontsize=20, pad=20)
#         ax.set_xticklabels(np.round(x_ticks, 4), rotation=-90, fontsize=14)
#         ax.set_yticklabels(np.round(y_ticks, 4), fontsize=14)
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([vmin1, 0, vmax1])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)
#         cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

#     avg_GCV = f"{np.mean(compare_GCV):.2e}"
#     avg_DP = f"{np.mean(compare_DP):.2e}"
#     avg_LC = f"{np.mean(compare_LC):.2e}"
#     avg_oracle = f"{np.mean(compare_oracle):.2e}"
#     avg_LR2_upen = f"{np.mean(compare_LR2_upen):.2e}"

#     add_heatmap(axs[0], compare_GCV, tick_labels[0], f'LocReg Error - GCV Error ({err_type})\nAverage GCV Comparison Score: {avg_GCV}', x_ticksval, y_ticksval)
#     add_heatmap(axs[1], compare_DP, tick_labels[1], f'LocReg Error - DP Error ({err_type})\nAverage DP Comparison Score: {avg_DP}', x_ticksval, y_ticksval)
#     add_heatmap(axs[2], compare_LC, tick_labels[2], f'LocReg Error - L-Curve Error ({err_type})\nAverage LCurve Comparison Score: {avg_LC}', x_ticksval, y_ticksval)
#     add_heatmap(axs[3], compare_oracle, tick_labels[3], f'LocReg Error - Oracle Error ({err_type})\nAverage Oracle Comparison Score: {avg_oracle}', x_ticksval, y_ticksval)
#     add_heatmap(axs[4], compare_LR2_upen, tick_labels[4], f'LocReg2ndDeriv Error - UPEN Error ({err_type})\nAverage UPEN Comparison Score: {avg_LR2_upen}', x_ticksval, y_ticksval)
#     fig.delaxes(axs[5])
#     for ax in axs:
#         ax.xaxis.tick_bottom()
#         ax.xaxis.set_label_position('bottom')
#         ax.tick_params(labelleft=True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(file_path_final, "compare_heatmap.png"))
#     print("Saved Comparison Heatmap")
#     plt.close()

# def indiv_heatmap():
#     fig, axs = plt.subplots(4, 2, sharey=True, figsize=(18, 22))
#     plt.subplots_adjust(wspace=0.3, hspace=0.3)
#     tick_labels = [
#         ['Low LocReg Error', 'High LocReg Error'], ['Low DP Error', 'High DP Error'],  
#         ['Low LCurve Error', 'High LCurve Error'], ['Low Oracle Error', 'High Oracle Error'],  
#         ['Low GCV Error', 'High GCV Error'], ['Low UPEN Error', 'High UPEN Error'],
#         ['Low LocReg1stDer Error', 'High LocReg1stDer Error'], ['Low LocReg2ndDer Error', 'High LocReg2ndDer Error']
#     ]
#     axs = axs.flatten()
#     x_ticks, y_ticks = rps, unif_sigma
#     vmax1 = np.max([np.max(errs_LR), np.max(errs_LC), np.max(errs_GCV), np.max(errs_DP), np.max(errs_oracle), np.max(errs_upen), np.max(errs_LR1), np.max(errs_LR2)])

#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         fmt1 = ".4f" if err_type == "WassScore" else ".3f"
#         im = sns.heatmap(data, cmap='jet', ax=ax, cbar=True, vmin=0, vmax=vmax1, annot=True, fmt=fmt1,
#                          annot_kws={"size": 12, "weight": "bold"}, linewidths=0.5, linecolor='black',
#                          cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8}, xticklabels=1, yticklabels=1)
#         ax.set_xlabel('Peak Separation', fontsize=20)
#         ax.set_ylabel('Peak Width', fontsize=20)
#         ax.set_title(title, fontsize=20, pad=20)
#         ax.set_xticklabels(np.round(x_ticks, 4), rotation=-90)
#         ax.set_yticklabels(np.round(y_ticks, 4))
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([0, vmax1])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)
#         cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

#     avg_GCV, avg_DP, avg_LC = f"{np.mean(errs_GCV):.2e}", f"{np.mean(errs_DP):.2e}", f"{np.mean(errs_LC):.2e}"
#     avg_oracle, avg_LR, avg_upen = f"{np.mean(errs_oracle):.2e}", f"{np.mean(errs_LR):.2e}", f"{np.mean(errs_upen):.2e}"
#     avg_LR1, avg_LR2 = f"{np.mean(errs_LR1):.2e}", f"{np.mean(errs_LR2):.2e}"

#     add_heatmap(axs[0], errs_LR, tick_labels[0], f'LocReg Error ({err_type})\nAverage Score: {avg_LR}', x_ticks, y_ticks)
#     add_heatmap(axs[0], errs_DP, tick_labels[0], f'DP Error ({err_type})\nAverage Score: {avg_DP}', x_ticks, y_ticks)
#     add_heatmap(axs[1], errs_LC, tick_labels[1], f'L-Curve Error ({err_type})\nAverage Score: {avg_LC}', x_ticks, y_ticks)
#     add_heatmap(axs[2], errs_oracle, tick_labels[2], f'Oracle Error ({err_type})\nAverage Score: {avg_oracle}', x_ticks, y_ticks)
#     add_heatmap(axs[3], errs_GCV, tick_labels[3], f'GCV Error ({err_type})\nAverage Score: {avg_GCV}', x_ticks, y_ticks)
#     add_heatmap(axs[4], errs_upen, tick_labels[4], f'UPEN Error ({err_type})\nAverage Score: {avg_upen}', x_ticks, y_ticks)
#     add_heatmap(axs[6], errs_LR1, tick_labels[0], f'LocReg1stDer Error ({err_type})\nAverage Score: {avg_LR1}', x_ticks, y_ticks)
#     add_heatmap(axs[5], errs_LR2, tick_labels[5], f'LocReg2ndDer Error ({err_type})\nAverage Score: {avg_LR2}', x_ticks, y_ticks)

#     for ax in axs[:6]:
#         ax.xaxis.tick_bottom()
#         ax.xaxis.set_label_position('bottom')
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)
#         ax.tick_params(labelleft=True)

#     plt.tight_layout()
#     plt.savefig(os.path.join(file_path_final, "indiv_heatmap.png"))
#     print("Saved Individual Heatmap")
#     plt.close()

# def worker_init():
#     worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
#     np.random.seed(worker_id)

# def parallel_processed(func, shift=True):
#     with mp.Pool(processes=num_cpus_avail, initializer=worker_init) as pool, tqdm(total=len(target_iterator)) as pbar:
#         for estimates_dataframe, iter_sim, iter_sigma, iter_rps, noisereal, std_noisereal in pool.imap_unordered(func, range(len(target_iterator))):
#             lis.append(estimates_dataframe)
#             noise_arr[iter_sim, iter_sigma, iter_rps, :] = noisereal
#             stdnoise_data[iter_sim, iter_sigma, iter_rps, :] = std_noisereal
#             pbar.update()
#         pool.close()
#         pool.join()
#     return estimates_dataframe, noise_arr, stdnoise_data

# if __name__ == '__main__':
#     if 'TERM_PROGRAM' in os.environ and os.environ['TERM_PROGRAM'] == 'vscode':
#         print("Running in VS Code")
#     logging.info("Script started.")
#     freeze_support()
#     unif_sigma, diff_sigma = calc_diff_sigma(nsigma)
#     T2, TE, A, m, SNR = load_Gaus(Gaus_info)
#     print("TE", TE)
#     T2mu = calc_T2mu(rps)
#     string = "MRR_1D_LocReg_Comparison"
#     file_path_final = create_result_folder(string, SNR, lam_ini_val, dist_type)
#     print("Finished Assignments...")
#     lastpath = os.path.join(file_path_final, f'{data_tag}.pkl')
#     directory_path = os.path.dirname(lastpath)
#     lis, lis_L2, lis_w, sigma_rps_labels = [], [], [], []

#     if parallel:
#         estimates_dataframe, noise_arr, stdnoise_data = parallel_processed(generate_estimates, shift=True)
#     else:
#         for i in range(n_sim):
#             for j in range(nsigma):
#                 for k in range(nrps):
#                     iter = (i, j, k)
#                     estimates_dataframe, iter_sim, iter_sigma, iter_rps, noisereal, std_noisereal = generate_estimates(iter)
#                     lis.append(estimates_dataframe)
#                     noise_arr[iter_sim, iter_sigma, iter_rps, :] = noisereal
#                     stdnoise_data[iter_sim, iter_sigma, iter_rps, :] = std_noisereal

#     print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
#     df = pd.concat(lis, ignore_index=True)
#     if not os.path.exists(directory_path):
#         print(f"The folder {directory_path} does not exist. Creating it now...")
#         os.makedirs(directory_path, exist_ok=True)
#     else:
#         print("directory made")
#     df.to_pickle(lastpath)

#     df['Sigma'] = df['Sigma'].apply(tuple)
#     df_sorted = df.sort_values(by=['NR', 'Sigma', 'RPS_val'], ascending=[True, True, True])
#     print("df_sorted", df_sorted)
#     num_NRs = df_sorted['NR'].nunique()
#     na_count = df_sorted.isna().sum().sum()
#     print(f"Total number of NA values in the DataFrame: {na_count}")

#     if num_NRs == 1:
#         df_sorted.fillna(0)
#     else:
#         df_sorted.dropna()

#     na_count = df_sorted.isna().sum().sum()
#     print(f"Total number of NA values in the DataFrame: {na_count}")
#     print("df_sorted.shape", df_sorted.shape)

#     grouped = df_sorted.groupby(['Sigma', 'RPS_val']).agg({
#         'err_DP': 'sum', 'err_LC': 'sum', 'err_LR': 'sum', 'err_GCV': 'sum',
#         'err_oracle': 'sum', "err_upen": "sum", "err_LR_1stDer": "sum", "err_LR_2ndDer": "sum"
#     })
#     average_errors = grouped / num_NRs
#     print("num_NRs", num_NRs)
#     errors = average_errors

#     errs_oracle = np.array(errors["err_oracle"].to_numpy().reshape(nsigma, nrps))
#     errs_LC = np.array(errors["err_LC"].to_numpy().reshape(nsigma, nrps))
#     errs_GCV = np.array(errors["err_GCV"].to_numpy().reshape(nsigma, nrps))
#     errs_DP = np.array(errors["err_DP"].to_numpy().reshape(nsigma, nrps))
#     errs_LR = np.array(errors["err_LR"].to_numpy().reshape(nsigma, nrps))
#     errs_upen = np.array(errors["err_upen"].to_numpy().reshape(nsigma, nrps))
#     errs_LR1 = np.array(errors["err_LR_1stDer"].to_numpy().reshape(nsigma, nrps))
#     errs_LR2 = np.array(errors["err_LR_2ndDer"].to_numpy().reshape(nsigma, nrps))
#     compare_GCV = errs_LR2 - errs_GCV
#     compare_DP = errs_LR2 - errs_DP
#     compare_LC = errs_LR2 - errs_LC
#     compare_oracle = errs_LR2 - errs_oracle
#     compare_upen = errs_LR - errs_upen
#     compare_LR1_upen = errs_LR1 - errs_upen
#     compare_LR2_upen = errs_LR2 - errs_upen
#     if show == 1:
#         # compare_heatmap()
#         indiv_heatmap()
#         if not preset_noise:
#             np.save(file_path_final + f'/{data_tag}noise_arr', noise_arr)
#             print("noise array saved")
#             np.save(file_path_final + f'/{data_tag}stdnoise_data', stdnoise_data)
#             print("standard dev noise array saved")
#         else:
#             print("Used preset noise array and std. dev noise array")
#     logging.info("Script completed.")

# import sys
# import os
# import logging
# from datetime import date

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.ticker as ticker
# from scipy.stats import norm as normsci, wasserstein_distance
# from scipy.linalg import norm as linalg_norm
# import cvxpy as cp
# import multiprocess as mp
# from multiprocessing import Pool, freeze_support
# from tqdm import tqdm

# # Import custom modules (assuming these exist)
# sys.path.append(".")
# from regularization.reg_methods.dp.discrep_L2 import discrep_L2
# from regularization.reg_methods.gcv.GCV_NNLS import GCV_NNLS
# from regularization.reg_methods.lcurve.Lcurve import Lcurve
# from regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm
# from regularization.reg_methods.locreg.LRalgo import LocReg_Ito_mod, LocReg_Ito_mod_deriv, LocReg_Ito_mod_deriv2
# from regularization.reg_methods.upen.upenzama import UPEN_Zama, UPEN_Zama0th, UPEN_Zama1st

from dataclasses import dataclass
from typing import Tuple, List, Optional
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    n_sim: int = 4
    SNR_value: int = 1000
    npeaks: int = 2
    nsigma: int = 5
    nrps: int = None  # Will be calculated
    Kmax: int = 500
    beta_0: float = 1e-7
    tol_lam: float = 1e-5
    gamma_init: float = 0.5
    reg_param_lb: float = -5
    reg_param_ub: float = 1
    N_reg: int = 25
    err_type: str = "WassScore"
    lam_ini_val: str = "GCV"
    dist_type: str = "narrowL_broadR_parallel"
    parallel: bool = False
    show: bool = True
    preset_noise: bool = False

class RegularizationSimulator:
    """Main class for running regularization method comparison simulations."""
    
    def __init__(self, config: SimulationConfig, file_path: str):
        self.config = config
        self.file_path = file_path
        self._setup_environment()
        self._load_data()
        self._initialize_arrays()
        
    # def _setup_environment(self):
    #     """Setup logging and environment variables."""
    #     logging.basicConfig(
    #         filename='my_script.log', 
    #         level=logging.INFO, 
    #         format='%(asctime)s - %(levelname)s - %(message)s'
    #     )
        
    #     # Set MOSEK license path
    #     mosek_license_path = r"C:\Users\kimjosy\Downloads\mosek\mosek.lic"
    #     os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
    #     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    def _setup_environment(self):
        """Setup logging and environment variables."""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Remove all handlers associated with the root logger (if any)
        if logger.hasHandlers():
            logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler('my_script.log')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console (stream) handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Environment variables (keep these as before)
        mosek_license_path = r"C:\Users\kimjosy\Downloads\mosek\mosek.lic"
        os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        
    def _load_data(self):
        """Load and process the Gaussian data."""
        Gaus_info = np.load(self.file_path, allow_pickle=True)
        self.A = Gaus_info["A"]
        self.n, self.m = self.A.shape
        self.TE = Gaus_info["TE"].flatten()
        self.T2 = np.linspace(10, 200, self.m)
        
        # Generate regularization parameters
        self.Lambda = np.logspace(
            self.config.reg_param_lb, 
            self.config.reg_param_ub, 
            self.config.N_reg
        ).reshape(-1, 1)
        
    def _initialize_arrays(self):
        """Initialize arrays for simulation data."""
        # Calculate sigma and rps values
        self.rps = np.linspace(1.1, 4, self.config.nsigma).T
        self.config.nrps = len(self.rps)
        self.unif_sigma = np.linspace(2, 5, self.config.nsigma).T
        self.diff_sigma = np.column_stack((self.unif_sigma, 3 * self.unif_sigma))
        
        # Calculate T2 means for each rps
        self.T2mu = self._calc_T2mu()
        
        # Initialize noise arrays
        shape = (self.config.n_sim, self.config.nsigma, self.config.nrps, self.n)
        self.noise_arr = np.zeros(shape)
        self.stdnoise_data = np.zeros(shape)
        
        # Create target iterator for parallel processing
        self.target_iterator = [
            (a, b, c) 
            for a in range(self.config.n_sim) 
            for b in range(self.config.nsigma) 
            for c in range(self.config.nrps)
        ]
        
    def _calc_T2mu(self) -> np.ndarray:
        """Calculate T2 mean values for different peak separations."""
        mps = self.rps / 2
        T2_left = 40 * np.ones(len(self.rps))
        T2_right = T2_left * self.rps
        return np.column_stack((T2_left, T2_right))
    
    def _create_result_folder(self) -> str:
        """Create folder for saving results."""
        today = date.today()
        folder_name = f"MRR_1D_LocReg_Comparison_{today.strftime('%Y-%m-%d')}_nsim{self.config.n_sim}_SNR{self.config.SNR_value}"
        
        base_path = os.path.join("SimulationSets", "MRR", "SpanRegFig")
        full_path = os.path.join(os.getcwd(), base_path, folder_name)
        
        os.makedirs(full_path, exist_ok=True)
        return full_path
    
    def _generate_ideal_model(self, iter_rps: int, sigma_i: np.ndarray) -> np.ndarray:
        """Generate the ideal/ground truth model."""
        T2mu_sim = self.T2mu[iter_rps, :]
        p = np.array([
            normsci.pdf(self.T2, mu, sigma) 
            for mu, sigma in zip(T2mu_sim, sigma_i)
        ])
        return (p.T @ np.ones(self.config.npeaks)) / self.config.npeaks
    
    def _add_noise(self, dat_noiseless: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Add noise to noiseless data."""
        if seed is not None:
            np.random.seed(seed)
            
        SD_noise = np.max(np.abs(dat_noiseless)) / self.config.SNR_value
        noise = np.random.normal(0, SD_noise, size=dat_noiseless.shape)
        return dat_noiseless + noise, noise, SD_noise
    
    def _minimize_oracle(self, Alpha_vec: np.ndarray, data_noisy: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, float, float, int]:
        """Find oracle solution by minimizing error over regularization parameters."""
        nT2 = len(self.T2)
        OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
        OP_rhos = np.zeros(len(Alpha_vec))
        
        for j, alpha in enumerate(Alpha_vec):
            try:
                sol, _, _ = nonnegtik_hnorm(self.A, data_noisy, alpha, '0', nargin=4)
                if np.all(sol == 0):
                    raise ValueError("Zero vector detected")
            except Exception:
                # Fallback to CVXPY solver
                sol = self._solve_with_cvxpy(alpha, data_noisy)
                
            OP_x_lc_vec[:, j] = sol
            OP_rhos[j] = self._calculate_error(g, sol)
        
        min_index = np.argmin(OP_rhos)
        return (OP_x_lc_vec[:, min_index], 
                Alpha_vec[min_index][0], 
                OP_rhos[min_index], 
                min_index)
    
    def _solve_with_cvxpy(self, alpha: float, data_noisy: np.ndarray) -> np.ndarray:
        """Solve regularization problem using CVXPY."""
        lam_vec = alpha * np.ones(self.A.shape[1])
        A_reg = self.A.T @ self.A + np.diag(lam_vec)
        eps = 1e-2
        ep4 = np.ones(A_reg.shape[1]) * eps
        b = self.A.T @ data_noisy + self.A.T @ self.A @ ep4 + ep4 * lam_vec
        
        y = cp.Variable(self.A.shape[1])
        cost = cp.norm(A_reg @ y - b, 2)**2
        constraints = [y >= 0]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.MOSEK, verbose=False)
        
        sol = y.value - eps
        return np.maximum(sol, 0)
    
    def _calculate_error(self, true_signal: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate error between true and reconstructed signals."""
        if self.config.err_type == "WassScore":
            return wasserstein_distance(true_signal, reconstructed)
        else:
            true_norm = linalg_norm(true_signal)
            return linalg_norm(true_signal - reconstructed) / true_norm
    
    def _normalize_solution(self, solution: np.ndarray) -> np.ndarray:
        """Normalize solution using trapezoidal integration."""
        integral = np.trapz(solution, self.T2)
        return solution / integral if integral != 0 else solution
    
    def _run_all_methods(self, data_noisy: np.ndarray, IdealModel_weighted: np.ndarray, noise_norm: float) -> dict:
        """Run all regularization methods and return results."""
        results = {}
        
        # Standard methods
        results['f_rec_DP'], results['lambda_DP'] = discrep_L2(data_noisy, self.A, self.config.SNR_value, self.Lambda, noise_norm)
        results['f_rec_LC'], results['lambda_LC'] = Lcurve(data_noisy, self.A, self.Lambda)
        results['f_rec_GCV'], results['lambda_GCV'] = GCV_NNLS(data_noisy, self.A, self.Lambda)
        results['f_rec_GCV'] = results['f_rec_GCV'][:, 0]
        results['lambda_GCV'] = np.squeeze(results['lambda_GCV'])
        
        # Oracle method
        (results['f_rec_oracle'], 
         results['lambda_oracle'], 
         results['min_rhos'], 
         results['min_index']) = self._minimize_oracle(self.Lambda, data_noisy, IdealModel_weighted)
        
        # LocReg methods
        initial_lambda = results['lambda_GCV']  # Use GCV as initial
        maxiter = 50
        
        results['f_rec_LocReg'], results['lambda_LR'], _, _, _ = LocReg_Ito_mod(
            data_noisy, self.A, initial_lambda, self.config.gamma_init, maxiter)
        results['f_rec_LocReg1'], results['lambda_LR1'], _, _, _ = LocReg_Ito_mod_deriv(
            data_noisy, self.A, initial_lambda, self.config.gamma_init, maxiter)
        results['f_rec_LocReg2'], results['lambda_LR2'], _, _, _ = LocReg_Ito_mod_deriv2(
            data_noisy, self.A, initial_lambda, self.config.gamma_init, maxiter)
        
        # UPEN methods
        try:
            results['f_rec_upen'], results['lambda_upen'] = UPEN_Zama(
                self.A, data_noisy, IdealModel_weighted, noise_norm, 
                self.config.beta_0, self.config.Kmax, self.config.tol_lam)
            results['f_rec_upen1D'], results['lambda_upen1D'] = UPEN_Zama1st(
                self.A, data_noisy, IdealModel_weighted, noise_norm, 
                self.config.beta_0, self.config.Kmax, self.config.tol_lam)
            results['f_rec_upen0D'], results['lambda_upen0D'] = UPEN_Zama0th(
                self.A, data_noisy, IdealModel_weighted, noise_norm, 
                self.config.beta_0, self.config.Kmax, self.config.tol_lam)
        except Exception as e:
            print(f"UPEN methods failed: {e}")
            # Set default values
            for key in ['f_rec_upen', 'f_rec_upen1D', 'f_rec_upen0D']:
                results[key] = np.zeros_like(IdealModel_weighted)
            for key in ['lambda_upen', 'lambda_upen1D', 'lambda_upen0D']:
                results[key] = 0.0
        
        # Normalize all solutions
        solution_keys = [k for k in results.keys() if k.startswith('f_rec_')]
        for key in solution_keys:
            results[key] = self._normalize_solution(results[key].flatten())
            
        return results
    
    def _calculate_all_errors(self, results: dict, IdealModel_weighted: np.ndarray) -> dict:
        """Calculate all error metrics for the results."""
        errors = {}
        solution_keys = [k for k in results.keys() if k.startswith('f_rec_')]
        
        for key in solution_keys:
            method_name = key.replace('f_rec_', 'err_')
            errors[method_name] = self._calculate_error(IdealModel_weighted, results[key])
            
            # Calculate additional metrics
            diff = results[key] - IdealModel_weighted
            errors[method_name.replace('err_', 'bias_')] = np.mean(diff)
            errors[method_name.replace('err_', 'var_')] = np.var(diff)
            errors[method_name.replace('err_', 'MSE_')] = np.mean(diff**2)
            
        return errors
    
    def generate_single_estimate(self, params: Tuple[int, int, int]) -> pd.DataFrame:
        """Generate estimates for a single parameter combination."""
        iter_sim, iter_sigma, iter_rps = params
        
        # Get parameters for this iteration
        sigma_i = self.diff_sigma[iter_sigma, :]
        IdealModel_weighted = self._generate_ideal_model(iter_rps, sigma_i)
        
        # Generate noisy data
        dat_noiseless = self.A @ IdealModel_weighted
        if not self.config.preset_noise:
            dat_noisy, noise, stdnoise = self._add_noise(dat_noiseless, seed=iter_sim)
            self.noise_arr[iter_sim, iter_sigma, iter_rps, :] = noise
            self.stdnoise_data[iter_sim, iter_sigma, iter_rps, :] = stdnoise
        else:
            noise = self.noise_arr[iter_sim, iter_sigma, iter_rps, :]
            dat_noisy = dat_noiseless + noise
            stdnoise = self.stdnoise_data[iter_sim, iter_sigma, iter_rps, 0]
        
        noise_norm = np.linalg.norm(noise)
        
        # Run all methods
        results = self._run_all_methods(dat_noisy, IdealModel_weighted, noise_norm)
        
        # Calculate errors
        errors = self._calculate_all_errors(results, IdealModel_weighted)
        
        # Check if oracle is actually optimal
        oracle_err = errors['err_oracle']
        other_errs = [errors['err_LC'], errors['err_GCV'], errors['err_DP']]
        if not all(oracle_err <= err for err in other_errs):
            print(f"Warning: Oracle not optimal at iter {iter_sim}, sigma {iter_sigma}, rps {iter_rps}")
            logging.info(f"Warning: Oracle not optimal at iter {iter_sim}, sigma {iter_sigma}, rps {iter_rps}")
            return self._create_empty_dataframe(iter_sim, sigma_i, self.rps[iter_rps], IdealModel_weighted)
        
        # Create results dataframe
        return self._create_results_dataframe(
            iter_sim, sigma_i, self.rps[iter_rps], IdealModel_weighted, 
            results, errors
        )
    
    def _create_results_dataframe(self, iter_sim: int, sigma_i: np.ndarray, rps_val: float, 
                                IdealModel_weighted: np.ndarray, results: dict, errors: dict) -> pd.DataFrame:
        """Create a pandas DataFrame with all results."""
        data = {
            'NR': [iter_sim],
            'Sigma': [sigma_i],
            'RPS_val': [rps_val],
            'GT': [IdealModel_weighted]
        }
        
        # Add all results and errors to the dataframe
        for key, value in {**results, **errors}.items():
            data[key] = [value]
            
        return pd.DataFrame(data)
    
    def _create_empty_dataframe(self, iter_sim: int, sigma_i: np.ndarray, 
                              rps_val: float, IdealModel_weighted: np.ndarray) -> pd.DataFrame:
        """Create an empty dataframe for failed cases."""
        return pd.DataFrame({
            'NR': [iter_sim],
            'Sigma': [sigma_i],
            'RPS_val': [rps_val],
            'GT': [IdealModel_weighted]
        })
    
    def run_simulation(self) -> pd.DataFrame:
        """Run the complete simulation."""
        logging.info("Starting simulation...")
        
        self.result_folder = self._create_result_folder()
        results_list = []
        
        if self.config.parallel:
            with mp.Pool(processes=os.cpu_count()) as pool:
                with tqdm(total=len(self.target_iterator)) as pbar:
                    for result in pool.imap_unordered(self.generate_single_estimate, self.target_iterator):
                        results_list.append(result)
                        pbar.update()
        else:
            for params in tqdm(self.target_iterator):
                result = self.generate_single_estimate(params)
                results_list.append(result)
        
        # Combine all results
        final_df = pd.concat(results_list, ignore_index=True)
        
        # Save results
        self._save_results(final_df)
        
        # Generate plots if requested
        if self.config.show:
            self._generate_plots(final_df)
        
        logging.info("Simulation completed.")
        return final_df
    
    def _save_results(self, df: pd.DataFrame):
        """Save simulation results to file."""
        today = date.today()
        filename = f"est_table_SNR{self.config.SNR_value}_iter{self.config.n_sim}_{today.strftime('%d%b%y')}.pkl"
        filepath = os.path.join(self.result_folder, filename)
        
        df.to_pickle(filepath)
        print(f"Results saved to: {filepath}")
        
        # Save noise arrays if not using preset noise
        if not self.config.preset_noise:
            np.save(os.path.join(self.result_folder, 'noise_arr.npy'), self.noise_arr)
            np.save(os.path.join(self.result_folder, 'stdnoise_data.npy'), self.stdnoise_data)
    
    def _generate_plots(self, df: pd.DataFrame):
        """Generate heatmap plots for visualization."""
        # Group and average results
        df['Sigma'] = df['Sigma'].apply(tuple)
        grouped = df.groupby(['Sigma', 'RPS_val']).agg({
            col: 'mean' for col in df.columns if col.startswith('err_')
        }).reset_index()
        
        # Create heatmaps (simplified version)
        self._create_error_heatmaps(grouped)
    
    def _create_error_heatmaps(self, grouped_df: pd.DataFrame):
        """Create heatmap visualizations of errors."""
        # This is a simplified version - you can expand based on your specific needs
        error_cols = [col for col in grouped_df.columns if col.startswith('err_')]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(error_cols[:8]):  # Plot first 8 error types
            if i < len(axes):
                # Reshape data for heatmap
                pivot_data = grouped_df.pivot(index='Sigma', columns='RPS_val', values=col)
                sns.heatmap(pivot_data, ax=axes[i], cmap='viridis', annot=True, fmt='.2e')
                axes[i].set_title(col)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_folder, 'error_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the simulation."""
    # Configuration
    config = SimulationConfig(
        n_sim=4,
        SNR_value=1000,
        parallel=False,  # Set to True for parallel processing
        show=True
    )
    
    # File path to the data
    file_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\Simulations\num_of_basis_functions\lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_6_basis2_40110lmbda_min-6lmbda_max008Oct24.pkl"
    
    # Create simulator and run
    simulator = RegularizationSimulator(config, file_path)
    results_df = simulator.run_simulation()
    
    print(f"Simulation completed. Results shape: {results_df.shape}")
    return results_df

if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    results = main()