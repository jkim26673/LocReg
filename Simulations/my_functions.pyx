# my_functions.pyx
# cython: language_level=3
import numpy as np
cimport numpy as cnp
cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import cvxpy as xpy  # Regular import, not cimport
# Cython directives
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
from scipy.stats import wasserstein_distance, entropy
from Utilities_functions.discrep_L2 import discrep_L2
from Utilities_functions.GCV_NNLS import GCV_NNLS
from Utilities_functions.Lcurve import Lcurve
import pandas as pd
import cvxpy as cp
from scipy.linalg import svd
from regu.csvd import csvd
from regu.discrep import discrep
from Simulations.Ito_LocReg import Ito_LocReg
from Simulations.Ito_LocReg import *
from Utilities_functions.pasha_gcv import Tikhonov
from regu.l_curve import l_curve
from tqdm import tqdm
from Utilities_functions.tikhonov_vec import tikhonov_vec
import mosek
from ItoLocRegConst import LocReg_Ito_C,LocReg_Ito_C_2,LocReg_Ito_C_4
from regu.nonnegtik_hnorm import nonnegtik_hnorm
import multiprocess as mp
from multiprocessing import Pool, freeze_support
from multiprocessing import set_start_method
import functools
from datetime import date
import random
import cProfile
import pstats
import os
from datetime import date
from cython.parallel import prange
# cython: language_level=3
@cython.boundscheck(False)
@cython.wraparound(False)

# Function to create a result folder
def create_result_folder(string: str, SNR: float, lam_ini_val: float, dist_type: str) -> str:
    cdef str folder_name
    folder_name = f"{cwd_full}/{string}_{date.today()}_SNR_{SNR}_lamini_{lam_ini_val}_dist_{dist_type}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    return folder_name

# Function to minimize objective function
def minimize_OP(cnp.ndarray Alpha_vec, cnp.ndarray L, cnp.ndarray  data_noisy, cnp.ndarray G, int nT2, cnp.ndarray g):
    cdef int j
    cdef int alpha_len = len(Alpha_vec)
    # Prepare output arrays
    cdef cnp.ndarray[cnp.float64_t, ndim=2] OP_x_lc_vec = np.zeros((nT2, alpha_len), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] OP_rhos = np.zeros((alpha_len,), dtype=np.float64)
    for j in (range(len(Alpha_vec))):
        try:
            # Fallback to nonnegtik_hnorm
            sol, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j], '0', nargin=4)
        except Exception as e:
            print(f"Error in nonnegtik_hnorm: {e}")
            # If both methods fail, solve using cvxpy
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

        OP_x_lc_vec[:, j] = sol
        OP_rhos[j] = np.linalg.norm(OP_x_lc_vec[:,j] - g, 2)**2

    OP_log_err_norm = np.log10(OP_rhos)
    min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)
    min_x = Alpha_vec[min_index[0]]
    min_z = np.min(OP_log_err_norm)
    OP_min_alpha1 = min_x
    OP_min_alpha1_ind = min_index[0]
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    return f_rec_OP_grid, OP_min_alpha1

# Function to calculate T2mu
def calc_T2mu(cnp.ndarray rps):
    cdef int nrps = len(rps)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] mps = rps / 2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] T2_left = 40 * np.ones(nrps, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] T2_mid = T2_left * mps
    cdef cnp.ndarray[cnp.float64_t, ndim=1] T2_right = T2_left * rps
    T2mu = np.column_stack((T2_left, T2_right))
    return T2mu

# Function to calculate sigma_i
def calc_sigma_i(int iter_i, cnp.ndarray diff_sigma):
    return diff_sigma[iter_i, :]

# Function to calculate rps_val
def calc_rps_val(int iter_j, cnp.ndarray rps):
    return rps[iter_j]

# Function to calculate difference in sigma
def calc_diff_sigma(int nsigma):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unif_sigma = np.linspace(2, 5, nsigma)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] diff_sigma = np.column_stack((unif_sigma, 3 * unif_sigma))
    return unif_sigma, diff_sigma

# Function to load Gaussian information
def load_Gaus(dict Gaus_info):
    T2 = Gaus_info['T2'].flatten()
    TE = Gaus_info['TE'].flatten()
    A = Gaus_info['A']
    Lambda = Gaus_info['Lambda'].reshape(-1, 1)
    n, m = A.shape
    SNR = Gaus_info['SNR']  # Ensure this is defined in Gaus_info
    return T2, TE, Lambda, A, m, SNR

# Function to calculate noisy data
def calc_dat_noisy(cnp.ndarray A, cnp.ndarray TE, cnp.ndarray IdealModel_weighted,  float SNR):
    dat_noiseless = A @ IdealModel_weighted
    noise = np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE))
    dat_noisy = dat_noiseless + noise
    return dat_noisy, noise

# Function to get weighted ideal model
def get_IdealModel_weighted(int iter_j, int m, int npeaks, 
                             cnp.ndarray T2, 
                             cnp.ndarray T2mu, 
                             cnp.ndarray sigma_i):
    T2mu_sim = T2mu[iter_j, :]
    p = np.array([normsci.pdf(T2, mu, sigma) for mu, sigma in zip(T2mu_sim, sigma_i)])
    IdealModel_weighted = p.T @ f_coef / npeaks  # Ensure f_coef is defined
    return IdealModel_weighted

# Function to calculate L2 error
def l2_error(cnp.ndarray IdealModel, 
             cnp.ndarray reconstr):
    true_norm = linalg_norm(IdealModel)
    err = linalg_norm(IdealModel - reconstr) / true_norm
    return err

# Function to calculate L2 error with shift
def l2_error_shift(float gamma, 
                   cnp.ndarray IdealModel, 
                   cnp.ndarray reconstr):
    
    true_norm = linalg_norm(IdealModel)
    shift_reconstr = np.interp(T2 + gamma, T2, reconstr)
    err = linalg_norm(IdealModel - shift_reconstr) / true_norm
    return err

# Function to calculate Wasserstein shift
def wass_shift(cnp.ndarray T2, 
               float gamma, 
               cnp.ndarray IdealModel, 
               cnp.ndarray reconstr):
    emd = wasserstein_distance(T2, T2 + gamma, u_weights=IdealModel, v_weights=reconstr)
    return emd

# Function to calculate Wasserstein error
def wass_error(cnp.ndarray T2, 
               cnp.ndarray IdealModel, 
               cnp.ndarray reconstr):
    emd = wasserstein_distance(T2, T2, u_weights=IdealModel, v_weights=reconstr)
    return emd

# Function to find the minimum beta
def find_min_beta(cnp.ndarray beta_list, 
                  cnp.ndarray metric_list):
    opt_ind = np.argmin(metric_list)
    opt_gam = beta_list[opt_ind]
    opt_err_score = metric_list[opt_ind]
    return opt_gam, opt_err_score

# Function to get scores
def get_scores(float gamma, cnp.ndarray T2, cnp.ndarray g, cnp.ndarray locreg, cnp.ndarray l1, cnp.ndarray l2):
    cdef list kl_scores_list = []
    cdef list l2_rmsscores_list = []
    cdef list wass_scores_list = []

    shifted_locreg = np.interp(T2 + gamma, T2, locreg)
    shifted_l2 = np.interp(T2 + gamma, T2, l2)
    shifted_l1 = np.interp(T2 + gamma, T2, l1)

    wass_scores_gamma = wass_shift(T2, gamma, g, locreg)

    return wass_scores_gamma

# Cython function to generate estimates


# Function for the Wasserstein shift plots
cpdef get_wass_shift_plots(int iter_sim, int iter_sigma, int iter_rps):
    
    plt.figure(figsize=(12.06, 4.2))
    # First subplot
    plt.subplot(1, 3, 1)
    plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
    plt.plot(T2, shift_w_LR, linestyle=':', linewidth=3, color='red', 
             label=f'LocReg {lam_ini_val} Wass. Shift (Wass. Error: {round(err_LR_w_shift, 3)})')
    plt.plot(T2, shift_w_oracle, linestyle='-.', linewidth=3, color='gold', 
             label=f'Oracle Wass. Shift (Wass. Error: {round(err_oracle_w_shift, 3)})')
    plt.plot(T2, shift_w_DP, linewidth=3, color='green', 
             label=f'DP Wass. Shift (Wass. Error: {round(err_DP_w_shift, 3)})')
    plt.plot(T2, shift_w_GCV, linestyle='--', linewidth=3, color='blue', 
             label=f'GCV Wass. Shift (Wass. Error: {round(err_GCV_w_shift, 3)})')
    plt.plot(T2, shift_w_LC, linestyle='-.', linewidth=3, color='purple', 
             label=f'L-curve Wass. Shift (Wass. Error: {round(err_LC_w_shift, 3)})')
    plt.legend(fontsize=10, loc='best')
    plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
    plt.ylim(0, np.max(IdealModel_weighted) * 1.15)

    # Second subplot
    plt.subplot(1, 3, 2)
    plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
    plt.plot(TE, A @ shift_w_LR, linestyle=':', linewidth=3, color='red', label=f'LocReg Wass. Shift {lam_ini_val}')
    plt.plot(TE, A @ shift_w_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle Wass. Shift')
    plt.plot(TE, A @ shift_w_DP, linewidth=3, color='green', label='DP Wass. Shift')
    plt.plot(TE, A @ shift_w_GCV, linestyle='--', linewidth=3, color='blue', label='GCV Wass. Shift')
    plt.plot(TE, A @ shift_w_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve Wass. Shift')
    plt.legend(fontsize=10, loc='best')
    plt.xlabel('TE', fontsize=20, fontweight='bold')
    plt.ylabel('Intensity', fontsize=20, fontweight='bold')

    # Third subplot
    plt.subplot(1, 3, 3)
    plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
    plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
    plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
    plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')
    plt.semilogy(T2, lambda_oracle * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')

    plt.legend(fontsize=10, loc='best')
    plt.xlabel('T2', fontsize=20, fontweight='bold')
    plt.ylabel('Lambda', fontsize=20, fontweight='bold')

    plt.tight_layout()
    save_plot(iter_sim, iter_sigma, iter_rps, "wass_shifts")

# Function for the L2 shift plots
cpdef get_L2_shift_plots(int iter_sim, int iter_sigma, int iter_rps):
    
    plt.figure(figsize=(12.06, 4.2))
    # First subplot
    plt.subplot(1, 3, 1)
    plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
    plt.plot(T2, shift_L2_LR, linestyle=':', linewidth=3, color='red', 
             label=f'LocReg {lam_ini_val} L2 Shift (Rel. L2 Error: {round(err_LR_L2_shift, 3)})')
    plt.plot(T2, shift_L2_oracle, linestyle='-.', linewidth=3, color='gold', 
             label=f'Oracle L2 Shift (Rel. L2 Error: {round(err_oracle_L2_shift, 3)})')
    plt.plot(T2, shift_L2_DP, linewidth=3, color='green', 
             label=f'DP L2 Shift (Rel. L2 Error: {round(err_DP_L2_shift, 3)})')
    plt.plot(T2, shift_L2_GCV, linestyle='--', linewidth=3, color='blue', 
             label=f'GCV L2 Shift (Rel. L2 Error: {round(err_GCV_L2_shift, 3)})')
    plt.plot(T2, shift_L2_LC, linestyle='-.', linewidth=3, color='purple', 
             label=f'L-curve L2 Shift (Rel. L2 Error: {round(err_LC_L2_shift, 3)})')
    plt.legend(fontsize=10, loc='best')
    plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
    plt.ylim(0, np.max(IdealModel_weighted) * 1.15)

    # Second subplot
    plt.subplot(1, 3, 2)
    plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
    plt.plot(TE, A @ shift_L2_LR, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} L2 Shift')
    plt.plot(TE, A @ shift_L2_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle L2 Shift')
    plt.plot(TE, A @ shift_L2_DP, linewidth=3, color='green', label='DP L2 Shift')
    plt.plot(TE, A @ shift_L2_GCV, linestyle='--', linewidth=3, color='blue', label='GCV L2 Shift')
    plt.plot(TE, A @ shift_L2_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve L2 Shift')
    plt.legend(fontsize=10, loc='best')
    plt.xlabel('TE', fontsize=20, fontweight='bold')
    plt.ylabel('Intensity', fontsize=20, fontweight='bold')

    # Third subplot
    plt.subplot(1, 3, 3)
    plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
    plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
    plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
    plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')
    plt.semilogy(T2, lambda_oracle * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')

    plt.legend(fontsize=10, loc='best')
    plt.xlabel('T2', fontsize=20, fontweight='bold')
    plt.ylabel('Lambda', fontsize=20, fontweight='bold')

    plt.tight_layout()
    save_plot(iter_sim, iter_sigma, iter_rps, "L2_shifts")


def generate_estimates_shift(i_param_combo):
    # Extract parameters
    iter_sim, iter_sigma, iter_rps = target_iterator[i_param_combo]
    sigma_i = diff_sigma[iter_sigma, :]

    rps_val = calc_rps_val(iter_rps, rps)

    # Compute Ideal Model
    IdealModel_weighted = get_IdealModel_weighted(iter_rps, m, npeaks, T2, T2mu, sigma_i)

    if not preset_noise:
        dat_noisy, noise = calc_dat_noisy(A, TE, IdealModel_weighted, SNR)
    else:
        dat_noiseless = A @ IdealModel_weighted
        noise = noise_arr[iter_sim, iter_sigma, iter_rps, :]
        dat_noisy = dat_noiseless + np.ravel(noise)
        noisy_data[iter_sim, iter_sigma, iter_rps, :] = dat_noisy
        noiseless_data[iter_sim, iter_sigma, iter_rps, :] = dat_noiseless

    f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)
    f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
    f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)

    f_rec_GCV = f_rec_GCV[:, 0]
    lambda_GCV = np.squeeze(lambda_GCV)

    if lam_ini_val in ["LCurve", "L-Curve"]:
        LRIto_ini_lam = lambda_LC
        f_rec_ini = f_rec_LC
    elif lam_ini_val.lower() == "gcv":
        LRIto_ini_lam = lambda_GCV
        f_rec_ini = f_rec_GCV
    elif lam_ini_val.lower() == "dp":
        LRIto_ini_lam = lambda_DP
        f_rec_ini = f_rec_DP

    f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1 = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, f_rec_ini, gamma_init, maxiter)

    if testing and iter_sigma == 0 and iter_rps == 0:
        meanfrec1 = np.mean(test_frec1)
        meanlam1 = np.mean(test_lam1)
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, test_frec1, linestyle=':', linewidth=3, color='red', label=f'frec1 gamma init {gamma_init:.2f}')
        plt.legend(fontsize=10, loc='best')
        
        plt.subplot(1, 2, 2)
        plt.semilogy(T2, test_lam1 * np.ones(len(T2)), linewidth=3, color='green', label=f'test_lam1 with meanlam1 {meanlam1:.2f}')
        plt.legend(fontsize=10, loc='best')

        print(f"{numiterate} iters for Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}")
        plt.savefig(f'plot_output_gamma_init{gamma_init}.png')

    f_rec_oracle, lambda_oracle = minimize_OP(Lambda, np.eye(A.shape[1]), dat_noisy, A, len(T2), IdealModel_weighted)

    # Normalization
    sum_x = np.sum(f_rec_LocReg_LC)
    f_rec_LocReg_LC /= sum_x
    sum_oracle = np.sum(f_rec_oracle)
    f_rec_oracle /= sum_oracle
    sum_GCV = np.sum(f_rec_GCV)
    f_rec_GCV /= sum_GCV
    sum_LC = np.sum(f_rec_LC)
    f_rec_LC /= sum_LC
    sum_DP = np.sum(f_rec_DP)
    f_rec_DP /= sum_DP

    # Compute relative L2 norm error and Wasserstein distances without shifts
    err_LC_L2 = l2_error(IdealModel_weighted, f_rec_LC)
    err_DP_L2 = l2_error(IdealModel_weighted, f_rec_DP)
    err_GCV_L2 = l2_error(IdealModel_weighted, f_rec_GCV)
    err_oracle_L2 = l2_error(IdealModel_weighted, f_rec_oracle)
    err_LR_L2 = l2_error(IdealModel_weighted, f_rec_LocReg_LC)

    err_LC_w = wass_error(T2, IdealModel_weighted, f_rec_LC)
    err_DP_w = wass_error(T2, IdealModel_weighted, f_rec_DP)
    err_GCV_w = wass_error(T2, IdealModel_weighted, f_rec_GCV)
    err_oracle_w = wass_error(T2, IdealModel_weighted, f_rec_oracle)
    err_LR_w = wass_error(T2, IdealModel_weighted, f_rec_LocReg_LC)

    # Compute Wasserstein scores and L2 errors using list comprehensions
    w_LR_scores = np.array([wass_shift(T2, beta, IdealModel_weighted, f_rec_LocReg_LC) for beta in beta_list])
    w_LC_scores = np.array([wass_shift(T2, beta, IdealModel_weighted, f_rec_LC) for beta in beta_list])
    w_GCV_scores = np.array([wass_shift(T2, beta, IdealModel_weighted, f_rec_GCV) for beta in beta_list])
    w_DP_scores = np.array([wass_shift(T2, beta, IdealModel_weighted, f_rec_DP) for beta in beta_list])
    w_oracle_scores = np.array([wass_shift(T2, beta, IdealModel_weighted, f_rec_oracle) for beta in beta_list])

    L2_LR_scores = np.array([l2_error_shift(beta, IdealModel_weighted, f_rec_LocReg_LC) for beta in beta_list])
    L2_LC_scores = np.array([l2_error_shift(beta, IdealModel_weighted, f_rec_LC) for beta in beta_list])
    L2_GCV_scores = np.array([l2_error_shift(beta, IdealModel_weighted, f_rec_GCV) for beta in beta_list])
    L2_DP_scores = np.array([l2_error_shift(beta, IdealModel_weighted, f_rec_DP) for beta in beta_list])
    L2_oracle_scores = np.array([l2_error_shift(beta, IdealModel_weighted, f_rec_oracle) for beta in beta_list])

    # Find optimal beta
    w_LR_opt_beta, err_LR_w_shift = find_min_beta(beta_list, w_LR_scores)
    w_LC_opt_beta, err_LC_w_shift = find_min_beta(beta_list, w_LC_scores)
    w_GCV_opt_beta, err_GCV_w_shift = find_min_beta(beta_list, w_GCV_scores)
    w_DP_opt_beta, err_DP_w_shift = find_min_beta(beta_list, w_DP_scores)
    w_oracle_opt_beta, err_oracle_w_shift = find_min_beta(beta_list, w_oracle_scores)

    L2_LR_opt_beta, err_LR_L2_shift = find_min_beta(beta_list, L2_LR_scores)
    L2_LC_opt_beta, err_LC_L2_shift = find_min_beta(beta_list, L2_LC_scores)
    L2_GCV_opt_beta, err_GCV_L2_shift = find_min_beta(beta_list, L2_GCV_scores)
    L2_DP_opt_beta, err_DP_L2_shift = find_min_beta(beta_list, L2_DP_scores)
    L2_oracle_opt_beta, err_oracle_L2_shift = find_min_beta(beta_list, L2_oracle_scores)

    # Find shifted reconstructions
    shift_w_LR = np.interp(T2 + w_LR_opt_beta, T2, f_rec_LocReg_LC)
    shift_w_DP = np.interp(T2 + w_DP_opt_beta, T2, f_rec_DP)
    shift_w_oracle = np.interp(T2 + w_oracle_opt_beta, T2, f_rec_oracle)
    shift_w_LC = np.interp(T2 + w_LC_opt_beta, T2, f_rec_LC)
    shift_w_GCV = np.interp(T2 + w_GCV_opt_beta, T2, f_rec_GCV)

    shift_L2_LR = np.interp(T2 + L2_LR_opt_beta, T2, f_rec_LocReg_LC)
    shift_L2_DP = np.interp(T2 + L2_DP_opt_beta, T2, f_rec_DP)
    shift_L2_oracle = np.interp(T2 + L2_oracle_opt_beta, T2, f_rec_oracle)
    shift_L2_LC = np.interp(T2 + L2_LC_opt_beta, T2, f_rec_LC)
    shift_L2_GCV = np.interp(T2 + L2_GCV_opt_beta, T2, f_rec_GCV)

    if iter_sim == 0:
        get_orig_plot(iter_sim, iter_sigma, iter_rps)
        print(f"Finished Unshifted Plots for iteration {iter_sim} sigma {iter_sigma} rps {iter_rps}")
        get_both_shift_plots_GCV(iter_sim, iter_sigma, iter_rps)
        print(f"Finished GCV Compare Plots for iteration {iter_sim} sigma {iter_sigma} rps {iter_rps}")
        get_both_shift_plots_LR(iter_sim, iter_sigma, iter_rps)
        print(f"Finished LocReg Compare Plots for iteration {iter_sim} sigma {iter_sigma} rps {iter_rps}")
        get_wass_shift_plots(iter_sim, iter_sigma, iter_rps)
        print(f"Finished Wass. Shift Plots for iteration {iter_sim} sigma {iter_sigma} rps {iter_rps}")
        get_L2_shift_plots(iter_sim, iter_sigma, iter_rps)
        print(f"Finished L2 Shift Plots for iteration {iter_sim} sigma {iter_sigma} rps {iter_rps}")

    # Create DataFrames for features
    feature_df_L2 = pd.DataFrame(columns=["NR", 'Sigma', 'RPS_val', 'err_DP', "err_LC", "err_LR", "err_GCV", "err_oracle",  'shift_err_DP', "shift_err_LC", "shift_err_LR", "shift_err_GCV", "shift_err_oracle"])
    feature_df_L2.loc[0] = [iter_sim, sigma_i, rps_val, err_DP_L2, err_LC_L2, err_LR_L2, err_GCV_L2, err_oracle_L2, err_DP_L2_shift, err_LC_L2_shift, err_LR_L2_shift, err_GCV_L2_shift, err_oracle_L2_shift]

    feature_df_w = pd.DataFrame(columns=["NR", 'Sigma', 'RPS_val', 'err_DP', "err_LC", "err_LR", "err_GCV", "err_oracle",  'shift_err_DP', "shift_err_LC", "shift_err_LR", "shift_err_GCV", "shift_err_oracle"])
    feature_df_w.loc[0] = [iter_sim, sigma_i, rps_val, err_DP_w, err_LC_w, err_LR_w, err_GCV_w, err_oracle_w, err_DP_w_shift, err_LC_w_shift, err_LR_w_shift, err_GCV_w_shift, err_oracle_w_shift]

    return feature_df_L2, feature_df_w, noise


def parallel_processed(func, bint shift=True):

    if shift:
        with mp.Pool(processes=num_cpus_avail) as pool:
            with tqdm(total=len(target_iterator)) as pbar:
                for estimates_dataframe_L2, estimates_dataframe_w, noisereal in pool.imap_unordered(func, range(len(target_iterator))):
                    lis_L2.append(estimates_dataframe_L2)
                    lis_w.append(estimates_dataframe_w)
                    noise_list.append(noisereal)
                    pbar.update()
        pool.close()
        pool.join()
        
        noise_arr = np.array(noise_list)
        noise_arr = noise_arr.reshape(n_sim, nsigma, nrps, A.shape[0])
        
        return lis_L2, lis_w, noise_arr
    else:
        with mp.Pool(processes=num_cpus_avail) as pool:
            with tqdm(total=len(target_iterator)) as pbar:
                for estimates_dataframe, noisereal in pool.imap_unordered(func, range(len(target_iterator))):
                    lis.append(estimates_dataframe)
                    noise_list.append(noisereal)
                    pbar.update()
        pool.close()
        pool.join()
        
        noise_arr = np.array(noise_list)
        noise_arr = noise_arr.reshape(n_sim, nsigma, nrps, A.shape[0])
        
        return lis, noise_arr
    

# Cython directives
# cython: language_level=3
import numpy as np
cimport numpy as cnp
cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import cvxpy as xpy  # Regular import, not cimport
cimport cython
# Cython directives
# cython: language_level=3
@cython.boundscheck(False)
@cython.wraparound(False)
def LocReg_Ito_mod(cnp.ndarray data_noisy,
                   cnp.ndarray G,
                   double lam_ini,
                   double gamma_init,
                   int maxiter):
    cdef double eps = 1e-2
    cdef int n_cols = G.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] lam_vec = np.full(n_cols, lam_ini, dtype=np.float64)
    cdef double choice_val = 1e-5

    # No cdef here for lam_vec, just take it as an argument
    def minimize(cnp.ndarray lam_vec):
        cdef cnp.ndarray[cnp.float64_t, ndim=2] A
        cdef cnp.ndarray[cnp.float64_t, ndim=1] b
        y = xpy.Variable(n_cols)  # Use cvxpy Variable

        A = G.T @ G + np.diag(lam_vec)
        ep4 = np.ones(G.shape[1]) * eps
        b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)

        cost = xpy.norm(A @ y - b, 'fro')**2
        constraints = [y >= 0]
        problem = xpy.Problem(xpy.Minimize(cost), constraints)
        problem.solve(solver=xpy.MOSEK, mosek_params={
            'MSK_IPAR_INTPNT_MAX_ITERATIONS': '100',
            'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
        }, verbose=False)

        sol = y.value
        sol = np.maximum(sol - eps, 0)

        if sol is None or np.any(np.isnan(sol)):
            print("Solution contains None or NaN values, switching to fallback method")
            return None, A, b
        
        return sol, A, b

    def phi_resid(cnp.ndarray G, cnp.ndarray param_vec, cnp.ndarray data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

    def fixed_point_algo(double gamma, cnp.ndarray lam_vec, double choice_val):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] lam_curr = lam_vec
        cdef cnp.ndarray[cnp.float64_t, ndim=1] f_old = np.ones(n_cols)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] curr_f_rec
        cdef cnp.ndarray[cnp.float64_t, ndim=1] curr_noise
        cdef cnp.ndarray[cnp.float64_t, ndim=1] delta_p
        cdef int k = 1

        f_rec, LHS, _ = minimize(lam_curr)

        while k <= maxiter:
            curr_f_rec, LHS, _ = minimize(lam_curr)
            if curr_f_rec is None or np.any(np.isnan(curr_f_rec)):
                print(f"curr_f_rec returns None after minimization for iteration {k}")
                continue
            
            curr_noise = G @ curr_f_rec - data_noisy
            LHS_sparse = csr_matrix(LHS)
            delta_p = spsolve(LHS_sparse, G.T @ curr_noise)

            prev_norm = np.linalg.norm(delta_p)
            iterationval = 1
            
            while iterationval < 300:
                curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
                curr_noise = G @ curr_f_rec - data_noisy
                delta_p = spsolve(LHS_sparse, G.T @ curr_noise)

                if np.abs(np.linalg.norm(delta_p) / prev_norm - 1) < choice_val:
                    break
                prev_norm = np.linalg.norm(delta_p)
                iterationval += 1

            phi_new = phi_resid(G, curr_f_rec, data_noisy)
            c = 1 / gamma
            lam_new = c * (phi_new / (np.abs(curr_f_rec) + eps))

            if np.linalg.norm(curr_f_rec - f_old) < eps or k == maxiter:
                if k == maxiter:
                    print("Maximum Iteration Reached")
                return curr_f_rec, lam_curr, k
            else:
                lam_curr = lam_new
                f_old = curr_f_rec
                k += 1

    try:
        best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, choice_val)
    except Exception as e:
        print("Error in locreg:", e)
    
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G, zero_vec, data_noisy)
    gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

    best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, choice_val)

    return best_f_rec2, fin_lam2, best_f_rec1, fin_lam1, iternum

@cython.boundscheck(False)
@cython.wraparound(False)
def LocReg_Ito_mod2(cnp.ndarray data_noisy,
                   cnp.ndarray G,
                   double lam_ini,
                   double gamma_init,
                   int maxiter):
    cdef double eps = 1e-2
    cdef int n_cols = G.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] lam_vec = np.full(n_cols, lam_ini, dtype=np.float64)
    cdef double choice_val = 1e-5

    def minimize(cnp.ndarray lam_vec):
        cdef cnp.ndarray[cnp.float64_t, ndim=2] A = np.empty((n_cols, n_cols), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] b = np.empty(n_cols, dtype=np.float64)
        y = xpy.Variable(n_cols)
        # Using Cython's direct assignment to fill matrix A
        A = G.T @ G + np.diag(lam_vec)
        ep4 = np.ones(G.shape[1]) * eps
        b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
        cost = xpy.norm(A @ y - b, 'fro')**2
        constraints = [y >= 0]
        problem = xpy.Problem(xpy.Minimize(cost), constraints)
        problem.solve(solver=xpy.MOSEK, mosek_params={
            'MSK_IPAR_INTPNT_MAX_ITERATIONS': '100',
            'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
        }, verbose=False)

        sol = y.value
        if sol is None or np.any(np.isnan(sol)):
            print("Solution contains None or NaN values, switching to fallback method")
            return None, A, b
        
        return np.clip(sol - eps, 0, None), A, b  # Using np.clip for efficiency

    def phi_resid(cnp.ndarray G, cnp.ndarray param_vec, cnp.ndarray data_noisy):
        return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

    def fixed_point_algo(double gamma, cnp.ndarray lam_vec, double choice_val):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] lam_curr = lam_vec.copy()
        cdef cnp.ndarray[cnp.float64_t, ndim=1] f_old = np.ones(n_cols, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] curr_f_rec
        cdef cnp.ndarray[cnp.float64_t, ndim=1] curr_noise
        cdef cnp.ndarray[cnp.float64_t, ndim=1] delta_p
        cdef int k = 1

        curr_f_rec, LHS, _ = minimize(lam_curr)

        while k <= maxiter:
            curr_f_rec, LHS, _ = minimize(lam_curr)
            if curr_f_rec is None or np.any(np.isnan(curr_f_rec)):
                print(f"curr_f_rec returns None after minimization for iteration {k}")
                continue
            
            curr_noise = G @ curr_f_rec - data_noisy
            LHS_sparse = csr_matrix(LHS)  # Only create this once if LHS doesn't change
            delta_p = spsolve(LHS_sparse, G.T @ curr_noise)

            prev_norm = np.linalg.norm(delta_p)
            iterationval = 1
            
            while iterationval < 300:
                curr_f_rec -= delta_p  # Directly modifying curr_f_rec
                curr_f_rec = np.maximum(curr_f_rec, 0)  # Avoid using np.maximum repeatedly
                curr_noise = G @ curr_f_rec - data_noisy
                delta_p = spsolve(LHS_sparse, G.T @ curr_noise)

                if np.abs(np.linalg.norm(delta_p) / prev_norm - 1) < choice_val:
                    break
                prev_norm = np.linalg.norm(delta_p)
                iterationval += 1

            phi_new = phi_resid(G, curr_f_rec, data_noisy)
            c = 1 / gamma
            lam_new = c * (phi_new / (np.abs(curr_f_rec) + eps))

            if np.linalg.norm(curr_f_rec - f_old) < eps or k == maxiter:
                if k == maxiter:
                    print("Maximum Iteration Reached")
                return curr_f_rec, lam_curr, k
            else:
                lam_curr = lam_new
                f_old = curr_f_rec.copy()  # Use copy to avoid reference issues
                k += 1

    try:
        best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, choice_val)
    except Exception as e:
        print("Error in locreg:", e)
    
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G, zero_vec, data_noisy)
    gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

    best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, choice_val)

    return best_f_rec2, fin_lam2, best_f_rec1, fin_lam1, iternum