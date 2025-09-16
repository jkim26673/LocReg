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
from regularization.reg_methods.dp.discrep_L2 import discrep_L2
from regularization.reg_methods.gcv.GCV_NNLS import GCV_NNLS
from regularization.reg_methods.lcurve.Lcurve import Lcurve
import pandas as pd
import cvxpy as cp
from scipy.linalg import svd
from regularization.subfunc.csvd import csvd
from regularization.reg_methods.dp.discrep import discrep
from regularization.reg_methods.locreg.LRalgo import LocReg_Ito_mod, LocReg_Ito_mod_deriv, LocReg_Ito_mod_deriv2
from tools.trips_py.pasha_gcv import Tikhonov
from regularization.reg_methods.lcurve import l_curve
from tqdm import tqdm
from regularization.reg_methods.nnls.tikhonov_vec import tikhonov_vec
import mosek
import seaborn as sns
from regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm
import multiprocess as mp
from multiprocessing import Pool, freeze_support
from multiprocessing import set_start_method
import functools
from datetime import date
import random
import cProfile
import pstats
from sim_scripts.peak_resolution.resolutionanalysis import find_min_between_peaks, check_resolution
import logging
import time
from scipy.stats import wasserstein_distance
import matplotlib.ticker as ticker  # Add this import
from regularization.reg_methods.upen.upencode import upen_param_setup, upen_setup

# Configure logging
logging.basicConfig(
    filename='my_script.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
print("setting license path")
# mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
# mosek_license_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\mosek\mosek.lic"
# mosek_license_path = f"C:\Users\kimjosy\Downloads\mosek.lic"
mosek_license_path = r"C:\Users\kimjosy\Downloads\mosek\mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# parent = os.path.dirname(os.path.abspath(''))
# sys.path.append(parent)
# cwd = os.getcwd()

# cwd_temp = os.getcwd()
# base_file = 'LocReg_Regularization-1'
# cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

# pat_tag = "MRR"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
# series_tag = "SpanRegFig"
# simulation_save_folder = f"SimulationSets/{pat_tag}/{series_tag}"
# # cwd_full = cwd_cut + output_folder + lam_ini
# cwd_full = cwd_cut + simulation_save_folder 

# #Number of simulations and SNR:
# n_sim = 1
# #n_sim 50-100
# SNR_value = 1000

# #Hyperparameters and Global Parameters
# ###Plotting hyperparameters
# npeaks = 2
# nsigma = 5
# f_coef = np.ones(npeaks)
# rps = np.linspace(1.1, 4, 5).T
# # rps = np.array([1.30])
# nrps = len(rps)

# ####Showing Plots
# show = 1

# ####Parallel processing
# parallel = False

# ####error metric
# err_type = "Wass. Score"
# # err_type = "Rel. L2 Norm"

# ####Resolution peak analysis
# peak_test_true = True
# if peak_test_true == True:
#     peak_test = []

# ####Shifting distribution by a constant beta value to find optimal error
# testing = False
# shift_beta = False
# if shift_beta == True:
#     beta_list = np.linspace(-100,100,1000)

# ###LocReg hyperparameters
# eps1 = 1e-2
# ep_min = 1e-2
# eps_cut = 1.2
# eps_floor = 1e-4
# exp = 0.5
# feedback = True
# lam_ini_val = "LCurve"
# # dist_type = f"narrowL_broadR_testing_new_noisearr_parallel_nsim{n_sim}"
# dist_type = f"narrowL_broadR_parallel_nsim{n_sim}_SNR_{SNR_value}_errtype_{err_type}"
# gamma_init = 0.5

# #Load Data File
# # file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_6_basis2_40110lmbda_min-6lmbda_max008Oct24.pkl"
# file_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\Simulations\num_of_basis_functions\lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_6_basis2_40110lmbda_min-6lmbda_max008Oct24.pkl"
# Gaus_info = np.load(file_path, allow_pickle=True)
# TEtest = Gaus_info["TE"]
# T2test = Gaus_info["T2"]
# # print("TE", TEtest[0])
# # print("TE", TEtest[-1])
# # print("T2", T2test[0])
# # print("T2", T2test[-1])
# print(f"File loaded from: {file_path}")
# A = Gaus_info["A"]
# n, m = Gaus_info['A'].shape
# print("Gaus_info['A'].shape", Gaus_info['A'].shape)

# reg_param_lb = -6
# reg_param_ub = 0
# N_reg = 50
# #Increase from 20-40
# Lambda = np.logspace(reg_param_lb, reg_param_ub, N_reg).reshape(-1,1)

# ###Number of noisy realizations; 20 NR is enough to until they ask for more noise realizations
# #Naming for Data Folder
# date = date.today()
# day = date.strftime('%d')
# month = date.strftime('%B')[0:3]
# year = date.strftime('%y')
# data_path = "SimulationsSets/MRR/SpanRegFig"
# add_tag = ""
# data_head = "est_table"
# data_tag = (f"{data_head}_SNR{SNR_value}_iter{n_sim}_lamini_{lam_ini_val}_dist_{dist_type}_{add_tag}{day}{month}{year}")
# data_folder = (os.getcwd() + f'/{data_path}')
# os.makedirs(data_folder, exist_ok = True)

# #Number of tasks to execute
# target_iterator = [(a,b,c) for a in range(n_sim) for b in range(nsigma) for c in range(nrps)]
# print("target_iterator", len(target_iterator))
# # num_cpus_avail = np.min([len(target_iterator),40])
# # Get the number of CPUs available
# num_cpus_avail = os.cpu_count()


# Adjusting the file paths for compatibility with Windows

# Get the parent directory
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)

# Get the current working directory
cwd = os.getcwd()

# Adjust the base file and create the correct path
base_file = 'LocReg_Regularization-1'
cwd_temp = os.getcwd()
cwd_cut = os.path.join(cwd_temp.split(base_file, 1)[0], base_file)

# Define simulation save folder
pat_tag = "MRR"
series_tag = "SpanRegFig"
simulation_save_folder = os.path.join("SimulationSets", pat_tag, series_tag)

# Full path for saving simulation data
cwd_full = os.path.join(cwd_cut, simulation_save_folder)

# Number of simulations and SNR
n_sim = 10
SNR_value = 1000

# Hyperparameters and Global Parameters
npeaks = 2
nsigma = 5
f_coef = np.ones(npeaks)
rps = np.linspace(1.1, 4, nsigma).T
nrps = len(rps)

# Show Plots
show = 1

# Parallel processing
parallel = False

# Error metric
err_type = "WassScore"

# Resolution peak analysis
peak_test_true = True
if peak_test_true:
    peak_test = []

# Shifting distribution by a constant beta value
testing = False
shift_beta = False
if shift_beta:
    beta_list = np.linspace(-100, 100, 1000)

# LocReg hyperparameters
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
exp = 0.5
feedback = True
lam_ini_val = "GCV"

# Distribution type for the simulation
dist_type = f"narrowL_broadR_parallel_nsim{n_sim}_SNR_{SNR_value}_errtype_{err_type}_compare1st2ndDeriv_UPEN"

# Initial gamma value
gamma_init = 0.5

# Load Data File (adjusted for Windows path)
file_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\Simulations\num_of_basis_functions\lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_6_basis2_40110lmbda_min-6lmbda_max008Oct24.pkl"

# Load the Gaussian data
Gaus_info = np.load(file_path, allow_pickle=True)
TEtest = Gaus_info["TE"]
T2test = Gaus_info["T2"]

A = Gaus_info["A"]
n, m = Gaus_info['A'].shape

# Define the regularization parameter range
reg_param_lb = -6
reg_param_ub = 0
N_reg = 50
Lambda = np.logspace(reg_param_lb, reg_param_ub, N_reg).reshape(-1, 1)

# Date information for the file name
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')

# Data folder path and creation (adjusted for Windows path)
data_path = os.path.join("SimulationsSets", "MRR", "SpanRegFig")
add_tag = ""
data_head = "est_table"
data_tag = (f"{data_head}_SNR{SNR_value}_iter{n_sim}_lamini_{lam_ini_val}_dist_{dist_type}_{add_tag}{day}{month}{year}")

# Combine the full path for the data folder
data_folder = os.path.join(os.getcwd(), data_path)
os.makedirs(data_folder, exist_ok=True)

# Number of tasks to execute
target_iterator = [(a, b, c) for a in range(n_sim) for b in range(nsigma) for c in range(nrps)]
print("target_iterator", len(target_iterator))

# Get the number of CPUs available
num_cpus_avail = os.cpu_count()


####Predefine noise path
preset_noise = False
# noise_file_path = "SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_narrowL_broadR_testing3_10Oct24noise_arr.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-11_SNR_1000_lamini_LCurve_dist_narrowL_broadR_testing_parallel_nsim2/est_table_SNR1000_iter2_lamini_LCurve_dist_narrowL_broadR_testing_parallel_nsim2_11Oct24noise_arr.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-11_SNR_1000_lamini_LCurve_dist_narrowL_broadR_testing_new_noisearr_noparallel_nsim2/est_table_SNR1000_iter2_lamini_LCurve_dist_narrowL_broadR_testing_new_noisearr_noparallel_nsim2_11Oct24noise_arr.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-14_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim5_SNR_1000/est_table_SNR1000_iter5_lamini_LCurve_dist_narrowL_broadR_parallel_nsim5_SNR_1000_14Oct24noise_arr.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-14_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_1000testing/est_table_SNR1000_iter1_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_1000testing_14Oct24noise_arr.npy"
noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-22_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Wass. Score/est_table_SNR1000_iter10_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Wass. Score_22Oct24noise_arr.npy"
noisy_data = np.zeros((n_sim, nsigma, nrps, n))
noiseless_data = np.zeros((n_sim, nsigma, nrps, n))

std_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-22_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Wass. Score/est_table_SNR1000_iter10_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Wass. Score_22Oct24stdnoise_data.npy"
stdnoise_data = np.zeros((n_sim, nsigma, nrps, n))

if preset_noise == True:
    noise_arr = np.load(noise_file_path, allow_pickle=True)
    stdnoise_data =  np.load(std_file_path, allow_pickle=True)
else:
    noise_arr = np.zeros((n_sim, nsigma, nrps, n))
    stdnoise_data = np.zeros((n_sim, nsigma, nrps, n))

#Functions
# def create_result_folder(string, SNR, lam_ini_val, dist_type):
#     folder_name = f"{cwd_full}/{string}_{date}_SNR_{SNR}_lamini_{lam_ini_val}_dist_{dist_type}"
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#     return folder_name

def create_result_folder(string, SNR, lam_ini_val, dist_type):
    # Get current date in the format 'dd_Mmm_yy'
    # Create the folder name using os.path.join to ensure cross-platform compatibility
    folder_name = os.path.join(cwd_full, f"{string}_{month}{day}{year}_nsim{n_sim}")
    
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    
    return folder_name

#run for 5x5;
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
            OP_rhos[j] = l2_error(g, OP_x_lc_vec[:, j])
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
    min_x = Alpha_vec[min_index][0]
    # print("min_lambda", min_x)
    OP_min_alpha1 = min_x
    # OP_min_alpha1_ind = min_index[0]
    OP_min_alpha1_ind = min_index
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    return f_rec_OP_grid, OP_min_alpha1, min_rhos , min_index

def calc_T2mu(rps):
    mps = rps / 2
    nrps = len(rps)
    T2_left = 40 * np.ones(nrps)
    T2_mid = T2_left * mps
    T2_right = T2_left * rps
    T2mu = np.column_stack((T2_left, T2_right))
    return T2mu

def calc_sigma_i(iter_i, diff_sigma):
    sigma_i = diff_sigma[iter_i, :]
    return sigma_i

def calc_rps_val(iter_j, rps):
    rps_val = rps[iter_j]
    return rps_val

def calc_diff_sigma(nsigma):
    unif_sigma = np.linspace(2, 5, nsigma).T
    diff_sigma = np.column_stack((unif_sigma, 3 *unif_sigma))
    return unif_sigma, diff_sigma

def load_Gaus(Gaus_info):
    n, m = Gaus_info['A'].shape
    # T2 = Gaus_info['T2'].flatten()
    T2 = np.linspace(10,200,m)
    TE = Gaus_info['TE'].flatten()
    A = Gaus_info['A']
    # Lambda = Gaus_info['Lambda'].reshape(-1,1)
    # Lambda = np.append(0, Lambda)
    # Lambda = np.append(0, np.logspace(-6,-1,20)).reshape(-1,1)
    SNR = SNR_value
    return T2, TE, A, m,  SNR

# def calc_dat_noisy(A, TE, IdealModel_weighted, SNR):
#     dat_noiseless = A @ IdealModel_weighted
#     # noise = np.column_stack([np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE), 1)]) 
#     SD_noise =  np.max(np.abs(dat_noiseless)) / SNR 
#     noise = np.random.normal(0, SD_noise, size=dat_noiseless.shape)
#     # print("noise", noise)
#     # noise = np.max(np.abs(dat_noiseless)) / SNR
#     # stdnoise = np.max(np.abs(dat_noiseless)) / SNR 
#     # noise  = np.ravel(noise)
#     dat_noisy = dat_noiseless + noise
#     return dat_noisy, noise, SD_noise

def calc_dat_noisy(A, TE, IdealModel_weighted, SNR, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dat_noiseless = A @ IdealModel_weighted  # Compute noiseless data
    SD_noise = np.max(np.abs(dat_noiseless)) / SNR  # Standard deviation of noise
    noise = np.random.normal(0, SD_noise, size=dat_noiseless.shape)  # Add noise
    dat_noisy = dat_noiseless + noise
    return dat_noisy, noise, SD_noise

def get_IdealModel_weighted(iter_j, m, npeaks, T2, T2mu, sigma_i):
    p = np.zeros((npeaks, m))
    T2mu_sim = T2mu[iter_j, :]
    p = np.array([normsci.pdf(T2, mu, sigma) for mu, sigma in zip(T2mu_sim, sigma_i)])
    IdealModel_weighted = (p.T @ f_coef) / npeaks
    return IdealModel_weighted

def l2_error(IdealModel,reconstr):
    true_norm = linalg_norm(IdealModel)
    err = linalg_norm(IdealModel - reconstr) / true_norm
    return err

def wass_error(IdealModel,reconstr):
    true_norm = linalg_norm(IdealModel)
    #check the absolute errors pattern vs SNRs.
    # err = wasserstein_distance(IdealModel,reconstr)/true_norm
    err = wasserstein_distance(IdealModel,reconstr)
    return err

def generate_estimates(i_param_combo, seed=None):
    def plot(iter_sim, iter_sigma, iter_rps):
        plt.figure(figsize=(17, 8))
        # Plotting the first subplot
        # plt.subplot(1, 3, 1) 
        # plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        # plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} (Error: {"{:.2e}".format(err_LR)})')
        # plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle (Error: {"{:.2e}".format(err_oracle)})')
        # plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {"{:.2e}".format(err_DP)})')
        # plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {"{:.2e}".format(err_GCV)})')
        # plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Error: {"{:.2e}".format(err_LC)})')
        errors = {'LocReg': err_LR, 'Oracle': err_oracle, 'DP': err_DP, 'GCV': err_GCV, 'L-curve': err_LC, "UPEN": err_upen, 'LocReg_1st_Der': err_LR1, 'LocReg_2nd_Der': err_LR2}
        min_method = min(errors, key=errors.get)
        # Modify the plot labels to include a star next to the method with the lowest error
        plt.subplot(1, 3, 1)
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        locreg_label = f'LocReg {lam_ini_val} (Error: {"{:.2e}".format(err_LR)})'
        if min_method == 'LocReg':
            locreg_label += ' *'
        plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=locreg_label)

        locreg_label1 = f'LocReg 1st Deriv {lam_ini_val} (Error: {"{:.2e}".format(err_LR1)})'
        if min_method == 'LocReg_1st_Der':
            locreg_label1 += ' *'
        plt.plot(T2, f_rec_LocReg_LC1, linestyle=':', linewidth=3, color='green', label=locreg_label1)

        locreg2_label = f'LocReg 2nd Deriv {lam_ini_val} (Error: {"{:.2e}".format(err_LR2)})'
        if min_method == 'LocReg_2nd_Der':
            locreg2_label += ' *'
        plt.plot(T2, f_rec_LocReg_LC2, linestyle=':', linewidth=3, color='cyan', label=locreg2_label)

        oracle_label = f'Oracle (Error: {"{:.2e}".format(err_oracle)})'
        if min_method == 'Oracle':
            oracle_label += ' *'
        plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=oracle_label)
        # dp_label = f'DP (Error: {"{:.2e}".format(err_DP)})'
        # if min_method == 'DP':
        #     dp_label += ' *'
        # plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=dp_label)
        gcv_label = f'GCV (Error: {"{:.2e}".format(err_GCV)})'
        if min_method == 'GCV':
            gcv_label += ' *'
        plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=gcv_label)
        plt.legend(fontsize=10, loc='best')
        # lc_label = f'LCurve (Error: {"{:.2e}".format(err_LC)})'
        # if min_method == 'LC':
        #     lc_label += ' *'
        # plt.plot(T2, f_rec_LC, linestyle='--', linewidth=3, color='purple', label=lc_label)
        upen_label = f'UPEN (Error: {"{:.2e}".format(err_upen)})'
        if min_method == 'UPEN':
            upen_label += ' *'

        plt.plot(T2, upen_sol, linestyle='--', linewidth=3, color='purple', label=upen_label)
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
        plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
        ymax = np.max(IdealModel_weighted) * 1.15
        plt.ylim(0, ymax)

        # Plotting the second subplot
        plt.subplot(1, 3, 2)
        plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label= f'LocReg {lam_ini_val}')
        plt.plot(TE, A @ f_rec_LocReg_LC1, linestyle=':', linewidth=3, color='green', label= f'LocReg 1stDer {lam_ini_val}')
        plt.plot(TE, A @ f_rec_LocReg_LC2, linestyle=':', linewidth=3, color='cyan', label= f'LocReg 2ndDer {lam_ini_val}')
        plt.plot(TE, A @ f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle')
        # plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
        plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
        # plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
        plt.plot(TE, A @ upen_sol, linestyle='-.', linewidth=3, color='purple', label='UPEN')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('TE', fontsize=20, fontweight='bold')
        plt.ylabel('Intensity', fontsize=20, fontweight='bold')
        
        plt.subplot(1, 3, 3)
        # plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
        plt.semilogy(T2, (lambda_GCV) * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
        # plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
        plt.semilogy(T2, (lambda_locreg_LC) * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')
        plt.semilogy(T2, (lambda_locreg_LC1) * np.ones(len(T2)), linestyle=':', linewidth=3, color='green', label=f'LocReg 1stDeriv {lam_ini_val}')
        plt.semilogy(T2, (lambda_locreg_LC2) * np.ones(len(T2)), linestyle=':', linewidth=3, color='cyan', label=f'LocReg 2ndDeriv {lam_ini_val}')
        plt.semilogy(T2, (lambda_oracle) * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')
        plt.semilogy(T2, upen_lams * np.ones(len(T2)), linestyle='-.', linewidth=3, color='purple', label='UPEN')
        # plt.plot(T2, (lambda_GCV) * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
        # plt.plot(T2, (lambda_locreg_LC) * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')
        # plt.plot(T2, (lambda_oracle) * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')
        # plt.plot(T2, upen_lams * np.ones(len(T2)), linestyle='-.', linewidth=3, color='purple', label='UPEN')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2', fontsize=20, fontweight='bold')
        plt.ylabel('Lambda', fontsize=20, fontweight='bold')
        plt.tight_layout()
        # plt.ylim(bottom=0, top=2)
        plt.savefig(os.path.join(file_path_final, f"Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}.png"))
        plt.close() 
    # Identify which sigma and rps for a given n_sim iteration
    if parallel == True:
        iter_sim, iter_sigma, iter_rps = target_iterator[i_param_combo]
    else:
        iter_sim, iter_sigma, iter_rps = i_param_combo 
    # L = np.eye(A.shape[1])
    sigma_i = diff_sigma[iter_sigma, :]
    rps_val = calc_rps_val(iter_rps, rps)

    # Generate Ground Truth and add random noise or specific noise array
    IdealModel_weighted = get_IdealModel_weighted(iter_rps, m, npeaks, T2, T2mu, sigma_i)
    dT = T2[-1]/m
    # sum_x = np.sum(IdealModel_weighted) * dT
    # IdealModel_weighted = IdealModel_weighted / sum_x
    if preset_noise == False:
        dat_noisy,noise, stdnoise = calc_dat_noisy(A, TE, IdealModel_weighted, SNR, seed)
    else:
        dat_noiseless = A @ IdealModel_weighted
        try:
            noise = noise_arr[iter_sim, iter_sigma, iter_rps,:]
        except IndexError:
            print("Preset Noise Array Doesn't Match Current Number of Simulations")
        dat_noisy = dat_noiseless + np.ravel(noise)
        noisy_data[iter_sim, iter_sigma, iter_rps,:] = dat_noisy
        noiseless_data[iter_sim, iter_sigma, iter_rps,:] = dat_noiseless
        stdnoise = stdnoise_data[iter_sim,iter_sigma,iter_rps,:]
        stdnoise = stdnoise[0]


    # print("noise", noise)
    # area = np.trapz(IdealModel_weighted, T2)
    # if np.isclose(area, 1.0):
    #     print("The curve is normalized.")
    # else:
    #     print(f"The curve is not normalized (area = {area}).")
    #Recovery using Regularization Based Methods
    f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda, stdnoise)
    f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
    f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
    f_rec_GCV = f_rec_GCV[:, 0]
    lambda_GCV = np.squeeze(lambda_GCV)
    f_rec_oracle, lambda_oracle, min_rhos , min_index = minimize_OP(Lambda, dat_noisy, A, len(T2), IdealModel_weighted)

    if lam_ini_val == "LCurve" or lam_ini_val == "L-Curve":
        LRIto_ini_lam = lambda_LC
        f_rec_ini = f_rec_LC
    elif lam_ini_val == "GCV" or lam_ini_val == "gcv":
        LRIto_ini_lam = lambda_GCV
        f_rec_ini = f_rec_GCV
    elif lam_ini_val == "DP" or lam_ini_val == "dp":
        LRIto_ini_lam = lambda_DP
        f_rec_ini = f_rec_DP

    # maxiter = 500
    maxiter = 50
    LRIto_ini_lam =lambda_GCV
    f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)

    f_rec_LocReg_LC1, lambda_locreg_LC1, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)

    f_rec_LocReg_LC2, lambda_locreg_LC2, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv2(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)

    result = upen_param_setup(TE, T2, A, dat_noisy)
    upen_sol, _ ,_ , upen_lams= upen_setup(result, dat_noisy, LRIto_ini_lam, True)
    
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
    sum_x1 = np.trapz(f_rec_LocReg_LC1, T2)
    f_rec_LocReg_LC1 = f_rec_LocReg_LC1 / sum_x1

    sum_x2 = np.trapz(f_rec_LocReg_LC2, T2)
    f_rec_LocReg_LC2 = f_rec_LocReg_LC2 / sum_x2

    sum_x = np.trapz(f_rec_LocReg_LC, T2)
    f_rec_LocReg_LC = f_rec_LocReg_LC / sum_x

    sum_oracle = np.trapz(f_rec_oracle, T2)
    f_rec_oracle = f_rec_oracle / sum_oracle
    sum_GCV = np.trapz(f_rec_GCV, T2)
    f_rec_GCV = f_rec_GCV / sum_GCV
    sum_LC = np.trapz(f_rec_LC, T2)
    f_rec_LC = f_rec_LC / sum_LC
    sum_DP = np.trapz(f_rec_DP, T2)
    f_rec_DP = f_rec_DP / sum_DP
    upen_sum = np.trapz(upen_sol, T2)
    upen_sol = upen_sol / upen_sum

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
    f_rec_DP = f_rec_DP.flatten()
    f_rec_LC = f_rec_LC.flatten()
    f_rec_LocReg_LC = f_rec_LocReg_LC.flatten()
    f_rec_LocReg_LC1 = f_rec_LocReg_LC1.flatten()
    f_rec_LocReg_LC2 = f_rec_LocReg_LC2.flatten()
    f_rec_oracle = f_rec_oracle.flatten()
    upen_sol = upen_sol.flatten()

    # Calculate Relative L2 Error
    if err_type == "WassScore":
        err_LC = wass_error(IdealModel_weighted, f_rec_LC)
        err_DP = wass_error(IdealModel_weighted, f_rec_DP)
        err_GCV = wass_error(IdealModel_weighted, f_rec_GCV)
        err_oracle = wass_error( IdealModel_weighted, f_rec_oracle)
        err_LR = wass_error(IdealModel_weighted, f_rec_LocReg_LC)
        err_LR1 = wass_error(IdealModel_weighted, f_rec_LocReg_LC1)
        err_LR2 = wass_error(IdealModel_weighted, f_rec_LocReg_LC2)
        err_upen = wass_error(IdealModel_weighted, upen_sol)
    else:
        err_LC = l2_error(IdealModel_weighted, f_rec_LC)
        err_DP = l2_error(IdealModel_weighted, f_rec_DP)
        err_GCV = l2_error(IdealModel_weighted, f_rec_GCV)
        err_oracle = l2_error( IdealModel_weighted, f_rec_oracle)
        err_LR = l2_error( IdealModel_weighted, f_rec_LocReg_LC)
        err_LR1 = l2_error( IdealModel_weighted, f_rec_LocReg_LC1)
        err_LR2 = l2_error( IdealModel_weighted, f_rec_LocReg_LC2)
        err_upen = l2_error(IdealModel_weighted, upen_sol)

    realresult = 1
    # Assuming you are inside a loop or block where you want to check these conditions
    if not(err_oracle <= err_LC and err_oracle <= err_GCV and err_oracle <= err_DP):
        print("err oracle", err_oracle)
        print("oracle", f_rec_oracle)
        print("err_LC", err_LC)
        print("err_GCV", err_GCV)
        print("err_DP", err_DP)
        print("lambda_oracle", lambda_oracle)
        print("min_rhos", min_rhos)
        print("min_index", min_index)
        realresult = 0
        print("Oracle Error should not be larger than other single parameter methods")

    # Plot a set of 25 reconstructions for simulation 0
    if iter_sim == 0:
        plot(iter_sim, iter_sigma, iter_rps)
        print(f"Finished Plots for iteration {iter_sim} sigma {iter_sigma} rps {iter_rps}")
    else:
        pass
    # Create DataFrame
    if realresult == 1:
        feature_df = pd.DataFrame(columns=["NR", 'Sigma', 'RPS_val', 'err_DP', "err_LC", "err_LR", "err_GCV", "err_oracle", 
                                        "LR_vect", "oracle_vect", "DP_vect", "LC_vect", "GCV_vect"])
        feature_df["NR"] = [iter_sim]
        feature_df["Sigma"] = [sigma_i]
        feature_df["RPS_val"] = [rps_val]
        feature_df["err_DP"] = [err_DP]
        feature_df["err_LC"] = [err_LC]
        feature_df["err_LR"] = [err_LR]
        feature_df["err_LR_1stDer"] = [err_LR1]
        feature_df["err_LR_2ndDer"] = [err_LR2]
        feature_df["err_GCV"] = [err_GCV]
        feature_df["err_oracle"] = [err_oracle]
        feature_df["err_upen"] = [err_upen]
        feature_df["LR_vect"] = [f_rec_LocReg_LC]
        feature_df["LR_vect_1stDer"] = [f_rec_LocReg_LC1]
        feature_df["LR_vect_2ndDer"] = [f_rec_LocReg_LC2]
        feature_df["oracle_vect"] = [f_rec_oracle]
        feature_df["DP_vect"] = [f_rec_DP]
        feature_df["LC_vect"] = [f_rec_LC]
        feature_df["GCV_vect"] = [f_rec_GCV]
        feature_df["upen_vect"] = [upen_sol]
        feature_df["DP_lam"] = [lambda_DP]
        feature_df["LC_lam"] = [lambda_LC]
        feature_df["GCV_lam"] = [lambda_GCV]
        feature_df["upen_lam"] = [upen_lams]
        feature_df["oracle_lam"] = [lambda_oracle]
        feature_df["LR_lam"] = [lambda_locreg_LC]
        feature_df["LR_lam_1stDer"] = [lambda_locreg_LC1]
        feature_df["LR_lam_2ndDer"] = [lambda_locreg_LC2]

    else:
        print("Skipped because not a good noise realization where Oracle is not the lowest value")
        feature_df = pd.DataFrame(columns=["NR", 'Sigma', 'RPS_val', 'err_DP', "err_LC", "err_LR", "err_GCV", "err_oracle", 
                                        "LR_vect", "oracle_vect", "DP_vect", "LC_vect", "GCV_vect"])
        feature_df["NR"] = [iter_sim]
        feature_df["Sigma"] = [sigma_i]
        feature_df["RPS_val"] = [rps_val]
        feature_df["err_DP"] = [None]
        feature_df["err_LC"] = [None]
        feature_df["err_LR"] = [None]
        feature_df["err_GCV"] = [None]
        feature_df["err_oracle"] = [None]
        feature_df["err_upen"] = [None]
        feature_df["LR_vect"] = [None]
        feature_df["oracle_vect"] = [None]
        feature_df["DP_vect"] = [None]
        feature_df["LC_vect"] = [None]
        feature_df["GCV_vect"] = [None]
        feature_df["upen_vect"] = [None]
        feature_df["DP_lam"] = [lambda_DP]
        feature_df["LC_lam"] = [lambda_LC]
        feature_df["GCV_lam"] = [lambda_GCV]
        feature_df["upen_lam"] = [upen_lams]
        feature_df["oracle_lam"] = [lambda_oracle]
        feature_df["LR_lam"] = [lambda_locreg_LC]
        feature_df["LR_lam_1stDer"] = [lambda_locreg_LC1]
        feature_df["LR_lam_2ndDer"] = [lambda_locreg_LC2]
    return feature_df, iter_sim, iter_sigma, iter_rps, noise, stdnoise

# def generate_random_numbers(seed, size):
#     rng = np.random.default_rng(seed)
#     return rng.random(size)

# def parallel_processed(func, target_iterator, num_cpus_avail, shift=True):
#     # Create a SeedSequence to generate unique seeds for each process
#     seed_seq = np.random.SeedSequence(12345)  # You can set any base seed here
#     child_seeds = seed_seq.spawn(num_cpus_avail)  # Generate `num_cpus_avail` child seeds
    
#     # Placeholder for results
#     # estimates_dataframe = []
#     # noise_arr = np.zeros((len(target_iterator), len(target_iterator), len(target_iterator), 100))  # Adjust size as needed
#     # stdnoise_data = np.zeros_like(noise_arr)

#     # Function to parallelize
#     def worker(index, seed):
#         # Generate random numbers for the current task using the seed
#         random_numbers = generate_random_numbers(seed, 1000)  # Adjust size as needed
#         estimates_dataframe, iter_sim, iter_sigma, iter_rps, noisereal, std_noisereal = func(index, random_numbers)
        
#         return estimates_dataframe, iter_sim, iter_sigma, iter_rps, noisereal, std_noisereal

#     # Parallel processing using multiprocessing pool
#     with mp.Pool(processes=num_cpus_avail) as pool:
#         with tqdm(total=len(target_iterator)) as pbar:
#             for result in pool.starmap(worker, zip(range(len(target_iterator)), child_seeds)):
#                 estimates_dataframe, iter_sim, iter_sigma, iter_rps, noisereal, std_noisereal = result
#                 noise_arr[iter_sim, iter_sigma, iter_rps, :] = noisereal
#                 stdnoise_data[iter_sim, iter_sigma, iter_rps, :] = std_noisereal
#                 estimates_dataframe.append(estimates_dataframe)
#                 pbar.update()
#     return estimates_dataframe, noise_arr, stdnoise_data

# def parallel_processed(generate_estimates):
#     # Create a SeedSequence to generate unique seeds for each process
#     seed_seq = np.random.SeedSequence(12345)  # You can set any base seed here
#     child_seeds = seed_seq.spawn(num_cpus_avail)  # Generate `num_cpus_avail` child seeds
#     # Function to parallelize
#     def worker(index, seed):
#         # Call generate_estimates with the appropriate parameters and unique seed
#         estimates_dataframe, noise, SD_noise = generate_estimates(seed)
#         return estimates_dataframe, noise, SD_noise
#     # Parallel processing using multiprocessing pool
#     with mp.Pool(processes=num_cpus_avail) as pool:
#         with tqdm(total=len(target_iterator)) as pbar:
#             for result in pool.starmap(worker, zip(range(len(target_iterator)), child_seeds)):
#                 estimates_dataframe, iter_sim, iter_sigma, iter_rps, noisereal, std_noisereal = result
#                 noise_arr[iter_sim, iter_sigma, iter_rps, :] = noisereal
#                 stdnoise_data[iter_sim, iter_sigma, iter_rps, :] = std_noisereal
#                 estimates_dataframe.append(estimates_dataframe)
#                 pbar.update()
#     return estimates_dataframe, noise_arr, stdnoise_data

# def parallel_processed(func, shift = True):
#     # Create a SeedSequence to generate unique seeds for each process
#     seed_seq = np.random.SeedSequence(12345)  # You can set any base seed here
#     child_seeds = seed_seq.spawn(num_cpus_avail)  # Generate `num_cpus_avail` child seeds

#     # Function to parallelize
#     def worker(index, seed, target_iter_item):
#         # Unpack the target iterator item (index, iter_sim, iter_sigma, iter_rps)
#         # iter_sim, iter_sigma, iter_rps = target_iter_item
        
#         # Call generate_estimates with the appropriate parameters and unique seed
#         estimates_dataframe, iter_sim, iter_sigma, iter_rps, noise, SD_noise = generate_estimates(target_iter_item, seed)
        
#         # Return all the necessary data: estimates, noise, stdnoise
#         return estimates_dataframe, iter_sim, iter_sigma, iter_rps, noise, SD_noise
    
#     # Parallel processing using multiprocessing pool
#     with mp.Pool(processes=num_cpus_avail) as pool:
#         # Use tqdm for progress bar and iterate through the target_iterator
#         with tqdm(total=len(target_iterator)) as pbar:
#             # Mapping target_iterator to workers with the seed and target parameters
#             # results = pool.starmap(worker, [(i, child_seeds[i], target_iterator[i]) for i in range(len(target_iterator))])
#             results = pool.starmap(worker, [(i, child_seeds[i], target_iterator[i]) for i in range(len(target_iterator))])

#             for result in results:
#                 estimates_dataframe, iter_sim, iter_sigma, iter_rps, noise, SD_noise = result
#                 # Store the noise and standard deviation of the noise in the respective arrays
#                 noise_arr[iter_sim, iter_sigma, iter_rps, :] = noise
#                 stdnoise_data[iter_sim, iter_sigma, iter_rps, :] = SD_noise
                
#                 # Append the result DataFrame from the worker
#                 estimates_dataframe.append(estimates_dataframe)
                
#                 # Update progress bar
#                 pbar.update()
#     return estimates_dataframe, noise_arr, stdnoise_data
# def compare_heatmap():
#     fig, axs = plt.subplots(2, 2, sharey=True, figsize=(12, 10))
#     plt.subplots_adjust(wspace=0.3, hspace=0.3)
#     # Define tick labels for each method
#     tick_labels = [
#         ['LocReg is better', 'Neutral', 'GCV is better'],  
#         ['LocReg is better', 'Neutral', 'DP is better'],    
#         ['LocReg is better', 'Neutral', 'L-Curve is better'],
#         ['LocReg is better', 'Neutral', 'Oracle is better']
#     ]
#     # Flatten the axes array for easier indexing
#     axs = axs.flatten()
#     x_ticks = rps
#     y_ticks = unif_sigma
#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         im = sns.heatmap(data, cmap='jet', ax=ax, cbar=True, vmin=-0.5, vmax=0.5,
#                         annot=True, fmt=".4f", annot_kws={"size": 12, "weight": "bold"},  
#                         linewidths=0.5, linecolor='black', 
#                         cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8})  # Adjust padding and size
#         ax.set_xlabel('Peak Separation', fontsize=18)
#         ax.set_ylabel('Peak Width', fontsize=18)
#         ax.set_title(title, fontsize=18, pad=20)  # Adjust pad to increase space above title
#         # Set x and y ticks
#         ax.set_xticklabels(x_ticks, rotation=-90)
#         ax.set_yticklabels(y_ticks)
#         # ax.set_xticks(range(len(x_ticks)))  # Ensure ticks correspond to the length of x_ticks
#         # ax.set_xticklabels(x_ticks, rotation=-90)
#         # ax.set_yticks(range(len(y_ticks)))  # Ensure ticks correspond to the length of y_ticks
#         # ax.set_yticklabels(y_ticks)
#         # ax.set_xticks(range(len(x_ticks)))  # Ensure ticks correspond to the length of x_ticks
#         # ax.set_xticklabels(x_ticks, rotation=-90)
#         # ax.set_yticks(range(len(y_ticks)))  # Ensure ticks correspond to the length of y_ticks
#         # ax.set_yticklabels(y_ticks)
#         ax.set_yticks(range(len(y_ticks)))
#         ax.set_yticklabels(y_ticks, fontsize=14)  # Set font size for better visibility

#         # Make sure the y-axis is visible
#         ax.yaxis.set_visible(True)

#         # # Get the colorbar from the heatmap
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([-0.5, 0, 0.5])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)  # Set the tick label size
#     # Add heatmaps and colorbars for each method
#     add_heatmap(axs[0], compare_GCV, tick_labels[0], 'LocReg Error - GCV Error (Rel. L2)', x_ticks, y_ticks)
#     add_heatmap(axs[1], compare_DP, tick_labels[1], 'LocReg Error - DP Error (Rel. L2)', x_ticks, y_ticks)
#     add_heatmap(axs[2], compare_LC, tick_labels[2], 'LocReg Error - L-Curve Error (Rel. L2)', x_ticks, y_ticks)
#     add_heatmap(axs[3], compare_oracle, tick_labels[3], 'LocReg Error - Oracle Error (Rel. L2)', x_ticks, y_ticks)
#     # Adjust tick labels to be on top
#     for ax in axs:
#         ax.xaxis.tick_bottom()
#         ax.xaxis.set_label_position('bottom')
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)
#         # ax.set_yticklabels(ax.get_yticklabels())

#     # Optimize layout to remove whitespace
#     plt.tight_layout()
#     # Save the figure
#     plt.savefig(os.path.join(file_path_final, f"compare_heatmap.png"))
#     print("Saved Comparison Heatmap")
#     plt.close()

#BEST ONE SO FAR
# def compare_heatmap():
#     fig, axs = plt.subplots(2, 2, sharey=True, figsize=(14, 12))
#     plt.subplots_adjust(wspace=0.3, hspace=0.4)

#     # Define tick labels for each method
#     tick_labels = [
#         ['LocReg is better', 'Neutral', 'GCV is better'],
#         ['LocReg is better', 'Neutral', 'DP is better'],
#         ['LocReg is better', 'Neutral', 'L-Curve is better'],
#         ['LocReg is better', 'Neutral', 'Oracle is better']
#     ]
#     axs = axs.flatten()
#     x_ticks = rps
#     y_ticks = unif_sigma
#     if err_type == "Wass. Score":
#         maxLR = np.max(errs_LR)
#         maxLC = np.max(errs_LC)
#         maxoracle = np.max(errs_oracle)
#         maxGCV = np.max(errs_GCV)
#         maxDP = np.max(errs_DP)
#         vmax1 = np.max([maxLR, maxLC, maxGCV, maxDP, maxoracle])
#         vmin1 = -vmax1
#         print("vmax1", vmax1)
#         print("vmin1", vmin1)
#     else:
#         vmin1 = -0.5
#         vmax1 = 0.5

#     if err_type == "Wass. Score":
#         fmt1 = ".5f"
#     else:
#         fmt1 = ".3f"
#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         im = sns.heatmap(data, cmap='jet', ax=ax, cbar=True, vmin=vmin1, vmax=vmax1,
#                           annot=True, fmt=fmt1, annot_kws={"size": 12, "weight": "bold"},  
#                           linewidths=0.5, linecolor='black', 
#                           cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8}, xticklabels= 1, yticklabels= 1)

#         ax.set_xlabel('Peak Separation', fontsize=20)
#         ax.set_ylabel('Peak Width', fontsize=20)
#         ax.set_title(title, fontsize=20, pad=20)
#         x_ticks = np.round(x_ticks, 4)
#         y_ticks = np.round(y_ticks, 4)
#         # Set x and y ticks
#         ax.set_xticklabels(x_ticks, rotation=-90, fontsize=14)
#         ax.set_yticklabels(y_ticks, fontsize=14)
#         # Ensure y-axis is visible
#         ax.yaxis.set_visible(True)
#         # Get the colorbar from the heatmap
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([vmin1, 0, vmax1])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)

#     # Add heatmaps for each method
#     add_heatmap(axs[0], compare_GCV, tick_labels[0], f'LocReg Error - GCV Error ({err_type})', x_ticks, y_ticks)
#     add_heatmap(axs[1], compare_DP, tick_labels[1], f'LocReg Error - DP Error ({err_type})', x_ticks, y_ticks)
#     add_heatmap(axs[2], compare_LC, tick_labels[2], f'LocReg Error - L-Curve Error ({err_type})', x_ticks, y_ticks)
#     add_heatmap(axs[3], compare_oracle, tick_labels[3], f'LocReg Error - Oracle Error ({err_type})', x_ticks, y_ticks)

#     # Ensure y-axis labels show for all plots
#     for ax in axs:
#         ax.xaxis.tick_bottom()
#         ax.xaxis.set_label_position('bottom')
#         # if ax != axs[2] and ax != axs[3]:  # Only show y-ticks for the left plots
#         # ax.set_yticks(np.arange(len(y_ticks)))  # Set the ticks for all axes
#         # ax.set_yticklabels(y_ticks, fontsize=14)  # Set the tick labels for all axes
#         ax.tick_params(labelleft=True)
#         # else:
#         #     ax.yaxis.set_visible(False)
#     # Optimize layout to remove whitespace
#     plt.tight_layout()
#     # Save the figure
#     plt.savefig(os.path.join(file_path_final, f"compare_heatmap.png"))
#     print("Saved Comparison Heatmap")
#     plt.close()


def compare_heatmap():
    fig, axs = plt.subplots(4, 2, sharey=True, figsize=(18, 22))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # Define tick labels for each method
    tick_labels = [
        ['LocReg is better', 'Neutral', 'GCV is better'],
        ['LocReg is better', 'Neutral', 'DP is better'],
        ['LocReg is better', 'Neutral', 'L-Curve is better'],
        ['LocReg is better', 'Neutral', 'Oracle is better'],
        ['LocReg is better', 'Neutral', 'UPEN is better'],
        ['LocReg1stDeriv is better', 'Neutral', 'UPEN is better'],
        ['LocReg2ndDeriv is better', 'Neutral', 'UPEN is better'],
    ]
    axs = axs.flatten()
    x_ticksval = rps
    y_ticksval = unif_sigma

    if err_type == "WassScore":
        maxLR = np.max(errs_LR)
        maxLC = np.max(errs_LC)
        maxoracle = np.max(errs_oracle)
        maxGCV = np.max(errs_GCV)
        maxDP = np.max(errs_DP)
        maxupen = np.max(errs_upen)
        maxLR1 = np.max(errs_LR1)
        maxLR2 = np.max(errs_LR2)
        vmax1 = np.max([maxLR, maxLC, maxGCV, maxDP, maxoracle, maxupen, maxLR1, maxLR2])
        vmin1 = -vmax1
        print("vmax1", vmax1)
        print("vmin1", vmin1)
    else:
        vmin1 = -0.5
        vmax1 = 0.5

    if err_type == "WassScore":
        fmt1 = ".2e"  # Change format to scientific notation
    else:
        fmt1 = ".3e"  # Change format to scientific notation

    def add_heatmap(ax, data, tick_labels, title,  x_ticks, y_ticks):
        im = sns.heatmap(data, cmap='jet', ax=ax, cbar=True, vmin=vmin1, vmax=vmax1,
                          annot=True, fmt=fmt1, annot_kws={"size": 12, "weight": "bold"},  
                          linewidths=0.5, linecolor='black', 
                          cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8}, xticklabels=1, yticklabels=1)

        ax.set_xlabel('Peak Separation', fontsize=20)
        ax.set_ylabel('Peak Width', fontsize=20)
        ax.set_title(title, fontsize=20, pad=20)

        x_ticks = np.round(x_ticks, 4)
        y_ticks = np.round(y_ticks, 4)
        # Set x and y ticks
        ax.set_xticklabels(x_ticks, rotation=-90, fontsize=14)
        ax.set_yticklabels(y_ticks, fontsize=14)
        ax.yaxis.set_visible(True)

        # Get the colorbar from the heatmap
        cbar = im.collections[0].colorbar
        cbar.set_ticks([vmin1, 0, vmax1])
        # cbar.set_ticklabels([f'{vmin1:.1e}', '0', f'{vmax1:.1e}'])  # Use scientific notation for colorbar ticks
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=16)

        # Set scientific formatting for colorbar ticks
        cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

    avg_GCV = f"{np.mean(compare_GCV):.2e}"
    avg_DP = f"{np.mean(compare_DP):.2e}"
    avg_LC = f"{np.mean(compare_LC):.2e}"
    avg_oracle = f"{np.mean(compare_oracle):.2e}"
    avg_upen = f"{np.mean(compare_upen):.2e}"
    avg_LR1_upen = f"{np.mean(compare_LR1_upen):.2e}"
    avg_LR2_upen = f"{np.mean(compare_LR2_upen):.2e}"

    # Add heatmaps for each method
    add_heatmap(axs[0], compare_GCV, tick_labels[0], title = f'LocReg Error - GCV Error ({err_type})\n'+ f'Average GCV Comparison Score: {avg_GCV}',  x_ticks = x_ticksval, y_ticks = y_ticksval)
    add_heatmap(axs[1], compare_DP, tick_labels[1], title = f'LocReg Error - DP Error ({err_type})\n' + f'Average DP Comparison Score: {avg_DP}', x_ticks = x_ticksval, y_ticks = y_ticksval)
    add_heatmap(axs[2], compare_LC, tick_labels[2], title = f'LocReg Error - L-Curve Error ({err_type})\n' +  f'Average LCurve Comparison Score: {avg_LC}', x_ticks = x_ticksval, y_ticks = y_ticksval)
    add_heatmap(axs[3], compare_oracle, tick_labels[3], title = f'LocReg Error - Oracle Error ({err_type})\n' + f'Average Oracle Comparison Score: {avg_oracle}',  x_ticks = x_ticksval, y_ticks = y_ticksval)
    add_heatmap(axs[4], compare_upen, tick_labels[4], title = f'LocReg Error - UPEN Error ({err_type})\n' + f'Average UPEN Comparison Score: {avg_upen}',  x_ticks = x_ticksval, y_ticks = y_ticksval)
    add_heatmap(axs[5], compare_LR1_upen, tick_labels[5], title = f'LocReg1stDeriv Error - UPEN Error ({err_type})\n' + f'Average UPEN Comparison Score: {avg_LR1_upen}',  x_ticks = x_ticksval, y_ticks = y_ticksval)
    add_heatmap(axs[6], compare_LR2_upen, tick_labels[6], title = f'LocReg2ndDeriv Error - UPEN Error ({err_type})\n' + f'Average UPEN Comparison Score: {avg_LR2_upen}',  x_ticks = x_ticksval, y_ticks = y_ticksval)

    # add_heatmap(axs[0], compare_GCV, tick_labels[0], title = f'LocReg Error - GCV Error (Rel. {err_type})',  x_ticks = x_ticksval, y_ticks = y_ticksval)
    # add_heatmap(axs[1], compare_DP, tick_labels[1], title = f'LocReg Error - DP Error (Rel. {err_type})',  x_ticks = x_ticksval, y_ticks = y_ticksval)
    # add_heatmap(axs[2], compare_LC, tick_labels[2], title = f'LocReg Error - L-Curve Error (Rel. {err_type})', x_ticks = x_ticksval, y_ticks = y_ticksval)
    # add_heatmap(axs[3], compare_oracle, tick_labels[3], title = f'LocReg Error - Oracle Error (Rel.{err_type})',  x_ticks = x_ticksval, y_ticks = y_ticksval)
    fig.delaxes(axs[7])

    # Ensure y-axis labels show for all plots
    for ax in axs:
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        ax.tick_params(labelleft=True)

    # Optimize layout to remove whitespace
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(file_path_final, f"compare_heatmap.png"))
    print("Saved Comparison Heatmap")
    plt.close()

def indiv_heatmap():
    # Create 5 subplots arranged in 3 rows and 2 columns (last plot will be empty)
    fig, axs = plt.subplots(4, 2, sharey=True, figsize=(18, 22))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Define tick labels for each method
    tick_labels = [
        ['Low LocReg Error', 'High LocReg Error'],
        ['Low DP Error', 'High DP Error'],  
        ['Low LCurve Error', 'High LCurve Error'],  
        ['Low Oracle Error', 'High Oracle Error'],  
        ['Low GCV Error', 'High GCV Error'],  
        ['Low UPEN Error', 'High UPEN Error'],
        ['Low LocReg1stDer Error', 'High LocReg1stDer Error'],  
        ['Low LocReg2ndDer Error', 'High LocReg2ndDer Error']    
    ]

    # Flatten the axes array for easier indexing
    axs = axs.flatten()
    
    x_ticks = rps
    y_ticks = unif_sigma
    
    maxLR = np.max(errs_LR)
    maxLC = np.max(errs_LC)
    maxoracle = np.max(errs_oracle)
    maxGCV = np.max(errs_GCV)
    maxDP = np.max(errs_DP)
    maxUPEN = np.max(errs_upen)
    maxLR1UPEN = np.max(errs_LR1)
    maxLR2UPEN = np.max(errs_LR2)

    vmax1 = np.max([maxLR, maxLC, maxGCV, maxDP, maxoracle, maxUPEN, maxLR1UPEN, maxLR2UPEN])
    # Inner function to create a heatmap
    def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
        # if dist == "LR":
        #     vmax1 = maxLR
        # elif dist == "LC":
        #     vmax1 = maxLC
        # elif dist == "GCV":
        #     vmax1 = maxGCV
        # elif dist == "DP":
        #     vmax1 = maxDP
        # elif dist == "oracle":
        #     vmax1 = maxoracle
        if err_type == "WassScore":
            fmt1 = ".4f"
        else:
            fmt1 = ".3f"
        im = sns.heatmap(data, cmap='jet', ax=ax, cbar=True, vmin=0, vmax=vmax1,
                         annot=True, fmt=fmt1, annot_kws={"size": 12, "weight": "bold"},  
                         linewidths=0.5, linecolor='black', 
                         cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8}, xticklabels= 1, yticklabels= 1)  # Adjust padding and size
        ax.set_xlabel('Peak Separation', fontsize=20)
        ax.set_ylabel('Peak Width', fontsize=20)
        ax.set_title(title, fontsize=20, pad=20)  # Adjust pad to increase space above title
        # Set x and y ticks
        x_ticks = np.round(x_ticks, 4)
        y_ticks = np.round(y_ticks, 4)
        ax.set_xticklabels(x_ticks, rotation=-90)
        ax.set_yticklabels(y_ticks)
        # Get the colorbar from the heatmap
        cbar = im.collections[0].colorbar
        cbar.set_ticks([0, vmax1])
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=16)  # Set the tick label size
        # Set scientific formatting for colorbar ticks
        cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

    avg_GCV = f"{np.mean(errs_GCV):.2e}"
    avg_DP = f"{np.mean(errs_DP):.2e}"
    avg_LC = f"{np.mean(errs_LC):.2e}"
    avg_oracle = f"{np.mean(errs_oracle):.2e}"
    avg_LR = f"{np.mean(errs_LR):.2e}"
    avg_upen = f"{np.mean(errs_upen):.2e}"
    avg_LR1 = f"{np.mean(errs_LR1):.2e}"
    avg_LR2 = f"{np.mean(errs_LR2):.2e}"

    # Add heatmaps and colorbars for each method
    add_heatmap(axs[0], errs_LR, tick_labels[0], f'LocReg Error ({err_type})\n'+ f'Average Score: {avg_LR}', x_ticks, y_ticks)
    add_heatmap(axs[1], errs_DP, tick_labels[1], f'DP Error ({err_type})\n'+ f'Average Score: {avg_DP}', x_ticks, y_ticks)
    add_heatmap(axs[2], errs_LC, tick_labels[2], f'L-Curve Error ({err_type})\n'+ f'Average Score: {avg_LC}', x_ticks, y_ticks)
    add_heatmap(axs[3], errs_oracle, tick_labels[3], f'Oracle Error ({err_type})\n'+ f'Average Score: {avg_oracle}', x_ticks, y_ticks)
    add_heatmap(axs[4], errs_GCV, tick_labels[4], f'GCV Error ({err_type})\n'+ f'Average Score: {avg_GCV}', x_ticks, y_ticks)
    add_heatmap(axs[5], errs_upen, tick_labels[4], f'UPEN Error ({err_type})\n'+ f'Average Score: {avg_upen}', x_ticks, y_ticks)
    add_heatmap(axs[6], errs_LR1 , tick_labels[0], f'LocReg1stDer Error ({err_type})\n'+ f'Average Score: {avg_LR1}', x_ticks, y_ticks)
    add_heatmap(axs[7], errs_LR2, tick_labels[0], f'LocReg2ndDer Error ({err_type})\n'+ f'Average Score: {avg_LR2}', x_ticks, y_ticks)

    # The 6th subplot is not needed, hide it
    # axs[5].axis('off')
    
    # Adjust tick labels to be on the bottom
    for ax in axs[:7]:
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)
        ax.tick_params(labelleft=True)
    
    # Optimize layout to remove excess whitespace
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(file_path_final, f"indiv_heatmap.png"))
    print("Saved Individual Heatmap")
    plt.close()


def worker_init():
    # Use current_process()._identity to get a unique worker ID for each worker
    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    np.random.seed(worker_id)  # Set a random seed for each worker


def parallel_processed(func, shift = True):
    with mp.Pool(processes = num_cpus_avail, initializer=worker_init) as pool:
        with tqdm(total = len(target_iterator)) as pbar:
            for estimates_dataframe, iter_sim, iter_sigma, iter_rps, noisereal, std_noisereal in pool.imap_unordered(func, range(len(target_iterator))):
                lis.append(estimates_dataframe)
                noise_arr[iter_sim, iter_sigma, iter_rps,:] = noisereal
                stdnoise_data[iter_sim, iter_sigma, iter_rps,:] = std_noisereal
                pbar.update()
        pool.close()
        pool.join()
    return estimates_dataframe, noise_arr, stdnoise_data

if __name__ == '__main__':
    if 'TERM_PROGRAM' in os.environ and os.environ['TERM_PROGRAM'] == 'vscode':
        print("Running in VS Code")
    logging.info("Script started.")
    freeze_support()
    unif_sigma, diff_sigma = calc_diff_sigma(nsigma)
    T2, TE, A, m,  SNR = load_Gaus(Gaus_info)
    print("TE",TE)
    T2mu = calc_T2mu(rps)
    string = "MRR_1D_LocReg_Comparison"
    file_path_final = create_result_folder(string, SNR, lam_ini_val, dist_type)
    print("Finished Assignments...")  
    lis = []
    lis_L2 = []
    lis_w = []
    sigma_rps_labels = []
 
    if parallel == True:
        estimates_dataframe, noise_arr, stdnoise_data = parallel_processed(generate_estimates, shift = True)
        # lis.append(estimates_dataframe)
    else:
        for i in range(n_sim):
            for j in range(nsigma):
                for k in range(nrps):
                    iter = (i,j,k)
                    estimates_dataframe, iter_sim, iter_sigma, iter_rps, noisereal, std_noisereal = generate_estimates(iter) 
                    lis.append(estimates_dataframe)
                    noise_arr[iter_sim, iter_sigma, iter_rps,:] = noisereal
                    stdnoise_data[iter_sim, iter_sigma, iter_rps,:] = std_noisereal
        #check if the shape of arr is correct, add up iterations; 
        # for i in range(n_sim):
        #     for j in range(nsigma):
        #         for k in range(nrps):
        #             iter = (i,j,k)
        #             estimates_dataframe, noisereal = generate_estimates(iter) 
        #             lis.append(estimates_dataframe)
        #             noise_list.append(noisereal)
        # noise_arr = np.array(noise_list)
        # #check if the shape of arr is correct, add up iterations; 
        # noise_arr = noise_arr.reshape(n_sim, nsigma, nrps, A.shape[0])

    print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
    df = pd.concat(lis, ignore_index= True)
    # df.to_pickle(file_path_final + f'/' + data_tag +'.pkl')
    # os.makedirs(file_path_final, exist_ok=True)
    # df.to_pickle(os.path.join(file_path_final, f'{data_tag}.pkl'))
    lastpath = os.path.join(file_path_final, f'{data_tag}.pkl')
    directory_path = os.path.dirname(lastpath)

    # Check if the directory exists and create it if not
    if not os.path.exists(directory_path):
        print(f"The folder {directory_path} does not exist. Creating it now...")
        os.makedirs(directory_path, exist_ok=True)
    else:
        print("directory made")

    # Use a UNC path to bypass Windows path length limitations
    # Now save the DataFrame as a pickle file
    # df = pd.concat(lis, ignore_index=True)
    df.to_pickle(lastpath)

    df['Sigma'] = df['Sigma'].apply(tuple)
    df_sorted = df.sort_values(by=['NR','Sigma', 'RPS_val'], ascending=[True, True, True])
    print("df_sorted", df_sorted)
    num_NRs = df_sorted['NR'].nunique()

    na_count = df_sorted.isna().sum().sum()
    print(f"Total number of NA values in the DataFrame: {na_count}")

    if num_NRs == 1:
        df_sorted.fillna(0)
    else:
        df_sorted.dropna()

    na_count = df_sorted.isna().sum().sum()
    print(f"Total number of NA values in the DataFrame: {na_count}")

    print("df_sorted.shape", df_sorted.shape)
    grouped = df_sorted.groupby(['Sigma', 'RPS_val']).agg({
        'err_DP': 'sum',
        'err_LC': 'sum',
        'err_LR': 'sum',
        'err_GCV': 'sum',
        'err_oracle': 'sum',
        "err_upen": "sum",
        "err_LR_1stDer": "sum",
        "err_LR_2ndDer": "sum"
    })
    # Average the errors
    average_errors = grouped / num_NRs
    print("num_NRs", num_NRs)
    errors = average_errors

    errs_oracle = errors["err_oracle"].to_numpy().reshape(nsigma,nrps)
    errs_oracle = np.array(errs_oracle)
    errs_LC= errors["err_LC"].to_numpy().reshape(nsigma,nrps)
    errs_LC = np.array(errs_LC)
    errs_GCV = errors["err_GCV"].to_numpy().reshape(nsigma,nrps)
    errs_GCV = np.array(errs_GCV)
    errs_DP = errors["err_DP"].to_numpy().reshape(nsigma,nrps)
    errs_DP = np.array(errs_DP)
    errs_LR = errors["err_LR"].to_numpy().reshape(nsigma,nrps)
    errs_LR = np.array(errs_LR)
    errs_upen = errors["err_upen"].to_numpy().reshape(nsigma,nrps)
    errs_upen = np.array(errs_upen)
    errs_LR1 = errors["err_LR_1stDer"].to_numpy().reshape(nsigma,nrps)
    errs_LR1 = np.array(errs_LR1)
    errs_LR2 = errors["err_LR_2ndDer"].to_numpy().reshape(nsigma,nrps)
    errs_LR2 = np.array(errs_LR2)
        
    compare_GCV = errs_LR - errs_GCV
    compare_DP = errs_LR - errs_DP
    compare_LC = errs_LR - errs_LC
    compare_oracle = errs_LR - errs_oracle
    compare_upen = errs_LR - errs_upen
    compare_LR1_upen = errs_LR1 - errs_upen
    compare_LR2_upen = errs_LR2 - errs_upen

    if show == 1:
        compare_heatmap()
        indiv_heatmap()
        if preset_noise == False:
            np.save(file_path_final + f'/' + data_tag + "noise_arr", noise_arr)
            print("noise array saved")
            np.save(file_path_final + f'/' + data_tag + "stdnoise_data", stdnoise_data)
            print("standard dev noise array saved")
        else:
            print("Used preset noise array and std. dev noise array")
            pass
    logging.info("Script completed.")
