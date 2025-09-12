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
from locreg_ito import LocReg_Ito_mod as LocReg_Ito_mod_cy
# from funcs import *

print("setting license path")
mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ["MSK_IPAR_OPTIMIZER"] = 'MSK_OPTIMIZER_INTPNT'
# os.environ["MSK_IPAR_BI_MAX_ITERATIONS "] = "1000"

parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
cwd = os.getcwd()

cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

pat_tag = "MRR"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
series_tag = "SpanRegFig"
simulation_save_folder = f"SimulationSets/{pat_tag}/{series_tag}"
# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + simulation_save_folder 

#Hyperparameters and Global Parameters
preset_noise = True
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_15Aug24noise_arr_modifiedalgo.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_test_21Aug24noise_arr.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_13Aug24noise_arr.txt.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_narrowL_broadR_21Aug24noise_arr.npy"

# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_narrowL_broadR_13Sep24noise_arr.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR300_iter10_lamini_LCurve_dist_narrowL_broadR_26Aug24noise_arr.npy"
noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR100_iter1_lamini_LCurve_dist_narrowL_broadR_30Sep24noise_arr.npy"
testing = False
shift_beta = False
if shift_beta == True:
    beta_list = np.linspace(-100,100,1000)

show = 1
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_15Aug24noise_arr_modifiedalgo.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_13Aug24noise_arr.txt.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_06Aug24noise_arr.txt.npy"

#Number of simulations:
n_sim = 1
SNR_value = 100

###LocReg hyperparameters
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
exp = 0.5
feedback = True
lam_ini_val = "LCurve"
dist_type = "testingcython"
# gamma_init = 5
#gamma_init = 0.5 is best
gamma_init = 0.5

###Plotting hyperparameters
npeaks = 2
# npeaks = 3
nsigma = 5
f_coef = np.ones(npeaks)
rps = np.linspace(1, 4, 5).T


#orig/2bump:
#npeak = 2
# rps = np.linspace(1, 4, 5).T
# nsigma = 5
#single/hat:
#npeak = 1
# rps = np.linspace(1, 1, 5).T
# nsigma = 5

#threebump:
# npeaks = 3
# rps = np.linspace(4, 4, 1).T



# Lambda = Gaus_info['Lambda'].reshape(-1,1)
# Alpha_vec = np.logspace(-6, 0,16)
# Alpha_vec = Gaus_info['Lambda'].reshape(-1,1)
nrps = len(rps)

###SNR Values to Evaluate
# SNR_value = 50
# SNR_value = 200
# SNR_value = 500
# SNR_value = 1000

#Load Data File
# #narrow left, broad right, SNR 1000
# file_path ="/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max0.pkl"

# #narrow left, broad right, SNR 50
# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_50_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max015Aug24.pkl"

#broad left, narrow right, SNR 1000
# file_path ="/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_1000_nrun_20_sigma_min_4_sigma_max_2_basis2_5080lmbda_min-6lmbda_max015Aug24.pkl"
file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_1000_nrun_1_sigma_min_2_sigma_max_6_basis2_10040lmbda_min-6lmbda_max016Aug24.pkl"

# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_1000_nrun_10_sigma_min_2_sigma_max_6_basis2_5020lmbda_min-6lmbda_max019Aug24.pkl"


# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_50_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max031Jul24.pkl"
# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_200_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max0_73124.pkl"
# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max1_073124.pkl"

Gaus_info = np.load(file_path, allow_pickle=True)
print(f"File loaded from: {file_path}")
A = Gaus_info["A"]
n, m = Gaus_info['A'].shape

# print("Gaus_info['A'].shape", Gaus_info['A'].shape)
###Number of noisy realizations; 20 NR is enough to until they ask for more noise realizations

#Naming for Data Folder
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')
data_path = "SimulationsSets/MRR/SpanRegFig"
add_tag = ""
data_head = "est_table"
data_tag = (f"{data_head}_SNR{SNR_value}_iter{n_sim}_lamini_{lam_ini_val}_dist_{dist_type}_{add_tag}{day}{month}{year}")
data_folder = (os.getcwd() + f'/{data_path}')
os.makedirs(data_folder, exist_ok = True)

#Number of tasks to execute
target_iterator = [(a,b,c) for a in range(n_sim) for b in range(nsigma) for c in range(nrps)]
# num_cpus_avail = np.min([len(target_iterator),40])
num_cpus_avail = 100
noisy_data = np.zeros((n_sim, nsigma, nrps, n))
noiseless_data = np.zeros((n_sim, nsigma, nrps, n))

if preset_noise == True:
    noise_arr = np.load(noise_file_path, allow_pickle=True)
    noise_list = []
else:
    noise_arr = np.zeros((n_sim, nsigma, nrps, A.shape[0]))
    noise_list = []

#Functions
def create_result_folder(string, SNR, lam_ini_val, dist_type):
    folder_name = f"{cwd_full}/{string}_{date}_SNR_{SNR}_lamini_{lam_ini_val}_dist_{dist_type}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g):
    OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
    OP_rhos = np.zeros((len(Alpha_vec)))
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
    # OP_min_alpha1 = np.sqrt(OP_min_alpha1)
    return f_rec_OP_grid, OP_min_alpha1

def calc_T2mu(rps):
    #30 change 30 to 40
    mps = rps / 2
    #singlepeak:
    #mps = rps / 1
    #T2_left = 100 * np.ones(nrps)

    """
    nrps = len(rps)
    T2_left = 40 * np.ones(nrps)
    # T2_left = 30 * np.ones(nrps)
    T2_mid = T2_left * mps
    T2_right = T2_left * rps
    """
    nrps = len(rps)
    T2_left = 40 * np.ones(nrps)
    # T2_left = 30 * np.ones(nrps)
    T2_mid = T2_left * mps
    T2_right = T2_left * rps

    #Single peak on the right side of T2 axis:
    # T2_right = 140 * np.ones(nrps)
    # T2_left = T2_right/rps

    # T2_mid = T2_left * mps
    #twopeak
    T2mu = np.column_stack((T2_left, T2_right))
    #threepeak
    # T2mu = np.column_stack((T2_left, T2_mid,  T2_right))
    # T2mu = T2_left

    return T2mu

def calc_sigma_i(iter_i, diff_sigma):
    sigma_i = diff_sigma[iter_i, :]
    return sigma_i

def calc_rps_val(iter_j, rps):
    rps_val = rps[iter_j]
    return rps_val

def calc_diff_sigma(nsigma):
    # unif_sigma = np.linspace(2, 5, nsigma).T
    # unif_sigma = np.linspace(2, 4, nsigma).T

    unif_sigma = np.linspace(2, 5, nsigma).T
    diff_sigma = np.column_stack((unif_sigma, 3 * unif_sigma))
    # diff_sigma = np.column_stack((3 * unif_sigma, unif_sigma, unif_sigma))
    # diff_sigma = np.column_stack((3 * unif_sigma, unif_sigma))

    #hat/single peak dist
    # unif_sigma = np.linspace(2, 4, nsigma).T
    # diff_sigma = np.column_stack((9* unif_sigma, 9*unif_sigma))
    return unif_sigma, diff_sigma

def load_Gaus(Gaus_info):
    T2 = Gaus_info['T2'].flatten()
    TE = Gaus_info['TE'].flatten()
    A = Gaus_info['A']
    Lambda = Gaus_info['Lambda'].reshape(-1,1)
    n, m = Gaus_info['A'].shape
    # SNR = Gaus_info['SNR']
    SNR = SNR_value
    return T2, TE, Lambda, A, m,  SNR

def calc_dat_noisy(A, TE, IdealModel_weighted, SNR):
    dat_noiseless = A @ IdealModel_weighted
    noise = np.column_stack([np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE), 1)]) 
    noise  = np.ravel(noise)
    dat_noisy = dat_noiseless + np.ravel(noise)
    return dat_noisy, noise

def get_IdealModel_weighted(iter_j, m, npeaks, T2, T2mu, sigma_i):
    p = np.zeros((npeaks, m))
    T2mu_sim = T2mu[iter_j, :]
    p = np.array([normsci.pdf(T2, mu, sigma) for mu, sigma in zip(T2mu_sim, sigma_i)])
    IdealModel_weighted = p.T @ f_coef / npeaks
    return IdealModel_weighted

def l2_error(IdealModel,reconstr):
    true_norm = linalg_norm(IdealModel)
    err = linalg_norm(IdealModel - reconstr) / true_norm
    return err

def l2_error_shift(gamma, IdealModel,reconstr):
    true_norm = linalg_norm(IdealModel)
    shift_reconstr = np.interp(T2 + gamma, T2, reconstr)
    err = linalg_norm(IdealModel - shift_reconstr) / true_norm
    return err

# def l2_error_shift(IdealModel,reconstr):
#     true_norm = linalg_norm(IdealModel)
#     err = linalg_norm(IdealModel - reconstr) / true_norm
#     return err

# def kl_div(IdealModel,reconstr):
#     """
#     Calculate the Kullback-Leibler divergence for Loc-Reg, L2 L-curve, and L1 L-curve solutions,
#     compared to the ground truth g.

#     Parameters:
#         locreg (ndarray): Loc-Reg solution.
#         l2 (ndarray): L2 L-curve solution.
#         l1 (ndarray): L1 L-curve solution.
#         g (ndarray): Ground truth.

#     Returns:
#         ndarray: Array containing the KL divergence for Loc-Reg, L2 L-curve, and L1 L-curve.
#     """
#     #compare locreg with ground truth
#     locreg_err = entropy(locreg, g)
#     #compare L2 with ground truth
#     l2_error = entropy(l2, g)
#     #compare L1 with ground truth
#     l1_error = entropy(l1, g)
#     return np.array([locreg_err,l2_error,l1_error])

def wass_shift(T2, gamma, IdealModel,reconstr):
    #IdealModel,reconstr
    emd= wasserstein_distance(T2, T2+gamma, u_weights=IdealModel, v_weights=reconstr)
    # emd_l2 = wasserstein_distance(T2, T2+gamma, u_weights=IdealModel, v_weights=l2)
    # emd_l1 = wasserstein_distance(T2, T2+gamma, u_weights=IdealModel, v_weights=l1)
    #np.array([emd_locreg,emd_l2,emd_l1])
    return emd

def wass_error(T2, IdealModel, reconstr):
    emd= wasserstein_distance(T2, T2, u_weights=IdealModel, v_weights=reconstr)
    return emd

def find_min_beta(beta_list, metric_list):
    opt_ind = np.argmin(metric_list)
    # print("opt_ind", opt_ind)
    opt_gam = beta_list[opt_ind]
    # print("opt_gam", opt_gam)
    opt_err_score = metric_list[opt_ind]
    # print("opt_err_score", opt_err_score)
    return opt_gam, opt_err_score

def get_scores(gamma, T2, g, locreg, l1, l2):
    kl_scores_list = []
    l2_rmsscores_list = []
    wass_scores_list = []

    shifted_locreg = np.interp(T2 + gamma, T2, locreg)
    shifted_l2 = np.interp(T2 + gamma, T2, l2)
    shifted_l1 = np.interp(T2 + gamma, T2, l1)
    # fig, ax = plt.subplots()
    # ax.plot(T2, g, linewidth=2)
    #ax.plot(T2, locreg, linewidth=2)
    #ax.plot(T2, l2, linewidth=2)
    #ax.plot(T2, l1, linewidth=2)
    
    # Calculate the kl_score and l2_rmsscore for this gamma value
    # kl_scores_gamma = kl_div(shifted_locreg, shifted_l2, shifted_l1, g)
    # l2_rmsscores_gamma = l2_rms(shifted_locreg, shifted_l2, shifted_l1, g)
    wass_scores_gamma = wass_shift(T2, gamma, g, locreg, l2, l1)

    # Add the kl_score and l2_rmsscore as text annotations on the plot
    # ax.text(0.7, 0.9, f"KL Score:\nLoc-Reg: {kl_scores_gamma[0]:.4f}\nl2 L-curve: {kl_scores_gamma[1]:.4f}\nl1 L-curve: {kl_scores_gamma[2]:.4f}",
    #         transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # ax.text(0.7, 0.6, f"L2 RMS Score:\nLoc-Reg: {l2_rmsscores_gamma[0]:.4f}\nl2 L-curve: {l2_rmsscores_gamma[1]:.4f}\nl1 L-curve: {l2_rmsscores_gamma[2]:.4f}",
    #         transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # ax.text(0.7, 0.3, f"Wass. Score:\nLoc-Reg: {wass_scores_gamma[0]:.4f}\nl2 L-curve: {wass_scores_gamma[1]:.4f}\nl1 L-curve: {wass_scores_gamma[2]:.4f}",
    #         transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # # Set the legend size
    # ax.legend(['True', 'Loc-Reg', 'l2 L-curve', 'l1 L-curve'], fontsize=10, loc = "upper left")

    # # Adjust the title position
    # ax.set_title(f"Gamma = {gamma}", fontsize=14, y=1.02)
    #fig
    return wass_scores_gamma

# # def wasserstein
# # def 

# # def profile_function(func, *args, **kwargs):
# #     profiler = cProfile.Profile()
# #     profiler.enable()
# #     result = func(*args, **kwargs)
# #     profiler.disable()
# #     stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
# #     stats.print_stats(10)
# #     return result

def generate_estimates(i_param_combo):
    def plot(iter_sim, iter_sigma, iter_rps):
        plt.figure(figsize=(12.06, 4.2))
        # Plotting the first subplot
        plt.subplot(1, 3, 1) 
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} (Error: {round(err_LR,3)})')
        # plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle (Error: {round(err_oracle,3)})')
        # plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {round(err_DP,3)})')
        plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {round(err_GCV,3)})')
        # plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Error: {round(err_LC,3)})')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
        plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
        ymax = np.max(IdealModel_weighted) * 1.15
        plt.ylim(0, ymax)

        # Plotting the second subplot
        plt.subplot(1, 3, 2)
        plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label= f'LocReg {lam_ini_val}')
        # plt.plot(TE, A @ f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle')
        # plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
        plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
        # plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('TE', fontsize=20, fontweight='bold')
        plt.ylabel('Intensity', fontsize=20, fontweight='bold')
        
        plt.subplot(1, 3, 3)
        # plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
        plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
        # plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
        plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')
        # plt.semilogy(T2, lambda_oracle * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')

        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2', fontsize=20, fontweight='bold')
        plt.ylabel('Lambda', fontsize=20, fontweight='bold')

        plt.tight_layout()
        string = "MRR_1D_LocReg_Comparison"
        file_path = create_result_folder(string, SNR, lam_ini_val, dist_type)
        plt.savefig(os.path.join(file_path, f"Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}.png"))
        plt.close() 

    iter_sim, iter_sigma, iter_rps = target_iterator[i_param_combo]
    L = np.eye(A.shape[1])
    sigma_i = diff_sigma[iter_sigma, :]
    rps_val = calc_rps_val(iter_rps, rps)

    # Profile the function calls individually
    IdealModel_weighted = get_IdealModel_weighted(iter_rps, m, npeaks, T2, T2mu, sigma_i)

    if preset_noise == False:
        dat_noisy,noise = calc_dat_noisy( A, TE, IdealModel_weighted, SNR)
    else:
        dat_noiseless = A @ IdealModel_weighted
        # print("noise_arr.shape", noise_arr.shape)
        # print("iter_sim", iter_sim)
        # print("iter_sigma", iter_sigma)
        # print("iter_rps", iter_rps)

        # noise_arr = noise_arr.flatten()
        # noise_arr = noise_arr.reshape(z,y)
        # print("dat_noiseless.shape", dat_noiseless.shape)
        # print("noise_iter shape",noise_arr[])
        noise = noise_arr[iter_sim, iter_sigma, iter_rps,:]
        # print("noise", noise.shape)
        # print("noise.shape", noise.shape)
        dat_noisy = dat_noiseless + np.ravel(noise)
        # print("dat_noiseless.shape", dat_noiseless.shape)

        noisy_data[iter_sim, iter_sigma, iter_rps,:] = dat_noisy
        noiseless_data[iter_sim, iter_sigma, iter_rps,:] = dat_noiseless

        # noise = noise_arr.tolist()
        # noise_arr = np.zeros((n_sim, nsigma, nrps))
    f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)
    f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
    f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
    f_rec_GCV = f_rec_GCV[:, 0]
    lambda_GCV = np.squeeze(lambda_GCV)

    if lam_ini_val == "LCurve" or lam_ini_val == "L-Curve":
        LRIto_ini_lam = lambda_LC
        f_rec_ini = f_rec_LC
    elif lam_ini_val == "GCV" or lam_ini_val == "gcv":
        LRIto_ini_lam = lambda_GCV
        f_rec_ini = f_rec_GCV
    elif lam_ini_val == "DP" or lam_ini_val == "dp":
        LRIto_ini_lam = lambda_DP
        f_rec_ini = f_rec_DP

    # LRIto_ini_lam = lambda_LC
    maxiter = 200
    # maxiter = 200
    # maxiter = 600
    # f_rec_LocReg_LC, lambda_locreg_LC = profile_function(LocReg_Ito_mod, dat_noisy, A, lambda_LC, gamma_init, maxiter)
    # f_rec_oracle, lambda_oracle = profile_function(minimize_OP, Alpha_vec, L, dat_noisy, A, len(T2), IdealModel_weighted)

    # f_rec_LocReg_LC, lambda_locreg_LC = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)

    #as of 9/13/24, this is the best/orgianl 
    f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_cy(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
    # f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1 = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, f_rec_ini, gamma_init, maxiter)

    # f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1 = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, f_rec_ini, gamma_init, maxiter)

    if testing == True and iter_sigma == 0 and iter_rps == 0:
        meanfrec1 = np.mean(test_frec1)
        meanlam1 = np.mean(test_lam1)
        plt.figure(figsize=(10, 5))  # Create a new figure
        
        # First subplot
        plt.subplot(1, 2, 1) 
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, test_frec1, linestyle=':', linewidth=3, color='red', label=f'frec1 gamma init {gamma_init:.2f}')
        plt.legend(fontsize=10, loc='best')
        
        # Second subplot
        plt.subplot(1, 2, 2)  # Changed to 2 for the second subplot
        plt.semilogy(T2, test_lam1 * np.ones(len(T2)), linewidth=3, color='green', label=f'test_lam1 with meanlam1 {meanlam1:.2f}')
        plt.legend(fontsize=10, loc='best')
        
        # Print information
        print(f"{numiterate} iters for Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}")
        
        # Show plots
        plt.savefig(f'plot_output_gamma_init{gamma_init}.png')
    else:
        pass

    f_rec_oracle, lambda_oracle = minimize_OP(Lambda, L, dat_noisy, A, len(T2), IdealModel_weighted)

    #normalization
    sum_x = np.sum(f_rec_LocReg_LC)
    f_rec_LocReg_LC = f_rec_LocReg_LC / sum_x
    sum_oracle = np.sum(f_rec_oracle)
    f_rec_oracle = f_rec_oracle / sum_oracle
    sum_GCV = np.sum(f_rec_GCV)
    f_rec_GCV = f_rec_GCV / sum_GCV
    sum_LC = np.sum(f_rec_LC)
    f_rec_LC = f_rec_LC / sum_LC
    sum_DP = np.sum(f_rec_DP)
    f_rec_DP = f_rec_DP / sum_DP

    # Flatten results
    f_rec_GCV = f_rec_GCV.flatten()
    f_rec_DP = f_rec_DP.flatten()
    f_rec_LC = f_rec_LC.flatten()
    f_rec_LocReg_LC = f_rec_LocReg_LC.flatten()
    f_rec_oracle = f_rec_oracle.flatten()

    err_LC = l2_error(IdealModel_weighted, f_rec_LC)
    err_DP = l2_error(IdealModel_weighted, f_rec_DP)
    err_GCV = l2_error(IdealModel_weighted, f_rec_GCV)
    err_oracle = l2_error( IdealModel_weighted, f_rec_oracle)
    err_LR = l2_error( IdealModel_weighted, f_rec_LocReg_LC)
    # if iter_rps == 0 and iter_sigma == 0:
    #     true_norm = linalg_norm(IdealModel_weighted)
    #     print("true_norm", linalg_norm(IdealModel_weighted))
    #     print("locreg error",err_LR)
    #     print("gcv error", err_GCV)
    #     print("oracle error", err_oracle)

    # Plot a random simulation
    # random_sim = random.randint(0, n_sim)
    if iter_sim == 0:
        plot(iter_sim, iter_sigma, iter_rps)
        print(f"Finished Plots for iteration {iter_sim} sigma {iter_sigma} rps {iter_rps}")
    else:
        pass

    # Create DataFrame
    # sol_strct_uneql = {}
    # sol_strct_uneql['noiseless_data'] = noiseless_data
    # sol_strct_uneql['noisy_data'] = noisy_data  
    feature_df = pd.DataFrame(columns=["NR", 'Sigma', 'RPS_val', 'err_DP', "err_LC", "err_LR", "err_GCV", "err_oracle"])
    feature_df["NR"] = [iter_sim]
    feature_df["Sigma"] = [sigma_i]
    feature_df["RPS_val"] = [rps_val]
    feature_df["err_DP"] = [err_DP]
    feature_df["err_LC"] = [err_LC]
    feature_df["err_LR"] = [err_LR]
    feature_df["err_GCV"] = [err_GCV]
    feature_df["err_oracle"] = [err_oracle]

    # noise_df = pd.DataFrame(columns= ["Noise"])
    return feature_df, noise

def generate_estimates_shift(i_param_combo):

    def get_orig_plot(iter_sim, iter_sigma, iter_rps):
        """
        This function generates gt and all the reconstructions of regularization methods (DP, LC, GCV, LocReg) without any shifts and their respective wassterstein and l2 errors 
        Returns for example: plot of gt, locreg/DP/LC/GCV with current relative l2 errror and wassterstein score,
        """
        plt.figure(figsize=(12.06, 4.2))
        # Plotting the first subplot
        plt.subplot(1, 3, 1) 
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} (Rel. L2 Error: {round(err_LR_L2,3)}; Wass. Score = {round(err_LR_w,3)})')
        plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle (Rel. L2 Error: {round(err_oracle_L2,3)} ; Wass. Score = {round(err_oracle_w,3)})')
        plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Rel. L2 Error: {round(err_DP_L2,3)} ; Wass. Score = {round(err_DP_w,3)})')
        plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Rel. L2 Error: {round(err_GCV_L2,3)} ; Wass. Score = {round(err_GCV_w,3)})' )
        plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Rel. L2 Error: {round(err_LC_L2,3)} ; Wass. Score = {round(err_LC_w,3)})')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
        plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
        ymax = np.max(IdealModel_weighted) * 1.15
        plt.ylim(0, ymax)

        # Plotting the second subplot
        plt.subplot(1, 3, 2)
        plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label= f'LocReg {lam_ini_val}')
        plt.plot(TE, A @ f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle')
        plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
        plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
        plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('TE', fontsize=20, fontweight='bold')
        plt.ylabel('Intensity', fontsize=20, fontweight='bold')
        
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
        string = "MRR_1D_LocReg_Comparison"
        file_path = create_result_folder(string, SNR, lam_ini_val, dist_type)
        add_path_name = "unshifted"
        new_file_path = file_path + f"/" + add_path_name
        if not os.path.exists(new_file_path):
            os.makedirs(new_file_path)
        plt.savefig(os.path.join(new_file_path, f"Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}.png"))
        plt.close() 
        return 

    def get_both_shift_plots_LR(iter_sim, iter_sigma, iter_rps):
        """
        This function generates the original reconstruction, the shifted wassterstein, and the shifted L2 norm error for a given regularization method/LocReg (ex. locreg)
        Returns for example: plot of gt, locreg with current relative l2 errror and wassterstein score, locreg with shifted wassterstein, locreg with shifted relative l2 error
        """
        plt.figure(figsize=(12.06, 4.2))
        # Plotting the first subplot
        plt.subplot(1, 3, 1) 
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} (Rel. L2 Error: {round(err_LR_L2,3)}) ; Wass. Score = {round(err_LR_w,3)})')
        plt.plot(T2, shift_w_LR, linestyle='-.', linewidth=3, color='yellow', label=f'LocReg Wass. Shift (Wass. Error: {round(err_LR_w_shift,3)})')
        plt.plot(T2, shift_L2_LR, linewidth=3, color='blue', label=f'LocReg L2 Shift (Rel. L2 Error: {round(err_LR_L2_shift,3)})')

        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
        plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
        ymax = np.max(IdealModel_weighted) * 1.15
        plt.ylim(0, ymax)

        # Plotting the second subplot
        plt.subplot(1, 3, 2)
        plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label= f'LocReg {lam_ini_val}')
        plt.plot(TE, A @ shift_w_LR, linestyle='-.', linewidth=3, color='yellow', label='LocReg Wass. Shift')
        plt.plot(TE, A @ shift_L2_LR, linewidth=3, color='blue', label='LocReg L2 Shift')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('TE', fontsize=20, fontweight='bold')
        plt.ylabel('Intensity', fontsize=20, fontweight='bold')
        
        plt.subplot(1, 3, 3)
        plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')

        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2', fontsize=20, fontweight='bold')
        plt.ylabel('Lambda', fontsize=20, fontweight='bold')

        plt.tight_layout()
        string = "MRR_1D_LocReg_Comparison"
        file_path = create_result_folder(string, SNR, lam_ini_val, dist_type)
        add_path_name = "LR_compare_shifts"
        new_file_path = file_path + f"/" + add_path_name
        if not os.path.exists(new_file_path):
            os.makedirs(new_file_path)
        plt.savefig(os.path.join(new_file_path, f"Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}.png"))
        plt.close() 
        return 

    def get_both_shift_plots_GCV(iter_sim, iter_sigma, iter_rps):
        """
        This function generates the original reconstruction, the shifted wassterstein, and the shifted L2 norm error for a given regularization method/LocReg (ex. locreg)
        Returns for example: plot of gt, locreg with current relative l2 errror and wassterstein score, locreg with shifted wassterstein, locreg with shifted relative l2 error
        """
        plt.figure(figsize=(12.06, 4.2))
        # Plotting the first subplot
        plt.subplot(1, 3, 1) 
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, f_rec_GCV, linestyle=':', linewidth=3, color='red', label=f'GCV (Rel. L2 Error: {round(err_GCV_L2,3)}) ; Wass. Score = {round(err_GCV_w,3)})')
        plt.plot(T2, shift_w_GCV, linestyle='-.', linewidth=3, color='yellow', label=f'GCV Wass. Shift (Wass. Error: {round(err_GCV_w_shift,3)})')
        plt.plot(T2, shift_L2_GCV, linewidth=3, color='blue', label=f'GCV L2 Shift (Rel. L2 Error: {round(err_GCV_L2_shift,3)})')

        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
        plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
        ymax = np.max(IdealModel_weighted) * 1.15
        plt.ylim(0, ymax)

        # Plotting the second subplot
        plt.subplot(1, 3, 2)
        plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(TE, A @ f_rec_GCV,  linestyle=':', linewidth=3, color='red', label= f'GCV')
        plt.plot(TE, A @ shift_w_GCV, linestyle='-.', linewidth=3, color='yellow', label='GCV Wass. Shift')
        plt.plot(TE, A @ shift_L2_GCV, linewidth=3, color='blue', label='GCV L2 Shift')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('TE', fontsize=20, fontweight='bold')
        plt.ylabel('Intensity', fontsize=20, fontweight='bold')
        
        plt.subplot(1, 3, 3)
        plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'GCV')

        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2', fontsize=20, fontweight='bold')
        plt.ylabel('Lambda', fontsize=20, fontweight='bold')

        plt.tight_layout()
        string = "MRR_1D_LocReg_Comparison"
        file_path = create_result_folder(string, SNR, lam_ini_val, dist_type)
        add_path_name = "GCV_compare_shifts"
        new_file_path = file_path + f"/" + add_path_name
        if not os.path.exists(new_file_path):
            os.makedirs(new_file_path)
        plt.savefig(os.path.join(new_file_path, f"Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}.png"))
        plt.close() 
        return 

    def get_wass_shift_plots(iter_sim, iter_sigma, iter_rps):
        """
        This function generates the gt, and the best  shifted wassterstein  error for all regularization methods
        Returns: plot of gt, locreg, LC, GCV, DP all with their respective best shifted wassterstein
        """
        plt.figure(figsize=(12.06, 4.2))
        # Plotting the first subplot
        plt.subplot(1, 3, 1) 
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, shift_w_LR, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} Wass. Shift (Wass. Error: {round(err_LR_w_shift,3)})')
        plt.plot(T2, shift_w_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle Wass. Shift (Wass. Error: {round(err_oracle_w_shift,3)})')
        plt.plot(T2, shift_w_DP, linewidth=3, color='green', label=f'DP Wass. Shift (Wass. Error: {round(err_DP_w_shift,3)})')
        plt.plot(T2, shift_w_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV Wass. Shift (Wass. Error: {round(err_GCV_w_shift,3)})')
        plt.plot(T2, shift_w_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve Wass. Shift (Wass. Error: {round(err_LC_w_shift,3)})')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
        plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
        ymax = np.max(IdealModel_weighted) * 1.15
        plt.ylim(0, ymax)

        # Plotting the second subplot
        plt.subplot(1, 3, 2)
        plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(TE, A @ shift_w_LR, linestyle=':', linewidth=3, color='red', label= f'LocReg Wass. Shift {lam_ini_val}')
        plt.plot(TE, A @ shift_w_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle Wass. Shift')
        plt.plot(TE, A @ shift_w_DP, linewidth=3, color='green', label='DP Wass. Shift')
        plt.plot(TE, A @ shift_w_GCV, linestyle='--', linewidth=3, color='blue', label='GCV Wass. Shift')
        plt.plot(TE, A @ shift_w_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve Wass. Shift')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('TE', fontsize=20, fontweight='bold')
        plt.ylabel('Intensity', fontsize=20, fontweight='bold')
        
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
        string = "MRR_1D_LocReg_Comparison"
        file_path = create_result_folder(string, SNR, lam_ini_val, dist_type)
        add_path_name = "wass_shifts"
        new_file_path = file_path + f"/" + add_path_name
        if not os.path.exists(new_file_path):
            os.makedirs(new_file_path)
        plt.savefig(os.path.join(new_file_path, f"Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}.png"))
        plt.close() 
        return 
    def get_L2_shift_plots(iter_sim, iter_sigma, iter_rps):
        """
        This function generates the gt, and the best the shifted L2 norm error for all regularization methods
        Returns: plot of gt, locreg, LC, GCV, DP all with their respective best shifted relative l2 error
        """
        plt.figure(figsize=(12.06, 4.2))
        # Plotting the first subplot
        plt.subplot(1, 3, 1) 
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, shift_L2_LR, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} L2 Shift (Rel. L2 Error: {round(err_LR_L2_shift,3)})')
        plt.plot(T2, shift_L2_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle L2 Shift (Rel. L2 Error: {round(err_oracle_L2_shift,3)})')
        plt.plot(T2, shift_L2_DP, linewidth=3, color='green', label=f'DP L2 Shift (Rel. L2 Error: {round(err_DP_L2_shift,3)})')
        plt.plot(T2, shift_L2_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV L2 Shift (Rel. L2 Error: {round(err_GCV_L2_shift,3)})')
        plt.plot(T2, shift_L2_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve L2 Shift (Rel. L2 Error: {round(err_LC_L2_shift,3)})')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
        plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
        ymax = np.max(IdealModel_weighted) * 1.15
        plt.ylim(0, ymax)

        # Plotting the second subplot
        plt.subplot(1, 3, 2)
        plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(TE, A @ shift_L2_LR, linestyle=':', linewidth=3, color='red', label= f'LocReg {lam_ini_val} L2 Shift')
        plt.plot(TE, A @ shift_L2_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle L2 Shift')
        plt.plot(TE, A @ shift_L2_DP, linewidth=3, color='green', label='DP L2 Shift')
        plt.plot(TE, A @ shift_L2_GCV, linestyle='--', linewidth=3, color='blue', label='GCV L2 Shift')
        plt.plot(TE, A @ shift_L2_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve L2 Shift')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('TE', fontsize=20, fontweight='bold')
        plt.ylabel('Intensity', fontsize=20, fontweight='bold')
        
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
        string = "MRR_1D_LocReg_Comparison"
        add_path_name = "L2_shifts"
        file_path = create_result_folder(string, SNR, lam_ini_val, dist_type)
        new_file_path = file_path + f"/" + add_path_name
        if not os.path.exists(new_file_path):
            os.makedirs(new_file_path)
        plt.savefig(os.path.join(new_file_path, f"Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}.png"))
        plt.close() 
        return 


    iter_sim, iter_sigma, iter_rps = target_iterator[i_param_combo]
    L = np.eye(A.shape[1])
    sigma_i = diff_sigma[iter_sigma, :]
    rps_val = calc_rps_val(iter_rps, rps)

    # Profile the function calls individually
    IdealModel_weighted = get_IdealModel_weighted(iter_rps, m, npeaks, T2, T2mu, sigma_i)

    if preset_noise == False:
        dat_noisy,noise = calc_dat_noisy( A, TE, IdealModel_weighted, SNR)
    else:
        dat_noiseless = A @ IdealModel_weighted
        # print("noise_arr.shape", noise_arr.shape)
        # print("iter_sim", iter_sim)
        # print("iter_sigma", iter_sigma)
        # print("iter_rps", iter_rps)

        # noise_arr = noise_arr.flatten()
        # noise_arr = noise_arr.reshape(z,y)
        # print("dat_noiseless.shape", dat_noiseless.shape)
        # print("noise_iter shape",noise_arr[])
        noise = noise_arr[iter_sim, iter_sigma, iter_rps,:]
        # print("noise", noise.shape)
        # print("noise.shape", noise.shape)
        dat_noisy = dat_noiseless + np.ravel(noise)
        # print("dat_noiseless.shape", dat_noiseless.shape)

        noisy_data[iter_sim, iter_sigma, iter_rps,:] = dat_noisy
        noiseless_data[iter_sim, iter_sigma, iter_rps,:] = dat_noiseless

        # noise = noise_arr.tolist()
        # noise_arr = np.zeros((n_sim, nsigma, nrps))
    f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)
    f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
    f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
    f_rec_GCV = f_rec_GCV[:, 0]
    lambda_GCV = np.squeeze(lambda_GCV)

    if lam_ini_val == "LCurve" or lam_ini_val == "L-Curve":
        LRIto_ini_lam = lambda_LC
        f_rec_ini = f_rec_LC
    elif lam_ini_val == "GCV" or lam_ini_val == "gcv":
        LRIto_ini_lam = lambda_GCV
        f_rec_ini = f_rec_GCV
    elif lam_ini_val == "DP" or lam_ini_val == "dp":
        LRIto_ini_lam = lambda_DP
        f_rec_ini = f_rec_DP

    # LRIto_ini_lam = lambda_LC
    maxiter = 200
    # maxiter = 200
    # maxiter = 600
    # f_rec_LocReg_LC, lambda_locreg_LC = profile_function(LocReg_Ito_mod, dat_noisy, A, lambda_LC, gamma_init, maxiter)
    # f_rec_oracle, lambda_oracle = profile_function(minimize_OP, Alpha_vec, L, dat_noisy, A, len(T2), IdealModel_weighted)

    # f_rec_LocReg_LC, lambda_locreg_LC = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)

    #9/13/24
    # f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
  
    f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1 = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, f_rec_ini, gamma_init, maxiter)

    if testing == True and iter_sigma == 0 and iter_rps == 0:
        meanfrec1 = np.mean(test_frec1)
        meanlam1 = np.mean(test_lam1)
        plt.figure(figsize=(10, 5))  # Create a new figure
        
        # First subplot
        plt.subplot(1, 2, 1) 
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, test_frec1, linestyle=':', linewidth=3, color='red', label=f'frec1 gamma init {gamma_init:.2f}')
        plt.legend(fontsize=10, loc='best')
        
        # Second subplot
        plt.subplot(1, 2, 2)  # Changed to 2 for the second subplot
        plt.semilogy(T2, test_lam1 * np.ones(len(T2)), linewidth=3, color='green', label=f'test_lam1 with meanlam1 {meanlam1:.2f}')
        plt.legend(fontsize=10, loc='best')
        
        # Print information
        print(f"{numiterate} iters for Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}")
        
        # Show plots
        plt.savefig(f'plot_output_gamma_init{gamma_init}.png')
    else:
        pass

    f_rec_oracle, lambda_oracle = minimize_OP(Lambda, L, dat_noisy, A, len(T2), IdealModel_weighted)

    #normalization
    sum_x = np.sum(f_rec_LocReg_LC)
    f_rec_LocReg_LC = f_rec_LocReg_LC / sum_x
    sum_oracle = np.sum(f_rec_oracle)
    f_rec_oracle = f_rec_oracle / sum_oracle
    sum_GCV = np.sum(f_rec_GCV)
    f_rec_GCV = f_rec_GCV / sum_GCV
    sum_LC = np.sum(f_rec_LC)
    f_rec_LC = f_rec_LC / sum_LC
    sum_DP = np.sum(f_rec_DP)
    f_rec_DP = f_rec_DP / sum_DP

    # Flatten results
    f_rec_GCV = f_rec_GCV.flatten()
    f_rec_DP = f_rec_DP.flatten()
    f_rec_LC = f_rec_LC.flatten()
    f_rec_LocReg_LC = f_rec_LocReg_LC.flatten()
    f_rec_oracle = f_rec_oracle.flatten()

    #compute original relative l2 norm error ans wass dist without shifts
    err_LC_L2 = l2_error(IdealModel_weighted, f_rec_LC)
    err_DP_L2 = l2_error(IdealModel_weighted, f_rec_DP)
    err_GCV_L2 = l2_error(IdealModel_weighted, f_rec_GCV)
    err_oracle_L2 = l2_error(IdealModel_weighted, f_rec_oracle)
    err_LR_L2 = l2_error(IdealModel_weighted, f_rec_LocReg_LC)

    err_LC_w = wass_error(T2, IdealModel_weighted, f_rec_LC)
    err_DP_w = wass_error(T2, IdealModel_weighted, f_rec_DP)
    err_GCV_w = wass_error(T2,IdealModel_weighted, f_rec_GCV)
    err_oracle_w = wass_error(T2, IdealModel_weighted, f_rec_oracle)
    err_LR_w = wass_error(T2, IdealModel_weighted, f_rec_LocReg_LC)

    #compute  relative l2 norm error ans wass dist with shifts (using grid search of gamma)
    # w_LR_scores = []
    # w_LC_scores = []
    # w_GCV_scores = []
    # w_DP_scores = []
    # w_oracle_scores = []

    # L2_LR_scores = []
    # L2_LC_scores = []
    # L2_GCV_scores = []
    # L2_DP_scores = []
    # L2_oracle_scores = []
    # Calculate shifting errors of wassterstein and relative L2 norm
        #Store a bunch of wass.metrics across a variety of gamma values
    # for beta in tqdm(beta_list):
    #     LR_wass_shift = wass_shift(T2, beta, IdealModel_weighted,f_rec_LocReg_LC, shift = True)
    #     LC_wass_shift = wass_shift(T2, beta, IdealModel_weighted,f_rec_LC,shift = True)
    #     GCV_wass_shift = wass_shift(T2, beta, IdealModel_weighted,f_rec_GCV, shift = True)
    #     DP_wass_shift = wass_shift(T2, beta, IdealModel_weighted,f_rec_DP, shift = True)
    #     oracle_wass_shift = wass_shift(T2, beta, IdealModel_weighted,f_rec_oracle, shift = True)

    #     w_LR_scores.append(LR_wass_shift)
    #     w_LC_scores.append(LC_wass_shift)
    #     w_GCV_scores.append(GCV_wass_shift)
    #     w_DP_scores.append(DP_wass_shift)
    #     w_oracle_scores.append(oracle_wass_shift)

    #     LR_l2_shift = l2_error_shift(beta, IdealModel_weighted,f_rec_LocReg_LC)
    #     LC_l2_shift = l2_error_shift(beta, IdealModel_weighted,f_rec_LC)
    #     GCV_l2_shift = l2_error_shift(beta, IdealModel_weighted,f_rec_GCV)
    #     DP_l2_shift = l2_error_shift(beta, IdealModel_weighted,f_rec_DP)
    #     oracle_l2_shift = l2_error_shift(beta, IdealModel_weighted,f_rec_oracle)

    #     L2_LR_scores.append(LR_l2_shift)
    #     L2_LC_scores.append(LC_l2_shift)
    #     L2_GCV_scores.append(GCV_l2_shift)
    #     L2_DP_scores.append(DP_l2_shift)
    #     L2_oracle_scores.append(oracle_l2_shift)

    # # Compute Wasserstein distances using list comprehensions
    # w_LR_scores = [wass_shift(T2, beta, IdealModel_weighted, f_rec_LocReg_LC, shift=True) for beta in beta_list]
    # w_LC_scores = [wass_shift(T2, beta, IdealModel_weighted, f_rec_LC, shift=True) for beta in beta_list]
    # w_GCV_scores = [wass_shift(T2, beta, IdealModel_weighted, f_rec_GCV, shift=True) for beta in beta_list]
    # w_DP_scores = [wass_shift(T2, beta, IdealModel_weighted, f_rec_DP, shift=True) for beta in beta_list]
    # w_oracle_scores = [wass_shift(T2, beta, IdealModel_weighted, f_rec_oracle, shift=True) for beta in beta_list]

    # # Compute L2 errors using list comprehensions
    # L2_LR_scores = [l2_error_shift(beta, IdealModel_weighted, f_rec_LocReg_LC) for beta in beta_list]
    # L2_LC_scores = [l2_error_shift(beta, IdealModel_weighted, f_rec_LC) for beta in beta_list]
    # L2_GCV_scores = [l2_error_shift(beta, IdealModel_weighted, f_rec_GCV) for beta in beta_list]
    # L2_DP_scores = [l2_error_shift(beta, IdealModel_weighted, f_rec_DP) for beta in beta_list]
    # L2_oracle_scores = [l2_error_shift(beta, IdealModel_weighted, f_rec_oracle) for beta in beta_list]

    w_LR_scores = np.array([wass_shift(T2, beta, IdealModel_weighted, f_rec_LocReg_LC) for beta in beta_list])
    w_LC_scores = np.array([wass_shift(T2, beta, IdealModel_weighted, f_rec_LC) for beta in beta_list])
    w_GCV_scores = np.array([wass_shift(T2, beta, IdealModel_weighted, f_rec_GCV) for beta in beta_list])
    w_DP_scores = np.array([wass_shift(T2, beta, IdealModel_weighted, f_rec_DP) for beta in beta_list])
    w_oracle_scores = np.array([wass_shift(T2, beta, IdealModel_weighted, f_rec_oracle) for beta in beta_list])

    # Compute L2 errors using list comprehensions
    L2_LR_scores = np.array([l2_error_shift(beta, IdealModel_weighted, f_rec_LocReg_LC) for beta in beta_list])
    L2_LC_scores = np.array([l2_error_shift(beta, IdealModel_weighted, f_rec_LC) for beta in beta_list])
    L2_GCV_scores = np.array([l2_error_shift(beta, IdealModel_weighted, f_rec_GCV) for beta in beta_list])
    L2_DP_scores = np.array([l2_error_shift(beta, IdealModel_weighted, f_rec_DP) for beta in beta_list])
    L2_oracle_scores = np.array([l2_error_shift(beta, IdealModel_weighted, f_rec_oracle) for beta in beta_list])

    #Find opt beta of the shifts for wassterstein and rel l2 norm error
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

    #Find shifted reconstructions by wassterstein and rel l2 norm error:
    shift_w_LR= np.interp(T2 + w_LR_opt_beta, T2, f_rec_LocReg_LC)
    shift_w_DP = np.interp(T2 + w_DP_opt_beta, T2, f_rec_DP)
    shift_w_oracle = np.interp(T2 + w_oracle_opt_beta, T2, f_rec_oracle)
    shift_w_LC = np.interp(T2 + w_LC_opt_beta, T2, f_rec_LC)
    shift_w_GCV = np.interp(T2 + w_GCV_opt_beta, T2, f_rec_GCV)

    shift_L2_LR = np.interp(T2 + L2_LR_opt_beta, T2, f_rec_LocReg_LC)
    shift_L2_DP = np.interp(T2 + L2_DP_opt_beta, T2, f_rec_DP)
    shift_L2_oracle = np.interp(T2 + L2_oracle_opt_beta, T2, f_rec_oracle)
    shift_L2_LC = np.interp(T2 + L2_LC_opt_beta, T2, f_rec_LC)
    shift_L2_GCV = np.interp(T2 + L2_GCV_opt_beta, T2, f_rec_GCV)
    # if iter_rps == 0 and iter_sigma == 0:
    #     true_norm = linalg_norm(IdealModel_weighted)
    #     print("true_norm", linalg_norm(IdealModel_weighted))
    #     print("locreg error",err_LR)
    #     print("gcv error", err_GCV)
    #     print("oracle error", err_oracle)

    # Plot a random simulation
    # random_sim = random.randint(0, n_sim)
    # if iter_sim == 0 and iter_sigma == 1 and iter_rps == 3:
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
    else:
        pass

    # Create DataFrame
    # sol_strct_uneql = {}
    # sol_strct_uneql['noiseless_data'] = noiseless_data
    # sol_strct_uneql['noisy_data'] = noisy_data  

    feature_df_L2 = pd.DataFrame(columns=["NR", 'Sigma', 'RPS_val', 'err_DP', "err_LC", "err_LR", "err_GCV", "err_oracle",  'shift_err_DP', "shift_err_LC", "shift_err_LR", "shift_err_GCV", "shift_err_oracle"])
    feature_df_L2["NR"] = [iter_sim]
    feature_df_L2["Sigma"] = [sigma_i]
    feature_df_L2["RPS_val"] = [rps_val]
    feature_df_L2["err_DP"] = [err_DP_L2]
    feature_df_L2["err_LC"] = [err_LC_L2]
    feature_df_L2["err_LR"] = [err_LR_L2]
    feature_df_L2["err_GCV"] = [err_GCV_L2]
    feature_df_L2["err_oracle"] = [err_oracle_L2]
    feature_df_L2["shift_err_DP"] = [err_DP_L2_shift]
    feature_df_L2["shift_err_LC"] = [err_LC_L2_shift]
    feature_df_L2["shift_err_LR"] = [err_LR_L2_shift]
    feature_df_L2["shift_err_GCV"] = [err_GCV_L2_shift]
    feature_df_L2["shift_err_oracle"] = [err_oracle_L2_shift]

    feature_df_w = pd.DataFrame(columns=["NR", 'Sigma', 'RPS_val', 'err_DP', "err_LC", "err_LR", "err_GCV", "err_oracle",  'shift_err_DP', "shift_err_LC", "shift_err_LR", "shift_err_GCV", "shift_err_oracle"])
    feature_df_w["NR"] = [iter_sim]
    feature_df_w["Sigma"] = [sigma_i]
    feature_df_w["RPS_val"] = [rps_val]
    feature_df_w["err_DP"] = [err_DP_w]
    feature_df_w["err_LC"] = [err_LC_w]
    feature_df_w["err_LR"] = [err_LR_w]
    feature_df_w["err_GCV"] = [err_GCV_w]
    feature_df_w["err_oracle"] = [err_oracle_w]
    feature_df_w["shift_err_DP"] = [err_DP_w_shift]
    feature_df_w["shift_err_LC"] = [err_LC_w_shift]
    feature_df_w["shift_err_LR"] = [err_LR_w_shift]
    feature_df_w["shift_err_GCV"] = [err_GCV_w_shift]
    feature_df_w["shift_err_oracle"] = [err_oracle_w_shift]
    
    return feature_df_L2, feature_df_w, noise
    # noise_df = pd.DataFrame(columns= ["Noise"])

def parallel_processed(func, shift = True):
    if shift == True:
        with mp.Pool(processes = num_cpus_avail) as pool:
            with tqdm(total = len(target_iterator)) as pbar:
                for estimates_dataframe_L2, estimates_dataframe_w, noisereal in pool.imap_unordered(func, range(len(target_iterator))):
                    lis_L2.append(estimates_dataframe_L2)
                    lis_w.append(estimates_dataframe_w)
                    noise_list.append(noisereal)
                    pbar.update()
            pool.close()
            pool.join()
        noise_arr = np.array(noise_list)
        noise_arr = noise_arr.reshape(n_sim, nsigma, nrps, A.shape[0])
        return estimates_dataframe_L2, estimates_dataframe_w, noise_arr
    else:
        with mp.Pool(processes = num_cpus_avail) as pool:
            with tqdm(total = len(target_iterator)) as pbar:
                for estimates_dataframe, noisereal in pool.imap_unordered(func, range(len(target_iterator))):
                    lis.append(estimates_dataframe)
                    noise_list.append(noisereal)
                    pbar.update()
            pool.close()
            pool.join()
        noise_arr = np.array(noise_list)
        noise_arr = noise_arr.reshape(n_sim, nsigma, nrps, A.shape[0])
        
        return estimates_dataframe, noise_arr

if __name__ == '__main__':
    freeze_support()
    unif_sigma, diff_sigma = calc_diff_sigma(nsigma)
    T2, TE, Lambda, A, m,  SNR = load_Gaus(Gaus_info)
    T2mu = calc_T2mu(rps)

    print("Finished Assignments...")  

    lis = []
    lis_L2 = []
    lis_w = []
    #insert noise_arr:
    # profiler = cProfile.Profile()
    # profiler.enable()
    if shift_beta == True:
        estimates_dataframe_L2, estimates_dataframe_w, noise_arr = parallel_processed(generate_estimates_shift, shift = True)
        print(f"Completed {len(lis_L2)} of {len(target_iterator)} L2 voxels")
        print(f"Completed {len(lis_w)} of {len(target_iterator)} wassterstein voxels")
        df_L2 = pd.concat(lis_L2, ignore_index= True)
        df_L2.to_pickle(data_folder + f'/' + data_tag +'L2.pkl')
        df_w = pd.concat(lis_w, ignore_index= True)
        df_w.to_pickle(data_folder + f'/' + data_tag +'wass.pkl')    
        if preset_noise == False:
            np.save(data_folder + f'/' + data_tag + "noise_arr_shift", noise_arr)
            print("noise array saved")
        else:
            print("Used preset noise array")
            # np.save(data_folder + f'/' + data_tag + "noiselessdata", noiseless_data)
            # np.save(data_folder + f'/' + data_tag + "noisydata", noisy_data)
            # print("test noise array saved")
            pass
    else:
        estimates_dataframe, noise_arr = parallel_processed(generate_estimates, shift = False)
        print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
        df = pd.concat(lis, ignore_index= True)
        df.to_pickle(data_folder + f'/' + data_tag +'.pkl')

        df['Sigma'] = df['Sigma'].apply(tuple)
        df_sorted = df.sort_values(by=['NR','Sigma', 'RPS_val'], ascending=[True, True, True])
        print("df_sorted", df_sorted)

        grouped = df_sorted.groupby(['Sigma', 'RPS_val']).agg({
            'err_DP': 'sum',
            'err_LC': 'sum',
            'err_LR': 'sum',
            'err_GCV': 'sum',
            'err_oracle': 'sum',
            # "shift_err_DP": 'sum',
            # "shift_err_LC": 'sum',
            # "shift_err_LR": 'sum',
            # "shift_err_GCV": 'sum',
            # "shift_err_oracle": 'sum'
        })
        num_NRs = df_sorted['NR'].nunique()
        # Average the errors
        average_errors = grouped / num_NRs
        errors = average_errors

        print(grouped)

        n = nsigma
        m = len(rps)

        # errors = df.groupby('Sigma').agg({
        #     'NR': 'sum',
        #     'err_DP': 'sum',
        #     'err_LC': 'sum',
        #     'err_LR': 'sum',
        #     'err_GCV': 'sum',
        #     'err_oracle': 'sum'
        #     }).reset_index()


        errs_oracle = errors["err_oracle"].to_numpy().reshape(n,m)
        errs_oracle = np.array(errs_oracle)
        errs_LC= errors["err_LC"].to_numpy().reshape(n,m)
        errs_LC = np.array(errs_LC)
        errs_GCV = errors["err_GCV"].to_numpy().reshape(n,m)
        errs_GCV = np.array(errs_GCV)
        errs_DP = errors["err_DP"].to_numpy().reshape(n,m)
        errs_DP = np.array(errs_DP)
        errs_LR = errors["err_LR"].to_numpy().reshape(n,m)
        errs_LR = np.array(errs_LR)

        # avg_oracle = np.mean(errs_oracle)
        # avg_LC = np.mean(errs_LC)
        # avg_GCV = np.mean(errs_GCV)
        # avg_DP = np.mean(errs_DP)
        # avg_LR = np.mean(errs_LR)
        # print(errs_LR)
        compare_GCV = errs_LR - errs_GCV
        # print("compare_GCV", compare_GCV)
        compare_DP = errs_LR - errs_DP
        compare_LC = errs_LR - errs_LC
        compare_oracle = errs_LR - errs_oracle

        if show == 1:
            def add_custom_colorbar(ax, im, title):
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(title, fontsize=18)
                cbar.ax.tick_params(labelsize=14)
                
                # Define custom ticks and labels
                ticks = [-0.5, 0, 0.5]
                tick_labels = ['LocReg is better', 'Neutral', 'LocReg is worse']
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(tick_labels)
                return cbar
            # feature_df['IdealModel_data'] = feature_df
            y_min, y_max = unif_sigma[0], unif_sigma[-1]
            if y_min == y_max:
                y_max += 0.01  # Small offset

            plt.figure(figsize=(8, 12))
            plt.subplots_adjust(wspace=0.3, hspace=0.8)
            plt.subplot(4, 1, 1)
            plt.imshow(compare_GCV, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
            plt.colorbar()
            plt.xlabel('Ratio of Peak Separation', fontsize=18)
            plt.ylabel('Gaussian Sigma', fontsize=18)
            plt.title('LocReg Error - GCV Error', fontsize=18)
            plt.clim(-0.5, 0.5)

            plt.subplot(4, 1, 2)
            plt.imshow(compare_DP, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
            plt.colorbar()
            plt.xlabel('Ratio of Peak Separation', fontsize=18)
            plt.ylabel('Gaussian Sigma', fontsize=18)
            plt.title('LocReg Error - DP Error', fontsize=18)
            plt.clim(-0.5, 0.5)

            plt.subplot(4, 1, 3)
            plt.imshow(compare_LC, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
            plt.colorbar()
            plt.xlabel('Ratio of Peak Separation', fontsize=18)
            plt.ylabel('Gaussian Sigma', fontsize=18)
            plt.title('LocReg Error - L-Curve Error', fontsize=18)
            plt.clim(-0.5, 0.5)

            plt.subplot(4, 1, 4)
            plt.imshow(compare_oracle, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
            plt.colorbar()
            plt.xlabel('Ratio of Peak Separation', fontsize=18)
            plt.ylabel('Gaussian Sigma', fontsize=18)
            plt.title('LocReg Error - Oracle Error', fontsize=18)
            plt.clim(-0.5, 0.5)

            # plt.subplot(4, 1, 4)
            # plt.imshow(compare_LocReg, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
            # plt.colorbar()
            # plt.xlabel('Ratio of Peak Separation', fontsize=18)
            # plt.ylabel('Gaussian Sigma', fontsize=18)
            # plt.title('Comparison with LocReg using LC lam', fontsize=18)
            # plt.clim(-0.5, 0.5)
            # plt.savefig('heatmap.png')
            string = "Heatmap"
            file_path = create_result_folder(string, SNR, lam_ini_val, dist_type)
            plt.savefig(os.path.join(file_path, f"heatmap.png"))
            print(f"Saved Heatmap")
            plt.close()

        if preset_noise == False:
            np.save(data_folder + f'/' + data_tag + "noise_arr", noise_arr)
            print("noise array saved")
        else:
            print("Used preset noise array")
            # np.save(data_folder + f'/' + data_tag + "noiselessdata", noiseless_data)
            # np.save(data_folder + f'/' + data_tag + "noisydata", noisy_data)
            # print("test noise array saved")
            pass

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(10) 