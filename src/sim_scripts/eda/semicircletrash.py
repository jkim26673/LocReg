# import sys
# import os
# print("Setting system path")
# sys.path.append(".")  # Replace this path with the actual path to the parent directory of Utilities_functions
# import numpy as np
# from scipy.stats import norm as normsci
# from scipy.linalg import norm as linalg_norm
# from scipy.optimize import nnls
# import matplotlib.pyplot as plt
# import pickle
# from Utilities_functions.discrep_L2 import discrep_L2
# from Utilities_functions.GCV_NNLS import GCV_NNLS
# from Utilities_functions.Lcurve import Lcurve
# import pandas as pd
# import cvxpy as cp
# from scipy.linalg import svd
# from Simulations.lcurve_functions import l_cuve,csvd,l_corner
# from Simulations.l_curve_corner import l_curve_corner
# from regu.csvd import csvd
# from regu.discrep import discrep
# from Simulations.Ito_LocReg import Ito_LocReg
# from Simulations.Ito_LocReg import *
# from Utilities_functions.pasha_gcv import Tikhonov
# from regu.l_curve import l_curve
# from tqdm import tqdm
# from Utilities_functions.tikhonov_vec import tikhonov_vec
# import mosek
# from ItoLocRegConst import LocReg_Ito_C,LocReg_Ito_C_2,LocReg_Ito_C_4
# from regu.nonnegtik_hnorm import nonnegtik_hnorm
# import multiprocess as mp
# from multiprocessing import Pool, freeze_support
# from multiprocessing import set_start_method
# import functools
# from datetime import date
# import random
# import cProfile
# import pstats

# print("setting license path")
# mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
# os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
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
noise_file_path="/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_improve_algo_SNR300_iter1_lamini_gcv_dist_narrowL_broadR_improvealgo_27Aug24noise_arr.npy"
testing = False
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_15Aug24noise_arr_modifiedalgo.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_13Aug24noise_arr.txt.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_06Aug24noise_arr.txt.npy"

#Number of simulations:
n_sim = 1
SNR_value = 50

###LocReg hyperparameters
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
exp = 0.5
feedback = True
lam_ini_val = "gcv"
dist_type = "semicircle"
# gamma_init = 5
#gamma_init = 0.5 is best
gamma_init = 0.5

###Plotting hyperparameters
npeaks = 1
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
num_cpus_avail = np.min([len(target_iterator),100])

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
    folder_name = f"{cwd_full}/{string}_{date}_SNR_{SNR}_lamini_{lam_ini_val}_dist_{dist_type}_modularized"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def error(IdealModel,reconstr):
    true_norm = linalg_norm(IdealModel)
    err = linalg_norm(IdealModel - reconstr) / true_norm
    return err

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


def semicircle_distribution(T2):
    # Calculate the center and radius of the semicircle
    start = T2[0]
    end = T2[-1]
    center = (start + end) / 2
    radius = min(center - start, end - center)

    # Define the semicircle distribution function
    def f(t):
        if center - radius <= t <= center + radius:
            return np.sqrt(radius**2 - (t - center)**2)
        else:
            return 0

    # Calculate the raw distribution values
    raw_distribution = np.array([f(t) for t in T2])

    # Find the maximum value in the raw distribution
    max_value = np.sum(raw_distribution)

    # Normalize the distribution to have a maximum value of 0.5
    normalized_distribution = raw_distribution / max_value 

    return normalized_distribution


def generate_estimates(i):
    def plot(i = 0):
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
        plt.savefig(os.path.join(file_path, f"Simulation{n_sim}.png"))
        plt.close() 

    L = np.eye(A.shape[1])

    # Profile the function calls individually
    # IdealModel_weighted = get_IdealModel_weighted(iter_rps, m, npeaks, T2, T2mu, sigma_i)
    IdealModel_weighted = semicircle_distribution(T2)
    dat_noisy,noise = calc_dat_noisy(A, TE, IdealModel_weighted, SNR)
        # noise = noise_arr.tolist()
        # noise_arr = np.zeros((n_sim, nsigma, nrps))
    # f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)
    # f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
    f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
    f_rec_GCV = f_rec_GCV[:, 0]
    lambda_GCV = np.squeeze(lambda_GCV)

    # if lam_ini_val == "LCurve" or lam_ini_val == "L-Curve":
    #     LRIto_ini_lam = lambda_LC
    # elif lam_ini_val == "GCV" or lam_ini_val == "gcv":
    #     LRIto_ini_lam = lambda_GCV
    # elif lam_ini_val == "DP" or lam_ini_val == "dp":
    #     LRIto_ini_lam = lambda_DP

    if lam_ini_val == "GCV" or lam_ini_val == "gcv":
        LRIto_ini_lam = lambda_GCV

    # LRIto_ini_lam = lambda_LC
    # maxiter = 75
    maxiter = 200
    # maxiter = 600
    # f_rec_LocReg_LC, lambda_locreg_LC = profile_function(LocReg_Ito_mod, dat_noisy, A, lambda_LC, gamma_init, maxiter)
    # f_rec_oracle, lambda_oracle = profile_function(minimize_OP, Alpha_vec, L, dat_noisy, A, len(T2), IdealModel_weighted)
    f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
  
    
    # if testing == True and iter_sigma == 0 and iter_rps == 0:
    #     meanfrec1 = np.mean(test_frec1)
    #     meanlam1 = np.mean(test_lam1)
    #     plt.figure(figsize=(10, 5))  # Create a new figure
        
    #     # First subplot
    #     plt.subplot(1, 2, 1) 
    #     plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
    #     plt.plot(T2, test_frec1, linestyle=':', linewidth=3, color='red', label=f'frec1 gamma init {gamma_init:.2f}')
    #     plt.legend(fontsize=10, loc='best')
        
    #     # Second subplot
    #     plt.subplot(1, 2, 2)  # Changed to 2 for the second subplot
    #     plt.semilogy(T2, test_lam1 * np.ones(len(T2)), linewidth=3, color='green', label=f'test_lam1 with meanlam1 {meanlam1:.2f}')
    #     plt.legend(fontsize=10, loc='best')
        
    #     # Print information
    #     print(f"{numiterate} iters for Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}")
        
    #     # Show plots
    #     plt.savefig(f'plot_output_gamma_init{gamma_init}.png')
    # else:
    #     pass

    # f_rec_oracle, lambda_oracle = minimize_OP(Lambda, L, dat_noisy, A, len(T2), IdealModel_weighted)

    #normalization
    sum_x = np.sum(f_rec_LocReg_LC)
    f_rec_LocReg_LC = f_rec_LocReg_LC / sum_x
    # sum_oracle = np.sum(f_rec_oracle)
    # f_rec_oracle = f_rec_oracle / sum_oracle
    sum_GCV = np.sum(f_rec_GCV)
    f_rec_GCV = f_rec_GCV / sum_GCV
    # sum_LC = np.sum(f_rec_LC)
    # f_rec_LC = f_rec_LC / sum_LC
    # sum_DP = np.sum(f_rec_DP)
    # f_rec_DP = f_rec_DP / sum_DP

    # Flatten results
    f_rec_GCV = f_rec_GCV.flatten()
    # f_rec_DP = f_rec_DP.flatten()
    # f_rec_LC = f_rec_LC.flatten()
    f_rec_LocReg_LC = f_rec_LocReg_LC.flatten()
    # f_rec_oracle = f_rec_oracle.flatten()'

    # Calculate errors
    # err_LC = error(IdealModel_weighted, f_rec_LC)
    # err_DP = error(IdealModel_weighted, f_rec_DP)
    err_GCV = error(IdealModel_weighted, f_rec_GCV)
    # err_oracle = error( IdealModel_weighted, f_rec_oracle)
    err_LR = error( IdealModel_weighted, f_rec_LocReg_LC)

    # if iter_rps == 0 and iter_sigma == 0:
    #     true_norm = linalg_norm(IdealModel_weighted)
    #     print("true_norm", linalg_norm(IdealModel_weighted))
    #     print("locreg error",err_LR)
    #     print("gcv error", err_GCV)
    #     print("oracle error", err_oracle)

    # Plot a random simulation
    # random_sim = random.randint(0, n_sim)
    if i == 0:
        plot(i)
        print(f"Finished Plots for iteration {i}")
    else:
        pass

    # Create DataFrame
    # sol_strct_uneql = {}
    # sol_strct_uneql['noiseless_data'] = noiseless_data
    # sol_strct_uneql['noisy_data'] = noisy_data  

    feature_df = pd.DataFrame(columns=["NR", "err_LR", "err_GCV"])
    # feature_df["NR"] = [iter_sim]
    # feature_df["Sigma"] = [sigma_i]
    # feature_df["RPS_val"] = [rps_val]
    # feature_df["err_DP"] = [err_DP]
    # feature_df["err_LC"] = [err_LC]
    feature_df["err_LR"] = [err_LR]
    feature_df["err_GCV"] = [err_GCV]
    # feature_df["err_oracle"] = [err_oracle]

    # noise_df = pd.DataFrame(columns= ["Noise"])
    return feature_df, noise

# Example usage
# T2_values = np.linspace(0.1, 200, 40)
# distribution = semicircle_distribution(T2)

# # Plot the distribution
# plt.plot(T2, distribution)
# plt.xlabel('T2')
# plt.ylabel('Semicircle Distribution')
# plt.title('Normalized Semicircle Distribution over T2 Axis')
# plt.grid(True)
# plt.show()

# plt.savefig("semicircularfig.png")
# print("save fig")

if __name__ == '__main__':
    freeze_support()
    T2, TE, Lambda, A, m,  SNR = load_Gaus(Gaus_info)

    print("Finished Assignments...")  
    for i in range(n_sim):
        df, noise = generate_estimates(i)
    # lis.append(generate_estimates(target_iterator[0]))

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(10) 

    # print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
    # df = pd.concat(lis, ignore_index= True)

    df.to_pickle(data_folder + f'/' + data_tag +'.pkl')    
    if preset_noise == False:
        np.save(data_folder + f'/' + data_tag + "noise_arr", noise_arr)
        print("noise array saved")
    else:
        print("Used preset noise array")
        # np.save(data_folder + f'/' + data_tag + "noiselessdata", noiseless_data)
        # np.save(data_folder + f'/' + data_tag + "noisydata", noisy_data)
        # print("test noise array saved")
        pass
