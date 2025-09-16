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
# from Utilities_functions.discrep_L2 import *
# from Utilities_functions.GCV_NNLS import GCV_NNLS
# from Utilities_functions.Lcurve import Lcurve
# import pandas as pd
# import cvxpy as cp
# from scipy.linalg import svd
# from regu.csvd import csvd
# from regu.discrep import discrep
# from Simulations.LRalgo import *
# from Utilities_functions.pasha_gcv import Tikhonov
# from regu.l_curve import l_curve
# from tqdm import tqdm
# from Utilities_functions.tikhonov_vec import tikhonov_vec
# import mosek
# import seaborn as sns
# from regu.nonnegtik_hnorm import nonnegtik_hnorm
# import multiprocess as mp
# from multiprocessing import Pool, freeze_support
# from multiprocessing import set_start_method
# import functools
# from datetime import date
# import random
# import cProfile
# import pstats
# from io import StringIO
# from Simulations.resolutionanalysis import find_min_between_peaks, check_resolution
# import logging
# import time
# from scipy.stats import wasserstein_distance
# import matplotlib.ticker as ticker  # Add this import
# import scipy
# import timeit
# import unittest
# from scipy.integrate import simpson

# import matlab.engine
# print("successfully")
from utils.load_imports.loading import *
logging.basicConfig(
    filename='braindatascript.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Log messages to a file named app.log
        logging.StreamHandler()  # Output log to console
    ]
)
# Create a logger
logger = logging.getLogger()
# Set up logging to file
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Set up logging to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(file_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

print("setting license path")
logging.info("setting license path")
mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
logging.info(f'MOSEK License Set from {mosek_license_path}')

#Naming Convention for Save Folder for Images:
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

pat_tag = "MRR"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
series_tag = "BrainData_Images"
simulation_save_folder = f"SimulationSets/{pat_tag}/{series_tag}"
# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + simulation_save_folder 
logging.info(f"Save Folder for Brain Images {cwd_full})")

#Simulation Test 1
#Simulation Test 2
#Simulation Test 3
addingNoise = True
#Load Brain data and SNR map from Chuan
# brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization-1/data/brain/braindata/mew_cleaned_brain_data_unfiltered.mat")["brain_data"]
# brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat")["final_data_2"]
brain_data = scipy.io.loadmat(r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\braindata\cleaned_brain_data (1).mat")["final_data_2"]
# SNR_map = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/SNR_map.mat")["SNR_map"]
# brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat")["new_BW"]
# SNR_map = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization-1/data/brain/SNRmap/new_SNR_Map.mat")["SNR_MAP"]
SNR_map =scipy.io.loadmat(r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\SNRmap\new_SNR_Map.mat")["SNR_MAP"]
logging.info(f"brain_data shape {brain_data.shape}")
logging.info(f"SNR_map shape {SNR_map.shape}")
logging.info(f"brain_data and SNR_map from Chuan have been successfully loaded")
print("SNR_map.shape",SNR_map.shape)

#Iterator for Parallel Processing

# brain_data = brain_data[80,160,:]
# brain_data2 = brain_data[80,160,:]
p,q,s = brain_data.shape
# print("p",p)
# print("q",q)
# print("s",s)
# p = 1
# q = 1
# s = 32
#replace minimum SNR of 0 to be 1
SNR_map = np.where(SNR_map == 0, 1, SNR_map)
# SNR_map = SNR_map[SNR_map!= 0]
# SNR_map = SNR_map[80,160]
# print("SNR_map.shape", SNR_map.shape)
# #create mask for newSNRmap
# newSNRmap = np.zeros((313,313)) @ SNR_map
# # print("newSNRmap", newSNRmap.shape)
# newSNRmap[80,160] = SNR_map[80,160]
# SNR_map = newSNRmap
print("max SNR_map",np.max(SNR_map))
print("min SNR_map",np.min(SNR_map))
#set a minimum SNR of <10; keep the pixel and set MWF to 0.

# brain_data = brain_data2
# print("brain_data.shape",brain_data.shape)
# print("SNR_map.shape",SNR_map.shape)
# SNR = 300
SpanReg_level = 800
# SpanReg_level = 200

SNRchoice = SpanReg_level
SNR = SpanReg_level

target_iterator = [(a,b) for a in range(p) for b in range(q)]
# target_iterator = [(80,160)]
print(target_iterator)
logging.debug(f'Target Iterator Length len({target_iterator})')

#Naming for Saving Data Collected from Script Folder
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')
data_path = f"data/Brain/results_{day}{month}{year}"
add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_filtered_myelinmaps_GCV_LR012_UPEN"
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_filtered_NA_GCV_LR012_UPEN"
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_nofilt_myelinmaps_GCV_LR012_UPEN"
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_nofilt_NA_GCV_LR012_UPEN"

# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_filtered_noiseadditionUPEN"
data_head = "est_table"
data_tag = (f"{data_head}_{add_tag}{day}{month}{year}")
data_folder = (os.getcwd() + f'/{data_path}')
os.makedirs(data_folder, exist_ok = True)
logging.info(f"Save Folder for Final Estimates Table {data_folder})")

#Parallelization Switch
# parallel = False
parallel = False
num_cpus_avail = os.cpu_count()
#LocReg Hyperparameters
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
exp = 0.5
feedback = True
lam_ini_val = "LCurve"
gamma_init = 0.5
maxiter = 500


#CVXPY global parameters
eps = 1e-2

#Error Metric
err_type = "Wass. Score"

#Lambda Space
Lambda = np.logspace(-6,1,50).reshape(-1,1)
# curr_SNR = 1

#Key Functions
def wass_error(IdealModel,reconstr):
    err = wasserstein_distance(IdealModel,reconstr)
    return err

def l2_error(IdealModel,reconstr):
    true_norm = linalg_norm(IdealModel)
    err = linalg_norm(IdealModel - reconstr) / true_norm
    return err

def cvxpy_tikhreg(Lambda, G, data_noisy):
    """
    Alternative way of performing non-negative Tikhonov regularization.
    """
    # lam_vec = Lambda * np.ones(G.shape[1])
    # A = (G.T @ G + np.diag(lam_vec))
    # ep4 = np.ones(A.shape[1]) * eps
    # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vec
    # y = cp.Variable(G.shape[1])
    # cost = cp.norm(A @ y - b, 2)**2
    # constraints = [y >= 0]
    # problem = cp.Problem(cp.Minimize(cost), constraints)
    # problem.solve(solver=cp.MOSEK, verbose=False)
    # sol = y.value
    # sol = sol - eps
    # sol = np.maximum(sol, 0)

    return nonnegtik_hnorm

def add_noise(data, SNR):
    SD_noise = np.max(np.abs(data)) / SNR  # Standard deviation of noise
    noise = np.random.normal(0, SD_noise, size=data.shape)  # Add noise
    dat_noisy = data + noise
    return dat_noisy, noise

# def calc_dat_noisy(A, TE, IdealModel_weighted, SNR, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     dat_noiseless = A @ IdealModel_weighted  # Compute noiseless data
#     SD_noise = np.max(np.abs(dat_noiseless)) / SNR  # Standard deviation of noise
#     noise = np.random.normal(0, SD_noise, size=dat_noiseless.shape)  # Add noise
#     dat_noisy = dat_noiseless + noise
#     return dat_noisy, noise, SD_noise

def choose_error(pdf1, pdf2,err_type):
    """
    choose_error: Helps select error type based on input
    err_type(string): "Wassterstein" or "Wass. Score"
    vec1(np.array): first probability density function (PDF) you wish to compare to.
    vec2(np.array): second probability density funciton (PDF) you wish to compare to vec1.
    
    typically we select pdf1 to be the ground truth, and we select pdf2 to be our reconstruction method from regularization.
    ex. pdf1 = gt
        pdf2 = GCV solution (f_rec_GCV)
    """
    if err_type == "Wass. Score" or "Wassterstein":
        err = wass_error(pdf1, pdf2)
    else:
        err = l2_error(pdf1, pdf2)
    return err

def minimize_OP(Lambda, data_noisy, G, nT2, g):
    """
    Calculates the oracle lambda solution. Iterates over many lambdas in Alpha_vec
    """
    OP_x_lc_vec = np.zeros((nT2, len(Lambda)))
    OP_rhos = np.zeros((len(Lambda)))
    for j in (range(len(Lambda))):
        # try:Lambda
        #     # Fallback to nonnegtik_hnorm
        #     sol, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j], '0', nargin=4)
        #     if np.all(sol == 0):
        #         logging.debug("Solution from nonnegtik_hnorm is a zero vector, switching to CVXPY solver.")
        #         print(f"Solution from nonnegtik_hnorm is a zero vector, switching to CVXPY solver.")
        #         raise ValueError("Zero vector detected, switching to CVXPY.")
        # except Exception as e:
            #CVXPY solution
            # logging.error("Error in nonnegtik_hnorm, using CVXPY")
            # print(f"Error in nonnegtik_hnorm: {e}")
        # sol = cvxpy_tikhreg(Lambda[j], G, data_noisy)
        sol = nonnegtik_hnorm(G, data_noisy, Lambda[j], '0', nargin=4)[0]
        OP_x_lc_vec[:, j] = sol
        # Calculate the error (rho)
        OP_rhos[j] = choose_error(g, OP_x_lc_vec[:, j], err_type = "Wass. Score")
    #Find the minimum value of errors, its index, and the corresponding lambda and reconstruction
    min_rhos = min(OP_rhos)
    min_index = np.argmin(OP_rhos)
    min_x = Lambda[min_index][0]
    OP_min_alpha1 = min_x
    OP_min_alpha1_ind = min_index
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    return f_rec_OP_grid, OP_min_alpha1, min_rhos , min_index

def compute_MWF(f_rec, T2, Myelin_idx):
    """
    Normalize f_rec and compute MWF.
    Inputs: 
    1. f_rec (np.array type): the reconstruction generated from Tikhonov regularization 
    2. T2 (np.array): the transverse relaxation time constant vector in ms
    3. Myelin_idx (np.array):the indices of T2 where myelin is present (e.g. T2 < 40 ms)
    
    Outputs:
    1. MWF (float): the myelin water fraction
    2. f_rec_normalized (np.array type): the normalized reconstruction generated from Tikhonov regularization. 
        Normalized to 1.
    
    Test Example:
    1) f_rec = np.array([1,2,3,4,5,6])
    2) np.trapz(f_rec_normalized, T2) ~ 1.
    """
    # f_rec_normalized = f_rec / np.trapz(f_rec, T2)
    # print("np.sum(f_rec)*dTE",np.sum(f_rec)*dT)
    # f_rec_normalized = f_rec / (np.sum(f_rec)*dT)
    try:
        f_rec_normalized = f_rec / (np.sum(f_rec) * dT)
        # f_rec_normalized = f_rec / np.trapz(f_rec, T2)
        # f_rec_normalized = f_rec / simpson(y = f_rec, x = T2)
    except ZeroDivisionError:
        epsilon = 0.0001
        f_rec_normalized = f_rec / (epsilon)
        print("Division by zero encountered, using epsilon:", epsilon)
    total_MWF = np.cumsum(f_rec_normalized)
    MWF = total_MWF[Myelin_idx[-1][-1]]
    return f_rec_normalized, MWF

def generate_noisy_brain_estimates(i_param_combo, seed= None):
    # print(f"Processing {i_param_combo}") 
    feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "DP_estimate", "LC_estimate", 
                                       "LR_estimate", "GCV_estimate", "OR_estimate", "LS_estimate", "MWF_DP", "MWF_LC", 
                                       "MWF_LR", "MWF_GCV", "MWF_OR", "MWF_LS", "Lam_DP", "Lam_LC", "Lam_GCV", "Lam_OR", "Lam_LR", "Flagged"])
    
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
        # print("x_coord", x_coord) 
        # print("y_coord", y_coord) 

    #eliminate voxels with pure noise
    if brain_data[x_coord, y_coord][0] < 50:
    # if brain_data[0] < 50:
        print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
        return feature_df
    # if SNR_map[x_coord, y_coord] == 0:
    #     print(f"SNR_map is 0 and pure noise for {x_coord} and {y_coord}")
    #     return feature_df
    else:
        passing_x_coord = x_coord
        passing_y_coord = y_coord
        curr_data = brain_data[passing_x_coord,passing_y_coord,:]
        # curr_data = brain_data
        #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
        # print("curr_data before factor", curr_data)
        # print("curr_SNR", curr_SNR)
        try:
            # sol1 = nnls(A, curr_data)[0]
            sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            # print(f"curr_data {curr_data}")
            # print(f"sol1 {sol1}")
            # print(f"A {A}")
            np.save("A_sol_debug.npy", A)  
            np.save("curr_data_debug.npy", curr_data)  
            np.save("sol1_debug.npy", sol1)  
        except RuntimeError as e:
            print(f"Crashed, {e}")
            #use the cvxpy one...solver...

            #if use previous...use the four adjacent back...
            # sol1 = prevsol.copy()
        # prevsol = sol1.copy()
        # print("prevsol",prevsol)

        #1/11: heres the problem here....
#   File "/home/kimjosy/LocReg_Regularization-1/Simulations/brainscripts/braindatascript.py", line 348, in generate_noisy_brain_estimates
#     noisy_f_rec_LS = nnls(A, noisy_curr_data,maxiter=1e5)[0]
#                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/kimjosy/miniconda3/envs/myenv/lib/python3.11/site-packages/scipy/optimize/_nnls.py", line 93, in nnls
#     raise RuntimeError("Maximum number of iterations reached.")
# RuntimeError: Maximum number of iterations reached.

        # result = scipy.optimize.lsq_linear(A, curr_data, bounds=(0, np.inf), max_iter=int(1e5))
        # sol1 = result.x
        # print("sol1", sol1)
        factor = np.trapz(sol1,T2)
        # factor = trapz(sol1,T2)
        # factor = np.sum(sol1) * dT
        curr_data = curr_data/factor
        # print("curr_datafactor", curr_data)
        if addingNoise == True:
            noisy_curr_data, noise = add_noise(curr_data, SNR = SNRchoice)
        else:
            pass
        #LS:
        # f_rec_LS = nnls(A, curr_data)[0]
        # f_rec_LS, MWF_LS = compute_MWF(f_rec_LS, T2, Myelin_idx)

        #real one 1/3/25
        print("Starting LS")
        try:
            # noisy_f_rec_LS = nnls(A, noisy_curr_data)[0]
            noisy_f_rec_LS = nonnegtik_hnorm(A, noisy_curr_data, 0, '0', nargin=4)[0]
            # print(f"noisy_curr_data {noisy_curr_data}")
            # print(f"noisy_f_rec_LS {noisy_f_rec_LS}")
            # print(f"A {A}")
            # np.save(A, "A_LS_debug.npy")  
            # np.save(noisy_curr_data, "noisy_curr_data_debug.npy")  
            # np.save(noisy_f_rec_LS, "noisy_f_rec_LS_debug.npy")  
            np.save("A_LS_debug.npy", A)  
            np.save("noisy_curr_data_debug.npy", noisy_curr_data)  
            np.save("noisy_f_rec_LS_debug.npy", noisy_f_rec_LS)
        except RuntimeError as e:
            print("Crashed",e)
            print(f"Takes too many iterations, {e}")
            #use the cvxpy one...solver...

            # noisy_f_rec_LS = prevsol2.copy()
        # prevsol2 = noisy_f_rec_LS.copy()
        # print("prevsol2",prevsol2)
        noisy_f_rec_LS, noisy_MWF_LS = compute_MWF(noisy_f_rec_LS, T2, Myelin_idx)

        # curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        #LC
        # f_rec_LC, lambda_LC = Lcurve(curr_data, A, Lambda)
        # f_rec_LC, MWF_LC = compute_MWF(f_rec_LC, T2, Myelin_idx)

        #real one 1/3/25
        print("Starting LCurve")
        noisy_f_rec_LC, noisy_lambda_LC = Lcurve(noisy_curr_data, A, Lambda)
        noisy_f_rec_LC, noisy_MWF_LC = compute_MWF(noisy_f_rec_LC, T2, Myelin_idx)


        # print("MWF_LC", MWF_LC)

        #GCV
        # f_rec_GCV, lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        # f_rec_GCV = f_rec_GCV[:, 0]
        # lambda_GCV = np.squeeze(lambda_GCV)
        # f_rec_GCV, MWF_GCV = compute_MWF(f_rec_GCV, T2, Myelin_idx)
        # print("MWF_GCV", MWF_GCV)

        # #real one 1/3/25
        print("Starting GCV")
        noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(noisy_curr_data, A, Lambda)
        noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
        noisy_f_rec_GCV, noisy_MWF_GCV = compute_MWF(noisy_f_rec_GCV, T2, Myelin_idx)
       
        #Oracle or GCV
        # IdealModel_weighted = f_rec_GCV
        # f_rec_OR, lambda_OR, min_rhos , min_index = minimize_OP(Lambda, curr_data, A, len(T2), IdealModel_weighted)
        # f_rec_OR, MWF_OR = compute_MWF(f_rec_OR, T2, Myelin_idx)
        # print("MWF_OR", MWF_OR)

        #DP
        # f_rec_DP, lambda_DP = discrep_L2(curr_data, A, curr_SNR, Lambda, noise = "brain")
        # f_rec_DP, MWF_DP = compute_MWF(f_rec_DP, T2, Myelin_idx)

        #real one 1/3/25
        # print("curr_SNR",curr_SNR)
        # noisy_f_rec_DP, noisy_lambda_DP = discrep_L2(noisy_curr_data, A, curr_SNR, Lambda, noise = True)
        # print("Starting DP")
        noisy_f_rec_DP, noisy_lambda_DP = discrep_L2_brain(noisy_curr_data, A, curr_SNR, Lambda, noise = True)
        noisy_f_rec_DP, noisy_MWF_DP = compute_MWF(noisy_f_rec_DP, T2, Myelin_idx)
        # print("MWF_DP", MWF_DP)
        
        #LR
        # LRIto_ini_lam = lambda_LC
        # f_rec_ini = f_rec_LC
        # f_rec_LR, lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter)
        # f_rec_LR, MWF_LR = compute_MWF(f_rec_LR, T2, Myelin_idx)
        # print("MWF_LR", MWF_LR)
        print("Starting LocReg")
        noisy_LRIto_ini_lam = noisy_lambda_LC
        noisy_f_rec_ini = noisy_f_rec_LC
        noisy_f_rec_LR, noisy_lambda_LR, noisy_test_frec1, noisy_test_lam1, noisy_numiterate = LocReg_Ito_mod(noisy_curr_data, A, noisy_LRIto_ini_lam, gamma_init, maxiter)
        noisy_f_rec_LR, noisy_MWF_LR = compute_MWF(noisy_f_rec_LR, T2, Myelin_idx)

        # gt = f_rec_GCV

        
        # def curve_plot(x_coord, y_coord, frec):
        #     plt.plot(T2, frec)
        #     plt.savefig(f"LR_recon_xcoord{x_coord}_ycoord{y_coord}.png")
        #     print(f"savefig xcoord{x_coord}_ycoord{y_coord}")
        
        # if passing_x_coord % 10 == 0 and passing_y_coord % 10 == 0:
        #     curve_plot(passing_x_coord, passing_y_coord, f_rec_LR)    
        #     curve_plot(passing_x_coord, passing_y_coord, noisy_f_rec_LR)    

        # # OR_err = choose_error(gt, f_rec_OR, err_type)
        # LC_err = choose_error(f_rec_LC, noisy_f_rec_LC, err_type)
        # LR_err = choose_error(f_rec_LR, noisy_f_rec_LR, err_type)
        # DP_err = choose_error(f_rec_DP, noisy_f_rec_DP, err_type)
        # GCV_err = choose_error(f_rec_GCV, noisy_f_rec_GCV, err_type)

        # Assuming you are inside a loop or block where you want to check these conditions
        # if not(OR_err <= LC_err and OR_err <= LR_err and OR_err <= DP_err):
        #     logging.warning("Oracle error should not be larger than other single parameter methods and GCV is ground truth solution")
        #     logging.info("oracle error", OR_err)
        #     logging.info("DP error", DP_err)
        #     logging.info("LC error", LC_err)
        #     logging.info("LR error", LR_err)
        #     logging.info("oracle lambda", lambda_OR)
        #     logging.info(f"oracle's minimum error score ({err_type})", min_rhos)
        #     logging.info(f"oracle's index for minimum error score ({err_type})", min_index)
        #     print("Oracle error should not be larger than other single parameter methods and GCV is ground truth solution")
        #     print("oracle error", OR_err)
        #     print("DP error", DP_err)
        #     print("LC error", LC_err)
        #     print("LR error", LR_err)
        #     print("oracle lambda", lambda_OR)
        #     print(f"oracle's minimum error score ({err_type})", min_rhos)
        #     print(f"oracle's index for minimum error score ({err_type})", min_index)
        #     feature_df["Flagged"] = [1]
        # else:
        #     feature_df["Flagged"] = [0]
        # feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "DP_estimate", "LC_estimate", 
        #                                     "LR_estimate", "GCV_estimate", "OR_estimate", "MWF_DP", "MWF_LC", 
        #                                     "MWF_LR", "MWF_GCV", "MWF_OR", "Lam_DP", "Lam_LC", "Lam_GCV", "Lam_OR", "Lam_LR"])
        feature_df["X_val"] = [passing_x_coord]
        feature_df["Y_val"] = [passing_y_coord]
        feature_df["curr_data"] = [curr_data]
        feature_df["noisy_curr_data"] = [noisy_curr_data]
        feature_df["noise"] = [noise]
        # feature_df["DP_estimate"] = [f_rec_DP]
        # feature_df["LC_estimate"] = [f_rec_LC]
        # feature_df["LR_estimate"] = [f_rec_LR]
        # feature_df["GCV_estimate"] = [f_rec_GCV]
        # feature_df["LS_estimate"] = [f_rec_LS]
        # feature_df["OR_estimate"] = [f_rec_OR]

        # real one 1/3/25
        feature_df["noisy_LS_estimate"] = [noisy_f_rec_LS]
        feature_df["noisy_MWF_LS"] = [noisy_MWF_LS]

        feature_df["noisy_DP_estimate"] = [noisy_f_rec_DP]
        feature_df["noisy_LC_estimate"] = [noisy_f_rec_LC]
        feature_df["noisy_LR_estimate"] = [noisy_f_rec_LR]
        feature_df["noisy_GCV_estimate"] = [noisy_f_rec_GCV]
        feature_df["noisy_MWF_DP"] = [noisy_MWF_DP]
        feature_df["noisy_MWF_LC"] = [noisy_MWF_LC]
        feature_df["noisy_MWF_LR"] = [noisy_MWF_LR]
        feature_df["noisy_MWF_GCV"] = [noisy_MWF_GCV]
        feature_df["noisy_Lam_DP"] = [noisy_lambda_DP]
        feature_df["noisy_Lam_LC"] = [noisy_lambda_LC]
        feature_df["noisy_Lam_LR"] = [noisy_lambda_LR]
        feature_df["noisy_Lam_GCV"] = [noisy_lambda_GCV]


        # feature_df["OR_estimate"] = [f_rec_OR]
        # feature_df["MWF_DP"] = [MWF_DP]
        # feature_df["MWF_LC"] = [MWF_LC]
        # feature_df["MWF_LR"] = [MWF_LR]
        # feature_df["MWF_GCV"] = [MWF_GCV]
        # feature_df["MWF_LS"] = [MWF_LS]

        # feature_df["MWF_OR"] = [MWF_OR]
        # feature_df["Lam_DP"] = [lambda_DP]
        # feature_df["Lam_LC"] = [lambda_LC]
        # feature_df["Lam_LR"] = [lambda_LR]
        # feature_df["Lam_GCV"] = [lambda_GCV]

        # feature_df["LC_err"] = [LC_err]
        # feature_df["LR_err"] = [LR_err]
        # feature_df["DP_err"] = [DP_err]
        # feature_df["GCV_err"] = [GCV_err]
        # feature_df["Lam_OR"] = [lambda_OR]
        print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
        return feature_df

def generate_noisy_brain_estimates2(i_param_combo, seed= None):
    # print(f"Processing {i_param_combo}") 
    feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "DP_estimate", "LC_estimate", 
                                       "LR_estimate", "GCV_estimate", "OR_estimate", "LS_estimate", "MWF_DP", "MWF_LC", 
                                       "MWF_LR", "MWF_GCV", "MWF_OR", "MWF_LS", "Lam_DP", "Lam_LC", "Lam_GCV", "Lam_OR", "Lam_LR", "Flagged"])
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
    #eliminate voxels with pure noise
    if brain_data[x_coord, y_coord][0] < 50:
        print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
        return feature_df
    else:
        passing_x_coord = x_coord
        passing_y_coord = y_coord
        curr_data = brain_data[passing_x_coord,passing_y_coord,:]
        curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
        try:
            sol1 = nnls(A, curr_data, maxiter=1e6)[0]
        except:
            try:
                sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            except:
                print("sol1 is ones")
                sol1 = np.ones(len(T2))
                Flag_Val = 1

        X = sol1
        tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"

        all_close_to_zero = np.all(np.abs(X) < tolerance)

        factor = np.sum(sol1) * dT
        curr_data = curr_data/factor
        curr_data, noise = add_noise(curr_data, SNR = SNRchoice)
        # curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        #LS:
        try:
            noisy_f_rec_LS = nnls(A, curr_data, maxiter=1e6)[0]
        except:
            try:
                noisy_f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            except:
                noisy_f_rec_LS = np.ones(len(T2))
                Flag_Val = 2
        # f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        
        X = noisy_f_rec_LS
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_LS[:-1] == 0) or all_close_to_zero:
            noisy_MWF_LS = 0
            method = "LS"
            Flag_Val = 1
        else:
            noisy_f_rec_LS, noisy_MWF_LS = compute_MWF(noisy_f_rec_LS, T2, Myelin_idx)
            Flag_Val = 0

        #LC
        noisy_f_rec_LC, noisy_lambda_LC = Lcurve(curr_data, A, Lambda)
        X = noisy_f_rec_LC

        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_LC[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LC is 0 or all_close_to_zero")
            noisy_MWF_LC = 0
            method = "LC"
            Flag_Val = 1
        else:
            noisy_f_rec_LC, noisy_MWF_LC = compute_MWF(noisy_f_rec_LC, T2, Myelin_idx)
            Flag_Val = 0

        #GCV
        noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)

        X = noisy_f_rec_GCV
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_GCV[:-1] == 0) or all_close_to_zero:
            noisy_MWF_GCV = 0
            method = "GCV"
            Flag_Val = 1
        else:
            noisy_f_rec_GCV, noisy_MWF_GCV = compute_MWF(noisy_f_rec_GCV, T2, Myelin_idx)
            Flag_Val = 0
        noisy_f_rec_DP, noisy_lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
        X = noisy_f_rec_DP

        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_DP[:-1] == 0) or all_close_to_zero:
            # print("f_rec_DP is 0 or all_close_to_zero")
            noisy_MWF_DP = 0
            method = "DP"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_DP, curr_data, lambda_DP, curr_SNR, MWF_DP, filepath)
        else:
            noisy_f_rec_DP, noisy_MWF_DP = compute_MWF(noisy_f_rec_DP, T2, Myelin_idx)
            Flag_Val = 0
        # print("MWF_DP", MWF_DP)

        #LR
        LRIto_ini_lam = noisy_lambda_LC
        f_rec_ini = noisy_f_rec_LC
        # print("curr_data", curr_data)
        noisy_f_rec_LR, noisy_lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter)
        X = noisy_f_rec_LR
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)

        if np.all(noisy_f_rec_LR[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LR is 0")
            noisy_MWF_LR = 0
            method = "LocReg"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_LR, curr_data, lambda_LR, curr_SNR, MWF_LR, filepath)
        else:
            noisy_f_rec_LR, noisy_MWF_LR = compute_MWF(noisy_f_rec_LR, T2, Myelin_idx)
            Flag_Val = 0
        # print("MWF_LR", MWF_LR)
        gt = noisy_f_rec_GCV
        feature_df["X_val"] = [passing_x_coord]
        feature_df["Y_val"] = [passing_y_coord]
        feature_df["noisy_curr_data"] = [curr_data]
        # feature_df["noisy_curr_data"] = [noisy_curr_data]
        feature_df["noise"] = [noise]
        feature_df["curr_SNR"] = [curr_SNR]
        # real one 1/3/25
        feature_df["noisy_LS_estimate"] = [noisy_f_rec_LS]
        feature_df["noisy_MWF_LS"] = [noisy_MWF_LS]
        feature_df["noisy_DP_estimate"] = [noisy_f_rec_DP]
        feature_df["noisy_LC_estimate"] = [noisy_f_rec_LC]
        feature_df["noisy_LR_estimate"] = [noisy_f_rec_LR]
        feature_df["noisy_GCV_estimate"] = [noisy_f_rec_GCV]
        feature_df["noisy_MWF_DP"] = [noisy_MWF_DP]
        feature_df["noisy_MWF_LC"] = [noisy_MWF_LC]
        feature_df["noisy_MWF_LR"] = [noisy_MWF_LR]
        feature_df["noisy_MWF_GCV"] = [noisy_MWF_GCV]
        feature_df["noisy_Lam_DP"] = [noisy_lambda_DP]
        feature_df["noisy_Lam_LC"] = [noisy_lambda_LC]
        feature_df["noisy_Lam_LR"] = [noisy_lambda_LR]
        feature_df["noisy_Lam_GCV"] = [noisy_lambda_GCV]
        feature_df["Flagged"] = [Flag_Val]
        print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
        return feature_df

# 4) SpanReg
    # original brain data (filtered NESMA)
    # collect T2 from spanreg
    # then A * F(T2) = d, then add noise with uniform SNR
    # recovery using that noisy data 
    # (using DP, GCV, LocReg, LCurve, non-regular NNLS)
    # 200, 800 SNR levels

tolerance = 1e-6
def spanreg_unif_SNR(i_param_combo, seed= None):
    # print(f"Processing {i_param_combo}") 
    feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "DP_estimate", "LC_estimate", 
                                       "LR_estimate", "GCV_estimate", "reference_estimate", "LS_estimate", "MWF_DP", "MWF_LC", 
                                       "MWF_LR", "MWF_GCV", "MWF_LS", "reference_MWF",  "Lam_DP", "Lam_LC", "Lam_GCV", "Lam_Ref_LR", "Lam_LR", "Flagged"])
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
    #eliminate voxels with pure noise
    if brain_data[x_coord, y_coord][0] < 50:
        print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
        return feature_df
    else:
        passing_x_coord = x_coord
        passing_y_coord = y_coord
        curr_data = brain_data[passing_x_coord,passing_y_coord,:]
        curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
        #LC
        noisy_f_rec_LC, noisy_lambda_LC = Lcurve(curr_data, A, Lambda)
        #LR
        LRIto_ini_lam = noisy_lambda_LC
        f_rec_ini = noisy_f_rec_LC
        # print("curr_data", curr_data)
        noisy_f_rec_LR, Lam_Ref_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter)
        X = noisy_f_rec_LR
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        reference_estimate, reference_MWF = compute_MWF(noisy_f_rec_LR, T2, Myelin_idx)

        data_noiseless = A @ noisy_f_rec_LR

        factor = np.sum(noisy_f_rec_LR) * dT
        curr_data = curr_data/factor

        curr_data, noise = add_noise(curr_data, SNR = SpanReg_level)

        # curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        #LS:
        try:
            noisy_f_rec_LS = nnls(A, curr_data, maxiter=1e6)[0]
        except:
            try:
                noisy_f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            except:
                noisy_f_rec_LS = np.ones(len(T2))
                Flag_Val = 2
        # f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        
        X = noisy_f_rec_LS
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_LS[:-1] == 0) or all_close_to_zero:
            noisy_MWF_LS = 0
            method = "LS"
            Flag_Val = 1
        else:
            noisy_f_rec_LS, noisy_MWF_LS = compute_MWF(noisy_f_rec_LS, T2, Myelin_idx)
            Flag_Val = 0

        #LC
        noisy_f_rec_LC, noisy_lambda_LC = Lcurve(curr_data, A, Lambda)
        X = noisy_f_rec_LC

        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_LC[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LC is 0 or all_close_to_zero")
            noisy_MWF_LC = 0
            method = "LC"
            Flag_Val = 1
        else:
            noisy_f_rec_LC, noisy_MWF_LC = compute_MWF(noisy_f_rec_LC, T2, Myelin_idx)
            Flag_Val = 0

        #GCV
        noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)

        X = noisy_f_rec_GCV
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_GCV[:-1] == 0) or all_close_to_zero:
            noisy_MWF_GCV = 0
            method = "GCV"
            Flag_Val = 1
        else:
            noisy_f_rec_GCV, noisy_MWF_GCV = compute_MWF(noisy_f_rec_GCV, T2, Myelin_idx)
            Flag_Val = 0
        noisy_f_rec_DP, noisy_lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
        X = noisy_f_rec_DP

        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_DP[:-1] == 0) or all_close_to_zero:
            # print("f_rec_DP is 0 or all_close_to_zero")
            noisy_MWF_DP = 0
            method = "DP"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_DP, curr_data, lambda_DP, curr_SNR, MWF_DP, filepath)
        else:
            noisy_f_rec_DP, noisy_MWF_DP = compute_MWF(noisy_f_rec_DP, T2, Myelin_idx)
            Flag_Val = 0
        # print("MWF_DP", MWF_DP)

        #LR
        LRIto_ini_lam = noisy_lambda_LC
        f_rec_ini = noisy_f_rec_LC
        # print("curr_data", curr_data)
        noisy_f_rec_LR, noisy_lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter)
        X = noisy_f_rec_LR
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)

        if np.all(noisy_f_rec_LR[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LR is 0")
            noisy_MWF_LR = 0
            method = "LocReg"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_LR, curr_data, lambda_LR, curr_SNR, MWF_LR, filepath)
        else:
            noisy_f_rec_LR, noisy_MWF_LR = compute_MWF(noisy_f_rec_LR, T2, Myelin_idx)
            Flag_Val = 0
        # print("MWF_LR", MWF_LR)
        gt = noisy_f_rec_GCV

        feature_df["X_val"] = [passing_x_coord]
        feature_df["Y_val"] = [passing_y_coord]
        feature_df["reference_estimate"] = [reference_estimate]
        feature_df["reference_MWF"] = [reference_MWF]
        feature_df["Lam_Ref_LR"] = [Lam_Ref_LR]
        feature_df["noisy_curr_data"] = [curr_data]
        # feature_df["noisy_curr_data"] = [noisy_curr_data]
        feature_df["noise"] = [noise]
        feature_df["curr_SNR"] = [curr_SNR]
        # real one 1/3/25
        feature_df["noisy_LS_estimate"] = [noisy_f_rec_LS]
        feature_df["noisy_MWF_LS"] = [noisy_MWF_LS]
        feature_df["noisy_DP_estimate"] = [noisy_f_rec_DP]
        feature_df["noisy_LC_estimate"] = [noisy_f_rec_LC]
        feature_df["noisy_LR_estimate"] = [noisy_f_rec_LR]
        feature_df["noisy_GCV_estimate"] = [noisy_f_rec_GCV]
        feature_df["noisy_MWF_DP"] = [noisy_MWF_DP]
        feature_df["noisy_MWF_LC"] = [noisy_MWF_LC]
        feature_df["noisy_MWF_LR"] = [noisy_MWF_LR]
        feature_df["noisy_MWF_GCV"] = [noisy_MWF_GCV]
        feature_df["noisy_Lam_DP"] = [noisy_lambda_DP]
        feature_df["noisy_Lam_LC"] = [noisy_lambda_LC]
        feature_df["noisy_Lam_LR"] = [noisy_lambda_LR]
        feature_df["noisy_Lam_GCV"] = [noisy_lambda_GCV]
        feature_df["Flagged"] = [Flag_Val]
        print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
        return feature_df



def profile_section(section_name, func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    print(f"Profiling results for {section_name}:")
    ps.print_stats(10)  # Show top 10 functions
    print(s.getvalue())
    return result


def curve_plot(method, x_coord, y_coord, frec, curr_data, lambda_vals, curr_SNR, MWF, filepath):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # First plot: T2 vs f_rec
    axs[0].plot(T2, frec)
    axs[0].set_title('T2 vs f_rec')
    axs[0].set_xlabel('T2')
    axs[0].set_ylabel('f_rec')

    # Second plot: TE vs curr_data
    axs[1].plot(TE, curr_data)
    axs[1].set_title('TE vs Decay Data')
    axs[1].set_xlabel('TE')
    axs[1].set_ylabel('curr_data')

    # Third plot: T2 vs lambda
    axs[2].plot(T2, lambda_vals * np.ones(len(T2)))
    axs[2].set_title('T2 vs Lambda')
    axs[2].set_xlabel('T2')
    axs[2].set_ylabel('lambda')
    # Set the main title with curr_SNR and MWF value
    fig.suptitle(f'{method} Plots for x={x_coord}, y={y_coord} | SNR={curr_SNR}, MWF={MWF}', fontsize=16)
    # Save the figure
    plt.savefig(f"{filepath}/{method}_recon_xcoord{x_coord}_ycoord{y_coord}.png")
    print(f"savefig xcoord{x_coord}_ycoord{y_coord}")
    plt.close('all')
    return 
#We set GCV as ground truth
filepath = data_folder

def generate_brain_estimates(i_param_combo, seed= None):
    # print(f"Processing {i_param_combo}") 
    feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "DP_estimate", "LC_estimate", 
                                       "LR_estimate", "GCV_estimate", "LS_estimate", "MWF_DP", "MWF_LC", 
                                       "MWF_LR", "MWF_GCV", "MWF_LS", "Lam_DP", "Lam_LC", "Lam_GCV", "Lam_LR", "Flagged"])
    
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
        # print("x_coord", x_coord) 
        # print("y_coord", y_coord) 

    #eliminate voxels with pure noise
    if brain_data[x_coord, y_coord][0] < 50:
    # if brain_data[0] < 50:
        print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
        return feature_df
    # if SNR_map[x_coord, y_coord] == 0:
    #     print(f"SNR_map is 0 and pure noise for {x_coord} and {y_coord}")
    #     return feature_df
    else:
        passing_x_coord = x_coord
        passing_y_coord = y_coord
        curr_data = brain_data[passing_x_coord,passing_y_coord,:]
        curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        # curr_data = brain_data
        #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
        try:
            sol1 = nnls(A, curr_data, maxiter=1e6)[0]
        except:
            try:
                sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            except:
                # print("sol1 is ones")
                sol1 = np.ones(len(T2))
                # print("all methods fail")
                Flag_Val = 1

        # sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        # sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]

        X = sol1
        tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        # if np.all(sol1[:-1] == 0) or all_close_to_zero:
        #     print("factor is close to 0 or all_close_to_zero")
        #     # factor = 1e-6
        #     factor = sol1[-1] * dT
        # else:
        #     factor = np.sum(sol1) * dT
        #     pass
        factor = np.sum(sol1) * dT

        # print("sol1[0]",np.max(sol1))
        # factor2 = sol1[-1] * dT
        # print("factor", factor)
        # print("factor2", factor2)
        curr_data = curr_data/factor

        # if curr_data[0] < 0.5:
        #     print(f"normalized data  satisfies <0.5 requirement for {x_coord} and {y_coord}")
        #     return feature_df

        # factor = np.trapz(sol1,T2)
        # factor = simpson(y = sol1, x = T2)

        # factor = np.sum(sol1) * dT
        curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        #LS:
        try:
            f_rec_LS = nnls(A, curr_data, atol = 1e-3, maxiter=1e6)[0]
        except:
            try:
                f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            except:
                print("all methods fail")
                f_rec_LS = np.ones(len(T2))
                #try another
        # f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        
        X = f_rec_LS
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        # f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        if np.all(f_rec_LS[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LS is 0 or all_close_to_zero")
            MWF_LS = 0
            method = "LS"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_LS, curr_data, 0, curr_SNR, MWF_LS, filepath)
        else:
            f_rec_LS, MWF_LS = compute_MWF(f_rec_LS, T2, Myelin_idx)
            Flag_Val = 0
        # f_rec_LS, MWF_LS = compute_MWF(f_rec_LS, T2, Myelin_idx)
        # curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        #LC
        f_rec_LC, lambda_LC = Lcurve(curr_data, A, Lambda)
        X = f_rec_LC
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        # f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        if np.all(f_rec_LC[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LC is 0 or all_close_to_zero")
            MWF_LC = 0
            method = "LC"
            # curve_plot(method, x_coord, y_coord, f_rec_LC, curr_data, lambda_LC, curr_SNR, MWF_LC, filepath)
            Flag_Val = 1
            #Save the results for the pixel...; plot:SNR, decay, lambda..., f_rec..MWF.
        else:
            f_rec_LC, MWF_LC = compute_MWF(f_rec_LC, T2, Myelin_idx)
            Flag_Val = 0
        # f_rec_LC, MWF_LC = compute_MWF(f_rec_LC, T2, Myelin_idx)
        # print("MWF_LC", MWF_LC)
        #GCV
        f_rec_GCV, lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        f_rec_GCV = f_rec_GCV[:, 0]
        lambda_GCV = np.squeeze(lambda_GCV)

        X = f_rec_GCV
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(f_rec_GCV[:-1] == 0) or all_close_to_zero:
            # print("f_rec_GCV is 0 or all_close_to_zero")
            MWF_GCV = 0
            method = "GCV"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_GCV, curr_data, lambda_GCV, curr_SNR, MWF_GCV, filepath)
        else:
            f_rec_GCV, MWF_GCV = compute_MWF(f_rec_GCV, T2, Myelin_idx)
            Flag_Val = 0

        # f_rec_GCV, MWF_GCV = compute_MWF(f_rec_GCV, T2, Myelin_idx)
        # print("MWF_GCV", MWF_GCV)
       
        #Oracle or GCV
        # IdealModel_weighted = f_rec_GCV
        # f_rec_OR, lambda_OR, min_rhos , min_index = minimize_OP(Lambda, curr_data, A, len(T2), IdealModel_weighted)
        # f_rec_OR, MWF_OR = compute_MWF(f_rec_OR, T2, Myelin_idx)
        # print("MWF_OR", MWF_OR)

        #DP
        # f_rec_DP, lambda_DP = discrep_L2(curr_data, A, curr_SNR, Lambda, noise = "brain")
        f_rec_DP, lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
        X = f_rec_DP
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(f_rec_DP[:-1] == 0) or all_close_to_zero:
            # print("f_rec_DP is 0 or all_close_to_zero")
            MWF_DP = 0
            method = "DP"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_DP, curr_data, lambda_DP, curr_SNR, MWF_DP, filepath)
        else:
            f_rec_DP, MWF_DP = compute_MWF(f_rec_DP, T2, Myelin_idx)
            Flag_Val = 0
        # print("MWF_DP", MWF_DP)

        #LR
        # LRIto_ini_lam = lambda_LC
        # f_rec_ini = f_rec_LC
        LRIto_ini_lam = lambda_GCV
        f_rec_ini = f_rec_GCV
        # print("curr_data", curr_data)
        f_rec_LR, lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter)
        X = f_rec_LR
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)

        if np.all(f_rec_LR[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LR is 0")
            MWF_LR = 0
            method = "LocReg"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_LR, curr_data, lambda_LR, curr_SNR, MWF_LR, filepath)
        else:
            f_rec_LR, MWF_LR = compute_MWF(f_rec_LR, T2, Myelin_idx)
            Flag_Val = 0

        # print("MWF_LR", MWF_LR)

        gt = f_rec_GCV
        
        # def curve_plot(x_coord, y_coord, frec):
        #     plt.plot(T2, frec)
        #     plt.legend()

        #     plt.savefig(f"LR_recon_xcoord{x_coord}_ycoord{y_coord}.png")
        #     print(f"savefig xcoord{x_coord}_ycoord{y_coord}")

        # if passing_x_coord % 10 == 0 and passing_y_coord % 10 == 0:
        #     curve_plot(passing_x_coord, passing_y_coord, f_rec_LR)    

        # OR_err = choose_error(gt, f_rec_OR, err_type)
        # LC_err = choose_error(gt, f_rec_LC, err_type)
        # LR_err = choose_error(gt, f_rec_LR, err_type)
        # DP_err = choose_error(gt, f_rec_DP, err_type)

        # Assuming you are inside a loop or block where you want to check these conditions
        # if not(OR_err <= LC_err and OR_err <= LR_err and OR_err <= DP_err):
        #     logging.warning("Oracle error should not be larger than other single parameter methods and GCV is ground truth solution")
        #     logging.info("oracle error", OR_err)
        #     logging.info("DP error", DP_err)
        #     logging.info("LC error", LC_err)
        #     logging.info("LR error", LR_err)
        #     logging.info("oracle lambda", lambda_OR)
        #     logging.info(f"oracle's minimum error score ({err_type})", min_rhos)
        #     logging.info(f"oracle's index for minimum error score ({err_type})", min_index)
        #     print("Oracle error should not be larger than other single parameter methods and GCV is ground truth solution")
        #     print("oracle error", OR_err)
        #     print("DP error", DP_err)
        #     print("LC error", LC_err)
        #     print("LR error", LR_err)
        #     print("oracle lambda", lambda_OR)
        #     print(f"oracle's minimum error score ({err_type})", min_rhos)
        #     print(f"oracle's index for minimum error score ({err_type})", min_index)
        #     feature_df["Flagged"] = [1]
        # else:
            # feature_df["Flagged"] = [0]

        # feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "DP_estimate", "LC_estimate", 
        #                                     "LR_estimate", "GCV_estimate", "OR_estimate", "MWF_DP", "MWF_LC", 
        #                                     "MWF_LR", "MWF_GCV", "MWF_OR", "Lam_DP", "Lam_LC", "Lam_GCV", "Lam_OR", "Lam_LR"])
        feature_df["X_val"] = [passing_x_coord]
        feature_df["Y_val"] = [passing_y_coord]
        feature_df["Normalized Data"] = [curr_data]
        feature_df["curr_SNR"] = [curr_SNR]
        feature_df["DP_estimate"] = [f_rec_DP]
        feature_df["LC_estimate"] = [f_rec_LC]
        feature_df["LR_estimate"] = [f_rec_LR]
        feature_df["GCV_estimate"] = [f_rec_GCV]
        # feature_df["OR_estimate"] = [f_rec_OR]
        feature_df["LS_estimate"] = [f_rec_LS]
        feature_df["MWF_DP"] = [MWF_DP]
        feature_df["MWF_LC"] = [MWF_LC]
        feature_df["MWF_LR"] = [MWF_LR]
        feature_df["MWF_GCV"] = [MWF_GCV]
        # feature_df["MWF_OR"] = [MWF_OR]
        feature_df["MWF_LS"] = [MWF_LS]
        feature_df["Lam_DP"] = [lambda_DP]
        feature_df["Lam_LC"] = [lambda_LC]
        feature_df["Lam_LR"] = [lambda_LR]
        feature_df["Lam_GCV"] = [lambda_GCV]
        #Flag_Val = 0 is good, 1 is bad value
        feature_df["Flagged"] = [Flag_Val]
        # feature_df["Lam_OR"] = [lambda_OR]
        print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
        return feature_df

def generate_brain_estimates(i_param_combo, seed= None):
    # print(f"Processing {i_param_combo}") 
    feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "DP_estimate", "LC_estimate", 
                                       "LR_estimate", "GCV_estimate", "LS_estimate", "MWF_DP", "MWF_LC", 
                                       "MWF_LR", "MWF_GCV", "MWF_LS", "Lam_DP", "Lam_LC", "Lam_GCV", "Lam_LR", "Flagged"])
    
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
        # print("x_coord", x_coord) 
        # print("y_coord", y_coord) 

    #eliminate voxels with pure noise
    if brain_data[x_coord, y_coord][0] < 50:
    # if brain_data[0] < 50:
        print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
        return feature_df
    # if SNR_map[x_coord, y_coord] == 0:
    #     print(f"SNR_map is 0 and pure noise for {x_coord} and {y_coord}")
    #     return feature_df
    else:
        passing_x_coord = x_coord
        passing_y_coord = y_coord
        curr_data = brain_data[passing_x_coord,passing_y_coord,:]
        curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        # curr_data = brain_data
        #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
        try:
            sol1 = nnls(A, curr_data, maxiter=1e6)[0]
        except:
            try:
                sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            except:
                # print("sol1 is ones")
                sol1 = np.ones(len(T2))
                # print("all methods fail")
                Flag_Val = 1

        # sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        # sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]

        X = sol1
        tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        # if np.all(sol1[:-1] == 0) or all_close_to_zero:
        #     print("factor is close to 0 or all_close_to_zero")
        #     # factor = 1e-6
        #     factor = sol1[-1] * dT
        # else:
        #     factor = np.sum(sol1) * dT
        #     pass
        factor = np.sum(sol1) * dT

        # print("sol1[0]",np.max(sol1))
        # factor2 = sol1[-1] * dT
        # print("factor", factor)
        # print("factor2", factor2)
        curr_data = curr_data/factor

        # if curr_data[0] < 0.5:
        #     print(f"normalized data  satisfies <0.5 requirement for {x_coord} and {y_coord}")
        #     return feature_df

        # factor = np.trapz(sol1,T2)
        # factor = simpson(y = sol1, x = T2)

        # factor = np.sum(sol1) * dT
        curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        #LS:
        try:
            f_rec_LS = nnls(A, curr_data, atol = 1e-3, maxiter=1e6)[0]
        except:
            try:
                f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            except:
                print("all methods fail")
                f_rec_LS = np.ones(len(T2))
                #try another
        # f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        
        X = f_rec_LS
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        # f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        if np.all(f_rec_LS[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LS is 0 or all_close_to_zero")
            MWF_LS = 0
            method = "LS"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_LS, curr_data, 0, curr_SNR, MWF_LS, filepath)
        else:
            f_rec_LS, MWF_LS = compute_MWF(f_rec_LS, T2, Myelin_idx)
            Flag_Val = 0
        # f_rec_LS, MWF_LS = compute_MWF(f_rec_LS, T2, Myelin_idx)
        # curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        #LC
        f_rec_LC, lambda_LC = Lcurve(curr_data, A, Lambda)
        X = f_rec_LC
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        # f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        if np.all(f_rec_LC[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LC is 0 or all_close_to_zero")
            MWF_LC = 0
            method = "LC"
            # curve_plot(method, x_coord, y_coord, f_rec_LC, curr_data, lambda_LC, curr_SNR, MWF_LC, filepath)
            Flag_Val = 1
            #Save the results for the pixel...; plot:SNR, decay, lambda..., f_rec..MWF.
        else:
            f_rec_LC, MWF_LC = compute_MWF(f_rec_LC, T2, Myelin_idx)
            Flag_Val = 0
        # f_rec_LC, MWF_LC = compute_MWF(f_rec_LC, T2, Myelin_idx)
        # print("MWF_LC", MWF_LC)
        #GCV
        f_rec_GCV, lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        f_rec_GCV = f_rec_GCV[:, 0]
        lambda_GCV = np.squeeze(lambda_GCV)

        X = f_rec_GCV
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(f_rec_GCV[:-1] == 0) or all_close_to_zero:
            # print("f_rec_GCV is 0 or all_close_to_zero")
            MWF_GCV = 0
            method = "GCV"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_GCV, curr_data, lambda_GCV, curr_SNR, MWF_GCV, filepath)
        else:
            f_rec_GCV, MWF_GCV = compute_MWF(f_rec_GCV, T2, Myelin_idx)
            Flag_Val = 0
        #DP
        # f_rec_DP, lambda_DP = discrep_L2(curr_data, A, curr_SNR, Lambda, noise = "brain")
        f_rec_DP, lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
        X = f_rec_DP
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(f_rec_DP[:-1] == 0) or all_close_to_zero:
            # print("f_rec_DP is 0 or all_close_to_zero")
            MWF_DP = 0
            method = "DP"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_DP, curr_data, lambda_DP, curr_SNR, MWF_DP, filepath)
        else:
            f_rec_DP, MWF_DP = compute_MWF(f_rec_DP, T2, Myelin_idx)
            Flag_Val = 0
        # print("MWF_DP", MWF_DP)

        #LR
        # LRIto_ini_lam = lambda_LC
        # f_rec_ini = f_rec_LC
        LRIto_ini_lam = lambda_GCV
        f_rec_ini = f_rec_GCV
        # print("curr_data", curr_data)
        f_rec_LR, lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter)
        X = f_rec_LR
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = all(abs(x) < tolerance for x in X)
        # X = np.array(X)
        all_close_to_zero = np.all(np.abs(X) < tolerance)

        if np.all(f_rec_LR[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LR is 0")
            MWF_LR = 0
            method = "LocReg"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_LR, curr_data, lambda_LR, curr_SNR, MWF_LR, filepath)
        else:
            f_rec_LR, MWF_LR = compute_MWF(f_rec_LR, T2, Myelin_idx)
            Flag_Val = 0

        gt = f_rec_GCV
    
        feature_df["X_val"] = [passing_x_coord]
        feature_df["Y_val"] = [passing_y_coord]
        feature_df["Normalized Data"] = [curr_data]
        feature_df["curr_SNR"] = [curr_SNR]
        feature_df["DP_estimate"] = [f_rec_DP]
        feature_df["LC_estimate"] = [f_rec_LC]
        feature_df["LR_estimate"] = [f_rec_LR]
        feature_df["GCV_estimate"] = [f_rec_GCV]
        # feature_df["OR_estimate"] = [f_rec_OR]
        feature_df["LS_estimate"] = [f_rec_LS]
        feature_df["MWF_DP"] = [MWF_DP]
        feature_df["MWF_LC"] = [MWF_LC]
        feature_df["MWF_LR"] = [MWF_LR]
        feature_df["MWF_GCV"] = [MWF_GCV]
        # feature_df["MWF_OR"] = [MWF_OR]
        feature_df["MWF_LS"] = [MWF_LS]
        feature_df["Lam_DP"] = [lambda_DP]
        feature_df["Lam_LC"] = [lambda_LC]
        feature_df["Lam_LR"] = [lambda_LR]
        feature_df["Lam_GCV"] = [lambda_GCV]
        #Flag_Val = 0 is good, 1 is bad value
        feature_df["Flagged"] = [Flag_Val]
        # feature_df["Lam_OR"] = [lambda_OR]
        print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
        return feature_df


def calculate_noise(signals):
    """
    Calculates the noise by computing the root sum of squared deviations
    between each signal and the mean signal, and randomly flipping signs.
    
    Args:
        signals (numpy.ndarray): Array of normalized signals.
    
    Returns:
        numpy.ndarray: The noise array with random signs.
    """
    # normalized_signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)
    # signals = normalized_signals
    # normalized_signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)
    mean_sig = np.mean(signals, axis=0)
    # squared_diff = (signals - mean_sig) ** 2
    # deviations = signals - mean_sig
    # sum_squared_diff = np.sum(squared_diff, axis=0)
    # noise = np.sqrt(sum_squared_diff)
    # random_signs = np.random.choice([-1, 1], size=noise.shape)
    # noise_stddev = np.std(deviations, axis=0)
    squared_diff = (signals - mean_sig) ** 2
    sum_squared_diff = np.sum(squared_diff, axis=0)
    noise_stddev = np.sqrt(sum_squared_diff)[0]
    mean_sig_val = mean_sig[0]
    return  mean_sig_val, noise_stddev 

def filter_and_compute_MWF(reconstr, tol = 1e-6):
    all_close_to_zero = np.all(np.abs(reconstr) < tol)
    if np.all(reconstr[:-1] == 0) or all_close_to_zero:
        noisy_MWF = 0
        noisy_f_rec = reconstr
        Flag_Val = 1
    else:
        noisy_f_rec, noisy_MWF = compute_MWF(reconstr, T2, Myelin_idx)
        Flag_Val = 0
    return noisy_f_rec, noisy_MWF, Flag_Val


def get_signals(coord_pairs, mask_array, unfiltered_arr, A, dT):
    """
    Extracts signals from the brain data at the specified coordinates,
    normalizes them using NNLS, and returns the signals list.
    
    Args:
        coord_pairs (list of tuple): List of coordinates to extract signals from.
        mask_array (numpy.ndarray): Mask array.
        unfiltered_arr (numpy.ndarray): Unfiltered brain data array.
        A (numpy.ndarray): The matrix A.
        dT (float): Time step.
    
    Returns:
        list: List of normalized signals.
    """
    signals = []
    SNRs = []
    for (x_coord, y_coord) in coord_pairs:
        mask_value = mask_array[x_coord, y_coord]
        signal = unfiltered_arr[x_coord, y_coord, :]
        SNR = SNR_map[x_coord,y_coord]
        # sol1 = nnls(A, signal)[0]
        # factor = np.sum(sol1) * dT
        # signal = signal / factor
        signals.append(signal)
        SNRs.append(SNR)
        print(f"Coordinate: ({x_coord}, {y_coord}), Mask value: {mask_value}")
    return np.array(signals), np.array(SNR)

# from Simulations.upencode import upen_param_setup, upen_setup
from Simulations.upenzama import UPEN_Zama
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\ZamaUPEN\1D_test', nargout=0)
unif_noise = False
preset_noise = False
# presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14Apr25\est_table_xcoordlen_313_ycoordlen_313_filtered_noise_addition_uniform_noise_UPEN_LR1D2D14Apr25 copy.pkl"

presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"
# presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14Apr25\est_table_xcoordlen_313_ycoordlen_313_filtered_noise_addition_uniform_noise_UPEN_LR1D2D14Apr25.pkl"
def generate_brain_estimates2(i_param_combo, seed= None):
    # print(f"Processing {i_param_combo}") 
    feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "curr_data", "noise", "curr_SNR", 
                                    "LS_estimate", "MWF_LS", "DP_estimate", "LC_estimate", "LR_estimate", "GCV_estimate", 
                                    "MWF_DP", "MWF_LC", "MWF_LR", "MWF_GCV", "Lam_DP", "Lam_LC", "Lam_LR", "Lam_GCV"])
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
    #eliminate voxels with pure noise
    if brain_data[x_coord, y_coord][0] < 50:
        print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
        return feature_df
    else:
        #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
        passing_x_coord = x_coord
        passing_y_coord = y_coord
        curr_data = brain_data[passing_x_coord,passing_y_coord,:]
        curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        # if unif_noise == True:
        #     unnormalized_data = curr_data
        #     if preset_noise == True:
        #         with open(presetfilepath, 'rb') as file:
        #             df = pickle.load(file)
        #         unif_noise_val = df["noise"][0]
        #         noise = unif_noise_val
        #         try:
        #             curr_data = df[(df["X_val"] == passing_x_coord) & (df["Y_val"] == passing_y_coord)]["curr_data"].tolist()[0]
        #         except:
        #             sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        #             X = sol1
        #             tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        #             all_close_to_zero = np.all(np.abs(X) < tolerance)
        #             factor = np.sum(sol1) * dT
        #             curr_data = curr_data/factor
        #             curr_data = curr_data + unif_noise_val
        #         try:
        #             noisy_f_rec_GCV = df[(df["X_val"] == passing_x_coord) & (df["Y_val"] == passing_y_coord)]["GCV_estimate"].tolist()[0]
        #         except:
        #             noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        #             noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        #             noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
        #             noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV, tol = 1e-6)
        #     else:
        #         noise = unif_noise_val
        #         curr_data = curr_data + unif_noise_val
            # try:
            #     sol1 = nnls(A, curr_data, maxiter=1e6)[0]
            # except:
            #     try:
            #         sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            #     except:
            #         print("need to skip, cannot find solution to LS solutions for normalizaiton")
            #         return feature_df
            # X = sol1
            # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
            # all_close_to_zero = np.all(np.abs(X) < tolerance)
            # factor = np.sum(sol1) * dT
            # curr_data = curr_data/factor

        try:
            sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        except:
                print("need to skip, cannot find solution to LS solutions for normalizaiton")
                return feature_df
        X = sol1
        tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        factor = np.sum(sol1) * dT
        curr_data = curr_data/factor

        # with open(presetfilepath, 'rb') as file:
        #     df = pickle.load(file)
        # unif_noise_val = df["noise"][0]
        # noise = unif_noise_val
        #After normalizaing, do regularization techniques
        #LS reconstruction and MWF calculation

        # try:
        #     noisy_f_rec_LS = nnls(A, curr_data, maxiter=1e6)[0]
        # except:
        #     try:
        #         noisy_f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        #     except:
        #         noisy_f_rec_LS = np.zeros(len(T2))
        #         Flag_Val = 2
        # noisy_f_rec_LS, noisy_MWF_LS, LS_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LS, tol = 1e-6)
        
        # #LCurve reconstruction and MWF calculation
        # noisy_f_rec_LC, noisy_lambda_LC = Lcurve(curr_data, A, Lambda)
        # noisy_f_rec_LC, noisy_MWF_LC, LC_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LC, tol = 1e-6)

        # #GCV reconstruction and MWF calculation

        # #DP reconstruction and MWF calculation
        # noisy_f_rec_DP, noisy_lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
        # noisy_f_rec_DP, noisy_MWF_DP, DP_Flag_Val = filter_and_compute_MWF(noisy_f_rec_DP, tol = 1e-6)
        noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
        noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV, tol = 1e-6)

        #LocReg reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        noisy_f_rec_LR, noisy_lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        noisy_f_rec_LR, noisy_MWF_LR, LR_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LR, tol = 1e-6)
        gt = noisy_f_rec_GCV

        # #LocReg1stDeriv reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        f_rec_LR1D, noisy_lambda_LR1D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        noisy_f_rec_LR1D, noisy_MWF_LR1D, LR_Flag_Val1D = filter_and_compute_MWF(f_rec_LR1D, tol = 1e-6)

        #LocReg2ndDeriv reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        f_rec_LR2D, noisy_lambda_LR2D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv2(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        noisy_f_rec_LR2D, noisy_MWF_LR2D, LR_Flag_Val2D = filter_and_compute_MWF(f_rec_LR2D, tol = 1e-6)

        #UPEN reconstruction and MWF calculation
        # result = upen_param_setup(TE, T2, A, curr_data)
        # noisy_f_rec_UPEN, _ ,_ , noisy_lambda_UPEN= upen_setup(result, curr_data, LRIto_ini_lam, True)
        # noise = unif_noise_val
        # noise_norm = np.linalg.norm(unif_noise_val)
        threshold = 1.1 * np.sqrt(A.shape[0]) * np.max(curr_data) / SNR
        # noise_norm = np.linalg.norm(noise)
        noise_norm = threshold
        xex = noisy_f_rec_GCV
        Kmax = 50
        beta_0 = 1e-7
        tol_lam = 1e-5
        noisy_f_rec_UPEN, noisy_lambda_UPEN = UPEN_Zama(A, curr_data, xex, noise_norm, beta_0, Kmax, tol_lam)
        noisy_f_rec_UPEN, noisy_MWF_UPEN, UPEN_Flag_Val = filter_and_compute_MWF(noisy_f_rec_UPEN, tol = 1e-6)

        feature_df["X_val"] = [passing_x_coord]
        feature_df["Y_val"] = [passing_y_coord]
        feature_df["curr_data"] = [curr_data]
        # feature_df["noise"] = [noise]
        feature_df["curr_SNR"] = [curr_SNR]
        # feature_df["LS_estimate"] = [noisy_f_rec_LS]
        # feature_df["MWF_LS"] = [noisy_MWF_LS]
        # feature_df["DP_estimate"] = [noisy_f_rec_DP]
        # feature_df["LC_estimate"] = [noisy_f_rec_LC]
        feature_df["LR_estimate"] = [noisy_f_rec_LR]
        feature_df["LR1D_estimate"] = [noisy_f_rec_LR1D]
        feature_df["LR2D_estimate"] = [noisy_f_rec_LR2D]
        feature_df["GCV_estimate"] = [noisy_f_rec_GCV]
        feature_df["UPEN_estimate"] = [noisy_f_rec_UPEN]
        # feature_df["MWF_DP"] = [noisy_MWF_DP]
        # feature_df["MWF_LC"] = [noisy_MWF_LC]
        feature_df["MWF_LR"] = [noisy_MWF_LR]
        feature_df["MWF_LR1D"] = [noisy_MWF_LR1D]
        feature_df["MWF_LR2D"] = [noisy_MWF_LR2D]
        feature_df["MWF_GCV"] = [noisy_MWF_GCV]
        feature_df["MWF_UPEN"] = [noisy_MWF_UPEN]
        # feature_df["Lam_DP"] = [noisy_lambda_DP]
        # feature_df["Lam_LC"] = [noisy_lambda_LC]
        feature_df["Lam_LR"] = [noisy_lambda_LR]
        feature_df["Lam_LR1D"] = [noisy_lambda_LR1D]
        feature_df["Lam_LR2D"] = [noisy_lambda_LR2D]
        feature_df["Lam_GCV"] = [noisy_lambda_GCV]
        feature_df["Lam_UPEN"] = [noisy_lambda_UPEN]
        # feature_df["LS_Flag_Val"] = [LS_Flag_Val]
        # feature_df["LC_Flag_Val"] = [LC_Flag_Val]
        feature_df["GCV_Flag_Val"] = [GCV_Flag_Val]
        # feature_df["DP_Flag_Val"] = [DP_Flag_Val]
        feature_df["LR_Flag_Val"] = [LR_Flag_Val]
        feature_df["LR1D_Flag_Val"] = [LR_Flag_Val1D]
        feature_df["LR2D_Flag_Val"] = [LR_Flag_Val2D]
        feature_df["UPEN_Flag_Val"] = [UPEN_Flag_Val]
        print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
        return feature_df

unif_noise = True
preset_noise = True
presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"
def generate_brain_estimatesNA(i_param_combo, seed= None):
    # print(f"Processing {i_param_combo}") 
    feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "curr_data", "noise", "curr_SNR", 
                                    "LS_estimate", "MWF_LS", "DP_estimate", "LC_estimate", "LR_estimate", "GCV_estimate", 
                                    "MWF_DP", "MWF_LC", "MWF_LR", "MWF_GCV", "Lam_DP", "Lam_LC", "Lam_LR", "Lam_GCV"])
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
    #eliminate voxels with pure noise
    if brain_data[x_coord, y_coord][0] < 50:
        print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
        return feature_df
    else:
        #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
        passing_x_coord = x_coord
        passing_y_coord = y_coord
        curr_data = brain_data[passing_x_coord,passing_y_coord,:]
        curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        if unif_noise == True:
            unnormalized_data = curr_data
            if preset_noise == True:
                with open(presetfilepath, 'rb') as file:
                    df = pickle.load(file)
                unif_noise_val = df["noise"][0]
                noise = unif_noise_val
                curr_data = curr_data + unif_noise_val
            # try:
            #     sol1 = nnls(A, curr_data, maxiter=1e6)[0]
            # except:
            #     try:
            #         sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            #     except:
            #         print("need to skip, cannot find solution to LS solutions for normalizaiton")
            #         return feature_df
            # X = sol1
            # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
            # all_close_to_zero = np.all(np.abs(X) < tolerance)
            # factor = np.sum(sol1) * dT
            # curr_data = curr_data/factor

        try:
            sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        except:
                print("need to skip, cannot find solution to LS solutions for normalizaiton")
                return feature_df
        X = sol1
        tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        factor = np.sum(sol1) * dT
        curr_data = curr_data/factor

        # with open(presetfilepath, 'rb') as file:
        #     df = pickle.load(file)
        # unif_noise_val = df["noise"][0]
        # noise = unif_noise_val
        #After normalizaing, do regularization techniques
        #LS reconstruction and MWF calculation

        # try:
        #     noisy_f_rec_LS = nnls(A, curr_data, maxiter=1e6)[0]
        # except:
        #     try:
        #         noisy_f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        #     except:
        #         noisy_f_rec_LS = np.zeros(len(T2))
        #         Flag_Val = 2
        # noisy_f_rec_LS, noisy_MWF_LS, LS_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LS, tol = 1e-6)
        
        # #LCurve reconstruction and MWF calculation
        # noisy_f_rec_LC, noisy_lambda_LC = Lcurve(curr_data, A, Lambda)
        # noisy_f_rec_LC, noisy_MWF_LC, LC_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LC, tol = 1e-6)

        # #GCV reconstruction and MWF calculation

        # #DP reconstruction and MWF calculation
        # noisy_f_rec_DP, noisy_lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
        # noisy_f_rec_DP, noisy_MWF_DP, DP_Flag_Val = filter_and_compute_MWF(noisy_f_rec_DP, tol = 1e-6)
        noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
        noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV, tol = 1e-6)

        #LocReg reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        noisy_f_rec_LR, noisy_lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        noisy_f_rec_LR, noisy_MWF_LR, LR_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LR, tol = 1e-6)
        gt = noisy_f_rec_GCV

        # #LocReg1stDeriv reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        f_rec_LR1D, noisy_lambda_LR1D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        noisy_f_rec_LR1D, noisy_MWF_LR1D, LR_Flag_Val1D = filter_and_compute_MWF(f_rec_LR1D, tol = 1e-6)

        #LocReg2ndDeriv reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        f_rec_LR2D, noisy_lambda_LR2D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv2(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        noisy_f_rec_LR2D, noisy_MWF_LR2D, LR_Flag_Val2D = filter_and_compute_MWF(f_rec_LR2D, tol = 1e-6)

        #UPEN reconstruction and MWF calculation
        # result = upen_param_setup(TE, T2, A, curr_data)
        # noisy_f_rec_UPEN, _ ,_ , noisy_lambda_UPEN= upen_setup(result, curr_data, LRIto_ini_lam, True)
        # noise = unif_noise_val
        # noise_norm = np.linalg.norm(unif_noise_val)
        threshold = 1.1 * np.sqrt(A.shape[0]) * np.max(curr_data) / SNR
        # noise_norm = np.linalg.norm(noise)
        noise_norm = threshold
        xex = noisy_f_rec_GCV
        Kmax = 50
        beta_0 = 1e-7
        tol_lam = 1e-5
        noisy_f_rec_UPEN, noisy_lambda_UPEN = UPEN_Zama(A, curr_data, xex, noise_norm, beta_0, Kmax, tol_lam)
        noisy_f_rec_UPEN, noisy_MWF_UPEN, UPEN_Flag_Val = filter_and_compute_MWF(noisy_f_rec_UPEN, tol = 1e-6)

        feature_df["X_val"] = [passing_x_coord]
        feature_df["Y_val"] = [passing_y_coord]
        feature_df["curr_data"] = [curr_data]
        feature_df["noise"] = [noise]
        feature_df["curr_SNR"] = [curr_SNR]
        # feature_df["LS_estimate"] = [noisy_f_rec_LS]
        # feature_df["MWF_LS"] = [noisy_MWF_LS]
        # feature_df["DP_estimate"] = [noisy_f_rec_DP]
        # feature_df["LC_estimate"] = [noisy_f_rec_LC]
        feature_df["LR_estimate"] = [noisy_f_rec_LR]
        feature_df["LR1D_estimate"] = [noisy_f_rec_LR1D]
        feature_df["LR2D_estimate"] = [noisy_f_rec_LR2D]
        feature_df["GCV_estimate"] = [noisy_f_rec_GCV]
        feature_df["UPEN_estimate"] = [noisy_f_rec_UPEN]
        # feature_df["MWF_DP"] = [noisy_MWF_DP]
        # feature_df["MWF_LC"] = [noisy_MWF_LC]
        feature_df["MWF_LR"] = [noisy_MWF_LR]
        feature_df["MWF_LR1D"] = [noisy_MWF_LR1D]
        feature_df["MWF_LR2D"] = [noisy_MWF_LR2D]
        feature_df["MWF_GCV"] = [noisy_MWF_GCV]
        feature_df["MWF_UPEN"] = [noisy_MWF_UPEN]
        # feature_df["Lam_DP"] = [noisy_lambda_DP]
        # feature_df["Lam_LC"] = [noisy_lambda_LC]
        feature_df["Lam_LR"] = [noisy_lambda_LR]
        feature_df["Lam_LR1D"] = [noisy_lambda_LR1D]
        feature_df["Lam_LR2D"] = [noisy_lambda_LR2D]
        feature_df["Lam_GCV"] = [noisy_lambda_GCV]
        feature_df["Lam_UPEN"] = [noisy_lambda_UPEN]
        # feature_df["LS_Flag_Val"] = [LS_Flag_Val]
        # feature_df["LC_Flag_Val"] = [LC_Flag_Val]
        feature_df["GCV_Flag_Val"] = [GCV_Flag_Val]
        # feature_df["DP_Flag_Val"] = [DP_Flag_Val]
        feature_df["LR_Flag_Val"] = [LR_Flag_Val]
        feature_df["LR1D_Flag_Val"] = [LR_Flag_Val1D]
        feature_df["LR2D_Flag_Val"] = [LR_Flag_Val2D]
        feature_df["UPEN_Flag_Val"] = [UPEN_Flag_Val]
        print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
        return feature_df
    
def worker_init():
    # Use current_process()._identity to get a unique worker ID for each worker
    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    np.random.seed(worker_id)  # Set a random seed for each worker

# def parallel_processed(func, shift = True):
#     with mp.Pool(processes = num_cpus_avail, initializer=worker_init) as pool:
#         with tqdm(total = len(target_iterator)) as pbar:
#             for estimates_dataframe in pool.imap_unordered(func, range(len(target_iterator))):
#                 lis.append(estimates_dataframe)
#                 pbar.update()
#         pool.close()
#         pool.join()
#     return estimates_dataframe

def parallel_processed(func, shift=True):
    with mp.Pool(processes=num_cpus_avail, initializer=worker_init) as pool:
        with tqdm(total=len(target_iterator)) as pbar:
            for estimates_dataframe in pool.imap_unordered(func, range(len(target_iterator))):  # Pass target_iterator directly
                lis.append(estimates_dataframe)
                pbar.update()
        pool.close()
        pool.join()
    return estimates_dataframe
#Unit Tests

if __name__ == "__main__":
    logging.info("Script started.")
    freeze_support()
    dTE = 11.3
    n = 32
    TE = dTE * np.linspace(1,n,n)
    m = 150
    T2 = np.linspace(10,200,m)
    A= np.zeros((n,m))
    dT = T2[1] - T2[0]
    logging.info(f"T2 range is from 10ms to 200ms with {m} discretizations")
    logging.info(f"dT is {dT}")
    logging.info(f"TE range is {TE}")
    for i in range(n):
        for j in range(m):
            A[i,j] = np.exp(-TE[i]/T2[j]) * dT

    if unif_noise == True:
        num_signals = 1000
        coord_pairs = set()
        
        # Generate random coordinates
        # for i in range(num_signals):
        #     x = random.randint(155, 160)
        #     y = random.randint(150, 160)
        #     mask_value = mask[x, y]
        #     coord_pairs.add((x, y))
        mask_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\masks\new_mask.mat" 
        mask = scipy.io.loadmat(mask_path)["new_BW"]
        # unfilt_brain_data_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\braindata\mew_cleaned_brain_data_unfiltered.mat"
        brain_data = scipy.io.loadmat(r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\braindata\cleaned_brain_data (1).mat")["final_data_2"]

        # unfilt_brain_data = scipy.io.loadmat(unfilt_brain_data_path)["brain_data"]
        while len(coord_pairs) < num_signals:
            # Generate random coordinates
            x = random.randint(130, 200)
            y = random.randint(100, 200)
            # Get the mask_value at the coordinate
            mask_value = mask[x, y]
            
            # Check if the mask_value is 0 and add the coordinate to the set
            if mask_value == 0:
                coord_pairs.add((x, y))
        coord_pairs = list(coord_pairs)
        print("Length", len(coord_pairs))
        
        signals, SNRs = get_signals(coord_pairs, mask , brain_data, A, dT)
        # mean_sig, unif_noise_val = calculate_noise(signals)
        # SNR_mean = np.mean(SNRs)
        # normalized_signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)
        # signals = normalized_signals
        mean_sig = np.mean(signals, axis=0)
        tail_length = 2  # Example length, adjust as needed
        # # Get the tail end of the signals
        tail = mean_sig[-tail_length:]
        # # Calculate the standard deviation of the tail ends
        tail_std = np.std(tail)
        # _, unif_noise_val = add_noise(mean_sig, SNR = SNR_mean)
        print("tail_std", tail_std)
        # _, unif_noise_val = add_noise(mean_sig, SNR = 1)
        unif_noise_val = np.random.normal(0, tail_std, size=32)  # Add noise
        plt.figure()
        print("unif_noise_val", unif_noise_val)
        plt.plot(unif_noise_val)
        plt.xlabel('Time/Index')
        plt.ylabel('Noise Standard Deviation')
        plt.title('Noise Standard Deviation Across Signals')
        plt.grid(True)
        plt.savefig("testfignoise.png")
        plt.close()

        plt.figure()
        plt.plot(mean_sig)
        plt.xlabel('TE')
        plt.ylabel('Amplitude')
        plt.title('Mean Signal')
        plt.grid(True)
        plt.savefig("testfigmeansig.png")
        plt.close()
    else:
        pass

    logging.info(f"Kernel matrix is size {A.shape} and is form np.exp(-TE[i]/T2[j]) * dT")
    LS_estimates = np.zeros((p,q,m))
    MWF_LS = np.zeros((p,q))
    LR_estimates = np.zeros((p,q,m))
    MWF_LR = np.zeros((p,q))
    LC_estimates = np.zeros((p,q,m))
    MWF_LC = np.zeros((p,q))    
    GCV_estimates = np.zeros((p,q,m))
    MWF_GCV = np.zeros((p,q))
    DP_estimates = np.zeros((p,q,m))
    MWF_DP = np.zeros((p,q))
    OR_estimates = np.zeros((p,q,m))
    MWF_OR = np.zeros((p,q))
    Myelin_idx = np.where(T2<=40)
    logging.info("We define myelin index to be less than 40 ms.")
    logging.info("Since this is experimental patient brain data, we do not have the ground truth. Set the ground truth to be GCV.")
    lis = []
    # prevsol = np.ones(len(T2)) 
    # prevsol2 = np.ones(len(T2)) 
    if parallel == True:
        logging.info("Generating Brain Estimates Using Parallel Processing.")
        estimates_dataframe = parallel_processed(generate_brain_estimates, shift = True)
        # estimates_dataframe = parallel_processed(profile_generate_brain_estimates, shift = True)
    else:
        logging.info("Generating Brain Estimates Without Parallel Processing.")
        # for j in range(p):
        #     for k in range(q):
        #         iteration = (j,k)
        #         # print(f"Processing no parallel: {iteration}")  # Debugging line
        #         estimates_dataframe = generate_brain_estimates(iteration) 
        #         lis.append(estimates_dataframe)
        for j in tqdm(range(p), desc="Processing rows", unit="row"):
            for k in tqdm(range(q), desc="Processing columns", unit="col", leave=False):  # leave=False to not overwrite the row progress
                iteration = (j, k)
                # print(f"Processing no parallel: {iteration}")  # Debugging line
                # profile_generate_brain_estimates()
                # estimates_dataframe = generate_brain_estimates(iteration) 
                estimates_dataframe = generate_brain_estimates2(iteration) 
                # estimates_dataframe = generate_brain_estimatesNA(iteration) 

                # profile_generate_brain_estimates(iteration)
                # estimates_dataframe = generate_noisy_brain_estimates(iteration)
                # estimates_dataframe = generate_noisy_brain_estimates2(iteration)
                # estimates_dataframe = spanreg_unif_SNR(iteration)
                # print(f"saved estimates dataframe iteration{j}_{k}")
                # estimates_dataframe.to_pickle(data_folder + f'/' + data_tag + f"iteration{j}_{k}"+'.pkl')
                lis.append(estimates_dataframe)
    #Save file
    print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
    df = pd.concat(lis, ignore_index= True)
    df.to_pickle(data_folder + f'/' + data_tag +'.pkl')