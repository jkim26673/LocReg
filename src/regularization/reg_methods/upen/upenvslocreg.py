# import sys
# import os
# from datetime import datetime
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
# import matplotlib.ticker as ticker
# import scipy
# import timeit
# import unittest
# from scipy.integrate import simpson
# from Simulations.upenzama import UPEN_Zama
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
import matlab.engine

# Start MATLAB engine
try:
    eng = matlab.engine.start_matlab()
    # Add paths to MATLAB functions
    eng.addpath(r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\ZamaUPEN\1D_test', nargout=0)
    eng.addpath('.', nargout=0)  # Add current directory for T1.m
    print("MATLAB engine started successfully")
except Exception as e:
    print(f"Error starting MATLAB engine: {e}")
    raise

def create_result_folder(string, SNR):
    """Create a folder based on the current date and time"""
    date = datetime.now().strftime("%Y%m%d")
    folder_name = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\classical_prob\upen"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def load_T1_problem(N, noise_level, eng):
    """Load T1 problem using MATLAB engine"""
    try:
        # Call T1.m function through MATLAB engine
        A_mat, b_mat, xex_mat, noise_norm_mat, b_exact_mat = eng.T1(N, noise_level, nargout=5)
        
        # Convert MATLAB arrays to numpy arrays
        A = np.array(A_mat)
        b = np.array(b_mat).flatten()
        xex = np.array(xex_mat).flatten()
        noise_norm = float(noise_norm_mat)
        b_exact = np.array(b_exact_mat).flatten()
        
        print(f"T1 problem loaded: A shape = {A.shape}, b shape = {b.shape}, xex shape = {xex.shape}")
        return A, b, xex, noise_norm, b_exact
        
    except Exception as e:
        print(f"Error loading T1 problem: {e}")
        raise

def run_single_noise_realization(noise_idx, A, b_exact, xex, T2, TE, SNR, gamma_init):
    """Run simulation for a single noise realization"""
    print(f"Running noise realization {noise_idx + 1}/10")
    
    # Generate noise for this realization
    SD_noise = 1/SNR * max(abs(b_exact))
    noise = np.random.normal(0, SD_noise, b_exact.shape)

    noise_norm = np.linalg.norm(noise)
    curr_data = b_exact + noise
    
    # SVD decomposition for regularization
    U, s, V = csvd(A, tst=None, nargin=1, nargout=3)


    Lambda_vec=np.logspace(-3,0,200)
    
    # GCV reconstruction
    noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda_vec)
    noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
    noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
    
    # L-curve reconstruction
    # f_rec_LC, lambda_LC = Lcurve(A, curr_data, U, s, V)
    
    # LocReg reconstruction (original)
    eps1 = 1e-2
    ep_min = 1e-2
    eps_cut = 1.2
    eps_floor = 1e-4
    exp = 0.5
    feedback = True
    gamma_init = 0.5
    # maxiter = 500
    LRIto_ini_lam = noisy_lambda_GCV
    f_rec_ini = noisy_f_rec_GCV
    noisy_f_rec_LR, noisy_lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(
        curr_data, A, LRIto_ini_lam, gamma_init, maxiter=50
    )
    f_rec_ini = noisy_f_rec_GCV
    # LocReg 1st Derivative reconstruction
    f_rec_LR1D, noisy_lambda_LR1D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv(
        curr_data, A, LRIto_ini_lam, gamma_init, maxiter=50
    )
    f_rec_ini = noisy_f_rec_GCV
    # LocReg 2nd Derivative reconstruction
    f_rec_LR2D, noisy_lambda_LR2D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv2(
        curr_data, A, LRIto_ini_lam, gamma_init, maxiter=50
    )
    
    # UPEN reconstruction
    # std_dev = np.std(curr_data[len(curr_data)-5:])
    # SNR_est = np.max(np.abs(curr_data))/std_dev
    # threshold = 1.05 * np.sqrt(A.shape[0]) * np.max(curr_data) / SNR_est
    # noise_norm = threshold
    xex_init = xex
    Kmax = 500
    beta_0 = 1e-3
    tol_lam=1.e-5
    noisy_f_rec_UPEN, noisy_lambda_UPEN = UPEN_Zama(A, curr_data, xex_init, noise_norm, beta_0, Kmax, tol_lam)
    
    # Calculate errors
    err_GCV = np.linalg.norm(xex - noisy_f_rec_GCV)
    # err_LC = np.linalg.norm(xex - f_rec_LC)
    err_LR = np.linalg.norm(xex - noisy_f_rec_LR)
    err_LR1D = np.linalg.norm(xex - f_rec_LR1D)
    err_LR2D = np.linalg.norm(xex - f_rec_LR2D)
    err_UPEN = np.linalg.norm(xex - noisy_f_rec_UPEN)
    
    results = {
        'data_noisy': curr_data,
        'f_rec_GCV': noisy_f_rec_GCV,
        # 'f_rec_LC': f_rec_LC,
        'f_rec_LR': noisy_f_rec_LR,
        'f_rec_LR1D': f_rec_LR1D,
        'f_rec_LR2D': f_rec_LR2D,
        'f_rec_UPEN': noisy_f_rec_UPEN,
        'lambda_GCV': noisy_lambda_GCV,
        # 'lambda_LC': lambda_LC,
        'lambda_LR': noisy_lambda_LR,
        'lambda_LR1D': noisy_lambda_LR1D,
        'lambda_LR2D': noisy_lambda_LR2D,
        'lambda_UPEN': noisy_lambda_UPEN,
        'err_GCV': err_GCV,
        'err_LR': err_LR,
        'err_LR1D': err_LR1D,
        'err_LR2D': err_LR2D,
        'err_UPEN': err_UPEN
    }
    
    return results

def plot_all_realizations(all_results, xex, A, T2, TE, SNR, file_path):
    """Plot results for all noise realizations"""
    n_realizations = len(all_results)
    
    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot 1: Reconstructed signals
    ymax = np.max(xex) * 1.15
    axs[0, 0].plot(T2, xex, color="black", linewidth=3, label="Ground Truth")
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    method_names = ['GCV', 'LocReg', 'LocReg-1D', 'LocReg-2D', 'UPEN']
    
    for i, (method, color) in enumerate(zip(['f_rec_GCV', 'f_rec_LR', 'f_rec_LR1D', 'f_rec_LR2D', 'f_rec_UPEN'], colors)):
        for j, result in enumerate(all_results):
            alpha = 0.3 if j > 0 else 0.8
            label = method_names[i] if j == 0 else None
            axs[0, 0].plot(T2, result[method], color=color, alpha=alpha, label=label)
    
    axs[0, 0].set_xlabel('T1 Time', fontsize=14, fontweight='bold')
    axs[0, 0].set_ylabel('Amplitude', fontsize=14, fontweight='bold')
    axs[0, 0].legend(fontsize=10, loc='best')
    axs[0, 0].set_ylim(0, ymax)
    axs[0, 0].set_title('Reconstructed Signals (All Realizations)', fontsize=14, fontweight='bold')
    
    # Plot 2: Data fitting
    axs[0, 1].plot(TE, A @ xex, linewidth=3, color='black', label='Ground Truth')
    
    for i, (method, color) in enumerate(zip(['f_rec_GCV', 'f_rec_LR', 'f_rec_LR1D', 'f_rec_LR2D', 'f_rec_UPEN'], colors)):
        for j, result in enumerate(all_results):
            alpha = 0.3 if j > 0 else 0.8
            label = method_names[i] if j == 0 else None
            axs[0, 1].plot(TE, A @ result[method], color=color, alpha=alpha, label=label)
    
    axs[0, 1].legend(fontsize=10, loc='best')
    axs[0, 1].set_xlabel('Measurement Index', fontsize=14, fontweight='bold')
    axs[0, 1].set_ylabel('Intensity', fontsize=14, fontweight='bold')
    axs[0, 1].set_title('Data Fitting (All Realizations)', fontsize=14, fontweight='bold')
    
    # Plot 3: Regularization parameters
    for i, (method, color) in enumerate(zip(['lambda_GCV', 'lambda_LR', 'lambda_LR1D', 'lambda_LR2D', 'lambda_UPEN'], colors)):
        for j, result in enumerate(all_results):
            alpha = 0.3 if j > 0 else 0.8
            label = method_names[i] if j == 0 else None
            
            if method in ['lambda_GCV']:
                # Scalar lambdas
                axs[1, 0].semilogy(T2, result[method] * np.ones(len(T2)), 
                                  color=color, alpha=alpha, label=label)
            else:
                # Vector lambdas
                if hasattr(result[method], '__len__') and len(result[method]) == len(T2):
                    axs[1, 0].semilogy(T2, result[method], color=color, alpha=alpha, label=label)
                else:
                    # Handle scalar case
                    axs[1, 0].semilogy(T2, result[method] * np.ones(len(T2)), 
                                      color=color, alpha=alpha, label=label)
    
    axs[1, 0].legend(fontsize=10, loc='best')
    axs[1, 0].set_xlabel('T1 Index', fontsize=14, fontweight='bold')
    axs[1, 0].set_ylabel('Lambda', fontsize=14, fontweight='bold')
    axs[1, 0].set_title('Regularization Parameters', fontsize=14, fontweight='bold')
    
    # Plot 4: Error comparison
    methods = ['GCV', 'LocReg', 'LocReg-1D', 'LocReg-2D', 'UPEN']
    error_keys = ['err_GCV', 'err_LR', 'err_LR1D', 'err_LR2D', 'err_UPEN']
    
    errors_by_method = {method: [] for method in methods}
    
    for result in all_results:
        for method, error_key in zip(methods, error_keys):
            errors_by_method[method].append(result[error_key])
    
    # Box plot
    box_data = [errors_by_method[method] for method in methods]
    bp = axs[1, 1].boxplot(box_data, labels=methods, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axs[1, 1].set_ylabel('Reconstruction Error', fontsize=14, fontweight='bold')
    axs[1, 1].set_title('Error Distribution Across Realizations', fontsize=14, fontweight='bold')
    axs[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 5: Error statistics table
    axs[2, 0].axis('off')
    
    # Calculate statistics
    stats_data = []
    for method, error_key in zip(methods, error_keys):
        errors = [result[error_key] for result in all_results]
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        min_err = np.min(errors)
        max_err = np.max(errors)
        stats_data.append([method, f'{mean_err:.4f}', f'{std_err:.4f}', f'{min_err:.4f}', f'{max_err:.4f}'])
    
    table = axs[2, 0].table(cellText=stats_data, 
                           colLabels=['Method', 'Mean Error', 'Std Error', 'Min Error', 'Max Error'],
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axs[2, 0].set_title('Error Statistics', fontsize=14, fontweight='bold')
    
    # Plot 6: Mean reconstruction
    axs[2, 1].plot(T2, xex, color="black", linewidth=3, label="Ground Truth")
    
    # Calculate mean reconstructions
    for i, (method, color) in enumerate(zip(['f_rec_GCV', 'f_rec_LR', 'f_rec_LR1D', 'f_rec_LR2D', 'f_rec_UPEN'], colors)):
        mean_reconstruction = np.mean([result[method] for result in all_results], axis=0)
        axs[2, 1].plot(T2, mean_reconstruction, color=color, linewidth=2, label=method_names[i])
    
    axs[2, 1].set_xlabel('T1 Time', fontsize=14, fontweight='bold')
    axs[2, 1].set_ylabel('Amplitude', fontsize=14, fontweight='bold')
    axs[2, 1].legend(fontsize=10, loc='best')
    axs[2, 1].set_ylim(0, ymax)
    axs[2, 1].set_title('Mean Reconstructions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, f"T1_LocReg_UPEN_comparison_10_realizations.png"), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(file_path, f"T1_LocReg_UPEN_comparison_10_realizations.pdf"), 
                bbox_inches='tight')
    print(f"Plots saved to {file_path}")

def main():
    """Main simulation function"""
    # Parameters

    N = 100  # Problem size
    noise_level = 0.01  # Noise level for T1 problem (1% noise)
    SNR = 1/noise_level  # Convert to SNR
    gamma_init = 0.5  # Initialize gamma parameter for LocReg
    print(f"Loading T1 problem with N={N}, noise_level={noise_level}")
    
    # Load T1 problem using MATLAB engine
    A, b_noisy, xex, noise_norm_matlab, b_exact = load_T1_problem(N, noise_level, eng)
    
    # Create time/index vectors for plotting
    T2 = np.linspace(0, 1, len(xex))  # T1 time vector
    TE = np.arange(len(b_exact))      # Measurement indices
    
    # Create result folder
    string = "T1"
    # file_path = create_result_folder(string, int(SNR))
    file_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\classical_prob\upen"
    
    # Run simulations for 10 noise realizations
    print("Running simulations for 10 noise realizations...")
    all_results = []
    
    for i in range(10):
        try:
            result = run_single_noise_realization(i, A, b_exact, xex, T2, TE, SNR, gamma_init)
            all_results.append(result)
        except Exception as e:
            print(f"Error in realization {i+1}: {str(e)}")
            continue
    
    if not all_results:
        print("No successful realizations completed!")
        return
    
    print(f"Completed {len(all_results)} realizations successfully")
    
    # Plot all results
    plot_all_realizations(all_results, xex, A, T2, TE, SNR, file_path)
    
    # Save results to pickle file
    results_dict = {
        'all_results': all_results,
        'ground_truth': xex,
        'T2': T2,
        'TE': TE,
        'A': A,
        'SNR': SNR,
        'b_exact': b_exact,
        'noise_norm_matlab': noise_norm_matlab,
        'N': N,
        'noise_level': noise_level
    }
    
    with open(os.path.join(file_path, 'T1_simulation_results.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"All results saved to {file_path}")
    print("T1 simulation completed successfully!")
    
    # Print summary statistics
    print("\n=== SIMULATION SUMMARY ===")
    methods = ['GCV',  'LocReg', 'LocReg-1D', 'LocReg-2D', 'UPEN']
    error_keys = ['err_GCV',  'err_LR', 'err_LR1D', 'err_LR2D', 'err_UPEN']
    
    for method, error_key in zip(methods, error_keys):
        errors = [result[error_key] for result in all_results]
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        print(f"{method}: Mean Error = {mean_err:.4f} ± {std_err:.4f}")

if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    main()
    
    # Close MATLAB engine
    try:
        eng.quit()
        print("MATLAB engine closed successfully")
    except:
        pass