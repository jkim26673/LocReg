# import numpy as np
# from scipy.stats import norm as normsci
# from scipy.linalg import norm as linalg_norm
# from scipy.optimize import nnls
# import matplotlib.pyplot as plt
# from scipy.stats import wasserstein_distance, entropy
# import pickle
# from tqdm import tqdm
# from regularization.reg_methods.locreg.LocReg import LocReg
# from regularization.reg_methods.locreg.LocReg_unconstrainedB import LocReg_unconstrainedB
# from regularization.reg_methods.locreg.LocReg_NEW_NNLS import LocReg_NEW_NNLS
# from regularization.reg_methods.dp.discrep_L2 import discrep_L2
# from regularization.reg_methods.gcv.GCV_NNLS import GCV_NNLS
# from regularization.reg_methods.lcurve.Lcurve import Lcurve
# from regularization.reg_methods.spanreg.Multi_Reg_Gaussian_Sum1 import Multi_Reg_Gaussian_Sum1

# import os
# from datetime import datetime
# import pandas as pd
# import sys
# # sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
# import cvxpy as cp
# import os
# import scipy
# from scipy.linalg import svd
# # from lsqnonneg import lsqnonneg
# from regularization.subfunc.lcurve_functions import l_cuve,csvd,l_corner
# from regularization.subfunc.l_curve_corner import l_curve_corner
# from regularization.subfunc.csvd import csvd
# from regularization.reg_methods.dp.discrep import discrep
# # from Simulations.Ito_LocReg import blur_ito, grav_ito
# from regularization.reg_methods.locreg.Ito_LocReg import *
# from regularization.reg_methods.locreg.LocReg import LocReg as Chuan_LR
# # from regu.ito_blur import blur
# # from regu.ito_gravity import gravity
# from tools.trips_py.pasha_gcv import Tikhonov
# # from regu.tikhonov import tikhonov
# from regularization.reg_methods.lcurve import l_curve
# from datetime import datetime
# from tqdm import tqdm
# from regularization.reg_methods.nnls.tikhonov_vec import tikhonov_vec
# import sys
# import os
# import mosek
# # from ItoLocRegConst import LocReg_Ito_C,LocReg_Ito_C_2,LocReg_Ito_C_4

# # print("setting license path")
# mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
# os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path

from utils.load_imports.loading import *

parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
cwd = os.getcwd()

cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

pat_tag = "MRR"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
series_tag = "SpanRegFig"
output_folder = f"SimulationSets/{pat_tag}/{series_tag}"
date = datetime.now().strftime("%Y%m%d")

# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + output_folder 

def create_result_folder(string, SNR):
    # Create a folder based on the current date and time
    #folder_name = f"c:/Users/kimjosy/LocReg_Regularization/{string}_{date}_SNR_{SNR}"
    folder_name = f"{cwd_full}/{string}_{date}_SNR_{SNR}"
    # folder_name = f"/Volumes/Lexar/NIH/Experiments/GridSearch/{string}_{date}_Run"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name



def load_Gaus(Gaus_info):
    T2 = Gaus_info['T2'].flatten()
    TE = Gaus_info['TE'].flatten()
    A = Gaus_info['A']
    LGBs = Gaus_info['LGBs']
    Lambda = Gaus_info['Lambda'].reshape(-1,1)
    Lambda_1 = np.logspace(np.log10(Lambda[0]), np.log10(Lambda[-1]), len(Lambda))
    nLambda_1 = len(Lambda_1)
    nLambda = len(Lambda)
    nGaus = Gaus_info['LGBs'].shape[1]
    n, m = Gaus_info['A'].shape
    return m


def gen_rps(rps):
    rps = np.linspace(1, 4, 5).T
    mps = rps / 2
    nrps = len(rps)
    T2_left = 30 * np.ones(nrps)
    T2_mid = T2_left * mps
    T2_right = T2_left * rps
    T2mu = np.column_stack((T2_left,  T2_right))
    return rps, T2mu

def gen_peaks():
    npeaks = 2
    nsigma = 5
    unif_sigma = np.linspace(2, 5, nsigma).T
    diff_sigma = np.column_stack((unif_sigma, 3 * unif_sigma))
    f_coef = np.ones(npeaks)
    return nsigma, npeaks, diff_sigma, f_coef


    
    df = pd.Series(["nsigma"])
    return nsigma, npeaks, rps, m, diff_sigma, f_coef, 

def get_func_name(func):
    f_name = func.__name__
    print
    raise Exception("not a valid function")
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
exp = 0.5
feedback = False

nsigma
diff_sigma
rps
npeaks
m
T2mu
T2
f_coef
A
SNR
TE
def run():


sigma_i = diff_sigma[i, :]

            for j in range(len(rps)):
def rps_run():
    kl_scores_list = []
    l2_rmsscores_list = []
    wass_scores_list = []
    p = np.zeros((npeaks, m))
    T2mu_sim = T2mu[j, :]
    for ii in range(npeaks):
        p[ii, :] = normsci.pdf(T2, T2mu_sim[ii], sigma_i[ii])
    IdealModel_weighted = p.T @ (f_coef) / npeaks

    dat_noiseless = A @ IdealModel_weighted
    noise = np.column_stack([np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE), 1)]) 
    dat_noisy = dat_noiseless + np.ravel(noise)
    # dat_noisy = dat_noiseless + np.ravel(noise[l,i,j,:])

    print('evaluating ' + str(i) + '-th sigma and ' + str(j) + '-th ratio peak separation test')
    # Online computation
    # DP
    f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)
    # L curve
    f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
    # GCV
    f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
    # Multi_Reg_Gaussian_Sum1
    # f_rec, alpha_L2, F_info, C_L2 = Multi_Reg_Gaussian_Sum1(dat_noisy, Gaus_info)
    # LocReg
    #maxiter = 400
    ep1 = 1e-2; # 1/(|x|+ep1)
    ep2 = 1e-2; # norm(dx)/norm(x)
    ep3 = 1e-2; # norm(x_(k-1) - x_k)/norm(x_(k-1))
    ep4 = 1e-4; # lb for ep1
    # maxiter = 400
    LRIto_ini_lam = 1e-3
    # print(LRIto_ini_lam)
            # gamma_init = 10
    gamma_init = 5
    maxiter = 75
            # best_f_rec, fin_etas = Ito_LocReg(data_noisy, G, LRIto_ini_lam, gamma_init, param_num, B_mats, maxiter)
            # # print("Completed 2P Ito")
            # com_vec_ItoLR2P[i] = norm(g - best_f_rec)
            # lam_ItoLR2P[:,i] = fin_etas

            # res_vec_ItoLR2P[:,i] =

            # print("Starting NP Ito")
    f_rec_LocReg_LC, lambda_locreg_LC = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
    # f_rec_LocReg_LC, lambda_locreg_LC = LocReg(dat_noisy, A, f_rec_LC, maxiter)
    # f_rec_LocReg_DP, lambda_locreg_DP = LocReg(dat_noisy, A, f_rec_DP, maxiter)
    # f_rec_LocReg_GCV, lambda_locreg_GCV = LocReg(dat_noisy, A, f_rec_GCV.ravel(), maxiter)
    #Deriv
    LR_mod_rec1, fin_lam1 = LocReg_Ito_C(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
    LR_Ito_lams = fin_lam1

    #No Deriv
    LR_mod_rec2, fin_lam2 = LocReg_Ito_C_2(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
    LR_Ito_lams2 = fin_lam2

    #Gamma Deriv
    LR_mod_rec4, fin_lam4, gamma_new2 = LocReg_Ito_C_4(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
    LR_Ito_lams4 = fin_lam4

    # f_rec_LocReg_NE, lambda_locreg_NE = LocReg_NEW_NNLS(dat_noisy, A, f_rec_LC, maxiter)
    # f_rec_LocReg, lambda_locreg = LocReg_unconstrainedB(dat_noisy, A, f_rec_GCV,lambda_GCV, ep1,ep2,ep3,ep4)
    # Save to dataset
    # MultiReg_data[i, j, :] = f_rec
    GCV_data[i, j, :] = f_rec_GCV.ravel()
    DP_data[i, j, :] = f_rec_DP
    LC_data[i, j, :] = f_rec_LC
    LocReg_data_LC[i, j, :] = f_rec_LocReg_LC

    ss_LocReg_deriv_data[i, j, :] = LR_mod_rec1
    ss_LocReg_data[i, j, :] = LR_mod_rec2
    ts_LocReg_deriv_data[i, j, :] =LR_mod_rec4


                for j in range(len(rps)):
                    p = np.zeros((npeaks, m))
                    T2mu_sim = T2mu[j, :]
                    for ii in range(npeaks):
                        p[ii, :] = normsci.pdf(T2, T2mu_sim[ii], sigma_i[ii])
                    IdealModel_weighted = p.T @ f_coef / npeaks

                    dat_noiseless = A @ IdealModel_weighted
                    # dat_noiseless = np.column_stack([dat_noiseless])
                    noise = np.column_stack([np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE), 1)]) 
                    
                    dat_noisy = dat_noiseless + np.ravel(noise)
        
                    #dat_noisy = dat_noiseless + np.ravel(noise[l,i,j,:])

                    #dat_noisy = dat_noiseless + np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE))

                    # Online computation
                    # DP
                    f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)
                    # L curve
                    f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
                    # GCV
                    f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
                    f_rec_GCV = f_rec_GCV[:,0]
                    # Multi_Reg_Gaussian_Sum1
                    f_rec, alpha_L2, F_info, C_L2 = Multi_Reg_Gaussian_Sum1(dat_noisy, Gaus_info)
                    # print("f_rec_SpanReg:", f_rec)
                    # LocReg
                    lambda_GCV = np.squeeze(lambda_GCV)
                    # maxiter = 400
                    #original: maxiter = 400
                    # f_rec_LocReg, lambda_locreg = LocReg(dat_noisy, A, f_rec_LC, maxiter)
                    # f_rec_LocReg_LC, lambda_locreg_LC = LocReg(dat_noisy, A, f_rec_LC, maxiter)
                    # f_rec_LocReg_DP, lambda_locreg_DP = LocReg(dat_noisy, A, f_rec_DP, maxiter)
                    # f_rec_LocReg_GCV, lambda_locreg_GCV = LocReg(dat_noisy, A, f_rec_GCV.ravel(), maxiter)
                    # f_rec_LocReg_NE, lambda_locreg_NE = LocReg_NEW_NNLS(dat_noisy, A, f_rec_LC, maxiter)
                    LRIto_ini_lam = 1e-3
                            # gamma_init = 10
                    gamma_init = 5
                    maxiter = 75
                    f_rec_LocReg_LC, lambda_locreg_LC = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
                    
                    LR_mod_rec1, fin_lam1 = LocReg_Ito_C(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                    LR_Ito_lams = fin_lam1

                    #No Deriv
                    LR_mod_rec2, fin_lam2 = LocReg_Ito_C_2(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                    LR_Ito_lams2 = fin_lam2

                    #Gamma Deriv
                    LR_mod_rec4, fin_lam4, gamma_new2 = LocReg_Ito_C_4(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                    LR_Ito_lams4 = fin_lam4

                    # print("f_rec_GCV:",f_rec_GCV.shape)
                    # print("f_rec_LocReg_NE:",f_rec_LocReg_NE)

                    # Save to dataset
                    MultiReg_data[i, j, :] = f_rec.flatten()
                    GCV_data[i, j, :] = f_rec_GCV.flatten()
                    DP_data[i, j, :] = f_rec_DP.flatten()
                    LC_data[i, j, :] = f_rec_LC.flatten()
                    # LocReg_data[i, j, :] = f_rec_LocReg.flatten()
                    LocReg_data_LC[i, j, :] = f_rec_LocReg_LC
                    ss_LocReg_deriv_data[i, j, :] = LR_mod_rec1
                    ss_LocReg_data[i, j, :] = LR_mod_rec2
                    ts_LocReg_deriv_data[i, j, :] =LR_mod_rec4
                    # LocReg_data_GCV[i, j, :] = f_rec_LocReg_GCV
                    # LocReg_data_DP[i, j, :] = f_rec_LocReg_DP
                    # LocReg_NE_data[i, j, :] = f_rec_LocReg_NE.flatten()

                    true_norm = linalg_norm(IdealModel_weighted)
                    MDL_err[i, j] = linalg_norm(IdealModel_weighted - f_rec) / true_norm
                    MDL_err_GCV[i, j] = linalg_norm(IdealModel_weighted - f_rec_GCV) / true_norm
                    gcverr = linalg_norm(IdealModel_weighted - f_rec_GCV) / true_norm
                    MDL_err_DP[i, j] = linalg_norm(IdealModel_weighted - f_rec_DP) / true_norm
                    dperr = linalg_norm(IdealModel_weighted - f_rec_DP) / true_norm
                    MDL_err_LC[i, j] = linalg_norm(IdealModel_weighted - f_rec_LC) / true_norm
                    lcerr =  linalg_norm(IdealModel_weighted - f_rec_LC) / true_norm
                    # MDL_err_LocReg[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg) / true_norm
                    MDL_err_LocReg_LC[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_LC) / true_norm
                    locregerr = linalg_norm(IdealModel_weighted - f_rec_LocReg_LC) / true_norm
                    # MDL_err_LocReg_DP[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_DP) / true_norm
                    # MDL_err_LocReg_GCV[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_GCV) / true_norm
                    # MDL_err_LocReg_NE[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_NE) / true_norm
                    MDL_err_ss_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm
                    ss_LocReg_deriv_err = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm

                    MDL_err_ss_LocReg[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm
                    ss_LocReg_err = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm

                    MDL_err_ts_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm
                    ts_LocReg_deriv_err = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm

                    error_multi_reg[i, j] = MDL_err[i, j]
                    error_GCV[i, j] = MDL_err_GCV[i, j]
                    error_DP[i, j] = MDL_err_DP[i, j]
                    error_LC[i, j] = MDL_err_LC[i, j]
                    # error_LocReg[i, j] = MDL_err_LocReg[i, j]
                    error_LocReg_LC[i, j] = MDL_err_LocReg_LC[i, j]
                    error_ss_LocReg_deriv[i, j] = MDL_err_ss_LocReg_deriv[i, j]
                    error_ss_LocReg[i, j] = MDL_err_ss_LocReg[i, j]
                    error_ts_LocReg_deriv[i, j] = MDL_err_ts_LocReg_deriv[i, j]
                    # error_LocReg_DP[i, j, l] = MDL_err_LocReg_DP[i, j]
                    # error_LocReg_GCV[i, j, l] = MDL_err_LocReg_GCV[i, j]

                    # error_LocReg_NE[i, j, l] = MDL_err_LocReg_NE[i, j]

                    plt.figure(figsize=(12.06, 4.2))
                    # plt.figure(figsize=(6.03,4.2))
                        # Set y-axis limits directly
                    # plt.subplots_adjust(wspace=0.3)

                    # Plotting the first subplot
                    plt.subplot(1, 3, 1) 
                    plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
                    plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg (Error: {round(locregerr,3)})')
                    plt.plot(T2, LR_mod_rec1, linestyle='-', linewidth=3, color='green', label=f'ss_LocReg_deriv (Error: {round(ss_LocReg_deriv_err,3)})')
                    plt.plot(T2, LR_mod_rec2, linestyle='-.', linewidth=3, color='magenta', label=f'ss_LocReg (Error: {round(ss_LocReg_err,3)})')
                    plt.plot(T2, LR_mod_rec4, linestyle='-.', linewidth=3, color='blue', label=f'ts_LocReg_deriv (Error: {round(ts_LocReg_deriv_err,3)})')

                    # plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {round(dperr,3)})')
                    # plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {round(gcverr,3)})')
                    # plt.plot(T2, f_rec_LocReg_NE, linestyle='--', linewidth=3, color='red', label='New LocReg Expression')
                    plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Error: {round(lcerr,3)})')
                    # plt.plot(T2, f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
                    plt.legend(fontsize=10, loc='best')
                    plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
                    plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
                    ymax = np.max(IdealModel_weighted) * 1.15
                    plt.ylim(0, ymax)

                    # Plotting the second subplot
                    plt.subplot(1, 3, 2)
                    # print("f_rec_LocReg_LC.shape", f_rec_LocReg_LC.shape)
                    plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
                    # plt.plot(TE, A @ f_rec_LocReg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
                    plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label='LocReg')
                    
                    plt.plot(TE, A @ LR_mod_rec1, linestyle='-', linewidth=3, color='green', label='ss_LocReg_deriv')
                    plt.plot(TE, A @ LR_mod_rec2, linestyle='-.', linewidth=3, color='magenta', label='ss_LocReg')
                    plt.plot(TE, A @ LR_mod_rec4, linestyle=':', linewidth=3, color='blue', label='ts_LocReg_deriv')

                    # plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
                    # plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
                    # plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
                    # plt.plot(TE, A @ f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
                    plt.legend(fontsize=10, loc='best')
                    plt.xlabel('TE', fontsize=20, fontweight='bold')
                    plt.ylabel('Intensity', fontsize=20, fontweight='bold')
                    
                    plt.subplot(1, 3, 3)
                    # plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
                    # plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
                    plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
                    plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label='LocReg')
                    plt.semilogy(T2, LR_Ito_lams * np.ones(len(T2)), linestyle='-', linewidth=3, color='green', label='ss_LocReg_deriv')
                    plt.semilogy(T2, LR_Ito_lams2 * np.ones(len(T2)), linestyle='-.', linewidth=3, color='magenta', label='ss_LocReg')
                    plt.semilogy(T2, LR_Ito_lams4 * np.ones(len(T2)), linestyle='-.', linewidth=3, color='blue', label='ts_LocReg_deriv')

                    plt.legend(fontsize=10, loc='best')
                    plt.xlabel('T2', fontsize=20, fontweight='bold')
                    plt.ylabel('Lambda', fontsize=20, fontweight='bold')

                    plt.tight_layout()
                    string = "MRR_1D_LocReg_Comparison"
                    file_path = create_result_folder(string, SNR)
                    plt.savefig(os.path.join(file_path, f"Comparison_figure_{i}_{j}.png"))
                    print(f"Saved Comparison Plot for {i}_{j}")
                    # plt.savefig(f'figure_{i}_{j}.png')
                    plt.close()  

def plot():
#    for j in range(len(rps)):
    p = np.zeros((npeaks, m))
    T2mu_sim = T2mu[j, :]
    for ii in range(npeaks):
        p[ii, :] = normsci.pdf(T2, T2mu_sim[ii], sigma_i[ii])
    IdealModel_weighted = p.T @ f_coef / npeaks

    dat_noiseless = A @ IdealModel_weighted
    # dat_noiseless = np.column_stack([dat_noiseless])
    noise = np.column_stack([np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE), 1)]) 
    
    dat_noisy = dat_noiseless + np.ravel(noise)

    #dat_noisy = dat_noiseless + np.ravel(noise[l,i,j,:])

    #dat_noisy = dat_noiseless + np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE))

    # Online computation
    # DP
    f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)
    # L curve
    f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
    # GCV
    f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
    f_rec_GCV = f_rec_GCV[:,0]
    # Multi_Reg_Gaussian_Sum1
    f_rec, alpha_L2, F_info, C_L2 = Multi_Reg_Gaussian_Sum1(dat_noisy, Gaus_info)
    # print("f_rec_SpanReg:", f_rec)
    # LocReg
    lambda_GCV = np.squeeze(lambda_GCV)
    # maxiter = 400
    #original: maxiter = 400
    # f_rec_LocReg, lambda_locreg = LocReg(dat_noisy, A, f_rec_LC, maxiter)
    # f_rec_LocReg_LC, lambda_locreg_LC = LocReg(dat_noisy, A, f_rec_LC, maxiter)
    # f_rec_LocReg_DP, lambda_locreg_DP = LocReg(dat_noisy, A, f_rec_DP, maxiter)
    # f_rec_LocReg_GCV, lambda_locreg_GCV = LocReg(dat_noisy, A, f_rec_GCV.ravel(), maxiter)
    # f_rec_LocReg_NE, lambda_locreg_NE = LocReg_NEW_NNLS(dat_noisy, A, f_rec_LC, maxiter)
    LRIto_ini_lam = 1e-3
            # gamma_init = 10
    gamma_init = 5
    maxiter = 75
    f_rec_LocReg_LC, lambda_locreg_LC = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
    
    LR_mod_rec1, fin_lam1 = LocReg_Ito_C(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
    LR_Ito_lams = fin_lam1

    #No Deriv
    LR_mod_rec2, fin_lam2 = LocReg_Ito_C_2(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
    LR_Ito_lams2 = fin_lam2

    #Gamma Deriv
    LR_mod_rec4, fin_lam4, gamma_new2 = LocReg_Ito_C_4(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
    LR_Ito_lams4 = fin_lam4

    # print("f_rec_GCV:",f_rec_GCV.shape)
    # print("f_rec_LocReg_NE:",f_rec_LocReg_NE)

    # Save to dataset
    MultiReg_data[i, j, :] = f_rec.flatten()
    GCV_data[i, j, :] = f_rec_GCV.flatten()
    DP_data[i, j, :] = f_rec_DP.flatten()
    LC_data[i, j, :] = f_rec_LC.flatten()
    # LocReg_data[i, j, :] = f_rec_LocReg.flatten()
    LocReg_data_LC[i, j, :] = f_rec_LocReg_LC
    ss_LocReg_deriv_data[i, j, :] = LR_mod_rec1
    ss_LocReg_data[i, j, :] = LR_mod_rec2
    ts_LocReg_deriv_data[i, j, :] =LR_mod_rec4
    # LocReg_data_GCV[i, j, :] = f_rec_LocReg_GCV
    # LocReg_data_DP[i, j, :] = f_rec_LocReg_DP
    # LocReg_NE_data[i, j, :] = f_rec_LocReg_NE.flatten()

    true_norm = linalg_norm(IdealModel_weighted)
    MDL_err[i, j] = linalg_norm(IdealModel_weighted - f_rec) / true_norm
    MDL_err_GCV[i, j] = linalg_norm(IdealModel_weighted - f_rec_GCV) / true_norm
    gcverr = linalg_norm(IdealModel_weighted - f_rec_GCV) / true_norm
    MDL_err_DP[i, j] = linalg_norm(IdealModel_weighted - f_rec_DP) / true_norm
    dperr = linalg_norm(IdealModel_weighted - f_rec_DP) / true_norm
    MDL_err_LC[i, j] = linalg_norm(IdealModel_weighted - f_rec_LC) / true_norm
    lcerr =  linalg_norm(IdealModel_weighted - f_rec_LC) / true_norm
    # MDL_err_LocReg[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg) / true_norm
    MDL_err_LocReg_LC[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_LC) / true_norm
    locregerr = linalg_norm(IdealModel_weighted - f_rec_LocReg_LC) / true_norm
    # MDL_err_LocReg_DP[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_DP) / true_norm
    # MDL_err_LocReg_GCV[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_GCV) / true_norm
    # MDL_err_LocReg_NE[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_NE) / true_norm
    MDL_err_ss_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm
    ss_LocReg_deriv_err = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm

    MDL_err_ss_LocReg[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm
    ss_LocReg_err = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm

    MDL_err_ts_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm
    ts_LocReg_deriv_err = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm

    error_multi_reg[i, j] = MDL_err[i, j]
    error_GCV[i, j] = MDL_err_GCV[i, j]
    error_DP[i, j] = MDL_err_DP[i, j]
    error_LC[i, j] = MDL_err_LC[i, j]
    # error_LocReg[i, j] = MDL_err_LocReg[i, j]
    error_LocReg_LC[i, j] = MDL_err_LocReg_LC[i, j]
    error_ss_LocReg_deriv[i, j] = MDL_err_ss_LocReg_deriv[i, j]
    error_ss_LocReg[i, j] = MDL_err_ss_LocReg[i, j]
    error_ts_LocReg_deriv[i, j] = MDL_err_ts_LocReg_deriv[i, j]
    # error_LocReg_DP[i, j, l] = MDL_err_LocReg_DP[i, j]
    # error_LocReg_GCV[i, j, l] = MDL_err_LocReg_GCV[i, j]

    # error_LocReg_NE[i, j, l] = MDL_err_LocReg_NE[i, j]

    plt.figure(figsize=(12.06, 4.2))
    # plt.figure(figsize=(6.03,4.2))
        # Set y-axis limits directly
    # plt.subplots_adjust(wspace=0.3)

    # Plotting the first subplot
    plt.subplot(1, 3, 1) 
    plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
    plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg (Error: {round(locregerr,3)})')
    plt.plot(T2, LR_mod_rec1, linestyle='-', linewidth=3, color='green', label=f'ss_LocReg_deriv (Error: {round(ss_LocReg_deriv_err,3)})')
    plt.plot(T2, LR_mod_rec2, linestyle='-.', linewidth=3, color='magenta', label=f'ss_LocReg (Error: {round(ss_LocReg_err,3)})')
    plt.plot(T2, LR_mod_rec4, linestyle='-.', linewidth=3, color='blue', label=f'ts_LocReg_deriv (Error: {round(ts_LocReg_deriv_err,3)})')

    # plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {round(dperr,3)})')
    # plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {round(gcverr,3)})')
    # plt.plot(T2, f_rec_LocReg_NE, linestyle='--', linewidth=3, color='red', label='New LocReg Expression')
    plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Error: {round(lcerr,3)})')
    # plt.plot(T2, f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
    plt.legend(fontsize=10, loc='best')
    plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
    ymax = np.max(IdealModel_weighted) * 1.15
    plt.ylim(0, ymax)

    # Plotting the second subplot
    plt.subplot(1, 3, 2)
    # print("f_rec_LocReg_LC.shape", f_rec_LocReg_LC.shape)
    plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
    # plt.plot(TE, A @ f_rec_LocReg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
    plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label='LocReg')
    
    plt.plot(TE, A @ LR_mod_rec1, linestyle='-', linewidth=3, color='green', label='ss_LocReg_deriv')
    plt.plot(TE, A @ LR_mod_rec2, linestyle='-.', linewidth=3, color='magenta', label='ss_LocReg')
    plt.plot(TE, A @ LR_mod_rec4, linestyle=':', linewidth=3, color='blue', label='ts_LocReg_deriv')

    # plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
    # plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
    # plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
    # plt.plot(TE, A @ f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
    plt.legend(fontsize=10, loc='best')
    plt.xlabel('TE', fontsize=20, fontweight='bold')
    plt.ylabel('Intensity', fontsize=20, fontweight='bold')
    
    plt.subplot(1, 3, 3)
    # plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
    # plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
    plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
    plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label='LocReg')
    plt.semilogy(T2, LR_Ito_lams * np.ones(len(T2)), linestyle='-', linewidth=3, color='green', label='ss_LocReg_deriv')
    plt.semilogy(T2, LR_Ito_lams2 * np.ones(len(T2)), linestyle='-.', linewidth=3, color='magenta', label='ss_LocReg')
    plt.semilogy(T2, LR_Ito_lams4 * np.ones(len(T2)), linestyle='-.', linewidth=3, color='blue', label='ts_LocReg_deriv')

    plt.legend(fontsize=10, loc='best')
    plt.xlabel('T2', fontsize=20, fontweight='bold')
    plt.ylabel('Lambda', fontsize=20, fontweight='bold')

    plt.tight_layout()
    string = "MRR_1D_LocReg_Comparison"
    file_path = create_result_folder(string, SNR)
    plt.savefig(os.path.join(file_path, f"Comparison_figure_{i}_{j}.png"))
    print(f"Saved Comparison Plot for {i}_{j}")
    # plt.savefig(f'figure_{i}_{j}.png')
    plt.close()  
        
def heatmap_unequal_width_All(Gaus_info, show):
    # print("Gaus_info keys:", Gaus_info.keys())
    # print("Gaus_info type:", type(Gaus_info))



    # # Create arrays to store intermediate results
    # avg_MDL_err = np.zeros((nsigma, nrps))
    # # avg_MDL_err_GCV = np.zeros((nsigma, nrps))
    # avg_MDL_err_GCV = []
    # avg_MDL_err_DP = np.zeros((nsigma, nrps))
    # avg_MDL_err_LC = np.zeros((nsigma, nrps))
    # avg_MDL_err_LocReg_LC = np.zeros((nsigma, nrps))

    # avg_MDL_err_ss_LocReg_deriv = np.zeros((nsigma, nrps))
    # avg_MDL_err_ss_LocReg = np.zeros((nsigma, nrps))
    # avg_MDL_err_ts_LocReg_deriv = np.zeros((nsigma, nrps))
    # # avg_MDL_err_LocReg_DP = np.zeros((nsigma, nrps))
    # # avg_MDL_err_LocReg_GCV = np.zeros((nsigma, nrps))
    # avg_MDL_err_LocReg_GCV = []

    # avg_MDL_err_LocReg_NE = np.zeros((nsigma, nrps))

    # error_multi_reg = np.zeros((nsigma, nrps))
    # error_GCV = np.zeros((nsigma, nrps))
    # error_DP = np.zeros((nsigma, nrps))
    # error_LC = np.zeros((nsigma, nrps))
    # error_LocReg_LC = np.zeros((nsigma, nrps))
    # # error_LocReg_GCV = np.zeros((nsigma, nrps, n_sim))
    # # error_LocReg_DP = np.zeros((nsigma, nrps, n_sim))
    # error_ss_LocReg_deriv = np.zeros((nsigma, nrps))
    # error_ss_LocReg = np.zeros((nsigma, nrps))
    # error_ts_LocReg_deriv = np.zeros((nsigma, nrps))

    # # error_LocReg_NE = np.zeros((nsigma, nrps, n_sim))

    # # error_multi_reg_wass = np.zeros((nsigma, nrps, n_sim))
    # # error_GCV_wass = np.zeros((nsigma, nrps, n_sim))
    # # error_DP_wass = np.zeros((nsigma, nrps, n_sim))
    # # error_LC_wass = np.zeros((nsigma, nrps, n_sim))
    # # error_LocReg_wass = np.zeros((nsigma, nrps, n_sim))

    # # error_multi_reg_KL = np.zeros((nsigma, nrps, n_sim))
    # # error_GCV_KL = np.zeros((nsigma, nrps, n_sim))
    # # error_DP_KL = np.zeros((nsigma, nrps, n_sim))
    # # error_LC_KL = np.zeros((nsigma, nrps, n_sim))
    # # error_LocReg_KL = np.zeros((nsigma, nrps, n_sim))

    # SNR = Gaus_info['SNR']

    # # Create dataset arrays to save the results
    # # feature_df = pd.DataFrame(columns = ["TI1*g","TI2*g","TI_DATA", "MSE", "var", "bias", "pEst_cvn", "pEst_AIC", "pEst_cf"])

    # MultiReg_data = np.zeros(( nsigma, nrps, m))
    # GCV_data = np.zeros(( nsigma, nrps, m))
    # DP_data = np.zeros(( nsigma, nrps, m))
    # LC_data = np.zeros(( nsigma, nrps, m))
    # # LocReg_data_LC = np.zeros(( nsigma, nrps, m))
    # # LocReg_data_GCV = np.zeros(( nsigma, nrps, m))
    # # LocReg_data_DP = np.zeros(( nsigma, nrps, m))
    # LocReg_data_LC = np.zeros(( nsigma, nrps, m))
    # # LocReg_data_GCV = np.zeros(( nsigma, nrps, m))
    # # LocReg_data_DP = np.zeros(( nsigma, nrps, m))
    # IdealModel_data = np.zeros(( nsigma, nrps, m))
    # # LocReg_NE_data = np.zeros(( nsigma, nrps, m))
    # # LocReg_data_LC = np.zeros(( nsigma, nrps, m))
    # LocReg_data_LC = np.zeros(( nsigma, nrps, m))

    # ss_LocReg_deriv_data = np.zeros(( nsigma, nrps, m))
    # ss_LocReg_data = np.zeros(( nsigma, nrps, m))
    # ts_LocReg_deriv_data = np.zeros(( nsigma, nrps, m))
    # gammas_list = np.linspace(0,10,2)
    # kl_scores_list = []
    # l2_rmsscores_list = []
    # wass_scores_list = []
    # gammas_list = np.linspace(-4,4,5)

    # print("This is simulation " + f"{l}")
    # MDL_err = np.zeros((nsigma, nrps))
    # MDL_err_GCV = np.zeros((nsigma, nrps))
    # MDL_err_DP = np.zeros((nsigma, nrps))
    # MDL_err_LC = np.zeros((nsigma, nrps))
    # # MDL_err_LocReg_DP = np.zeros((nsigma, nrps))
    # MDL_err_LocReg_LC = np.zeros((nsigma, nrps))
    # # MDL_err_LocReg_GCV = np.zeros((nsigma, nrps))
    # # MDL_err_LocReg_NE = np.zeros((nsigma, nrps))
    # MDL_err_ss_LocReg_deriv = np.zeros((nsigma, nrps))
    # MDL_err_ss_LocReg = np.zeros((nsigma, nrps))
    # MDL_err_ts_LocReg_deriv = np.zeros((nsigma, nrps))


    # MDL_err_wass = np.zeros((nsigma, nrps))
    # MDL_err_GCV_wass = np.zeros((nsigma, nrps))
    # MDL_err_DP_wass = np.zeros((nsigma, nrps))
    # MDL_err_LC_wass = np.zeros((nsigma, nrps))
    # MDL_err_LocReg_wass = np.zeros((nsigma, nrps))

    # MDL_err_KL = np.zeros((nsigma, nrps))
    # MDL_err_GCV_KL = np.zeros((nsigma, nrps))
    # MDL_err_DP_KL = np.zeros((nsigma, nrps))
    # MDL_err_LC_KL = np.zeros((nsigma, nrps))
    # MDL_err_LocReg_KL = np.zeros((nsigma, nrps))

    eps1 = 1e-2
    ep_min = 1e-2
    eps_cut = 1.2
    eps_floor = 1e-4
    exp = 0.5
    feedback = False

    for i in range(nsigma):
        sigma_i = diff_sigma[i, :]

        if show == 0:
            for j in range(len(rps)):
                kl_scores_list = []
                l2_rmsscores_list = []
                wass_scores_list = []
                p = np.zeros((npeaks, m))
                T2mu_sim = T2mu[j, :]
                for ii in range(npeaks):
                    p[ii, :] = normsci.pdf(T2, T2mu_sim[ii], sigma_i[ii])
                IdealModel_weighted = p.T @ (f_coef) / npeaks

                dat_noiseless = A @ IdealModel_weighted
                noise = np.column_stack([np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE), 1)]) 
                dat_noisy = dat_noiseless + np.ravel(noise)
                # dat_noisy = dat_noiseless + np.ravel(noise[l,i,j,:])

                print('evaluating ' + str(i) + '-th sigma and ' + str(j) + '-th ratio peak separation test')
                # Online computation
                # DP
                f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)
                # L curve
                f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
                # GCV
                f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
                # Multi_Reg_Gaussian_Sum1
                # f_rec, alpha_L2, F_info, C_L2 = Multi_Reg_Gaussian_Sum1(dat_noisy, Gaus_info)
                # LocReg
                #maxiter = 400
                ep1 = 1e-2; # 1/(|x|+ep1)
                ep2 = 1e-2; # norm(dx)/norm(x)
                ep3 = 1e-2; # norm(x_(k-1) - x_k)/norm(x_(k-1))
                ep4 = 1e-4; # lb for ep1
                # maxiter = 400
                LRIto_ini_lam = 1e-3
                # print(LRIto_ini_lam)
                        # gamma_init = 10
                gamma_init = 5
                maxiter = 75
                        # best_f_rec, fin_etas = Ito_LocReg(data_noisy, G, LRIto_ini_lam, gamma_init, param_num, B_mats, maxiter)
                        # # print("Completed 2P Ito")
                        # com_vec_ItoLR2P[i] = norm(g - best_f_rec)
                        # lam_ItoLR2P[:,i] = fin_etas

                        # res_vec_ItoLR2P[:,i] =

                        # print("Starting NP Ito")
                f_rec_LocReg_LC, lambda_locreg_LC = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
                # f_rec_LocReg_LC, lambda_locreg_LC = LocReg(dat_noisy, A, f_rec_LC, maxiter)
                # f_rec_LocReg_DP, lambda_locreg_DP = LocReg(dat_noisy, A, f_rec_DP, maxiter)
                # f_rec_LocReg_GCV, lambda_locreg_GCV = LocReg(dat_noisy, A, f_rec_GCV.ravel(), maxiter)
                #Deriv
                LR_mod_rec1, fin_lam1 = LocReg_Ito_C(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                LR_Ito_lams = fin_lam1

                #No Deriv
                LR_mod_rec2, fin_lam2 = LocReg_Ito_C_2(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                LR_Ito_lams2 = fin_lam2

                #Gamma Deriv
                LR_mod_rec4, fin_lam4, gamma_new2 = LocReg_Ito_C_4(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                LR_Ito_lams4 = fin_lam4

                # f_rec_LocReg_NE, lambda_locreg_NE = LocReg_NEW_NNLS(dat_noisy, A, f_rec_LC, maxiter)
                # f_rec_LocReg, lambda_locreg = LocReg_unconstrainedB(dat_noisy, A, f_rec_GCV,lambda_GCV, ep1,ep2,ep3,ep4)
                # Save to dataset
                # MultiReg_data[i, j, :] = f_rec
                GCV_data[i, j, :] = f_rec_GCV.ravel()
                DP_data[i, j, :] = f_rec_DP
                LC_data[i, j, :] = f_rec_LC
                LocReg_data_LC[i, j, :] = f_rec_LocReg_LC

                ss_LocReg_deriv_data[i, j, :] = LR_mod_rec1
                ss_LocReg_data[i, j, :] = LR_mod_rec2
                ts_LocReg_deriv_data[i, j, :] =LR_mod_rec4
                # LocReg_data_GCV[i, j, :] = f_rec_LocReg_GCV
                # LocReg_data_DP[i, j, :] = f_rec_LocReg_DP

                # LocReg_NE_data[i, j, :] = f_rec_LocReg_NE
                # for gamma in tqdm(gammas_list):
                #     kl_scores = kl_div(T2, gamma, f_rec, IdealModel_weighted)
                #     l2_rmsscores = l2_rms(T2, gamma, data, g)
                #     wass_scores =  wass_m(T2, gamma,data, gnd_truth_dist)
                #     kl_scores_list.append(kl_scores)
                #     l2_rmsscores_list.append(l2_rmsscores)
                #     wass_scores_list.append(wass_scores)
                # for gamma in tqdm(gammas_list):
                #     kl_scores, l2_rmsscores, wass_scores = process_gamma(gamma)
                #     # kl_scores_list.append(kl_scores)
                #     l2_rmsscores_list.append(l2_rmsscores)
                #     # wass_scores_list.append(wass_scores)
                # np.argmin(l2_rmsscores_list)

                true_norm = linalg_norm(IdealModel_weighted)
                # MDL_err[i, j] = linalg_norm(IdealModel_weighted - f_rec) / true_norm
                MDL_err_GCV[i, j] = linalg_norm(IdealModel_weighted - f_rec_GCV) / true_norm
                MDL_err_DP[i, j] = linalg_norm(IdealModel_weighted - f_rec_DP) / true_norm
                MDL_err_LC[i, j] = linalg_norm(IdealModel_weighted - f_rec_LC) / true_norm
                MDL_err_LocReg_LC[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_LC) / true_norm
                MDL_err_ss_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm
                MDL_err_ss_LocReg[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm
                MDL_err_ts_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm
                # MDL_err_LocReg_DP[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_DP) / true_norm
                # MDL_err_LocReg_GCV[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_GCV) / true_norm

                # MDL_err_LocReg_NE[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_NE) / true_norm


                # MDL_err_wass[i, j] = linalg_norm(IdealModel_weighted - f_rec) / true_norm
                # MDL_err_GCV_wass[i, j] = linalg_norm(IdealModel_weighted - f_rec_GCV) / true_norm
                # MDL_err_DP_wass[i, j] = linalg_norm(IdealModel_weighted - f_rec_DP) / true_norm
                # MDL_err_LC_wass[i, j] = linalg_norm(IdealModel_weighted - f_rec_LC) / true_norm
                # MDL_err_LocReg_wass[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg) / true_norm
                
                # #Pay attention to order of arguments for interpretations
                # MDL_err_KL[i, j] = entropy(f_rec, IdealModel_weighted)
                # MDL_err_GCV_KL[i, j] = entropy(f_rec_GCV, IdealModel_weighted)
                # MDL_err_DP_KL[i, j] = entropy(f_rec_DP, IdealModel_weighted)
                # MDL_err_LC_KL[i, j] = entropy(f_rec_LC, IdealModel_weighted)
                # MDL_err_LocReg_KL[i, j] = entropy(f_rec_LocReg, IdealModel_weighted) 

                # error_multi_reg[i, j, l] = MDL_err[i, j]
                error_GCV[i, j, l] = MDL_err_GCV[i, j]
                error_DP[i, j, l] = MDL_err_DP[i, j]
                error_LC[i, j, l] = MDL_err_LC[i, j]
                error_LocReg_LC[i, j, l] = MDL_err_LocReg_LC[i, j]
                error_ss_LocReg_deriv[i, j, l] = MDL_err_ss_LocReg_deriv[i, j]
                error_ss_LocReg[i, j, l] = MDL_err_ss_LocReg[i, j]
                error_ts_LocReg_deriv[i, j, l] = MDL_err_ts_LocReg_deriv[i, j]

                # error_LocReg_DP[i, j, l] = MDL_err_LocReg_DP[i, j]
                # error_LocReg_GCV[i, j, l] = MDL_err_LocReg_GCV[i, j]

                # error_LocReg_NE[i, j, l] = MDL_err_LocReg_NE[i, j]

                # error_multi_reg_wass[i, j, l] = MDL_err_wass[i, j]
                # error_GCV_wass[i, j, l] = MDL_err_GCV_wass[i, j]
                # error_DP_wass[i, j, l] = MDL_err_DP_wass[i, j]
                # error_LC_wass[i, j, l] = MDL_err_LC_wass[i, j]
                # error_LocReg_wass[i, j, l] = MDL_err_LocReg_wass[i, j]

                # error_multi_reg_KL[i, j, l] = MDL_err_KL[i, j]
                # error_GCV_KL[i, j, l] = MDL_err_GCV_KL[i, j]
                # error_DP_KL[i, j, l] = MDL_err_DP_KL[i, j]
                # error_LC_KL[i, j, l] = MDL_err_LC_KL[i, j]
                # error_LocReg_KL[i, j, l] = MDL_err_LocReg_KL[i, j]
                # # Assuming you have the arrays T2, IdealModel_weighted, f_rec, f_rec_LocReg, f_rec_DP, f_rec_GCV, f_rec_LC, TE, and A

                # plt.figure(figsize=(12.06, 4.2))

                # # Plotting the first subplot
                # plt.subplot(1, 2, 1)
                # plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='True Dist.')
                # plt.plot(T2, f_rec, linestyle=':', linewidth=3, color='red', label='Multi-Reg')
                # plt.plot(T2, f_rec_LocReg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
                # plt.plot(T2, f_rec_DP, linewidth=3, color='yellow', label='DP')
                # plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
                # plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
                # plt.legend(fontsize=20, loc='best')
                # plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
                # plt.ylabel('Intensity', fontsize=20, fontweight='bold')
                # plt.title('Comparison of Recovered Distributions', fontsize=20, fontweight='bold')

                # # Plotting the second subplot
                # plt.subplot(1, 2, 2)
                # plt.plot(TE, A * IdealModel_weighted, linewidth=3, color='black', label='True Dist.')
                # plt.plot(TE, A * f_rec, linestyle=':', linewidth=3, color='red', label='Multi-Reg')
                # plt.plot(TE, A * f_rec_LocReg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
                # plt.plot(TE, A * f_rec_DP, linewidth=3, color='yellow', label='DP')
                # plt.plot(TE, A * f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
                # plt.plot(TE, A * f_rec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
                # plt.legend(fontsize=20, loc='best')
                # plt.xlabel('TE', fontsize=20, fontweight='bold')
                # plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
                # plt.title('Comparison of Recovered Data', fontsize=20, fontweight='bold')

                # plt.tight_layout()
                # plt.draw()
                # plt.show()
        else:
            for j in range(len(rps)):
                p = np.zeros((npeaks, m))
                T2mu_sim = T2mu[j, :]
                for ii in range(npeaks):
                    p[ii, :] = normsci.pdf(T2, T2mu_sim[ii], sigma_i[ii])
                IdealModel_weighted = p.T @ f_coef / npeaks

                dat_noiseless = A @ IdealModel_weighted
                # dat_noiseless = np.column_stack([dat_noiseless])
                noise = np.column_stack([np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE), 1)]) 
                
                dat_noisy = dat_noiseless + np.ravel(noise)
    
                #dat_noisy = dat_noiseless + np.ravel(noise[l,i,j,:])

                #dat_noisy = dat_noiseless + np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE))

                # Online computation
                # DP
                f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)
                # L curve
                f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
                # GCV
                f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
                f_rec_GCV = f_rec_GCV[:,0]
                # Multi_Reg_Gaussian_Sum1
                f_rec, alpha_L2, F_info, C_L2 = Multi_Reg_Gaussian_Sum1(dat_noisy, Gaus_info)
                # print("f_rec_SpanReg:", f_rec)
                # LocReg
                lambda_GCV = np.squeeze(lambda_GCV)
                # maxiter = 400
                #original: maxiter = 400
                # f_rec_LocReg, lambda_locreg = LocReg(dat_noisy, A, f_rec_LC, maxiter)
                # f_rec_LocReg_LC, lambda_locreg_LC = LocReg(dat_noisy, A, f_rec_LC, maxiter)
                # f_rec_LocReg_DP, lambda_locreg_DP = LocReg(dat_noisy, A, f_rec_DP, maxiter)
                # f_rec_LocReg_GCV, lambda_locreg_GCV = LocReg(dat_noisy, A, f_rec_GCV.ravel(), maxiter)
                # f_rec_LocReg_NE, lambda_locreg_NE = LocReg_NEW_NNLS(dat_noisy, A, f_rec_LC, maxiter)
                LRIto_ini_lam = 1e-3
                        # gamma_init = 10
                gamma_init = 5
                maxiter = 75
                f_rec_LocReg_LC, lambda_locreg_LC = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
                
                LR_mod_rec1, fin_lam1 = LocReg_Ito_C(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                LR_Ito_lams = fin_lam1

                #No Deriv
                LR_mod_rec2, fin_lam2 = LocReg_Ito_C_2(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                LR_Ito_lams2 = fin_lam2

                #Gamma Deriv
                LR_mod_rec4, fin_lam4, gamma_new2 = LocReg_Ito_C_4(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                LR_Ito_lams4 = fin_lam4

                # print("f_rec_GCV:",f_rec_GCV.shape)
                # print("f_rec_LocReg_NE:",f_rec_LocReg_NE)

                # Save to dataset
                MultiReg_data[i, j, :] = f_rec.flatten()
                GCV_data[i, j, :] = f_rec_GCV.flatten()
                DP_data[i, j, :] = f_rec_DP.flatten()
                LC_data[i, j, :] = f_rec_LC.flatten()
                # LocReg_data[i, j, :] = f_rec_LocReg.flatten()
                LocReg_data_LC[i, j, :] = f_rec_LocReg_LC
                ss_LocReg_deriv_data[i, j, :] = LR_mod_rec1
                ss_LocReg_data[i, j, :] = LR_mod_rec2
                ts_LocReg_deriv_data[i, j, :] =LR_mod_rec4
                # LocReg_data_GCV[i, j, :] = f_rec_LocReg_GCV
                # LocReg_data_DP[i, j, :] = f_rec_LocReg_DP
                # LocReg_NE_data[i, j, :] = f_rec_LocReg_NE.flatten()

                true_norm = linalg_norm(IdealModel_weighted)
                MDL_err[i, j] = linalg_norm(IdealModel_weighted - f_rec) / true_norm
                MDL_err_GCV[i, j] = linalg_norm(IdealModel_weighted - f_rec_GCV) / true_norm
                gcverr = linalg_norm(IdealModel_weighted - f_rec_GCV) / true_norm
                MDL_err_DP[i, j] = linalg_norm(IdealModel_weighted - f_rec_DP) / true_norm
                dperr = linalg_norm(IdealModel_weighted - f_rec_DP) / true_norm
                MDL_err_LC[i, j] = linalg_norm(IdealModel_weighted - f_rec_LC) / true_norm
                lcerr =  linalg_norm(IdealModel_weighted - f_rec_LC) / true_norm
                # MDL_err_LocReg[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg) / true_norm
                MDL_err_LocReg_LC[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_LC) / true_norm
                locregerr = linalg_norm(IdealModel_weighted - f_rec_LocReg_LC) / true_norm
                # MDL_err_LocReg_DP[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_DP) / true_norm
                # MDL_err_LocReg_GCV[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_GCV) / true_norm
                # MDL_err_LocReg_NE[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_NE) / true_norm
                MDL_err_ss_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm
                ss_LocReg_deriv_err = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm

                MDL_err_ss_LocReg[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm
                ss_LocReg_err = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm

                MDL_err_ts_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm
                ts_LocReg_deriv_err = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm

                error_multi_reg[i, j] = MDL_err[i, j]
                error_GCV[i, j] = MDL_err_GCV[i, j]
                error_DP[i, j] = MDL_err_DP[i, j]
                error_LC[i, j] = MDL_err_LC[i, j]
                # error_LocReg[i, j] = MDL_err_LocReg[i, j]
                error_LocReg_LC[i, j] = MDL_err_LocReg_LC[i, j]
                error_ss_LocReg_deriv[i, j] = MDL_err_ss_LocReg_deriv[i, j]
                error_ss_LocReg[i, j] = MDL_err_ss_LocReg[i, j]
                error_ts_LocReg_deriv[i, j] = MDL_err_ts_LocReg_deriv[i, j]
                # error_LocReg_DP[i, j, l] = MDL_err_LocReg_DP[i, j]
                # error_LocReg_GCV[i, j, l] = MDL_err_LocReg_GCV[i, j]

                # error_LocReg_NE[i, j, l] = MDL_err_LocReg_NE[i, j]

                plt.figure(figsize=(12.06, 4.2))
                # plt.figure(figsize=(6.03,4.2))
                    # Set y-axis limits directly
                # plt.subplots_adjust(wspace=0.3)

                # Plotting the first subplot
                plt.subplot(1, 3, 1) 
                plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
                plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg (Error: {round(locregerr,3)})')
                plt.plot(T2, LR_mod_rec1, linestyle='-', linewidth=3, color='green', label=f'ss_LocReg_deriv (Error: {round(ss_LocReg_deriv_err,3)})')
                plt.plot(T2, LR_mod_rec2, linestyle='-.', linewidth=3, color='magenta', label=f'ss_LocReg (Error: {round(ss_LocReg_err,3)})')
                plt.plot(T2, LR_mod_rec4, linestyle='-.', linewidth=3, color='blue', label=f'ts_LocReg_deriv (Error: {round(ts_LocReg_deriv_err,3)})')

                # plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {round(dperr,3)})')
                # plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {round(gcverr,3)})')
                # plt.plot(T2, f_rec_LocReg_NE, linestyle='--', linewidth=3, color='red', label='New LocReg Expression')
                plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Error: {round(lcerr,3)})')
                # plt.plot(T2, f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
                plt.legend(fontsize=10, loc='best')
                plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
                plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
                ymax = np.max(IdealModel_weighted) * 1.15
                plt.ylim(0, ymax)

                # Plotting the second subplot
                plt.subplot(1, 3, 2)
                # print("f_rec_LocReg_LC.shape", f_rec_LocReg_LC.shape)
                plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
                # plt.plot(TE, A @ f_rec_LocReg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
                plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label='LocReg')
                
                plt.plot(TE, A @ LR_mod_rec1, linestyle='-', linewidth=3, color='green', label='ss_LocReg_deriv')
                plt.plot(TE, A @ LR_mod_rec2, linestyle='-.', linewidth=3, color='magenta', label='ss_LocReg')
                plt.plot(TE, A @ LR_mod_rec4, linestyle=':', linewidth=3, color='blue', label='ts_LocReg_deriv')

                # plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
                # plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
                # plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
                # plt.plot(TE, A @ f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
                plt.legend(fontsize=10, loc='best')
                plt.xlabel('TE', fontsize=20, fontweight='bold')
                plt.ylabel('Intensity', fontsize=20, fontweight='bold')
                
                plt.subplot(1, 3, 3)
                # plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
                # plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
                plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
                plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label='LocReg')
                plt.semilogy(T2, LR_Ito_lams * np.ones(len(T2)), linestyle='-', linewidth=3, color='green', label='ss_LocReg_deriv')
                plt.semilogy(T2, LR_Ito_lams2 * np.ones(len(T2)), linestyle='-.', linewidth=3, color='magenta', label='ss_LocReg')
                plt.semilogy(T2, LR_Ito_lams4 * np.ones(len(T2)), linestyle='-.', linewidth=3, color='blue', label='ts_LocReg_deriv')

                plt.legend(fontsize=10, loc='best')
                plt.xlabel('T2', fontsize=20, fontweight='bold')
                plt.ylabel('Lambda', fontsize=20, fontweight='bold')

                plt.tight_layout()
                string = "MRR_1D_LocReg_Comparison"
                file_path = create_result_folder(string, SNR)
                plt.savefig(os.path.join(file_path, f"Comparison_figure_{i}_{j}.png"))
                print(f"Saved Comparison Plot for {i}_{j}")
                # plt.savefig(f'figure_{i}_{j}.png')
                plt.close()  
                # plt.draw()
                # plt.show()

        avg_MDL_err += MDL_err
        avg_MDL_err_GCV.append(MDL_err_GCV)
        avg_MDL_err_DP += MDL_err_DP
        avg_MDL_err_LC += MDL_err_LC
        avg_MDL_err_LocReg_LC += MDL_err_LocReg_LC
        avg_MDL_err_ss_LocReg_deriv += MDL_err_ss_LocReg_deriv
        avg_MDL_err_ss_LocReg += MDL_err_ss_LocReg
        avg_MDL_err_ts_LocReg_deriv += MDL_err_ts_LocReg_deriv

        # avg_MDL_err_LocReg_GCV.append(MDL_err_LocReg_GCV)
        # avg_MDL_err_LocReg_DP += MDL_err_LocReg_DP
        # avg_MDL_err_LocReg_NE += MDL_err_LocReg_NE

    
    avg_MDL_err /= n_sim
    # avg_MDL_err_GCV /= n_sim
    avg_MDL_err_DP /= n_sim
    avg_MDL_err_LC /= n_sim
    avg_MDL_err_LocReg_LC /= n_sim
    # avg_MDL_err_LocReg_GCV /= n_sim
    avg_MDL_err_GCV = np.mean(avg_MDL_err_GCV)
    avg_MDL_err_ss_LocReg_deriv /= n_sim
    avg_MDL_err_ss_LocReg /= n_sim
    avg_MDL_err_ts_LocReg_deriv /= n_sim
    # avg_MDL_err_GCV/=n_sim
    # avg_MDL_err_LocReg_GCV = np.median(avg_MDL_err_LocReg_GCV)
    # avg_MDL_err_LocReg_DP /= n_sim
    # avg_MDL_err_LocReg_NE /= n_sim

    # compare_GCV = avg_MDL_err - avg_MDL_err_GCV
    # compare_DP = avg_MDL_err - avg_MDL_err_DP
    # compare_LC = avg_MDL_err - avg_MDL_err_LC
    # compare_LocReg = avg_MDL_err - avg_MDL_err_LocReg_LC
    compare_GCV = avg_MDL_err_LocReg_LC - avg_MDL_err_GCV
    compare_DP = avg_MDL_err_LocReg_LC - avg_MDL_err_DP
    compare_LC = avg_MDL_err_LocReg_LC - avg_MDL_err_LC
    # compare_LocReg = avg_MDL_err - avg_MDL_err_LocReg_LC
    # compare_LocReg_NE = avg_MDL_err - avg_MDL_err_LocReg_NE

    errors = {
        'avg_MDL_err': avg_MDL_err,
        'avg_MDL_err_DP': avg_MDL_err_DP,
        'avg_MDL_err_LC': avg_MDL_err_LC,
        'avg_MDL_err_LocReg_LC': avg_MDL_err_LocReg_LC,
        'avg_MDL_err_GCV': avg_MDL_err_GCV,
        'avg_MDL_err_ss_LocReg_deriv': avg_MDL_err_ss_LocReg_deriv,
        'avg_MDL_err_ss_LocReg': avg_MDL_err_ss_LocReg,
        'avg_MDL_err_ts_LocReg_deriv': avg_MDL_err_ts_LocReg_deriv
    }

    file_path = f'{cwd_full}/avg_error_vals.pkl'

    # Load the error data from the pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(errors, file)

    errors = {k: np.mean(v) if isinstance(v, np.ndarray) else v for k, v in errors.items()}

    # Sort metrics by error value
    sorted_errors = sorted(errors.items(), key=lambda item: item[1])

    # Assign ranks based on sorted order
    rankings = {metric: rank + 1 for rank, (metric, _) in enumerate(sorted_errors)}

    # Print the rankings
    print("Error Metric Rankings (1 = Lowest Error):")
    for metric, rank in rankings.items():
        print(f"{rank}: {metric} with an error value of {errors[metric]:.4f}")

    sol_strct_uneql = {}
    # sol_strct_uneql = pd.DataFrame(columns = ["avg_MDL_err","avg_MDL_err_GCV","avg_MDL_err_DP", "avg_MDL_err_LC", "var", "bias", "pEst_cvn", "pEst_AIC", "pEst_cf"])
    sol_strct_uneql['avg_MDL_err'] = avg_MDL_err
    sol_strct_uneql['avg_MDL_err_GCV'] = avg_MDL_err_GCV
    sol_strct_uneql['compare_GCV'] = compare_GCV
    sol_strct_uneql['error_GCV'] = error_GCV
    sol_strct_uneql['Gaus_info'] = Gaus_info
    sol_strct_uneql['error_multi_reg'] = error_multi_reg
    sol_strct_uneql['avg_MDL_err_DP'] = avg_MDL_err_DP
    sol_strct_uneql['compare_DP'] = compare_DP
    sol_strct_uneql['error_DP'] = error_DP
    sol_strct_uneql['avg_MDL_err_LC'] = avg_MDL_err_LC
    sol_strct_uneql['compare_LC'] = compare_LC
    sol_strct_uneql['error_LC'] = error_LC
    sol_strct_uneql['avg_MDL_err_LocReg_LC'] = avg_MDL_err_LocReg_LC
    sol_strct_uneql['avg_MDL_err_LocReg_GCV'] = avg_MDL_err_LocReg_GCV
    # sol_strct_uneql['avg_MDL_err_LocReg_DP'] = avg_MDL_err_LocReg_DP
    # sol_strct_uneql['compare_LocReg'] = compare_LocReg
    sol_strct_uneql['avg_MDL_err_ss_LocReg_deriv'] = avg_MDL_err_ss_LocReg_deriv
    sol_strct_uneql['avg_MDL_err_ss_LocReg'] = avg_MDL_err_ss_LocReg
    sol_strct_uneql['avg_MDL_err_ts_LocReg_deriv'] = avg_MDL_err_ts_LocReg_deriv

    sol_strct_uneql['error_ss_LocReg_deriv'] = error_ss_LocReg_deriv
    sol_strct_uneql['error_ss_LocReg'] = error_ss_LocReg
    sol_strct_uneql['error_ts_LocReg_deriv'] = error_ts_LocReg_deriv

    # sol_strct_uneql['error_LocReg_GCV'] = error_LocReg_GCV
    # sol_strct_uneql['error_LocReg_DP'] = error_LocReg_DP
    # sol_strct_uneql['avg_MDL_err_LocReg_NEW_NNLS'] = avg_MDL_err_LocReg_NE
    # sol_strct_uneql['compare_LocReg_NEW_NNLS'] = compare_LocReg_NE
    # sol_strct_uneql['error_LocReg_NEW_NNLS'] = error_LocReg_NE

    # Save history
    sol_strct_uneql['MultiReg_data'] = MultiReg_data
    sol_strct_uneql['GCV_data'] = GCV_data
    sol_strct_uneql['DP_data'] = DP_data
    sol_strct_uneql['LC_data'] = LC_data
    sol_strct_uneql['LocReg_data_LC'] = LocReg_data_LC

    sol_strct_uneql['ss_LocReg_deriv_data'] = ss_LocReg_deriv_data
    sol_strct_uneql['ss_LocReg_data'] = ss_LocReg_data
    sol_strct_uneql['ts_LocReg_deriv_data'] = ts_LocReg_deriv_data

    # sol_strct_uneql['LocReg_data_GCV'] = LocReg_data_GCV
    # sol_strct_uneql['LocReg_data_DP'] = LocReg_data_DP
    # sol_strct_uneql['LocReg_NEW_NNLS_data'] = LocReg_NE_data

    if show == 1:
        sol_strct_uneql['IdealModel_data'] = IdealModel_data
        plt.figure(figsize=(8, 12))
        plt.subplots_adjust(wspace=0.3, hspace=0.8)
        plt.subplot(3, 1, 1)
        plt.imshow(compare_GCV, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
        plt.colorbar()
        plt.xlabel('Ratio of Peak Separation', fontsize=18)
        plt.ylabel('Gaussian Sigma', fontsize=18)
        plt.title('LocReg Error - GCV Error', fontsize=18)
        plt.clim(-0.5, 0.5)

        plt.subplot(3, 1, 2)
        plt.imshow(compare_DP, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
        plt.colorbar()
        plt.xlabel('Ratio of Peak Separation', fontsize=18)
        plt.ylabel('Gaussian Sigma', fontsize=18)
        plt.title('LocReg Error - DP Error', fontsize=18)
        plt.clim(-0.5, 0.5)

        plt.subplot(3, 1, 3)
        plt.imshow(compare_LC, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
        plt.colorbar()
        plt.xlabel('Ratio of Peak Separation', fontsize=18)
        plt.ylabel('Gaussian Sigma', fontsize=18)
        plt.title('LocReg Error - L-Curve Error', fontsize=18)
        plt.clim(-0.5, 0.5)

        # plt.subplot(4, 1, 4)
        # plt.imshow(compare_LocReg, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
        # plt.colorbar()
        # plt.xlabel('Ratio of Peak Separation', fontsize=18)
        # plt.ylabel('Gaussian Sigma', fontsize=18)
        # plt.title('Comparison with LocReg using LC lam', fontsize=18)
        # plt.clim(-0.5, 0.5)
        # plt.savefig('heatmap.png')
        string = "heatmap"
        file_path = create_result_folder(string, SNR)
        plt.savefig(os.path.join(file_path, f"heatmap.png"))
        print(f"Saved Heatmap")
        plt.close()

    if show == 0:
        # Define the filename for saving the data
        # filename = '/Users/steveh/Downloads/NIH 23-24/LocReg_Python/figure_data/sol_struct_uneql2_noshow_11_21_23.pickle'
        filename = f"{cwd_full}/{string}_{date}_SNR_{SNR}_dol_struct_uneql2_noshow.pkl"
        # Save the dictionary to a file using pickle
        errors = {
                'avg_MDL_err': avg_MDL_err,
                'avg_MDL_err_DP': avg_MDL_err_DP,
                'avg_MDL_err_LC': avg_MDL_err_LC,
                'avg_MDL_err_LocReg_LC': avg_MDL_err_LocReg_LC,
                'avg_MDL_err_GCV': avg_MDL_err_GCV,
                'avg_MDL_err_ss_LocReg_deriv': avg_MDL_err_ss_LocReg_deriv,
                'avg_MDL_err_ss_LocReg': avg_MDL_err_ss_LocReg,
                'avg_MDL_err_ts_LocReg_deriv': avg_MDL_err_ts_LocReg_deriv
            }

        file_path = f'{cwd_full}/avg_error_vals.pkl'

        # Load the error data from the pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(errors, file)

        errors = {k: np.mean(v) if isinstance(v, np.ndarray) else v for k, v in errors.items()}

        # Sort metrics by error value
        sorted_errors = sorted(errors.items(), key=lambda item: item[1])

        # Assign ranks based on sorted order
        rankings = {metric: rank + 1 for rank, (metric, _) in enumerate(sorted_errors)}

        # Print the rankings
        print("Error Metric Rankings (1 = Lowest Error):")
        for metric, rank in rankings.items():
            print(f"{rank}: {metric} with an error value of {errors[metric]:.4f}")

        with open(filename, 'wb') as file:
            pickle.dump(sol_strct_uneql, file)
        print(f"Data saved to {filename}")
    
    return sol_strct_uneql