# import sys
# print("Setting system path")
# sys.path.append(".")  # Replace this path with the actual path to the parent directory of Utilities_functions
# import numpy as np
# from scipy.stats import norm as normsci
# from scipy.linalg import norm as linalg_norm
# from scipy.optimize import nnls
# import matplotlib.pyplot as plt
# from scipy.stats import wasserstein_distance, entropy
# import pickle
# from tqdm import tqdm
# # from Utilities_functions.LocReg import LocReg
# # from Utilities_functions.LocReg_unconstrainedB import LocReg_unconstrainedB
# from regularization.reg_methods.dp.discrep_L2 import discrep_L2
# from regularization.reg_methods.gcv.GCV_NNLS import GCV_NNLS
# from regularization.reg_methods.lcurve.Lcurve import Lcurve
# from regularization.reg_methods.spanreg.Multi_Reg_Gaussian_Sum1 import Multi_Reg_Gaussian_Sum1
# from regularization.reg_methods.dp.discrep import discrep
# from regularization.reg_methods.locreg.Ito_LocReg import *
# from regularization.reg_methods.locreg.LocReg import LocReg as Chuan_LR
# from regularization.reg_methods.lcurve import l_curve
# from regularization.reg_methods.nnls.tikhonov_vec import tikhonov_vec
# from regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm

# # from Utilities_functions.LocReg_NEW_NNLS import LocReg_NEW_NNLS
# import os
# from datetime import datetime
# import pandas as pd
# # sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
# import cvxpy as cp
# import os
# import scipy
# from scipy.linalg import svd
# # from lsqnonneg import lsqnonneg
# from regularization.subfunc.lcurve_functions import l_cuve,csvd,l_corner
# from regularization.subfunc.l_curve_corner import l_curve_corner
# from regularization.subfunc.csvd import csvd
# # from Simulations.Ito_LocReg import Ito_LocReg
# # from Simulations.Ito_LocReg import blur_ito, grav_ito
# # from regu.ito_blur import blur
# # from regu.ito_gravity import gravity
# from tools.trips_py.pasha_gcv import Tikhonov
# # from regu.tikhonov import tikhonov
# from datetime import datetime
# from tqdm import tqdm
# import sys
# import os
# import mosek
# # from ItoLocRegConst import LocReg_Ito_C,LocReg_Ito_C_2,LocReg_Ito_C_4
from utils.load_imports.loading import *

# print("setting license path")
mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path

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

lam_ini_val = "LCurve"
normset = True
# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + output_folder 

preset_noise = True

if preset_noise == True:
    noisearr = np.load("/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_13Aug24noise_arr.txt.npy")
else:
    pass
def create_result_folder(string, SNR,lam_ini_val):
    # Create a folder based on the current date and time
    #folder_name = f"c:/Users/kimjosy/LocReg_Regularization/{string}_{date}_SNR_{SNR}"
    folder_name = f"{cwd_full}/{string}_{date}_SNR_{SNR}_lamini_{lam_ini_val}_original"
    # folder_name = f"/Volumes/Lexar/NIH/Experiments/GridSearch/{string}_{date}_Run"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# def minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g):
#     OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
#     OP_rhos = np.zeros((len(Alpha_vec)))
#     for j in (range(len(Alpha_vec))):
#         X, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j]**2, '0', nargin = 4)
#         # A  = G.T @ G + Alpha_vec[j]**2 * L.T @ L 
#         # b =  G.T @ data_noisy
#         # val = Alpha_vec[j]**2 * np.ones(G.shape[1])
#         # eps = 1e-2
#         # A = (G.T @ G + np.diag(val))
#         # ep4 = np.ones(G.shape[1]) * eps
#         # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * (val)
#         # # exp = np.linalg.inv(A) @ b
#         # # exp = iterative_algorithm(G, data_noisy, B_1, B_2, Alpha_vec[j], Alpha_vec2[j], max_iter=1000, tol=1e-6)
#         # try:
#         #     # Solve the problem using nnls
#         #     sol = nnls(A, b, maxiter=1000)[0]
#         #     sol = sol - eps
#         #     # Ensure non-negative solution
#         #     sol = np.array(sol)
#         #     sol[sol < 0] = 0
            
#         # except Exception as e:
#         #     # Handle exceptions if nnls fails
#         #     y = cp.Variable(G.shape[1])
#         #     cost = cp.norm(A @ y - b, 2)**2
#         #     constraints = [y >= 0]
#         #     problem = cp.Problem(cp.Minimize(cost), constraints)
#         #     try:
#         #         problem.solve(solver=cp.MOSEK, verbose=False)
#         #         #you can try a different solver if you don't have the license for MOSEK doesn't work (should be free for one year)
#         #     except Exception as e:
#         #         print(e)
#         #     # reconst = y.value
#         #     # reconst,_ = nnls(A,b, maxiter = 10000)
#         #     sol = y.value
#         #     # print(f"An error occurred during nnls optimization, using MOSEK instead: {e}")
#         #     sol = sol - eps
#         #     # Ensure non-negative solution
#         #     sol = np.array(sol)
#         #     sol[sol < 0] = 0
#         # print("exp shape", exp.shape)
#         # print("g.shape", g.shape)
#         # sol = np.linalg.lstsq(A,b, rcond=None)[0]
#         # exp = np.linalg.solve(A,b)
#         # exp = fe.fnnls(A, b) 
#         # print(np.linalg.cond(A))
#         # exp = nnls(A, b, maxiter = 1e30)[0]
#         # x = cp.Variable(nT2)
#         # cost = cp.sum_squares(A @ x - b)
#         # cost = cp.norm(A @ x - b, 2)**2
#         # constraints = [x >= 0]
#         # problem = cp.Problem(cp.Minimize(cost), constraints)
#         # problem.solve(solver=cp.MOSEK, verbose=False)
#         # exp,_ = fnnls(A,b)
#         OP_x_lc_vec[:, j] = X
#         # print("L_x_lc_vec[:,j,k]", L_x_lc_vec[:,j,k])
#         OP_rhos[j] = np.linalg.norm(OP_x_lc_vec[:,j] - g, 2)**2
    
#     OP_log_err_norm = np.log10(OP_rhos)
#     min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)

#     min_x = Alpha_vec[min_index[0]]
#     min_z = np.min(OP_log_err_norm)

#     OP_min_alpha1 = min_x
#     OP_min_alpha1_ind = min_index[0]
#     f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
#     return f_rec_OP_grid, OP_min_alpha1

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
    return f_rec_OP_grid, OP_min_alpha1

def heatmap_unequal_width_All(Gaus_info, show, n_sim):
    # print("Gaus_info keys:", Gaus_info.keys())
    # print("Gaus_info type:", type(Gaus_info))
    T2 = Gaus_info['T2'].flatten()
    TE = Gaus_info['TE'].flatten()
    A = Gaus_info['A']
    # print("A.shape",A.shape)
    npeaks = 2
    f_coef = np.ones(npeaks)

    #Get T2mu
    rps = np.linspace(1, 4, 5).T
    mps = rps / 2
    nrps = len(rps)
    T2_left = 30 * np.ones(nrps)
    T2_mid = T2_left * mps
    T2_right = T2_left * rps
    T2mu = np.column_stack((T2_left,  T2_right))

    #Get diff_sigma
    nsigma = 5
    unif_sigma = np.linspace(2, 5, nsigma).T
    diff_sigma = np.column_stack((unif_sigma, 3 * unif_sigma))

    LGBs = Gaus_info['LGBs']
    Lambda = Gaus_info['Lambda'].reshape(-1,1)
    Lambda_1 = np.logspace(np.log10(Lambda[0]), np.log10(Lambda[-1]), len(Lambda))
    nLambda_1 = len(Lambda_1)
    nLambda = len(Lambda)
    nGaus = Gaus_info['LGBs'].shape[1]
    n, m = Gaus_info['A'].shape

    print(Lambda[0])
    print(Lambda[0])
    print(Lambda[0])

    # Create arrays to store intermediate results
    avg_MDL_err = np.zeros((nsigma, nrps))
    # avg_MDL_err_GCV = np.zeros((nsigma, nrps))
    avg_MDL_err_GCV = np.zeros((nsigma, nrps))
    avg_MDL_err_DP = np.zeros((nsigma, nrps))
    avg_MDL_err_LC = np.zeros((nsigma, nrps))
    avg_MDL_err_LocReg_LC = np.zeros((nsigma, nrps))
    avg_MDL_err_oracle =  np.zeros((nsigma, nrps))
    avg_MDL_err_ss_LocReg_deriv = np.zeros((nsigma, nrps))
    avg_MDL_err_ss_LocReg = np.zeros((nsigma, nrps))
    avg_MDL_err_ts_LocReg_deriv = np.zeros((nsigma, nrps))
    # avg_MDL_err_LocReg_DP = np.zeros((nsigma, nrps))
    # avg_MDL_err_LocReg_GCV = np.zeros((nsigma, nrps))
    avg_MDL_err_LocReg_GCV = []
    avg_MDL_err_LocReg_NE = np.zeros((nsigma, nrps))

    error_multi_reg = np.zeros((nsigma, nrps, n_sim))
    error_GCV = np.zeros((nsigma, nrps, n_sim))
    error_DP = np.zeros((nsigma, nrps, n_sim))
    error_LC = np.zeros((nsigma, nrps, n_sim))
    error_LocReg_LC = np.zeros((nsigma, nrps, n_sim))
    error_oracle =  np.zeros((nsigma, nrps, n_sim))
    # error_LocReg_GCV = np.zeros((nsigma, nrps, n_sim))
    # error_LocReg_DP = np.zeros((nsigma, nrps, n_sim))
    error_ss_LocReg_deriv = np.zeros((nsigma, nrps, n_sim))
    error_ss_LocReg = np.zeros((nsigma, nrps, n_sim))
    error_ts_LocReg_deriv = np.zeros((nsigma, nrps, n_sim))

    # error_LocReg_NE = np.zeros((nsigma, nrps, n_sim))

    # error_multi_reg_wass = np.zeros((nsigma, nrps, n_sim))
    # error_GCV_wass = np.zeros((nsigma, nrps, n_sim))
    # error_DP_wass = np.zeros((nsigma, nrps, n_sim))
    # error_LC_wass = np.zeros((nsigma, nrps, n_sim))
    # error_LocReg_wass = np.zeros((nsigma, nrps, n_sim))

    # error_multi_reg_KL = np.zeros((nsigma, nrps, n_sim))
    # error_GCV_KL = np.zeros((nsigma, nrps, n_sim))
    # error_DP_KL = np.zeros((nsigma, nrps, n_sim))
    # error_LC_KL = np.zeros((nsigma, nrps, n_sim))
    # error_LocReg_KL = np.zeros((nsigma, nrps, n_sim))

    SNR = Gaus_info['SNR']

    # Create dataset arrays to save the results
    # feature_df = pd.DataFrame(columns = ["TI1*g","TI2*g","TI_DATA", "MSE", "var", "bias", "pEst_cvn", "pEst_AIC", "pEst_cf"])

    MultiReg_data = np.zeros((n_sim, nsigma, nrps, m))
    GCV_data = np.zeros((n_sim, nsigma, nrps, m))
    DP_data = np.zeros((n_sim, nsigma, nrps, m))
    LC_data = np.zeros((n_sim, nsigma, nrps, m))
    # LocReg_data_LC = np.zeros((n_sim, nsigma, nrps, m))
    # LocReg_data_GCV = np.zeros((n_sim, nsigma, nrps, m))
    # LocReg_data_DP = np.zeros((n_sim, nsigma, nrps, m))
    LocReg_data_LC = np.zeros((n_sim, nsigma, nrps, m))
    # LocReg_data_GCV = np.zeros((n_sim, nsigma, nrps, m))
    # LocReg_data_DP = np.zeros((n_sim, nsigma, nrps, m))
    IdealModel_data = np.zeros((n_sim, nsigma, nrps, m))
    # LocReg_NE_data = np.zeros((n_sim, nsigma, nrps, m))
    # LocReg_data_LC = np.zeros((n_sim, nsigma, nrps, m))
    LocReg_data_LC = np.zeros((n_sim, nsigma, nrps, m))
    oracle_data = np.zeros((n_sim, nsigma, nrps, m))

    noisy_data = np.zeros((n_sim, nsigma, nrps, n))
    noiseless_data = np.zeros((n_sim, nsigma, nrps, n))

    ss_LocReg_deriv_data = np.zeros((n_sim, nsigma, nrps, m))
    ss_LocReg_data = np.zeros((n_sim, nsigma, nrps, m))
    ts_LocReg_deriv_data = np.zeros((n_sim, nsigma, nrps, m))
    # gammas_list = np.linspace(0,10,2)
    # kl_scores_list = []
    # l2_rmsscores_list = []
    # wass_scores_list = []
    # gammas_list = np.linspace(-4,4,5)

    # print("This is simulation " + f"{l}")
    MDL_err = np.zeros((nsigma, nrps))
    MDL_err_GCV = np.zeros((nsigma, nrps))
    MDL_err_DP = np.zeros((nsigma, nrps))
    MDL_err_LC = np.zeros((nsigma, nrps))
    # MDL_err_LocReg_DP = np.zeros((nsigma, nrps))
    MDL_err_LocReg_LC = np.zeros((nsigma, nrps))
    # MDL_err_LocReg_GCV = np.zeros((nsigma, nrps))
    # MDL_err_LocReg_NE = np.zeros((nsigma, nrps))
    MDL_err_ss_LocReg_deriv = np.zeros((nsigma, nrps))
    MDL_err_ss_LocReg = np.zeros((nsigma, nrps))
    MDL_err_ts_LocReg_deriv = np.zeros((nsigma, nrps))
    
    MDL_err_oracle= np.zeros((nsigma, nrps))


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
    

    L = np.eye(A.shape[1])
    for l in tqdm(range(n_sim)):
        for i in range(nsigma):
            sigma_i = diff_sigma[i, :]
            if show == 0:
                for j in range(len(rps)):
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
                    
                    Alpha_vec = Lambda
                    # Alpha_vec = Lambda
                    f_rec_oracle, lambda_oracle = minimize_OP(Alpha_vec, L, dat_noisy, A, A.shape[1], IdealModel_weighted)
                    # # Multi_Reg_Gaussian_Sum1
                    # f_rec, alpha_L2, F_info, C_L2 = Multi_Reg_Gaussian_Sum1(dat_noisy, Gaus_info)
                    # LocReg
                    #maxiter = 400
                    ep1 = 1e-2; # 1/(|x|+ep1)
                    ep2 = 1e-2; # norm(dx)/norm(x)
                    ep3 = 1e-2; # norm(x_(k-1) - x_k)/norm(x_(k-1))
                    ep4 = 1e-4; # lb for ep1
                    # maxiter = 400
                    LRIto_ini_lam = lambda_LC
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
                        #normalization
                    # f_rec_LocReg_LC, lambda_locreg_LC = LocReg(dat_noisy, A, f_rec_LC, maxiter)
                    # f_rec_LocReg_DP, lambda_locreg_DP = LocReg(dat_noisy, A, f_rec_DP, maxiter)
                    # f_rec_LocReg_GCV, lambda_locreg_GCV = LocReg(dat_noisy, A, f_rec_GCV.ravel(), maxiter)
                    # # Deriv
                    # LR_mod_rec1, fin_lam1 = LocReg_Ito_C(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                    # LR_Ito_lams = fin_lam1

                    # # No Deriv
                    # LR_mod_rec2, fin_lam2 = LocReg_Ito_C_2(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                    # LR_Ito_lams2 = fin_lam2

                    # #Gamma Deriv
                    LR_mod_rec4, fin_lam4, gamma_new2 = LocReg_Ito_C_4(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                    LR_Ito_lams4 = fin_lam4

                    # f_rec_LocReg_NE, lambda_locreg_NE = LocReg_NEW_NNLS(dat_noisy, A, f_rec_LC, maxiter)
                    # f_rec_LocReg, lambda_locreg = LocReg_unconstrainedB(dat_noisy, A, f_rec_GCV,lambda_GCV, ep1,ep2,ep3,ep4)
                    # Save to dataset
                    # MultiReg_data[l, i, j, :] = f_rec
                    GCV_data[l, i, j, :] = f_rec_GCV.ravel()
                    DP_data[l, i, j, :] = f_rec_DP
                    LC_data[l, i, j, :] = f_rec_LC
                    LocReg_data_LC[l, i, j, :] = f_rec_LocReg_LC
                    oracle_data[l, i, j, :] = f_rec_oracle

                    # ss_LocReg_deriv_data[l, i, j, :] = LR_mod_rec1
                    # ss_LocReg_data[l, i, j, :] = LR_mod_rec2
                    ts_LocReg_deriv_data[l, i, j, :] =LR_mod_rec4
                    # LocReg_data_GCV[l, i, j, :] = f_rec_LocReg_GCV
                    # LocReg_data_DP[l, i, j, :] = f_rec_LocReg_DP

                    # LocReg_NE_data[l, i, j, :] = f_rec_LocReg_NE
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
                    MDL_err_oracle[i, j] = linalg_norm(IdealModel_weighted - f_rec_oracle) / true_norm

                    # MDL_err_ss_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm
                    # MDL_err_ss_LocReg[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm
                    MDL_err_ts_LocReg_deriv[i, j] = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm
                    # # MDL_err_LocReg_DP[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_DP) / true_norm
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
                    error_oracle[i, j, l] = MDL_err_oracle[i, j]

                    # error_ss_LocReg_deriv[i, j, l] = MDL_err_ss_LocReg_deriv[i, j]
                    # error_ss_LocReg[i, j, l] = MDL_err_ss_LocReg[i, j]
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
                    if preset_noise == False:
                        noise = np.column_stack([np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE), 1)])
                        dat_noisy = dat_noiseless + np.ravel(noise) 
                    else:
                        noise = noisearr[l,i,j,:]
                        dat_noisy = dat_noiseless + np.ravel(noise)
                        noiseless_data[l, i, j, :] = dat_noiseless
                        noisy_data[l, i, j, :] = dat_noisy

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
                    
                    if lam_ini_val == "LCurve" or lam_ini_val == "L-Curve":
                        LRIto_ini_lam = lambda_LC
                    elif lam_ini_val == "GCV" or lam_ini_val == "gcv":
                        LRIto_ini_lam = lambda_GCV
                    elif lam_ini_val == "DP" or lam_ini_val == "dp":
                        LRIto_ini_lam = lambda_DP
                            # gamma_init = 10
                    gamma_init = 5
                    maxiter = 75
                    # normset = True
                    Alpha_vec = Lambda
                    # print("dat_noiseless",dat_noiseless.shape)
                    f_rec_oracle, lambda_oracle = minimize_OP(Alpha_vec, L, dat_noisy, A, len(T2), IdealModel_weighted)

                    print(f"finished all except locreg")

                    f_rec_LocReg_LC, lambda_locreg_LC = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
                    # f_rec_LocReg_DP, lambda_locreg_DP = LocReg_Ito_mod(dat_noisy, A, lambda_DP, gamma_init, maxiter)
                    # f_rec_LocReg_GCV, lambda_locreg_GCV = LocReg_Ito_mod(dat_noisy, A, lambda_GCV, gamma_init, maxiter)
                    print(f"finished locreg for {i} and {j}")

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
                    # normset = False
                    # f_rec_LocReg_LCnn, lambda_locreg_LCnn = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)

                    # print(dat_noiseless.shape)
                    # print(L.shape)
                    # print(Lambda.shape)

                    # Alpha_vec = Lambda
                    # LR_mod_rec1, fin_lam1 = LocReg_Ito_C(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                    # LR_Ito_lams = fin_lam1

                    # #No Deriv
                    # LR_mod_rec2, fin_lam2 = LocReg_Ito_C_2(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                    # LR_Ito_lams2 = fin_lam2

                    #Gamma Deriv
                    # LR_mod_rec4, fin_lam4, gamma_new2 = LocReg_Ito_C_4(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
                    # LR_Ito_lams4 = fin_lam4

                    # print("f_rec_GCV:",f_rec_GCV.shape)
                    # print("f_rec_LocReg_NE:",f_rec_LocReg_NE)

                    # Save to dataset
                    MultiReg_data[l, i, j, :] = f_rec.flatten()
                    GCV_data[l, i, j, :] = f_rec_GCV.flatten()
                    DP_data[l, i, j, :] = f_rec_DP.flatten()
                    LC_data[l, i, j, :] = f_rec_LC.flatten()
                    # LocReg_data[l, i, j, :] = f_rec_LocReg.flatten()
                    LocReg_data_LC[l, i, j, :] = f_rec_LocReg_LC
                    oracle_data[l, i, j, :] = f_rec_oracle

                    # ss_LocReg_deriv_data[l, i, j, :] = LR_mod_rec1
                    # ss_LocReg_data[l, i, j, :] = LR_mod_rec2
                    # ts_LocReg_deriv_data[l, i, j, :] =LR_mod_rec4

                    # LocReg_data_GCV[l, i, j, :] = f_rec_LocReg_GCV
                    # LocReg_data_DP[l, i, j, :] = f_rec_LocReg_DP
                    # LocReg_NE_data[l, i, j, :] = f_rec_LocReg_NE.flatten()

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
                    # locregdperr =  linalg_norm(IdealModel_weighted - f_rec_LocReg_DP) / true_norm
                    # locreggcverr =  linalg_norm(IdealModel_weighted - f_rec_LocReg_GCV) / true_norm

                    MDL_err_oracle[i, j] = linalg_norm(IdealModel_weighted - f_rec_oracle) / true_norm
                    oracleerr = linalg_norm(IdealModel_weighted - f_rec_oracle) / true_norm

                    if j == 0 and i == 0:
                        true_norm = linalg_norm(IdealModel_weighted)
                        print("true_norm", linalg_norm(IdealModel_weighted))
                        print("locreg error", linalg_norm(IdealModel_weighted - f_rec_LocReg_LC)/true_norm)
                        print("gcv error", linalg_norm(IdealModel_weighted - f_rec_GCV) / true_norm)
                        print("oracle error",linalg_norm(IdealModel_weighted - f_rec_oracle) / true_norm )

                    # locregnnerr = linalg_norm(IdealModel_weighted - f_rec_LocReg_LCnn) / true_norm
                    # MDL_err_ss_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm
                    # ss_LocReg_deriv_err = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm

                    # MDL_err_ss_LocReg[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm
                    # ss_LocReg_err = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm

                    # MDL_err_ts_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm
                    # ts_LocReg_deriv_err = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm


                    # MDL_err_LocReg_DP[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_DP) / true_norm
                    # MDL_err_LocReg_GCV[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_GCV) / true_norm
                    # MDL_err_LocReg_NE[i, j] = linalg_norm(IdealModel_weighted - f_rec_LocReg_NE) / true_norm
                    # MDL_err_ss_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm
                    # ss_LocReg_deriv_err = linalg_norm(IdealModel_weighted - LR_mod_rec1) / true_norm

                    # MDL_err_ss_LocReg[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm
                    # ss_LocReg_err = linalg_norm(IdealModel_weighted - LR_mod_rec2) / true_norm

                    # MDL_err_ts_LocReg_deriv[i,j] = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm
                    # ts_LocReg_deriv_err = linalg_norm(IdealModel_weighted - LR_mod_rec4) / true_norm

                    error_multi_reg[i, j, l] = MDL_err[i, j]
                    error_GCV[i, j, l] = MDL_err_GCV[i, j]
                    error_DP[i, j, l] = MDL_err_DP[i, j]
                    error_LC[i, j, l] = MDL_err_LC[i, j]
                    # error_LocReg[i, j, l] = MDL_err_LocReg[i, j]
                    error_LocReg_LC[i, j, l] = MDL_err_LocReg_LC[i, j]
                    error_oracle[i, j, l] = MDL_err_oracle[i, j]

                    error_ss_LocReg_deriv[i, j, l] = MDL_err_ss_LocReg_deriv[i, j]
                    error_ss_LocReg[i, j, l] = MDL_err_ss_LocReg[i, j]
                    error_ts_LocReg_deriv[i, j, l] = MDL_err_ts_LocReg_deriv[i, j]
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
                    plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} (Error: {round(locregerr,3)})')
                    # plt.plot(T2, f_rec_LocReg_LCnn, linestyle=':', linewidth=3, color='red', label=f'LocReg No Norm {lam_ini_val} (Error: {round(locregnnerr,3)})')
                    # plt.plot(T2, f_rec_LocReg_DP, linestyle=':', linewidth=3, color='orange', label=f'LocReg DP (Error: {round(locregdperr,3)})')
                    # plt.plot(T2, f_rec_LocReg_GCV, linestyle=':', linewidth=3, color='magenta', label=f'LocReg GCV (Error: {round(locreggcverr,3)})')

                    # plt.plot(T2, LR_mod_rec1, linestyle='-', linewidth=3, color='green', label=f'ss_LocReg_deriv (Error: {round(ss_LocReg_deriv_err,3)})')
                    # plt.plot(T2, LR_mod_rec2, linestyle='-.', linewidth=3, color='magenta', label=f'ss_LocReg (Error: {round(ss_LocReg_err,3)})')
                    # plt.plot(T2, LR_mod_rec4, linestyle='-.', linewidth=3, color='orange', label=f'LocReg Deriv (Error: {round(ts_LocReg_deriv_err,3)})')
                    plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle (Error: {round(oracleerr,3)})')

                    plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {round(dperr,3)})')
                    plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {round(gcverr,3)})')
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
                    plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')
                    # plt.plot(TE, A @ f_rec_LocReg_DP, linestyle=':', linewidth=3, color='orange', label='LocReg DP')
                    # plt.plot(TE, A @ f_rec_LocReg_GCV, linestyle=':', linewidth=3, color='magenta', label='LocReg GCV')

                    plt.plot(TE, A @ f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle')
                    # plt.plot(TE, A @ LR_mod_rec1, linestyle='-', linewidth=3, color='green', label='ss_LocReg_deriv')
                    # plt.plot(TE, A @ LR_mod_rec2, linestyle='-.', linewidth=3, color='magenta', label='ss_LocReg')
                    # plt.plot(TE, A @ LR_mod_rec4, linestyle=':', linewidth=3, color='orange', label='LocReg Deriv')

                    plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
                    plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
                    plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
                    # plt.plot(TE, A @ f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
                    plt.legend(fontsize=10, loc='best')
                    plt.xlabel('TE', fontsize=20, fontweight='bold')
                    plt.ylabel('Intensity', fontsize=20, fontweight='bold')
                    
                    plt.subplot(1, 3, 3)
                    plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
                    plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
                    plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
                    plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')
                    # plt.semilogy(T2, lambda_locreg_DP * np.ones(len(T2)), linestyle=':', linewidth=3, color='orange', label='LocReg DP')
                    # plt.semilogy(T2, lambda_locreg_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='magenta', label='LocReg GCV')

                    # plt.semilogy(T2, LR_Ito_lams * np.ones(len(T2)), linestyle='-', linewidth=3, color='green', label='ss_LocReg_deriv')
                    # plt.semilogy(T2, LR_Ito_lams2 * np.ones(len(T2)), linestyle='-.', linewidth=3, color='magenta', label='ss_LocReg')
                    # plt.semilogy(T2, LR_Ito_lams4 * np.ones(len(T2)), linestyle='-.', linewidth=3, color='orange', label='LocReg Deriv')
                    plt.semilogy(T2, lambda_oracle * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')

                    plt.legend(fontsize=10, loc='best')
                    plt.xlabel('T2', fontsize=20, fontweight='bold')
                    plt.ylabel('Lambda', fontsize=20, fontweight='bold')

                    plt.tight_layout()
                    string = "MRR_1D_LocReg_Comparison_origcodetest"
                    file_path = create_result_folder(string, SNR, lam_ini_val)
                    plt.savefig(os.path.join(file_path, f"Comparison_figure_{i}_{j}.png"))
                    print(f"Saved Comparison Plot for {i}_{j}")
                    # plt.savefig(f'figure_{i}_{j}.png')
                    plt.close()  
                    if j == 1 and i == 1:
                        print("MDL_err_LocReg_LC", MDL_err_LocReg_LC)
                        print("MDL_err_oracle", MDL_err_oracle)

                    # plt.draw()
                    # plt.show()

            avg_MDL_err += MDL_err
            avg_MDL_err_GCV += MDL_err_GCV
            avg_MDL_err_DP += MDL_err_DP
            avg_MDL_err_LC += MDL_err_LC
            avg_MDL_err_LocReg_LC += MDL_err_LocReg_LC
            avg_MDL_err_ss_LocReg_deriv += MDL_err_ss_LocReg_deriv
            avg_MDL_err_ss_LocReg += MDL_err_ss_LocReg
            avg_MDL_err_ts_LocReg_deriv += MDL_err_ts_LocReg_deriv
            avg_MDL_err_oracle += MDL_err_oracle

            # avg_MDL_err_LocReg_GCV.append(MDL_err_LocReg_GCV)
            # avg_MDL_err_LocReg_DP += MDL_err_LocReg_DP
            # avg_MDL_err_LocReg_NE += MDL_err_LocReg_NE

    print("MDL_err_LocReg_LC after 25reconstructions", avg_MDL_err_LocReg_LC)
    print("MDL_err_oracle after 25 reconstructions", avg_MDL_err_oracle)

    avg_MDL_err /= n_sim
    # avg_MDL_err_GCV /= n_sim
    avg_MDL_err_DP /= n_sim
    avg_MDL_err_LC /= n_sim
    avg_MDL_err_LocReg_LC /= n_sim
    # avg_MDL_err_LocReg_GCV /= n_sim
    avg_MDL_err_GCV/= n_sim
    avg_MDL_err_ss_LocReg_deriv /= n_sim
    avg_MDL_err_ss_LocReg /= n_sim
    avg_MDL_err_ts_LocReg_deriv /= n_sim
    avg_MDL_err_oracle /= n_sim

    print("avg_MDL_err", avg_MDL_err)
    print("avg_MDL_err_GCV", avg_MDL_err_GCV)
    print("avg_MDL_err_LC", avg_MDL_err_LC)
    print("avg_MDL_err_DP", avg_MDL_err_DP)
    print("avg_MDL_err_LocReg_LC", MDL_err_LocReg_LC)

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
        'avg_MDL_err_oracle': avg_MDL_err_oracle
        # 'avg_MDL_err_ss_LocReg_deriv': avg_MDL_err_ss_LocReg_deriv,
        # 'avg_MDL_err_ss_LocReg': avg_MDL_err_ss_LocReg,
        # 'avg_MDL_err_ts_LocReg_deriv': avg_MDL_err_ts_LocReg_deriv
    }

    file_path = f'{cwd_full}/avg_error_vals_{date}_SNR_{SNR}_lamini_{lam_ini_val}_show_{show}.pkl'

    # Load the error data from the pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(errors, file)

    print("avg_MDL_err_LocReg_LC", np.mean(avg_MDL_err_LocReg_LC.flatten()))
    # errors = {k: np.mean(v) if isinstance(v, np.ndarray) else v for k, v in errors.items()}

    # # Sort metrics by error value
    # sorted_errors = sorted(errors.items(), key=lambda item: item[1])

    # # Assign ranks based on sorted order
    # rankings = {metric: rank + 1 for rank, (metric, _) in enumerate(sorted_errors)}

    # # Print the rankings
    # print("Error Metric Rankings (1 = Lowest Error):")
    # for metric, rank in rankings.items():
    #     print(f"{rank}: {metric} with an error value of {errors[metric]:.4f}")

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
    sol_strct_uneql['avg_MDL_err_oracle'] = avg_MDL_err_oracle

    # sol_strct_uneql['avg_MDL_err_LocReg_DP'] = avg_MDL_err_LocReg_DP
    # sol_strct_uneql['compare_LocReg'] = compare_LocReg
    # sol_strct_uneql['avg_MDL_err_ss_LocReg_deriv'] = avg_MDL_err_ss_LocReg_deriv
    # sol_strct_uneql['avg_MDL_err_ss_LocReg'] = avg_MDL_err_ss_LocReg
    # sol_strct_uneql['avg_MDL_err_ts_LocReg_deriv'] = avg_MDL_err_ts_LocReg_deriv

    # sol_strct_uneql['error_ss_LocReg_deriv'] = error_ss_LocReg_deriv
    # sol_strct_uneql['error_ss_LocReg'] = error_ss_LocReg
    # sol_strct_uneql['error_ts_LocReg_deriv'] = error_ts_LocReg_deriv

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
    sol_strct_uneql['oracle_data'] = oracle_data
    sol_strct_uneql['ss_LocReg_deriv_data'] = ss_LocReg_deriv_data
    sol_strct_uneql['ss_LocReg_data'] = ss_LocReg_data
    sol_strct_uneql['ts_LocReg_deriv_data'] = ts_LocReg_deriv_data
    sol_strct_uneql['noiseless_data'] = noiseless_data
    sol_strct_uneql['noisy_data'] = noisy_data  
    # sol_strct_uneql['LocReg_data_GCV'] = LocReg_data_GCV
    # sol_strct_uneql['LocReg_data_DP'] = LocReg_data_DP
    # sol_strct_uneql['LocReg_NEW_NNLS_data'] = LocReg_NE_data

    solfilename = f"{cwd_full}/{string}_{date}_SNR_{SNR}_sol_struct_uneql2_noshow.pkl"
    with open(solfilename, 'wb') as file:
        pickle.dump(sol_strct_uneql, file)

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
        file_path = create_result_folder(string, SNR,lam_ini_val)
        plt.savefig(os.path.join(file_path, f"heatmap.png"))
        print(f"Saved Heatmap")
        plt.close()

    if show == 0:
        # Define the filename for saving the data
        # filename = '/Users/steveh/Downloads/NIH 23-24/LocReg_Python/figure_data/sol_struct_uneql2_noshow_11_21_23.pickle'
        filename = f"{cwd_full}/{string}_{date}_SNR_{SNR}_sol_struct_uneql2_noshow.pkl"
        # Save the dictionary to a file using pickle
        errors = {
                'avg_MDL_err': avg_MDL_err,
                'avg_MDL_err_DP': avg_MDL_err_DP,
                'avg_MDL_err_LC': avg_MDL_err_LC,
                'avg_MDL_err_LocReg_LC': avg_MDL_err_LocReg_LC,
                'avg_MDL_err_GCV': avg_MDL_err_GCV,
                'avg_MDL_err_oracle': avg_MDL_err_oracle
                # 'avg_MDL_err_ss_LocReg_deriv': avg_MDL_err_ss_LocReg_deriv,
                # 'avg_MDL_err_ss_LocReg': avg_MDL_err_ss_LocReg,
                # 'avg_MDL_err_ts_LocReg_deriv': 
                
            }

        file_path = f'{cwd_full}/avg_error_vals_{date}_SNR_{SNR}_lam_ini_val_{lam_ini_val}_show_{show}.pkl'

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