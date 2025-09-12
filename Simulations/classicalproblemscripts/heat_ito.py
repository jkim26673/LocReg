import sys
import os
# sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
import numpy as np
import matplotlib.pyplot as plt
from regu.Lcurve import Lcurve
from regu.baart import baart
from regu.csvd import csvd
from regu.l_curve import l_curve
from regu.tikhonov import tikhonov
from regu.gcv import gcv
from regu.discrep import discrep
from numpy.linalg import norm
from Utilities_functions.LocReg_unconstrained import LocReg_unconstrained
from regu.heat import heat
from concurrent.futures import ThreadPoolExecutor
from functools import partial
# from Utilities_functions.LocReg_v2 import LocReg_v2
from tqdm import tqdm
from Utilities_functions.LocReg_NE import LocReg_unconstrained_NE
from Utilities_functions.pasha_gcv import Tikhonov
from Ito_LocReg import *
from tqdm import tqdm
from datetime import datetime
from Utilities_functions.tikhonov_vec import tikhonov_vec
from regu.baart import baart

# import scipy.io
# noise = scipy.io.loadmat('/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Matlab NRs/heat_NR.mat')
# #/Users/steveh/Downloads/noisearr1000for10diffnoisereal3.mat
# noise = noise['noise_arr']
import sys
import os
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
cwd = os.getcwd()

cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

pat_tag = "classical_prob"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
series_tag = "heat"
output_folder = f"SimulationSets/{pat_tag}/{series_tag}"

# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + output_folder 

def create_result_folder(string, SNR):
    # Create a folder based on the current date and time
    date = datetime.now().strftime("%Y%m%d")
    #folder_name = f"c:/Users/kimjosy/LocReg_Regularization/{string}_{date}_SNR_{SNR}"
    folder_name = f"{cwd_full}/{string}_{date}_SNR_{SNR}"
    # folder_name = f"/Volumes/Lexar/NIH/Experiments/GridSearch/{string}_{date}_Run"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g):
    OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
    OP_rhos = np.zeros((len(Alpha_vec)))
    for j in (range(len(Alpha_vec))):
        A  = G.T @ G + Alpha_vec[j]**2 * L.T @ L 
        b =  G.T @ data_noisy
        # exp = np.linalg.inv(A) @ b
        # exp = iterative_algorithm(G, data_noisy, B_1, B_2, Alpha_vec[j], Alpha_vec2[j], max_iter=1000, tol=1e-6)
        # exp, _ = nnls(A, b, maxiter = 400)
        # sol = np.linalg.lstsq(A,b, rcond=None)[0]
        exp = np.linalg.solve(A,b)
        # exp = fe.fnnls(A, b) 
        # print(np.linalg.cond(A))
        # exp = nnls(A, b, maxiter = 1e30)[0]
        # x = cp.Variable(nT2)
        # cost = cp.sum_squares(A @ x - b)
        # cost = cp.norm(A @ x - b, 2)**2
        # constraints = [x >= 0]
        # problem = cp.Problem(cp.Minimize(cost), constraints)
        # problem.solve(solver=cp.MOSEK, verbose=False)
        # exp,_ = fnnls(A,b)
        OP_x_lc_vec[:, j] = exp
        # print("L_x_lc_vec[:,j,k]", L_x_lc_vec[:,j,k])
        OP_rhos[j] = norm(OP_x_lc_vec[:,j] - g, 2)**2
    
    OP_log_err_norm = np.log10(OP_rhos)
    min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)

    min_x = Alpha_vec[min_index[0]]
    min_z = np.min(OP_log_err_norm)

    OP_min_alpha1 = min_x
    OP_min_alpha1_ind = min_index[0]
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    return f_rec_OP_grid, OP_min_alpha1

# n = 1000
n = 500
nT2 = n
T2 = np.linspace(-np.pi/2,np.pi/2,n)
TE = T2

G, data_noiseless, g = heat(n, kappa = 1, nargin = 1, nargout=3)
# _,_,g = baart(n)

U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 300
SD_noise= 1/SNR*max(abs(data_noiseless))
Lambda_vec = np.logspace(-5,3,40)
nLambda = len(Lambda_vec)

if np.any(g) < 0:
    print("g is not non-neg")
else:
    print("heat is non-neg problem")

# nrun = 50
# com_vec_DP = np.zeros(nrun)
# com_vec_GCV = np.zeros(nrun)
# com_vec_LC = np.zeros(nrun)
# com_vec_ItoLR = np.zeros(nrun)
# com_vec_ItoLR2 = np.zeros(nrun)
# com_vec_ItoLR3 = np.zeros(nrun)
# com_vec_ItoLR4 = np.zeros(nrun)
# com_vec_oracle = np.zeros(nrun)
# com_vec_ItoLR_feed = np.zeros(nrun)
# com_vec_ItoLR2_feed = np.zeros(nrun)
# com_vec_ItoLR3_feed = np.zeros(nrun)
# com_vec_ItoLR4_feed = np.zeros(nrun)
# Alpha_vec = np.logspace(-8,3,300)

# # eps1 = 1e-3
# # ep_min = 1e-2
# # eps_cut = 10
# # eps_floor = 1e-4
# # eps1 = 1e-2
# # ep_min = 1e-2
# # eps_cut = 6
# # eps_floor = 1e-3
# # feedback = True
# # exp = 0.5

# eps1 = 1e-1
# ep_min = 1e-3
# eps_cut = 6
# eps_floor = 1e-3
# feedback = False
# exp = 2/3

# for i in tqdm(range(nrun)):

#     SD_noise= 1/SNR*max(abs(data_noiseless))

#     noise = np.random.normal(0,SD_noise, data_noiseless.shape)
#     #noise_arr[:,i] = noise.reshape(1,-1)
#     #data_noisy = data_noiseless + noise[:,i]
#     data_noisy = data_noiseless + noise
#     lambda_LC,rho,eta,_ = l_curve(U,s, data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
#     f_rec_LC,_,_ = tikhonov(U,s,V,data_noisy,lambda_LC, nargin=5, nargout=1)
#     com_vec_LC[i] = norm(g - f_rec_LC)

#     delta1 = norm(noise)*1.05
#     x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta1, x_0= None, nargin = 5)
#     # f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)
#     L = np.eye(G.shape[1])
#     x_true = None
#     f_rec_DP, lambda_DP = Tikhonov(G, data_noisy, L, x_true, regparam = 'DP', delta = delta1)
#     f_rec_DP, lambda_DP = tikhonov_vec(U, s, V, data_noisy, np.sqrt(lambda_DP), x_0 = None, nargin = 5)
#     com_vec_DP[i] = norm(g - f_rec_DP)

#     L = np.eye(G.shape[1])
#     x_true = None
#     # f_rec_GCV, reg_min = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')
#     f_rec_GCV, lambda_GCV = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')
#     f_rec_GCV, lambda_GCV = tikhonov_vec(U, s, V, data_noisy, np.sqrt(lambda_GCV), x_0 = None, nargin = 5)
#     com_vec_GCV[i] = norm(g - f_rec_GCV)

#     # x0_ini = f_rec_LC
#     # ep1 = 1e-8
#     # ep2 = 1e-1
#     # ep3 = 1e-3 
#     # ep4 = 1e-4 
#     # f_rec_Chuan, lambda_Chuan = LocReg_unconstrained(data_noisy, G, x0_ini, lambda_LC, ep1, ep2, ep3)

#     gamma_init = 10
#     param_num = 2
#     maxiter = 50
#     LRIto_ini_lam = lambda_LC
#     #Deriv
#     LR_mod_rec, fin_lam, c_array, _ , _ = LocReg_Ito_UC(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
#     LR_Ito_lams = fin_lam
#     com_vec_ItoLR[i] = norm(g - LR_mod_rec)

#     #No Deriv
#     LR_mod_rec2, fin_lam, c_array, _ , _ = LocReg_Ito_UC_2(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
#     LR_Ito_lams2 = fin_lam
#     com_vec_ItoLR2[i] = norm(g - LR_mod_rec2)

#     #Gamma No Deriv
#     LR_mod_rec3, fin_lam, gamma_new1, c_array, _ , _ = LocReg_Ito_UC_3(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
#     LR_Ito_lams3 = fin_lam
#     com_vec_ItoLR3[i] = norm(g - LR_mod_rec3)
    
#     #Gamma Deriv
#     LR_mod_rec4, fin_lam, gamma_new2, c_array, _ , _ = LocReg_Ito_UC_4(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
#     LR_Ito_lams4 = fin_lam
#     com_vec_ItoLR4[i] = norm(g - LR_mod_rec4)

#     LR_mod_recfeed, fin_lamfeed, c_array, _ , _ = LocReg_Ito_UC(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
#     LR_Ito_lamsfeed = fin_lamfeed
#     com_vec_ItoLR_feed[i] = norm(g - LR_mod_recfeed)

#     LR_mod_rec2feed, fin_lamfeed, c_array, _ , _ = LocReg_Ito_UC_2(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
#     LR_Ito_lams2feed = fin_lamfeed
#     com_vec_ItoLR2_feed[i] = norm(g - LR_mod_rec2feed)

#     LR_mod_rec3feed, fin_lamfeed, gamma_new3, c_array, _ , _ = LocReg_Ito_UC_3(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
#     LR_Ito_lams3feed = fin_lamfeed
#     com_vec_ItoLR3_feed[i] = norm(g - LR_mod_rec3feed)

#     LR_mod_rec4feed, fin_lamfeed, gamma_new4, c_array, _ , _ = LocReg_Ito_UC_4(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
#     LR_Ito_lams4feed = fin_lamfeed
#     com_vec_ItoLR4_feed[i] = norm(g - LR_mod_rec4feed)

#     Alpha_vec = np.logspace(-8,3,300)
#     f_rec_OP_grid, oracle_lam = minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g)
#     com_vec_oracle[i] = norm(g - f_rec_OP_grid)

#     err_DP = np.linalg.norm(g - f_rec_DP)
#     err_Lcurve = np.linalg.norm(g - f_rec_LC)
#     # err_Ito2P = np.linalg.norm(g - best_f_rec)
#     err_LR_Ito = np.linalg.norm(g - LR_mod_rec)
#     err_LR_Ito2 = np.linalg.norm(g - LR_mod_rec2)
#     err_LR_Ito3 = np.linalg.norm(g - LR_mod_rec3)
#     err_LR_Ito4 = np.linalg.norm(g - LR_mod_rec4)

#     err_LR_Itofeed = np.linalg.norm(g - LR_mod_recfeed)
#     err_LR_Ito2feed = np.linalg.norm(g - LR_mod_rec2feed)
#     err_LR_Ito3feed = np.linalg.norm(g - LR_mod_rec3feed)
#     err_LR_Ito4feed = np.linalg.norm(g - LR_mod_rec4feed)

#     # err_Chuan = np.linalg.norm(g - f_rec_Chuan)
#     err_GCV = np.linalg.norm(g - f_rec_GCV)
#     err_oracle = np.linalg.norm(g - f_rec_OP_grid)
#     #add 1/n for formalism
#     if i % 15 == 0:
#         #Plot the curves
#         fig, axs = plt.subplots(2, 2, figsize=(12 ,12))
#         # plt.subplots_adjust(wspace=0.3)

#         # Plotting the first subplot
#         # plt.subplot(1, 3, 1)
#         ymax = np.max(g) * 1.15
#         axs[0, 0].plot(T2, g, color = "black",  label = "Ground Truth")
#         axs[0, 0].plot(T2, f_rec_OP_grid, color = "purple",  label = f"Oracle 1P (Error: {round(err_oracle,3)})")
#         axs[0, 0].plot(T2, LR_mod_rec, color = "blue",  label = f"Ito Derivative (Error: {round(err_LR_Ito,3)})")
#         axs[0, 0].plot(T2, LR_mod_rec2, color = "gold",  label = f"Ito No Derivative (Error: {round(err_LR_Ito2,3)})")
#         axs[0, 0].plot(T2, LR_mod_rec3, color = "cyan",  label = f"Ito Gamma No Derivative (Error: {round(err_LR_Ito3,3)})")
#         axs[0, 0].plot(T2, LR_mod_rec4, color = "brown",  label = f"Ito Gamma Derivative (Error: {round(err_LR_Ito4,3)})")
#         axs[0, 0].plot(T2, f_rec_LC, color = "orange", label = f"LC 1P (Error: {round(err_Lcurve,3)})")
#         # axs[0, 0].plot(T2, f_test, color = "cyan", label = "test")
#         axs[0, 0].plot(T2, f_rec_GCV, color = "green", label = f"GCV 1P (Error:{round(err_GCV,3)}")
#         axs[0, 0].plot(T2, f_rec_DP, color = "red", label = f"DP 1P (Error:{round(err_DP,3)})")
#         # axs[0, 0].plot(T2, f_rec_Chuan, color = "red", label = "Chuan")
#         axs[0, 0].set_xlabel('t', fontsize=20, fontweight='bold')
#         axs[0, 0].set_ylabel('f(t)', fontsize=20, fontweight='bold')
#         axs[0, 0].legend(fontsize=10, loc='best')
#         axs[0, 0].set_ylim(0, ymax)
#         axs[0, 0].set_title('Reconstruction', fontsize=16, fontweight='bold')  # Add title here

#         # Plotting the second subplot
#         # plt.subplot(1, 3, 2)
#         axs[0, 1].plot(TE, G @ g, linewidth=3, color='black', label='Ground Truth')
#         axs[0, 1].plot(TE, G @ f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
#         axs[0, 1].plot(TE, G @ LR_mod_rec, color = "blue",  label = "Ito Derivative")
#         axs[0, 1].plot(TE, G @ LR_mod_rec2, color = "gold",  label = "Ito No Derivative")
#         axs[0, 1].plot(TE, G @ LR_mod_rec3, color = "cyan",  label = "Ito Gamma No Derivative")
#         axs[0, 1].plot(TE, G @ LR_mod_rec4, color = "brown",  label = "Ito Gamma Derivative")
#         axs[0, 1].plot(TE, G @ f_rec_LC, color = "orange", label = "LC 1P")
#         axs[0, 1].plot(TE, G @ f_rec_GCV, color = "green", label = "GCV 1P")
#         axs[0, 1].plot(TE, G @ f_rec_DP, color = "red", label = "DP 1P")

#         # axs[0, 1].plot(TE, G @ f_rec_Chuan, color = "red", label = "Chuan")
#         axs[0, 1].legend(fontsize=10, loc='best')
#         axs[0, 1].set_xlabel('s', fontsize=20, fontweight='bold')
#         axs[0, 1].set_ylabel('g(s)', fontsize=20, fontweight='bold')
#         axs[0, 1].set_title('Data', fontsize=16, fontweight='bold')  # Add title here

#         # plt.subplot(1, 3, 3)
#         axs[1, 0].semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='orange', label='LC 1P')
#         axs[1, 0].semilogy(T2, oracle_lam * np.ones(len(T2)), linewidth=3, color='purple', label='Oracle 1P')
#         axs[1, 0].semilogy(T2, LR_Ito_lams, color = "blue", linewidth=3,  label = "Ito Derivative")
#         axs[1, 0].semilogy(T2, LR_Ito_lams2, color = "gold", linewidth=3,  label = "Ito No Derivative")
#         axs[1, 0].semilogy(T2, LR_Ito_lams3, color = "cyan", linewidth=3,  label = "Ito Gamma No Derivative")
#         axs[1, 0].semilogy(T2, LR_Ito_lams4, color = "brown", linewidth=3,  label = "Ito Gamma Derivative")

#         # axs[1,0].semilogy(T2, lambda_Chuan, color = "red",  label = "Chuan")
#         axs[1, 0].semilogy(T2, lambda_GCV * np.ones(len(T2)),linewidth=3,  color = "green", label = "GCV 1P")
#         axs[1, 0].semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color = "red", label = "DP 1P")
#         axs[1, 0].legend(fontsize=10, loc='best')
#         axs[1, 0].set_xlabel('t', fontsize=20, fontweight='bold')
#         axs[1, 0].set_ylabel('Lambda', fontsize=20, fontweight='bold')
#         axs[1, 0].set_title('Regularization Distribution', fontsize=16, fontweight='bold')  # Add title here

#         # ymax2 = 1.5 * np.max(lambda_LC)
#         # axs[1, 0].set_ylim(0, ymax2)

#         table_ax = axs[1, 1]
#         table_ax.axis('off')

#         # Define the data for the table (This is part of the plot)
#         data = [
#             # ["L-Curve Lambda", lambda_LC.item()],
#             # ["Initial Lambda for Ito", LRIto_ini_lam],
#             # ["Initial Eta2 for Ito", round(lam_ini, 4)],
#             # ["Initial Eta2 for Ito", LRIto_ini_lam],
#             # ["Final Eta1 for Ito", fin_etas[0].item()],
#             # ["Final Eta2 for Ito", fin_etas[1].item()],
#             ["Error 1P DP", err_DP.item()],
#             ["Error 1P L-Curve", err_Lcurve.item()],
#             ["Error 1P GCV", err_GCV.item()],
#             ["Error Ito Derivative No Feedback", err_LR_Ito.item()],
#             ["Error Ito No Derivative No Feedback", err_LR_Ito2.item()],
#             ["Error Ito Gamma Derivative No Feedback", err_LR_Ito4.item()],
#             ["Error Ito Gamma No Derivative No Feedback", err_LR_Ito3.item()],
#             # ["error Ito 2P", err_Ito2P.item()],
#             ["Error 1P Oracle", err_oracle.item()],
#             # ["error test", err_test.item()],
#             # ["error Chuan", err_Chuan.item()],
#             ["SNR", SNR],
#             ["Feedback", feedback],
#             ["Exponent", exp]

#             # ["Initial Lambdas for Ito Loc", LR_ini_lam],
#             # ["Final Lambdas for Ito Loc", LR_Ito_lams]
#         ]

#         # Create the table
#         table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
#         table.auto_set_font_size(False)
#         table.set_fontsize(12)
#         table.scale(1.2, 2.5)
#         table_ax.set_title('Heat Problem No Feedback Table', fontsize=16, fontweight='bold', y=1.2)  # Adjust the y position

#         #Save the results in the save results folder
#         plt.tight_layout()
#         string = "Heat"
#         file_path = create_result_folder(string, SNR)
#         plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve_nofeed{i}"))
#         print(f"Saving comparison plot is complete")

#         fig, axs = plt.subplots(2, 2, figsize=(12 ,12))

#         ymax = np.max(g) * 1.15
#         axs[0, 0].plot(T2, g, color = "black",  label = "Ground Truth")
#         axs[0, 0].plot(T2, f_rec_OP_grid, color = "purple",  label = f"Oracle 1P (Error: {round(err_oracle,3)})")
#         axs[0, 0].plot(T2, LR_mod_recfeed, color = "blue",  label = f"Ito Derivative feedback (Error: {round(err_LR_Itofeed,3)})")
#         axs[0, 0].plot(T2, LR_mod_rec2feed, color = "gold",  label = f"Ito No Derivative feedback (Error: {round(err_LR_Ito2feed,3)})")
#         axs[0, 0].plot(T2, LR_mod_rec3feed, color = "cyan",  label = f"Error Ito Gamma No Derivative feedback (Error: {round(err_LR_Ito3feed,3)})")
#         axs[0, 0].plot(T2, LR_mod_rec4feed, color = "brown",  label = f"Ito Gamma Derivative feedback (Error: {round(err_LR_Ito4feed,3)})")

#         # axs[0, 0].plot(T2, f_rec_Chuan, color = "red", label = "Chuan")
#         axs[0, 0].set_xlabel('t', fontsize=20, fontweight='bold')
#         axs[0, 0].set_ylabel('f(t)', fontsize=20, fontweight='bold')
#         axs[0 ,0].legend(fontsize=10, loc='best')
#         axs[0 ,0].set_ylim(0, ymax)
#         axs[0 ,0].set_title('Reconstruction', fontsize=16, fontweight='bold')  # Add title here

#         axs[0 ,1].plot(TE, G @ g, linewidth=3, color='black', label='Ground Truth')
#         axs[0 ,1].plot(TE, G @ f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
#         axs[0 ,1].plot(TE, G @ LR_mod_recfeed, color = "blue",  label = "Ito Derivative Feedback")
#         axs[0 ,1].plot(TE, G @ LR_mod_rec2feed, color = "gold",  label = "Ito No Derivative Feedback")
#         axs[0 ,1].plot(TE, G @ LR_mod_rec3feed, color = "cyan",  label = "Ito Gamma No Derivative Feedback")
#         axs[0 ,1].plot(TE, G @ LR_mod_rec4feed, color = "brown",  label = "Ito Gamma Derivative Feedback")
#         axs[0 ,1].plot(TE, G @ f_rec_LC, color = "orange", label = "LC 1P")
#         axs[0 ,1].plot(TE, G @ f_rec_GCV, color = "green", label = "GCV 1P")
#         axs[0 ,1].plot(TE, G @ f_rec_DP, color = "red", label = "DP 1P")

#         # axs[0, 1].plot(TE, G @ f_rec_Chuan, color = "red", label = "Chuan")
#         axs[0 ,1].legend(fontsize=10, loc='best')
#         axs[0 ,1].set_xlabel('s', fontsize=20, fontweight='bold')
#         axs[0 ,1].set_ylabel('g(s)', fontsize=20, fontweight='bold')
#         axs[0 ,1].set_title('Data', fontsize=16, fontweight='bold')  # Add title here


#         axs[1, 0].semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='orange', label='LC 1P')
#         axs[1, 0].semilogy(T2, oracle_lam * np.ones(len(T2)), linewidth=3, color='purple', label='Oracle 1P')
#         axs[1, 0].semilogy(T2, LR_Ito_lamsfeed, color = "blue", linewidth=3,  label = "Ito Derivative Feedback")
#         axs[1, 0].semilogy(T2, LR_Ito_lams2feed, color = "gold", linewidth=3,  label = "Ito No Derivative Feedback")
#         axs[1, 0].semilogy(T2, LR_Ito_lams3feed, color = "cyan", linewidth=3,  label = "Ito Gamma No Derivative Feedback")
#         axs[1, 0].semilogy(T2, LR_Ito_lams4feed, color = "brown", linewidth=3,  label = "Ito Gamma Derivative Feedback")

#         # axs[1,0].semilogy(T2, lambda_Chuan, color = "red",  label = "Chuan")
#         axs[1, 0].semilogy(T2, lambda_GCV * np.ones(len(T2)),linewidth=3,  color = "green", label = "GCV 1P")
#         axs[1, 0].semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color = "red", label = "DP 1P")
#         axs[1, 0].legend(fontsize=10, loc='best')
#         axs[1, 0].set_xlabel('t', fontsize=20, fontweight='bold')
#         axs[1, 0].set_ylabel('Lambda', fontsize=20, fontweight='bold')
#         axs[1, 0].set_title('Regularization Distribution', fontsize=16, fontweight='bold')  # Add title here


#         table_ax = axs[1, 1]
#         table_ax.axis('off')

#         # Define the data for the table (This is part of the plot)
#         data = [
#             # ["L-Curve Lambda", lambda_LC.item()],
#             # ["Initial Lambda for Ito", LRIto_ini_lam],
#             # ["Initial Eta2 for Ito", round(lam_ini, 4)],
#             # ["Initial Eta2 for Ito", LRIto_ini_lam],
#             # ["Final Eta1 for Ito", fin_etas[0].item()],
#             # ["Final Eta2 for Ito", fin_etas[1].item()],
#             ["Error 1P DP", err_DP.item()],
#             ["Error 1P L-Curve", err_Lcurve.item()],
#             ["Error 1P GCV", err_GCV.item()],
#             ["Error Ito Derivative Feedback", err_LR_Itofeed.item()],
#             ["Error Ito No Derivative Feedback", err_LR_Ito2feed.item()],
#             ["Error Ito Gamma Derivative Feedback", err_LR_Ito4feed.item()],
#             ["Error Ito Gamma No Derivative Feedback", err_LR_Ito3feed.item()],
#             # ["error Ito 2P", err_Ito2P.item()],
#             ["Error 1P Oracle ", err_oracle.item()],
#             # ["error test", err_test.item()],
#             # ["error Chuan", err_Chuan.item()],
#             ["SNR", SNR],
#             ["Feedback", True],
#             ["Exponent", exp]

#             # ["Initial Lambdas for Ito Loc", LR_ini_lam],
#             # ["Final Lambdas for Ito Loc", LR_Ito_lams]
#         ]

#         # Create the table
#         table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
#         table.auto_set_font_size(False)
#         table.set_fontsize(12)
#         table.scale(1.2, 2.5)
#         table_ax.set_title('Heat Problem with Feedback Table', fontsize=16, fontweight='bold', y=1.2)  # Adjust the y position
#         plt.tight_layout()
#         string = "Heat"
#         file_path = create_result_folder(string, SNR)
#         plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve_feedback{i}"))
#         print(f"Saving comparison plot is complete")

# #save into a pickle file
# import pickle
# hyp = {'dp_err': com_vec_DP,
#            'lc_err': com_vec_LC,
#            'gcv_err': com_vec_GCV,
#            'oracle_err': com_vec_oracle,
#            'ito_deriv': com_vec_ItoLR,
#            'ito_noderiv': com_vec_ItoLR2,
#            'ito_gamma_noderiv': com_vec_ItoLR3,
#            'ito_gamma_deriv': com_vec_ItoLR4,
#            'ito_deriv_feed': com_vec_ItoLR_feed,
#            'ito_noderiv_feed': com_vec_ItoLR2_feed,
#            'ito_gamma_noderiv_feed': com_vec_ItoLR3_feed,
#            'ito_gamma_deriv_feed': com_vec_ItoLR4_feed}


# hyp2 = {'snr': SNR, 
#            'dp_err': com_vec_DP,
#            'lc_err': com_vec_LC,
#            'gcv_err': com_vec_GCV,
#            'oracle_err': com_vec_oracle,
#            'ito_deriv': com_vec_ItoLR,
#            'ito_noderiv': com_vec_ItoLR2,
#            'ito_gamma_noderiv': com_vec_ItoLR3,
#            'ito_gamma_deriv': com_vec_ItoLR4,
#            'ito_deriv_feed': com_vec_ItoLR_feed,
#            'ito_noderiv_feed': com_vec_ItoLR2_feed,
#            'ito_gamma_noderiv_feed': com_vec_ItoLR3_feed,
#            'ito_gamma_deriv_feed': com_vec_ItoLR4_feed,
#            'noise_real': nrun,
#            'gamma_init': gamma_init,
#            'gamma_new_nd_nf': gamma_new1,
#            'gamma_new_d_nf': gamma_new2,
#            'gamma_new_nd_f': gamma_new3,
#            'gamma_new_d_f': gamma_new4}

# # Calculate the medians
# medians = {key: np.median(values) for key, values in hyp.items()}

# # Rank the medians
# sorted_keys = sorted(medians, key=medians.get)
# ranks = {key: rank + 1 for rank, key in enumerate(sorted_keys)}

# # Display the results
# print("Medians:")
# for key, median in medians.items():
#     print(f"{key}: {median}")
# print("\nRanks:")
# for key, rank in ranks.items():
#     print(f"{key}: {rank}")

# results = {'medians': medians, 'ranks': ranks}
# with open(f'{file_path}/{string}_rankings.pkl', 'wb') as file:
#     pickle.dump(results, file)
    
# with open(f'{file_path}/{string}_hyp_data.pkl', 'wb') as file:
#     pickle.dump(hyp2, file)

# print("SNR",SNR)
# print("Median DP:", np.median(com_vec_DP))
# print("Median LC:", np.median(com_vec_LC))
# print("Median GCV:", np.median(com_vec_GCV))
# print("Median Oracle:", np.median(com_vec_oracle))
# print("Median Ito Derivative:", np.median(com_vec_ItoLR))
# print("Median Ito No Derivative:", np.median(com_vec_ItoLR2))
# print("Median Ito Gamma:", np.median(com_vec_ItoLR3))
# print("Median Ito Derivative Gamma:", np.median(com_vec_ItoLR4))
# print("Median Ito Derivative feed:", np.median(com_vec_ItoLR_feed))
# print("Median Ito No Derivative feed:", np.median(com_vec_ItoLR2_feed))
# print("Median Ito Gamma No Derivative feed:", np.median(com_vec_ItoLR3_feed))
# print("Median Ito Gamma Derivative feed:", np.median(com_vec_ItoLR4_feed))
# print("Noise Realizations:", nrun)
# print("Gamma Initial:", gamma_init)

# print("SNR",SNR)
# print("Median DP:", np.median(com_vec_DP))
# print("Median LC:", np.median(com_vec_LC))
# print("Median GCV:", np.median(com_vec_GCV))
# print("Median Oracle:", np.median(com_vec_oracle))
# print("Median Ito Derivative:", np.median(com_vec_ItoLR))
# print("Median Ito No Derivative:", np.median(com_vec_ItoLR2))
# print("Median Ito Gamma:", np.median(com_vec_ItoLR3))
# print("Median Ito Derivative Gamma:", np.median(com_vec_ItoLR4))
# print("Median Ito Derivative feed:", np.median(com_vec_ItoLR_feed))
# print("Median Ito No Derivative feed:", np.median(com_vec_ItoLR2_feed))
# print("Median Ito Gamma No Derivative feed:", np.median(com_vec_ItoLR3_feed))
# print("Median Ito Gamma Derivative feed:", np.median(com_vec_ItoLR4_feed))
# print("Noise Realizations:", nrun)
# print("Gamma Initial:", gamma_init)


def run_script(nrun, expvalue, SNR, lam_ini, fignum):
    com_vec_DP = np.zeros(nrun)
    com_vec_GCV = np.zeros(nrun)
    com_vec_LC = np.zeros(nrun)
    com_vec_ItoLR = np.zeros(nrun)
    com_vec_ItoLR2 = np.zeros(nrun)
    com_vec_ItoLR3 = np.zeros(nrun)
    com_vec_ItoLR4 = np.zeros(nrun)
    com_vec_ItoLR_feed = np.zeros(nrun)
    com_vec_ItoLR2_feed = np.zeros(nrun)
    com_vec_ItoLR3_feed = np.zeros(nrun)
    com_vec_ItoLR4_feed = np.zeros(nrun)
    com_vec_oracle = np.zeros(nrun)

    eps1 = 1e-2
    ep_min = 1e-2
    eps_cut = 1.2
    eps_floor = 1e-4
    # eps1 = 1e-3
    # ep_min = 1e-2
    # eps_cut = 10
    # eps_floor = 1e-4
    # exp = 0.5
    exp = expvalue

    def gen_noiseset(SNR,data_noiseless,nrun):
        noise_set = []
        for i in range(nrun):
            SD_noise= 1/SNR*max(abs(data_noiseless))
            noise = np.random.normal(0,SD_noise, data_noiseless.shape)
            noise_set.append(noise)
        return np.array(noise_set)


    ############## Save Folder ###########
    # cwd_temp = os.getcwd()
    # base_file = 'LocReg_Regularization-1'
    # cwd_full = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

    # pat_tag = "classical_prob"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"

    # output_folder = f"SimulationSets/{pat_tag}/Noise"
    # seriesFolder = (f'{cwd_full}/{output_folder}')
    # os.makedirs(seriesFolder, exist_ok = True)
    # if add_noise:
    #     noise_iter_folder = f'Noise_Generation/Noise_Sets/{pat_tag}_SNR{SNR_oi}_{date_of_data}'
    # # else:
    # #     noise_iter_folder = f'NoNoise/{pat_tag}'

    # ####### File Naming Section #########


    # date = date.today()
    # day = date.strftime('%d')
    # month = date.strftime('%B')[0:3]
    # year = date.strftime('%y')

    # if add_noise:
    #     seriesTag = (pat_tag + f"SNR_{SNR_oi}_")
    # else:
    #     seriesTag = (pat_tag + f"NoNoise_")

    # seriesTag = (seriesTag + addTag + day + month + year)



    if SNR == 300:
        SNR300dataset = gen_noiseset(SNR, data_noiseless, nrun)
        noiseset = SNR300dataset
    else:
        SNR30dataset = gen_noiseset(SNR, data_noiseless, nrun)
        noiseset = SNR30dataset

    for i in tqdm(range(nrun)):

        # SD_noise= 1/SNR*max(abs(data_noiseless))

        # noise = np.random.normal(0,SD_noise, data_noiseless.shape)
        #noise_arr[:,i] = noise.reshape(1,-1)
        #data_noisy = data_noiseless + noise[:,i]
        noise = noiseset[i]
        data_noisy = data_noiseless + noise
        lambda_LC,rho,eta,_ = l_curve(U,s,data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
        f_rec_LC,_,_ = tikhonov(U,s,V,data_noisy,lambda_LC, nargin=5, nargout=1)
        com_vec_LC[i] = norm(g - f_rec_LC)

        delta1 = norm(noise)*1.05
        x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta1, x_0= None, nargin = 5)
        # f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)
        L = np.eye(G.shape[1])
        x_true = None
        f_rec_DP, lambda_DP = Tikhonov(G, data_noisy, L, x_true, regparam = 'DP', delta = delta1)
        f_rec_DP, lambda_DP = tikhonov_vec(U, s, V, data_noisy, np.sqrt(lambda_DP), x_0 = None, nargin = 5)
        com_vec_DP[i] = norm(g - f_rec_DP)

        L = np.eye(G.shape[1])
        x_true = None
        # f_rec_GCV, reg_min = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')
        f_rec_GCV, lambda_GCV = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')
        f_rec_GCV, lambda_GCV = tikhonov_vec(U, s, V, data_noisy, np.sqrt(lambda_GCV), x_0 = None, nargin = 5)
        com_vec_GCV[i] = norm(g - f_rec_GCV)

        # x0_ini = f_rec_LC
        # ep1 = 1e-8
        # ep2 = 1e-1
        # ep3 = 1e-3 
        # ep4 = 1e-4 
        # f_rec_Chuan, lambda_Chuan = LocReg_unconstrained(data_noisy, G, x0_ini, lambda_LC, ep1, ep2, ep3)

        gamma_init = 10
        param_num = 2
        maxiter = 50
        if lam_ini == "LC":
            LRIto_ini_lam = lambda_LC
        elif lam_ini == "GCV":
            LRIto_ini_lam = lambda_GCV
        elif lam_ini == "DP":
            LRIto_ini_lam = lambda_DP
        else:
            pass

        feedback = False
        #Deriv
        LR_mod_rec, fin_lam = LocReg_Ito_UC(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
        LR_Ito_lams = fin_lam
        com_vec_ItoLR[i] = norm(g - LR_mod_rec)

        #No Deriv
        LR_mod_rec2, fin_lam = LocReg_Ito_UC_2(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
        LR_Ito_lams2 = fin_lam
        com_vec_ItoLR2[i] = norm(g - LR_mod_rec2)

        #Gamma No Deriv
        LR_mod_rec3, fin_lam, gamma_new1 = LocReg_Ito_UC_3(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
        LR_Ito_lams3 = fin_lam
        com_vec_ItoLR3[i] = norm(g - LR_mod_rec3)
        
        #Gamma Deriv
        LR_mod_rec4, fin_lam, gamma_new2 = LocReg_Ito_UC_4(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
        LR_Ito_lams4 = fin_lam
        com_vec_ItoLR4[i] = norm(g - LR_mod_rec4)

        feedback = True
        LR_mod_recfeed, fin_lamfeed = LocReg_Ito_UC(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
        LR_Ito_lamsfeed = fin_lamfeed
        com_vec_ItoLR_feed[i] = norm(g - LR_mod_recfeed)

        LR_mod_rec2feed, fin_lamfeed = LocReg_Ito_UC_2(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
        LR_Ito_lams2feed = fin_lamfeed
        com_vec_ItoLR2_feed[i] = norm(g - LR_mod_rec2feed)


        LR_mod_rec3feed, fin_lamfeed, gamma_new3  = LocReg_Ito_UC_3(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
        LR_Ito_lams3feed = fin_lamfeed
        com_vec_ItoLR3_feed[i] = norm(g - LR_mod_rec3feed)

        LR_mod_rec4feed, fin_lamfeed, gamma_new4 = LocReg_Ito_UC_4(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
        LR_Ito_lams4feed = fin_lamfeed
        com_vec_ItoLR4_feed[i] = norm(g - LR_mod_rec4feed)

        Alpha_vec = np.logspace(-8,3,300)
        f_rec_OP_grid, oracle_lam = minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g)
        com_vec_oracle[i] = norm(g - f_rec_OP_grid)

        err_DP = np.linalg.norm(g - f_rec_DP)
        err_Lcurve = np.linalg.norm(g - f_rec_LC)
        # err_Ito2P = np.linalg.norm(g - best_f_rec)
        err_LR_Ito = np.linalg.norm(g - LR_mod_rec)
        err_LR_Ito2 = np.linalg.norm(g - LR_mod_rec2)
        err_LR_Ito3 = np.linalg.norm(g - LR_mod_rec3)
        err_LR_Ito4 = np.linalg.norm(g - LR_mod_rec4)

        err_LR_Itofeed = np.linalg.norm(g - LR_mod_recfeed)
        err_LR_Ito2feed = np.linalg.norm(g - LR_mod_rec2feed)
        err_LR_Ito3feed = np.linalg.norm(g - LR_mod_rec3feed)
        err_LR_Ito4feed = np.linalg.norm(g - LR_mod_rec4feed)

        # err_Chuan = np.linalg.norm(g - f_rec_Chuan)
        err_GCV = np.linalg.norm(g - f_rec_GCV)
        err_oracle = np.linalg.norm(g - f_rec_OP_grid)
        #add 1/n for formalism
        if i % fignum == 0:
            #Plot the curves
            fig, axs = plt.subplots(2, 2, figsize=(12 ,12))
            # plt.subplots_adjust(wspace=0.3)

            # Plotting the first subplot
            # plt.subplot(1, 3, 1)
            
            ymax = np.max(g) * 1.15
            axs[0, 0].plot(T2, g, color = "black",  label = "Ground Truth")
            axs[0, 0].plot(T2, f_rec_OP_grid, color = "purple",  label = f"Oracle 1P (Error: {round(err_oracle,3)})")
            axs[0, 0].plot(T2, LR_mod_rec, color = "blue",  label = f"Ito Derivative (Error: {round(err_LR_Ito,3)})")
            axs[0, 0].plot(T2, LR_mod_rec2, color = "gold",  label = f"Ito No Derivative (Error: {round(err_LR_Ito2,3)})")
            axs[0, 0].plot(T2, LR_mod_rec3, color = "cyan",  label = f"Ito Gamma No Derivative (Error: {round(err_LR_Ito3,3)})")
            axs[0, 0].plot(T2, LR_mod_rec4, color = "brown",  label = f"Ito Gamma Derivative (Error: {round(err_LR_Ito4,3)})")
            axs[0, 0].plot(T2, f_rec_LC, color = "orange", label = f"LC 1P (Error: {round(err_Lcurve,3)})")
            # axs[0, 0].plot(T2, f_test, color = "cyan", label = "test")
            axs[0, 0].plot(T2, f_rec_GCV, color = "green", label = f"GCV 1P (Error:{round(err_GCV,3)}")
            axs[0, 0].plot(T2, f_rec_DP, color = "red", label = f"DP 1P (Error:{round(err_DP,3)})")
            # axs[0, 0].plot(T2, f_rec_Chuan, color = "red", label = "Chuan")
            axs[0, 0].set_xlabel('t', fontsize=20, fontweight='bold')
            axs[0, 0].set_ylabel('f(t)', fontsize=20, fontweight='bold')
            axs[0, 0].legend(fontsize=10, loc='best')
            axs[0, 0].set_ylim(0, ymax)
            axs[0, 0].set_title('Reconstruction', fontsize=16, fontweight='bold')  # Add title here

            # Plotting the second subplot
            # plt.subplot(1, 3, 2)
            axs[0, 1].plot(TE, G @ g, linewidth=3, color='black', label='Ground Truth')
            axs[0, 1].plot(TE, G @ f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
            axs[0, 1].plot(TE, G @ LR_mod_rec, color = "blue",  label = "Ito Derivative")
            axs[0, 1].plot(TE, G @ LR_mod_rec2, color = "gold",  label = "Ito No Derivative")
            axs[0, 1].plot(TE, G @ LR_mod_rec3, color = "cyan",  label = "Ito Gamma No Derivative")
            axs[0, 1].plot(TE, G @ LR_mod_rec4, color = "brown",  label = "Ito Gamma Derivative")
            axs[0, 1].plot(TE, G @ f_rec_LC, color = "orange", label = "LC 1P")
            axs[0, 1].plot(TE, G @ f_rec_GCV, color = "green", label = "GCV 1P")
            axs[0, 1].plot(TE, G @ f_rec_DP, color = "red", label = "DP 1P")

            # axs[0, 1].plot(TE, G @ f_rec_Chuan, color = "red", label = "Chuan")
            axs[0, 1].legend(fontsize=10, loc='best')
            axs[0, 1].set_xlabel('s', fontsize=20, fontweight='bold')
            axs[0, 1].set_ylabel('g(s)', fontsize=20, fontweight='bold')
            axs[0, 1].set_title('Data', fontsize=16, fontweight='bold')  # Add title here

            # plt.subplot(1, 3, 3)
            axs[1, 0].semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='orange', label='LC 1P')
            axs[1, 0].semilogy(T2, oracle_lam * np.ones(len(T2)), linewidth=3, color='purple', label='Oracle 1P')
            axs[1, 0].semilogy(T2, LR_Ito_lams, color = "blue", linewidth=3,  label = "Ito Derivative")
            axs[1, 0].semilogy(T2, LR_Ito_lams2, color = "gold", linewidth=3,  label = "Ito No Derivative")
            axs[1, 0].semilogy(T2, LR_Ito_lams3, color = "cyan", linewidth=3,  label = "Ito Gamma No Derivative")
            axs[1, 0].semilogy(T2, LR_Ito_lams4, color = "brown", linewidth=3,  label = "Ito Gamma Derivative")

            # axs[1,0].semilogy(T2, lambda_Chuan, color = "red",  label = "Chuan")
            axs[1, 0].semilogy(T2, lambda_GCV * np.ones(len(T2)),linewidth=3,  color = "green", label = "GCV 1P")
            axs[1, 0].semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color = "red", label = "DP 1P")
            axs[1, 0].legend(fontsize=10, loc='best')
            axs[1, 0].set_xlabel('t', fontsize=20, fontweight='bold')
            axs[1, 0].set_ylabel('Lambda', fontsize=20, fontweight='bold')
            axs[1, 0].set_title('Regularization Distribution', fontsize=16, fontweight='bold')  # Add title here

            # ymax2 = 1.5 * np.max(lambda_LC)
            # axs[1, 0].set_ylim(0, ymax2)

            table_ax = axs[1, 1]
            table_ax.axis('off')

            # Define the data for the table (This is part of the plot)
            data = [
                # ["L-Curve Lambda", lambda_LC.item()],
                # ["Initial Lambda for Ito", LRIto_ini_lam],
                # ["Initial Eta2 for Ito", round(lam_ini, 4)],
                # ["Initial Eta2 for Ito", LRIto_ini_lam],
                # ["Final Eta1 for Ito", fin_etas[0].item()],
                # ["Final Eta2 for Ito", fin_etas[1].item()],
                ["Error 1P DP", err_DP.item()],
                ["Error 1P L-Curve", err_Lcurve.item()],
                ["Error 1P GCV", err_GCV.item()],
                ["Error Ito Derivative No Feedback", err_LR_Ito.item()],
                ["Error Ito No Derivative No Feedback", err_LR_Ito2.item()],
                ["Error Ito Gamma Derivative No Feedback", err_LR_Ito4.item()],
                ["Error Ito Gamma No Derivative No Feedback", err_LR_Ito3.item()],
                # ["error Ito 2P", err_Ito2P.item()],
                ["Error 1P Oracle", err_oracle.item()],
                # ["error test", err_test.item()],
                # ["error Chuan", err_Chuan.item()],
                ["SNR", SNR],
                ["Feedback", feedback],
                ["Exponent", exp]

                # ["Initial Lambdas for Ito Loc", LR_ini_lam],
                # ["Final Lambdas for Ito Loc", LR_Ito_lams]
            ]

            # Create the table
            table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2.5)
            table_ax.set_title('Heat Problem No Feedback Table', fontsize=16, fontweight='bold', y=1.2)  # Adjust the y position

            #Save the results in the save results folder
            string = "heat"
            file_path = create_result_folder(string, SNR)
            exp_str = str(int(exp * 100))
            plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve_feedback{i}_{lam_ini}_exp_{exp_str}"))
            print(f"Saving comparison plot is complete")
            plt.close()

            fig, axs = plt.subplots(2, 2, figsize=(12 ,12))

            ymax = np.max(g) * 1.15
            axs[0, 0].plot(T2, g, color = "black",  label = "Ground Truth")
            axs[0, 0].plot(T2, f_rec_OP_grid, color = "purple",  label = f"Oracle 1P (Error: {round(err_oracle,3)})")
            axs[0, 0].plot(T2, LR_mod_recfeed, color = "blue",  label = f"Ito Derivative feedback (Error: {round(err_LR_Itofeed,3)})")
            axs[0, 0].plot(T2, LR_mod_rec2feed, color = "gold",  label = f"Ito No Derivative feedback (Error: {round(err_LR_Ito2feed,3)})")
            axs[0, 0].plot(T2, LR_mod_rec3feed, color = "cyan",  label = f"Error Ito Gamma No Derivative feedback (Error: {round(err_LR_Ito3feed,3)})")
            axs[0, 0].plot(T2, LR_mod_rec4feed, color = "brown",  label = f"Ito Gamma Derivative feedback (Error: {round(err_LR_Ito4feed,3)})")

            # axs[0, 0].plot(T2, f_rec_Chuan, color = "red", label = "Chuan")
            axs[0, 0].set_xlabel('t', fontsize=20, fontweight='bold')
            axs[0, 0].set_ylabel('f(t)', fontsize=20, fontweight='bold')
            axs[0 ,0].legend(fontsize=10, loc='best')
            axs[0 ,0].set_ylim(0, ymax)
            axs[0 ,0].set_title('Reconstruction', fontsize=16, fontweight='bold')  # Add title here

            axs[0 ,1].plot(TE, G @ g, linewidth=3, color='black', label='Ground Truth')
            axs[0 ,1].plot(TE, G @ f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
            axs[0 ,1].plot(TE, G @ LR_mod_recfeed, color = "blue",  label = "Ito Derivative Feedback")
            axs[0 ,1].plot(TE, G @ LR_mod_rec2feed, color = "gold",  label = "Ito No Derivative Feedback")
            axs[0 ,1].plot(TE, G @ LR_mod_rec3feed, color = "cyan",  label = "Ito Gamma No Derivative Feedback")
            axs[0 ,1].plot(TE, G @ LR_mod_rec4feed, color = "brown",  label = "Ito Gamma Derivative Feedback")
            axs[0 ,1].plot(TE, G @ f_rec_LC, color = "orange", label = "LC 1P")
            axs[0 ,1].plot(TE, G @ f_rec_GCV, color = "green", label = "GCV 1P")
            axs[0 ,1].plot(TE, G @ f_rec_DP, color = "red", label = "DP 1P")

            # axs[0, 1].plot(TE, G @ f_rec_Chuan, color = "red", label = "Chuan")
            axs[0 ,1].legend(fontsize=10, loc='best')
            axs[0 ,1].set_xlabel('s', fontsize=20, fontweight='bold')
            axs[0 ,1].set_ylabel('g(s)', fontsize=20, fontweight='bold')
            axs[0 ,1].set_title('Data', fontsize=16, fontweight='bold')  # Add title here


            axs[1, 0].semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='orange', label='LC 1P')
            axs[1, 0].semilogy(T2, oracle_lam * np.ones(len(T2)), linewidth=3, color='purple', label='Oracle 1P')
            axs[1, 0].semilogy(T2, LR_Ito_lamsfeed, color = "blue", linewidth=3,  label = "Ito Derivative Feedback")
            axs[1, 0].semilogy(T2, LR_Ito_lams2feed, color = "gold", linewidth=3,  label = "Ito No Derivative Feedback")
            axs[1, 0].semilogy(T2, LR_Ito_lams3feed, color = "cyan", linewidth=3,  label = "Ito Gamma No Derivative Feedback")
            axs[1, 0].semilogy(T2, LR_Ito_lams4feed, color = "brown", linewidth=3,  label = "Ito Gamma Derivative Feedback")

            # axs[1,0].semilogy(T2, lambda_Chuan, color = "red",  label = "Chuan")
            axs[1, 0].semilogy(T2, lambda_GCV * np.ones(len(T2)),linewidth=3,  color = "green", label = "GCV 1P")
            axs[1, 0].semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color = "red", label = "DP 1P")
            axs[1, 0].legend(fontsize=10, loc='best')
            axs[1, 0].set_xlabel('t', fontsize=20, fontweight='bold')
            axs[1, 0].set_ylabel('Lambda', fontsize=20, fontweight='bold')
            axs[1, 0].set_title('Regularization Distribution', fontsize=16, fontweight='bold')  # Add title here


            table_ax = axs[1, 1]
            table_ax.axis('off')

            # Define the data for the table (This is part of the plot)
            data = [
                # ["L-Curve Lambda", lambda_LC.item()],
                # ["Initial Lambda for Ito", LRIto_ini_lam],
                # ["Initial Eta2 for Ito", round(lam_ini, 4)],
                # ["Initial Eta2 for Ito", LRIto_ini_lam],
                # ["Final Eta1 for Ito", fin_etas[0].item()],
                # ["Final Eta2 for Ito", fin_etas[1].item()],
                ["Error 1P DP", err_DP.item()],
                ["Error 1P L-Curve", err_Lcurve.item()],
                ["Error 1P GCV", err_GCV.item()],
                ["Error Ito Derivative Feedback", err_LR_Itofeed.item()],
                ["Error Ito No Derivative Feedback", err_LR_Ito2feed.item()],
                ["Error Ito Gamma Derivative Feedback", err_LR_Ito4feed.item()],
                ["Error Ito Gamma No Derivative Feedback", err_LR_Ito3feed.item()],
                # ["error Ito 2P", err_Ito2P.item()],
                ["Error 1P Oracle ", err_oracle.item()],
                # ["error test", err_test.item()],
                # ["error Chuan", err_Chuan.item()],
                ["SNR", SNR],
                ["Feedback", True],
                ["Exponent", exp]

                # ["Initial Lambdas for Ito Loc", LR_ini_lam],
                # ["Final Lambdas for Ito Loc", LR_Ito_lams]
            ]

            # Create the table
            table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2.5)
            table_ax.set_title('Heat Problem with Feedback Table', fontsize=16, fontweight='bold', y=1.2)  # Adjust the y position
            string = "heat"
            file_path = create_result_folder(string, SNR)
            exp_str = str(int(exp * 100))
            plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve_feedback{i}_{lam_ini}_exp_{exp_str}"))
            print(f"Saving comparison plot is complete")
            plt.close()


    #save into a pickle file
    import pickle
    hyp = {'dp_err': com_vec_DP,
            'lc_err': com_vec_LC,
            'gcv_err': com_vec_GCV,
            'oracle_err': com_vec_oracle,
            'ito_deriv': com_vec_ItoLR,
            'ito_noderiv': com_vec_ItoLR2,
            'ito_gamma_noderiv': com_vec_ItoLR3,
            'ito_gamma_deriv': com_vec_ItoLR4,
            'ito_deriv_feed': com_vec_ItoLR_feed,
            'ito_noderiv_feed': com_vec_ItoLR2_feed,
            'ito_gamma_noderiv_feed': com_vec_ItoLR3_feed,
            'ito_gamma_deriv_feed': com_vec_ItoLR4_feed}


    hyp2 = {'snr': SNR, 
            'dp_err': com_vec_DP,
            'lc_err': com_vec_LC,
            'gcv_err': com_vec_GCV,
            'oracle_err': com_vec_oracle,
            'ito_deriv': com_vec_ItoLR,
            'ito_noderiv': com_vec_ItoLR2,
            'ito_gamma_noderiv': com_vec_ItoLR3,
            'ito_gamma_deriv': com_vec_ItoLR4,
            'ito_deriv_feed': com_vec_ItoLR_feed,
            'ito_noderiv_feed': com_vec_ItoLR2_feed,
            'ito_gamma_noderiv_feed': com_vec_ItoLR3_feed,
            'ito_gamma_deriv_feed': com_vec_ItoLR4_feed,
            'noise_real': nrun,
            'gamma_init': gamma_init,
            'gamma_new_nd_nf': gamma_new1,
            'gamma_new_d_nf': gamma_new2,
            'gamma_new_nd_f': gamma_new3,
            'gamma_new_d_f': gamma_new4}

    # Calculate the medians
    medians = {key: np.median(values) for key, values in hyp.items()}

    # Rank the medians
    sorted_keys = sorted(medians, key=medians.get)
    ranks = {key: rank + 1 for rank, key in enumerate(sorted_keys)}

    # Display the results
    print("Medians:")
    for key, median in medians.items():
        print(f"{key}: {median}")
    print("\nRanks:")
    for key, rank in ranks.items():
        print(f"{key}: {rank}")

    results = {'medians': medians, 'ranks': ranks}
    try:
        with open(f'{file_path}/{string}_rankings_SNR_{SNR}_exp_{exp}_{lam_ini}.pkl', 'wb') as file:
            pickle.dump(results, file)
        print(f'Saved {string}_rankings_SNR_{SNR}_exp_{exp}_{lam_ini}.pkl')
    except Exception as e:
        print(f'Failed to save {string}_rankings_SNR_{SNR}_exp_{exp}_{lam_ini}.pkl: {e}')

    # Save hyperparameters data
    try:
        with open(f'{file_path}/{string}_hyp_data_SNR_{SNR}_exp_{exp}_{lam_ini}.pkl', 'wb') as file:
            pickle.dump(hyp2, file)
        print(f'Saved {string}_hyp_data_SNR_{SNR}_exp_{exp}_{lam_ini}.pkl')
    except Exception as e:
        print(f'Failed to save {string}_hyp_data_SNR_{SNR}_exp_{exp}_{lam_ini}.pkl: {e}')

    # Save dataset based on SNR
    # if SNR == 300:
    #     try:
    #         file_path2 = {}
    #         np.save(f'{file_path}/SNR300dataset.npy', SNR300dataset)
    #         print('Saved SNR300dataset.npy')
    #     except Exception as e:
    #         print(f'Failed to save SNR300dataset.npy: {e}')
    # else:
    #     try:
    #         np.save(f'{file_path}/SNR30dataset.npy', SNR30dataset)
    #         print('Saved SNR30dataset.npy')
    #     except Exception as e:
    #         print(f'Failed to save SNR30dataset.npy: {e}')

    # print("SNR",SNR)
    # print("Median DP:", np.median(com_vec_DP))
    # print("Median LC:", np.median(com_vec_LC))
    # print("Median GCV:", np.median(com_vec_GCV))
    # print("Median Oracle:", np.median(com_vec_oracle))
    # print("Median Ito Derivative:", np.median(com_vec_ItoLR))
    # print("Median Ito No Derivative:", np.median(com_vec_ItoLR2))
    # print("Median Ito Gamma:", np.median(com_vec_ItoLR3))
    # print("Median Ito Derivative Gamma:", np.median(com_vec_ItoLR4))
    # print("Median Ito Derivative feed:", np.median(com_vec_ItoLR_feed))
    # print("Median Ito No Derivative feed:", np.median(com_vec_ItoLR2_feed))
    # print("Median Ito Gamma No Derivative feed:", np.median(com_vec_ItoLR3_feed))
    # print("Median Ito Gamma Derivative feed:", np.median(com_vec_ItoLR4_feed))
    # print("Noise Realizations:", nrun)
    # print("Gamma Initial:", gamma_init)
    return 

# vecs = np.asarray([com_vec_DP, com_vec_LC, com_vec_GCV, com_vec_ItoLR])
# date = datetime.now().strftime("%Y%m%d")
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/classical_problems/Testing_Ito_{date}_heat.csv', vecs , delimiter=',')
nrun_values = [20]  # Example: vary nrun between 20 and 30
expvalue_values = [0, 1, 0.66]  # Example: only one value for expvalue
SNR_values = [300, 30]  # Example: vary SNR between 300 and 30
# feedback_values = [False, True]  # Example: vary feedback between False and True
lam_ini_values = ['LC', 'GCV', "DP"]  # Example: vary lam_ini between 'LC' and 'other'
fignum_values = [5]  # Example: only one value for fignum

# Nested loops to iterate over all parameter combinations
parameter_combinations = [(nrun, expvalue, SNR, lam_ini, fignum)
                          for nrun in nrun_values
                          for expvalue in expvalue_values
                          for SNR in SNR_values
                          for lam_ini in lam_ini_values
                          for fignum in fignum_values]

# Function to run scripts in parallel
def run_parallel(func, args_list, max_workers=None):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, *args) for args in args_list]
        for future in tqdm(futures, total=len(args_list), desc='Running scripts in parallel'):
            future.result()  # Wait for completion and get the result

# Run scripts in parallel
run_parallel(run_script, parameter_combinations, max_workers=None)  # Adjust max_workers as needed

# vecs = np.asarray([com_vec_DP, com_vec_LC, com_vec_GCV, com_vec_ItoLR])
# date = datetime.now().strftime("%Y%m%d")
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/classical_problems/Testing_Ito_{date}_heat.csv', vecs , delimiter=',')


# noise = np.random.normal(0,SD_noise, data_noiseless.shape)
# #noise_arr[:,i] = noise.reshape(1,-1)
# #data_noisy = data_noiseless + noise[:,i]
# data_noisy = data_noiseless + noise
# lambda_LC,rho,eta,_ = l_curve(U,s,data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
# f_rec_LC,_,_ = tikhonov(U,s,V,data_noisy,lambda_LC, nargin=5, nargout=1)

# delta1 = norm(noise)*1.05
# x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta1, x_0= None, nargin = 5)
# # f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)
# L = np.eye(G.shape[1])
# x_true = None
# f_rec_DP, lambda_DP = Tikhonov(G, data_noisy, L, x_true, regparam = 'DP', delta = delta1)
# f_rec_DP, lambda_DP = tikhonov_vec(U, s, V, data_noisy, np.sqrt(lambda_DP), x_0 = None, nargin = 5)


# L = np.eye(G.shape[1])
# x_true = None
# # f_rec_GCV, reg_min = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')
# f_rec_GCV, lambda_GCV = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')
# f_rec_GCV, lambda_GCV = tikhonov_vec(U, s, V, data_noisy, np.sqrt(lambda_GCV), x_0 = None, nargin = 5)

# # x0_ini = f_rec_LC
# # ep1 = 1e-8
# # ep2 = 1e-1
# # ep3 = 1e-3 
# # ep4 = 1e-4 
# # f_rec_Chuan, lambda_Chuan = LocReg_unconstrained(data_noisy, G, x0_ini, lambda_LC, ep1, ep2, ep3)

# gamma_init = 10
# param_num = 2
# maxiter = 50
# LRIto_ini_lam = lambda_LC


# LR_mod_rec, fin_lam, c_array, _ , _ = LocReg_Ito_UC(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor)
# LR_Ito_lams = fin_lam

# Alpha_vec = np.logspace(-8,3,300)
# f_rec_OP_grid, oracle_lam = minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g)

# err_DP = np.linalg.norm(g - f_rec_DP)
# err_Lcurve = np.linalg.norm(g - f_rec_LC)
# # err_Ito2P = np.linalg.norm(g - best_f_rec)
# err_LR_Ito = np.linalg.norm(g - LR_mod_rec)
# # err_Chuan = np.linalg.norm(g - f_rec_Chuan)
# err_GCV = np.linalg.norm(g - f_rec_GCV)
# err_oracle = np.linalg.norm(g - f_rec_OP_grid)

# #Plot the curves
# fig, axs = plt.subplots(2, 2, figsize=(12, 12))
# # plt.subplots_adjust(wspace=0.3)

# # Plotting the first subplot
# # plt.subplot(1, 3, 1)
# ymax = np.max(g) * 1.15
# axs[0, 0].plot(T2, g, color = "black",  label = "Ground Truth")
# axs[0, 0].plot(T2, f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
# axs[0, 0].plot(T2, LR_mod_rec, color = "blue",  label = "Ito NP")
# axs[0, 0].plot(T2, f_rec_LC, color = "orange", label = "LC 1P")
# # axs[0, 0].plot(T2, f_test, color = "pink", label = "test")
# axs[0, 0].plot(T2, f_rec_GCV, color = "green", label = "GCV 1P")
# axs[0, 0].plot(T2, f_rec_DP, color = "red", label = "DP 1P")

# # axs[0, 0].plot(T2, f_rec_Chuan, color = "red", label = "Chuan")
# axs[0, 0].set_xlabel('t', fontsize=20, fontweight='bold')
# axs[0, 0].set_ylabel('f(t)', fontsize=20, fontweight='bold')
# axs[0, 0].legend(fontsize=10, loc='best')
# axs[0, 0].set_ylim(0, ymax)
# axs[0, 0].set_title('Reconstruction', fontsize=16, fontweight='bold')  # Add title here

# # Plotting the second subplot
# # plt.subplot(1, 3, 2)
# axs[0, 1].plot(TE, G @ g, linewidth=3, color='black', label='Ground Truth')
# axs[0, 1].plot(TE, G @ f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
# axs[0, 1].plot(TE, G @ LR_mod_rec, color = "blue",  label = "Ito NP")
# axs[0, 1].plot(TE, G @ f_rec_LC, color = "orange", label = "LC 1P")
# axs[0, 1].plot(TE, G @ f_rec_GCV, color = "green", label = "GCV 1P")
# axs[0, 1].plot(TE, G @ f_rec_DP, color = "red", label = "DP 1P")

# # axs[0, 1].plot(TE, G @ f_rec_Chuan, color = "red", label = "Chuan")
# axs[0, 1].legend(fontsize=10, loc='best')
# axs[0, 1].set_xlabel('s', fontsize=20, fontweight='bold')
# axs[0, 1].set_ylabel('g(s)', fontsize=20, fontweight='bold')
# axs[0, 1].set_title('Data', fontsize=16, fontweight='bold')  # Add title here

# # plt.subplot(1, 3, 3)
# axs[1, 0].semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='orange', label='LC 1P')
# axs[1, 0].semilogy(T2, oracle_lam * np.ones(len(T2)), linewidth=3, color='purple', label='Oracle 1P')
# axs[1, 0].semilogy(T2, fin_lam, color = "blue", linewidth=3,  label = "Ito NP")
# # axs[1,0].semilogy(T2, lambda_Chuan, color = "red",  label = "Chuan")
# axs[1, 0].semilogy(T2, lambda_GCV * np.ones(len(T2)),linewidth=3,  color = "green", label = "GCV 1P")
# axs[1, 0].semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color = "red", label = "DP 1P")
# axs[1, 0].legend(fontsize=10, loc='best')
# axs[1, 0].set_xlabel('t', fontsize=20, fontweight='bold')
# axs[1, 0].set_ylabel('Lambda', fontsize=20, fontweight='bold')
# axs[1, 0].set_title('Regularization Distribution', fontsize=16, fontweight='bold')  # Add title here

# # ymax2 = 1.5 * np.max(lambda_LC)
# # axs[1, 0].set_ylim(0, ymax2)

# table_ax = axs[1, 1]
# table_ax.axis('off')

# # Define the data for the table (This is part of the plot)
# data = [
#     ["L-Curve Lambda", lambda_LC.item()],
#     ["Initial Lambda for Ito", LRIto_ini_lam],
#     # ["Initial Eta2 for Ito", round(lam_ini, 4)],
#     # ["Initial Eta2 for Ito", LRIto_ini_lam],
#     # ["Final Eta1 for Ito", fin_etas[0].item()],
#     # ["Final Eta2 for Ito", fin_etas[1].item()],
#     ["Error Conventional DP", err_DP.item()],
#     ["Error Conventional L-Curve", err_Lcurve.item()],
#     ["Error Conventional GCV", err_GCV.item()],
#     ["Error Ito NP", err_LR_Ito.item()],
#     # ["error Ito 2P", err_Ito2P.item()],
#     ["Error Oracle 1P", err_oracle.item()],
#     # ["error test", err_test.item()],
#     # ["error Chuan", err_Chuan.item()],
#     ["SNR", SNR]

#     # ["Initial Lambdas for Ito Loc", LR_ini_lam],
#     # ["Final Lambdas for Ito Loc", LR_Ito_lams]
# ]

# # Create the table
# table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.scale(1.2, 2.5)
# axs[1, 1].set_title('L2 Error Summary Table', fontsize=16, fontweight='bold')  # Add title here


# #Save the results in the save results folder
# plt.tight_layout()
# string = "heat"
# file_path = create_result_folder(string, SNR)
# plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve"))
# print(f"Saving comparison plot is complete")

"""
nrun = 50
com_vec_DP = np.zeros(nrun)
com_vec_GCV = np.zeros(nrun)
com_vec_LC = np.zeros(nrun)
com_vec_locreg = np.zeros(nrun)
com_vec_locreg_ne = np.zeros(nrun)

res_vec_DP = np.zeros((n,nrun))
res_vec_GCV = np.zeros((n,nrun))
res_vec_LC = np.zeros((n,nrun))
res_vec_locreg = np.zeros((n,nrun))
res_vec_locreg_ne = np.zeros((n,nrun))

noise_size = int(data_noiseless.shape[0])
noise_arr = np.zeros((noise_size, nrun))
lam_LC = np.zeros((n,nrun))
lam_GCV = np.zeros((n,nrun))
lam_LocReg = np.zeros((n,nrun))
lam_DP = np.zeros((n,nrun))
lam_LocReg_ne = np.zeros((n,nrun))

for i in tqdm(range(nrun)):
    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    #data_noisy = data_noiseless + noise[:,i]
    noise_arr[:,i] = noise.reshape(1,-1)
    data_noisy = data_noiseless + noise
    
    # L-curve
    #reg_corner,rho,eta = l_curve(U,s,data_noisy)
    reg_corner,rho,eta,_ = l_curve(U,s,data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
    lam_LC[:,i] = reg_corner
    f_rec_LC,_,_ = tikhonov(U,s,V,data_noisy,reg_corner, nargin=5, nargout=1)
    com_vec_LC[i] = norm(g - f_rec_LC)
    res_vec_LC[:,i] = f_rec_LC
    # %% GCV
    reg_min,_,reg_param = gcv(U,s,data_noisy, method = 'Tikh', nargin = 3, nargout = 3)
    lam_GCV[:,i] = reg_min
    f_rec_GCV,_,_ = tikhonov(U,s,V,data_noisy,reg_min, nargin=5, nargout=1)
    com_vec_GCV[i] = norm(g - f_rec_GCV)
    res_vec_GCV[:,i] = f_rec_GCV
    
    # %% DP
    #delta = norm(noise[:,i])*1.05
    delta = norm(noise)*1.05
    x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta, x_0= None, nargin = 5)
    lam_DP[:,i] = lambda_DP
    f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)
    com_vec_DP[i] = norm(g - f_rec_DP)
    res_vec_DP[:,i] = f_rec_DP
    
    # %% locreg
    x0_ini = f_rec_LC
    # %     lambda_ini = reg_corner;
    ep1 = 1e-1
    # % 1/(|x|+ep1)
    ep2 = 1e-1
    # % norm(dx)/norm(x)
    ep3 = 1e-3
    # % norm(x_(k-1) - x_k)/norm(x_(k-1))
    ep4 = 1e-4 
    # % lb for ep1

    # f_rec_locreg, lambda_locreg = LocReg_v2(data_noisy, G, reg_corner)
    f_rec_locreg, lambda_locreg = LocReg_unconstrained_NE(data_noisy, G, x0_ini, reg_corner, ep1, ep2, ep3)
    lam_LocReg_ne[:,i] = lambda_locreg
    com_vec_locreg_ne[i] = norm(g - f_rec_locreg)
    res_vec_locreg_ne[:,i] = f_rec_locreg

    f_rec_locreg, lambda_locreg = LocReg_unconstrained(data_noisy, G, x0_ini, reg_corner, ep1, ep2, ep3)
    lam_LocReg[:,i] = lambda_locreg
    com_vec_locreg[i] = norm(g - f_rec_locreg)
    # val_arr.append(np.mean(com_vec_locreg[i]))
    res_vec_locreg[:,i] = f_rec_locreg

# ind = np.argmin(val_arr)
# print("value index", ind)
# print("value opt.", val_arr[ind])

print('The mean error for DP is', str(np.median(com_vec_DP)))
print('The mean error for L-curve is', str(np.median(com_vec_LC)))
print('The mean error for GCV is', str(np.median(com_vec_GCV)))
print('The mean error for locreg is', str(np.median(com_vec_locreg)))
print('The mean error for locreg_ne is', str(np.median(com_vec_locreg_ne)))

threshold = 3 * np.median(com_vec_GCV)  # Adjust as needed

# Identify outliers by comparing with the threshold
outliers = com_vec_GCV > threshold

# Replace outliers with NaN (Not a Number) or any other value of your choice
com_vec_GCV[outliers] = np.median(com_vec_GCV)

print('The SD for DP is', str(np.std(com_vec_DP)))
print('The SD for L-curve is', str(np.std(com_vec_LC)))
print('The SD for GCV is', str(np.std(com_vec_GCV)))
print('The SD for locreg is', str(np.std(com_vec_locreg)))
print('The SD for locreg_ne is', str(np.std(com_vec_locreg_ne)))

vecs = np.asarray([com_vec_DP, com_vec_LC, com_vec_GCV, com_vec_locreg])
np.savetxt('heat_vec.csv', vecs , delimiter=',')
np.savetxt('heat_noise_arr_1000', noise_arr)


rid_GCV = np.where(com_vec_GCV >= 100)[0]
res_vec_GCV = np.delete(res_vec_GCV, rid_GCV, axis=1)

# #Save files
# info = {
#         'LC': res_vec_LC,
#         'GCV': res_vec_GCV,
#         'DP': res_vec_DP,
#         'LocReg' : res_vec_locreg,
#         'Lambda': Lambda_vec,
#         'Noise Array': noise_arr,
#         'G': G,
#         'T2': T2,
#         'TE': TE,
#         'SNR': SNR
#     }
# print("Finished computing info")
# # Save file
# FileName = f"Heat_info_09_11_23_{nrun}NR"

#     # Construct the directory path
# directory_path = os.path.join(os.getcwd(), "classical_problems", "problem_info")
# os.makedirs(directory_path, exist_ok=True)

#     # Construct the full file path using os.path.join()
# file_path = os.path.join(directory_path, FileName + '.pkl')
#      # Save file
# with open(file_path, 'wb') as file:
#     pickle.dump(info, file)
# print(f"File saved at: {file_path}")

####################################################################
####For figures

#Plot the ground truth with the mehtod
name = "locreg_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_locreg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'heat_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format

name = "gcv_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'heat_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format

name = "dp_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_DP, linewidth=3, color='brown', label='DP')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'heat_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format

name = "lc_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'heat_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format

#Plot the lambda plots
name="lambda_plt"
plt.figure(figsize=(12, 6))
plt.semilogy(T2, lam_LocReg * np.ones(len(T2)), linestyle=':', linewidth=3, color='magenta', label='LocReg')
plt.semilogy(T2, lam_DP * np.ones(len(T2)),linewidth=3, color='brown', label='DP')
plt.semilogy(T2, lam_GCV * np.ones(len(T2)), linestyle='--', linewidth=3, color='blue', label='GCV')
plt.semilogy(T2, lam_LC * np.ones(len(T2)), linestyle='-.',linewidth=3, color='cyan', label='L-curve')
handles, labels = plt.gca().get_legend_handles_labels()
dict_of_labels = dict(zip(labels, handles))
plt.legend(dict_of_labels.values(), dict_of_labels.keys(), fontsize='small', loc='best')
#plt.legend(fontsize='small', loc='upper left')
#plt.legend(['DP', 'GCV', 'L-curve', 'LocReg'], fontsize=20, loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Lambda', fontsize=20, fontweight='bold')
plt.savefig(f'heat_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format

############################################














#Print graph
plt.figure(figsize=(40, 20))
plt.subplots_adjust(wspace=0.3)

plt.subplot(1, 3, 1)
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_locreg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
plt.plot(T2, res_vec_locreg_ne, linestyle=':', linewidth=3, color='yellow', label='LocReg_NE')

plt.plot(T2, res_vec_DP, linewidth=3, color='brown', label='DP')
plt.plot(T2, res_vec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
plt.plot(T2, res_vec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
# plt.plot(T2, f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
# get legend handles and their corresponding labels
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.show()
# Plotting the second subplot
plt.subplot(1, 3, 2)
plt.plot(TE, G @ g, linewidth=3, color='black', label='True Dist.')
plt.plot(TE, G @ res_vec_locreg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
plt.plot(TE, G @ res_vec_DP, linewidth=3, color='brown', label='DP')
plt.plot(TE, G @ res_vec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
plt.plot(TE, G @ res_vec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
plt.plot(TE, G @ res_vec_locreg_ne, linestyle='-.', linewidth=3, color='yellow', label='LocReg_ne')

#plt.plot(TE, G @ f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
plt.legend(fontsize='small', loc='best')
plt.xlabel('TE', fontsize=20, fontweight='bold')
plt.ylabel('Intensity', fontsize=20, fontweight='bold')

plt.subplot(1,3,3)
plt.semilogy(T2, lam_LocReg * np.ones(len(T2)), linestyle=':', linewidth=3, color='magenta', label='LocReg')
plt.semilogy(T2, lam_DP * np.ones(len(T2)),linewidth=3, color='brown', label='DP')
plt.semilogy(T2, lam_GCV * np.ones(len(T2)), linestyle='--', linewidth=3, color='blue', label='GCV')
plt.semilogy(T2, lam_LC * np.ones(len(T2)), linestyle='-.',linewidth=3, color='cyan', label='L-curve')
plt.semilogy(T2, lam_LocReg_ne * np.ones(len(T2)), linestyle='-.',linewidth=3, color='yellow', label='LocReg_NE')

handles, labels = plt.gca().get_legend_handles_labels()
dict_of_labels = dict(zip(labels, handles))
plt.legend(dict_of_labels.values(), dict_of_labels.keys(), fontsize=10, loc='best')
#plt.legend(fontsize='small', loc='upper left')
#plt.legend(['DP', 'GCV', 'L-curve', 'LocReg'], fontsize=20, loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Lambda', fontsize=20, fontweight='bold')
plt.subplots_adjust(bottom=0.15, top=0.85)
# plt.savefig(f'heat_prob_nrun{nrun}_modified.png')  # You can specify the desired filename and format
plt.show()
"""


# plt.figure(figsize=(10, 8))

# plt.subplot(2, 2, 1)
# plt.plot(T2, g, linewidth=1.5)
# plt.plot(T2, res_vec_DP)
# plt.title('DP')

# plt.subplot(2, 2, 2)
# plt.plot(T2, g, linewidth=1.5)
# plt.plot(T2, res_vec_LC)
# plt.title('L curve')

# plt.subplot(2, 2, 3)
# plt.plot(T2, g, linewidth=1.5)
# plt.plot(T2, res_vec_GCV)
# plt.title('GCV')

# plt.subplot(2, 2, 4)
# plt.plot(T2, g, linewidth=1.5)
# plt.plot(T2, res_vec_locreg)
# plt.title('LocReg')
