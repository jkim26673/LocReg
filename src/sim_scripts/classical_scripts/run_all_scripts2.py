# #Run packages
# import numpy as np
# import os
# import pickle
# import matplotlib.pyplot as plt
# from regu.Lcurve import Lcurve
# from regu.baart import baart
# from regu.blur import blur
# from regu.deriv2 import deriv2
# from regu.foxgood import foxgood
# from regu.gravity import gravity
# from regu.phillips import phillips
# from regu.shaw import shaw
# from regu.wing import wing
# from regu.csvd import csvd
# from regu.l_curve import l_curve
# from regu.tikhonov import tikhonov
# from regu.gcv import gcv
# from regu.discrep import discrep
# from numpy.linalg import norm
# from Utilities_functions.LocReg_unconstrained import LocReg_unconstrained
# from tqdm import tqdm
# # from Utilities_functions.LocReg_v2 import LocReg_v2
# from Utilities_functions.LocReg_NE import LocReg_unconstrained_NE
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor
# from functools import partial
# from numpy.linalg import norm
# from Utilities_functions.pasha_gcv import Tikhonov
# from Ito_LocReg import *
# from tqdm import tqdm
# from datetime import datetime
# from Utilities_functions.pasha_gcv import Tikhonov
# from Utilities_functions.tikhonov_vec import tikhonov_vec
# import multiprocessing as mp
from utils.load_imports.load_classical import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cwd = os.getcwd()
#parameter settings:
add_noise = True
blur = False

#Save folders

cwd = os.getcwd()

cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

pat_tag = "classical_prob"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"

output_folder = f"SimulationSets/{pat_tag}"
lam_ini = 'LC'

# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + output_folder + lam_ini

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

#Global Parameters:
#Functions
def minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g):
    OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
    OP_rhos = np.zeros((len(Alpha_vec)))
    for j in (range(len(Alpha_vec))):
        A  = G.T @ G + Alpha_vec[j]**2 * L.T @ L 
        b =  G.T @ data_noisy
        exp = np.linalg.solve(A,b)
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

def plot(i):

    return 


def sequence(i):
    SD_noise= 1/SNR*max(abs(data_noiseless))

    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    #noise_arr[:,i] = noise.reshape(1,-1)
    #data_noisy = data_noiseless + noise[:,i]
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
    LRIto_ini_lam = lambda_LC

    #Deriv
    LR_mod_rec, fin_lam, c_array, _ , _ = LocReg_Ito_UC(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
    LR_Ito_lams = fin_lam
    com_vec_ItoLR[i] = norm(g - LR_mod_rec)

    #No Deriv
    LR_mod_rec2, fin_lam, c_array, _ , _ = LocReg_Ito_UC_2(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
    LR_Ito_lams2 = fin_lam
    com_vec_ItoLR2[i] = norm(g - LR_mod_rec2)

    #Gamma No Deriv
    LR_mod_rec3, fin_lam, gamma_new1, c_array, _ , _ = LocReg_Ito_UC_3(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
    LR_Ito_lams3 = fin_lam
    com_vec_ItoLR3[i] = norm(g - LR_mod_rec3)
    
    #Gamma Deriv
    LR_mod_rec4, fin_lam, gamma_new2, c_array, _ , _ = LocReg_Ito_UC_4(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
    LR_Ito_lams4 = fin_lam
    com_vec_ItoLR4[i] = norm(g - LR_mod_rec4)

    LR_mod_recfeed, fin_lamfeed, c_array, _ , _ = LocReg_Ito_UC(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
    LR_Ito_lamsfeed = fin_lamfeed
    com_vec_ItoLR_feed[i] = norm(g - LR_mod_recfeed)

    LR_mod_rec2feed, fin_lamfeed, c_array, _ , _ = LocReg_Ito_UC_2(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
    LR_Ito_lams2feed = fin_lamfeed
    com_vec_ItoLR2_feed[i] = norm(g - LR_mod_rec2feed)


    LR_mod_rec3feed, fin_lamfeed, gamma_new3, c_array, _ , _ = LocReg_Ito_UC_3(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
    LR_Ito_lams3feed = fin_lamfeed
    com_vec_ItoLR3_feed[i] = norm(g - LR_mod_rec3feed)

    LR_mod_rec4feed, fin_lamfeed, gamma_new4, c_array, _ , _ = LocReg_Ito_UC_4(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback =True)
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

    if i % 15 == 0:
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
        table_ax.set_title('Baart Problem No Feedback Table', fontsize=16, fontweight='bold', y=1.2)  # Adjust the y position

        #Save the results in the save results folder
        plt.tight_layout()
        string = "baart"
        file_path = create_result_folder(string, SNR)
        plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve_nofeed{i}"))
        print(f"Saving comparison plot is complete")

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
        table_ax.set_title('Baart Problem with Feedback Table', fontsize=16, fontweight='bold', y=1.2)  # Adjust the y position
        plt.tight_layout()
        string = "baart"
        file_path = create_result_folder(string, SNR)
        plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve_feedback{i}"))
        print(f"Saving comparison plot is complete")  
    return 


if __name__ == '__main__':
    nrun = 100
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
    feedback = False
    # exp = 0.5
    exp = 2/3

    SNR = 300

    num_discret = 500
    T2 = np.linspace(-np.pi/2,np.pi/2,num_discret)
    TE = T2
    nT2 = len(T2)

    G,data_noiseless,g = baart(num_discret)
    U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)

    # Initialize arrays or lists to store results

    # Initialize other arrays or lists as needed

    # Define other variables such as SNR, U, s, V, data_noiseless, etc.

    # Create a pool of processes
    batch_size = 10  # Process 10 runs per batch

    num_processes = 10  # Number of processes to use
    pool = mp.Pool(processes=num_processes)

    # Use tqdm for progress bar
    with tqdm(total=nrun) as pbar:
        for batch_start in range(0, nrun, batch_size):
            batch_indices = range(batch_start, min(batch_start + batch_size, nrun))
            results = pool.map(sequence, batch_indices)
            pbar.update(len(batch_indices))

    pool.close()
    pool.join()


# import concurrent.futures
# import time

# def process_task(i):
#     # Simulate a task that takes time
#     time.sleep(0.1)
#     return f"Task {i} completed"

# def main():
#     tasks = range(50)  # 50 iterations
#     results = []

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = {executor.submit(process_task, i): i for i in tasks}
#         for future in concurrent.futures.as_completed(futures):
#             results.append(future.result())

#     for result in results:
#         print(result)

# if __name__ == "__main__":
#     main()