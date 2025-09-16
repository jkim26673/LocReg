# import numpy as np
# import sys
# sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
# import os
# import pickle
# import matplotlib.pyplot as plt
# from regu.Lcurve import Lcurve
# from regu.baart import baart
# from regu.csvd import csvd
# from regu.l_curve import l_curve
# from regu.tikhonov import tikhonov
# from regu.gcv import gcv
# from regu.discrep import discrep
# from numpy.linalg import norm
# from Utilities_functions.LocReg_unconstrained import LocReg_unconstrained
# from regu.gravity import gravity
# from tqdm import tqdm
# from Utilities_functions.LocReg_NE import LocReg_unconstrained_NE
# from Utilities_functions.pasha_gcv import Tikhonov
# from Simulations.Ito_LocReg import *
# from datetime import datetime
# from Utilities_functions.tikhonov_vec import tikhonov_vec

from utils.load_imports.load_classical import *
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

def minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g):
    OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
    OP_rhos = np.zeros((len(Alpha_vec)))
    for j in tqdm(range(len(Alpha_vec))):
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
# import scipy.io
# noise = scipy.io.loadmat('/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Matlab NRs/gravity_prob_NR.mat')
# #/Users/steveh/Downloads/noisearr1000for10diffnoisereal3.mat
# noise = noise['noise_arr']

def create_result_folder(string, SNR):
    # Create a folder based on the current date and time
    date = datetime.now().strftime("%Y%m%d")
    folder_name = f"/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/classical_problems/{string}_{date}_SNR_{SNR}"
    # folder_name = f"/Volumes/Lexar/NIH/Experiments/GridSearch/{string}_{date}_Run"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

n = 500
nT2 = n
T2 = np.linspace(-np.pi/2,np.pi/2,n)
TE = T2

G, data_noiseless, g = gravity(n, nargin=1)
U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 30
SD_noise= 1/SNR*max(abs(data_noiseless))
Lambda_vec = np.logspace(-5,3,40)
nLambda = len(Lambda_vec)

noise = np.random.normal(0,SD_noise, data_noiseless.shape)
#noise_arr[:,i] = noise.reshape(1,-1)
#data_noisy = data_noiseless + noise[:,i]
data_noisy = data_noiseless + noise
lambda_LC,rho,eta,_ = l_curve(U,s,data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
# f_rec_LC,_,_ = tikhonov(U,s,V,data_noisy,lambda_LC, nargin=5, nargout=1)
f_rec_LC, lambda_LC = tikhonov_vec(U, s, V, data_noisy, lambda_LC, x_0 = None, nargin = 5)

delta1 = norm(noise)*1.05
x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta1, x_0= None, nargin = 5)
# f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)
L = np.eye(G.shape[1])
x_true = None
f_rec_DP, lambda_DP = Tikhonov(G, data_noisy, L, x_true, regparam = 'DP', delta = delta1)
f_rec_DP, lambda_DP = tikhonov_vec(U, s, V, data_noisy, np.sqrt(lambda_DP), x_0 = None, nargin = 5)


L = np.eye(G.shape[1])
x_true = None
# f_rec_GCV, reg_min = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')
f_rec_GCV, lambda_GCV = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')
f_rec_GCV, lambda_GCV = tikhonov_vec(U, s, V, data_noisy, np.sqrt(lambda_GCV), x_0 = None, nargin = 5)

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

LR_mod_rec, fin_lam, c_array, _ , _ = LocReg_Ito_UC(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter)
LR_Ito_lams = fin_lam
    
Alpha_vec = np.logspace(-4,-1,250)
f_rec_OP_grid, oracle_lam = minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g)

# err_Lcurve = np.linalg.norm(g - f_rec_LC)
# # err_Ito2P = np.linalg.norm(g - best_f_rec)
# err_LR_Ito = np.linalg.norm(g - LR_mod_rec)
# # err_Chuan = np.linalg.norm(g - f_rec_Chuan)
# err_GCV = np.linalg.norm(g - f_rec_GCV)
err_DP = np.linalg.norm(g - f_rec_DP)
err_Lcurve = np.linalg.norm(g - f_rec_LC)
# err_Ito2P = np.linalg.norm(g - best_f_rec)
err_LR_Ito = np.linalg.norm(g - LR_mod_rec)
# err_Chuan = np.linalg.norm(g - f_rec_Chuan)
err_GCV = np.linalg.norm(g - f_rec_GCV)
err_oracle = np.linalg.norm(g - f_rec_OP_grid)

#Plot the curves
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
# plt.subplots_adjust(wspace=0.3)

# Plotting the first subplot
# plt.subplot(1, 3, 1)
ymax = np.max(g) * 1.15
axs[0, 0].plot(T2, g, color = "black",  label = "Ground Truth")
axs[0, 0].plot(T2, f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
axs[0, 0].plot(T2, LR_mod_rec, color = "blue",  label = "Ito NP")
axs[0, 0].plot(T2, f_rec_LC, color = "orange", label = "LC 1P")
# axs[0, 0].plot(T2, f_test, color = "pink", label = "test")
axs[0, 0].plot(T2, f_rec_GCV, color = "green", label = "GCV 1P")
axs[0, 0].plot(T2, f_rec_DP, color = "red", label = "DP 1P")

# axs[0, 0].plot(T2, f_rec_Chuan, color = "red", label = "Chuan")
axs[0, 0].set_xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
axs[0, 0].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
axs[0, 0].legend(fontsize=10, loc='best')
axs[0, 0].set_ylim(0, ymax)

# Plotting the second subplot
# plt.subplot(1, 3, 2)
axs[0, 1].plot(TE, G @ g, linewidth=3, color='black', label='Ground Truth')
axs[0, 1].plot(TE, G @ f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
axs[0, 1].plot(TE, G @ LR_mod_rec, color = "blue",  label = "Ito NP")
axs[0, 1].plot(TE, G @ f_rec_LC, color = "orange", label = "LC 1P")
axs[0, 1].plot(TE, G @ f_rec_GCV, color = "green", label = "GCV 1P")
axs[0, 1].plot(TE, G @ f_rec_DP, color = "red", label = "DP 1P")

# axs[0, 1].plot(TE, G @ f_rec_Chuan, color = "red", label = "Chuan")
axs[0, 1].legend(fontsize=10, loc='best')
axs[0, 1].set_xlabel('TE', fontsize=20, fontweight='bold')
axs[0, 1].set_ylabel('Intensity', fontsize=20, fontweight='bold')

# plt.subplot(1, 3, 3)
axs[1, 0].semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='orange', label='LC 1P')
axs[1, 0].semilogy(T2, oracle_lam * np.ones(len(T2)), linewidth=3, color='purple', label='Oracle 1P')
axs[1, 0].semilogy(T2, fin_lam, color = "blue", linewidth=3,  label = "Ito NP")
# axs[1,0].semilogy(T2, lambda_Chuan, color = "red",  label = "Chuan")
axs[1, 0].semilogy(T2, lambda_GCV * np.ones(len(T2)),linewidth=3,  color = "green", label = "GCV 1P")
axs[1, 0].semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color = "red", label = "DP 1P")
axs[1, 0].legend(fontsize=10, loc='best')
axs[1, 0].set_xlabel('T2', fontsize=20, fontweight='bold')
axs[1, 0].set_ylabel('Lambda', fontsize=20, fontweight='bold')
# ymax2 = 1.5 * np.max(lambda_LC)
# axs[1, 0].set_ylim(0, ymax2)

table_ax = axs[1, 1]
table_ax.axis('off')

# Define the data for the table (This is part of the plot)
data = [
    ["L-Curve Lambda", lambda_LC.item()],
    ["Initial Lambda for Ito", LRIto_ini_lam],
    # ["Initial Eta2 for Ito", round(lam_ini, 4)],
    # ["Initial Eta2 for Ito", LRIto_ini_lam],
    # ["Final Eta1 for Ito", fin_etas[0].item()],
    # ["Final Eta2 for Ito", fin_etas[1].item()],
    ["error Conventional DP", err_DP.item()],
    ["error Conventional L-Curve", err_Lcurve.item()],
    ["error Conventional GCV", err_GCV.item()],
    ["error Ito NP", err_LR_Ito.item()],
    # ["error Ito 2P", err_Ito2P.item()],
    ["error Oracle 1P", err_oracle.item()],
    # ["error test", err_test.item()],
    # ["error Chuan", err_Chuan.item()],
    ["SNR", SNR]

    # ["Initial Lambdas for Ito Loc", LR_ini_lam],
    # ["Final Lambdas for Ito Loc", LR_Ito_lams]
]

# Create the table
table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

#Save the results in the save results folder
plt.tight_layout()
string = "gravity"
file_path = create_result_folder(string, SNR)
plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve"))
print(f"Saving comparison plot is complete")



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
lam_LocReg_ne = np.zeros((n,nrun))
lam_DP = np.zeros((n,nrun))

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
    ep1 = 1e-2
    # % 1/(|x|+ep1)
    ep2 = 1e-1
    # % norm(dx)/norm(x)
    ep3 = 1e-2
    # % norm(x_(k-1) - x_k)/norm(x_(k-1))
    ep4 = 1e-4 
    # % lb for ep1
    f_rec_locreg, lambda_locreg = LocReg_unconstrained_NE(data_noisy, G, x0_ini, reg_corner, ep1, ep2, ep3)
    lam_LocReg_ne[:,i] = lambda_locreg
    com_vec_locreg_ne[i] = norm(g - f_rec_locreg)
    res_vec_locreg_ne[:,i] = f_rec_locreg

    f_rec_locreg, lambda_locreg = LocReg_unconstrained(data_noisy, G, x0_ini, reg_corner, ep1, ep2, ep3)
    lam_LocReg[:,i] = lambda_locreg
    com_vec_locreg[i] = norm(g - f_rec_locreg)
    res_vec_locreg[:,i] = f_rec_locreg

print('The mean error for DP is', str(np.median(com_vec_DP)))
print('The mean error for L-curve is', str(np.median(com_vec_LC)))
print('The mean error for GCV is', str(np.median(com_vec_GCV)))
print('The mean error for locreg is', str(np.median(com_vec_locreg)))

# threshold = 3 * np.median(com_vec_GCV)  # Adjust as needed

# # Identify outliers by comparing with the threshold
# outliers = com_vec_GCV > threshold

# # Replace outliers with NaN (Not a Number) or any other value of your choice
# com_vec_GCV[outliers] = np.median(com_vec_GCV)

print('The SD for DP is', str(np.std(com_vec_DP)))
print('The SD for L-curve is', str(np.std(com_vec_LC)))
print('The SD for GCV is', str(np.std(com_vec_GCV)))
print('The SD for locreg is', str(np.std(com_vec_locreg)))
print('The SD for locreg is', str(np.std(com_vec_locreg_ne)))

vecs = np.asarray([com_vec_DP, com_vec_LC, com_vec_GCV, com_vec_locreg, com_vec_locreg_ne])
np.savetxt('gravity_vec_1121.csv', vecs , delimiter=',')
# np.savetxt('gravity_vec_1023.csv', vecs , delimiter=',')
# np.savetxt('gravity_noise_arr_1000', noise_arr)


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
# FileName = f"gravity_info_09_11_23_{nrun}NR"

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
plt.savefig(f'gravity_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format
plt.show()

name = "gcv_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'gravity_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format
plt.show()
name = "dp_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_DP, linewidth=3, color='brown', label='DP')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'gravity_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format
plt.show()

name = "lc_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'gravity_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format
plt.show()

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
plt.savefig(f'gravity_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format
plt.show()

############################################











#Print graph
plt.figure(figsize=(40, 20))
plt.subplots_adjust(wspace=0.3)

plt.subplot(1, 3, 1)
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_locreg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
plt.plot(T2, res_vec_DP, linewidth=3, color='brown', label='DP')
plt.plot(T2, res_vec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
plt.plot(T2, res_vec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
# plt.plot(T2, f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
# get legend handles and their corresponding labels
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')

# Plotting the second subplot
plt.subplot(1, 3, 2)
plt.plot(TE, G @ g, linewidth=3, color='black', label='True Dist.')
plt.plot(TE, G @ res_vec_locreg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
plt.plot(TE, G @ res_vec_DP, linewidth=3, color='brown', label='DP')
plt.plot(TE, G @ res_vec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
plt.plot(TE, G @ res_vec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
#plt.plot(TE, G @ f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
plt.legend(fontsize='small', loc='best')
plt.xlabel('TE', fontsize=20, fontweight='bold')
plt.ylabel('Intensity', fontsize=20, fontweight='bold')

plt.subplot(1,3,3)
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
plt.subplots_adjust(bottom=0.15, top=0.85)
plt.savefig(f'gravity_prob_nrun{nrun}_09_11_23.png')  # You can specify the desired filename and format
plt.show()
"""