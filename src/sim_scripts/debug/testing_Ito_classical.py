# #Packages
# import sys
# sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
# import numpy as np
# import cvxpy as cp
# import os
# import scipy
# import matplotlib.pyplot as plt
# from scipy.linalg import svd
# from scipy.optimize import nnls
# # from lsqnonneg import lsqnonneg
# #from Simulations.lcurve_functions import l_cuve,csvd,l_corner
# from Simulations.l_curve_corner import l_curve_corner
# from regu.csvd import csvd
# from regu.discrep import discrep
# from Simulations.Ito_LocReg import Ito_LocReg
# from Simulations.Ito_LocReg import blur_ito, grav_ito
# from Simulations.Ito_LocReg import LocReg_Ito_mod
# from Utilities_functions.LocReg import LocReg as Chuan_LR
# from regu.ito_blur import blur
# from regu.ito_gravity import gravity


# from regu.tikhonov import tikhonov
# from regu.l_curve import l_curve
# from Utilities_functions.Lcurve import Lcurve
# from datetime import datetime
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

def create_result_folder(string):
    # Create a folder based on the current date and time
    date = datetime.now().strftime("%Y%m%d")
    folder_name = f"/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito/{string}_{date}_Run"
    # folder_name = f"/Volumes/Lexar/NIH/Experiments/GridSearch/{string}_{date}_Run"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return folder_name

def generate_B_matrices(N, M):
    B_matrices = []

    if M % N != 0:
        ValueError("M must be divisible by N")
    ones_per_matrix = M // N  # Calculate number of ones per matrix

    for i in range(N):
        start = ones_per_matrix * i  # Calculate the starting index for ones in this matrix
        end = min(ones_per_matrix * (i + 1), M)  # Calculate the ending index for ones in this matrix
        diagonal_values = np.zeros(M)
        diagonal_values[start:end] = 1  # Set ones in the appropriate range
        B_matrices.append(np.diag(diagonal_values))

    return B_matrices

#Generate the TE values/ time
TE = np.arange(1,512,4).T
#Generate the T2 values
T2 = np.arange(1,201).T
#Generate G_matrix
G = np.zeros((len(TE),len(T2)))
#For every column in each row, fill in the e^(-TE(i))
for i in range(len(TE)):
    for j in range(len(T2)):
        G[i,j] = np.exp(-TE[i]/T2[j])
nTE = len(TE)
nT2 = len(T2)
sigma1 = 3
mu1 = 40
sigma2 = 10
mu2 = 160

#Create ground truth
g = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
g = g + (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
g = g/2

#Generate Penalty Matrices:

N = 2
M = nT2
B_matrices = generate_B_matrices(N, M)

#DP
data_noiseless = np.dot(G, g)
Jacobian = np.dot(G.T, G)

U, s, V = csvd(G, tst = None, nargin = 1, nargout = 3)

SNR = 200
SD_noise = 1 / SNR
noise = np.random.normal(0, SD_noise, size=data_noiseless.shape)
data_noisy = data_noiseless + noise

maxiter = 50
blur_n = 50
gamma_init = 1
LRIto_ini_lam = 1e-3

GravIto_ini_lam = LRIto_ini_lam
grav_n = 1000

n = grav_n
t = np.linspace(-np.pi/2,np.pi/2,n**2)
t = np.linspace(-np.pi/2,np.pi/2,n)
# G, data_noiseless, g = blur(N = n)
# G = G.toarray()

G, data_noiseless, g = gravity(n, nargin=1)
# U,s,V = csvd(G, tst = None, nargin = 1, nargout = 3)
SNR = 200
SD_noise = 1 / SNR
noise = np.random.normal(0, SD_noise, size=data_noiseless.shape)
noise = 5e-2
data_noisy = data_noiseless + noise
# blur_rec, fin_lam, c_array, lam_arr_fin, sol_arr_fin = blur_ito(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter)
blur_rec, fin_lam, c_array, lam_arr_fin, sol_arr_fin = grav_ito(data_noisy, G, GravIto_ini_lam, gamma_init, maxiter)

# has_negative_values = np.any(blur_rec < 0)

# if has_negative_values:
#     print("blur_rec has negative values.")
# else:
#     print("blur_rec does not have any negative values.")

# blur_rec[blur_rec < 0] = 0
err_ItoBlur = 1/n * np.linalg.norm(g - blur_rec, 2)

fig, axs = plt.subplots(2, 2, figsize=(6, 6))
# plt.subplots_adjust(wspace=0.3)

# Plotting the first subplot
# plt.subplot(1, 3, 1)
ymax = np.max(g) * 1.15
axs[0, 0].plot(t, g, color = "black",  label = "Ground Truth")
axs[0, 0].plot(t, blur_rec, color = "purple",  label = "Ito Gravity")
axs[0, 0].set_xlabel('T', fontsize=20, fontweight='bold')
axs[0, 0].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
axs[0, 0].legend(fontsize=10, loc='best')
axs[0, 0].set_ylim(0, ymax)

# Plotting the second subplot
# plt.subplot(1, 3, 2)
# axs[0, 1].plot(t, G @ g, linewidth=3, color='black', label='Ground Truth')
# # plt.plot(TE, A @ f_rec_LocReg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
# axs[0, 1].plot(t, G @ blur_rec, color = "purple",  label = "Ito Blur")

# # plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
# # plt.plot(TE, A @ f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
# axs[0, 1].legend(fontsize=10, loc='best')
# axs[0, 1].set_xlabel('t', fontsize=20, fontweight='bold')
# axs[0, 1].set_ylabel('Intensity', fontsize=20, fontweight='bold')

# plt.subplot(1, 3, 3)
# axs[1, 0].semilogy(t, fin_lam * np.ones(len(t)), linewidth=3, color='purple', label='Ito Blur')
# axs[1, 0].legend(fontsize=10, loc='best')
# axs[1, 0].set_xlabel('T2', fontsize=20, fontweight='bold')
# axs[1, 0].set_ylabel('Lambda', fontsize=20, fontweight='bold')

table_ax = axs[1, 1]
table_ax.axis('off')

# Define the data for the table
data = [
    ["Initial Eta for Ito Gravity", GravIto_ini_lam],
    ["Final Eta for Ito Gravity", fin_lam],
    ["error Ito Gravity", err_ItoBlur]
    # ["Initial Lambdas for Ito Loc", LR_ini_lam],
    # ["Final Lambdas for Ito Loc", LR_Ito_lams]
]

# Create the table
table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

plt.tight_layout()
string = "comparison"
file_path = create_result_folder(string)
plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve"))
print(f"Saving comparison plot is complete")
plt.show()
plt.close()

# sol_arr_fin[6] - sol_arr_fin[7] 0.001182667585284297
