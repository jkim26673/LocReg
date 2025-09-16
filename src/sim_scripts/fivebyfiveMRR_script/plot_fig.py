# import matplotlib.pyplot as plt
# import numpy as np
# import sys
# sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
# from numpy.linalg import norm
# from regu.csvd import csvd
# from regu.tikhonov import tikhonov
# from regu.discrep import discrep
# from regu.l_curve import l_curve
# from Utilities_functions.discrep_L2 import discrep_L2
# from scipy.optimize import nnls
# from Utilities_functions.GCV_NNLS import GCV_NNLS
# from regu.gcv import gcv
# from Simulations.l_curve_corner import l_curve_corner
from utils.load_imports.loading import *

#Get Example Voxel (same size)
#Animated Regularization Two Peak
#Better

TE = np.arange(1,512,4).T
#Generate the T2 values
T2 = np.arange(1,201).T
G_mat = np.zeros((len(TE),len(T2)))
for i in range(len(TE)):
    for j in range(len(T2)):
        G_mat[i,j] = np.exp(-TE[i]/T2[j])

sigma1 = 2
mu1 = 40
sigma2 = 6
mu2 = 120
sigma3 = 1
mu3 = 150
sigma4 = 5
mu4 = 75

# mu3 = 60
# sigma3 = 3

g1 = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
g2 = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
g3 = (1 / (np.sqrt(2 * np.pi) * sigma3)) * np.exp(-((T2 - mu3) ** 2) / (2 * sigma3 ** 2))
g4 = (1 / (np.sqrt(2 * np.pi) * sigma4)) * np.exp(-((T2 - mu4) ** 2) / (2 * sigma4 ** 2))

g  = (1/4) * (g1 + g2 + g3 + g4)

# g  = (1/2) * (g1 + g2)

SNR = 300

data_noiseless = np.dot(G_mat, g)
SD_noise= 1/SNR*max(abs(data_noiseless))
noise = np.random.normal(0,SD_noise, data_noiseless.shape)
data_noisy = data_noiseless + noise


#####


U, s, V = csvd(G_mat, tst = None, nargin = 1, nargout = 3)

delta = norm(noise)*1.05 
# delta = norm(noise)**2
x_delta,lambda_OP = discrep(U,s,V,data_noisy,delta, x_0= None, nargin = 5)
lam_DP_OP = lambda_OP
Lambda = np.ones(len(T2))* lambda_OP

f_rec_DP_OP,_ = discrep_L2(data_noisy, G_mat, SNR, Lambda)
f_rec_DP_OP = f_rec_DP_OP.flatten()

reg_min,_,_ = gcv(U,s,data_noisy, method = 'Tikh', nargin = 3, nargout = 3)
lam_GCV_OP = reg_min
# f_rec_GCV_OP,_,_ = tikhonov(U,s,V,data_noisy,reg_min, nargin=5, nargout=1)
# f_rec_GCV_OP = f_rec_GCV_OP
Lambda = np.ones(len(T2))* reg_min
f_rec,_ = GCV_NNLS(data_noisy, G_mat, Lambda)
f_rec_GCV_OP = f_rec[:,0]

no_reg,_ = nnls(G_mat,data_noisy)

import cvxpy as cp
import os
import mosek
import cvxpy as cp
mosek_lic_path = "/Users/steveh/Downloads/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_lic_path
# nT2 = len(T2)
# Lambda_vec = np.logspace(-3, 2, 10)
# x_lc_vec = np.zeros((len(T2), len(Lambda_vec)))
# rhos = np.zeros(len(Lambda_vec)).T
# etas = np.zeros(len(Lambda_vec)).T
# for j in range(len(Lambda_vec)):
#     #x_lc_vec[:, j] = lsqnonneg((G.T @ G) +  (Lambda_vec[j]**2) * np.eye(nT2), G.T @ data_noisy, tol = 1e-3)[0]
#     x_lc_vec[:, j] = nnls((G_mat.T @ G_mat) +  (Lambda_vec[j]) * np.eye(nT2), G_mat.T @ data_noisy)[0]
#     rhos[j] = np.linalg.norm(data_noisy - G_mat @ x_lc_vec[:, j]) ** 2
#     etas[j] = Lambda_vec[j] * np.linalg.norm(x_lc_vec[:, j]) ** 2
# reg_corner, ireg_corner, b = l_curve_corner(rhos,etas,Lambda_vec)
# x0_ini = x_lc_vec[:, ireg_corner]
#1st plot

file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/figure_data/LCIRetreatFigs"
plt.plot(TE, data_noisy, 'bo',label = "Noisy Data")
plt.plot(TE, data_noiseless, 'k--', label = "Noiseless Data")
plt.title("Voxel Data from Multi-Echo MRI Sequence", fontweight = "bold")
plt.xlabel("Echo Acquisition Time (ms)", fontsize = 12)
plt.ylabel("Signal Amplitude", fontsize = 12)
plt.legend()
# plt.savefig(os.path.join(file_path, f"Fig1.png"))
plt.show()
plt.close()

#2nd plot
plt.plot(T2, g, 'k--', label = "Ground Truth")
plt.title("Recovery Comparison", fontweight = "bold")
plt.xlabel("T2 Relaxation Time (ms)", fontsize = 12)
plt.ylabel("Signal Amplitude", fontsize = 12)
plt.legend()
# plt.savefig(os.path.join(file_path, f"Fig2.png"))
plt.show()
plt.close()

plt.plot(T2, g, 'k--', label = "Ground Truth")
plt.plot(T2, no_reg, label = "No Regularization")
plt.title("Recovery Comparison", fontweight = "bold")
plt.xlabel("T2 Relaxation Time (ms)", fontsize = 12)
plt.ylabel("Signal Amplitude", fontsize = 12)
plt.legend()
# plt.savefig(os.path.join(file_path, f"Fig3.png"))
plt.show()
plt.close()

plt.plot(T2, g, 'k--', label = "Ground Truth")
plt.plot(T2, no_reg, label = "No Regularization")
plt.plot(T2, f_rec_GCV_OP, label = "Regularization")
plt.title("Recovery Comparison", fontweight = "bold")
plt.xlabel("T2 Relaxation Time (ms)", fontsize = 12)
plt.ylabel("Signal Amplitude", fontsize = 12)
plt.legend()
# plt.savefig(os.path.join(file_path, f"Fig4.png"))
plt.show()
plt.close()


plt.plot(T2, g, 'k--', label = "Ground Truth")
plt.title("Unknown Distribution Function", fontweight = "bold")
plt.xlabel("T2 Relaxation Time (ms)", fontsize = 12)
plt.ylabel("Signal Amplitude", fontsize = 12)
plt.legend()
plt.savefig(os.path.join(file_path, f"Fig5.png"))
plt.show()
plt.close()