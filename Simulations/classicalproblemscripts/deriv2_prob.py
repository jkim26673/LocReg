import sys
sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
import numpy as np
import os
import pickle
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
from regu.deriv2 import deriv2
from tqdm import tqdm
from Utilities_functions.LocReg_v2 import LocReg_v2
from Utilities_functions.LocReg_NE import LocReg_unconstrained_NE

import scipy.io
noise = scipy.io.loadmat('/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Matlab NRs/deriv2_prob_NR.mat')
#/Users/steveh/Downloads/noisearr1000for10diffnoisereal3.mat
noise = noise['noise_arr']

n = 1000
nT2 = n
T2 = np.linspace(-np.pi/2,np.pi/2,n)
TE = T2

G, data_noiseless, g = deriv2(n)

U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 300
SD_noise= 1/SNR*max(abs(data_noiseless))
Lambda_vec = np.logspace(-10,5,40)
nLambda = len(Lambda_vec)
nT2 = len(T2)
half_nT2 = int(len(T2)/2)
half_mat = np.eye(half_nT2)
L_1 = np.zeros((nT2, nT2))
L_2 = np.zeros((nT2, nT2))
L_1[:half_nT2, :half_nT2] = half_mat
L_2[half_nT2:, half_nT2:] = half_mat
c = 1
# delta2 = 0.1
k = 0
ep = 1e-4

nrun = 100
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

for i in range(nrun):
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
    x0_ini = f_rec_DP
    # %     lambda_ini = reg_corner;
    ep1 = 1e-2
    # % 1/(|x|+ep1)
    ep2 = 1e-1
    # % norm(dx)/norm(x)
    ep3 = 1e-2
    # % norm(x_(k-1) - x_k)/norm(x_(k-1))
    ep4 = 1e-4 
    # % lb for ep1
    f_rec_locreg, lambda_locreg = LocReg_unconstrained_NE(data_noisy, G, x0_ini, lambda_DP, ep1, ep2, ep3)
    lam_LocReg_ne[:,i] = lambda_locreg
    com_vec_locreg_ne[i] = norm(g - f_rec_locreg)
    res_vec_locreg_ne[:,i] = f_rec_locreg

    f_rec_locreg, lambda_locreg = LocReg_unconstrained(data_noisy, G, x0_ini, lambda_DP, ep1, ep2, ep3)
    lam_LocReg[:,i] = lambda_locreg
    com_vec_locreg[i] = norm(g - f_rec_locreg)
    res_vec_locreg[:,i] = f_rec_locreg

print('The mean error for DP is', str(np.mean(com_vec_DP)))
print('The mean error for L-curve is', str(np.mean(com_vec_LC)))
print('The mean error for GCV is', str(np.mean(com_vec_GCV)))
print('The mean error for locreg is', str(np.mean(com_vec_locreg)))
print('The mean error for locreg_ne is', str(np.mean(com_vec_locreg_ne)))

# threshold = 3 * np.median(com_vec_GCV)  # Adjust as needed

# # Identify outliers by comparing with the threshold
# outliers = com_vec_GCV > threshold

# # Replace outliers with NaN (Not a Number) or any other value of your choice
# com_vec_GCV[outliers] = np.median(com_vec_GCV)

print('The SD for DP is', str(np.std(com_vec_DP)))
print('The SD for L-curve is', str(np.std(com_vec_LC)))
print('The SD for GCV is', str(np.std(com_vec_GCV)))
print('The SD for locreg is', str(np.std(com_vec_locreg)))
print('The SD for locreg_ne is', str(np.std(com_vec_locreg_ne)))

vecs = np.asarray([com_vec_DP, com_vec_LC, com_vec_GCV, com_vec_locreg, com_vec_locreg_ne])
# np.savetxt('deriv2_vec_1023.csv', vecs , delimiter=',')
np.savetxt('deriv2_vec_1121.csv', vecs , delimiter=',')

# np.savetxt('deriv2_noise_arr_1000_09_28', noise_arr)

rid_GCV = np.where(com_vec_GCV >= 100)[0]
res_vec_GCV = np.delete(res_vec_GCV, rid_GCV, axis=1)

#Save files
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
# FileName = f"deriv2_info_09_11_23_{nrun}NR"

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
plt.savefig(f'deriv2_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format
plt.show()

name = "gcv_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'deriv2_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format
plt.show()

name = "dp_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_DP, linewidth=3, color='brown', label='DP')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'deriv2_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format
plt.show()

name = "lc_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'deriv2_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format
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
plt.savefig(f'deriv2_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format
plt.show()

############################################

















#Print graph
plt.figure(figsize=(40, 20))
plt.subplots_adjust(wspace=0.3)

plt.subplot(1, 3, 1)
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_locreg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
plt.plot(T2, res_vec_locreg_ne, linestyle=':', linewidth=3, color='green', label='LocReg_New')
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
#plt.plot(TE, G @ f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
plt.legend(fontsize='small', loc='best')
plt.xlabel('TE', fontsize=20, fontweight='bold')
plt.ylabel('Intensity', fontsize=20, fontweight='bold')

plt.subplot(1,3,3)
plt.semilogy(T2, lam_LocReg * np.ones(len(T2)), linestyle=':', linewidth=3, color='magenta', label='LocReg')
plt.semilogy(T2, lam_LocReg_ne * np.ones(len(T2)), linestyle=':', linewidth=3, color='green', label='LocReg_New')
plt.semilogy(T2, lam_DP * np.ones(len(T2)),linewidth=3, color='brown', label='DP')
plt.semilogy(T2, lam_GCV * np.ones(len(T2)), linestyle='--', linewidth=3, color='blue', label='GCV')
plt.semilogy(T2, lam_LC * np.ones(len(T2)), linestyle='-.',linewidth=3, color='cyan', label='L-curve')
handles, labels = plt.gca().get_legend_handles_labels()
dict_of_labels = dict(zip(labels, handles))
plt.legend(dict_of_labels.values(), dict_of_labels.keys(), fontsize='small', loc='best')
#plt.legend(fontsize=10, loc='upper left')
#plt.legend(['DP', 'GCV', 'L-curve', 'LocReg'], fontsize=20, loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Lambda', fontsize=20, fontweight='bold')

plt.show()
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
plt.savefig(f'deriv2_prob_nrun{nrun}_09_11_23.png')  # You can specify the desired filename and format
plt.show()
