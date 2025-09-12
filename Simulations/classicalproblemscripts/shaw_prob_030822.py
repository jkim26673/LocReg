import numpy as np
import sys
sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
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
from Utilities_functions.TwoParam_LR import Multi_Param_LR
from regu.shaw import shaw
from regu.tikhonov_multi_param import tikhonov_multi_param
from tqdm import tqdm
import scipy.io
noise = scipy.io.loadmat('/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Matlab NRs/shaw_NR.mat')
#/Users/steveh/Downloads/noisearr1000for10diffnoisereal3.mat
noise = noise['noise_arr']

n = 1000
nT2 = n
T2 = np.linspace(-np.pi/2,np.pi/2,n)
TE = T2

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
Lambda_vec = np.logspace(-5,3,40)
nLambda = len(Lambda_vec)


G, data_noiseless, g = shaw(n, nargout=3)

U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 100
SD_noise= 1/SNR*max(abs(data_noiseless))

nrun = 100
com_vec_DP = np.zeros(nrun)
com_vec_GCV = np.zeros(nrun)
com_vec_LC = np.zeros(nrun)
com_vec_locreg = np.zeros(nrun)
# com_vec_MP =np.zeros(nrun)
# com_vec_MP_DP =np.zeros(nrun)
# com_vec_MP_LC =np.zeros(nrun)
# com_vec_MP_GCV =np.zeros(nrun)

res_vec_DP = np.zeros((n,nrun))
res_vec_GCV = np.zeros((n,nrun))
res_vec_LC = np.zeros((n,nrun))
res_vec_locreg = np.zeros((n,nrun))
# res_vec_MP = np.zeros((n,nrun))
# res_vec_MP_DP = np.zeros((n,nrun))
# res_vec_MP_GCV = np.zeros((n,nrun))
# res_vec_MP_LC = np.zeros((n,nrun))

noise_size = int(data_noiseless.shape[0])
noise_arr = np.zeros((noise_size, nrun))
lam_LC = np.zeros((n,nrun))
lam_GCV = np.zeros((n,nrun))
lam_LocReg = np.zeros((n,nrun))
lam_DP = np.zeros((n,nrun))
# lam_MP_1 = np.zeros((n,nrun))
# lam_MP_2 = np.zeros((n,nrun))
# lam_MP_1_DP = np.zeros((n,nrun))
# lam_MP_2_DP = np.zeros((n,nrun))
# lam_MP_1_GCV = np.zeros((n,nrun))
# lam_MP_2_GCV = np.zeros((n,nrun))
# lam_MP_1_LC = np.zeros((n,nrun))
# lam_MP_2_LC = np.zeros((n,nrun))

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
    print("LC completed")

    # %% GCV
    reg_min,_,reg_param = gcv(U,s,data_noisy, method = 'Tikh', nargin = 3, nargout = 3)
    lam_GCV[:,i] = reg_min
    f_rec_GCV,_,_ = tikhonov(U,s,V,data_noisy,reg_min, nargin=5, nargout=1)
    com_vec_GCV[i] = norm(g - f_rec_GCV)
    res_vec_GCV[:,i] = f_rec_GCV
    print("GCV completed")

    # %% DP
    #delta = norm(noise[:,i])*1.05
    delta = norm(noise)*1.05
    x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta, x_0= None, nargin = 5)
    lam_DP[:,i] = lambda_DP
    f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)
    com_vec_DP[i] = norm(g - f_rec_DP)
    res_vec_DP[:,i] = f_rec_DP
    
    print("DP completed")

    # %% locreg
    x0_ini = f_rec_LC
    # %     lambda_ini = reg_corner;
    ep1 = 1e-2
    # % 1/(|x|+ep1)
    ep2 = 1e-2
    # % norm(dx)/norm(x)
    ep3 = 1e-3
    # % norm(x_(k-1) - x_k)/norm(x_(k-1))
    ep4 = 1e-4 
    # % lb for ep1

    f_rec_locreg, lambda_locreg = LocReg_unconstrained(data_noisy, G, x0_ini, lambda_DP, ep1, ep2, ep3)
    lam_LocReg[:,i] = lambda_locreg
    com_vec_locreg[i] = norm(g - f_rec_locreg)
    res_vec_locreg[:,i] = f_rec_locreg

    print("LocReg completed")
    
    # # delta = 1.05 * norm(noise)
    # alpha_1 = 0.2
    # alpha_2 = 0.2
    # lambda1, lambda2, f_rec_MP = Multi_Param_LR(data_noisy, G, L_1, L_2, alpha_1, alpha_2, ep, c, delta)
    # lam_MP_1[:,i] = lambda1
    # lam_MP_2[:,i] = lambda2
    # com_vec_MP[i] = norm(g - f_rec_MP)
    # res_vec_MP[:,i] = f_rec_MP

    # alpha_1 = lambda_DP
    # alpha_2 = lambda_DP
    # lambda1, lambda2, f_rec_MP = Multi_Param_LR(data_noisy, G, L_1, L_2, alpha_1, alpha_2, ep, c, delta)
    # lam_MP_1_DP[:,i] = lambda1
    # lam_MP_2_DP[:,i] = lambda2
    # com_vec_MP_DP[i] = norm(g - f_rec_MP)
    # res_vec_MP_DP[:,i] = f_rec_MP

    # alpha_1 = reg_min
    # alpha_2 = reg_min
    # lambda1, lambda2, f_rec_MP = Multi_Param_LR(data_noisy, G, L_1, L_2, alpha_1, alpha_2, ep, c, delta)
    # lam_MP_1_GCV[:,i] = lambda1
    # lam_MP_2_GCV[:,i] = lambda2
    # com_vec_MP_GCV[i] = norm(g - f_rec_MP)
    # res_vec_MP_GCV[:,i] = f_rec_MP

    # alpha_1 = reg_corner
    # alpha_2 = reg_corner
    # lambda1, lambda2, f_rec_MP = Multi_Param_LR(data_noisy, G, L_1, L_2, alpha_1, alpha_2, ep, c, delta)
    # lam_MP_1_LC[:,i] = lambda1
    # lam_MP_2_LC[:,i] = lambda2
    # com_vec_MP_LC[i] = norm(g - f_rec_MP)
    # res_vec_MP_LC[:,i] = f_rec_MP
    # print("MultiReg completed")

print('The median error for DP is', str(np.median(com_vec_DP)))
print('The median error for L-curve is', str(np.median(com_vec_LC)))
print('The median error for GCV is', str(np.median(com_vec_GCV)))
print('The median error for locreg is', str(np.median(com_vec_locreg)))
# print('The median error for Multi_Reg is', str(np.median(com_vec_MP)))

threshold = 3 * np.median(com_vec_GCV)  # Adjust as needed

# Identify outliers by comparing with the threshold
outliers = com_vec_GCV > threshold

# Replace outliers with NaN (Not a Number) or any other value of your choice
com_vec_GCV[outliers] = np.median(com_vec_GCV)

print('The SD for DP is', str(np.std(com_vec_DP)))
print('The SD for L-curve is', str(np.std(com_vec_LC)))
print('The SD for GCV is', str(np.std(com_vec_GCV)))
print('The SD for locreg is', str(np.std(com_vec_locreg)))
# print('The SD for MultiReg is', str(np.std(com_vec_MP)))

# plt.plot(g)
# plt.plot(f_rec_MP)
# plt.legend(["Ground Truth","Multi_Reg"])
# plt.show()

vecs = np.asarray([com_vec_DP, com_vec_LC, com_vec_GCV, com_vec_locreg])
np.savetxt('shaw_vec_1023.csv', vecs , delimiter=',')
np.savetxt('shaw_noise_arr_1000', noise_arr)


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
plt.savefig(f'shaw_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format

name = "gcv_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'shaw_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format

name = "dp_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_DP, linewidth=3, color='brown', label='DP')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'shaw_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format

name = "lc_gt"
plt.figure(figsize=(12, 6))
plt.plot(T2, g, linewidth=3, color='black', label='True Dist.')
plt.plot(T2, res_vec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
plt.legend(fontsize='small', loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
plt.savefig(f'shaw_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format

#Plot the lambda plots
name="lambda_plt"
plt.figure(figsize=(12, 6))
plt.semilogy(T2, lam_LocReg * np.ones(len(T2)), linestyle=':', linewidth=3, color='magenta', label='LocReg')
plt.semilogy(T2, lam_DP * np.ones(len(T2)),linewidth=3, color='brown', label='DP')
plt.semilogy(T2, lam_GCV * np.ones(len(T2)), linestyle='--', linewidth=3, color='blue', label='GCV')
plt.semilogy(T2, lam_LC * np.ones(len(T2)), linestyle='-.',linewidth=3, color='cyan', label='L-curve')
handles, labels = plt.gca().get_legend_handles_labels()
dict_of_labels = dict(zip(labels, handles))
plt.legend(dict_of_labels.values(), dict_of_labels.keys(), fontsize=10, loc='best')
#plt.legend(fontsize='small', loc='upper left')
#plt.legend(['DP', 'GCV', 'L-curve', 'LocReg'], fontsize=20, loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Lambda', fontsize=20, fontweight='bold')
plt.savefig(f'shaw_prob_nrun_{nrun}_{name}.png')  # You can specify the desired filename and format

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
plt.savefig(f'shaw_prob_nrun{nrun}_09_11_23.png')  # You can specify the desired filename and format
plt.show()
