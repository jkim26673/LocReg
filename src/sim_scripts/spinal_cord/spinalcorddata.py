# import sys
# import os
# print("Setting system path")
# sys.path.append(".")  # Replace this path with the actual path to the parent directory of Utilities_functions
# import numpy as np
# from scipy.stats import norm as normsci
# from scipy.linalg import norm as linalg_norm
# from scipy.optimize import nnls
# import matplotlib.pyplot as plt
# import pickle
# from Utilities_functions.discrep_L2 import discrep_L2, discrep_L2_brain, discrep_L2_sp
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
# from Simulations.resolutionanalysis import find_min_between_peaks, check_resolution
# import logging
# import time
# from scipy.stats import wasserstein_distance
# import matplotlib.ticker as ticker  # Add this import
# import scipy
# import timeit
# import unittest
# import xlrd
from utils.load_imports.loading import *

#Hyperparameters LocReg
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
exp = 0.5
feedback = True
lam_ini_val = "LCurve"
gamma_init = 0.5
maxiter = 500

#CVXPY global parameters
eps = 1e-2

#Error Metric
err_type = "Wass. Score"

# Load the data
spinalcord_data = "/home/kimjosy/LocReg_Regularization-1/data/spinalcord/SPT2_MiceSpinalCord.xls"
df = pd.DataFrame(pd.read_excel(spinalcord_data))
df = df.drop(columns=["Unnamed: 0"])
study1signal = np.array(df["Study1"].values)
study2signal = np.array(df["Study2"].values)
print("study1signal", study1signal)
print("study2signal", study2signal) 

#Parameters
dTE = 11.3
# n = 32
n = len(study1signal)
# TE = dTE * np.linspace(1,n,n)
TE = np.array(df["TE(ms)"].values)
m = 150
T2 = np.linspace(10,400,m)
A= np.zeros((n,m))
dT = T2[1] - T2[0]

# Define the tail end length
tail_length = 50  # Example length, adjust as needed

# Get the tail end of the signals
study1_tail = study1signal[-tail_length:]
study2_tail = study2signal[-tail_length:]

# Calculate the standard deviation of the tail ends
study1_tail_std = np.std(study1_tail)
study2_tail_std = np.std(study2_tail)

print("Study1 Tail Standard Deviation:", study1_tail_std)
print("Study2 Tail Standard Deviation:", study2_tail_std)

plt.figure()
plt.plot(TE, study1signal, label="Study 1 Signal")
plt.plot(TE, study2signal, label="Study 2 Signal")
plt.legend()
plt.title("Study 1 and Study 2 Signals")
plt.savefig("/home/kimjosy/LocReg_Regularization-1/data/spinalcord/results/Study1andstudy2test.png")
print("study2signal.shape", study2signal.shape) 
print("study1signal.shape", study1signal.shape) 


#Lambda Space
Lambda = np.logspace(-6,1,50).reshape(-1,1)


#Kernel

def kernel(n,m,TE,T2):
    for i in range(n):
        for j in range(m):
            A[i,j] = np.exp(-TE[i]/T2[j]) * dT
    return A

#check spinal code paper for reference for what most people are interested in MR relaxation time
# T2 times



# class SpinalCord_Data:
#     def __init__(self, signal, curr_SNR, TE, T2, Lambda, A):
#         self.TE = TE
#         self.Lambda = Lambda
#         self.noisy_signal = signal
#         self.T2 = T2
#         self.A = A
#         self.SNR = curr_SNR
    
#     def LCurve(self):
#         print('self.noisy_signal', self.noisy_signal.dtype)
#         print('self.A', self.A.dtype)
#         print('self.Lambda', self.Lambda.dtype)
#         f_rec_LC, lambda_LC = Lcurve(self.noisy_signal, self.A, self.Lambda)
#         self.f_rec_LC = f_rec_LC
#         self.lambda_LC = lambda_LC

#     def LS(self):
#         f_rec_LS = nnls(self.A, self.noisy_signal)[0]
#         self.f_rec_LS = f_rec_LS

#     def GCV(self):
#         print(self.noisy_signal.ddtype)
#         f_rec_GCV, lambda_GCV = GCV_NNLS(self.noisy_signal, self.A, self.Lambda)
#         f_rec_GCV = f_rec_GCV[:,0]
#         lambda_GCV = np.squeeze(lambda_GCV)
#         self.f_rec_GCV = f_rec_GCV
#         self.lambda_GCV = lambda_GCV

#     def LocReg(self):
#         noisy_LRIto_ini_lam = self.lambda_LC
#         # noisy_f_rec_ini = noisy_f_rec_LC
#         f_rec_LR, lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(self.noisy_signal, self.A, noisy_LRIto_ini_lam, gamma_init, maxiter)
#         self.f_rec_LR = f_rec_LR
#         self.lambda_LR = lambda_LR

#     def DP(self):
#         f_rec_DP, lambda_DP = discrep_L2(self.noisy_signal, self.A, self.SNR, self.Lambda, noise = 1)


# A = kernel(n,m,TE,T2)
# print(A.shape)
# print("TE",TE)
# print("T2",T2)

# recovery1 = SpinalCord_Data(study1signal, study1_tail_std, TE, T2, Lambda, A)
# print("recover1", recovery1)
# f_rec_LC, lambda_LC = recovery1.LCurve()
# f_rec_GCV, lambda_GCV = recovery1.GCV()
# f_rec_LR, lambda_LR = recovery1.LocReg()
# f_rec_LS, lambda_LS = recovery1.LS()
# f_rec_DP, lambda_DP = recovery1.DP()

# import matplotlib.pyplot as plt

# # Assuming f_rec_LC, f_rec_GCV, f_rec_LR, f_rec_LS, f_rec_DP are arrays or lists of the same length

# # Plot for recovery1
# plt.figure(figsize=(10, 6))
# plt.plot(f_rec_LC, label='LCurve')
# plt.plot(f_rec_GCV, label='GCV')
# plt.plot(f_rec_LR, label='LocReg')
# plt.plot(f_rec_LS, label='LS')
# plt.plot(f_rec_DP, label='DP')
# plt.title('Recovery 1')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
# plt.savefig("/home/kimjosy/LocReg_Regularization-1/Simulations/spinalcordscript/SpinalCordRecovery1.png")

# recovery2 = SpinalCord_Data(study2signal, study2_tail_std, TE, T2, Lambda, A)
# f_rec_LC, lambda_LC = recovery2.LCurve()
# f_rec_GCV, lambda_GCV = recovery2.GCV()
# f_rec_LR, lambda_LR = recovery2.LocReg()
# f_rec_LS, lambda_LS = recovery2.LS()
# f_rec_DP, lambda_DP = recovery2.DP()

# # Plot for recovery2
# plt.figure(figsize=(10, 6))
# plt.plot(f_rec_LC, label='LCurve')
# plt.plot(f_rec_GCV, label='GCV')
# plt.plot(f_rec_LR, label='LocReg')
# plt.plot(f_rec_LS, label='LS')
# plt.plot(f_rec_DP, label='DP')
# plt.title('Recovery 2')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
# plt.savefig("/home/kimjosy/LocReg_Regularization-1/Simulations/spinalcordscript/SpinalCordRecovery2.png")

A = kernel(n, m, TE, T2)
print("A.shape", A.shape)

# normalized the signals to 1
sol1 = nnls(A,study1signal)[0]
factor = np.trapz(sol1,T2) * dT
study1signal = study1signal/factor

sol2 = nnls(A,study2signal)[0]
factor = np.trapz(sol2,T2) * dT
study2signal = study2signal/factor

plt.figure()
plt.plot(TE, study1signal, label="Study 1 Signal")
plt.plot(TE, study2signal, label="Study 2 Signal")
plt.legend()
plt.title("Study 1 and Study 2 Normalized Signals")
plt.savefig("/home/kimjosy/LocReg_Regularization-1/data/spinalcord/results/Study1andstudy2testnormalized.png")

# LCurve Function
def LCurve(noisy_signal, A, Lambda):
    # Implementation of Lcurve method
    f_rec_LC, lambda_LC = Lcurve(noisy_signal, A, Lambda)
    return f_rec_LC, lambda_LC

# LS Function
def LS(noisy_signal, A):
    f_rec_LS = nnls(A, noisy_signal)[0]
    return f_rec_LS

# GCV Function
def GCV(noisy_signal, A, Lambda):
    f_rec_GCV, lambda_GCV = GCV_NNLS(noisy_signal, A, Lambda)
    return f_rec_GCV, lambda_GCV

# LocReg Function
def LocReg(noisy_signal, A, lambda_LC):
    noisy_LRIto_ini_lam = lambda_LC
    # noisy_f_rec_ini = noisy_f_rec_LC
    f_rec_LR, lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_sp(noisy_signal, A, noisy_LRIto_ini_lam, gamma_init, maxiter)
    return f_rec_LR, lambda_LR

# DP Function
def DP(noisy_signal, A, SNR, Lambda):
    # f_rec_DP, lambda_DP = discrep_L2(noisy_signal, A, SNR, Lambda, noise=1)
    f_rec_DP, lambda_DP = discrep_L2_sp(noisy_signal, A, SNR, Lambda, noise = True)
    return f_rec_DP, lambda_DP


# Define the recovery functions for Study 1 and Study 2
def process_recovery(signal, tail_std, TE, T2, Lambda, A):
    f_rec_LC, lambda_LC = LCurve(signal, A, Lambda)
    print("Lcurve done")
    f_rec_GCV, lambda_GCV = GCV(signal, A, Lambda)
    print("Gcv done")
    lambda_LR = lambda_LC[0]
    print("lambda_LC", lambda_LR)
    print("signal",signal)
    print("A",A)
    print("lambda_LR",lambda_LR)
    f_rec_LR, lambda_LR = LocReg(signal, A, lambda_LR)
    print("loc reg done")
    f_rec_LS = LS(signal, A)
    print("LS done")
    f_rec_DP, lambda_DP = DP(signal, A, tail_std, Lambda)
    print("DP done")
    return f_rec_LC, f_rec_GCV, f_rec_LR, f_rec_LS, f_rec_DP

# Process Recovery for Study 1
f_rec_LC1, f_rec_GCV1, f_rec_LR1, f_rec_LS1, f_rec_DP1 = process_recovery(study1signal, study1_tail_std, TE, T2, Lambda, A)

# Plot for Recovery 1
plt.figure(figsize=(10, 6))
plt.plot(f_rec_LC1, label='LCurve')
plt.plot(T2, f_rec_GCV1, label='GCV')
plt.plot(T2, f_rec_LR1, label='LocReg')
plt.plot(T2, f_rec_LS1, label='LS')
plt.plot(f_rec_DP1, label='DP')
plt.title('Recovery 1 Cut off from T2 (0, 200)')
plt.xlabel('T2')
plt.ylabel('Intensity')
plt.xlim((0,200))
plt.legend()
plt.show()
plt.savefig("/home/kimjosy/LocReg_Regularization-1/data/spinalcord/results/SpinalCordRecovery1newtest.png")

# Process Recovery for Study 2
f_rec_LC2, f_rec_GCV2, f_rec_LR2, f_rec_LS2, f_rec_DP2 = process_recovery(study2signal, study2_tail_std, TE, T2, Lambda, A)

# Plot for Recovery 2
plt.figure(figsize=(10, 6))
plt.plot(T2,f_rec_LC2, label='LCurve')
plt.plot(T2,f_rec_GCV2, label='GCV')
plt.plot(T2,f_rec_LR2, label='LocReg')
plt.plot(T2,f_rec_LS2, label='LS')
plt.plot(T2,f_rec_DP2, label='DP')
plt.title('Recovery 2 Cut off from T2 (0, 200)')
plt.xlabel('T2')
plt.ylabel('Intensity')
plt.xlim((0,200))
plt.legend()
plt.show()
plt.savefig("/home/kimjosy/LocReg_Regularization-1/data/spinalcord/results/SpinalCordRecovery2newtest.png")
