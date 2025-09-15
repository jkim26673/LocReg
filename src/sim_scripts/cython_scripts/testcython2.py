#testcython
import cProfile
import pstats
import io
import sys
sys.path.append(".")  # Replace this path with the actual path to the parent directory of Utilities_functions
import numpy as np
import time
# from locreg_ito import LocReg_Ito_mod as LocReg_Ito_mod_cy1
from Simulations.cythonminimize import minimize

import matplotlib.pyplot as plt
# from locreg_ito import LocReg_Ito_mod2 as LocReg_Ito_mod_cy2

from Simulations.Ito_LocReg import *

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
sigma1 = 2
mu1 = 40
sigma2 = 6
mu2 = 100

#Create ground truth
g = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
g = g + (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
g = g/2

data_noiseless = np.dot(G, g)

#Setting the SNR and standard deviation of noise; creating the noisy data
SNR = 130
SD_noise = 1 / SNR
noise = np.random.normal(0, SD_noise, size=data_noiseless.shape)
data_noisy = data_noiseless + noise

lam_ini = 0.01
gamma_init = 0.5
maxiter = 200

def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_cy(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)

# Measure time for the original Python function
original_result, original_time = time_function(LocReg_Ito_mod, data_noisy, G, lam_ini, gamma_init, maxiter)

# # Measure time for the Cython function
cython_result1, cython_time1 = time_function(LocReg_Ito_mod_cy1, data_noisy, G, lam_ini, gamma_init, maxiter)

# sparse_result, sparse_time = time_function(LocReg_Ito_modsparse, data_noisy, G, lam_ini, gamma_init, maxiter)


# Measure time for the Cython function
# cython_result2, cython_time2 = time_function(LocReg_Ito_mod_cy2, data_noisy, G, lam_ini, gamma_init, maxiter)

# # Print the results
# print(f"cholesky function time: {original_time:.6f} seconds")
print(f"Cython function1 time: {cython_time1:.6f} seconds")
# print(f"spsparse solve function1 time: {sparse_time:.6f} seconds")

# print(f"Cython function2 time: {cython_time2:.6f} seconds")

# Assuming both results are lists of arrays
# for i, (array1, array2) in enumerate(zip(cython_result1, original_result)):
#     print(f"Comparing result {i}: equal", np.allclose(array1, array2))

for i, (array1, array2) in enumerate(zip(cython_result1, original_result)):
    print(f"Comparing result {i}: equal", np.allclose(array1, array2))

# cython_result1 = sparse_result
best_f_rec_cython = cython_result1[0]
fin_lam_cython = cython_result1[1]

best_f_rec_orig = original_result[0]
fin_lam_orig = original_result[1]

plt.figure(figsize=(12.06, 4.2))
# Plotting the first subplot
plt.subplot(1, 3, 1) 
plt.plot(T2, g, linewidth=3, color='black', label='Ground Truth')
plt.plot(T2, best_f_rec_cython, linestyle=':', linewidth=3, color='red', label=f'Cholesky Solve')
plt.plot(T2, best_f_rec_orig, linestyle='-.', linewidth=3, color='gold', label=f'Numpy Linalg Solve')
plt.legend(fontsize=10, loc='best')
plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
ymax = np.max(g) * 1.15
plt.ylim(0, ymax)

# Plotting the second subplot
plt.subplot(1, 3, 2)
plt.plot(TE, G @ g, linewidth=3, color='black', label='Ground Truth')
plt.plot(TE, G @ best_f_rec_cython, linestyle=':', linewidth=3, color='red', label= f'Cholesky Solve')
plt.plot(TE, G @ best_f_rec_orig, linestyle='-.', linewidth=3, color='gold', label='Numpy Linalg Solve')
plt.legend(fontsize=10, loc='best')
plt.xlabel('TE', fontsize=20, fontweight='bold')
plt.ylabel('Intensity', fontsize=20, fontweight='bold')

plt.subplot(1, 3, 3)
plt.semilogy(T2, fin_lam_cython * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'Cholesky Solve')
plt.semilogy(T2, fin_lam_orig * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Numpy Linalg Solve')

plt.legend(fontsize=10, loc='best')
plt.xlabel('T2', fontsize=20, fontweight='bold')
plt.ylabel('Lambda', fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig("testingcythonLR.png")
plt.close() 

plt.plot()


# def run_profile():
#     TE = np.arange(1,512,4).T
#     #Generate the T2 values
#     T2 = np.arange(1,201).T
#     #Generate G_matrix

#     G = np.zeros((len(TE),len(T2)))
#     #For every column in each row, fill in the e^(-TE(i))
#     for i in range(len(TE)):
#         for j in range(len(T2)):
#             G[i,j] = np.exp(-TE[i]/T2[j])
#     nTE = len(TE)
#     nT2 = len(T2)
#     sigma1 = 2
#     mu1 = 40
#     sigma2 = 6
#     mu2 = 100

#     #Create ground truth
#     g = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
#     g = g + (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
#     g = g/2

#     data_noiseless = np.dot(G, g)

#     #Setting the SNR and standard deviation of noise; creating the noisy data
#     SNR = 130
#     SD_noise = 1 / SNR
#     noise = np.random.normal(0, SD_noise, size=data_noiseless.shape)
#     data_noisy = data_noiseless + noise

#     lam_ini = 0.01
#     gamma_init = 0.5
#     maxiter = 200
#     # f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter)
#     return  LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter)


# if __name__ == "__main__":
#     pr = cProfile.Profile()
#     pr.enable()
    
#     run_profile()
    
#     pr.disable()
    
#     s = io.StringIO()
#     sortby = pstats.SortKey.CUMULATIVE
#     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     ps.print_stats(30)
    
#     print(s.getvalue())