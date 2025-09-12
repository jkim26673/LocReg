#Replicating the results from Lu and Pereverzev
import numpy as np
import sys
sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
import matplotlib.pyplot as plt
from regu.Lcurve import Lcurve
from regu.gravity import gravity
from regu.baart import baart
from regu.heat import heat
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
from regu.i_laplace import i_laplace


#Find the relative error:
def min_max(arr):
    minimum = np.min(arr)
    maximum = np.max(arr)
    mean = np.mean(arr)
    median = np.median(arr)
    std_dev = np.std(arr)
    return mean, median, std_dev

n = 100
nT2 = n
T2 = np.linspace(-np.pi/2,np.pi/2,n)
TE = T2
T2 = np.arange(1,201)
TE = np.arange(1,512,4).T
nT2 = len(T2)
half_nT2 = int(len(T2)/2)
half_mat = np.eye(half_nT2)
L_1 = np.zeros((nT2, nT2))
L_2 = np.zeros((nT2, nT2))
L_1[:half_nT2, :half_nT2] = half_mat
L_2[half_nT2:, half_nT2:] = half_mat

# #Creating the 1st derivative and 2nd derivative matrices
# I = np.eye(n)
# # Unsure about the D matrix
# D = np.diff(I, n=1).T * -1

# D_tilde = np.diff(I, n = 2).T
# half_nT2 = int(len(T2)/2)
# half_mat = np.eye(half_nT2)

# D = np.zeros((n-1,n))
# for i in range(n-1):
#     D[i, i] = 1          # Set the diagonal of the first column to 1
#     D[i, 1 + i] = -1     # Set the diagonal of the second column to -2
# D

# D_tilde = np.zeros((n-2,n))
# for i in range(n-2):
#     D_tilde[i, i] = 1          # Set the diagonal of the first column to 1
#     D_tilde[i, 1 + i] = -2     # Set the diagonal of the second column to -2
#     D_tilde[i, 2 + i] = 1      # Set the diagonal of the third column to 1
# D_tilde

##Lowest error
# B_1 = (D.T @ D)
# B_2 = (D_tilde.T @ D_tilde)

B_1 = L_1
B_2 =L_2


# B_1[np.isnan(B_1)] = 1

# B_2[np.isnan(B_2)] = 1

# B_1 = np.abs(D.T @ D)
# B_2 = np.abs(D_tilde.T @ D_tilde)
# B_1 = np.abs(D.T @ D)
# B_2 = np.abs(D_tilde.T @ D_tilde)

# D = np.diff(I, n=1, axis = 0)
# # B_1 = np.abs(D.T @ D)**0.5
# B_1 = D
# B_1.shape
# D_tilde = np.diff(I, n =2, axis = 0)
# B_2 = D_tilde
# # B_2 = np.abs(D_tilde.T @ D_tilde)**0.5
# B_2.shape

# D = np.diff(I, n=1, axis = -2)
# # B_1 = np.abs(D.T @ D)**0.5
# B_1 = D
# D_tilde = np.diff(I, n =2, axis = -2)
# # B_2 = np.abs(D_tilde.T @ D_tilde)**0.5
# B_2 = D_tilde
# B_2.shape
# delta2 = 0.1
Lambda_vec = np.logspace(-5,3,40)
nLambda = len(Lambda_vec)

G, data_noiseless, g = shaw(n, nargout=3)

U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 100

#delta = 0.01 ||Ax^cross||, where x^cross is data_noiseless and A is the G matrix; this is for 1% noise


############################################################################
# 1% noise for shaw problem 
alpha_1 = 0.2
alpha_2 = 0.2
beta = 0
c = 1
ep = 1e-8
omega = 0.5
delta = 0.01 * norm(data_noiseless)
nrun = 100

SD_noise= 1/SNR*max(abs(data_noiseless))
noise = np.random.normal(0,SD_noise, data_noiseless.shape)
data_noisy = data_noiseless + delta * noise

rel_error_beta = np.zeros(nrun)
rel_error_alpha1 = np.zeros(nrun)
rel_error_alpha2 = np.zeros(nrun)
rel_error_three_param = np.zeros(nrun)
final_alpha_1s = np.zeros(nrun)
final_alpha_2s = np.zeros(nrun)
final_betas = np.zeros(nrun)
final_xdelta = np.zeros((nrun,n))
for i in range(nrun):
    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    data_noisy = data_noiseless + noise
    # _, _, _, _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, 0, beta, omega, ep, c, delta)
    # rel_error_beta[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, 0, 0, omega, ep, c, delta)
    # rel_error_alpha1[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, alpha_2, 0, omega, ep, c, delta)
    # rel_error_alpha2[i] = rel_error
    final_alpha_1, final_alpha_2, final_beta,  x_delta, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, alpha_2, beta, omega, ep, c, delta)
    final_alpha_1s[i] = final_alpha_1
    final_alpha_2s[i] = final_alpha_2
    final_betas[i] = final_beta
    final_xdelta[i] = x_delta
    rel_error_three_param[i] = rel_error

#Reconstruct parameters from the Lu Paper for the 1% Shaw Problem
Lu_alpha_1 = 1.8e-8
Lu_alpha_2 = 0.0051
Lu_beta = 0.0026

x_delta = np.linalg.inv(G.T @ G + Lu_alpha_1 * B_1.T @ B_1 + Lu_alpha_2 * B_2.T @ B_2 + Lu_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy
Lu_rel_error = norm(g - x_delta)/norm(g)
Lu_rel_error

np.linalg.cond(G.T @ G + Lu_alpha_1 * (B_1.T @ B_1)**2 + Lu_alpha_2 * (B_2.T @ B_2)**2 + Lu_beta * np.eye(data_noisy.shape[0]))


#Plot and reconstruct the best solution with lowest relative error
ind = np.where((rel_error_three_param) == min(rel_error_three_param))[0][0]
test = final_xdelta[ind]
lowest_alpha1 = final_alpha_1s[ind]
lowest_alpha2 = final_alpha_2s[ind]
lowest_beta = final_betas[ind]

test_alpha1 = min_max(final_alpha_1s)[0]
test_alpha2 = min_max(final_alpha_2s)[0]
test_beta = min_max(final_betas)[0]
test_alpha1
test_alpha2
test_beta
test = np.linalg.inv(G.T @ G + test_alpha1 * B_1.T @ B_1 + test_alpha2 * B_2.T @ B_2 + test_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy

np.linalg.cond(G.T @ G + test_alpha1 * (B_1.T @ B_1)**2 + test_alpha2 * (B_2.T @ B_2)**2 + test_beta * np.eye(data_noisy.shape[0]))

Josh_rel_error = norm(g - test)/norm(g)
Josh_rel_error

plt.plot(g)
plt.plot(test)
plt.plot(x_delta)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.legend(['Ground Truth', 'Replicated Version', 'Lu Paper'])
plt.title("Shaw Problem 1% Noise")
plt.show()

#Get the parameters for the Replicated Results
min_max(final_alpha_1s)
min_max(final_alpha_2s)
min_max(final_betas)
min_max(rel_error_three_param)


############################################################################

# 5% noise for shaw problem 
alpha_1 = 0.2
alpha_2 = 0.2
beta = 0.1
c = 1
ep = 1e-8
omega = 0.5
delta = 0.05 * norm(data_noiseless)
nrun = 100

SD_noise= 1/SNR*max(abs(data_noiseless))
noise = np.random.normal(0,SD_noise, data_noiseless.shape)
data_noisy = data_noiseless + delta * noise

rel_error_beta = np.zeros(nrun)
rel_error_alpha1 = np.zeros(nrun)
rel_error_alpha2 = np.zeros(nrun)
rel_error_three_param = np.zeros(nrun)
final_alpha_1s = np.zeros(nrun)
final_alpha_2s = np.zeros(nrun)
final_betas = np.zeros(nrun)
final_xdelta = np.zeros((nrun,n))
for i in range(nrun):
    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    data_noisy = data_noiseless + noise
    # _, _, _, _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, 0, beta, omega, ep, c, delta)
    # rel_error_beta[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, 0, 0, omega, ep, c, delta)
    # rel_error_alpha1[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, alpha_2, 0, omega, ep, c, delta)
    # rel_error_alpha2[i] = rel_error
    final_alpha_1, final_alpha_2, final_beta,  x_delta, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, alpha_2, beta, omega, ep, c, delta)
    final_alpha_1s[i] = final_alpha_1
    final_alpha_2s[i] = final_alpha_2
    final_betas[i] = final_beta
    final_xdelta[i] = x_delta
    rel_error_three_param[i] = rel_error

#Reconstructions using mean parameters from the Lu Paper for the 5% Shaw Problem
Lu_alpha_1 = 4.3e-4
Lu_alpha_2 = 0.0263
Lu_beta = 0.0183

x_delta = np.linalg.inv(G.T @ G + Lu_alpha_1 * B_1.T @ B_1 + Lu_alpha_2 * B_2.T @ B_2 + Lu_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy
Lu_rel_error = norm(g - x_delta)/norm(g)
Lu_rel_error

np.linalg.cond(G.T @ G + Lu_alpha_1 * (B_1.T @ B_1)**2 + Lu_alpha_2 * (B_2.T @ B_2)**2 + Lu_beta * np.eye(data_noisy.shape[0]))

#Plot and reconstruct the best solution with lowest relative error
ind = np.where((rel_error_three_param) == min(rel_error_three_param))[0][0]
test = final_xdelta[ind]

test_alpha1 = min_max(final_alpha_1s)[0]
test_alpha2 = min_max(final_alpha_2s)[0]
test_beta = min_max(final_betas)[0]
test_alpha1
test_alpha2
test_beta
test = np.linalg.inv(G.T @ G + test_alpha1 * B_1.T @ B_1 + test_alpha2 * B_2.T @ B_2 + test_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy
Josh_rel_error = norm(g - test)/norm(g)
Josh_rel_error
plt.plot(g)
plt.plot(test)
plt.plot(x_delta)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.legend(['Ground Truth', 'Replicated Version', 'Lu Paper'])
plt.title("Shaw Problem 5% Noise")
plt.show()

min_max(final_alpha_1s)
min_max(final_alpha_2s)
min_max(final_betas)
min_max(rel_error_three_param)


############################################################################
# 1% noise for inverse laplace problem 
############################################################################

n = 100
nT2 = n
T2 = np.linspace(0,1000,n)
TE = T2
G,data_noiseless,g,_ = i_laplace(n, example = 1)
U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 1000
SD_noise= 1/SNR*max(abs(data_noiseless))
Lambda_vec = np.logspace(-10,5,40)
nLambda = len(Lambda_vec)

alpha_1 = 0.2
alpha_2 = 0.2
beta = 0.1
c = 1
ep = 1e-8
omega = 0.5
delta = 0.01 * norm(data_noiseless)
nrun = 100

# B_1 = (D.T @ D)**0.5
# B_2 = (D_tilde.T @ D_tilde)**0.5

# B_1[np.isnan(B_1)] = -1
# B_2[np.isnan(B_2)] = 1

SD_noise= 1/SNR*max(abs(data_noiseless))
noise = np.random.normal(0,SD_noise, data_noiseless.shape)
data_noisy = data_noiseless + delta * noise

rel_error_beta = np.zeros(nrun)
rel_error_alpha1 = np.zeros(nrun)
rel_error_alpha2 = np.zeros(nrun)
rel_error_three_param = np.zeros(nrun)
final_alpha_1s = np.zeros(nrun)
final_alpha_2s = np.zeros(nrun)
final_betas = np.zeros(nrun)
final_xdelta = np.zeros((nrun,n))
for i in range(nrun):
    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    data_noisy = data_noiseless + noise
    # _, _, _, _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, 0, beta, omega, ep, c, delta)
    # rel_error_beta[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, 0, 0, omega, ep, c, delta)
    # rel_error_alpha1[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, alpha_2, 0, omega, ep, c, delta)
    # rel_error_alpha2[i] = rel_error
    final_alpha_1, final_alpha_2, final_beta,  x_delta, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, alpha_2, beta, omega, ep, c, delta)
    final_alpha_1s[i] = final_alpha_1
    final_alpha_2s[i] = final_alpha_2
    final_betas[i] = final_beta
    final_xdelta[i] = x_delta
    rel_error_three_param[i] = rel_error

#Reconstruct mean regularization parameters from the Lu Paper for the 1% ilaplace Problem
Lu_alpha_1 = 3.5e-4
Lu_alpha_2 = 0.0823
Lu_beta = 0.0012

x_delta = np.linalg.inv(G.T @ G + Lu_alpha_1 * (B_1.T @ B_1)**2 + Lu_alpha_2 * (B_2.T @ B_2)**2 + Lu_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy
Lu_rel_error = norm(g - x_delta)/norm(g)
Lu_rel_error

np.linalg.cond(G.T @ G + Lu_alpha_1 * (B_1.T @ B_1)**2 + Lu_alpha_2 * (B_2.T @ B_2)**2 + Lu_beta * np.eye(data_noisy.shape[0]))

#Plot and reconstruct the best solution with lowest relative error

test_alpha1 = min_max(final_alpha_1s)[0]
test_alpha2 = min_max(final_alpha_2s)[0]
test_beta = min_max(final_betas)[0]
test_alpha1
test_alpha2
test_beta
test = np.linalg.inv(G.T @ G + test_alpha1 * B_1.T @ B_1 + test_alpha2 * B_2.T @ B_2 + test_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy

np.linalg.cond(G.T @ G + test_alpha1 * (B_1.T @ B_1)**2 + test_alpha2 * (B_2.T @ B_2)**2 + test_beta * np.eye(data_noisy.shape[0]))

Josh_rel_error = norm(g - test)/norm(g)
Josh_rel_error
plt.plot(g)
plt.plot(test)
plt.plot(x_delta)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.legend(['Ground Truth', 'Replicated Version', 'Lu Paper'])
plt.title("Inverse Laplace Problem 1% Noise")
plt.show()

plt.plot(g)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.title("Inverse Laplace Problem Ground Truth")
plt.show()

#Get the parameters for the Replicated Results
min_max(final_alpha_1s)
min_max(final_alpha_2s)
min_max(final_betas)
min_max(rel_error_three_param)
############################################################################

# 5% noise for inverse laplace problem 


############################################################################

# 1% noise for heat problem

n = 1000
nT2 = n
T2 = np.linspace(-np.pi/2,np.pi/2,n)
TE = T2

I = np.eye(n)
# Unsure about the D matrix
D = np.diff(I, n=1).T * -1

D_tilde = np.diff(I, n = 2).T
B_1 = D
B_2 =D_tilde

G, data_noiseless, g = heat(n, kappa = 1, nargin = 1, nargout=3)

U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 100
SD_noise= 1/SNR*max(abs(data_noiseless))
Lambda_vec = np.logspace(-5,3,40)
nLambda = len(Lambda_vec)

alpha_1 = 0.2
alpha_2 = 0.2
beta = 0.1
c = 1
ep = 1e-8
omega = 0.5
delta = 0.01 * norm(data_noiseless)
nrun = 100

# B_1 = (D.T @ D)**0.5
# B_2 = (D_tilde.T @ D_tilde)**0.5

# B_1[np.isnan(B_1)] = -1
# B_2[np.isnan(B_2)] = 1

SD_noise= 1/SNR*max(abs(data_noiseless))
noise = np.random.normal(0,SD_noise, data_noiseless.shape)
data_noisy = data_noiseless + delta * noise

rel_error_beta = np.zeros(nrun)
rel_error_alpha1 = np.zeros(nrun)
rel_error_alpha2 = np.zeros(nrun)
rel_error_three_param = np.zeros(nrun)
final_alpha_1s = np.zeros(nrun)
final_alpha_2s = np.zeros(nrun)
final_betas = np.zeros(nrun)
final_xdelta = np.zeros((nrun,n))
for i in range(nrun):
    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    data_noisy = data_noiseless + noise
    # _, _, _, _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, 0, beta, omega, ep, c, delta)
    # rel_error_beta[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, 0, 0, omega, ep, c, delta)
    # rel_error_alpha1[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, alpha_2, 0, omega, ep, c, delta)
    # rel_error_alpha2[i] = rel_error
    final_alpha_1, final_alpha_2, final_beta,  x_delta, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, alpha_2, beta, omega, ep, c, delta)
    final_alpha_1s[i] = final_alpha_1
    final_alpha_2s[i] = final_alpha_2
    final_betas[i] = final_beta
    final_xdelta[i] = x_delta
    rel_error_three_param[i] = rel_error


test_alpha1 = min_max(final_alpha_1s)[0]
test_alpha2 = min_max(final_alpha_2s)[0]
test_beta = min_max(final_betas)[0]
test_alpha1
test_alpha2
test_beta
test = np.linalg.inv(G.T @ G + test_alpha1 * B_1.T @ B_1 + test_alpha2 * B_2.T @ B_2 + test_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy

np.linalg.cond(G.T @ G + test_alpha1 * (B_1.T @ B_1)**2 + test_alpha2 * (B_2.T @ B_2)**2 + test_beta * np.eye(data_noisy.shape[0]))

Josh_rel_error = norm(g - test)/norm(g)
Josh_rel_error
plt.plot(g)
plt.plot(test)
# plt.plot(x_delta)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.legend(['Ground Truth', 'Replicated Version', 'Lu Paper'])
plt.title("Heat Problem 1% Noise")
plt.show()

# plt.plot(g)
# plt.xlabel("T2")
# plt.ylabel("Amplitude")
# plt.title("Heat Problem Ground Truth")
# plt.show()

#Get the parameters for the Replicated Results
min_max(final_alpha_1s)
min_max(final_alpha_2s)
min_max(final_betas)
min_max(rel_error_three_param)


############################################################################

# 1% noise for gravity problem

n = 1000
nT2 = n
T2 = np.linspace(-np.pi/2,np.pi/2,n)
TE = T2

I = np.eye(n)
# Unsure about the D matrix
D = np.diff(I, n=1).T * -1

D_tilde = np.diff(I, n = 2).T
B_1 = D
B_2 =D_tilde

G, data_noiseless, g = gravity(n, nargin = 1)

U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 100
SD_noise= 1/SNR*max(abs(data_noiseless))
Lambda_vec = np.logspace(-5,3,40)
nLambda = len(Lambda_vec)

alpha_1 = 0.2
alpha_2 = 0.2
beta = 0.1
c = 1
ep = 1e-8
omega = 0.5
delta = 0.01 * norm(data_noiseless)
nrun = 100

# B_1 = (D.T @ D)**0.5
# B_2 = (D_tilde.T @ D_tilde)**0.5

# B_1[np.isnan(B_1)] = -1
# B_2[np.isnan(B_2)] = 1

SD_noise= 1/SNR*max(abs(data_noiseless))
noise = np.random.normal(0,SD_noise, data_noiseless.shape)
data_noisy = data_noiseless + delta * noise

rel_error_beta = np.zeros(nrun)
rel_error_alpha1 = np.zeros(nrun)
rel_error_alpha2 = np.zeros(nrun)
rel_error_three_param = np.zeros(nrun)
final_alpha_1s = np.zeros(nrun)
final_alpha_2s = np.zeros(nrun)
final_betas = np.zeros(nrun)
final_xdelta = np.zeros((nrun,n))
for i in range(nrun):
    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    data_noisy = data_noiseless + noise
    # _, _, _, _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, 0, beta, omega, ep, c, delta)
    # rel_error_beta[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, 0, 0, omega, ep, c, delta)
    # rel_error_alpha1[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, alpha_2, 0, omega, ep, c, delta)
    # rel_error_alpha2[i] = rel_error
    final_alpha_1, final_alpha_2, final_beta,  x_delta, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, alpha_2, beta, omega, ep, c, delta)
    final_alpha_1s[i] = final_alpha_1
    final_alpha_2s[i] = final_alpha_2
    final_betas[i] = final_beta
    final_xdelta[i] = x_delta
    rel_error_three_param[i] = rel_error


test_alpha1 = min_max(final_alpha_1s)[0]
test_alpha2 = min_max(final_alpha_2s)[0]
test_beta = min_max(final_betas)[0]
test_alpha1
test_alpha2
test_beta
test = np.linalg.inv(G.T @ G + test_alpha1 * B_1.T @ B_1 + test_alpha2 * B_2.T @ B_2 + test_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy

np.linalg.cond(G.T @ G + test_alpha1 * (B_1.T @ B_1)**2 + test_alpha2 * (B_2.T @ B_2)**2 + test_beta * np.eye(data_noisy.shape[0]))

Josh_rel_error = norm(g - test)/norm(g)
Josh_rel_error
plt.plot(g)
plt.plot(test)
# plt.plot(x_delta)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.legend(['Ground Truth', 'Replicated Lu Version'])
plt.title("Gravity Problem 1% Noise")
plt.show()

# plt.plot(g)
# plt.xlabel("T2")
# plt.ylabel("Amplitude")
# plt.title("Heat Problem Ground Truth")
# plt.show()

#Get the parameters for the Replicated Results
min_max(final_alpha_1s)
min_max(final_alpha_2s)
min_max(final_betas)
min_max(rel_error_three_param)
############################################################################
#Make Pretty Tables

############################################################################
#TESTING EDGE CASES
# 1% noise for inverse laplace problem 
######################################################################
# alpha_1 =0, alpha_2 = 0, beta = 50

n = 100
nT2 = n
T2 = np.linspace(0,1000,n)
TE = T2
G,data_noiseless,g,_ = i_laplace(n, example = 1)
U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 1000
SD_noise= 1/SNR*max(abs(data_noiseless))
Lambda_vec = np.logspace(-10,5,40)
nLambda = len(Lambda_vec)

alpha_1 = 0
alpha_2 = 0
beta = 50
c = 1
ep = 1e-8
omega = 0.5
delta = 0.01 * norm(data_noiseless)
nrun = 100

# B_1 = (D.T @ D)**0.5
# B_2 = (D_tilde.T @ D_tilde)**0.5

# B_1[np.isnan(B_1)] = -1
# B_2[np.isnan(B_2)] = 1

SD_noise= 1/SNR*max(abs(data_noiseless))
noise = np.random.normal(0,SD_noise, data_noiseless.shape)
data_noisy = data_noiseless + delta * noise

rel_error_beta = np.zeros(nrun)
rel_error_alpha1 = np.zeros(nrun)
rel_error_alpha2 = np.zeros(nrun)
rel_error_three_param = np.zeros(nrun)
final_alpha_1s = np.zeros(nrun)
final_alpha_2s = np.zeros(nrun)
final_betas = np.zeros(nrun)
final_xdelta = np.zeros((nrun,n))
for i in range(nrun):
    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    data_noisy = data_noiseless + noise
    # _, _, _, _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, 0, beta, omega, ep, c, delta)
    # rel_error_beta[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, 0, 0, omega, ep, c, delta)
    # rel_error_alpha1[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, alpha_2, 0, omega, ep, c, delta)
    # rel_error_alpha2[i] = rel_error
    final_alpha_1, final_alpha_2, final_beta,  x_delta, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, alpha_2, beta, omega, ep, c, delta)
    final_alpha_1s[i] = final_alpha_1
    final_alpha_2s[i] = final_alpha_2
    final_betas[i] = final_beta
    final_xdelta[i] = x_delta
    rel_error_three_param[i] = rel_error

#Reconstruct mean regularization parameters from the Lu Paper for the 1% ilaplace Problem
Lu_alpha_1 = 3.5e-4
Lu_alpha_2 = 0.0823
Lu_beta = 0.0012

x_delta = np.linalg.inv(G.T @ G + Lu_alpha_1 * (B_1.T @ B_1)**2 + Lu_alpha_2 * (B_2.T @ B_2)**2 + Lu_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy
Lu_rel_error = norm(g - x_delta)/norm(g)
Lu_rel_error

np.linalg.cond(G.T @ G + Lu_alpha_1 * (B_1.T @ B_1)**2 + Lu_alpha_2 * (B_2.T @ B_2)**2 + Lu_beta * np.eye(data_noisy.shape[0]))

#Plot and reconstruct the best solution with lowest relative error

test_alpha1 = min_max(final_alpha_1s)[0]
test_alpha2 = min_max(final_alpha_2s)[0]
test_beta = min_max(final_betas)[0]
test_alpha1
test_alpha2
test_beta
test = np.linalg.inv(G.T @ G + test_alpha1 * B_1.T @ B_1 + test_alpha2 * B_2.T @ B_2 + test_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy

np.linalg.cond(G.T @ G + test_alpha1 * (B_1.T @ B_1)**2 + test_alpha2 * (B_2.T @ B_2)**2 + test_beta * np.eye(data_noisy.shape[0]))

Josh_rel_error = norm(g - test)/norm(g)
Josh_rel_error
min_max(final_alpha_1s)
min_max(final_alpha_2s)
min_max(final_betas)
min_max(rel_error_three_param)
plt.plot(g)
plt.plot(test)
plt.plot(x_delta)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.legend(['Ground Truth', 'Replicated', 'Lu Paper'])
plt.title("Inverse Laplace Problem 1% Noise")
plt.show()

plt.plot(g)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.title("Inverse Laplace Problem Ground Truth")
plt.show()


######################################################################
# alpha_1 =50, alpha_2 = 50, beta = 0
n = 100
nT2 = n
T2 = np.linspace(0,1000,n)
TE = T2
G,data_noiseless,g,_ = i_laplace(n, example = 1)
U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 1000
SD_noise= 1/SNR*max(abs(data_noiseless))
Lambda_vec = np.logspace(-10,5,40)
nLambda = len(Lambda_vec)

alpha_1 = 50
alpha_2 = 50
beta = 0
c = 1
ep = 1e-8
omega = 0.5
delta = 0.01 * norm(data_noiseless)
nrun = 100

# B_1 = (D.T @ D)**0.5
# B_2 = (D_tilde.T @ D_tilde)**0.5

# B_1[np.isnan(B_1)] = -1
# B_2[np.isnan(B_2)] = 1

SD_noise= 1/SNR*max(abs(data_noiseless))
noise = np.random.normal(0,SD_noise, data_noiseless.shape)
data_noisy = data_noiseless + delta * noise

rel_error_beta = np.zeros(nrun)
rel_error_alpha1 = np.zeros(nrun)
rel_error_alpha2 = np.zeros(nrun)
rel_error_three_param = np.zeros(nrun)
final_alpha_1s = np.zeros(nrun)
final_alpha_2s = np.zeros(nrun)
final_betas = np.zeros(nrun)
final_xdelta = np.zeros((nrun,n))
for i in range(nrun):
    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    data_noisy = data_noiseless + noise
    # _, _, _, _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, 0, beta, omega, ep, c, delta)
    # rel_error_beta[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, 0, 0, omega, ep, c, delta)
    # rel_error_alpha1[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, alpha_2, 0, omega, ep, c, delta)
    # rel_error_alpha2[i] = rel_error
    final_alpha_1, final_alpha_2, final_beta,  x_delta, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, alpha_2, beta, omega, ep, c, delta)
    final_alpha_1s[i] = final_alpha_1
    final_alpha_2s[i] = final_alpha_2
    final_betas[i] = final_beta
    final_xdelta[i] = x_delta
    rel_error_three_param[i] = rel_error

#Reconstruct mean regularization parameters from the Lu Paper for the 1% ilaplace Problem
Lu_alpha_1 = 3.5e-4
Lu_alpha_2 = 0.0823
Lu_beta = 0.0012

x_delta = np.linalg.inv(G.T @ G + Lu_alpha_1 * (B_1.T @ B_1)**2 + Lu_alpha_2 * (B_2.T @ B_2)**2 + Lu_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy
Lu_rel_error = norm(g - x_delta)/norm(g)
Lu_rel_error

np.linalg.cond(G.T @ G + Lu_alpha_1 * (B_1.T @ B_1)**2 + Lu_alpha_2 * (B_2.T @ B_2)**2 + Lu_beta * np.eye(data_noisy.shape[0]))

#Plot and reconstruct the best solution with lowest relative error

test_alpha1 = min_max(final_alpha_1s)[0]
test_alpha2 = min_max(final_alpha_2s)[0]
test_beta = min_max(final_betas)[0]
test_alpha1
test_alpha2
test_beta
test = np.linalg.inv(G.T @ G + test_alpha1 * B_1.T @ B_1 + test_alpha2 * B_2.T @ B_2 + test_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy

np.linalg.cond(G.T @ G + test_alpha1 * (B_1.T @ B_1)**2 + test_alpha2 * (B_2.T @ B_2)**2 + test_beta * np.eye(data_noisy.shape[0]))

Josh_rel_error = norm(g - test)/norm(g)
Josh_rel_error
min_max(final_alpha_1s)
min_max(final_alpha_2s)
min_max(final_betas)
min_max(rel_error_three_param)
plt.plot(g)
plt.plot(test)
plt.plot(x_delta)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.legend(['Ground Truth', 'Replicated', 'Lu Paper'])
plt.title("Inverse Laplace Problem 1% Noise")
plt.show()

plt.plot(g)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.title("Inverse Laplace Problem Ground Truth")
plt.show()
