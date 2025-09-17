# #Packages
# # np.random.seed(34)
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
print("import complete")

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
sigma1 = 2
mu1 = 40
sigma2 = 6
mu2 = 100

#Create ground truth
g = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
g = g + (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
g = g/2

plt.plot(g)
plt.show()
#need to shift T2 - gamma

data_noiseless = np.dot(G, g)

Jacobian = np.dot(G.T, G)

#Singular value decomposition
U, s, V = csvd(G, tst = None, nargin = 1, nargout = 3)
s = np.diag(s)

#Setting the SNR and standard deviation of noise; creating the noisy data
SNR = 130
SD_noise = 1 / SNR
noise = np.random.normal(0, SD_noise, size=data_noiseless.shape)
data_noisy = data_noiseless + noise

#Performing the DP
delta = np.linalg.norm(noise)*1.05
x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta, x_0= None, nargin = 5)

# l1 regularization
Lambda_vec_l1 = np.logspace(-5, 2, 20)
x_lc_vec_l1 = np.zeros((len(T2), len(Lambda_vec_l1)))
rhos_l1 = np.zeros(len(Lambda_vec_l1)).T
etas_l1 = np.zeros(len(Lambda_vec_l1)).T

for j in range(len(Lambda_vec_l1)):
    x = cp.Variable(nT2)
    objective = cp.Minimize(cp.norm(G.T @ G @ x - G.T @ data_noisy, 2) + Lambda_vec_l1[j] * cp.norm(x, 1))
    constraints = [x >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    #ECOS
    #x_lc_vec_l1[:, j] = x.value.flatten()
    x_lc_vec_l1[:, j] = x.value.flatten()
    rhos_l1[j] = np.linalg.norm(data_noisy - G @ x_lc_vec_l1[:, j]) ** 2
    etas_l1[j] = np.linalg.norm(x_lc_vec_l1[:, j], 1)

ireg_corner_l1 = l_curve_corner(rhos_l1,etas_l1,Lambda_vec_l1)[1]
x0_ini_l1 = x_lc_vec_l1[:, ireg_corner_l1]

## l2 regularization

# Lambda_vec = np.logspace(-3, 2, 10)
# x_lc_vec = np.zeros((len(T2), len(Lambda_vec)))
# rhos = np.zeros(len(Lambda_vec)).T
# etas = np.zeros(len(Lambda_vec)).T
# for j in range(len(Lambda_vec)):
#     #x_lc_vec[:, j] = lsqnonneg((G.T @ G) +  (Lambda_vec[j]**2) * np.eye(nT2), G.T @ data_noisy, tol = 1e-3)[0]
#     x_lc_vec[:, j] = nnls((G.T @ G) +  (Lambda_vec[j]**2) * np.eye(nT2), G.T @ data_noisy)[0]
#     rhos[j] = np.linalg.norm(data_noisy - G @ x_lc_vec[:, j]) ** 2
#     etas[j] = np.linalg.norm(x_lc_vec[:, j]) ** 2

Lambda_vec = np.logspace(-3, 2, 10)
x_lc_vec = np.zeros((len(T2), len(Lambda_vec)))
rhos = np.zeros(len(Lambda_vec)).T
etas = np.zeros(len(Lambda_vec)).T
for j in range(len(Lambda_vec)):
    #x_lc_vec[:, j] = lsqnonneg((G.T @ G) +  (Lambda_vec[j]**2) * np.eye(nT2), G.T @ data_noisy, tol = 1e-3)[0]
    x_lc_vec[:, j] = nnls((G.T @ G) +  (Lambda_vec[j]) * np.eye(nT2), G.T @ data_noisy)[0]
    rhos[j] = np.linalg.norm(data_noisy - G @ x_lc_vec[:, j]) ** 2
    etas[j] = Lambda_vec[j] * np.linalg.norm(x_lc_vec[:, j]) ** 2

plt.plot(Lambda_vec,rhos + etas)
plt.xlabel("Lambda")
plt.ylabel("||G(p)-d||_2^2 + lambda * ||p||_2^2")
plt.show()


reg_corner, ireg_corner, b = l_curve_corner(rhos,etas,Lambda_vec)
x0_ini = x_lc_vec[:, ireg_corner]

x0_LS,_,estimated_noise = lsqnonneg(G, data_noisy)
#x0_LS, estimated_noise = nnls(G, data_noisy)
x0 = x0_ini

ep = 1e-3
fig, ax = plt.subplots()
ax.set_xlabel('T2', fontsize=18)
ax.set_ylabel('Amplitude', fontsize=18)
ax.plot(T2, g, linewidth=2)
ax.plot(T2, x0_ini)
plt.show()

from scipy.optimize import nnls

search_lamb = np.logspace(-1,-6,3000)
reg_corner_test = np.ones(T2[-1]) * reg_corner 

test = x0_ini.copy()

n = 1
prev_x = test
estimated_noise_std = np.std(estimated_noise)
track = []
b = 100
sig = np.sin(2 * np.pi * 0.0000001 * test)
C = (np.linalg.norm(G.T @ data_noisy)/ (np.linalg.norm(G.T @ G @ test) + 1))
    # np.std(test - estimated_noise_std)
from scipy import signal

    #discoverY: np.abs(np.mean(test) - estimated_noise_std) --> 0, mimics the l2 curve
        #lambda_val = estimated_noise_std / ((np.abs(test) + ep))
    
    # plt.plot(lambda_val)
    # plt.show()
    # lambda_val = estimated_noise_std/(np.abs(test) + ep)
    # lambda_val = (np.linalg.norm(G.T @ data_noisy)/ (np.linalg.norm(G.T @ G @ G.T) + np.linalg.norm(test)))/((test) + ep)
    # lambda_val = (np.linalg.norm(G.T @ data_noisy, 2)/ (np.linalg.norm(G.T @ G @ G.T, 2)) )/((test) + ep)

    # lambda_val = (np.linalg.norm(G.T @ data_noisy, 2)/ (np.linalg.norm(G.T @ G @ G.T, 2) ))/((test) + ep)
while True:
    # lambda_val = estimated_noise_std / ((np.abs(test) + ep))
    # lambda_val = (np.linalg.norm(G.T**2 @ data_noisy, 1)/ (np.linalg.norm(G.T**2 @ G**2 @ G.T**2, 2)))/((test) + ep)
    lambda_val = (np.linalg.norm(G.T @ data_noisy, 2)/ (np.linalg.norm(G.T @ G @ G.T, 2)))/(np.abs(test) + ep)

    # lambda_val = np.exp(-(1/((test + ep)**2)))
    LHS = G.T @ G + np.diag(lambda_val)
            #RHS = G.T @ data_noisy + np.dot(G.T @ G * ep , np.ones(nT2)) + np.multiply(ep*lambda_val)
    # RHS = G.T @ data_noisy + (G.T @ G * ep) @ np.ones(nT2) + ep*lambda_val
    RHS = G.T @ data_noisy
        # RHS = G.T @ data_noisy + (G.T @ G * ep) @ np.ones(nT2) + ep*lambda_val
        # x0_ini = nnls(LHS, RHS, maxiter = 10000)[0]
    test = nnls(LHS,RHS, maxiter = 10000)[0]
        #x0_ini = nnls(LHS,RHS)[0]
    test = test - ep
    test[test < 0] = 0
        #curr_noise = np.dot(G, x0_ini) - data_noisy
    curr_noise = (G @ test) - data_noisy

    delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
    prev = np.linalg.norm(delta_p)
    LHS_temp = LHS.copy()
    while True:
        test = test - delta_p
        test[test < 0] = 0
        curr_noise = np.dot(G, test) - data_noisy
        delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
        if np.abs(np.linalg.norm(delta_p) / prev - 1) < 1e-3:
            break
        prev = np.linalg.norm(delta_p)

    test = test - delta_p
    test[test < 0] = 0
        # ax.plot(T2, test)
        # plt.draw()

    track.append(np.linalg.norm(test - prev_x) / np.linalg.norm(prev_x))
    if (np.linalg.norm(test - prev_x) / np.linalg.norm(prev_x)) < 1e-2:
        break

        # ax.plot(T2, test)
        # plt.draw()

    ep = ep / 1.1
    if ep <= 1e-6:
        ep = 1e-6
    n = n + 1
    prev_x = test

plt.plot(test)
plt.show()

old_locreg = test
np.linalg.norm(old_locreg - g)

new_lam = test
np.linalg.norm(new_lam - g)

plt.figure()
plt.plot(T2, g, linewidth=2)
#plt.hold(True)
plt.plot(T2, old_locreg, linewidth=2)
plt.plot(T2, test, linewidth=2)
plt.plot(T2, x_lc_vec[:, ireg_corner], linewidth=2)
# plt.plot(T2, x0_ini_l1, linewidth=2)
plt.legend(['True','LocReg','New_LocReg', 'l2 L-curve'], fontsize=10 )
plt.xlabel('T2', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.show()

a = np.linalg.pinv(G.T @ G) @ (G.T @ data_noisy - (np.linalg.norm(G.T @ data_noisy, 2)/ (np.linalg.norm(G.T @ G @ G.T, 2))))
plt.plot(a);plt.show()
#We show that L2 is perhaps not the best metric as shown below

import numpy as np

# Assuming you have an array of search_lamb values
search_lamb = np.logspace(-6, -2, 1000)

# Initialize an empty array to store the results for each lambda_val
results = []
test_arr = []

# Rest of your code
test = x0_ini.copy()

ep = 1e-3

n = 1
prev_x = test
estimated_noise_std = np.std(estimated_noise)
track = []
# Define the functions for LHS and RHS here
track = []
for lambda_val in search_lamb:
    # Update your code to work with vectorized operations
    lambda_val_arr = lambda_val / (np.abs(test) + ep)
    LHS = G.T @ G + np.diag(lambda_val_arr)
    RHS = G.T @ data_noisy

    test = nnls(LHS, RHS, maxiter=10000)[0]
    test = test - ep
    test[test < 0] = 0

    curr_noise = (G @ test) - data_noisy
    delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
    prev = np.linalg.norm(delta_p)
    LHS_temp = LHS.copy()

    while True:
        test = test - delta_p
        test[test < 0] = 0
        curr_noise = (G @ test) - data_noisy
        delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)

        if np.abs(np.linalg.norm(delta_p) / prev - 1) < 1e-3:
            break
        prev = np.linalg.norm(delta_p)

    test = test - delta_p
    test[test < 0] = 0
        # ax.plot(T2, test)
        # plt.draw()

    track.append(np.linalg.norm(test - prev_x) / np.linalg.norm(prev_x))
    if (np.linalg.norm(test - prev_x) / np.linalg.norm(prev_x)) < 1e-2:
        break

        # ax.plot(T2, test)
        # plt.draw()

    ep = ep / 1.1
    if ep <= 1e-6:
        ep = 1e-6

    test_arr.append(test)
    results.append(np.linalg.norm(g - test) / np.linalg.norm(test))

ind = np.argmin(results)
results[ind]
test = test_arr[ind]
search_lamb[ind]
# At this point, 'results' will contain the results for each lambda_val
plt.plot(results)
# plt.plot(track)
plt.show()

plt.figure()
plt.plot(T2, g, linewidth=2)
#plt.hold(True)
plt.plot(T2, test, linewidth=2)
plt.plot(T2, x_lc_vec[:, ireg_corner], linewidth=2)
plt.plot(T2, x0_ini_l1, linewidth=2)
plt.legend(['True', 'Loc-Reg', 'l2 L-curve', 'l1 L-curve'], fontsize=24)
plt.xlabel('T2', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.show()


#We show that L2 is perhaps not the best metric as shown above


import numpy as np
from scipy.stats import wasserstein_distance, entropy

# Assuming you have an array of search_lamb values
search_lamb = np.logspace(-4, -2, 1000)

# Initialize an empty array to store the results for each lambda_val
kl = []
l2 = []
test_arr = []

test = x0_ini.copy()

n = 1
prev_x = test
estimated_noise_std = np.std(estimated_noise)
track = []
ep = 1e-3

for lambda_val in search_lamb:
    # Update your code to work with vectorized operations
    lambda_val_arr = lambda_val / (np.abs(test) + ep)
    LHS = G.T @ G + np.diag(lambda_val_arr)
    RHS = G.T @ data_noisy

    test = nnls(LHS, RHS, maxiter=10000)[0]
    test = test - ep
    test[test < 0] = 0

    curr_noise = (G @ test) - data_noisy
    delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
    prev = np.linalg.norm(delta_p)
    LHS_temp = LHS.copy()

    while True:
        test = test - delta_p
        test[test < 0] = 0
        curr_noise = (G @ test) - data_noisy
        delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)

        if np.abs(np.linalg.norm(delta_p) / prev - 1) < 1e-3:
            break
        prev = np.linalg.norm(delta_p)

    test = test - delta_p
    test[test < 0] = 0
        # ax.plot(T2, test)
        # plt.draw()

    track.append(np.linalg.norm(test - prev_x) / np.linalg.norm(prev_x))
    if (np.linalg.norm(test - prev_x) / np.linalg.norm(prev_x)) < 1e-2:
        break

        # ax.plot(T2, test)
        # plt.draw()

    ep = ep / 1.1
    if ep <= 1e-6:
        ep = 1e-6

    test_arr.append(test)
    kl.append(entropy(test,g))
    l2.append(np.linalg.norm(g-test)/np.linalg.norm(test))

results = kl
results = l2
ind = np.argmin(results)
results[ind]
test = test_arr[ind]
search_lamb[ind]
# At this point, 'results' will contain the results for each lambda_val
plt.plot(results)
# plt.plot(track)
plt.show()

plt.figure()
plt.plot(T2, g, linewidth=2)
#plt.hold(True)
plt.plot(T2, test, linewidth=2)
plt.plot(T2, x_lc_vec[:, ireg_corner], linewidth=2)
plt.plot(T2, x0_ini_l1, linewidth=2)
plt.legend(['True', 'Loc-Reg', 'l2 L-curve', 'l1 L-curve'], fontsize=24)
plt.xlabel('T2', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)
plt.show()

plt.figure()
plt.plot(T2,g, linewidth =4)
plt.xlabel("T2", fontsize = 15, weight='bold')
plt.ylabel("Water Signal Amplitude (%)", fontsize = 15, weight='bold')
plt.show()

plt.figure()
plt.plot(TE,data_noisy, linewidth = 4)
plt.xlabel("Observation Time", fontsize = 15, weight='bold')
plt.ylabel("Signal Amplitude", fontsize = 15, weight='bold')
plt.title("Data From Multi-Echo MRI Sequence", fontsize = 15, weight='bold')
plt.show()