#Packages
# import matplotlib
# import numpy as np
# import cvxpy as cp
# import scipy
# import matplotlib.pyplot as plt
# from scipy.linalg import svd
# from scipy.optimize import nnls
# from lsqnonneg import lsqnonneg
# #from Simulations.lcurve_functions import l_cuve,csvd,l_corner
# from Simulations.l_curve_corner import l_curve_corner
# import os
# import mosek

# mosek_lic_path = "/Users/steveh/Downloads/mosek/mosek.lic"
# os.environ["MOSEKLM_LICENSE_FILE"] = mosek_lic_path
from utils.load_imports.loading import *


np.random.seed(34)


#Generate the TE values/ time
TE = np.arange(1,512,4).T
#Generate the T2 values
T2 = np.arange(1,201).T
#Generate G_matrix
G = np.exp(-TE[:, np.newaxis] / T2)

# shifted_T2 = T2 + 100  # Subtract gamma from T2 values


# # Generate shifted G matrix
# G = np.exp(-TE[:, np.newaxis] / shifted_T2)


#need to shift T2 - gamma
gammas = np.linspace(1,100,10)

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

data_noiseless = G @ g

Jacobian = G.T @ G

U, s, V = svd(G, full_matrices=False)
s = np.diag(s)

SNR = 1000
SD_noise = 1 / SNR
noise = np.random.normal(0, SD_noise, size=data_noiseless.shape)
data_noisy = data_noiseless + noise

# l1 regularization

Lambda_vec_l1 = np.logspace(-5, 2, 20)
x_lc_vec_l1 = np.zeros((len(T2), len(Lambda_vec_l1)))
rhos_l1 = np.zeros(len(Lambda_vec_l1)).T
etas_l1 = np.zeros(len(Lambda_vec_l1)).T

# for j in range(len(Lambda_vec_l1)):
#     x = lsqnonneg(G.T @ G, G.T @ data_noisy)[0]
#     x_lc_vec_l1[:,j] = x
#     rhos_l1[j] = np.linalg.norm(data_noisy - np.dot(G, x_lc_vec_l1[:, j])) ** 2
#     etas_l1[j] = np.linalg.norm(x_lc_vec_l1[:, j], 1)

# for j in range(len(Lambda_vec_l1)):
#     x = cp.Variable(nT2)
#     objective = cp.Minimize(cp.norm2(G.T @ G @ x - G.T @ data_noisy) + Lambda_vec_l1[j] * cp.norm1(x))
#     constraints = [x >= 0]
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver=cp.ECOS, verbose = False)

#     x_lc_vec_l1[:, j] = x.value.flatten()
#     rhos_l1[j] = np.linalg.norm(data_noisy - G @ x_lc_vec_l1[:, j])**2
#     etas_l1[j] = np.linalg.norm(x_lc_vec_l1[:, j], 1)

# ireg_corner_l1 = l_curve_corner(rhos_l1,etas_l1,Lambda_vec_l1)[1]
# x0_ini_l1 = x_lc_vec_l1[:, ireg_corner_l1]

for j, lambda_val in enumerate(Lambda_vec_l1):
    x = cp.Variable(nT2)
    objective = cp.Minimize(cp.norm(Jacobian @ x - G.T @ data_noisy, 2) + lambda_val * cp.norm(x, 1))
    constraints = [x >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)
    #ECOS
    #x_lc_vec_l1[:, j] = x.value.flatten()
    x_lc_vec_l1[:, j] = x.value.flatten()
    rhos_l1[j] = np.linalg.norm(data_noisy - G @ x_lc_vec_l1[:, j]) ** 2
    etas_l1[j] = np.linalg.norm(x_lc_vec_l1[:, j], 1)

# u, s, v = linalg.svd(h_mtx, full_matrices=False) #compute SVD without 0 singular values
#ireg_corner_l1 = l_cuve(U, s, V, plotit=False)
#ireg_corner_l1 = l_corner(rhos_l1,etas_l1,Lambda_vec_l1)[1]

ireg_corner_l1 = l_curve_corner(rhos_l1,etas_l1,Lambda_vec_l1)[1]
x0_ini_l1 = x_lc_vec_l1[:, ireg_corner_l1]

# ireg_corner_l1 = np.argmax(rhos_l1 / etas_l1)
# x0_ini_l1 = x_lc_vec_l1[:, ireg_corner_l1]

# scaler = StandardScaler()
# G_scaled = scaler.fit_transform(G)
# data_noisy_scaled = scaler.transform(data_noisy)

# for j in range(len(Lambda_vec_l1)):
#     lasso = Lasso(alpha=Lambda_vec_l1[j], positive=True)
#     #lasso.fit(G, data_noisy)
#     lasso.fit(G, data_noisy.reshape(-1))
#     #lasso.fit(G_scaled, data_noisy_scaled)
#     x_lc_vec_l1[:, j] = lasso.coef_
#     rhos_l1[j] = np.linalg.norm(data_noisy - G.dot(x_lc_vec_l1[:, j])) ** 2
#     etas_l1[j] = np.linalg.norm(x_lc_vec_l1[:, j], 1)

# ireg_corner_l1 = l_curve_corner(rhos_l1, etas_l1, Lambda_vec_l1)
# x0_ini_l1 = x_lc_vec_l1[:, ireg_corner_l1]



## l2 regularization

Lambda_vec = np.logspace(-2, 2, 10)
x_lc_vec = np.zeros((len(T2), len(Lambda_vec)))
rhos = np.zeros(len(Lambda_vec)).T
etas = np.zeros(len(Lambda_vec)).T
# for j, lambda_val in enumerate(Lambda_vec):
#     #A = G.T @ G + Lambda_vec[j]**2 * np.eye(nT2)
#     #x_lc_vec[:, j] = lsqnonneg(A, G.T @ data_noisy)
#     #x_lc_vec[:, j] = nnls(A, G.T @ data_noisy)[0]
#     #x_lc_vec[:, j] = lsqnonneg((G.T @ G) +  np.diag((Lambda_vec[j]**2) * np.eye(nT2)), G.T @ data_noisy)[0]
#     x_lc_vec[:, j] = lsqnonneg(Jacobian +  (lambda_val**2) * np.eye(nT2), G.T @ data_noisy)[0]
#     rhos[j] = np.linalg.norm(data_noisy - G @ x_lc_vec[:, j]) ** 2
#     etas[j] = np.linalg.norm(x_lc_vec[:, j]) ** 2

for j in range(len(Lambda_vec)):
    x_lc_vec[:, j] = lsqnonneg((G.T @ G) +  (Lambda_vec[j]**2) * np.eye(nT2), G.T @ data_noisy)[0]
    rhos[j] = np.linalg.norm(data_noisy - G @ x_lc_vec[:, j]) ** 2
    etas[j] = np.linalg.norm(x_lc_vec[:, j]) ** 2


#ireg_corner = l_curve_corner(rhos,etas,Lambda_vec)[1]
ireg_corner = l_curve_corner(rhos,etas,Lambda_vec)[1]
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

n = 1
prev_x = x0_ini
estimated_noise_std = np.std(estimated_noise)
track = []
while True:
    lambda_val = estimated_noise_std / (np.abs(x0_ini) + ep)
    LHS = Jacobian + np.diag(lambda_val)
    #RHS = G.T @ data_noisy + np.dot(G.T @ G * ep , np.ones(nT2)) + np.multiply(ep*lambda_val)
    RHS = G.T @ data_noisy + (Jacobian * ep) @ np.ones(nT2) + ep*lambda_val
    x0_ini = lsqnonneg(LHS, RHS)[0]
    #x0_ini = nnls(LHS,RHS)[0]
    x0_ini = x0_ini - ep
    x0_ini[x0_ini < 0] = 0
    #curr_noise = np.dot(G, x0_ini) - data_noisy
    curr_noise = (G @ x0_ini) - data_noisy

    delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
    prev = np.linalg.norm(delta_p)
    LHS_temp = LHS.copy()
    while True:
        x0_ini = x0_ini - delta_p
        x0_ini[x0_ini < 0] = 0
        curr_noise = np.dot(G, x0_ini) - data_noisy
        delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
        if np.abs(np.linalg.norm(delta_p) / prev - 1) < 1e-3:
            break
        prev = np.linalg.norm(delta_p)

    x0_ini = x0_ini - delta_p
    x0_ini[x0_ini < 0] = 0
    ax.plot(T2, x0_ini)
    plt.draw()

    track.append(np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x))
    if (np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x)) < 1e-2:
        break

    ax.plot(T2, x0_ini)
    plt.draw()

    ep = ep / 1.1
    if ep <= 1e-6:
        ep = 1e-6
    n = n + 1
    prev_x = x0_ini

# while True:
#     lambda_val = estimated_noise_std / (np.abs(x0_ini) + ep)
#     LHS = Jacobian + np.diag(lambda_val)
#     RHS = G.T @ data_noisy + (Jacobian * ep) @ np.ones(nT2) + ep * lambda_val

#     x0_ini = lsqnonneg(LHS, RHS)[0]
#     x0_ini -= ep
#     np.maximum(x0_ini, 0, out=x0_ini)

#     curr_noise = G @ x0_ini - data_noisy
#     delta_p = np.linalg.solve(LHS, G.T @ curr_noise)

#     prev = np.linalg.norm(delta_p)
#     LHS_temp = LHS

#     while True:
#         x0_ini -= delta_p
#         np.maximum(x0_ini, 0, out=x0_ini)
#         curr_noise = G @ x0_ini - data_noisy
#         delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
#         if np.abs(np.linalg.norm(delta_p) / prev - 1) < 1e-3:
#             break
#         prev = np.linalg.norm(delta_p)

#     x0_ini -= delta_p
#     np.maximum(x0_ini, 0, out=x0_ini)

#     ax.plot(T2, x0_ini)
#     plt.draw()

#     track.append(np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x))
#     if np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x) < 1e-2:
#         break

#     ax.plot(T2, x0_ini)
#     plt.draw()

#     ep /= 1.1
#     if ep <= 1e-6:
#         ep = 1e-6
#     n += 1
#     prev_x = x0_ini

# while True:
#     lambda_val = estimated_noise_std / (np.abs(x0_ini) + ep)
#     LHS = Jacobian + np.diag(lambda_val)
#     RHS = G.T @ data_noisy + (Jacobian * ep) @ np.ones(nT2) + ep * lambda_val

#     x0_ini = lsqnonneg(LHS, RHS)[0]
#     x0_ini -= ep
#     np.maximum(x0_ini, 0, out=x0_ini)

#     curr_noise = G @ x0_ini - data_noisy
#     delta_p = np.linalg.solve(LHS, G.T @ curr_noise)

#     prev = np.linalg.norm(delta_p)
#     LHS_temp = LHS

#     while True:
#         x0_ini -= delta_p
#         np.maximum(x0_ini, 0, out=x0_ini)
#         curr_noise = G @ x0_ini - data_noisy
#         delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
#         if np.abs(np.linalg.norm(delta_p) / prev - 1) < 1e-3:
#             break
#         prev = np.linalg.norm(delta_p)

#     x0_ini -= delta_p
#     np.maximum(x0_ini, 0, out=x0_ini)

#     # Plot the current estimate
#     ax.plot(T2, x0_ini)

#     # Check convergence condition and break the loop if satisfied
#     if np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x) < 1e-2:
#         break

#     # Update the previous estimate
#     prev_x = x0_ini

plt.figure()
gamma = 5
plt.plot(T2, g, linewidth=2)
#plt.hold(True)
#plt.plot(T2, x0_ini, linewidth=2)
plt.plot(T2, x0_ini + gamma, linewidth=2)
#plt.title("")
#plt.plot(T2, x_lc_vec[:, ireg_corner], linewidth=2)
plt.plot(T2 + gamma, x_lc_vec[:, ireg_corner], linewidth=2)

#plt.plot(T2, x0_ini_l1, linewidth=2)
plt.plot(T2 + gamma, x0_ini_l1, linewidth=2)

plt.legend(['True', 'Loc-Reg', 'l2 L-curve', 'l1 L-curve'], fontsize=24)
plt.xlabel('T2', fontsize=18)
plt.ylabel('Amplitude', fontsize=18)

gamma = 0

from scipy.stats import wasserstein_distance, entropy
from scipy.special import rel_entr

wasserstein_distance(T2, T2, u_weights=g, v_weights=x0_ini )
wasserstein_distance(T2, T2+gamma, u_weights=g, v_weights=x0_ini)


#shifted_y_values_B = np.interp(T2 + gamma, T2, x0_ini)
from scipy.interpolate import interp1d
interpolator = interp1d(T2, x0_ini, kind='cubic' )
shifted_y_values_cubic = interpolator(T2+gamma)

shifted_y_values_lin = np.interp(T2 + gamma, T2, x0_ini)


entropy(x0_ini,g)
entropy(shifted_y_values_cubic,g)
entropy(shifted_y_values_lin,g)

kl_divergence

g_truth = np.linalg.norm(g)
def error (exp,truth):
    return np.linalg.norm(exp-truth)
#normalized locreg error
locreg_err = error(x0_ini, g)/g_truth

locreg_err
plt.plot(T2, x0_ini)
plt.plot(T2,shifted_y_values_B)

plt.show()
#RMS ERROR EQUALS TO MINIMUM miinimum over gamma
#KL divergence
#washeterstein metric (symmetric)
#Generate all the exmples of SpanReg p

# def error (exp,truth):
#     return np.linalg.norm(exp-truth)

# g_truth = np.linalg.norm(g)

# #L2 metric

# #normalized locreg error
# locreg_err = error(x0_ini, g)/g_truth
# #normalized L2 error
# l2_error = error(x_lc_vec[:, ireg_corner],g)/g_truth
# # normalized L1 error
# l1_error = error(x0_ini_l1,g)/g_truth

# #KL divergence
# from scipy.special import rel_entr

# #compare locreg with ground truth
# sum(rel_entr(x0_ini, g))
# #compare L2 with ground truth
# sum(rel_entr(x_lc_vec[:, ireg_corner], g))
# #compare L1 with ground truth
# sum(rel_entr(x0_ini_l1, g))





#RMS ERROR EQUALS TO MINIMUM miinimum over gamma
#KL divergence
#washeterstein metric (symmetric)
#Generate all the exmples of SpanReg p