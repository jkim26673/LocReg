#Initializing the Inverse Laplace Problem

# https://www.cs.ubc.ca/sites/default/files/tr/1981/TR-81-10.pdf

# % Discretization of the inverse Laplace transformation by means of
# % Gauss-Laguerre quadrature.  The kernel K is given by
# %    K(s,t) = exp(-s*t) ,
# % and both integration intervals are [0,inf).
# %

#discretized using n-point Gauss-Laguerre quadrature
#Approximates are generated for f(t) at Gauss_Laguerra abscissae(t_j)^n

from src.utils.load_imports.load_classical import *
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
#n is the number of discretization
#Error in residual ||Kf -g||


#https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature


# def lag_weights_roots(n):
#     x = Symbol("x")
#     roots = Poly(laguerre(n, x)).all_roots()
#     x_i = [rt.evalf(10) for rt in roots]
#     w_i = [(rt / ((n + 1) * laguerre(n + 1, rt)) ** 2).evalf(10) for rt in roots]
#     return x_i, w_i


# def lag_weights_roots_2(n):
#     x = Symbol("x")
#     roots = Poly(laguerre(n, x)).all_roots()
    
#     x_i = [rt.evalf(10) for rt in roots]
    
#     # Use a lambda function to calculate w_i
#     laguerre_n_plus_1 = lambda rt: laguerre(n + 1, rt)
#     w_i = [((rt / ((n + 1) * laguerre_n_plus_1(rt)) ** 2).evalf(10)) for rt in roots]
    
#     return x_i, w_i


# What are you asked to find or show?[8]

#I'm asked to write my own inverse laplace transform function

# Can you restate the problem in your own words?

# The problem is that the usual way of evaluating the ILT which is using the gauss-laguerre
# quadrature methods is that it solves this by interpolating using multiple weights for certain discretizations
# As the number of discretizations increases, the it becomes difficult to solve/

# Can you think of a picture or a diagram that might help you understand the problem?
# Is there enough information to enable you to find a solution?
# Do you understand all the words used in stating the problem?
# Do you need to ask a question to get the answer?

# 3 conditions:
#singular vectors v are asympotoics to zero; 
# x-vectors are asymptotic to 1.0
# v^(i) and x^(i) each change signs (i-1) times

def min_max(arr):
    minimum = np.min(arr)
    maximum = np.max(arr)
    mean = np.mean(arr)
    median = np.median(arr)
    std_dev = np.std(arr)
    return mean, median, std_dev


def i_laplace2(n,example):
    s = np.linspace(1,n,n)
    # ds = np.diff(s)

    # sc = (s[:-1] + s[1:]) / 2

    # ds = np.append(ds, 1)
    # sc = np.append(sc, 0)

    # s = (10 / n) * np.arange(1, n+1)

    # % Compute abscissas x_i and weights w_i
    t_i, w_i = np.polynomial.laguerre.laggauss(n)

    t_i = np.array(t_i)
    # % Set up the coefficient matrix A.
    A = np.zeros((n,n))
    for i in range(len(s)):
        for j in range(len(t_i)):
            A[i,j] = np.exp(-(s[i]*t_i[j]))

    #Cover all example problems of ILT
    if example == 1:
        g = np.ones(n)/(s + 0.5)
        f = np.exp(-(t_i)/2)
    elif example == 2:
        g =  np.ones(n) / s - np.ones(n) / (s + 0.5)
        f = 1 - np.exp(-t_i / 2)
    elif example == 3:
        g = 2 / ((s + 0.5) ** 3)
        f= (t_i ** 2) * np.exp(-t_i / 2)
    elif example == 4:
        g = np.exp(-2 * s) / s
        f = np.ones(n)
        c = np.where(t_i <= 2)[0]
        f[c] = np.zeros(len(c))
    else:
        raise ValueError('Illegal example')

    return A, g, f, t_i


G,data_noiseless,g,_ = i_laplace2(n =100,example = 2)
A,b,g_orig,_ = i_laplace(n=100,example =2)
import matplotlib.pyplot as plt
plt.plot(g)
plt.plot(g_orig)
plt.legend(["josh","hansen"])
plt.title("I_laplace Replication Example 2")
plt.show()

############################################################################
# 1% noise for inverse laplace problem 
############################################################################

n = 100
nT2 = n
T2 = np.linspace(0,1000,n)
TE = T2
G,data_noiseless,g,_ = i_laplace2(n, example = 1)
U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
#Creating the 1st derivative and 2nd derivative matrices
I = np.eye(n)
# Unsure about the D matrix
D = np.diff(I, n=1).T * -1

D_tilde = np.diff(I, n = 2).T
B_1 = D
B_2 =D_tilde

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

SNR = 1000
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


################################################################################################
#Solve without regularization
n = 100
G1,data_noiseless1,g1,t_i1 = i_laplace2(n, example = 1)
G2,data_noiseless2,g2,t_i2 = i_laplace2(n, example = 2)
G3,data_noiseless3,g3,t_i3 = i_laplace2(n, example = 3)

SNR = 1000

data_noisy1 = data_noiseless1 + delta * noise
data_noisy2 = data_noiseless2 + delta * noise
data_noisy3 = data_noiseless3 + delta * noise
#Solve without regularization for example one:

def function_ex1(t, c):
    func = np.exp(-t/c)
    return func

def function_ex2(t, a, c):
    func = a - np.exp(-t/c)
    return func

def function_ex3(t, a, c):
    func = (t ** a) * (np.exp(-t/c))
    return func

true_param_ex1 = np.array([2])
true_param_ex2 = np.array([1,2])
true_param_ex3 = np.array([2,2])

#Solve without regularization for example 1
import scipy

lb_c = 0.0
ub_c = np.inf
popt1, pcov = scipy.optimize.curve_fit(function_ex1, t_i1, data_noisy1, p0 = true_param_ex1, bounds = ([lb_c],[ub_c]), method = 'trf', maxfev =10000)  # curve, xdata, ydata)

#Solve without regularization for example 2
lb_c = 0.0
ub_c = np.inf
lb_a = -np.inf
ub_a = np.inf
popt2, pcov = scipy.optimize.curve_fit(function_ex2, t_i2, data_noisy2, p0 = true_param_ex2, bounds = ([lb_a,lb_c],[ub_a,ub_c]), method = 'trf', maxfev =10000)  # curve, xdata, ydata)

#Solve without regularization for example 3
lb_c = 0.0
ub_c = np.inf
lb_a = 0.0
ub_a = np.inf
popt3, pcov = scipy.optimize.curve_fit(function_ex3, t_i3, data_noisy3, p0 = true_param_ex3, bounds = ([lb_a,lb_c],[ub_a,ub_c]), method = 'trf', maxfev =10000)  # curve, xdata, ydata)

plt.plot(t_i1,g1, c = "b", label = "Inverse Laplace Transform for Example 1") 
plt.plot(t_i1,function_ex1(t_i1, popt1[0]), c = "r", label= "fit curve") 
plt.plot(f_rec_DP, c = "g")
plt.legend(["Ground Truth", "Non-Regularized NLLS curve", "Regularized with DP_ Lambda = 0.0595"])
plt.title( "Inverse Laplace Transform for Example 1")
plt.show()

plt.plot(t_i2,g2, c = "b", label = "Inverse Laplace Transform for Example 2") 
plt.plot(t_i2,function_ex2(t_i2, popt2[0], popt2[1]), c = "r", label= "fit curve") 
plt.legend(["Ground Truth", "Non-Regularized NLLS curve"])
plt.title( "Inverse Laplace Transform for Example 2")
plt.show()

plt.plot(t_i3,g3, c = "b", label = "Inverse Laplace Transform for Example 3") 
plt.plot(t_i3,function_ex3(t_i3, popt3[0], popt3[1]), c = "r", label= "fit curve") 
plt.legend(["Ground Truth", "Non-Regularized NLLS curve"])
plt.title( "Inverse Laplace Transform for Example 3")
plt.show()

########################################Solve with Tikhonov Regularization

def error(exp_param, true_param, num_param):
    if num_param ==1:
        c_error = (exp_param[0] - true_param[0])**2
        error = c_error
    if num_param ==2:
        a_error = (exp_param[0] - true_param[0])**2
        c_error = (exp_param[0] - true_param[1])**2
        error = a_error + c_error
    return error

#Tikhonov Regularization for Example 1
#Use DP:
U,s,V = csvd(G1,tst = None, nargin = 1, nargout = 3)
#Creating the 1st derivative and 2nd derivative matrices
I = np.eye(n)
# Unsure about the D matrix
D = np.diff(I, n=1).T * -1

D_tilde = np.diff(I, n = 2).T
B_1 = D
B_2 =D_tilde

delta = norm(noise)*1.05
x_delta,lambda_DP = discrep(U,s,V,data_noisy1,delta, x_0= None, nargin = 5)
f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy1,lambda_DP, nargin=5, nargout=1)
norm(g - f_rec_DP)
lambda_DP

plt.plot(f_rec_DP)
plt.plot(g1)
plt.title("L2 Regularized Solution using Discepancy Principle" + "\n for Inverse Laplace Transform Example 1")
plt.legend(["Regularized with DP_ Lambda = 0.0595", "Ground Truth"])
plt.show()


x_delta,lambda_DP = discrep(U,s,V,data_noisy1,delta, x_0= None, nargin = 5)

#########################################Solve with Tikhonov Regularization using L1 regularization
import cvxpy as cp
Lambda_vec_l1 = np.logspace(-6, 2, 100)
x_lc_vec_l1 = np.zeros((len(T2), len(Lambda_vec_l1)))
rhos_l1 = np.zeros(len(Lambda_vec_l1)).T
etas_l1 = np.zeros(len(Lambda_vec_l1)).T

delta = norm(noise)*1.05

for j in range(len(Lambda_vec_l1)):
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(G1.T @ G1 @ x - G1.T @ data_noisy1, 2) + Lambda_vec_l1[j] * cp.norm(x, 1))
    # constraints = [x >= 0]
    problem = cp.Problem(objective)
    problem.solve(solver=cp.ECOS, verbose=False)
    #ECOS
    #x_lc_vec_l1[:, j] = x.value.flatten()
    x_lc_vec_l1[:, j] = x.value.flatten()
    rhos_l1[j] = np.linalg.norm(G1 @ x_lc_vec_l1[:, j] - data_noisy1)
    etas_l1[j] = np.linalg.norm(x_lc_vec_l1[:, j], 1)
index= np.where(np.abs(rhos_l1 - delta) == np.min(np.abs(rhos_l1 - delta)))[0]
rec = x_lc_vec_l1[:, index]
# plt.plot(Lambda_vec_l1,rhos_l1)
# plt.plot(np.ones(len(rhos_l1))*delta)
lambda_val = Lambda_vec_l1[index]
plt.plot(rec)
plt.plot(g)
plt.title("L1 Regularized Solution using Discepancy Principle" + "\n for Inverse Laplace Transform Example 1")
plt.show()

#########################################Solve with Tikhonov Regularization Add an additional term e.g 1st derivative and do a gridsearch
D = np.diff(I, n=1).T * -1
I = np.eye(n)
etas = np.zeros(len(Lambda_vec)).T

# Lambda_vec = np.logspace(-6, 2, 100)
# Lambda_vec2 = np.logspace(-6, 2, 100)

# x_lc_vec = np.zeros((len(T2), len(Lambda_vec), len(Lambda_vec2)))
# rhos = np.zeros((len(Lambda_vec), len(Lambda_vec2)))

# for j in range(len(Lambda_vec)):
#     for k in range(len(Lambda_vec2)):
#         x = cp.Variable(n)
#         # objective = cp.Minimize(cp.norm(G1.T @ G1 @ x - G1.T @ data_noisy1, 2) + Lambda_vec[j]**2 * cp.norm(x,2)**2)
#         objective = cp.Minimize(cp.norm(G1.T @ G1 @ x - G1.T @ data_noisy1, 2) + Lambda_vec[j]**2 * cp.norm(I @ x,2)**2 + Lambda_vec2[k]**2 * cp.norm(D @ x,2)**2 )

#         # constraints = [x >= 0]
#         problem = cp.Problem(objective)
#         problem.solve(solver=cp.SCS, verbose=False)

#         x_lc_vec[:, j, k] = x.value.flatten()
#         # x_lc_vec[:, j] = nnls((G.T @ G) +  (Lambda_vec[j]) * np.eye(n), G.T @ data_noisy)[0]
#         rhos[j, k] = np.linalg.norm(G @ x_lc_vec[:, j, k] - data_noisy1, 2)
#         # etas[j] = Lambda_vec[j] * np.linalg.norm(x_lc_vec[:, j]) ** 2

from itertools import product

Lambda_vec = np.logspace(-6, 2, 100)
Lambda_vec2 = np.logspace(-6, 2, 100)

x_lc_vec = np.zeros((len(T2), len(Lambda_vec), len(Lambda_vec2)))
rhos = np.zeros((len(Lambda_vec), len(Lambda_vec2)))
import tqdm
for j, k in (product(range(len(Lambda_vec)), range(len(Lambda_vec2)))):
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(G1.T @ G1 @ x - G1.T @ data_noisy1, 2) +
                            Lambda_vec[j]**2 * cp.norm(I @ x, 2)**2 +
                            Lambda_vec2[k]**2 * cp.norm(D @ x, 2)**2)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS, verbose=False)

    x_lc_vec[:, j, k] = x.value.flatten()
    rhos[j, k] = np.linalg.norm(G @ x_lc_vec[:, j, k] - data_noisy1, 2)

np.save('rhos_2d_grid_search_multi_param_1_9_24.npy', rhos)
np.save('x_lc_vec_2d_grid_search_multi_param_1_9_24.npy', x_lc_vec)


x_lc_vec = np.load("/Users/steveh/Downloads/x_lc_vec_2d_grid_search_multi_param_1_9_24.npy")
rhos = np.load("/Users/steveh/Downloads/rhos_2d_grid_search_multi_param_1_9_24.npy")

import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.linspace(1,100,100)
Y = np.linspace(1,100,100)
X, Y = np.meshgrid(X, Y)
Z = rhos

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.set_xlabel("Alpha Values")
ax.set_ylabel("Beta Values")
ax.set_zlabel("Residual Norm")
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



index= np.where(np.abs(rhos - delta) == np.min(np.abs(rhos - delta)))[0]
rec = x_lc_vec[:, index]
lambda_DPcvx = Lambda_vec[index]
plt.plot(rec)
plt.plot(f_rec_DP)
plt.plot(g)
plt.legend(["cvxpy", "DP tikh matlab", 'gt'])
plt.title("L1 Regularized Solution using Discepancy Principle" + "\n for Inverse Laplace Transform Example 1")
plt.show()

#Solve in closed form using grid search over two regularization parameters; ; specify kernel matrix and two regularization matices (Id and derivaiton)
#run these the two unknowns of alpha and beta; run and satisfy the DP

from itertools import product

Lambda_vec = np.logspace(-6, 2, 100)
Lambda_vec2 = np.logspace(-6, 2, 100)
Lambda_vec3 = np.logspace(-6, 2, 100)

D_tilde = np.diff(I, n = 2).T


x_lc_vec = np.zeros((len(T2), len(Lambda_vec), len(Lambda_vec2), len(Lambda_vec3)))
rhos = np.zeros((len(Lambda_vec), len(Lambda_vec2), len(Lambda_vec3)))
import tqdm
for j, k, l in (product(range(len(Lambda_vec)), range(len(Lambda_vec2)), range(len(Lambda_vec3)))):
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(G1.T @ G1 @ x - G1.T @ data_noisy1, 2) +
                            Lambda_vec[j]**2 * cp.norm(I @ x, 2)**2 +
                            Lambda_vec2[k]**2 * cp.norm(D @ x, 2)**2 + Lambda_vec3[l]**2 * cp.norm(D_tilde @ x, 2)**2)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS, verbose=False)

    x_lc_vec[:, j, k, l] = x.value.flatten()
    rhos[j, k, l] = np.linalg.norm(G @ x_lc_vec[:, j, k, l] - data_noisy1, 2)

np.save('rhos_3d_grid_search_multi_param_1_9_24.npy', rhos)
np.save('x_lc_vec_3d_grid_search_multi_param_1_9_24.npy', x_lc_vec)


#Now find alpha and beta using the Lu alogirthm


#Goal is take this solution and rederive
