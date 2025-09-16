
# import sys
# sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
# import numpy as np
# from regu.baart import baart
# from regu.foxgood import foxgood
# from regu.phillips import phillips
# from regu.csvd import csvd
# from regu.deriv2 import deriv2
# from regu.gravity import gravity
# from regu.heat import heat
# from regu.shaw import shaw
# from numpy.linalg import norm
# import matplotlib.pyplot as plt
# from regu.l_curve import l_curve
# import numpy as np
# from regu.discrep import discrep
# from math import factorial
# from regu.discrep import discrep
# import numpy as np
# from scipy.interpolate import lagrange
# # from pykalman import KalmanFilter, UnscentedKalmanFilter
# import numpy as np
# from regu.find_corner import find_corner
# import matplotlib.pyplot as plt
# from Utilities_functions.LocReg_unconstrained import LocReg_unconstrained
# from regu.tikhonov import tikhonov
from utils.load_imports.load_classical import *

#Try L1 method

n = 1000
nT2 = n
T2 = np.linspace(-np.pi/2,np.pi/2,n)
TE = T2
# G,data_noiseless,g = baart(n, nargout=3)

# G,data_noiseless,g = phillips(n, nargout = 3) #knee doesnt work here
# G,data_noiseless,g = baart(n, nargout=3) #knee doesn't work here
G, data_noiseless, g = gravity(n, nargin=1) #knee and corner doesn't work here

# G,data_noiseless,g = phillips(n, nargout=3) 

# G, data_noiseless, g = shaw(n, nargout=3)
U,s,V = csvd(G, tst = None, nargin = 1, nargout = 3)
SNR = 30


SD_noise= 1/SNR*max(abs(data_noiseless))
noise = np.random.normal(0,SD_noise, data_noiseless.shape)
data_noisy = data_noiseless + noise
Lambda_vec = np.logspace(-5,5,40)
nLambda = len(Lambda_vec)

UD_fix = U.T @ data_noisy
W = V @ np.diag(UD_fix)
GWpart = G @ W

# DP
delta = norm(noise)*1.05
f_rec_DPA,lambda_DP = discrep(U,s,V,data_noisy,delta,  x_0= None, nargin = 5)
x_DP = np.zeros(len(s))
for i in range(len(s)):
    x_DP[i] = s[i] / (s[i]**2 + lambda_DP**2)
f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)
x_ini = np.zeros(len(s))
for i in range(len(s)):
    x_ini[i] = s[i] / (s[i] ** 2 + lambda_DP ** 2)

ep1 = 1e-2
# % 1/(|x|+ep1)
ep2 = 1e-1
# % norm(dx)/norm(x)
ep3 = 1e-2
# % norm(x_(k-1) - x_k)/norm(x_(k-1))
ep4 = 1e-4 
x0_ini = f_rec_DP
f_rec_loc,lambda_loc = LocReg_unconstrained (data_noisy, G, x0_ini, lambda_DP, ep1,ep2,ep3 )

np.linalg.norm(g-f_rec_DP)
np.linalg.norm(g-f_rec_loc)

num_columns = G.shape[1]
x_rec = np.zeros(num_columns)
projection_temp = np.zeros(len(data_noisy))
projection_norms = np.zeros(num_columns)

for i in range(num_columns):
    data_noisy = data_noisy - projection_temp
    u = data_noisy
    x_rec[i] = np.dot(u, GWpart[:, i]) / (np.dot(GWpart[:, i], GWpart[:, i]) + 1e-9)
        # #run the below code for heat_prob
        # x_rec[i] = np.dot(u, GWpart[:, i]) / (np.dot(GWpart[:, i], GWpart[:, i]) + 1e-4)
    projection_temp = x_rec[i] * GWpart[:,i]
    projection_norms[i] = np.linalg.norm(projection_temp)

plt.plot(projection_norms)
plt.show()
from kneed import KneeLocator

def new_knee(proj_norms):
    end = min(proj_norms)
    index = 0
    mod_proj_norms = proj_norms
    while np.abs(proj_norms[index] - proj_norms[index+1]) > np.abs(np.median(projection_norms) + end)/2:
        scale = range(1, len(proj_norms)+1)
        kn = KneeLocator(scale, proj_norms, curve='convex', direction='decreasing', interp_method = "polynomial")
        index = kn.knee
        proj_norms = proj_norms[index:]
        # proj_norms = proj_norms[index:]
        print(len(proj_norms))
        print(proj_norms[index] - end)
        # kn.plot_knee()
    # scale = range(1, len(proj_norms)+1)
    # kn = KneeLocator(scale, proj_norms, curve='convex', direction='decreasing')
    # kn.plot_knee()
    return index

ireg_corner_nknee = new_knee(projection_norms)



scale = range(1,  len(projection_norms)+1)
    # Choose num_of_var_lambdas
kn = KneeLocator(scale, projection_norms, curve='convex', direction='decreasing', interp_method = "polynomial")

ireg_corner_kn = kn.knee
ireg_corner_kn

kn.plot_knee()
plt.show()

corner_reg = find_corner(projection_norms)
corner_reg
f_rec_DP = W @ x_DP

# Calculate p0 and lambda
DP = p0 = f_rec_DP

lambda_result = s / (np.linalg.solve(W, g)) - s**2
# lambda_approx = 0.1 / ((DP/2)**2 + np.log(np.abs(DP))**2 + 1)
lambda_approx = SD_noise/(DP + 1)

# lambda_approx_test = lambda_approx
# lambda_approx[:10] = lambda_result[:10]
# orig = lambda_approx
# test2 = lambda_approx.copy()
# test3 = lambda_approx.copy()
# test4 = lambda_approx.copy()


reconst = x_rec.copy()
new_err = []
DP_err = []
knee_err = []
corner_err = []
new_knee_err = []

# plt.plot(lambda_approx)
# plt.show()
for i in range(len(lambda_approx)):
    orig = reconst.copy()
    test2 = reconst.copy()
    test3 = reconst.copy()
    test4 = reconst.copy()

    # orig[:i] = lambda_result[:i]
    # test2[:ireg_corner_kn] = lambda_result[:ireg_corner_kn]
    # test3[:corner_reg] = lambda_result[:corner_reg]
    # test4[:ireg_corner_nknee] = lambda_result[:ireg_corner_nknee]
    orig[i:] = x_ini[i:]
    test2[ireg_corner_kn:] = x_ini[ireg_corner_kn:]
    test3[corner_reg:] = x_ini[corner_reg:]
    test4[ireg_corner_nknee:] = x_ini[ireg_corner_nknee:]


    # p1 = W @ (s / (s**2 + orig))
    # p2 = W @ (s / (s**2 + test2))
    # p3 = W @ (s / (s**2 + test3))
    # p4 = W @ (s / (s**2 + test4))

    p1 = W @ orig
    p2 = W @ test2
    p3 = W @ test3
    p4 = W @ test4

    DP =p0
    orange = p1
    knee = p2
    corner = p3
    newknee = p4

    #Calculate hte error between GT and p1 vs GT and p0
    new_err.append(np.linalg.norm(g-orange))
    knee_err.append(np.linalg.norm(g-knee))
    corner_err.append(np.linalg.norm(g-corner))
    new_knee_err.append(np.linalg.norm(g-newknee))

DP_err = [np.linalg.norm(g-DP)] * len(g)

DP_err[0]
knee_err[0]
new_err[0]
new_knee_err[0]
corner_err[0]

plt.plot(knee_err, label = 'knee')
plt.plot(DP_err, label = 'dp')
plt.plot(new_err, label = 'locreg')
plt.plot(new_knee_err, label = 'new_knee')
plt.plot(corner_err, label = 'corner')
plt.legend(['knee', "DP", "locreg_orig", 'new_knee','corner'])
plt.show()

np.argmin(new_err)
np.argmin(knee_err)
np.argmin(new_knee_err)

scale = range(1, len(knee_err)+1)
    # Choose num_of_var_lambdas
err_kn = KneeLocator(scale, knee_err, curve='convex', direction='decreasing')
err_kn.plot_knee()
plt.show()
find_corner(new_err)



test2[:ireg_corner_kn] = lambda_result[:ireg_corner_kn]
test3[:10] = lambda_result[:10]

# lambda_approx_test[:2] = lambda_mod_result[:2]

# Calculate p1
p1 = W @ (s / (s**2 + test2))

# p1_nn = W @ (s / (s**2 + lambda_approx_test))



plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(p0, label='DP')
plt.plot(p1, label='Method with first 10 lambdas of GT')
# plt.plot(p1_nn, label='p1_nn')
plt.plot(g, label='GT')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(lambda_result, label='lambda GT')
plt.plot(lambda_approx, label='lambda approx')
# plt.plot(lambda_approx_test, label='lambda approx_nn')

plt.legend()

plt.show()
