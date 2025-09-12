import numpy as np
from regu.baart import baart
from regu.foxgood import foxgood
from regu.phillips import phillips
from regu.csvd import csvd
from regu.deriv2 import deriv2
from regu.gravity import gravity
from regu.heat import heat
from regu.shaw import shaw
from numpy.linalg import norm
import matplotlib.pyplot as plt
from regu.l_curve import l_curve
import numpy as np
from math import factorial
from regu.discrep import discrep
import numpy as np
from scipy.interpolate import lagrange
# from pykalman import KalmanFilter, UnscentedKalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.signal import gaussian
from scipy.ndimage import convolve1d



def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    val = np.convolve( m[::-1], y, mode='valid')
    return val

def update_array_before_index(original_array, index, new_value):
    updated_array = original_array.copy()  # Create a copy of the original array to avoid modifying it directly
    for i in range(index):
        updated_array[i] = new_value
    return updated_array

n = 1000
nT2 = n
T2 = np.linspace(-np.pi/2,np.pi/2,n)
TE = T2
# G,data_noiseless,g = baart(n, nargout=3)
G,data_noiseless,g = foxgood(n)
#G,data_noiseless,g = phillips(n, nargout=3)
U,s,V = csvd(G, tst = None, nargin = 1, nargout = 3)
SNR = 30


SD_noise= 1/SNR*max(abs(data_noiseless))
noise = np.random.normal(0,SD_noise, data_noiseless.shape)
data_noisy = data_noiseless + noise
Lambda_vec = np.logspace(-5,5,40)
nLambda = len(Lambda_vec)

UD_fix = U.T @ data_noisy
W = V @ np.diag(UD_fix)
# DP
delta = norm(noise)*1.05
f_rec_DPA,lambda_DP = discrep(U,s,V,data_noisy,delta,  x_0= None, nargin = 5)
x_DP = np.zeros(len(s))
for i in range(len(s)):
    x_DP[i] = s[i] / (s[i]**2 + lambda_DP**2)
f_rec_DP = W @ x_DP

# Calculate p0 and lambda
DP = p0 = f_rec_DP

lambda_result = s / (np.linalg.solve(W, g)) - s**2
lambda_approx = 0.1 / (DP + 1)

# lambda_approx_test = lambda_approx
# lambda_approx[:10] = lambda_result[:10]

lambda_approx[:10] = lambda_result[:10]
# lambda_approx_test[:2] = lambda_mod_result[:2]

# Calculate p1
p1 = W @ (s / (s**2 + lambda_approx))

# p1_nn = W @ (s / (s**2 + lambda_approx_test))

DP =p0
orange = p1
#Calculate hte error between GT and p1 vs GT and p0
np.linalg.norm(g-DP)
np.linalg.norm(g-orange)

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




import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor


filter_sigma = 100.0
filter_size = int(6 * filter_sigma)
gaussian_filter = gaussian(filter_size, filter_sigma)
gaussian_filter /= gaussian_filter.sum()
data_filtered = convolve1d(p0, gaussian_filter, mode='constant')

smoothed_data = savitzky_golay(p0, window_size=51, order=10)
# kde = KernelDensity(bandwidth=1)  # Initialize the KDE model
# kde.fit(smoothed_data.reshape(-1, 1))  # Fit the KDE model to the smoothed data
# g_estimated = np.exp(kde.score_samples(T2.reshape(-1, 1)))  # Estimate the PDF at T2 points
T2_range = np.linspace(-np.pi/2, np.pi/2, 1000)  # Adjust the range and number of points as needed

kernel_regression_estimates = []

# Define the kernel function (e.g., Gaussian kernel)
def gaussian_kernel(x, x_i, bandwidth):
    return np.exp(-(x - x_i) ** 2 / (2 * bandwidth ** 2))

# Define the bandwidth for the kernel
bandwidth = 0.1  # Adjust the bandwidth as needed

# Iterate over each data point in T2_range and apply kernel regression
for point in T2_range:
    # Compute the kernel weights for each data point
    kernel_weights = np.array([gaussian_kernel(point, x_i, bandwidth) for x_i in T2])
    
    # Perform weighted averaging to estimate the function value
    estimated_value = np.sum(kernel_weights * smoothed_data) / np.sum(kernel_weights)
    
    kernel_regression_estimates.append(estimated_value)

# Convert the list of kernel regression estimates to a NumPy array
g_estimated = np.array(kernel_regression_estimates)


# Create a range of values for T2 for PDF estimation


# # Estimate the PDF using the KDE model for the specified range of values
# log_density_estimate = kde.score_samples(T2_range.reshape(-1, 1))

# # Convert the log-density estimate to actual density (PDF)
# g_estimated = np.exp(log_density_estimate)

plt.figure()
plt.plot(g, label='True g')
plt.plot(data_noiseless, label='Noiseless Data', color='green')
# plt.plot(data_noisy, label='Noisy Data', color='red')

# plt.plot(smoothed_data, label='Smoothed Data', color='blue')
plt.plot(g_estimated, label='Estimated g')
plt.legend()
plt.xlabel('T2')
plt.ylabel('Amplitude')
plt.title('True and Estimated g')
plt.show()



# kde = KernelDensity()  # Initialize the KDE model
# kde.fit(g_estimated.reshape(-1, 1))  # Fit the KDE model to the smoothed data
# g_estimated = np.exp(kde.score_samples(T2.reshape(-1, 1)))  # Estimate the PDF at T2 points


# Plotting
plt.figure()
plt.plot(g, label='True g')
plt.plot(g_estimated, label='Estimated g')
plt.legend()
plt.xlabel('T2')
plt.ylabel('Amplitude')
plt.title('True and Estimated g')
plt.show()

# Bandwidth selection for KDE
bandwidths = [0.1, 0.5, 1, 2]

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.api as sm


# #Assume our ground truth is unknown; use kernel density estimation


# # Generate random data similar to 'data_noisy'
# # Estimate the PDF using Kernel Density Estimation (KDE)
# # Kalman filter setup
# F = np.eye(n)  # Transition matrix
# H = np.ones((1, n))  # Observation matrix

# kf = KalmanFilter(transition_matrices=F, observation_matrices=H)
# filtered_state_means, filtered_state_covariances = kf.filter(data_noisy)

# # Smooth the filtered state estimates
# smoothed_state_means, smoothed_state_covariances = kf.smooth(data_noisy)

# # Obtain the filtered data
# filtered_data = smoothed_state_means.flatten()

# # Perform KDE on the filtered data
# kde = KernelDensity(bandwidth = 0.1)  # Initialize the KDE model
# kde.fit(filtered_data.reshape(-1, 1))  # Fit the KDE model to the filtered data
# g_estimated = np.exp(kde.score_samples(T2.reshape(-1, 1)))  # Estimate the PDF at T2 points
\
# from filterpy.kalman import ExtendedKalmanFilter, rts_smoother

# # Create an instance of the RTS smoother
# ekf = ExtendedKalmanFilter(2, 2)
# smoother = rts_smoother(ekf)

# # Smooth the state estimates
# smoothed_state_means, smoothed_state_covariances = smoother.smooth(filtered_state_mean_arr, filtered_state_covariances_arr)
# Perform KDE on the smoothed data


# kde = KernelDensity(bandwidth=0.1)  # Initialize the KDE model with a starting bandwidth

# # Perform AKDE to estimate the PDF of g
# kde.fit(data_noisy.reshape(-1, 1))  # Fit the KDE model to the noisy data
# g_estimated = np.exp(kde.score_samples(T2.reshape(-1, 1)))



# Plot the original data and the generated function for comparison
plt.plot(g, label='Original Data')
plt.plot(g_estimated, label='Approximated Function')
plt.xlabel('x')
plt.ylabel('g')
plt.legend()
plt.show() 


# %  if you know GT
# % then
# % x_GT = W\g 
# % we will know a GT lambda
##
# % p0 = g;
#Ground truth
lambda_result = s / (np.linalg.solve(W, g)) - s**2


# import numpy as np
# from scipy.interpolate import lagrange
# x= np.arange(1,len(g) + 1)
# y = np.linalg.solve(W, g)
# poly = lagrange(x, y)

#lambda_result[10:] = np.inf

# lsq = np.linalg.solve(W, g) 
# lsq[lsq<0] = 0
# ep = 1e-4
# lambda_mod_result = s / (lsq + ep) - s**2

import numpy as np

# Calculate lambda_approx
lambda_approx = 0.1 / (p0 + 1)


first_ten_lambda_approx = lambda_approx[:10]

# Calculate the corresponding 'lambda_result' values for the first ten 'lambda_approx' values
lambda_approxt = s / (1 / lambda_approx - s**2)

def sigmoid_transform(x):
    return 1 / (1 + np.exp(-x))

from scipy.stats import boxcox

lambda_approx_sing  = s / (np.linalg.solve(W, (p0 * data_noisy))) - s**2
lambda_approx_singt = 1/(np.exp(-lambda_approx_sing**g_estimated) - s)
# lambda_approx_sing = (s / (np.linalg.solve(W, - ((np.abs(p0*5))**(1/2)) * ((data_noisy)**2 + np.cos(W.T @ data_noisy))  / (np.cos(p0)))) - s**2))

# real = np.linalg.solve(W, g)
# test = np.linalg.solve(W,  (- (data_noisy) * np.sqrt(data_noisy /(p0 * np.cos(p0)))))

lambda_approx_sing[0:10]
#lambda_approx_sing = savitzky_golay(lambda_approx_sing, 51, 25)


plt.plot(lambda_result[0:10], label = 'gt')
plt.plot(lambda_approxt[0:10], label = 'exp')
# plt.plot(lambda_approx[0:10], label = 'p0')
plt.legend()
plt.show()



# lambda_approx_sing = (s / (test) - s**2).astype(np.float64)

# #Locate the elbow point
# from kneed import KneeLocator
# x = range(1, len(lambda_approx_sing)+1)
# kn = KneeLocator(x, lambda_approx_sing, curve='convex', direction='decreasing')
# print(kn.knee)

# lambda_approx_sing = update_array_before_index(lambda_approx_sing, kn.knee, lambda_approx_sing[kn.knee])


# plt.plot(g, label = 'gt')
# plt.legend()
# plt.show()
# 

# def gen_opt_lambda(lambda_list):
#     err_list = []
#     for i in range(len(lambda_list)):
#         lambda_approx[:i] = lambda_list[:i]
#         p1 = W @ (s / (s**2 + lambda_approx))
#         #Use l2 norm as a metric
#         err = np.linalg.norm(g-p1) 
#         err_list.append(err)
#     lam_indx = np.argmin(np.abs(np.array(err_list) - reg_corner))
#     return lam_indx, err_list

# lam_indx, err  = gen_opt_lambda(lambda_result)


# import matplotlib.pyplot as plt
# plt.xlabel('Values')
# plt.ylabel('Error')
# plt.plot(x, err, 'bx-')
# plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
# plt.show()
# #Error looks like log-log plot from lcurve


#Try non-negative

#It seems that if you include all the lambdas ,it's not suprising that you get the whole curve(b/c thats the ground truth)
#hard to do because you don't know the gt; try to get the first few values ( how many values do you need to )

lambda_approx_test = lambda_approx
# lambda_approx[:10] = lambda_result[:10]

lambda_approx[:10] = lambda_approx_sing[:10]
# lambda_approx_test[:2] = lambda_mod_result[:2]

# Calculate p1
p1 = W @ (s / (s**2 + lambda_approx))

p1s = W @ (s / (s**2 + lambda_approx_sing))

# p1_nn = W @ (s / (s**2 + lambda_approx_test))


#Calculate hte error between GT and p1 vs GT and p0
np.linalg.norm(g-p0)
np.linalg.norm(g-p1)
np.linalg.norm(g-p1s)




#Try the oracle lambda approach


#need to define a good cutoff point; goal is to find the threshold for lambda_approx


#find the cutoff point for baart (first 10 or 8)
#try to find what lambda_approx is based on p_0 or without gt


#Must be replicable

#Methods:

#Try with arbitrary constant

#tunes with lambda
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(p0, label='p0')
plt.plot(p1, label='p1')
# plt.plot(p1_nn, label='p1_nn')
plt.plot(g, label='GT')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(lambda_result, label='lambda GT')
plt.plot(lambda_approx, label='lambda approx')
# plt.plot(lambda_approx_test, label='lambda approx_nn')

plt.legend()

plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(p0, label='lambda_result')
plt.plot(p1, label='lambda_approx')
plt.plot(p1s, label='p1_s')
# plt.plot(g, label='GT')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(lambda_result, label='lambda GT')
plt.plot(lambda_approx, label='lambda approx')
plt.plot(lambda_approx_sing, label='lambda approx sing')

# plt.plot(lambda_approx_test, label='lambda approx_nn')

plt.legend()

plt.show()