# this code does conventional l2 regularization with positivity constraints
# on f based on the model
# y = argmin{ || y - Af ||_2^2 + lambda^2 || f ||_2^2 } such that f >= 0
# This code is a python adaptation of the Chuan's discrep_L2.m function

#Function discrep2 takes in dat_noisy,A,SNR, and Lambda.
#Data types: 
#Lambda -- int | dat_noisy

import numpy as np
from src.regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm
from scipy.optimize import nnls
import cvxpy as cp


# def discrep_L2(dat_noisy, A, SNR, Lambda):
#     """
#     Perform L2 regularization with positivity constraints and find the optimal lambda.

#     Parameters:
#         dat_noisy (numpy.ndarray): Noisy data vector.
#         A (numpy.ndarray): Design matrix.
#         SNR (float): Signal-to-noise ratio.
#         Lambda (numpy.ndarray): Array of lambda squared values.

#     Returns:
#         f_rec_dp (numpy.ndarray): Regularized solution for the optimal lambda.
#         lam (float): Selected lambda.
#     """
#     m = A.shape[1]
#     n = A.shape[0]

#     # Compute threshold
#     threshold = 1.05 * np.sqrt(n) * np.max(dat_noisy) / SNR
    
#     # Perform batch processing
#     X, rho, _ = nonnegtik_hnorm_batch(A, dat_noisy, Lambda, '0')
    
#     # Compute threshold differences
#     threshold_diffs = np.sqrt(rho) - threshold
    
#     # Find the first lambda where the condition is met
#     valid_indices = np.where(threshold_diffs > 0)[0]
#     if valid_indices.size > 0:
#         idx = valid_indices[0]
#         lam = Lambda[idx]
#         f_rec_dp = X[:, idx]
#     else:
#         print("failed")
#         pass
#         # Handle the case where no lambda meets the condition (optional)
#         lam = Lambda[-1]  # or some other default behavior
#         f_rec_dp = X[:, -1]

#     return f_rec_dp, lam


# def discrep_L2(dat_noisy, A, SNR, Lambda):
#     # This function performs conventional L2 regularization with positivity constraints
#     # on 'f' based on the model: y = argmin{ || y - Af ||_2^2 + lambda^2 || f ||_2^2 } such that f >= 0
    
#     #Seems to take in lambda**2 and return lambda**2
    
#     # Load information
#     #nLambda = len(Lambda)
#     m = A.shape[1]
#     n = A.shape[0]

#     # threshold = 1.05 * np.sqrt(n) * np.max(dat_noisy) / SNR
#     # threshold = (1.05)**2 * np.sqrt(n) * np.max(dat_noisy) / SNR
#     threshold = (1.05) * np.sqrt(n) * np.max(dat_noisy) / SNR

#     lam = Lambda[0]
#     diff_rho = -1
#     X = np.empty(0)
    
#     while diff_rho <= 0:
#         lam = 1.5 * lam
#         X, rho, trash = nonnegtik_hnorm(A, dat_noisy, lam, '0', nargin = 4)
#         diff_rho = np.sqrt(rho) - threshold

#     f_rec_dp = X
#     # y_rec_temp = A @ f_rec_dp
#     # y_ratio = (np.sum(dat_noisy)/len(dat_noisy))/(np.sum(y_rec_temp)/len(y_rec_temp))
#     # f_rec_dp = y_ratio * f_rec_dp
#     return f_rec_dp, lam


def discrep_L2(dat_noisy, A, SNR, Lambda, noise):
    # This function performs conventional L2 regularization with positivity constraints
    # on 'f' based on the model: y = argmin{ || y - Af ||_2^2 + lambda^2 || f ||_2^2 } such that f >= 0
    
    #Seems to take in lambda**2 and return lambda**2
    
    # Load information
    #nLambda = len(Lambda)
    m = A.shape[1]
    n = A.shape[0]

    # threshold = 1.05 * np.sqrt(n) * np.max(dat_noisy) / SNR
    # threshold = (1.05)**2 * np.sqrt(n) * np.max(dat_noisy) / SNR
    threshold = (1.1) * np.sqrt(n) * noise
    # threshold = 1.05 * np.linalg.norm(noise)

    lam = Lambda[0]
    diff_rho = -1
    X = np.empty(0)
    
    while diff_rho <= 0:
        lam = 1.5 * lam
        X, rho, trash = nonnegtik_hnorm(A, dat_noisy, lam, '0', nargin = 4)
        diff_rho = np.sqrt(rho) - threshold

    f_rec_dp = X
    # y_rec_temp = A @ f_rec_dp
    # y_ratio = (np.sum(dat_noisy)/len(dat_noisy))/(np.sum(y_rec_temp)/len(y_rec_temp))
    # f_rec_dp = y_ratio * f_rec_dp
    return f_rec_dp, lam

def discrep_L2_sp(dat_noisy, A, SNR, Lambda, noise):
    # This function performs conventional L2 regularization with positivity constraints
    # on 'f' based on the model: y = argmin{ || y - Af ||_2^2 + lambda^2 || f ||_2^2 } such that f >= 0
    
    #Seems to take in lambda**2 and return lambda**2
    
    # Load information
    #nLambda = len(Lambda)
    m = A.shape[1]
    n = A.shape[0]

    if isinstance(noise, str):
        # threshold = (1.1) * np.sqrt(n) * np.max(dat_noisy) / SNR
        rnorm = nnls(A, dat_noisy)[1]
        threshold = 1.05 * rnorm
    # threshold = 1.05 * np.sqrt(n) * np.max(dat_noisy) / SNR
    # threshold = (1.05)**2 * np.sqrt(n) * np.max(dat_noisy) / SNR
    else:
        #pass
        # threshold = (1.05) * np.sqrt(n) * noise
        # try:
        #     threshold = 1.05 * np.sqrt(n) * np.max(dat_noisy) / SNR
        #     print("threshold real", threshold)
        #     print("SNR", SNR)
        # except Exception as e:
        threshold = 1.5 * np.sqrt(n) * np.max(dat_noisy) / SNR
        # print("An error occurred, applying safety factor:")
        # print("threshold real", threshold)
        # print("SNR", SNR)
    # threshold = 1.05 * np.linalg.norm(noise)
    #try except with higher safety factor...1.1; higher safety factor if estimate of SNR is not confident... for conservative estimate.

    lam = Lambda[0]
    diff_rho = -1
    X = np.empty(0)
    
    while diff_rho <= 0:
        lam = 1.5 * lam
        # lam = 3 * lam
        # X, rho, trash = nonnegtik_hnorm(A, dat_noisy, lam, '0', nargin = 4)
        try:
            X, rho, trash = nonnegtik_hnorm(A, dat_noisy, lam, '0', nargin = 4)
            if np.all(X[:-1] == 0):
                print("X is all zeros")
                X[:-1] = 1e-8
                print("X", X[:-1])
                break
            # print("rho",rho)
        except Exception as e:
            print(f"nonnegtik_hnorm failed with SNR: {SNR}")
            print(f"Error: {e}")
            break
        # print("diff_rho", diff_rho) 
        diff_rho = np.sqrt(rho) - threshold


    f_rec_dp = X
    # y_rec_temp = A @ f_rec_dp
    # y_ratio = (np.sum(dat_noisy)/len(dat_noisy))/(np.sum(y_rec_temp)/len(y_rec_temp))
    # f_rec_dp = y_ratio * f_rec_dp
    return f_rec_dp, lam


def discrep_L2_brain(dat_noisy, A, SNR, Lambda, noise):
    # This function performs conventional L2 regularization with positivity constraints
    # on 'f' based on the model: y = argmin{ || y - Af ||_2^2 + lambda^2 || f ||_2^2 } such that f >= 0
    
    #Seems to take in lambda**2 and return lambda**2
    
    # Load information
    #nLambda = len(Lambda)
    m = A.shape[1]
    n = A.shape[0]

    if isinstance(noise, str):
        # threshold = (1.1) * np.sqrt(n) * np.max(dat_noisy) / SNR
        rnorm = nnls(A, dat_noisy)[1]
        threshold = 1.05 * rnorm
    # threshold = 1.05 * np.sqrt(n) * np.max(dat_noisy) / SNR
    # threshold = (1.05)**2 * np.sqrt(n) * np.max(dat_noisy) / SNR
    else:
        #pass
        # threshold = (1.05) * np.sqrt(n) * noise
        # try:
        #     threshold = 1.05 * np.sqrt(n) * np.max(dat_noisy) / SNR
        #     print("threshold real", threshold)
        #     print("SNR", SNR)
        # except Exception as e:
        threshold = 1.1 * np.sqrt(n) * np.max(dat_noisy) / SNR
        # print("An error occurred, applying safety factor:")
        # print("threshold real", threshold)
        # print("SNR", SNR)
    # threshold = 1.05 * np.linalg.norm(noise)
    #try except with higher safety factor...1.1; higher safety factor if estimate of SNR is not confident... for conservative estimate.

    lam = Lambda[0]
    diff_rho = -1
    X = np.empty(0)
    
    while diff_rho <= 0:
        lam = 1.5 * lam
        # lam = 3 * lam
        # X, rho, trash = nonnegtik_hnorm(A, dat_noisy, lam, '0', nargin = 4)
        try:
            X, rho, trash = nonnegtik_hnorm(A, dat_noisy, lam, '0', nargin = 4)
            # print("rho",rho)
            if np.all(X[:-1] == 0):
                print("X is all zeros")
                # X[:-1] = 1e-6
                # print("X", X)
                break
            tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
            # all_close_to_zero = all(abs(x) < tolerance for x in X)
            # X = np.array(X)
            all_close_to_zero = np.all(np.abs(X) < tolerance)
            if all_close_to_zero:
                print("X is all close to zero")
                break
        except Exception as e:
            print(f"nonnegtik_hnorm failed with SNR: {SNR}")
            print(f"Error: {e}")
            # X = nnls(A, dat_noisy)[0]
            break
        diff_rho = np.sqrt(rho) - threshold

        # print("diff_rho", diff_rho) 

    f_rec_dp = X
    # y_rec_temp = A @ f_rec_dp
    # y_ratio = (np.sum(dat_noisy)/len(dat_noisy))/(np.sum(y_rec_temp)/len(y_rec_temp))
    # f_rec_dp = y_ratio * f_rec_dp
    return f_rec_dp, lam

