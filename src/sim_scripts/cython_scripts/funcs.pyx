import cython
# my_functions.pyx
# cython: language_level=3
import numpy as np
cimport numpy as cnp
cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import cvxpy as xpy  # Regular import, not cimport
# Cython directives
import sys
import os
print("Setting system path")
sys.path.append(".")  # Replace this path with the actual path to the parent directory of Utilities_functions
import numpy as np
from scipy.stats import norm as normsci
from scipy.linalg import norm as linalg_norm
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import pickle
from scipy.stats import wasserstein_distance, entropy
from Utilities_functions.discrep_L2 import discrep_L2
from Utilities_functions.GCV_NNLS import GCV_NNLS
from Utilities_functions.Lcurve import Lcurve
import pandas as pd
import cvxpy as cp
from scipy.linalg import svd
from regu.csvd import csvd
from regu.discrep import discrep
from Simulations.Ito_LocReg import Ito_LocReg
from Simulations.Ito_LocReg import *
from Utilities_functions.pasha_gcv import Tikhonov
from regu.l_curve import l_curve
from tqdm import tqdm
from Utilities_functions.tikhonov_vec import tikhonov_vec
import mosek
from ItoLocRegConst import LocReg_Ito_C,LocReg_Ito_C_2,LocReg_Ito_C_4
from regu.nonnegtik_hnorm import nonnegtik_hnorm
import multiprocess as mp
from multiprocessing import Pool, freeze_support
from multiprocessing import set_start_method
import functools
from datetime import date
import random
import cProfile
import pstats
import os
from datetime import date
from cython.parallel import prange
# cython: language_level=3
@cython.boundscheck(False)
@cython.wraparound(False)


# Function to minimize objective function
def minimize_OP(cnp.ndarray Alpha_vec, cnp.ndarray L, cnp.ndarray  data_noisy, cnp.ndarray G, int nT2, cnp.ndarray g):
    cdef int j
    cdef int alpha_len = len(Alpha_vec)
    # Prepare output arrays
    cdef cnp.ndarray[cnp.float64_t, ndim=2] OP_x_lc_vec = np.zeros((nT2, alpha_len), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] OP_rhos = np.zeros((alpha_len,), dtype=np.float64)
    for j in (range(len(Alpha_vec))):
        try:
            # Fallback to nonnegtik_hnorm
            sol, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j], '0', nargin=4)
        except Exception as e:
            print(f"Error in nonnegtik_hnorm: {e}")
            # If both methods fail, solve using cvxpy
            lam_vec = Alpha_vec[j] * np.ones(G.shape[1])
            A = (G.T @ G + np.diag(lam_vec))
            eps = 1e-2
            ep4 = np.ones(A.shape[1]) * eps
            b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vec
            
            y = cp.Variable(G.shape[1])
            cost = cp.norm(A @ y - b, 2)**2
            constraints = [y >= 0]
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.MOSEK, verbose=False)
            
            sol = y.value
            sol = sol - eps
            sol = np.maximum(sol, 0)

        OP_x_lc_vec[:, j] = sol
        OP_rhos[j] = np.linalg.norm(OP_x_lc_vec[:,j] - g, 2)**2

    OP_log_err_norm = np.log10(OP_rhos)
    min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)
    min_x = Alpha_vec[min_index[0]]
    min_z = np.min(OP_log_err_norm)
    OP_min_alpha1 = min_x
    OP_min_alpha1_ind = min_index[0]
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    return f_rec_OP_grid, OP_min_alpha1

# Function to calculate T2mu
def calc_T2mu(cnp.ndarray rps):
    cdef int nrps = len(rps)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] mps = rps / 2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] T2_left = 40 * np.ones(nrps, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] T2_mid = T2_left * mps
    cdef cnp.ndarray[cnp.float64_t, ndim=1] T2_right = T2_left * rps
    T2mu = np.column_stack((T2_left, T2_right))
    return T2mu

# Function to calculate sigma_i
def calc_sigma_i(int iter_i, cnp.ndarray diff_sigma):
    return diff_sigma[iter_i, :]

# Function to calculate rps_val
def calc_rps_val(int iter_j, cnp.ndarray rps):
    return rps[iter_j]

# Function to calculate difference in sigma
def calc_diff_sigma(int nsigma):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unif_sigma = np.linspace(2, 5, nsigma)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] diff_sigma = np.column_stack((unif_sigma, 3 * unif_sigma))
    return unif_sigma, diff_sigma

# Function to load Gaussian information
def load_Gaus(dict Gaus_info):
    T2 = Gaus_info['T2'].flatten()
    TE = Gaus_info['TE'].flatten()
    A = Gaus_info['A']
    Lambda = Gaus_info['Lambda'].reshape(-1, 1)
    n, m = A.shape
    SNR = Gaus_info['SNR']  # Ensure this is defined in Gaus_info
    return T2, TE, Lambda, A, m, SNR

# Function to calculate noisy data
def calc_dat_noisy(cnp.ndarray A, cnp.ndarray TE, cnp.ndarray IdealModel_weighted,  float SNR):
    dat_noiseless = A @ IdealModel_weighted
    noise = np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE))
    dat_noisy = dat_noiseless + noise
    return dat_noisy, noise

# Function to get weighted ideal model
# Function to calculate L2 error
def l2_error(cnp.ndarray IdealModel, 
             cnp.ndarray reconstr):
    true_norm = linalg_norm(IdealModel)
    err = linalg_norm(IdealModel - reconstr) / true_norm
    return err

# Function to calculate L2 error with shift
def l2_error_shift(float gamma, 
                   cnp.ndarray IdealModel, 
                   cnp.ndarray reconstr):
    
    true_norm = linalg_norm(IdealModel)
    shift_reconstr = np.interp(T2 + gamma, T2, reconstr)
    err = linalg_norm(IdealModel - shift_reconstr) / true_norm
    return err

# Function to calculate Wasserstein shift
def wass_shift(cnp.ndarray T2, 
               float gamma, 
               cnp.ndarray IdealModel, 
               cnp.ndarray reconstr):
    emd = wasserstein_distance(T2, T2 + gamma, u_weights=IdealModel, v_weights=reconstr)
    return emd

# Function to calculate Wasserstein error
def wass_error(cnp.ndarray T2, 
               cnp.ndarray IdealModel, 
               cnp.ndarray reconstr):
    emd = wasserstein_distance(T2, T2, u_weights=IdealModel, v_weights=reconstr)
    return emd

# Function to find the minimum beta
def find_min_beta(cnp.ndarray beta_list, 
                  cnp.ndarray metric_list):
    opt_ind = np.argmin(metric_list)
    opt_gam = beta_list[opt_ind]
    opt_err_score = metric_list[opt_ind]
    return opt_gam, opt_err_score

# Function to get scores
def get_scores(float gamma, cnp.ndarray T2, cnp.ndarray g, cnp.ndarray locreg, cnp.ndarray l1, cnp.ndarray l2):
    cdef list kl_scores_list = []
    cdef list l2_rmsscores_list = []
    cdef list wass_scores_list = []

    shifted_locreg = np.interp(T2 + gamma, T2, locreg)
    shifted_l2 = np.interp(T2 + gamma, T2, l2)
    shifted_l1 = np.interp(T2 + gamma, T2, l1)

    wass_scores_gamma = wass_shift(T2, gamma, g, locreg)

    return wass_scores_gamma

# Cython function to generate estimates
