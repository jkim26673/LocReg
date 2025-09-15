#!/usr/bin/env python3
import sys
sys.path.append(".")
from scipy.optimize import nnls
import numpy as np
import cvxpy as cp
import scipy
import os
import mosek
import cvxpy as cp
from regu.nonnegtik_hnorm import nonnegtik_hnorm

mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
# cython: language_level=3
# Declare Cython memory views for efficient array access
cimport numpy as np

# Cython function to minimize
def minimize(double[:, :] G, double[:] data_noisy, double[:] lam_vec):
    cdef double eps = 1e-2
    cdef int G1 = G.shape[1]
    cdef double[:] ep4 = np.ones(G1)
    cdef double[:, :] A = np.zeros((G1, G1))  # Allocate space for A matrix
    A = np.dot(G.T, G) + np.diag(lam_vec)
    ep4 = np.ones(G1) * eps
    b = np.dot(G.T, data_noisy) + np.dot(np.dot(G.T, G), ep4) + (ep4 * lam_vec)
    
    # Set up and solve the convex optimization problem using cvxpy
    y = cp.Variable(G.shape[1])
    cost = cp.norm(A @ y - b, 'fro')**2
    constraints = [y >= 0]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.MOSEK, mosek_params={
        'MSK_IPAR_INTPNT_MAX_ITERATIONS': '100',
        'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
    }, verbose=False)
    
    sol = y.value
    sol = np.maximum(sol - eps, 0)
    
    # Check if the solution is None
    if sol is None or np.any([x is None for x in sol]):
        print("Solution contains None values, switching to nonnegtik_hnorm")
        # Assuming `nonnegtik_hnorm` is defined elsewhere in your code.
        sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
    
    return sol, A, b

# Residual function
def phi_resid(np.ndarray[np.double_t, ndim=2] G, np.ndarray[np.double_t, ndim=1] param_vec, np.ndarray[np.double_t, ndim=1] data_noisy):
    return np.linalg.norm(np.dot(G, param_vec) - data_noisy, 2)**2

# Fixed point algorithm
def fixed_point_algo(double gamma, np.ndarray[np.double_t, ndim=1] lam_vec, double choice_val):
    cdef np.ndarray[np.double_t, ndim=1] lam_curr = lam_vec
    cdef double ep = 1e-2
    cdef double ep_min = 1e-2
    cdef np.ndarray[np.double_t, ndim=1] f_old = np.ones(G.shape[1])
    cdef int k = 1
    while True and k <= maxiter:
        try:
            curr_f_rec, LHS, _ = minimize(lam_curr)
            if curr_f_rec is None or np.any([x is None for x in curr_f_rec]):
                print(f"curr_f_rec returns None after minimization for iteration {k}")
                continue
        except Exception as e:
            print("An error occurred during minimization:", e)
            continue
        curr_noise = np.dot(G, curr_f_rec) - data_noisy
        L = np.linalg.cholesky(LHS)
        delta_p = scipy.linalg.cho_solve((L, True), np.dot(G.T, curr_noise))
        prev = np.linalg.norm(delta_p)
        iterationval = 1
        while iterationval < 200:
            curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
            curr_noise = np.dot(G, curr_f_rec) - data_noisy
            try:
                delta_p = scipy.linalg.cho_solve((L, True), np.dot(G.T, curr_noise))
            except RuntimeWarning:
                print("Error with delta_p calculation")
            if np.abs((np.linalg.norm(delta_p) / prev) - 1) < 1e-3:
                break
            prev = np.linalg.norm(delta_p)
            iterationval += 1
        curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
        phi_new = phi_resid(G, curr_f_rec, data_noisy)
        psi_lam = np.array(curr_f_rec)
        c = 1 / gamma
        lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))
        
        # Convergence check
        if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < 1e-2 or k == maxiter:
            if k == maxiter:
                print("Maximum Iteration Reached")
            return curr_f_rec, lam_curr, k
        else:
            ep_min = ep_min / 2
            if ep_min <= 8e-5:
                ep_min = 8e-5
            lam_curr = lam_new
            f_old = curr_f_rec
            k += 1

# Main method
def LocReg_Ito_mod(np.ndarray[np.double_t, ndim=2] data_noisy, np.ndarray[np.double_t, ndim=2] G, 
                   np.ndarray[np.double_t, ndim=1] lam_ini, double gamma_init, int maxiter):
    lam_vec = lam_ini * np.ones(G.shape[1])
    choice_val = 1e-5
    try:
        best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, choice_val)
    except Exception as e:
        print("Error in LocReg")
        print("lam_vec", lam_vec)
    
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G, zero_vec, data_noisy)
    gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25
    new_choice2 = 1e-5
    best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, new_choice2)
    x_normalized = best_f_rec2
    
    return x_normalized, fin_lam2, best_f_rec1, fin_lam1, iternum
