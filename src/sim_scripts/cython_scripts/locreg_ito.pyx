# Cython directives
# cython: language_level=3
import numpy as np
cimport numpy as cnp
cnp.import_array()

# Use float64 for higher precision
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import cvxpy as xpy  # Regular import, not cimport
from regu.nonnegtik_hnorm import nonnegtik_hnorm
import cython
from sparse_solver import solvesparse


@cython.boundscheck(False)
@cython.wraparound(False)
def minimize(cnp.ndarray[DTYPE_t] lam_vec, G, cnp.ndarray[DTYPE_t] data_noisy):
    cdef int n_cols = G.shape[1]
    y = xpy.Variable(n_cols)  # Use cvxpy Variable
    
    # Convert memory views to NumPy arrays
    cdef cnp.ndarray[DTYPE_t] lam_np = lam_vec

    # Use NumPy operations here as needed
    cdef cnp.ndarray[DTYPE_t, ndim=2] A = G.T @ G + np.diag(lam_np)
    cdef cnp.ndarray[DTYPE_t] ep4 = np.ones(n_cols, dtype=DTYPE) * 1e-2  # eps
    cdef cnp.ndarray[DTYPE_t] b = G.T @ data_noisy + (G.T @ G @ ep4) + ep4 * lam_np

    cost = xpy.norm(A @ y - b, 'fro')**2
    constraints = [y >= 0]
    problem = xpy.Problem(xpy.Minimize(cost), constraints)
    problem.solve(solver=xpy.MOSEK, mosek_params={
        'MSK_IPAR_INTPNT_MAX_ITERATIONS': '100',
        'MSK_IPAR_INTPNT_SOLVE_FORM': 'MSK_SOLVE_DUAL'
    }, verbose=False)

    sol = y.value
    sol = np.maximum(sol - 1e-2, 0)  # eps

    if sol is None or np.any([x is None for x in sol]):
        print("Solution contains None values, switching to nonnegtik_hnorm")
        sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)

    return sol, A, b

@cython.boundscheck(False)
@cython.wraparound(False)
def phi_resid( G, cnp.ndarray[DTYPE_t] param_vec, cnp.ndarray[DTYPE_t] data_noisy):
    return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

@cython.boundscheck(False)
@cython.wraparound(False)
def fixed_point_algo( gamma, cnp.ndarray[DTYPE_t] lam_vec,  G, cnp.ndarray[DTYPE_t] data_noisy, double choice_val, int maxiter):
    cdef cnp.ndarray[DTYPE_t] lam_curr = lam_vec
    cdef int n_cols = G.shape[1]
    cdef cnp.ndarray[DTYPE_t] f_old = np.ones(n_cols, dtype=DTYPE)
    cdef int k = 1
    cdef double ep_min = 1e-2
    
    while k <= maxiter:
        curr_f_rec, LHS, _ = minimize(lam_curr, G, data_noisy)
        if curr_f_rec is None or np.any(np.isnan(curr_f_rec)):
            print(f"curr_f_rec returns None after minimization for iteration {k}")
            continue
        
        curr_noise = G @ curr_f_rec - data_noisy
        LHS_sparse = csr_matrix(LHS)
        delta_p = solvesparse(LHS_sparse, G.T @ curr_noise)

        prev_norm = np.linalg.norm(delta_p)
        iterationval = 1
        
        while iterationval < 300:
            curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
            curr_noise = G @ curr_f_rec - data_noisy
            delta_p = solvesparse(LHS_sparse, G.T @ curr_noise)
            if np.abs(np.linalg.norm(delta_p) / prev_norm - 1) < choice_val:
                break
            prev_norm = np.linalg.norm(delta_p)
            iterationval += 1
        curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)

        phi_new = phi_resid(G, curr_f_rec, data_noisy)
        c = 1 / gamma
        psi_lam = np.array(curr_f_rec)

        assert np.all(np.abs(psi_lam) + ep_min) != 0
        lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))

        if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < 1e-2 or k == maxiter:
            if k == maxiter:
                print("Maximum Iteration Reached")
            return curr_f_rec, lam_curr, k
        else:
            ep_min = ep_min / 1.2
            if ep_min <= 1e-4:
                ep_min = 1e-4
            lam_curr = lam_new
            f_old = curr_f_rec
            k += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def LocReg_Ito_mod(cnp.ndarray[DTYPE_t] data_noisy, 
                    G, 
                   lam_ini, gamma_init, int maxiter):
    lam_vec = lam_ini * np.ones(G.shape[1], dtype=DTYPE)
    best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, G, data_noisy, 1e-5, maxiter)
    if best_f_rec1 is None:
        raise ValueError("Fixed-point algorithm failed to produce a valid result.")
    
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    zero_vec = np.zeros(len(best_f_rec1), dtype=DTYPE)
    zero_resid = phi_resid(G, zero_vec, data_noisy)
    gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25
    best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, G, data_noisy, 1e-5, maxiter)

    return best_f_rec2, fin_lam2, best_f_rec1, fin_lam1, iternum
