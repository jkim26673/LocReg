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
    
    # Use NumPy operations here as needed
    cdef cnp.ndarray[DTYPE_t, ndim=2] A = G.T @ G + np.diag(lam_vec)
    cdef cnp.ndarray[DTYPE_t] ep4 = np.ones(n_cols, dtype=DTYPE) * 1e-2  # eps
    cdef cnp.ndarray[DTYPE_t] b = G.T @ data_noisy + (G.T @ G @ ep4) + ep4 * lam_vec

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