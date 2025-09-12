# sparse_solver.pyx
import numpy as np
cimport numpy as cnp
import cython
cnp.import_array()
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve

@cython.boundscheck(False)
@cython.wraparound(False)
def solvesparse(cnp.ndarray[cnp.float64_t, ndim=2] A, 
                 cnp.ndarray[cnp.float64_t, ndim=1] b):
    cdef int n = A.shape[0]
    # Use the sparse solver for efficiency
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.zeros(n, dtype=np.float64)
    # Solve the sparse linear system
    x[:] = spsolve(A, b)
    return x
