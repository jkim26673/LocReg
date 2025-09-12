import numpy as np
cimport numpy as cnp
from cython.parallel import prange
cnp.import_array()
def cylongloop():
    i = 1
    while i < 1e8:
        i += 1
    return i

ctypedef cnp.double_t DTYPE_t  # Use double for floating-point numbers

cpdef np_cylongloop(cnp.ndarray[cnp.float64_t, ndim=2] A, cnp.ndarray[cnp.float64_t, ndim=1] x):
    b = 1000 * (A @ x)  # Vectorized operation
    return b


