# cy_abs_sum.pyx
#import numpy as np
#cimport numpy as cnp

# Cython function to calculate absolute values and sum them
#def cy_abs_sum(cnp.ndarray[cnp.float64_t, ndim=1] data):
#    cdef int n = data.shape[0]
#    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
#    cdef double total_sum = 0.0
#    cdef int i

#    for i in range(n):
#        result[i] = abs(data[i])  # Python abs() is used here but should be fine for floats
#        total_sum += result[i]
    
#    return result, total_sum
def fibonacci(int n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)