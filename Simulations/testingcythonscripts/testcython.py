import numpy as np
import sys
import time

# Add the directory containing the .so file to the Python path if necessary
sys.path.append('/home/kimjosy/LocReg_Regularization-1/Simulations')

# Import the Cython Fibonacci function
from cy_abs_sum import fibonacci

# NumPy Fibonacci function
def numpy_fibonacci(n):
    if n <= 1:
        return n
    return numpy_fibonacci(n - 1) + numpy_fibonacci(n - 2)

# Function to time another function
def time_function(func, *args):
    start_time = time.time()  # Record the start time
    result = func(*args)      # Call the function
    end_time = time.time()    # Record the end time
    return result, end_time - start_time  # Return the result and elapsed time

# Example: Calculate Fibonacci of 30 (increased for better timing comparison)
n = 30

# Time the Cython implementation
cython_result, cython_time = time_function(fibonacci, n)
print(f"Cython Fibonacci({n}) =", cython_result, "Time:", cython_time)

# Time the NumPy implementation
numpy_result, numpy_time = time_function(numpy_fibonacci, n)
print(f"NumPy Fibonacci({n}) =", numpy_result, "Time:", numpy_time)