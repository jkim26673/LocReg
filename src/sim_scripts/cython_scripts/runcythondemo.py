import sys
sys.path.append(".")
from Simulations.purepythondemo import longloop, nplongloop, nplongloop_vec
from Simulations.cythondemo import cylongloop, np_cylongloop
import numpy as np
import timeit
import Simulations.cythondemo as cython_module

# List all available attributes and functions in the cythondemo module
print(dir(cython_module))


# # Create a random matrix A (150x200) and vector x (200)
# A = np.random.rand(150, 200)
# x = np.random.rand(200)

# # Test the functions
# result_longloop = cylongloop()
# result_nplongloop = np_cylongloop(A, x)

# print("Result of cylongloop:", result_longloop)
# print("Result of np_cylongloop:", result_nplongloop)

# Run and time the pure Python function
pure_python_result = longloop()
pure_python_execution_time = timeit.timeit(longloop, number=1)

# Run and time the Cython function
cython_result = cylongloop()
cython_execution_time = timeit.timeit(cylongloop, number=1)

# Calculate speedup factor
if cython_execution_time > 0:  # Prevent division by zero
    speedup_factor = pure_python_execution_time / cython_execution_time
else:
    speedup_factor = float('inf')  # Cython is infinitely faster if it runs instantly

# Print results and execution times in scientific notation
print(f"Pure Python Result: {pure_python_result}")
print(f"Execution time (Pure Python): {pure_python_execution_time:.2e} seconds")

print(f"Cython Result: {cython_result}")
print(f"Execution time (Cython): {cython_execution_time:.2e} seconds")

# Print the speedup factor
print(f"Speedup Factor (Cython vs. Pure Python): {speedup_factor:.2f}x")

####Incoroporating Numpy

A = np.random.rand(150, 200)
x = np.random.rand(200)

pure_python_result = nplongloop(A,x)
pure_python_execution_time = timeit.timeit(lambda: nplongloop(A, x), number=1)
print(f"Execution time (Pure Python): {pure_python_execution_time:.2e} seconds")

# Timing the Cython function
# Run and time the Cython function
cython_result = np_cylongloop(A,x)
cython_execution_time = timeit.timeit(lambda: np_cylongloop(A, x), number=1)
print(f"Execution time (Cython): {cython_execution_time:.2e} seconds")


# Calculate speedup factor
if cython_execution_time > 0:  # Prevent division by zero
    speedup_factor = pure_python_execution_time / cython_execution_time
else:
    speedup_factor = float('inf')  # Cython is infinitely faster if it runs instantly

# Print results and execution times in scientific notation
print(f"Numpy Python Result: {pure_python_result}")
print(f"Execution time (Numpy Python): {pure_python_execution_time:.2e} seconds")

print(f"Cython Result: {cython_result}")
print(f"Execution time (Cython): {cython_execution_time:.2e} seconds")

# Print the speedup factor
print(f"Speedup Factor (Cython vs. Numpy Python): {speedup_factor:.2f}x")

# parallel_cython_result = np_cylongloop_par(A,x)
# cython_execution_time = timeit.timeit(lambda: np_cylongloop_par(A, x), number=1)
# print(f"Execution time (Cython Parallelized): {cython_execution_time:.2e} seconds")

# if cython_execution_time > 0:  # Prevent division by zero
#     speedup_factor = pure_python_execution_time / cython_execution_time
# else:
#     speedup_factor = float('inf')  # Cython is infinitely faster if it runs instantly



####Incoroporating Numpy Vectorization

pure_python_result = nplongloop_vec(A,x)
pure_python_execution_time = timeit.timeit(lambda: nplongloop_vec(A, x), number=1)
print(f"Execution time (Numpy Vectorization): {pure_python_execution_time:.2e} seconds")

# Timing the Cython function
# Run and time the Cython function
cython_result = np_cylongloop(A,x)
cython_execution_time = timeit.timeit(lambda: np_cylongloop(A, x), number=1)
print(f"Execution time (Cython Vectorization): {cython_execution_time:.2e} seconds")


# Calculate speedup factor
if cython_execution_time > 0:  # Prevent division by zero
    speedup_factor = pure_python_execution_time / cython_execution_time
else:
    speedup_factor = float('inf')  # Cython is infinitely faster if it runs instantly

# Print results and execution times in scientific notation
print(f"Numpy Python Result: {pure_python_result}")
print(f"Execution time (Numpy Vectorization): {pure_python_execution_time:.2e} seconds")

print(f"Cython Result: {cython_result}")
print(f"Execution time (Cython Vectorization): {cython_execution_time:.2e} seconds")

# Print the speedup factor
print(f"Speedup Factor (Cython Vectorization vs. Numpy Vectorization): {speedup_factor:.2f}x")

