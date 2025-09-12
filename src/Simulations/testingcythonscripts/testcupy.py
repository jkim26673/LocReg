
# import cupy as cu
# import numpy as np
# import time

# # Create a larger sample data array
# data = np.array([1.0, -2.0, 3.0, -4.0] * 1000000)  # Adjust size as needed for noticeable performance difference

# # Measure time for NumPy operations
# start_time_np = time.time()
# np_abs = np.abs(data)
# np_sum = np.sum(np_abs)
# end_time_np = time.time()
# np_time = end_time_np - start_time_np

# # Transfer data to CuPy
# data_cp = cu.array(data)

# # Measure time for CuPy operations
# start_time_cu = time.time()
# cu_abs = cu.abs(data_cp)
# cu_sum = cu.sum(cu_abs)
# end_time_cu = time.time()
# cu_time = end_time_cu - start_time_cu

# # Transfer result back to NumPy for comparison
# cu_sum_np = cu.asnumpy(cu_sum)

# print("NumPy abs (first 10 elements):", np_abs[:10])  # Print first 10 elements for brevity
# print("NumPy sum of abs:", np_sum)

# print("CuPy abs (first 10 elements):", cu.asnumpy(cu_abs)[:10])  # Print first 10 elements for brevity
# print("CuPy sum of abs (converted to NumPy):", cu_sum_np)

# print(f"NumPy execution time: {np_time:.4f} seconds")
# print(f"CuPy execution time: {cu_time:.4f} seconds")


import numpy as np
import time
import cy_abs_sum

# Create a larger sample data array
data = np.array([1.0, -2.0, 3.0, -4.0] * 1000000, dtype=np.float64)  # Adjust size as needed for noticeable performance difference

# Measure time for NumPy operations
start_time_np = time.time()
np_abs = np.abs(data)
np_sum = np.sum(np_abs)
end_time_np = time.time()
np_time = end_time_np - start_time_np

# Measure time for Cython operations
start_time_cy = time.time()
cy_abs, cy_sum = cy_abs_sum.cy_abs_sum(data)
end_time_cy = time.time()
cy_time = end_time_cy - start_time_cy

print("NumPy abs (first 10 elements):", np_abs[:10])  # Print first 10 elements for brevity
print("NumPy sum of abs:", np_sum)

print("Cython abs (first 10 elements):", cy_abs[:10])  # Print first 10 elements for brevity
print("Cython sum of abs:", cy_sum)

print(f"NumPy execution time: {np_time:.4f} seconds")
print(f"Cython execution time: {cy_time:.4f} seconds")