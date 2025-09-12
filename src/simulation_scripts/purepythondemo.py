import numpy as np
def longloop():
    i = 1
    while i < 1e8:
        i += 1
    return i

# def longloop():
#     m = 23
#     for i in range(200):
#         for j in range(200):
#             for k in range(200):
#                 l = (i + j + k) * m
#                 m += 2
#     return l

def nplongloop(A, x):
    b = np.zeros(A.shape[0])
    for i in range(1000):
        b += A @ x
    return b

def nplongloop_vec(A, x):
    b = 1000 * (A @ x)  # This computes the dot product and scales the result
    return b


