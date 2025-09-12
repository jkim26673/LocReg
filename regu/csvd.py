#CSVD Compact singular value decomposition.

# s = csvd(A)
# [U,s,V] = csvd(A)
# [U,s,V] = csvd(A,'full')

# Computes the compact form of the SVD of A:
#    A = U*diag(s)*V',
# where
#    U  is  m-by-min(m,n)
#    s  is  min(m,n)-by-1
#    V  is  n-by-min(m,n).
#
# If a second argument is present, the full U and V are returned.

# Per Christian Hansen, IMM, 06/22/93.

import numpy as np
import mpmath
from scipy.sparse import csr_matrix
import scipy.linalg as sla

def csvd(A, tst, nargin, nargout):
    full_A = csr_matrix(A).toarray().astype('float')  # Convert to mpf (arbitrary precision floating-point)
    if (nargin == 1):
        if (nargout > 1):
            m,n = A.shape
            if (m >= n):
                U,s,V = np.linalg.svd(full_A)
                # U,s,V = sla.svd(A,full_matrices = False, lapack_driver = 'gesvd')
                V = V.T
            else:
                V,s,U = np.linalg.svd(full_A.T, full_matrices=False, compute_uv=True, hermitian=False)
                #V,s,U = mpmath.svd(full_A.T)
                U = U.T
        else:
            # U,s,V = sla.svd(full_A, full_matrices=True)
            U,s,V = np.linalg.svd(full_A, full_matrices=True)
            #U,s,V = mpmath.svd(full_A)
            U = s
    else:
        if (nargout > 1):
            U,s,V = np.linalg.svd(full_A, full_matrices=True)
            #U,s,V = mpmath.svd(full_A)
        else:
            U,s,V = np.linalg.svd(full_A, full_matrices=True)
            #U,s,V = mpmath.svd(full_A,lapack_driver = 'gesvd', full_matrices=False)
            U = s

    return U,s,V

# import numpy as np
# import mpmath
# import mpnum
# from scipy.sparse import csr_matrix

# def csvd(A, tst, nargin, nargout, precision=53):
#     # Set the precision for mpmath
#     mpmath.mp.dps = precision
    
#     full_A = csr_matrix(A).toarray().astype(mpmath.mpf)  # Convert to mpf (arbitrary precision floating-point)
    
#     if (nargin == 1):
#         if (nargout > 1):
#             U, s, V = mpnum.svd(full_A, full_matrices=False)
#             V = V.T
#         else:
#             U, s, V = mpnum.svd(full_A, full_matrices=False)
#             U = s
#     else:
#         if (nargout > 1):
#             U, s, V = mpnum.svd(full_A, full_matrices=False)
#         else:
#             U, s, V = mpnum.svd(full_A, full_matrices=False)
#             U = s

#     return U, s, V

# Example usage







# import numpy as np
# import mpmath
# from scipy.sparse import csr_matrix
# import scipy
# import torch

# def csvd(A, tst, nargin, nargout):
#     full_A = csr_matrix(A).toarray().astype(mpmath.mpf)  # Convert to mpf (arbitrary precision floating-point)
#     full_A_mpmath = mpmath.matrix(full_A)  # Convert to mpmath matrix
#     if (nargin == 1):
#         if (nargout > 1):
#             m, n = A.shape
#             if (m >= n):
#                 U, s, V = mpmath.svd(full_A_mpmath, full_matrices=False)
#                 V = V.T
#             else:
#                 V, s, U = mpmath.svd(full_A_mpmath.T, full_matrices=False)
#                 U = U.T
#         else:
#             U, s, V = mpmath.svd(full_A_mpmath, full_matrices=False)
#             U = s
#     else:
#         if (nargout > 1):
#             U, s, V = mpmath.svd(full_A_mpmath, full_matrices=False)
#         else:
#             U, s, V = mpmath.svd(full_A_mpmath, full_matrices=False)
#             U = s

#     return U, s, V