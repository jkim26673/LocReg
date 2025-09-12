import numpy as np
from scipy.optimize import nnls
#from Utilities_functions.lsqnonneg import lsqnonneg
import mosek 
import cvxpy as cp
import numpy as np
from cvxopt import solvers, matrix, spmatrix, mul
import itertools
from scipy import sparse
# import matlab.engine
# eng = matlab.engine.start_matlab()


def nonnegtik_hnorm_batch(A, b, lambdas, nm, R=None):
    """
    Batch processing for non-negative least squares with regularization.

    Parameters:
        A (numpy.ndarray): The design matrix.
        b (numpy.ndarray): The data vector.
        lambdas (numpy.ndarray): Array of lambda values (regularization parameters).
        nm (str): Regularization type ('0', '1', '2', '11', '22').
        R (numpy.ndarray, optional): Regularization matrix.

    Returns:
        X (numpy.ndarray): Solutions for each lambda.
        mdl_err (numpy.ndarray): Model errors for each lambda.
        xnorm (numpy.ndarray): Norms of the regularization terms for each lambda.
    """
    n = A.shape[1]
    num_lambdas = len(lambdas)
    
    # Initialize results
    X = np.zeros((n, num_lambdas))
    mdl_err = np.zeros(num_lambdas)
    xnorm = np.zeros(num_lambdas)

    # Build regularization matrices
    L0 = np.eye(n)
    L1 = -np.diag(np.ones(n)) + np.diag(np.ones(n - 1), 1)
    L1 = L1[:-1, :]
    L2 = (6 * np.diag(np.ones(n)) - 4 * np.diag(np.ones(n - 1), 1) - 4 * np.diag(np.ones(n - 1), -1) +
          np.diag(np.ones(n - 2), -2) + np.diag(np.ones(n - 2), 2))
    
    # Augment b
    Aug_b = np.concatenate((b, np.zeros(n)))

    for idx, lamb_da in enumerate(lambdas):
        if nm == '0':
            L = L0
        elif nm == '1':
            L = L0 + L1
        elif nm == '2':
            L = L0 + L1 + L2
        elif nm == '11':
            L = L1
        elif nm == '22':
            L = L2
        else:
            L = R

        # Construct augmented matrix
        Aug_A = np.concatenate((A, lamb_da * L), axis=0)

        # Solve the least squares problem
        x, _ = nnls(Aug_A, Aug_b)
        X[:, idx] = x

        # Compute the model error and norm of regularization term
        mdl_err[idx] = np.linalg.norm(A @ x - b) ** 2
        xnorm[idx] = np.linalg.norm(L @ x) ** 2

    return X, mdl_err, xnorm

def lsqnonneg(C, d):
    '''Linear least squares with nonnegativity constraints.

    (x, resnorm, residual) = lsqnonneg(C,d) returns the vector x that minimizes norm(d-C*x)
    subject to x >= 0, C and d must be real
    '''

    eps = 2.22e-16    # from matlab

    tol = 10*eps*np.linalg.norm(C,1)*(max(C.shape)+1)

    C = np.asarray(C)

    (m,n) = C.shape
    P = []
    R = [x for x in range(0,n)]

    x = np.zeros(n)

    resid = d - np.dot(C, x)
    w = np.dot(C.T, resid)

    count = 0

    # outer loop to put variables into set to hold positive coefficients
    while np.any(R) and np.max(w) > tol:

        j = np.argmax(w)
        P.append(j)
        R.remove(j)


        AP = np.zeros(C.shape)
        AP[:,P] = C[:,P]

        s=np.dot(np.linalg.pinv(AP), d)

        s[R] = 0
     
        while np.min(s) < 0:


            i = [i for i in P if s[i] <= 0]

            alpha = min(x[i]/(x[i] - s[i]))
            x = x + alpha*(s-x)

            j = [j for j in P if x[j] == 0]
            if j:
                R.append(*j)
                P.remove(j)
            
            AP = np.zeros(C.shape)
            AP[:,P] = C[:,P]
            s=np.dot(np.linalg.pinv(AP), d)
            s[R] = 0
     
        x = s
        resid = d - np.dot(C, x)

        w = np.dot(C.T, resid)

    return [x, sum(resid * resid), resid]

# '''
#     A simple library to solve constrained linear least squares problems
#     with sparse and dense matrices. Uses cvxopt library for 
#     optimization
# '''

# """__author__ = 'Valeriy Vishnevskiy'
# __email__ = 'valera.vishnevskiy@yandex.ru'
# __version__ = '1.0'
# __date__ = '22.11.2013'
# __license__ = 'WTFPL'"""


# def scipy_sparse_to_spmatrix(A):
#     coo = A.tocoo()
#     SP = spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())
#     return SP

# def spmatrix_sparse_to_scipy(A):
#     data = np.array(A.V).squeeze()
#     rows = np.array(A.I).squeeze()
#     cols = np.array(A.J).squeeze()
#     return sparse.coo_matrix( (data, (rows, cols)) )

# def sparse_None_vstack(A1, A2):
#     if A1 is None:
#         return A2
#     else:
#         return sparse.vstack([A1, A2])

# def numpy_None_vstack(A1, A2):
#     if A1 is None:
#         return A2
#     else:
#         return np.vstack([A1, A2])
        
# def numpy_None_concatenate(A1, A2):
#     if A1 is None:
#         return A2
#     else:
#         return np.concatenate([A1, A2])

# def get_shape(A):
#     if isinstance(C, spmatrix):
#         return C.size
#     else:
#         return C.shape

# def numpy_to_cvxopt_matrix(A):
#     if A is None:
#         return A
#     if sparse.issparse(A):
#         if isinstance(A, sparse.spmatrix):
#             return scipy_sparse_to_spmatrix(A)
#         else:
#             return A
#     else:
#         if isinstance(A, np.ndarray):
#             if A.ndim == 1:
#                 return matrix(A, (A.shape[0], 1), 'd')
#             else:
#                 return matrix(A, A.shape, 'd')
#         else:
#             return A

# def cvxopt_to_numpy_matrix(A):
#     if A is None:
#         return A
#     if isinstance(A, spmatrix):
#         return spmatrix_sparse_to_scipy(A)
#     elif isinstance(A, matrix):
#         return np.array(A).squeeze()
#     else:
#         return np.array(A).squeeze()
        

# def lsqlin(C, d, reg=0, A=None, b=None, Aeq=None, beq=None, \
#         lb=None, ub=None, x0=None, opts=None):
#     '''
#         Solve linear constrained l2-regularized least squares. Can 
#         handle both dense and sparse matrices. Matlab's lsqlin 
#         equivalent. It is actually wrapper around CVXOPT QP solver.

#             min_x ||C*x  - d||^2_2 + reg * ||x||^2_2
#             s.t.  A * x <= b
#                   Aeq * x = beq
#                   lb <= x <= ub

#         Input arguments:
#             C   is m x n dense or sparse matrix
#             d   is n x 1 dense matrix
#             reg is regularization parameter
#             A   is p x n dense or sparse matrix
#             b   is p x 1 dense matrix
#             Aeq is q x n dense or sparse matrix
#             beq is q x 1 dense matrix
#             lb  is n x 1 matrix or scalar
#             ub  is n x 1 matrix or scalar

#         Output arguments:
#             Return dictionary, the output of CVXOPT QP.

#         Dont pass matlab-like empty lists to avoid setting parameters,
#         just use None:
#             lsqlin(C, d, 0.05, None, None, Aeq, beq) #Correct
#             lsqlin(C, d, 0.05, [], [], Aeq, beq) #Wrong!
#     '''
#     sparse_case = False
#     if sparse.issparse(A): #detects both np and cxopt sparse
#         sparse_case = True
#         #We need A to be scipy sparse, as I couldn't find how 
#         #CVXOPT spmatrix can be vstacked
#         if isinstance(A, spmatrix):
#             A = spmatrix_sparse_to_scipy(A)
            
#     C =   numpy_to_cvxopt_matrix(C)
#     d =   numpy_to_cvxopt_matrix(d)
#     Q = C.T * C
#     q = - d.T * C
#     nvars = C.size[1]

#     if reg > 0:
#         if sparse_case:
#             I = scipy_sparse_to_spmatrix(sparse.eye(nvars, nvars,\
#                                           format='coo'))
#         else:
#             I = matrix(np.eye(nvars), (nvars, nvars), 'd')
#         Q = Q + reg * I

#     lb = cvxopt_to_numpy_matrix(lb)
#     ub = cvxopt_to_numpy_matrix(ub)
#     b  = cvxopt_to_numpy_matrix(b)
    
#     if lb is not None:  #Modify 'A' and 'b' to add lb inequalities 
#         if lb.size == 1:
#             lb = np.repeat(lb, nvars)
    
#         if sparse_case:
#             lb_A = -sparse.eye(nvars, nvars, format='coo')
#             A = sparse_None_vstack(A, lb_A)
#         else:
#             lb_A = -np.eye(nvars)
#             A = numpy_None_vstack(A, lb_A)
#         b = numpy_None_concatenate(b, -lb)
#     if ub is not None:  #Modify 'A' and 'b' to add ub inequalities
#         if ub.size == 1:
#             ub = np.repeat(ub, nvars)
#         if sparse_case:
#             ub_A = sparse.eye(nvars, nvars, format='coo')
#             A = sparse_None_vstack(A, ub_A)
#         else:
#             ub_A = np.eye(nvars)
#             A = numpy_None_vstack(A, ub_A)
#         b = numpy_None_concatenate(b, ub)

#     #Convert data to CVXOPT format
#     A =   numpy_to_cvxopt_matrix(A)
#     Aeq = numpy_to_cvxopt_matrix(Aeq)
#     b =   numpy_to_cvxopt_matrix(b)
#     beq = numpy_to_cvxopt_matrix(beq)

#     #Set up options
#     if opts is not None:
#         for k, v in opts.items():
#             solvers.options[k] = v
    
#     #Run CVXOPT.SQP solver
#     sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, x0)
#     return sol

# def lsqnonneg(C, d, opts):
#     '''
#     Solves nonnegative linear least-squares problem:
    
#     min_x ||C*x - d||_2^2,  where x >= 0
#     '''
#     return lsqlin(C, d, reg = 0, A = None, b = None, Aeq = None, \
#                  beq = None, lb = 0, ub = None, x0 = None, opts = opts)

def nonnegtik_hnorm(A, b, lamb_da, nm, nargin,  R=None):
    n = A.shape[1]

    L0 = np.eye(n)
    # print("L0:", L0)

    L1 = -np.diag(np.ones(n)) + np.diag(np.ones(n - 1), 1)
    L1 = L1[:-1, :]
    # print("L1:", L1)

    L2 = (6 * np.diag(np.ones(n)) - 4 * np.diag(np.ones(n - 1), 1) - 4 * np.diag(np.ones(n - 1), -1) + np.diag(np.ones(n - 2), -2) + np.diag(np.ones(n - 2), 2))
    # print("L2:", L2)

    Aug_b = np.concatenate((b, np.zeros(n)))

    if nargin == 4:
        if nm == '0':
            L = L0
        elif nm == '1':
            L = L0 + L1
        elif nm == '2':
            L = L0 + L1 + L2
        elif nm == '11':
            L = L1
        elif nm == '22':
            L = L2
    else:
        L = R

    #original
    Aug_A = np.concatenate((A, lamb_da * L))

    #new version Josh:
    # Aug_A = np.concatenate((A, lamb_da**2 * L))
    #test = np.vstack((A, lamb_da * L))
    #I had to change the tol = 1e-6 instead of none
    #x = lsqnonneg(Aug_A, Aug_b)[0]

    #uncomment below here
    # try:
    #     # Attempt Non-Negative Least Squares (NNLS) first
    #     x = nnls(Aug_A, Aug_b, maxiter=1e4)[0]
    # except Exception as e:
    #     print("NNLS failed:", e)
    #     # If NNLS fails, fallback to CVXPY solver (MOSEK)
    #     x = cp.Variable(Aug_A.shape[1])
    #     cost = cp.norm(Aug_A @ x - Aug_b, 2)**2
    #     constraints = [x >= 0]
    #     problem = cp.Problem(cp.Minimize(cost), constraints)
    #     try:
    #         # Try solving with MOSEK solver first
    #         problem.solve(solver=cp.MOSEK, verbose=False)
    #     except Exception as e:
    #         print("MOSEK solver failed:", e)
    #     #     try:
    #     #         # If MOSEK fails, try with another solver (e.g., ECOS)
    #     #         print("Attempting with ECOS solver...")
    #     #         problem.solve(solver=cp.CLARABEL, verbose=False)
    #     #     except Exception as e2:
    #     #         print("ECOS solver failed:", e2)
    #     #         try:
    #     #             # If ECOS also fails, try using SCS solver
    #     #             print("Attempting with SCS solver...")
    #     #             problem.solve(solver=cp.SCS, verbose=False)
    #     #         except Exception as e3:
    #     #             print("SCS solver failed:", e3)
    #     #             raise ValueError("All solvers failed. Unable to find a solution.")
    #     # # Retrieve the optimized value
    #     x = x.value
    x = nnls(Aug_A, Aug_b, maxiter=1e4)[0]

    # If `x` is all zeros, reattempt the optimization process
    # if np.all(x == 0):
    #     print("Solution is a zero vector, reattempting optimization with CVXPY.")
        
    #     # Reattempt the CVXPY optimization from scratch
    #     x = cp.Variable(Aug_A.shape[1])
    #     cost = cp.norm(Aug_A @ x - Aug_b, 2)**2
    #     constraints = [x >= 0]
    #     problem = cp.Problem(cp.Minimize(cost), constraints)
    #     try:
    #         # Try solving with MOSEK solver first
    #         problem.solve(solver=cp.MOSEK, verbose=False)
    #     except Exception as e:
    #         print("MOSEK solver failed:", e)
    #         try:
    #             # If MOSEK fails, try with ECOS solver
    #             print("Attempting with ECOS solver...")
    #             problem.solve(solver=cp.CLARABEL, verbose=False)
    #         except Exception as e2:
    #             print("ECOS solver failed:", e2)
    #             try:
    #                 # If ECOS also fails, try SCS solver
    #                 print("Attempting with SCS solver...")
    #                 problem.solve(solver=cp.SCS, verbose=False)
    #             except Exception as e3:
    #                 print("SCS solver failed:", e3)
    #                 raise ValueError("All solvers failed. Unable to find a solution.")
        
    #     # Retrieve the optimized value again
    #     x = x.value
    # else:
    #     pass
    x = np.maximum(x,0)
    mdl_err = np.linalg.norm(A @ x - b)** 2
    xnorm = np.linalg.norm(L @ x)**2
    return x, mdl_err, xnorm

def nonnegtik_hnorm(A, b, lamb_da, nm, nargin,  R=None):
    n = A.shape[1]

    L0 = np.eye(n)
    # print("L0:", L0)

    L1 = -np.diag(np.ones(n)) + np.diag(np.ones(n - 1), 1)
    L1 = L1[:-1, :]
    # print("L1:", L1)

    L2 = (6 * np.diag(np.ones(n)) - 4 * np.diag(np.ones(n - 1), 1) - 4 * np.diag(np.ones(n - 1), -1) + np.diag(np.ones(n - 2), -2) + np.diag(np.ones(n - 2), 2))
    # print("L2:", L2)

    Aug_b = np.concatenate((b, np.zeros(n)))

    if nargin == 4:
        if nm == '0':
            L = L0
        elif nm == '1':
            L = L0 + L1
        elif nm == '2':
            L = L0 + L1 + L2
        elif nm == '11':
            L = L1
        elif nm == '22':
            L = L2
    else:
        L = R

    #original
    Aug_A = np.concatenate((A, lamb_da * L))

    #new version Josh:
    # Aug_A = np.concatenate((A, lamb_da**2 * L))
    #test = np.vstack((A, lamb_da * L))
    #I had to change the tol = 1e-6 instead of none
    #x = lsqnonneg(Aug_A, Aug_b)[0]
    try:
        # Attempt Non-Negative Least Squares (NNLS) first
        x = nnls(Aug_A, Aug_b, maxiter=1e4)[0]
    except Exception as e:
        print("NNLS failed:", e)
        # # If NNLS fails, fallback to CVXPY solver (MOSEK)
        # x = cp.Variable(Aug_A.shape[1])
        # cost = cp.norm(Aug_A @ x - Aug_b, 2)**2
        # constraints = [x >= 0]
        # problem = cp.Problem(cp.Minimize(cost), constraints)
        # try:
        #     # Try solving with MOSEK solver first
        #     problem.solve(solver=cp.MOSEK, verbose=False)
        # except Exception as e:
        #     print("MOSEK solver failed:", e)
        #     try:
        #         # If MOSEK fails, try with another solver (e.g., ECOS)
        #         print("Attempting with ECOS solver...")
        #         problem.solve(solver=cp.CLARABEL, verbose=False)
        #     except Exception as e2:
        #         print("ECOS solver failed:", e2)
        #         try:
        #             # If ECOS also fails, try using SCS solver
        #             print("Attempting with SCS solver...")
        #             problem.solve(solver=cp.SCS, verbose=False)
        #         except Exception as e3:
        #             print("SCS solver failed:", e3)
        #             raise ValueError("All solvers failed. Unable to find a solution.")
        # # Retrieve the optimized value
        # x = x.value

    # If `x` is all zeros, reattempt the optimization process
    # if np.all(x == 0):
    #     print("Solution is a zero vector, reattempting optimization with CVXPY.")
        
    #     # Reattempt the CVXPY optimization from scratch
    #     x = cp.Variable(Aug_A.shape[1])
    #     cost = cp.norm(Aug_A @ x - Aug_b, 2)**2
    #     constraints = [x >= 0]
    #     problem = cp.Problem(cp.Minimize(cost), constraints)
    #     try:
    #         # Try solving with MOSEK solver first
    #         problem.solve(solver=cp.MOSEK, verbose=False)
    #     except Exception as e:
    #         print("MOSEK solver failed:", e)
    #         try:
    #             # If MOSEK fails, try with ECOS solver
    #             print("Attempting with ECOS solver...")
    #             problem.solve(solver=cp.CLARABEL, verbose=False)
    #         except Exception as e2:
    #             print("ECOS solver failed:", e2)
    #             try:
    #                 # If ECOS also fails, try SCS solver
    #                 print("Attempting with SCS solver...")
    #                 problem.solve(solver=cp.SCS, verbose=False)
    #             except Exception as e3:
    #                 print("SCS solver failed:", e3)
    #                 raise ValueError("All solvers failed. Unable to find a solution.")
        
    #     # Retrieve the optimized value again
    #     x = x.value
    # else:
    #     pass
    x = np.maximum(x,0)
    mdl_err = np.linalg.norm(A @ x - b)** 2
    xnorm = np.linalg.norm(L @ x)**2
    return x, mdl_err, xnorm


def nonnegtik_hnormLR(A, b, lamb_da, nm, nargin,  R=None):
    n = A.shape[1]

    L0 = np.eye(n)
    # print("L0:", L0)

    L1 = -np.diag(np.ones(n)) + np.diag(np.ones(n - 1), 1)
    L1 = L1[:-1, :]
    # print("L1:", L1)

    L2 = (6 * np.diag(np.ones(n)) - 4 * np.diag(np.ones(n - 1), 1) - 4 * np.diag(np.ones(n - 1), -1) +
          np.diag(np.ones(n - 2), -2) + np.diag(np.ones(n - 2), 2))
    # print("L2:", L2)
    # Aug_b = np.concatenate((b, np.zeros(n)))
    # Aug_b = (A.T @ b)
    eps = 1e-2
    ep4 = np.ones(A.shape[1]) * eps
    Aug_b = (A.T @ b) + ((A.T @ A) @ ep4) + ep4 * (lamb_da)

    if nargin == 4:
        if nm == '0':
            L = L0
        elif nm == '1':
            L = L0 + L1
        elif nm == '2':
            L = L0 + L1 + L2
        elif nm == '11':
            L = L1
        elif nm == '22':
            L = L2
    else:
        L = R

    #original
    # Aug_A = np.concatenate((A, lamb_da * L))
    Aug_A = (A.T @ A + (lamb_da) * L)

    #new version Josh:
    # Aug_A = np.concatenate((A, lamb_da**2 * L))
    #test = np.vstack((A, lamb_da * L))
    #I had to change the tol = 1e-6 instead of none
    #x = lsqnonneg(Aug_A, Aug_b)[0]
    x = nnls(Aug_A, Aug_b)[0]
    if np.all(x == 0):
        x = cp.Variable(A.shape[1])
        cost = cp.norm(Aug_A @ x - Aug_b, 2)**2
        constraints = [x >= 0]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            # Try solving with MOSEK
            problem.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            print("MOSEK solver failed:", e)
            try:
                # If MOSEK fails, try with another solver (e.g., ECOS or SCS)
                print("Attempting with ECOS solver...")
                problem.solve(solver=cp.ECOS, verbose=False)
            except Exception as e2:
                print("ECOS solver failed:", e2)
                try:
                    # If ECOS also fails, try using SCS solver
                    print("Attempting with SCS solver...")
                    problem.solve(solver=cp.SCS, verbose=False)
                except Exception as e3:
                    print("SCS solver failed:", e3)
                    raise ValueError("All solvers failed. Unable to find a solution.")
                # logging.debug("Solution from nonnegtik_hnorm is a zero vector, switching to CVXPY solver.")
                # print(f"Solution from nonnegtik_hnorm is a zero vector, switching to CVXPY solver.")
                # raise ValueError("Zero vector detected, switching to CVXPY.")
    else:
        pass
    # reconst = y.value
    # reconst,_ = nnls(A,b, maxiter = 10000)
    # x = x.value
    # x = x - eps
    x = np.maximum(x,0)
    mdl_err = np.linalg.norm(A @ x - b)** 2
    xnorm = np.linalg.norm(L @ x)**2

    return x, mdl_err, xnorm
