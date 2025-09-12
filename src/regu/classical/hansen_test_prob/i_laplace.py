# %I_LAPLACE Test problem: inverse Laplace transformation.
# %
# % [A,b,x,t] = i_laplace(n,example)
# %
# % Discretization of the inverse Laplace transformation by means of
# % Gauss-Laguerre quadrature.  The kernel K is given by
# %    K(s,t) = exp(-s*t) ,
# % and both integration intervals are [0,inf).
# %
# % The following examples are implemented, where f denotes
# % the solution, and g denotes the right-hand side:
# %    1: f(t) = exp(-t/2),        g(s) = 1/(s + 0.5)
# %    2: f(t) = 1 - exp(-t/2),    g(s) = 1/s - 1/(s + 0.5)
# %    3: f(t) = t^2*exp(-t/2),    g(s) = 2/(s + 0.5)^3
# %    4: f(t) = | 0 , t <= 2,     g(s) = exp(-2*s)/s.
# %              | 1 , t >  2
# %
# % The quadrature points are returned in the vector t.

# % Reference: J. M. Varah, "Pitfalls in the numerical solution of linear
# % ill-posed problems", SIAM J. Sci. Stat. Comput. 4 (1983), 164-176.

# % Per Christian Hansen, IMM, Oct. 21, 2006.
import numpy as np
import scipy

def i_laplace(n, example):
    # Initialization.
    if n <= 0:
        raise ValueError('The order n must be positive')
    if example == 1:
        example = 1

    # Compute equidistant collocation points s.
    s = (10 / n) * np.arange(1, n+1)

    # Compute abscissas t and weights v from the eigensystem of the
    # symmetric tridiagonal system derived from the recurrence
    # relation for the Laguerre polynomials. Sorting of the
    # eigenvalues and -vectors is necessary.
    t = np.diag(2 * np.arange(1, n+1) - 1) - np.diag(np.arange(1, n), 1) - np.diag(np.arange(1, n), -1)
    t,Q = scipy.linalg.eig(t)
    t = t.real
    t = np.sort(t)
    indx = [i[0] for i in sorted(enumerate(t), key=lambda x:x[1])]
    v = np.abs(Q[0,indx])

    # Find non-zero elements in v
    nz = np.where(v != 0)[0]

    # Set up the coefficient matrix A. Due to limitations caused
    # by finite-precision arithmetic, A has zero columns if n > 195.
    A = np.zeros((n, n))
    for i in range(n):
        for j in nz:
            A[i, j] = (1 - s[i]) * t[j] + 2 * np.log(v[j])
    A[:, nz] = np.exp(A[:, nz])

    # Compute the right-hand side b and the solution x by means of
    # simple collocation.
    if example == 1:
        b = np.ones(n) / (s + 0.5)
        x = np.exp(-t / 2)
    elif example == 2:
        b = 1 / s - 1 / (s + 0.5)
        x = 1 - np.exp(-t / 2)
    elif example == 3:
        b = 2 / ((s + 0.5) ** 3)
        x = (t ** 2) * np.exp(-t / 2)
    elif example == 4:
        b = np.exp(-2 * s) / s
        x = np.ones(n)
        f = np.where(t <= 2)[0]
        x[f] = np.zeros(len(f))
    else:
        raise ValueError('Illegal example')

    return A, b, x, t

# import numpy as np
# from scipy.linalg import eig

# def i_laplace(n, example=1):
#     """
#     I_LAPLACE Test problem: inverse Laplace transformation.

#     A, b, x, t = i_laplace(n, example)

#     Discretization of the inverse Laplace transformation by means of
#     Gauss-Laguerre quadrature. The kernel K is given by
#        K(s,t) = exp(-s*t),
#     and both integration intervals are [0,inf).

#     The function returns the coefficient matrix A, the right-hand side vector b,
#     the solution vector x, and the quadrature points t.

#     Parameters:
#     n : int
#         Order of the quadrature and size of the matrices.
#     example : int, optional
#         Example number (1, 2, 3, or 4) specifying the problem setup. Default is 1.

#     Returns:
#     A : numpy.ndarray
#         Coefficient matrix of size (n, n).
#     b : numpy.ndarray
#         Right-hand side vector of size (n,).
#     x : numpy.ndarray
#         Solution vector of size (n,).
#     t : numpy.ndarray
#         Quadrature points vector of size (n,).

#     Reference:
#     J. M. Varah, "Pitfalls in the numerical solution of linear
#     ill-posed problems", SIAM J. Sci. Stat. Comput. 4 (1983), 164-176.
#     """
    
#     # Initialization
#     if n <= 0:
#         raise ValueError('The order n must be positive')

#     # Compute equidistant collocation points s
#     s = (10/n) * np.arange(1, n+1)

#     # Compute abscissas t and weights v from the eigensystem
#     t_matrix = np.diag(2*np.arange(1, n+1)-1) - np.diag(np.arange(1, n), 1) - np.diag(np.arange(1, n), -1)
#     eigvals, Q = eig(t_matrix)
#     t = np.real(np.diag(eigvals))
#     Q = np.real(Q)
#     indx = np.argsort(t)
#     t = t[indx]
#     v = np.abs(Q[0, indx])
#     nz = np.nonzero(v)[0]

#     # print("s.shape",s.shape)
#     # print("t.shape",t.shape)
#     # print("v.shape",v.shape)
#     # print("nz.shape",nz.shape)

#     # Set up the coefficient matrix A
#     A = np.zeros((n, n))
#     for i in range(n):
#         for j in nz:
#             A[i, j] = (1 - s[i]) * t[j] + 2 * np.log(v[j])
#     A[:, nz] = np.exp(A[:, nz])

#     # Compute the right-hand side b and the solution x
#     if example == 1:
#         b = 1 / (s + 0.5)
#         x = np.exp(-t/2)
#     elif example == 2:
#         b = 1 / s - 1 / (s + 0.5)
#         x = 1 - np.exp(-t/2)
#     elif example == 3:
#         b = 2 / ((s + 0.5)**3)
#         x = t**2 * np.exp(-t/2)
#     elif example == 4:
#         b = np.exp(-2 * s) / s
#         x = np.ones(n)
#         f = np.where(t <= 2)[0]
#         x[f] = np.zeros(len(f))
#     else:
#         raise ValueError('Illegal example')

#     return A, b, x, t
