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
import scipy.linalg as la


def i_laplace(n, example, nargin):
    if (n <= 0):
        raise ValueError('The order n must be postiive')
    if nargin == 1:
        example = 1
    s = (10/n) * np.arange(1,n+1)
    s = s.reshape(-1,1)

    t = np.diag(2*np.arange(1,n+1) - 1) - np.diag((np.arange(1,n)), 1) - np.diag((np.arange(1,n)), -1)
    #check if eig are real and just in complex form...run in separate script...
    eigenvalues, Q = la.eig(t)
    # print("eigenvalues", eigenvalues.shape)
    # print("Q",Q.shape)
    t =np.real(eigenvalues)
    # t = np.diag(t)
    # print("t.shape",t.shape)
    #axis = 1 if t is row vector, else axis = 0
    t = np.sort(t)
    indx = np.argsort(np.real(eigenvalues))

    v = np.abs(Q[0, indx])
    nz = np.where(v != 0)[0]

    A = np.zeros((n, n))
    for i in range(n):
        for j in nz:
            A[i, j] = (1 - s[i]) * t[j] + 2 * np.log(v[j])
    A[:, nz] = np.exp(A[:, nz])

    if example == 1:
        b = 1 / (s + 0.5)
        x = np.exp(-t / 2)
    elif example == 2:
        b = 1 / s - 1 / (s + 0.5)
        x = 1 - np.exp(-t / 2)
    elif example == 3:
        b = 2 * 1 / ((s + 0.5) ** 3)
        x = (t ** 2) * np.exp(-t / 2)
    elif example == 4:
        b = np.exp(-2 * s) / s
        x = np.ones(n)
        f = np.where(t <= 2)[0]
        f_len = len(f)
        x[f] = np.zeros(f_len)
    else:
        raise ValueError('Illegal example')
    
    return A,b.flatten(),x,t
