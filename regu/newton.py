# %NEWTON Newton iteration (utility routine for DISCREP).
# %
# % lambda = newton(lambda_0,delta,s,beta,omega,delta_0)
# %
# % Uses Newton iteration to find the solution lambda to the equation
# %    || A x_lambda - b || = delta ,
# % where x_lambda is the solution defined by Tikhonov regularization.
# %
# % The initial guess is lambda_0.
# %
# % The norm || A x_lambda - b || is computed via s, beta, omega and
# % delta_0.  Here, s holds either the singular values of A, if L = I,
# % or the c,s-pairs of the GSVD of (A,L), if L ~= I.  Moreover,
# % beta = U'*b and omega is either V'*x_0 or the first p elements of
# % inv(X)*x_0.  Finally, delta_0 is the incompatibility measure.

# % Reference: V. A. Morozov, "Methods for Solving Incorrectly Posed
# % Problems", Springer, 1984; Chapter 26.

# % Per Christian Hansen, IMM, 12/29/97.
import numpy as np

def newton(lambda_0, delta, s, beta, omega, delta_0):

    thr = np.sqrt(np.finfo(float).eps)
    it_max = 50

    if (lambda_0 < 0):
        raise ValueError("Initial guess lambda_0 must be nonnegative")
    
    p,ps = s.reshape(-1,1).shape

    if (ps == 2):
        sigma = s[:,0]
        s = s[:,0]/s[:,1]
    s2 = s ** 2

    lamb_da = lambda_0
    step = 1
    it = 0
    while (np.abs(step) > thr * lamb_da and np.abs(step) > thr and it < it_max):
        it = it + 1
        f = s2/(s2 + lamb_da**2)
        if (ps == 1):
            r = (1-f) * (beta - s * omega)
            z = f * r
        else:
            r = (1-f) * (beta - sigma * omega)
            z = f * r
        step = (lamb_da/4) * (r.T @ r + (delta_0 + delta) * (delta_0 - delta))/(z.T @ r)
        lamb_da = lamb_da - step
        if (lamb_da <0):
            lamb_da = 0.5 * lambda_0
            lambda_0 = 0.5  * lambda_0
    
    if (np.abs(step) > thr*lamb_da and np.abs(step) > thr):
        raise ValueError(f"Max. number of iterations ({it_max}) reached")

    return lamb_da
