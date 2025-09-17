import numpy as np
from scipy.optimize import nnls

def tikh_norm(A, b, lamb_da, nm, nargin,  R=None):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """  
    n = A.shape[1]

    L0 = np.eye(n)
    # print("L0:", L0)

    L1 = -np.diag(np.ones(n)) + np.diag(np.ones(n - 1), 1)
    L1 = L1[:-1, :]
    # print("L1:", L1)

    L2 = (6 * np.diag(np.ones(n)) - 4 * np.diag(np.ones(n - 1), 1) - 4 * np.diag(np.ones(n - 1), -1) +
          np.diag(np.ones(n - 2), -2) + np.diag(np.ones(n - 2), 2))
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
    # x = nnls(Aug_A, Aug_b)[0]
    # x = nnls(Aug_A, Aug_b)[0]
    x = np.linalg.solve(Aug_A,Aug_b)
    mdl_err = np.linalg.norm(A @ x - b)** 2
    xnorm = np.linalg.norm(L @ x)**2

    return x, mdl_err, xnorm
