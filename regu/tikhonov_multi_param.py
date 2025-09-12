import numpy as np
from scipy.optimize import minimize

def tikhonov_multi_param(x,A,b,L1,L2,lambda1,lambda2):
# Define the objective function
    def objective_function(x, A, b, L1, L2, x_0, lambda1,lambda2):
        term1 = np.linalg.norm(np.dot(A, x) - b)**2
        term2 = lambda1**2 * np.linalg.norm(np.dot(L1, x - x_0))**2
        term3 = lambda2**2 * np.linalg.norm(np.dot(L2, x - x_0))**2
        return term1 + term2
    # Define the optimization problem
    x_0 = 0
    x_initial = np.zeros_like(b)
    result = minimize(objective_function, x_initial, args=(A, b, L1,L2, x_0, lambda1, lambda2))
    x_optimized = result.x
    return x_optimized


