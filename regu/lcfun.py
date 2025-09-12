import numpy as np

def lcfun(lambda_, s, beta, xi, nargin, fifth=False):
    """
    Auxiliary routine for l_corner; computes the NEGATIVE of the curvature.
    
    Parameters:
        lambda_ (ndarray): Regularization parameter values.
        s (ndarray): Singular values or norms of the solution.
        beta (ndarray): Solution coefficients.
        xi (ndarray): Vector related to the solution.
        fifth (bool): Flag for fifth-degree regularization.
    
    Returns:
        ndarray: Array of curvature values.
    """
    phi = np.zeros_like(lambda_)
    dphi = np.zeros_like(lambda_)
    psi = np.zeros_like(lambda_)
    dpsi = np.zeros_like(lambda_)
    eta = np.zeros_like(lambda_)
    rho = np.zeros_like(lambda_)
    
    if beta.size > s.size:
        LS = True
        rhoLS2 = beta[-1] ** 2
        beta = beta[:-1]
    else:
        LS = False
    
    for i in range(len(lambda_)):
        if nargin == 4:
            f = (s ** 2) / (s ** 2 + lambda_[i] ** 2)
        else:
             f = s / (s + lambda_[i])
        cf = 1 - f
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm(cf * beta)
        f1 = -2 * f * cf / lambda_[i]
        f2 = -f1 * (3 - 4 * f) / lambda_[i]
        phi[i] = np.sum(f * f1 * np.abs(xi) ** 2)
        psi[i] = np.sum(cf * f1 * np.abs(beta) ** 2)
        dphi[i] = np.sum((f1 ** 2 + f * f2) * np.abs(xi) ** 2)
        dpsi[i] = np.sum((-f1 ** 2 + cf * f2) * np.abs(beta) ** 2)
    
    if LS:
        rho = np.sqrt(rho ** 2 + rhoLS2)
    
    deta = phi / eta
    drho = -psi / rho
    ddeta = dphi / eta - deta * (deta / eta)
    ddrho = -dpsi / rho - drho * (drho / rho)
    
    dlogeta = deta / eta
    dlogrho = drho / rho
    ddlogeta = ddeta / eta - (dlogeta) ** 2
    ddlogrho = ddrho / rho - (dlogrho) ** 2
    
    g = - (dlogrho * ddlogeta - ddlogrho * dlogeta) / (dlogrho ** 2 + dlogeta ** 2) ** (3 / 2)
    
    return g
