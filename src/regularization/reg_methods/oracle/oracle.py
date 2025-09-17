from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

def oracle(Alpha_vec, L, data_noisy, G, nT2, g):
    OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
    OP_rhos = np.zeros((len(Alpha_vec)))
    for j in (range(len(Alpha_vec))):
        try:
            # Fallback to nonnegtik_hnorm
            sol, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j], '0', nargin=4)
        except Exception as e:
            print(f"Error in nonnegtik_hnorm: {e}")
            # If both methods fail, solve using cvxpy
            lam_vec = Alpha_vec[j] * np.ones(G.shape[1])
            A = (G.T @ G + np.diag(lam_vec))
            eps = 1e-2
            ep4 = np.ones(A.shape[1]) * eps
            b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vec
            
            y = cp.Variable(G.shape[1])
            cost = cp.norm(A @ y - b, 2)**2
            constraints = [y >= 0]
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.MOSEK, verbose=False)
            
            sol = y.value
            sol = sol - eps
            sol = np.maximum(sol, 0)

        OP_x_lc_vec[:, j] = sol
        OP_rhos[j] = np.linalg.norm(OP_x_lc_vec[:,j] - g, 2)**2

    OP_log_err_norm = np.log10(OP_rhos)
    min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)
    min_x = Alpha_vec[min_index[0]]
    min_z = np.min(OP_log_err_norm)
    OP_min_alpha1 = min_x
    OP_min_alpha1_ind = min_index[0]
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    return f_rec_OP_grid, OP_min_alpha1
