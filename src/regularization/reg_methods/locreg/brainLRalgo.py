from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

def LocReg_Ito_mod(data_noisy, G, lam_ini, gamma_init, maxiter):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    def minimize(lam_vec):
        """
        Description: Oracle selection method for regularization:

        :param power_pellet_active: bool - does the player have an active power pellet?
        :param touching_ghost: bool - is the player touching a ghost?
        :return: bool - can a ghost be eaten?

        Test Example:
        """
        try:
            eps = 1e-2
            A = G.T @ G + np.diag((lam_vec))        
            ep4 = np.ones(G.shape[1]) * eps
            b = G.T @ data_noisy + (G.T @ G @ ep4) + (ep4 * lam_vec)
            y = cp.Variable(G.shape[1])
            cost = cp.norm(A @ y - b, 'fro')**2
            constraints = [y >= 0]
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver = cp.MOSEK)
            sol = y.value
            sol = np.maximum(sol - eps, 0)
        except Exception as e:
            print(f"MOSEK failed {e}")
            sol, rho, trash = nonnegtik_hnorm(G, data_noisy, lam_vec, '0', nargin=4)
            sol = np.maximum(sol,0)
        return sol, A, b

    def phi_resid(G, param_vec, data_noisy):
        """
        Description: Oracle selection method for regularization:

        :param power_pellet_active: bool - does the player have an active power pellet?
        :param touching_ghost: bool - is the player touching a ghost?
        :return: bool - can a ghost be eaten?

        Test Example:
        """    
        return np.linalg.norm(G @ param_vec - data_noisy, 2) ** 2

    def fixed_point_algo(gamma, lam_vec,tol):
        """
        Description: Oracle selection method for regularization:

        :param power_pellet_active: bool - does the player have an active power pellet?
        :param touching_ghost: bool - is the player touching a ghost?
        :return: bool - can a ghost be eaten?

        Test Example:
        """    
        lam_curr = lam_vec
        ep = 1e-2
        ep_min = 1e-2
        f_old = np.ones(G.shape[1])
        k = 1
        while True and k <= maxiter:
            try:
                curr_f_rec, LHS, _ = minimize(lam_curr)
                if curr_f_rec is None or np.any([x is None for x in curr_f_rec]):
                    print(f"curr_f_rec returns a None after minimization for iteration {k}")
                    continue
            except Exception as e:
                print("An error occurred during minimization:", e)
                continue
            curr_noise = G @ curr_f_rec - data_noisy
            L = np.linalg.cholesky(LHS)
            delta_p = scipy.linalg.cho_solve((L,True), G.T @ curr_noise)
            prev = np.linalg.norm(delta_p)
            
            iterationval = 1
            # while iterationval < 300:
            # while iterationval < 550: testing and 1e-5; sufficient
            # while iterationval < 600: testing and 1e-4 2nd one; no good peak resolution
            # iterations < 700, 1e-5, similar to 550, worse overteim
            #best iterations 200, 1e-5
            #10-14-24 runs iteration val < 200, 1e-2
            #10-15-24 runs iteration val < 180, 1e-5
            #10-17-24 : iteration 180, 1e-3
            while iterationval < 200: 
                curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
                curr_noise = G @ curr_f_rec - data_noisy
                try:
                    delta_p = scipy.linalg.cho_solve((L,True), G.T @ curr_noise)
                except RuntimeWarning:
                    print("Error with delta_p calculation")
                if np.abs((np.linalg.norm(delta_p) / prev) - 1) < 1e-3:
                    break
                else:
                    pass
                prev = np.linalg.norm(delta_p)
                iterationval+=1
            curr_f_rec = np.maximum(curr_f_rec - delta_p, 0)
            phi_new = phi_resid(G, curr_f_rec, data_noisy)
            psi_lam = np.array(curr_f_rec)
            c = 1/gamma
            lam_new = c * (phi_new / (np.abs(psi_lam) + ep_min))
            if (np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) < tol or k == maxiter:
                if k == maxiter:
                    print("Maximum Iteration Reached")
                return curr_f_rec, lam_curr, k
            else:
                #10-17-24 : iteration 180, 1e-3
                # ep_min = ep_min / 1.7
                ep_min = ep_min / 2
                # if ep_min <= 8e-5:
                #     ep_min = 8e-5
                if ep_min <= 1e-5:
                    ep_min = 1e-5
                # ep_min = ep_min / 1.2
                # if ep_min <= 1e-4:
                #     ep_min = 1e-4
                lam_curr = lam_new
                f_old = curr_f_rec
                k += 1
                # return curr_f_rec, lam_curr, val

    #Main Code
    lam_vec = lam_ini * np.ones(G.shape[1])
    # choice_val = 9e-3
    try:
        best_f_rec1, fin_lam1, iternum = fixed_point_algo(gamma_init, lam_vec, tol=1e-2)
        # fin_lam1 = np.sqrt(fin_lam1)
    except Exception as e:
        print("Error in locreg")
        print("lam_vec", lam_vec)
    # fin_lam1 = np.sqrt(fin_lam1)
    new_resid = phi_resid(G, best_f_rec1, data_noisy)
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(G, zero_vec, data_noisy)
    gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25

    # new_choice2 = 5e-3
    best_f_rec2, fin_lam2, _ = fixed_point_algo(gamma_new, fin_lam1, tol=1e-3)
    x_normalized = best_f_rec2
    return x_normalized, fin_lam2, best_f_rec1, fin_lam1, iternum

