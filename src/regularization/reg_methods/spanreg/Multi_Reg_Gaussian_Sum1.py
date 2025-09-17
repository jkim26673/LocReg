import numpy as np
from scipy.sparse import kron
from scipy.optimize import nnls, linprog, minimize
from src.regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm
import matplotlib.pyplot as plt
import piqp
import os
import mosek
import nlopt
from src.utils.load_imports.loading import *

# import cvxpy as cp
# mosek_lic_path = "/Users/steveh/Downloads/mosek/mosek.lic"
# os.environ["MOSEKLM_LICENSE_FILE"] = mosek_lic_path
import os
os.environ['ARPACK_OPTIONS'] = 'tol=1e-6'

import numpy as np
import cvxopt


def quadprog(H, f, initial, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq.reshape(1,-1), tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq, initvals = initial, kktsolver='ldl', options={'kktreg':1e-1})

    return np.array(sol['x']).flatten()


def Multi_Reg_Gaussian_Sum1(dat_noisy, Gaus_info):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    # Load information
    Lambda = Gaus_info['Lambda'].reshape(-1,1)
    nLambda = len(Lambda)
    nGaus = Gaus_info['LGBs'].shape[1]
    LGBs = Gaus_info['LGBs']
    A = Gaus_info['A']
    m = A.shape[1]
    n = A.shape[0]
    T2 = np.linspace(1, 200, m)
    f_unknown_L2 = np.zeros((m, nLambda))
    rho_L2 = np.zeros(nLambda)
    eta_L2 = np.zeros(nLambda)

    # Online computation
    for kk in range(nLambda):
        f_unknown_L2[:, kk], rho_L2[kk], eta_L2[kk] = nonnegtik_hnorm(A, dat_noisy, Lambda[kk], '0', nargin = 4)

    #print("f_unknown_L2[:,6]:", f_unknown_L2[:,6])

    x_L2_weight = np.zeros((nGaus, nLambda))
    Err_f_unknown_L2 = np.zeros((m, nLambda))

    for j in range(nLambda):
        F_L2 = np.reshape(Gaus_info['L2_store'][:, :, j], (-1,m)).T
        Y_L2 = f_unknown_L2[:, j]
        xx = nnls(F_L2, Y_L2)[0]
        Err_f_unknown_L2[:, j] = F_L2 @ xx - Y_L2
        x_L2_weight[:, j] = xx
    # print("F_L2 shape: ", F_L2.shape)

    L2_store = Gaus_info['L2_store']
    #L2_store = L2_store.transpose(2,0,1)
    beta_L2 = Gaus_info['beta_L2']


    L2store_reshaped_alpha = np.empty((m,0))
    L2store_reshaped_c = np.empty((m,0))


    for i in range(nLambda):
        #L2store_reshaped_alpha = np.hstack((L2store_reshaped_alpha, L2_store[:, :, i]))
        L2store_reshaped_alpha = np.concatenate((L2store_reshaped_alpha, L2_store[:, :, i].T), axis = 1)
    
    # print("L2store_reshaped_alpha", L2store_reshaped_alpha.shape)
    # print("L2_store[:, :, 0].T", (L2_store[:, :, 0].T).shape)
    # print("L2store_reshaped_alpha shape:",L2store_reshaped_alpha.shape)
    # print("L2store_reshaped_alpha:",L2store_reshaped_alpha)

    # plt.plot(T2, L2store_reshaped_alpha)
    # plt.show()

    for i in range(nGaus):
        a = L2_store[i, :, :]
        L2store_reshaped_c = np.concatenate((L2store_reshaped_c, np.reshape(a, (m, nLambda))), axis = 1)
    
    # print("L2store_reshaped_c shape:",L2store_reshaped_c.shape)


    new_xweight_L2 = x_L2_weight.ravel()
    # print("new_x_weight_L2 shape:",new_xweight_L2.shape)
    # print("new_x_weight_L2:",new_xweight_L2)
    # has_nonzero = np.any(new_xweight_L2 != 0)

    # if has_nonzero:
    #     print("Array contains nonzero values.")
    # else:
    #     print("Array does not contain nonzero values.")

    beta_L2 = beta_L2.T.reshape(-1,1)
    new_beta_L2 = beta_L2.ravel()

    # print("new_beta_L2 shape:",new_beta_L2.shape)
    # print("new_beta_L2:",new_beta_L2)

    L2_mat_alpha = L2store_reshaped_alpha @ np.diag(new_xweight_L2)
    
    # print("L2_mat_alpha shape:",L2_mat_alpha.shape)
    # print("L2_mat_alpha:",L2_mat_alpha)

    L2_mat_c = L2store_reshaped_c @ np.diag(new_beta_L2)

    # print("L2_mat_c shape:",L2_mat_c.shape)
    # print("L2_mat_c:",L2_mat_c)

    kron_alpha = kron(np.eye(nLambda), np.ones((nGaus,1)))
    
    # print("kron_alpha shape:",kron_alpha.shape)
    # print("kron_alpha:",kron_alpha)

    Ind_alpha = np.column_stack((np.eye(nLambda), np.zeros((nLambda, nGaus))))
    
    # print("Ind_alpha shape:",Ind_alpha.shape)
    # print("Ind_alpha:",Ind_alpha)

    kron_c = kron(np.eye(nGaus), np.ones((nLambda,1)))
    
    # print("kron_c shape:",kron_c.shape)
    # print("kron_c:",kron_c)

    Ind_c = np.column_stack((np.zeros((nGaus, nLambda)), np.eye(nGaus)))

    # print("Ind_c shape:",Ind_c.shape)
    # print("Ind_c:",Ind_c)

    Aleft = L2_mat_alpha @ kron_alpha @ Ind_alpha - L2_mat_c @ kron_c @ Ind_c
    
    # print("Aleft condition number:", np.linalg.cond(Aleft))
    # print("Aleft shape:", Aleft.shape)


    bright = np.zeros(m)
    # Aeq = np.concatenate((np.zeros(nLambda), np.ones(nGaus))).reshape(1, -1)
    
    import scipy.sparse as sp

    #best2
    H = (Aleft.T @ Aleft)
    Aeq = np.concatenate((np.zeros(nLambda), np.ones(nGaus)))
    # #beq = 1
    #beq = np.ones(nLambda + nGaus)
    beq = 1
    # #beq = 3
    # # beq = np.array([1.0])  # A single equality constraint


    lb = np.zeros(nLambda + nGaus)
    ub = (nLambda + nGaus) * [None]
    # print("H shape:", H.shape)
    c = (np.zeros(nLambda + nGaus))
    def quadprog_constraint(x):
        return Aeq @ x - beq
    


    # P = sp.csc_matrix(H, dtype=np.float64)
    # c = np.zeros(nLambda + nGaus, dtype=np.float64)
    # A = sp.csc_matrix(Aeq, dtype=np.float64)  # Assuming Aeq_matrix is a sparse matrix
    # b = beq.astype(np.float64)
    # G = None  # If G is not needed, set it to None
    # h = None  # If h is not needed, set it to None

    # solver = piqp.SparseSolver()
    # solver.settings.verbose = True
    # solver.settings.compute_timings = True

    # # Ensure that the arguments are passed correctly to solver.setup
    # solver.setup(P, c, A, b, G, h, x_lb=lb, x_ub=None)

    # # Use solver.solve to find the solution
    # status = solver.solve()

    
    def quadprog_obj(x):
        return 0.5 * x.T @ H @ x + np.zeros((nLambda + nGaus,1)).T @ x

    # def obj(x):
    #     return H @ x


    
    #constraints=cons
    
    # from cvxopt import matrix, solvers
    # Print the results
    # if status == 'optimal':
    #     print('Optimal solution found:')
    #     print(f'x = {solver.result.x}')
    # else:
    #     print('No optimal solution found.')
    
    # result = solver.result.x

    
    # # Define your problem data
    # P = H
    # c = c

    # # Solve the QP problem
    # sol = solvers.qp(P, q, G, h, A, b)
    # print("H shape", H.shape)
    # print ("c shape", c.shape)
    # print("Aeq shape:", Aeq.shape)
    # print("Beq shape", beq.shape)

    # import qpsolvers
    # from qpsolvers import solve_qp
    # num_variables = H.shape[0]  # Assuming H is a square matrix
    # lb_value = np.array([0.0] * num_variables)  # Creating a vector of zeros as lower bounds
    # ub_value = np.array([np.inf] * num_variables)
    # num_columns_Aeq = Aeq.reshape(1,-1).shape[1]  # Number of columns in Aeq
    # beq = np.array([1.0] * num_columns_Aeq)  # Creating a vector of ones

    # result = solve_qp(H, c, G = None, h = None, A = Aeq, b = beq, lb = lb_value, ub = ub_value, solver = 'quadprog')
    # print("Solution found:", result)
    # try:
    #     sol_qp = solve_qp(H, c, G=None, h=None, A=  Aeq, b= np.array([1]).reshape(-1,1), lb=  lb_value, ub=None, solver='mosek', initvals = x0_guess)
    #     print("Solution found:", sol_qp)
    # except Exception as e:
    #     print("An error occurred:", e)

    # try:
    #     L = np.linalg.cholesky(H)
    #     print("Matrix is positive semidefinite")
    # except np.linalg.LinAlgError:
    #     print("Matrix is not positive semidefinite")
    def equality_constraint(x):
        return Aeq @ x - beq
    lb_value = 0.0
    cons = {'type': 'eq', 'fun': lambda x: Aeq @ x - beq}
    options = {'xtol': 1e-7, 'gtol': 1e-7, 'maxiter':400, 'disp': False}
    
    cons = [{'type': 'eq', 'fun': equality_constraint}]

    # def jacobian(beta):
    #     return H @ beta + c
            
    # def hessian(beta):
    #     return np.zeros_like(beta)
    

    x0_guess = np.ones(nLambda + nGaus)/(nLambda+nGaus)

    # result = minimize(fun = lambda beta: 0.5 * beta.T @ H @ beta + c.T @ beta, constraints= cons, method='trust-constr',
    #                   x0 = x0_guess, bounds= [(0, np.inf) for x in x0_guess],  options=options).x
    
    
            
    #result = minimize(fun = lambda beta: 0.5 * beta.T @ H @ beta + c.T @ beta,constraints= cons, x0 = x0_guess, bounds=[(0, None) for x in x0_guess], options=options).x

    #For testing cvxopt:
    Aeq = np.concatenate((np.zeros((nLambda,1)), np.ones((nGaus,1))))
    beq = 1
    c = (np.zeros((nLambda + nGaus, 1)))

    result = quadprog(H, c, L=None, k=None, Aeq=Aeq, beq=beq, lb=0, ub=None, initial = x0_guess)


    # from cvxopt import matrix, solvers

    # H = matrix(H)
    # #c = matrix(c)
    # Aeq = matrix(Aeq)
    # beq = matrix(beq)
    # lb = matrix(lb)
    # ub = matrix([float("inf")] * (nLambda + nGaus))
    # initvals = matrix(x0_guess)

    # # Define the decision variables
    # x = cp.Variable(nLambda + nGaus)

    # H = cp.psd_wrap(H)

    # # Define the objective function
    # objective = cp.Minimize(0.5 * cp.quad_form(x, H) + c @ x)

    # # Define the constraints
    # constraints = [Aeq @ x == beq]

    # # Apply variable bounds
    # for i in range(len(lb)):
    #     if lb[i] is not None:
    #         constraints.append(x[i] >= lb[i])
    #     if ub[i] is not None:
    #         constraints.append(x[i] <= ub[i])

    # # Create and solve the convex QP problem
    # problem = cp.Problem(objective, constraints)
    # result = problem.solve(solver=cp.MOSEK, verbose=True)

    # # Get the optimal solution
    # result = result.value
    # print("Optimal Solution:")
    # print(result)

    # method='interior-point'
    # result = quadprog(H, c, L=None, k=None, Aeq= Aeq, beq = beq, lb= lb, ub=None).flatten()
    # result = minimize(fun = lambda beta: 0.5 * (beta.T @ H @ beta) + (c.T @ beta), 
    #                   x0 = np.zeros(nLambda + nGaus), method = 'trust-constr', constraints = cons,
    #                   bounds=[(0, np.inf)] * (nLambda + nGaus), options = options).x
    # Solve the quadratic programming problem using the interior-point method
    # solver_options = {'initvals': initvals}

    # result = solvers.qp(H, c, Aeq=Aeq, beq=beq, lb=lb, ub=ub, options=solver_options)
    # # Extract the optimal solution
    # result = result['x']
    # result = np.array(result).flatten()
    # print("result:",result)


                                #bounds= [(0, np.inf) for x in x0_guess]
    #np.zeros((nLambda + nGaus,1), dtype=np.float64)
    # print("result", result)
    #np.ones(nLambda + nGaus)

    # test = Aeq @ result
    # print("This is the test:", test)

    C_L2 = result[nLambda:]
    alpha_L2 = result[:nLambda]

    # print("alpha_L2 shape:", alpha_L2.shape)
    # print("alpha_L2:", alpha_L2)
    # print("f_unknown_L2 shape:", f_unknown_L2.shape)
    # print("f_unknown_L2:", f_unknown_L2)

    # find the best representation
    f_rec_final = f_unknown_L2 @ alpha_L2
    y_rec_temp = A @ f_rec_final
    y_ratio = np.max(dat_noisy)/np.max(y_rec_temp)
    f_rec_final = y_ratio * f_rec_final

    f_rece_Final1 = LGBs @ C_L2

    # plt.plot(T2, f_rec_final)
    
    # plt.show()
    # plt.plot(T2, f_rece_Final1)
    # plt.show()

    # print("f_rec_final shape:", f_rec_final.shape)
    # print("f_rec_final:", f_rec_final)

    # Getting results and save data
    F_info = {
        'alpha_L2': alpha_L2,
        'f_unknown_L2': f_unknown_L2,
        'rho_L2': rho_L2,
        'eta_L2': eta_L2,
        'f_rec': f_rec_final,
        'C_L2': C_L2
    }

    return f_rec_final, alpha_L2, F_info, C_L2
