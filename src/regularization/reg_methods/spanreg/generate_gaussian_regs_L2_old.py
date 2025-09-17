import numpy as np
from src.regularization.reg_methods.spanreg.Gaussian_basis import Gaussian_basis
from src.regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm
import os
from scipy.optimize import minimize, LinearConstraint
import scipy.io

def generate_gaussian_regs_L2_old(A, T2, TE, SNR, n_run, reg_param_lb, reg_param_ub, N_reg, Nc, cmin, cmax, sigma_min, sigma_max):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    # Gaussian dictionary set up
    LGBs = Gaussian_basis(T2, cmin, cmax, Nc, sigma_min, sigma_max)
    options = {'disp': False}
    m = len(T2)
    n = A.shape[0]
    # setting up regularization parameters
    Lambda = np.logspace(reg_param_lb, reg_param_ub, N_reg)
    
    nLambda = len(Lambda)
    nc = sum(Nc)
    # initializations
    beta_L2 = np.zeros((nc, nLambda))
    L2_store = np.zeros((nc, m, nLambda))
    err_gaus_l2 = np.zeros((n_run, nc))

    for k in range(n_run):
        new_L2_store = np.zeros((nc, m, nLambda))
        new_beta_store = np.zeros((nc, nLambda))
        
        for i in range(nc):
            dat_noiseless = A @ LGBs[:, i]
            noise = np.max(dat_noiseless)/(SNR * np.random.randn(n,1))
            # add noise
            dat_noisy_ = dat_noiseless + noise.flatten()

            X_L2 = np.zeros((m, nLambda))
            rho_temp = np.zeros(nLambda)
            eta_temp = np.zeros(nLambda)
            
            #variability here at nnls,lsqnonnneg vs lsqnnoneg from matlab
            for kk in range(len(Lambda)):
                X_L2[:, kk], rho_temp[kk], eta_temp[kk] = nonnegtik_hnorm(A, dat_noisy_, Lambda[kk], nm = '0', nargin = 4)
            
            #check condition number X_L2 (link); check over many

            # def objective_function(x, H, f):
            #     return 0.5 * x.T @ H @ x + f.T @ x

            new_L2_store[i, :, :] = X_L2
            # print("X_L2:", X_L2.shape)
            # print(X_L2)
            # print("X_L2.T:", (X_L2.T).shape)
            # print(X_L2.T)
            H = X_L2.T @ X_L2
            # print("H:", H.shape)
            # print(H)
            f = -X_L2.T @ LGBs[:, i]

            # print("c:", c)
            # print("c shape:", c.shape)

            # nonneg_constraint = LinearConstraint(np.eye(nLambda), 0, np.inf)
            # constraints = [nonneg_constraint]

            # Define constraints for beta_temp
            # cons = ({'type': 'ineq', 'fun': lambda x: b - A @ x})

            def jacobian(beta):
                return H @ beta + f
            
            def hessian(beta):
                return H
    
            # A = None
            # b = None
            # x0 = np.zeros(nLambda)
            # lb = 0
            # ub = None
            # import cvxpy as cp
            # from cvxopt import matrix,solvers
            # n,m = H.shape
            # beta_temp = cp.Variable(n)
            # err_gaus_l2 = cp.Variable()
            # abs_tol = 1e-6
            # rel_tol = 1e-3

            # objective = cp.Minimize(0.5 * cp.quad_form(beta_temp, H) + f.T @ beta_temp)
            # constraints = [beta_temp >= 0 , err_gaus_l2 >= 0]
            # problem = cp.Problem(objective, constraints)
            # problem.solve(solver = cp.MOSEK)
            # beta_temp.value

            # H_solv = matrix(H)
            # f_solv = matrix(f)
            # solvers.qp(H_solv, f_solv, solver='mosek')


            # solver_instance = quadprog(H, f, A, b, x0, lb, ub)
            # con = lambda beta: 0.5 * beta.T @ H @ beta + f.T @ beta
            #nlc = scipy.optimize.NonlinearConstraint(con, 0, np.inf)

            x0_guess = np.zeros(nLambda)/(np.inf)
            # method = "trust-constr", jac = jacobian, hess=hessian,

            # result = minimize(fun = lambda beta: 0.5 * beta.T @ H @ beta + f.T @ beta,
            #                   x0 = x0_guess.T, method = 'trust-constr',
            #                     bounds= [(0, np.inf) for x in x0_guess],  options=options)
            def objective_function(x, H, f):
                return 0.5 * x.T @ H @ x + f.T @ x
            
            result = minimize(lambda x: objective_function(x, (X_L2.T @ X_L2), -X_L2.T @ LGBs[:,i]),np.zeros(nLambda), bounds=[(0, None)] * nLambda, options=options)

            beta_temp = result.x
            err_gaus_l2[k,i] = result.fun
            err_gaus_l2[k,i] = err_gaus_l2[k,i] + 0.5 * (LGBs[:, i].T @ LGBs[:, i])
            new_beta_store[i, :] = beta_temp

            # result = minimize(lambda x: objective_function(x, (X_L2.T @ X_L2), -X_L2.T @ LGBs[:,i]),np.zeros(nLambda), bounds=[(0, None)] * nLambda, options=options)
            # beta_temp = result.x
            # err_gaus_l2[k, i] = result.fun  # Set err_gaus_l2 to be fval
            # new_beta_store[i, :] = beta_temp
            # err_gaus_l2[k, i] = err_gaus_l2[k, i] + 0.5 * LGBs[:, i].T @ LGBs[:, i]
        L2_store += new_L2_store
        beta_L2 += new_beta_store
        
    L2_store /= n_run
    beta_L2 /= n_run
    #noise_arr[:,:,k] = new_noise_arr
    err_gaus_l2 = np.mean(err_gaus_l2)
    
    Gaus_info = {
        'L2_store': L2_store,
        'beta_L2': beta_L2,
        'LGBs': LGBs,
        'err_gaus_L2': err_gaus_l2,
        'Lambda': Lambda,
        'A': A,
        'T2': T2,
        'TE': TE,
        'SNR': SNR
    }
    
    return Gaus_info


# mosek_lic_path = "/Users/steveh/Downloads/mosek/mosek.lic"
# os.environ["MOSEKLM_LICENSE_FILE"] = mosek_lic_path

# # def generate_gaussian_regs_L2_old(A, T2, TE, SNR, n_run, reg_param_lb, reg_param_ub, N_reg, Nc, cmin, cmax,
# #                                   sigma_min, sigma_max):

# #     # Gaussian dictionary setup
# #     LGBs = Gaussian_basis(T2, cmin, cmax, Nc, sigma_min, sigma_max)

# #     m = len(T2)
# #     n = A.shape[0]

# #     # Setting up regularization parameters
# #     Lambda = np.logspace(reg_param_lb, reg_param_ub, N_reg)
# #     nLambda = len(Lambda)
# #     nc = np.sum(Nc)

# #     # Initialize arrays for storing results
# #     beta_L2 = np.zeros((nc, nLambda))
# #     L2_store = np.zeros((nc, m, nLambda))
# #     err_gaus_l2 = np.zeros(n_run)

# #     for k in range(n_run):
# #         new_L2_store = np.zeros((nc, m, nLambda))
# #         new_beta_store = np.zeros((nc, nLambda))

# #         for i in range(nc):
# #             # Calculate true exp.decaying signals
# #             dat_noiseless = A @ LGBs[:, i]
# #             # Add noise
# #             dat_noisy_ = dat_noiseless + max(dat_noiseless) / SNR * np.random.randn(n)

# #             X_L2 = np.zeros((m, nLambda))
# #             rho_temp = np.zeros(nLambda)
# #             eta_temp = np.zeros(nLambda)

# #             # For each noisy signal, calculate its L2 regularized solutions
# #             for kk in range(len(Lambda)):
# #                 X_L2[:, kk], rho_temp[kk], eta_temp[kk] = nonnegtik_hnorm(A, dat_noisy_, Lambda[kk], '0')

# #             # Store the solutions
# #             new_L2_store[i, :, :] = X_L2
# #             c = -np.dot(X_L2.T, LGBs[:, i])
# #             res = linprog(c, A_eq=X_L2.T, b_eq=LGBs[:, i], method='highs', bounds=(0, None))
# #             beta_temp = res.x
# #             new_beta_store[i, :] = beta_temp.T

# #         L2_store += new_L2_store
# #         beta_L2 += new_beta_store

# #     L2_store /= n_run
# #     beta_L2 /= n_run
# #     err_gaus_l2 = np.mean(err_gaus_l2)

# #     # Store the averaged values
# #     Gaus_info = {
# #         'L2_store': L2_store,
# #         'beta_L2': beta_L2,
# #         'LGBs': LGBs,
# #         'err_gaus_L2': err_gaus_l2,
# #         'Lambda': Lambda,
# #         'A': A,
# #         'T2': T2,
# #         'TE': TE,
# #         'SNR': SNR
# #     }

# #     return Gaus_info

# import numpy as np
# import cvxpy as cp

# def generate_gaussian_regs_L2_old(A, T2, TE, SNR, n_run, reg_param_lb, reg_param_ub, N_reg, Nc, cmin, cmax, sigma_min, sigma_max):
#     # Given the true model Ax = b, compute the L2_regularized solutions by running
#     # multiple times of regularizations with noise added to each Gaussian basis
#     # functions.

#     # Input:
#     # A: Discrete Laplace Transform Matrix
#     # SNR: defines noise level by max(b)/SNR, additive white noise
#     # n_run: number of running times to be averaged to get mean values of
#     #        the outputs.
#     # reg_param_lb: lower bound for regularization parameters
#     # reg_param_ub: upper bound for regularization parameters
#     # nc: number of means in the Gaussian basis
#     # nsigma: number of standard deviations for the Gaussian basis
#     # cmin: lower bound for means for Gaussian basis
#     # cmax: upper bound for means for Gaussian basis
#     # sigma_min: lower bound for stds for Gaussian basis
#     # sigma_max: upper bound for stds for Gaussian basis
#     # T2: discrete T2 values

#     # Output: Python dictionary named "Gaus_info", which contains
#     # beta: the beta values(coefficients) for each Gaussian basis
#     #           represented by their regularized solutions after averaging.
#     # L2_store: averaged L2_regularized solutions for all Gaussian basis
#     #           functions.
#     # LGBs: Gaussian basis.
#     # RHO_L2: averaged model error for each regularization for each Gaussian
#     #         basis.
#     # err_gaus_l2: averaged misfit for the combination of regularized solutions to
#     #              represent the true Gaussian basis.

#     # Created by Chuan Bi, 03/22/2019
#     ## setting up Gaussian basis for computations
#     # Gaussian dictionary set up
#     LGBs = Gaussian_basis(T2, cmin, cmax, Nc, sigma_min, sigma_max)

#     # reshape Gaussian dictionaries so that rows stand for Gaussian basis and
#     # columns stand for T2 values.
#     m = len(T2)
#     n = A.shape[0]

#     # setting up regularization parameters
#     Lambda = np.logspace(reg_param_lb, reg_param_ub, N_reg)

#     nLambda = len(Lambda)
#     nc = sum(Nc)
#     ## Offline computation starts
#     # initializations
#     beta_L2 = np.zeros((nc, nLambda))
#     L2_store = np.zeros((nc, m, nLambda))
#     err_gaus_l2 = np.zeros((n_run, nc))

#     for k in range(n_run):
#         new_L2_store = np.zeros((nc, m, nLambda))
#         new_beta_store = np.zeros((nc, nLambda))

#         for i in range(nc):
#             # calculate true exp.decaying signals
#             dat_noiseless = A @ LGBs[:, i]
#             dat_noiseless = np.column_stack([dat_noiseless])
#             noise =  np.column_stack([(np.max(dat_noiseless) / (SNR * np.random.randn(n, 1)))])
#             # add noise
#             dat_noisy_ = np.add(dat_noiseless, noise)
#             #dat_noisy_ = dat_noiseless + np.max(dat_noiseless) / SNR * np.random.randn(n, 1)
#             #convert to 1D array only.
#             dat_noisy_ = np.ravel(dat_noisy_)

#             X_L2 = np.zeros((m, nLambda))
#             rho_temp = np.zeros(nLambda)
#             eta_temp = np.zeros(nLambda)

#             # for each noisy signal, calculate its L2 regularized solutions
#             for kk in range(len(Lambda)):
#                 X_L2[:,kk],rho_temp[kk],eta_temp[kk] = nonnegtik_hnorm(A,dat_noisy_,Lambda[kk],'0')

#             # store the solutions
#             new_L2_store[i, :, :] = X_L2

#             #CHECK THIS
#             #c = -np.dot(X_L2.T, LGBs[:, i])

#             # Use cvxpy for quadratic programming

#             #Use scipy 

#             beta_temp = cp.Variable(nLambda)
#             #objective = cp.Minimize(0.5 * cp.quad_form(beta_temp, X_L2.T @ X_L2) + cp.sum(cp.multiply(c, beta_temp)))
#             objective = cp.Minimize(cp.sum_squares(X_L2 @ beta_temp - LGBs[:, i]))
#             constraints = [beta_temp >= 0]
#             prob = cp.Problem(objective, constraints)
#             prob.solve()
#             err_gaus_l2[k, i] = prob.value
#             err_gaus_l2[k, i] = err_gaus_l2[k, i] + 0.5 * np.dot(LGBs[:, i].T, LGBs[:, i])
#             new_beta_store[i, :] = beta_temp.value

#             # P = matrix(X_L2.T @ X_L2)
#             # q = matrix(-X_L2.T @ LGBs[:, i])
#             # #G = None  # Initialize G as an nLambda x n matrix filled with zeros
#             # #h = None  # Initialize h as an nLambda x 1 matrix filled with zeros
#             # G = matrix(0.0, (nLambda,nLambda))
#             # h = matrix(0.0, (nLambda, 1))
#             # lb = matrix(0.0, (nLambda, 1))
#             # ub = None  # No upper bounds specified
            
#             # def is_positive_semi_definite(matrix):
#             #     eigenvalues, _ = np.linalg.eig(matrix)
#             #     return np.all(eigenvalues >= 0)

#             # def is_negative_semi_definite(matrix):
#             #     eigenvalues, _ = np.linalg.eig(matrix)
#             #     return np.all(eigenvalues <= 0)
            
#             # if is_positive_semi_definite(P):
#             #     print("P is positive semi-definite.")
#             # else:
#             #     print("P is not positive semi-definite.")

#             # if is_negative_semi_definite(G):
#             #     print("G is negative semi-definite.")
#             # else:
#             #     print("G is not negative semi-definite.")

#             # print("P shape:", P.size)
#             # print("q shape:", q.size)
#             # print("G shape:", G.size)
#             # print("h shape:", h.size)
#             # # Define options for the solver
#             # options = {'show_progress': False}

            
#             # # Solve the quadratic programming problem
#             # sol = solvers.qp(P, q, lb=lb, ub=ub, options=options, solver='cvxopt')

#             # # Extract the optimal solution (beta_temp) and the primal objective value (err_gaus_l2(k, i))
#             # beta_temp = sol['x']
#             # err_gaus_l2[k,i] = sol['primal objective']


#             #err_gaus_l2[k, i] = err_gaus_l2[k, i] + 0.5 * np.dot(LGBs[:, i].T, LGBs[:, i])
#             # H = X_L2.T @ X_L2
#             # f = -X_L2.T @ LGBs[:, i]
#             # err_gaus_l2[k, i] = 0.5 * np.dot(beta_temp.T, np.dot(H, beta_temp)) + np.dot(f.T, beta_temp)

#             #err_gaus_l2[k, i] += 0.5 * np.dot(LGBs[:, i].T, LGBs[:, i])

#             #new_beta_store[i, :] = beta_temp

#             # beta_temp = cp.Variable(nLambda)
#             # objective = cp.Minimize(0.5 * cp.quad_form(beta_temp, X_L2.T @ X_L2) + cp.sum(cp.multiply(-X_L2.T @ LGBs[:, i], beta_temp)))
#             # constraints = [beta_temp >= 0]
#             # prob = cp.Problem(objective, constraints)
#             # prob.solve(solver=cp.MOSEK, verbose=False)
#             # err_gaus_l2[k, i] = err_gaus_l2[k, i] + 0.5 * np.dot(LGBs[:, i].T, LGBs[:, i])
#             # new_beta_store[i, :] = beta_temp.value

#             # beta_temp = cp.Variable(nLambda)
#             # #objective = cp.Minimize(0.5 * cp.quad_form(beta_temp, X_L2.T @ X_L2) + cp.sum(cp.multiply(-X_L2.T @ LGBs[:, i], beta_temp)))
#             # objective = cp.Minimize(0.5 * cp.sum_squares(X_L2 @ beta_temp - LGBs[:, i]) + cp.sum(cp.multiply(-X_L2.T @ LGBs[:, i], beta_temp)))
#             # #objective = cp.Minimize(0.5 * cp.sum_squares(X_L2 @ beta_temp) + cp.sum(cp.multiply(-X_L2.T @ LGBs[:, i], beta_temp)))
#             # constraints = [beta_temp >= 0]  # Additional constraint for non-negative coefficients
#             # prob = cp.Problem(objective,constraints)
#             # prob.solve(solver=cp.MOSEK, verbose=False)

#             # err_gaus_l2[k, i] = err_gaus_l2[k, i] + 0.5 * np.dot(LGBs[:, i].T, LGBs[:, i])
#             # #beta_temp_value = beta_temp.value
#             # #beta_temp_value = np.squeeze(beta_temp.value)  # Reshape to match dat_noisy_; CHECK THIS
#             # new_beta_store[i, :] = beta_temp.value
#             # beta_temp = cp.Variable(nLambda)
#             # objective = cp.Minimize(0.5 * cp.quad_form(beta_temp, X_L2.T @ X_L2) + cp.sum(cp.multiply(-X_L2.T @ LGBs[:, i], beta_temp)))
#             # constraints = [beta_temp >= 0]  # Additional constraint for non-negative coefficients
#             # prob = cp.Problem(objective, constraints)
#             # prob.solve()

#             # err_gaus_l2[k, i] = err_gaus_l2[k, i] + 0.5 * np.dot(LGBs[:, i].T, LGBs[:, i])
#             # new_beta_store[i, :] = beta_temp.value

#             # beta_temp = cp.Variable(nLambda)
#             # obj = cp.Minimize(cp.norm(A @ beta_temp - LGBs[:, i]) + 0.5 * cp.sum_squares(beta_temp))
#             # problem = cp.Problem(obj)
#             # problem.solve()

#             # err_gaus_l2[k, i] = problem.value + 0.5 * np.linalg.norm(LGBs[:, i])**2  # Add the 0.5 * ||LGBs[:, i]||^2 term

#             # new_beta_store[i, :] = beta_temp.value

#             # beta_temp = cp.Variable(nLambda)
#             #  # Define the objective function to minimize (quadratic)
#             # obj = cp.Minimize(0.5 * cp.quad_form(beta_temp, X_L2.T @ X_L2) - X_L2.T @ LGBs[:, i] @ beta_temp)
#             # #constraints = [beta_temp >= 0]
#             # problem = cp.Problem(obj)
#             # problem.solve()

#             # err_gaus_l2[k, i] = problem.value + 0.5 * np.linalg.norm(LGBs[:, i])**2  # Add the 0.5 * ||LGBs[:, i]||^2 term

#             # new_beta_store[i, :] = beta_temp.value
#             # beta_var = cp.Variable(nLambda)
#             # objective = cp.Minimize(cp.sum_squares(X_L2 @ beta_var - LGBs[:, i]) + 0.5 * cp.sum_squares(beta_var))
#             # prob = cp.Problem(objective, [cp.sum(beta_var) == 1])
#             # prob.solve()
#             #new_beta_store[i, :] = beta_var.value

#             # beta_temp = cp.Variable((nLambda, 1))
#             # objective = cp.Minimize(0.5 * cp.quad_form(beta_temp, X_L2.T @ X_L2) -
#             #                         cp.sum(X_L2.T @ LGBs[i, :].T @ beta_temp))
            
#             # err_gaus_l2_k_i = obj.value + 0.5 * np.dot(LGBs[:, i], LGBs[:, i])
#             # err_gaus_l2[k, i] = err_gaus_l2_k_i
#             # new_beta_store[i, :] = beta_temp.value

#         L2_store += new_L2_store
#         beta_L2 += new_beta_store

#     L2_store /= n_run
#     beta_L2 /= n_run
#     #err_gaus_l2 = np.mean(err_gaus_l2, axis=0)
#     err_gaus_l2 = np.mean(err_gaus_l2)


#     # store the averaged values
#     Gaus_info = {
#         'L2_store': L2_store,
#         'beta_L2': beta_L2,
#         'LGBs': LGBs,
#         'err_gaus_L2': err_gaus_l2,
#         'Lambda': Lambda,
#         'A': A,
#         'T2': T2,
#         'TE': TE,
#         'SNR': SNR
#     }

#     return Gaus_info

#Gaus_info = scipy.io.loadmat('/Users/steveh/Downloads/Gaus_noise_arr.mat')
#/Users/steveh/Downloads/noisearr1000for10diffnoisereal3.mat
#Schedule meetings with Chuan and Richard about the updates
#noise_arr = Gaus_info['noise_arr']

# import cvxopt

# class quadprog(object):

#     def __init__(self, H, f, A, b, x0, lb, ub):
#         self.H    = H
#         self.f    = f
#         self.A    = A
#         self.b    = b
#         self.x0   = x0
#         self.bnds = tuple([(lb, ub) for x in x0])
#         # call solver
#         self.result = self.solver()

#     def objective_function(self, x):
#         return 0.5*np.dot(np.dot(x.T, self.H), x) + np.dot(self.f.T, x)

#     def solver(self):
#         cons = ({'type': 'ineq', 'fun': lambda x: self.b - np.dot(self.A, x)})
#         optimum = minimize(self.objective_function, 
#                                     x0          = self.x0.T,
#                                     bounds      = self.bnds,
#                                     constraints = cons, 
#                                     tol         = 10**-3)
#         return optimum



# def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
#     """
#     Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
#     Output: Numpy array of the solution
#     """
#     n_var = H.shape[1]

#     P = cvxopt.matrix(H, tc='d')
#     q = cvxopt.matrix(f, tc='d')

#     if L is not None or k is not None:
#         assert(k is not None and L is not None)
#         if lb is not None:
#             L = np.vstack([L, -np.eye(n_var)])
#             k = np.vstack([k, -lb])

#         if ub is not None:
#             L = np.vstack([L, np.eye(n_var)])
#             k = np.vstack([k, ub])

#         L = cvxopt.matrix(L, tc='d')
#         k = cvxopt.matrix(k, tc='d')

#     if Aeq is not None or beq is not None:
#         assert(Aeq is not None and beq is not None)
#         Aeq = cvxopt.matrix(Aeq, tc='d')
#         beq = cvxopt.matrix(beq, tc='d')

#     sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq, options={'show_progress': False, 'maxiters': 100})

#     return np.array(sol['x'])

# from cvxopt import matrix, solvers