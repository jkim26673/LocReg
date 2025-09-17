import numpy as np
from src.regularization.reg_methods.spanreg.Gaussian_basis import Gaussian_basis
from src.regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm
import os
from scipy.optimize import minimize, LinearConstraint

def test_gen_gaussian(A, T2, TE, SNR, n_run, reg_param_lb, reg_param_ub, N_reg, Nc, cmin, cmax, sigma_min, sigma_max):
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
        cond_num_store = []
        
        for i in range(nc):
            dat_noiseless = A @ LGBs[:, 0]

            noise =  np.column_stack([(np.max(dat_noiseless) / (SNR * np.random.randn(n, 1)))])
            # add noise
            dat_noisy_ = dat_noiseless + np.ravel(noise)
            
            X_L2 = np.zeros((m, nLambda))
            rho_temp = np.zeros((nLambda,1))
            eta_temp = np.zeros((nLambda,1))
            
            #variability here at nnls,lsqnonnneg vs lsqnnoneg from matlab
            for kk in range(len(Lambda)):
                X_L2[:, kk], rho_temp[kk], eta_temp[kk] = nonnegtik_hnorm(A, dat_noisy_, Lambda[kk], '0')
            
            cond_num_store = np.append(cond_num_store, np.linalg.cond(X_L2))
            # print("X_L2 condition number:", np.linalg.cond(X_L2))
            #check condition number X_L2 (link); check over many iterations

            def objective_function(x, H, f):
                return 0.5 * x.T @ H @ x + f.T @ x

            new_L2_store[i, :, :] = X_L2
            # print("X_L2 shape:", X_L2.shape)
 
            H = np.dot(X_L2.T, X_L2)

            c = -X_L2.T @ LGBs[:, 0]

            # print("c:", c)
            # print("c shape:", c.shape)

            # nonneg_constraint = LinearConstraint(np.eye(nLambda), 0, np.inf)
            # constraints = [nonneg_constraint]

            # Define constraints for beta_temp
            cons = {'type': 'ineq', 'fun': lambda beta: -beta}
    
            result = minimize(fun = lambda beta: 0.5 * (beta.T @ H @ beta) + np.dot(c.T, beta),
                              x0 = np.ones(nLambda)/nLambda, bounds=[(0, None)] * nLambda, tol = 1e-6, options=options)
    
            beta_temp = result.x
            # print("beta_temp:", beta_temp)
            # print("beta_temp shape:", beta_temp.shape)

            err_gaus_l2[k,i] = result.fun
            err_gaus_l2[k,i] = err_gaus_l2[k,i] + 0.5 * np.dot(LGBs[:, 0].T, LGBs[:, 0])
            new_beta_store[i, :] = beta_temp

            # result = minimize(lambda x: objective_function(x, (X_L2.T @ X_L2), -X_L2.T @ LGBs[:,i]),np.zeros(nLambda), bounds=[(0, None)] * nLambda, options=options)
            # beta_temp = result.x
            # err_gaus_l2[k, i] = result.fun  # Set err_gaus_l2 to be fval
            # new_beta_store[i, :] = beta_temp
            # err_gaus_l2[k, i] = err_gaus_l2[k, i] + 0.5 * LGBs[:, i].T @ LGBs[:, i]
        L2_store += new_L2_store
        beta_L2 += new_beta_store
        
    L2_store = L2_store/n_run
    beta_L2 = beta_L2/n_run
    err_gaus_l2 = np.mean(err_gaus_l2)
    
    Gaus_info = {
        'L2_store': L2_store,
        'beta_L2': beta_L2,
        'LGBs': LGBs,
        'cond_num': cond_num_store,
        'err_gaus_L2': err_gaus_l2,
        'Lambda': Lambda,
        'A': A,
        'T2': T2,
        'TE': TE,
        'SNR': SNR
    }

    return Gaus_info
