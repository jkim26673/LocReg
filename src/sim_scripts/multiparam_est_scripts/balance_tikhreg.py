"""
Taken from Broydan method algorithm from Ito et al. Paper

Input: 

Rho: vector of regularization parameters
Psi: vector of roughening matrices
delta: noise level
G: kernel matrix
data_noisy:  data vector with polluted noise
s_factor: safety factor >= 1

Output:
x_vec : Solution Vector
"""
import numpy as np
import torch
from torch.autograd.functional import jacobian

def bal_tikhreg(rho, psi, delta, G, data_noisy, s_factor):
    #Construct the T(rho) according to the Ito Paper:
    #set up iteration number k
    k = 0
    eps = 1e-10
    alp_comp = 0
    bal_comp = 0
    for i in range(len(rho)):
        alp_comp += rho[i] * psi[i].T @ psi[i]
        bal_comp += rho[i] * psi[i]
    u_star = np.linalg.inv(G.T @ G + alp_comp) @ (G.T @ data_noisy)
    phi_u_g = 0.5 * np.linalg.norm(G @ u_star - data_noisy, 2) ** 2 
    # phi_u_g = 0.5 * s_factor**2 * delta**2
    T_rho = np.array([(phi_u_g - 0.5 * delta**2 + rho[0]*psi[0] -  rho[1]*psi[1]),(phi_u_g - 0.5 * delta**2  + rho[1]*psi[1] - rho[0]*psi[0])])
    #Compute jacobian
    def T_rho(rhos,psis):
        return np.array([(phi_u_g - 0.5 * delta**2 + rhos[0]*psis[0] -  rhos[1]*psis[1]),(phi_u_g - 0.5 * delta**2  + rhos[1]*psis[1] - rhos[0]*psis[0])])  
    init_rho = rho
    init_T_rho = T_rho(init_rho,psi)
    input = (torch.tensor([init_rho]), torch.tensor([psi]))
    init_jac_T_rho = jacobian(T_rho, input)

    curr_T_rho = init_T_rho
    curr_jac_T_rho = init_jac_T_rho
    curr_rho = init_rho

    while True:
        delta_rho = -np.linalg.inv(curr_jac_T_rho) @ T_rho(curr_rho,psi)
        new_rho = curr_rho + delta_rho
        new_T_rho = T_rho(new_rho, psi)
        delta_T = new_T_rho - curr_T_rho
        # new_jac_T_rho = curr_jac_T_rho + (1/np.linalg.norm(delta_rho)**2) * (delta_T - )
        new_jac_T_rho = (((delta_T @ delta_rho)/(np.linalg.norm(delta_rho)**2)) + curr_jac_T_rho) /( 1 + (delta_rho / np.linalg.norm(delta_rho)**2))
        if np.linalg.norm(curr_T_rho) < eps:
            break
        else:
            curr_jac_T_rho = new_jac_T_rho
            curr_rho = new_rho
            
    return new_rho

