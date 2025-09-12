#------------------------------------------------------------------------------+
#
#	Nathan A. Rooy
#	Simple Particle Swarm Optimization (PSO) with Python
#	Last update: 2018-JAN-26
#	Python 3.6
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import random
import math
import numpy as np
import sys
sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
from numpy.linalg import norm
from regu.csvd import csvd
from regu.tikhonov import tikhonov
from regu.discrep import discrep
from regu.l_curve import l_curve
from Utilities_functions.discrep_L2 import discrep_L2
from pyswarm import pso

#--- COST FUNCTION ------------------------------------------------------------+

# Define the size of matrices
nT2 = 200
half_nT2 = int(nT2/2)
T2 = np.arange(1,201)
TE = np.arange(1,512,4)
# Create half_mat1
half_mat1 = np.eye(half_nT2)

# Initialize B_1 and B_2 matrices
B_1 = np.zeros((nT2, nT2))
B_2 = np.zeros((nT2, nT2))

# Assign values to B_1 and B_2
B_1[:half_nT2, :half_nT2] = half_mat1[:half_nT2, :half_nT2]
B_2[-half_nT2:, -half_nT2:] = half_mat1[::-1, ::-1]

def generate_B_matrices(N, M):
    B_matrices = []

    if M % N != 0:
        ValueError("M must be divisible by N")
    ones_per_matrix = M // N  # Calculate number of ones per matrix

    for i in range(N):
        start = ones_per_matrix * i  # Calculate the starting index for ones in this matrix
        end = min(ones_per_matrix * (i + 1), M)  # Calculate the ending index for ones in this matrix
        diagonal_values = np.zeros(M)
        diagonal_values[start:end] = 1  # Set ones in the appropriate range
        B_matrices.append(np.diag(diagonal_values))

    return B_matrices

# Example usage
N = 200
M = nT2
B_matrices = generate_B_matrices(N, M)

# Print the generated B matrices
for i, B_matrix in enumerate(B_matrices):
    print(f"B_{i+1}:")
    print(B_matrix)


G_mat = np.zeros((len(TE), nT2))  
for i in range(len(TE)):
    for j in range(len(TE)):
        G_mat[i,j] = np.exp(-TE[i]/T2[j])
G = G_mat

mu1 = 40
mu2 = 150
sigma1 = 4
sigma2 = 20
SNR = 200
g1 = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
g2 = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
g  = 0.5 * (g1 + g2)

data_noiseless = np.dot(G, g)
SD_noise= 1/SNR*max(abs(data_noiseless))
noise = np.random.normal(0,SD_noise, data_noiseless.shape)
data_noisy = data_noiseless + noise
U, s, V = csvd(G, tst = None, nargin = 1, nargout = 3)

def minimize_DP(SNR):
    delta = np.linalg.norm(noise)*1.05 
    # delta = norm(self.noise)**2
    x_delta,lambda_OP = discrep(U,s,V,data_noisy,delta, x_0= None, nargin = 5)
    Lambda = np.ones(len(T2))* lambda_OP

    f_rec_DP_OP,_ = discrep_L2(data_noisy, G, SNR, Lambda)
    f_rec_DP_OP = f_rec_DP_OP.flatten()
    return lambda_OP, f_rec_DP_OP

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
    total=0
    # for i in range(len(x)):
    #     total+=x[i]**2
    if len(x) == 2:
        total = x[0]**2 - 10*np.cos(2 * np.pi * x[0]) + x[1]**2 - 10 * np.cos(2*np.pi* x[1])
    else:
        TypeError("Needs to be 2 dimensional")
    return total

def func2(alp,G,d,nT2):
    total=0
    # for i in range(len(x)):
    #     total+=x[i]**2
    I = np.eye(nT2)
    alpmat = 0
    for i in range(len(alp)):
        alpmat += alp[i] * B_matrices[i].T @ B_matrices[i]
    inv = np.linalg.inv(G.T @ G + alpmat)
    # print("cond num:", np.linalg.cond(G.T @ G + alp[0] * B_1.T @ B_1 + alp[1] * B_2.T @ B_2))
    # p = np.linalg.inv(G.T @ G + alp[0] * B_1.T @ B_1 + alp[1] * B_2.T @ B_2 + alp[2] * B_3.T @ B_3 + alp[3] * B_4.T @ B_4) @ G.T @ d
    p = inv @ G.T @ d
    # p = np.linalg.inv(G.T @ G + alp[0] * B_matrices[0].T @ B_matrices[0] + alp[1] * B_matrices[1].T @ B_matrices[1] + alp[2] * B_matrices[2].T @ B_matrices[2] + alp[3] * B_matrices[3].T @ B_matrices[3]) @ G.T @ d
    alp_norm = 0
    for i in range(len(alp)):
        alp_norm += alp[i] * np.linalg.norm(B_matrices[i] @ p , 2)**2
    # print(np.linalg.cond(np.linalg.inv(G.T @ G + alp[0] * B_matrices[0].T @ B_matrices[0] + alp[1] * B_matrices[1].T @ B_matrices[1] + alp[2] * B_matrices[2].T @ B_matrices[2] + alp[3] * B_matrices[3].T @ B_matrices[3])))
    total = np.linalg.norm(G @ p - d, 2)**2 + alp_norm
    # total = np.linalg.norm(G @ p - d, 2)**2 + alp[0] * np.linalg.norm(B_matrices[0] @ p ,2)**2 +  alp[1] * np.linalg.norm(B_matrices[1] @ p, 2)**2  + alp[2] * np.linalg.norm(B_matrices[2] @ p, 2)**2 + alp[3] * np.linalg.norm(B_matrices[3] @ p, 2)**2
    # print(total)
    # if len(x) == 2:
    #     total = x[0]**2 - 10*np.cos(2 * np.pi * x[0]) + x[1]**2 - 10 * np.cos(2*np.pi* x[1])
    # else:
    #     TypeError("Needs to be 2 dimensional")
    return total


#--- MAIN ---------------------------------------------------------------------+

# class Particle:
#     def __init__(self,x0):
#         self.position_i=[]          # particle position
#         self.velocity_i=[]          # particle velocity
#         self.pos_best_i=[]          # best position individual
#         self.err_best_i=-1          # best error individual
#         self.err_i=-1               # error individual

#         for i in range(0,num_dimensions):
#             self.velocity_i.append(random.uniform(-1,1))
#             self.position_i.append(x0[i])

#     # evaluate current fitness
#     def evaluate(self,costFunc):
#         self.err_i=costFunc(self.position_i, G, data_noisy, nT2)

#         # check to see if the current position is an individual best
#         if self.err_i<self.err_best_i or self.err_best_i==-1:
#             self.pos_best_i=self.position_i.copy()
#             self.err_best_i=self.err_i
                    
#     # update new particle velocity
#     def update_velocity(self,pos_best_g):
#         w=0.5       # constant inertia weight (how much to weigh the previous velocity)
#         c1=2        # cognative constant
#         c2=2        # social constant
        
#         for i in range(0,num_dimensions):
#             r1=random.random()
#             r2=random.random()
            
#             vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
#             vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
#             self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

#     # update the particle position based off new velocity updates
#     def update_position(self,bounds):
#         for i in range(0,num_dimensions):
#             self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            
#             # adjust maximum position if necessary
#             if self.position_i[i]>bounds[i][1]:
#                 self.position_i[i]=bounds[i][1]

#             # adjust minimum position if neseccary
#             if self.position_i[i]<bounds[i][0]:
#                 self.position_i[i]=bounds[i][0]

# class PSO():
#     def __init__(self, costFunc, x0, bounds, num_particles, maxiter, verbose=False):
#         global num_dimensions

#         num_dimensions=len(x0)
#         err_best_g=-1                   # best error for group
#         pos_best_g=[]                   # best position for group

#         # establish the swarm
#         swarm=[]
#         for i in range(0,num_particles):
#             swarm.append(Particle(x0))

#         # begin optimization loop
#         i=0
#         while i<maxiter:
#             if verbose: print(f'iter: {i:>4d}, best solution: {err_best_g:10.6f}')
#             # cycle through particles in swarm and evaluate fitness
#             for j in range(0,num_particles):
#                 swarm[j].evaluate(costFunc)

#                 # determine if current particle is the best (globally)
#                 if swarm[j].err_i<err_best_g or err_best_g==-1:
#                     pos_best_g=list(swarm[j].position_i)
#                     err_best_g=float(swarm[j].err_i)
            
#             # cycle through swarm and update velocities and position
#             for j in range(0,num_particles):
#                 swarm[j].update_velocity(pos_best_g)
#                 swarm[j].update_position(bounds)
#             i+=1

#         # print final results
#         print('\nFINAL SOLUTION:')
#         print(f'   > {pos_best_g}')
#         print(f'   > {err_best_g}\n')
#         self.alp1PSO = pos_best_g[0]
#         self.alp2PSO = pos_best_g[1]
#         self.alp3PSO = pos_best_g[2]
#         self.alp4PSO = pos_best_g[3]




#--- RUN ----------------------------------------------------------------------+

# best_lam, f_rec = minimize_DP(SNR)
# print("best_DP_lam:", best_lam)
# initial=[1,1,1,1]               # initial starting location [x1,x2...]
# bounds=[(0,1), (0,1), (0,1), (0,1)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
# pso_instance = PSO(func2, initial, bounds, num_particles=50, maxiter=100, verbose=True)

def main():
    from scipy.optimize import nnls 
    import os
    import mosek
    import cvxpy as cp
    mosek_lic_path = "/Users/steveh/Downloads/mosek/mosek.lic"
    os.environ["MOSEKLM_LICENSE_FILE"] = mosek_lic_path


    def pstar(alp):
        # alp1 = alp[0]
        # alp2 = alp[1]
        # alp3 = alp[2]
        # alp4 = alp[3]
        alpmat = 0
        for i in range(len(alp)):
            alpmat += alp[i] * B_matrices[i].T @ B_matrices[i]
        inv = np.linalg.inv(G.T @ G + alpmat)
        # print("cond num:", np.linalg.cond(G.T @ G + alp[0] * B_1.T @ B_1 + alp[1] * B_2.T @ B_2))
        # p = np.linalg.inv(G.T @ G + alp[0] * B_1.T @ B_1 + alp[1] * B_2.T @ B_2 + alp[2] * B_3.T @ B_3 + alp[3] * B_4.T @ B_4) @ G.T @ d
        p = inv @ G.T @ data_noisy
        # A = G.T @ G + alpmat
        # b =  G.T @ data_noisy
        # x = cp.Variable(nT2)
        # cost = cp.norm(A @ x - b, 2)**2
        # constraints = [x >= 0]
        # problem = cp.Problem(cp.Minimize(cost), constraints)
        # # exp,_ = fnnls(A,b)
        # problem.solve(solver=cp.MOSEK, verbose=False)
        # p = x.value
        # p = np.linalg.inv(G.T @ G + alp[0] * B_matrices[0].T @ B_matrices[0] + alp[1] * B_matrices[1].T @ B_matrices[1] + alp[2] * B_matrices[2].T @ B_matrices[2] + alp[3] * B_matrices[3].T @ B_matrices[3]) @ G.T @ data_noisy
        return p

    def func2(alp):
        # alp1 = alp[0]
        # alp2 = alp[1]
        # alp3 = alp[2]
        # alp4 = alp[3]
        # p = args
        # for i in range(len(x)):
        #     total+=x[i]**2
        I = np.eye(nT2)

        # print("cond num:", np.linalg.cond(G.T @ G + alp[0] * B_1.T @ B_1 + alp[1] * B_2.T @ B_2))
        # p = np.linalg.inv(G.T @ G + alp[0] * B_1.T @ B_1 + alp[1] * B_2.T @ B_2 + alp[2] * B_3.T @ B_3 + alp[3] * B_4.T @ B_4) @ G.T @ d
        p = pstar(alp)
        # total = np.linalg.norm(G @ p - data_noisy, 2)**2 + alp[0] * np.linalg.norm(B_matrices[0] @ p ,2)**2 +  alp[1] * np.linalg.norm(B_matrices[1] @ p, 2)**2  + alp[2] * np.linalg.norm(B_matrices[2] @ p, 2)**2 + alp[3] * np.linalg.norm(B_matrices[3] @ p, 2)**2
        alp_norm = 0
        for i in range(len(alp)):
            alp_norm += alp[i] * np.linalg.norm(B_matrices[i] @ p , 2)**2
        # print(np.linalg.cond(np.linalg.inv(G.T @ G + alp[0] * B_matrices[0].T @ B_matrices[0] + alp[1] * B_matrices[1].T @ B_matrices[1] + alp[2] * B_matrices[2].T @ B_matrices[2] + alp[3] * B_matrices[3].T @ B_matrices[3])))
        total = np.linalg.norm(G @ p - data_noisy, 2)**2 + alp_norm
        # A = G.T @ G + alp[0] * B_matrices[0].T @ B_matrices[0] + alp[1] * B_matrices[1].T @ B_matrices[1] + alp[2] * B_matrices[2].T @ B_matrices[2] + alp[3] * B_matrices[3].T @ B_matrices[3]
        # b =  G.T @ data_noisy
        # x = cp.Variable(nT2)
        # cost = cp.norm(A @ x - b, 2)**2
        # constraints = [x >= 0]
        # problem = cp.Problem(cp.Minimize(cost), constraints)
        # # exp,_ = fnnls(A,b)
        # problem.solve(solver=cp.MOSEK, verbose=False)
        # p = x.value
        # print("p val:", p)
        # total = np.linalg.norm(G @ p - data_noisy, 2)**2 + alp[0] * np.linalg.norm(B_matrices[0] @ p ,2)**2 +  alp[1] * np.linalg.norm(B_matrices[1] @ p, 2)**2  + alp[2] * np.linalg.norm(B_matrices[2] @ p, 2)**2 + alp[3] * np.linalg.norm(B_matrices[3] @ p, 2)**2
        # print(total)
        # if len(x) == 2:
        #     total = x[0]**2 - 10*np.cos(2 * np.pi * x[0]) + x[1]**2 - 10 * np.cos(2*np.pi* x[1])
        # else:
        #     TypeError("Needs to be 2 dimensional")
        return total


    def constraints(alp):
        # alp1 = alp[0]
        # alp2 = alp[1]
        # alp3 = alp[2]
        # alp4 = alp[3]
        p = pstar(alp)
        # print("p in constraint:", p)
        # p = np.linalg.inv(G.T @ G + alp[0] * B_matrices[0].T @ B_matrices[0] + alp[1] * B_matrices[1].T @ B_matrices[1] + alp[2] * B_matrices[2].T @ B_matrices[2] + alp[3] * B_matrices[3].T @ B_matrices[3]) @ G.T @ data_noisy
        return [1e-12-p]


    # def constraints(alp):
    #     p = pstar(alp)
    #     return [-np.min(p)]

    best_lam, f_rec = minimize_DP(SNR)
    print("best_DP_lam:", best_lam)
    initial= [1] * N               # initial starting location [x1,x2...]
    bounds=[(0,1)] * N  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    # pso_instance = PSO(func2, initial, bounds, num_particles=2, maxiter=20, verbose=True)
    lb = [1e-8] * N
    ub = [1] * N

    # pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
    #     swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8,
    #     minfunc=1e-8, debug=False)

    xopt, fopt = pso(func2, lb, ub, f_ieqcons=constraints, swarmsize=100, omega=0.5, phip=5, phig=5, maxiter=100, minstep=1e-12,minfunc=1e-12, debug=False)

    xoptval = 0
    for i in range(len(xopt)):
        xoptval += xopt[i] * B_matrices[i].T @ B_matrices[i]

    A = G.T @ G + xoptval
    b =  G.T @ data_noisy
    x = cp.Variable(nT2)
    cost = cp.norm(A @ x - b, 2)**2
    constraints = [x >= 0]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    # exp,_ = fnnls(A,b)
    problem.solve(solver=cp.MOSEK, verbose=False)
    p_PSO = x.value
    error_PSO = np.linalg.norm(g - p_PSO)
    error_DP = np.linalg.norm(g - f_rec)
    print("error_PSO:", error_PSO)
    print("error_DP:", error_DP)

    # inv = np.linalg.inv(G.T @ G + xoptval)
    # p_PSO = inv @ G.T @ data_noisy
    # p_PSO = np.linalg.inv(G.T @ G + xopt[0] * B_matrices[0].T @ B_matrices[0] + xopt[1] * B_matrices[1].T @ B_matrices[1] + xopt[2] * B_matrices[2].T @ B_matrices[2] + xopt[3] * B_matrices[3].T @ B_matrices[3] ) @ G.T @ data_noisy
    import matplotlib.pyplot as plt
    plt.plot(p_PSO, label = f"{N}P_PSO")
    plt.plot(g, label = "Ground Truth")
    plt.plot(f_rec, label = "1P DP")
    plt.legend(['PSO', 'Ground Truth', "DP"])
    # plt.text(0.5, 0.6, f"Error PSO: {error_PSO}", ha='center', va='center', fontsize=12, color='red')
    # plt.text(0.5, 0.6, f"Error DP: {error_DP}", ha='center', va='center', fontsize=12, color='red')
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
# def func2(alp):
#     total=0
#     # for i in range(len(x)):
#     #     total+=x[i]**2
#     I = np.eye(nT2)

#     # print("cond num:", np.linalg.cond(G.T @ G + alp[0] * B_1.T @ B_1 + alp[1] * B_2.T @ B_2))
#     # p = np.linalg.inv(G.T @ G + alp[0] * B_1.T @ B_1 + alp[1] * B_2.T @ B_2 + alp[2] * B_3.T @ B_3 + alp[3] * B_4.T @ B_4) @ G.T @ d
#     p = np.linalg.inv(G.T @ G + alp[0] * B_matrices[0].T @ B_matrices[0] + alp[1] * B_matrices[1].T @ B_matrices[1] + alp[2] * B_matrices[2].T @ B_matrices[2] + alp[3] * B_matrices[3].T @ B_matrices[3]) @ G.T @ data_noisy

#     total = np.linalg.norm(G @ p - data_noisy, 2)**2 + alp[0] * np.linalg.norm(B_matrices[0] @ p ,2)**2 +  alp[1] * np.linalg.norm(B_matrices[1] @ p, 2)**2  + alp[2] * np.linalg.norm(B_matrices[2] @ p, 2)**2 + alp[3] * np.linalg.norm(B_matrices[3] @ p, 2)**2
#     # print(total)
#     # if len(x) == 2:
#     #     total = x[0]**2 - 10*np.cos(2 * np.pi * x[0]) + x[1]**2 - 10 * np.cos(2*np.pi* x[1])
#     # else:
#     #     TypeError("Needs to be 2 dimensional")
#     return total

# def constraints(alp):
#     alp1 = alp[0]
#     alp2 = alp[1]
#     alp3 = alp[2]
#     alp4 = alp[3]
#     return [-alp1, -alp2, -alp3, -alp4]

# best_lam, f_rec = minimize_DP(SNR)
# print("best_DP_lam:", best_lam)
# initial=[0.3, 0.5, 0.4, 0.3]               # initial starting location [x1,x2...]
# bounds=[(0,1), (0,1), (0,1), (0,1)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
# # pso_instance = PSO(func2, initial, bounds, num_particles=2, maxiter=20, verbose=True)
# lb = [0, 0, 0,0]
# ub = [1, 1, 1,1]

# # pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
# #     swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8,
# #     minfunc=1e-8, debug=False)

# xopt, fopt = pso(func2, lb, ub, f_ieqcons=constraints, swarmsize=20, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8,minfunc=1e-8, debug=True)
