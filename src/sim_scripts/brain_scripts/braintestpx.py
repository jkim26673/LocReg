
# import pickle
# import scipy.io
# from scipy.ndimage import rotate
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import wasserstein_distance as wass
# import sys
# import os
# print("Setting system path")
# sys.path.append(".")  # Replace this path with the actual path to the parent directory of Utilities_functions
# import numpy as np
# from scipy.stats import norm as normsci
# from scipy.linalg import norm as linalg_norm
# from scipy.optimize import nnls
# import matplotlib.pyplot as plt
# import pickle
# from Utilities_functions.discrep_L2 import discrep_L2, discrep_L2_sp, discrep_L2_brain
# from Utilities_functions.GCV_NNLS import GCV_NNLS
# from Utilities_functions.Lcurve import Lcurve
# import pandas as pd
# import cvxpy as cp
# from scipy.linalg import svd
# from regu.csvd import csvd
# from regu.discrep import discrep
# from Simulations.LRalgo import LocReg_Ito_mod as oLocReg_Ito_mod
# from Simulations.brainLRalgo import LocReg_Ito_mod as bLocReg_Ito_mod
# from Utilities_functions.pasha_gcv import Tikhonov
# from regu.l_curve import l_curve
# from tqdm import tqdm
# from Utilities_functions.tikhonov_vec import tikhonov_vec
# import mosek
# import seaborn as sns
# from regu.nonnegtik_hnorm import nonnegtik_hnorm
# import multiprocess as mp
# from multiprocessing import Pool, freeze_support
# from multiprocessing import set_start_method
# import functools
# from datetime import date
# import random
# import cProfile
# import pstats
# from Simulations.resolutionanalysis import find_min_between_peaks, check_resolution
# import logging
# import time
# from scipy.stats import wasserstein_distance
# import matplotlib.ticker as ticker  # Add this import
# import scipy
# import timeit
from utils.load_imports.loading import *
import scipy.io
from scipy.ndimage import rotate
from Simulations.LRalgo import LocReg_Ito_mod as oLocReg_Ito_mod
from Simulations.brainLRalgo import LocReg_Ito_mod as bLocReg_Ito_mod


brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat")["final_data_2"]
maskfilepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/masks/mask_2.mat"
# estimatesfilepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results/est_table_xcoordlen_313_ycoordlen_31328Nov24.pkl"
# estimatesfilepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/results/est_table_xcoordlen_313_ycoordlen_313_withLS_OR30Nov24.pkl"
# estimatesfilepath =  "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_noiseadd_recovery/est_table_xcoordlen_313_ycoordlen_313_withLS_OR10Dec24.pkl"

estimatesfilepath = ""

with open (estimatesfilepath, "rb") as file:
    df = pickle.load(file)

#unique px
px = df[(df["X_val"] == 118) & (df["Y_val"] == 117)]
px["MWF_DP"]
px["MWF_LR"]

#Parallelization Switch
parallel = False
num_cpus_avail = os.cpu_count()
#LocReg Hyperparameters
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
feedback = True
# lam_ini_val = "LCurve"
lam_ini_val = "LCurve"
# gamma_init = 0.5
gamma_init = 0.5
maxiter = 1000

#CVXPY global parameters
eps = 1e-2

#Error Metric
err_type = "Wass. Score"

#Lambda Space
Lambda = np.logspace(-6,1,50).reshape(-1,1)
curr_SNR = 1
#Key Functions
passing_x_coord = 165
passing_y_coord = 154
def wass_error(IdealModel,reconstr):
    err = wasserstein_distance(IdealModel,reconstr)
    return err

def l2_error(IdealModel,reconstr):
    true_norm = linalg_norm(IdealModel)
    err = linalg_norm(IdealModel - reconstr) / true_norm
    return err

def cvxpy_tikhreg(Lambda, G, data_noisy):
    """
    Alternative way of performing non-negative Tikhonov regularization.
    """
    # lam_vec = Lambda * np.ones(G.shape[1])
    # A = (G.T @ G + np.diag(lam_vec))
    # ep4 = np.ones(A.shape[1]) * eps
    # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vec
    # y = cp.Variable(G.shape[1])
    # cost = cp.norm(A @ y - b, 2)**2
    # constraints = [y >= 0]
    # problem = cp.Problem(cp.Minimize(cost), constraints)
    # problem.solve(solver=cp.MOSEK, verbose=False)
    # sol = y.value
    # sol = sol - eps
    # sol = np.maximum(sol, 0)

    return nonnegtik_hnorm

def choose_error(pdf1, pdf2,err_type):
    """
    choose_error: Helps select error type based on input
    err_type(string): "Wassterstein" or "Wass. Score"
    vec1(np.array): first probability density function (PDF) you wish to compare to.
    vec2(np.array): second probability density funciton (PDF) you wish to compare to vec1.
    
    typically we select pdf1 to be the ground truth, and we select pdf2 to be our reconstruction method from regularization.
    ex. pdf1 = gt
        pdf2 = GCV solution (f_rec_GCV)
    """
    if err_type == "Wass. Score" or "Wassterstein":
        err = wass_error(pdf1, pdf2)
    else:
        err = l2_error(pdf1, pdf2)
    return err

def minimize_OP(Lambda, data_noisy, G, nT2, g):
    """
    Calculates the oracle lambda solution. Iterates over many lambdas in Alpha_vec
    """
    OP_x_lc_vec = np.zeros((nT2, len(Lambda)))
    OP_rhos = np.zeros((len(Lambda)))
    for j in (range(len(Lambda))):
        # try:Lambda
        #     # Fallback to nonnegtik_hnorm
        #     sol, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j], '0', nargin=4)
        #     if np.all(sol == 0):
        #         logging.debug("Solution from nonnegtik_hnorm is a zero vector, switching to CVXPY solver.")
        #         print(f"Solution from nonnegtik_hnorm is a zero vector, switching to CVXPY solver.")
        #         raise ValueError("Zero vector detected, switching to CVXPY.")
        # except Exception as e:
            #CVXPY solution
            # logging.error("Error in nonnegtik_hnorm, using CVXPY")
            # print(f"Error in nonnegtik_hnorm: {e}")
        # sol = cvxpy_tikhreg(Lambda[j], G, data_noisy)
        sol = nonnegtik_hnorm(G, data_noisy, Lambda[j], '0', nargin=4)[0]
        OP_x_lc_vec[:, j] = sol
        # Calculate the error (rho)
        OP_rhos[j] = choose_error(g, OP_x_lc_vec[:, j], err_type = "Wass. Score")
    #Find the minimum value of errors, its index, and the corresponding lambda and reconstruction
    min_rhos = min(OP_rhos)
    min_index = np.argmin(OP_rhos)
    min_x = Lambda[min_index][0]
    OP_min_alpha1 = min_x
    OP_min_alpha1_ind = min_index
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    return f_rec_OP_grid, OP_min_alpha1, min_rhos , min_index

def compute_MWF(f_rec, T2, Myelin_idx):
    """
    Normalize f_rec and compute MWF.
    Inputs: 
    1. f_rec (np.array type): the reconstruction generated from Tikhonov regularization 
    2. T2 (np.array): the transverse relaxation time constant vector in ms
    3. Myelin_idx (np.array):the indices of T2 where myelin is present (e.g. T2 < 40 ms)
    
    Outputs:
    1. MWF (float): the myelin water fraction
    2. f_rec_normalized (np.array type): the normalized reconstruction generated from Tikhonov regularization. 
        Normalized to 1.
    
    Test Example:
    1) f_rec = np.array([1,2,3,4,5,6])
    2) np.trapz(f_rec_normalized, T2) ~ 1.
    """
    f_rec_normalized = f_rec / np.trapz(f_rec, T2)
    total_MWF = np.cumsum(f_rec_normalized)
    MWF = total_MWF[Myelin_idx[-1][-1]]
    return f_rec_normalized, MWF

# passing_x_coord = 165
# passing_y_coord = 168

curr_data = brain_data[passing_x_coord,passing_y_coord,:]
# curr_data = brain_data
#normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
dTE = 11.3
n = 32
TE = dTE * np.linspace(1,n,n)
m = 150
T2 = np.linspace(10,200,m)
A= np.zeros((n,m))
dT = T2[1] - T2[0]
for i in range(n):
    for j in range(m):
        A[i,j] = np.exp(-TE[i]/T2[j]) * dT

Myelin_idx = np.where(T2<=40)
print("curr_data", curr_data)
sol1 = nnls(A, curr_data)[0]
print("sol1", sol1)
# factor = np.trapz(sol1,T2)
factor = np.sum(sol1) * dT

curr_data = curr_data/factor
if np.isnan(np.any(curr_data)):
    print("NA values")
    idx = np.where(curr_data == np.isnan(curr_data))
    print("idx",idx)

# #LS:

f_rec_LS = nnls(A, curr_data)[0]
f_rec_LS, MWF_LS = compute_MWF(f_rec_LS, T2, Myelin_idx)
# curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
#LC
f_rec_LC, lambda_LC = Lcurve(curr_data, A, Lambda)
f_rec_LC, MWF_LC = compute_MWF(f_rec_LC, T2, Myelin_idx)
# print("MWF_LC", MWF_LC)

# #GCV
f_rec_GCV, lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
f_rec_GCV = f_rec_GCV[:, 0]
lambda_GCV = np.squeeze(lambda_GCV)
f_rec_GCV, MWF_GCV = compute_MWF(f_rec_GCV, T2, Myelin_idx)
# print("MWF_GCV", MWF_GCV)

# # #Oracle or GCV
# IdealModel_weighted = f_rec_GCV
# f_rec_OR, lambda_OR, min_rhos , min_index = minimize_OP(Lambda, curr_data, A, len(T2), IdealModel_weighted)
# f_rec_OR, MWF_OR = compute_MWF(f_rec_OR, T2, Myelin_idx)
# print("MWF_OR", MWF_OR)

#DP
# f_rec_DP, lambda_DP = discrep_L2(curr_data, A, curr_SNR, Lambda, noise = "brain")
# f_rec_DP, lambda_DP = discrep_L2_sp(curr_data, A, curr_SNR, Lambda, noise = "brain")
f_rec_DP, lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
f_rec_DP, MWF_DP = compute_MWF(f_rec_DP, T2, Myelin_idx)
# print("MWF_DP", MWF_DP)

#LR
LRIto_ini_lam = lambda_LC
f_rec_ini = f_rec_LC
f_rec_LR, lambda_LR, test_frec1, test_lam1, numiterate = oLocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter)
f_rec_LR, MWF_LR = compute_MWF(f_rec_LR, T2, Myelin_idx)
print("MWF_LR original", MWF_LR)

# f_rec_LR, lambda_LR, test_frec1, test_lam1, numiterate = bLocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter)
# f_rec_LR, MWF_LR = compute_MWF(f_rec_LR, T2, Myelin_idx)


def plot_difference(slice1, slice2, slice1str, slice2str, filepath,xcoord=None,ycoord=None):
    """Plot and save the difference between two slices."""
    slice2cut = slice2[79:228, 103:215]
    slice1cut = slice1[79:228, 103:215]
    diff = slice2cut - slice1cut
    title = f"{slice2str} - {slice1str}"
    savepath = f"{filepath}/{slice2str}_minus_{slice1str}_brainfig"
    plt.figure()
    plt.title(title)
    plt.imshow(diff, cmap='viridis', vmin=0, vmax=0.2)
    plt.xlabel('X Index (103 to 215)')
    plt.ylabel('Y Index (79 to 228)')
    plt.xticks(ticks=range(0, slice1cut.shape[1], 25), labels=range(103, 216, 25))
    plt.yticks(ticks=range(0, slice1cut.shape[0], 25), labels=range(79, 229, 25))
    plt.axis('on')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    if xcoord is not None and ycoord is not None:
        title = f"{slice2str} - {slice1str} at Voxel X{xcoord} Y{ycoord}"
        plt.title(title)
        # Adjust for the zoom slice position (mapping the original coordinates to the zoomed coordinates)
        zoom_x = xcoord  - 103 # Adjusting xcoord for the zoom slice range
        zoom_y = ycoord  - 79 # Adjusting ycoord for the zoom slice range
        plt.scatter(zoom_x, zoom_y, color='red', s=20, label=f"MWF {slice1str} {round(slice1[xcoord,ycoord],4)} \n MWF {slice2str} {round(slice2[xcoord,ycoord],4)}")
        plt.legend()
        savepath = f"{filepath}/{slice2str}_minus_{slice1str}_xcoord{xcoord}_ycoord{ycoord}_brainfig_newLR"
    plt.savefig(savepath)

# filesavepath = "/home/kimjosy/LocReg_Regularization-1/data/debugfigures/brainpixelscript"
filesavepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/pixeldataresultstest"
MWF2_DP = np.zeros((313,313))
MWF2_LR = np.zeros((313,313))
MWF2_GCV = np.zeros((313,313))
MWF2_LC = np.zeros((313,313))
MWF2_OR = np.zeros((313,313))
MWF2_LS = np.zeros((313,313))

def load_MWF_values(df):
    """Load MWF values into respective arrays."""
    for i, row in df.iterrows():
        x = row['X_val']  # Adjust for 0-based index
        y = row['Y_val']  # Adjust for 0-based index
        
        # Assign the MWF values to the grid
        MWF2_DP[x, y] = row['MWF_DP']
        MWF2_LR[x, y] = row['MWF_LR']
        MWF2_LC[x, y] = row['MWF_LC']
        MWF2_GCV[x, y] = row['MWF_GCV']
        MWF2_OR[x, y] = row['MWF_OR']
        MWF2_LS[x, y] = row['MWF_LS']

df = load_MWF_values(df)
DPslice = MWF2_DP
LCslice = MWF2_LC
LRslice = MWF2_LR
LSslice = MWF2_LS
GCVslice = MWF2_GCV
print("max curr_data", np.max(curr_data))

def curve_plot(x_coord, y_coord, slice1, slice2, slice1str, slice2str):
    fig = plt.figure(figsize=(12.06, 4.2))
    # Plotting the first subplot
    # plt.subplot(1, 3, 1) 
    # plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
    # plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} (Error: {"{:.2e}".format(err_LR)})')
    # plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle (Error: {"{:.2e}".format(err_oracle)})')
    # plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {"{:.2e}".format(err_DP)})')
    # plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {"{:.2e}".format(err_GCV)})')
    # plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Error: {"{:.2e}".format(err_LC)})')
    # Modify the plot labels to include a star next to the method with the lowest error
    plt.subplot(1, 3, 1)
    plt.plot(T2, f_rec_LS, label = f"NNLS MWF:{MWF_LS:.3e}", color = 'black')
    plt.plot(T2, f_rec_LR, label = f"LocReg MWF:{MWF_LR:.3e}",  color = 'red')
    plt.plot(T2, f_rec_LC, label = f"LC MWF:{MWF_LC:.3e}",  color = 'purple')
    plt.plot(T2, f_rec_DP, label = f"DP MWF:{MWF_DP:.3e}",  color = 'green')
    plt.plot(T2, f_rec_GCV, label = f"GCV MWF:{MWF_GCV:.3e}",  color = 'blue')
    plt.legend(fontsize=10, loc='best')
    plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
    ymax = np.max(f_rec_LR) * 2
    plt.ylim(0, ymax)

    # Plotting the second subplot
    plt.subplot(1, 3, 2)
    plt.plot(TE, A @ f_rec_LS, linewidth=3, color='black', label='NNLS')
    plt.plot(TE, A @ f_rec_LR, linestyle=':', linewidth=3, color='red', label= f'LocReg {lam_ini_val}')
    plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
    plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
    plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
    plt.plot(TE, curr_data,linewidth=3,linestyle='solid',color='pink', label='Real Data')
    plt.legend(fontsize=10, loc='best')
    plt.xlabel('TE', fontsize=20, fontweight='bold')
    plt.ylabel('Intensity', fontsize=20, fontweight='bold')
    
    plt.subplot(1, 3, 3)
    plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
    plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
    plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
    plt.semilogy(T2, lambda_LR * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')

    plt.legend(fontsize=10, loc='best')
    plt.xlabel('T2', fontsize=20, fontweight='bold')
    plt.ylabel('Lambda', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.legend()
    fig.suptitle(f"Voxel: X {x_coord} and Y {y_coord} \n max real data {round(np.max(curr_data),2)}", fontsize=16)
    fig.subplots_adjust(top=0.85)  # Decrease the 'top' value to add space
    plt.savefig(f"{filesavepath}/compareLRDP_xcoord{x_coord}_ycoord{y_coord}_newLR.png")
    print(f"savefig xcoord{x_coord}_ycoord{y_coord}")
    plot_difference(slice1, slice2, slice1str, slice2str, filepath= filesavepath,xcoord=passing_x_coord,ycoord=passing_y_coord)

curve_plot(passing_x_coord, passing_y_coord, LRslice, DPslice, "LocReg", "DP")

curve_plot(passing_x_coord, passing_y_coord, LRslice, GCVslice, "LocReg", "GCV")

#1.) real brain data without tests; tune hyperparameters for LocReg find the pixels where NNLS fills in well, but Locreg doesnt...;
#best way is to See actual decay and fitted decay with diff. methods with actual noise decay to see underreg 
# (find out what techniques that can address it) or overreg(know the techniques that can address it); pick one or two pixels. normalization? rician?
#see the fit curve and see if its above or below the actual decay; plot lambda distributions with reconstructions;
#confrim whether the bad pixels normalized data is close to 1. if normalized data max value is 0.7 its a problem...; 
# hpyeratmeter: stopping criteria; figure it out.; check if 1st step of locreg is good or not; if two step method 
#       (usually relaxed method for 1st step (higher tol)) to leave more room for error in 2nd step (same tol);


#tell to do real brain data to get a better T2 distribution before fix as gt for test1 and test2


#2.) test1: stabilization across low and high SNR levels 180 and 400; uniform scenario; sensitivity and accuracy in diff. noise levels;
#arbitratily use locreg/gcv as gt.;#do unsupervised learning/naive bayes to find the two clusters of SNR_map;

#3.) test2 for last: SNR map from real data; acrooding to each pixel and then recovery; deviation; in real scenario; across number of SNR
# arbitratily use locreg/gcv as gt.; some SNR = 0; so we skip it where we assign MWF=0; 

