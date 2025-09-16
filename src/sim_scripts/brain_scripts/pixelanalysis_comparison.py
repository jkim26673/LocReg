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
from scipy.ndimage import rotate
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat")["final_data_2"]
maskfilepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat"
# estimatesfilepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results/est_table_xcoordlen_313_ycoordlen_31328Nov24.pkl"
# estimatesfilepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/results/est_table_xcoordlen_313_ycoordlen_313_withLS_OR30Nov24.pkl"
# estimatesfilepath =  "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_noiseadd_recovery/est_table_xcoordlen_313_ycoordlen_313_withLS_OR10Dec24.pkl"
BW = scipy.io.loadmat(maskfilepath)["new_BW"]
# filtered_estimatesfilepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_simexp117Jan25.pkl"
filtered_estimatesfilepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/filteredbrain/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps29Jan25.pkl"
unfiltered_estimatesfilepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_unfiltered_noNESMA21Jan25.pkl"
pixelanalysisfilepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/comparison_filtered_unfiltered_NESMA_30Jan25/pixelanalysis"


with open (filtered_estimatesfilepath, "rb") as file:
    filtered_data = pickle.load(file)

with open (unfiltered_estimatesfilepath, "rb") as file:
    unfiltered_data = pickle.load(file)

dTE = 11.3
n = 32
TE = dTE * np.linspace(1, n, n)
m = 150
T2 = np.linspace(10, 200, m)
A = np.zeros((n, m))
dT = T2[1] - T2[0]

for i in range(n):
    for j in range(m):
        A[i,j] = np.exp(-TE[i]/T2[j]) * dT

passing_x_coord = 128
passing_y_coord = 90

tag = f"x_{passing_x_coord}_y_{passing_y_coord}"
pixelanalysisfilepath = f"{pixelanalysisfilepath}/{tag}"
if not os.path.exists(pixelanalysisfilepath):
    os.makedirs(pixelanalysisfilepath)

methods = ['DP_estimate', 'LC_estimate', 'LR_estimate', 'GCV_estimate', 'LS_estimate']

# Initialize dictionaries to store the scores
wasserstein_scores = {}
SAD_scores = {}
L2_scores = {}
pixel_wasserstein_scores = {method: [] for method in methods}
pixel_SAD_scores = {method: [] for method in methods}
pixel_L2_scores = {method: [] for method in methods}


wasserstein_error_maps = {method: np.zeros((313, 313)) for method in methods}
SAD_maps = {method: np.zeros((313, 313)) for method in methods}
L2_maps = {method: np.zeros((313, 313)) for method in methods}
merged_data = pd.merge(unfiltered_data, filtered_data, on=['X_val', 'Y_val'], suffixes=('_unfiltered', '_filtered'), how='outer')
merged_data.fillna(0, inplace=True)

# Initialize empty lists to store the scores
# wassscores = []
# SADscores = []
# L2scores = []

methods = ['DP_estimate', 'LC_estimate', 'LR_estimate', 'GCV_estimate', 'LS_estimate']
lam_methods = ['DP', 'LC', 'LR', 'GCV']

# Iterate over each row of the merged DataFrame and calculate the scores
for index, row in merged_data.iterrows():
    # Extract the DP_estimates from both DataFrames (or any other columns you want to compare)
    x_val = row['X_val']
    y_val = row['Y_val']
    unfilt_curr_data = row["Normalized Data_unfiltered"]
    filt_curr_data = row["Normalized Data_filtered"]

    if isinstance(unfilt_curr_data, int):
        merged_data.at[index, "Normalized Data_unfiltered"] = np.zeros(n)
        merged_data.at[index, "Normalized Data_unfiltered"] = merged_data.at[index, "Normalized Data_unfiltered"].flatten()
        unfilt_curr_data = np.zeros(n)
    elif isinstance(filt_curr_data, int):
        merged_data.at[index, "Normalized Data_filtered"] = np.zeros(n)
        merged_data.at[index, "Normalized Data_filtered"] = merged_data.at[index, "Normalized Data_filtered"].flatten()
        filt_curr_data = np.zeros(n)
    else:
        pass

    for method in lam_methods:
        lam_unfiltered_method = f"Lam_{method}_unfiltered"
        lam_filtered_method = f"Lam_{method}_filtered"
        lam_unfiltered_estimate = row[lam_unfiltered_method]
        lam_filtered_estimate = row[lam_filtered_method]
        if isinstance(lam_unfiltered_estimate, int):
            merged_data.at[index, lam_unfiltered_method] = np.zeros(len(T2))
            merged_data.at[index, lam_unfiltered_method] = merged_data.at[index, lam_unfiltered_method].flatten()
            lam_unfiltered_estimate = np.zeros(len(T2))
        elif isinstance(lam_filtered_estimate, int):
            merged_data.at[index, lam_filtered_method] = np.zeros(len(T2))
            merged_data.at[index, lam_filtered_method] = merged_data.at[index, lam_filtered_method].flatten()
            lam_filtered_estimate = np.zeros(len(T2))
        else:
            pass
        
    for method in methods:
        # unfiltered_estimate = unfiltered_data.loc[index, method]
        # filtered_estimate = filtered_data.loc[index, method]
        unfiltered_method = f"{method}_unfiltered"
        filtered_method = f"{method}_filtered"

        # Extract the corresponding estimates
        unfiltered_estimate = row[unfiltered_method]
        filtered_estimate = row[filtered_method]


        if isinstance(unfiltered_estimate, int):
            merged_data.at[index, unfiltered_method] = np.zeros(len(T2))
            merged_data.at[index, unfiltered_method] = merged_data.at[index, unfiltered_method].flatten()
            unfiltered_estimate = np.zeros(len(T2))
        elif isinstance(filtered_estimate, int):
            merged_data.at[index, filtered_method] = np.zeros(len(T2))
            merged_data.at[index, filtered_method] = merged_data.at[index, filtered_method].flatten()
            filtered_estimate = np.zeros(len(T2))
        else:
            pass

        wassscore = wasserstein_distance(filtered_estimate, unfiltered_estimate)
        # Calculate the Sum of Absolute Differences (SAD)
        SADscore = np.sum(np.abs(unfiltered_estimate - filtered_estimate)) / np.sum(unfiltered_estimate)
        # Calculate the L2 score (Euclidean distance normalized)
        L2score = np.linalg.norm(unfiltered_estimate - filtered_estimate) / np.linalg.norm(unfiltered_estimate)
        pixel_wasserstein_scores[method].append((x_val, y_val, wassscore))
        pixel_SAD_scores[method].append((x_val, y_val, SADscore))
        pixel_L2_scores[method].append((x_val, y_val, L2score))
        # curve_plot(method, x_val, y_val, filtered_estimate, curr_data, lambda_vals, curr_SNR, MWF, filepath)

#unique px
# px = df[(df["X_val"] == 118) & (df["Y_val"] == 117)]
# px["MWF_DP"]
# px["MWF_LR"]

# filesavepath = "/home/kimjosy/LocReg_Regularization-1/data/debugfigures/brainpixelscript"
def curve_plot(x_val,y_val, merged_data, wasserstein_error_maps, savepath, filtered = True):
    px = merged_data[(merged_data["X_val"] == x_val) & (merged_data["Y_val"] == y_val)]
    if filtered == True:
        f_rec_LR = px["LR_estimate_filtered"].iloc[0].flatten()
        f_rec_LC = px["LC_estimate_filtered"].iloc[0].flatten()
        f_rec_GCV = px["GCV_estimate_filtered"].iloc[0].flatten()
        f_rec_DP = px["DP_estimate_filtered"].iloc[0].flatten()
        f_rec_LS = px["LS_estimate_filtered"].iloc[0].flatten()
        MWF_LR = px["MWF_LR_filtered"].iloc[0].flatten()
        MWF_LC = px["MWF_LC_filtered"].iloc[0].flatten()
        MWF_GCV = px["MWF_GCV_filtered"].iloc[0].flatten()
        MWF_DP = px["MWF_DP_filtered"].iloc[0].flatten()
        curr_data = px["Normalized Data_filtered"].iloc[0].flatten()
        lambda_DP = px["Lam_DP_filtered"].iloc[0].flatten()
        lambda_GCV = px["Lam_GCV_filtered"].iloc[0].flatten()
        lambda_LC = px["Lam_LC_filtered"].iloc[0].flatten()
        lambda_LR = px["Lam_LR_filtered"].iloc[0].flatten()
    else:
        f_rec_LR = px["LR_estimate_unfiltered"].iloc[0].flatten()
        f_rec_LC = px["LC_estimate_unfiltered"].iloc[0].flatten()
        f_rec_GCV = px["GCV_estimate_unfiltered"].iloc[0].flatten()
        f_rec_DP = px["DP_estimate_unfiltered"].iloc[0].flatten()
        f_rec_LS = px["LS_estimate_unfiltered"].iloc[0].flatten()
        MWF_LR = px["MWF_LR_unfiltered"].iloc[0].flatten()
        MWF_LC = px["MWF_LC_unfiltered"].iloc[0].flatten()
        MWF_GCV = px["MWF_GCV_unfiltered"].iloc[0].flatten()
        MWF_DP = px["MWF_DP_unfiltered"].iloc[0].flatten()
        curr_data = px["Normalized Data_unfiltered"].iloc[0].flatten()
        lambda_DP = px["Lam_DP_unfiltered"].iloc[0].flatten()
        lambda_GCV = px["Lam_GCV_unfiltered"].iloc[0].flatten()
        lambda_LC = px["Lam_LC_unfiltered"].iloc[0].flatten()
        lambda_LR = px["Lam_LR_unfiltered"].iloc[0].flatten()

    DP_wass = wasserstein_error_maps["DP_estimate"][x_val, y_val]
    GCV_wass = wasserstein_error_maps["GCV_estimate"][x_val, y_val]
    LC_wass = wasserstein_error_maps["LC_estimate"][x_val, y_val]
    LR_wass = wasserstein_error_maps["LR_estimate"][x_val, y_val]
    LS_wass = wasserstein_error_maps["LS_estimate"][x_val, y_val]

    fig = plt.figure(figsize=(12.06, 4.2))

    # Check if all values are zero
    if np.all(f_rec_LR == 0) and np.all(f_rec_LC == 0) and np.all(f_rec_GCV == 0) and np.all(f_rec_DP == 0) and np.all(f_rec_LS == 0):
        LR_wass = 0
        LS_wass = 0
        LC_wass = 0
        GCV_wass = 0
        DP_wass = 0
        fig.suptitle(f"Voxel: X {x_val} and Y {y_val} \n No pixel exists", fontsize=16)
    else:
        # Your existing plotting code here
        fig.suptitle(f"Voxel: X {x_val} and Y {y_val} \n max real data {round(np.max(curr_data),2)}", fontsize=16)
        pass


    plt.subplot(1, 3, 1)
    plt.plot(T2, f_rec_LR, label = f"LocReg Wass:{LR_wass:.3e}",  color = 'red')
    plt.plot(T2, f_rec_LC, label = f"LC Wass:{LC_wass:.3e}",  color = 'purple')
    plt.plot(T2, f_rec_DP, label = f"DP Wass:{DP_wass:.3e}",  color = 'green')
    plt.plot(T2, f_rec_GCV, label = f"GCV Wass:{GCV_wass:.3e}",  color = 'blue')
    plt.plot(T2, f_rec_LS, label = f"NNLS Wass:{LS_wass:.3e}",  color = 'black')
    plt.legend(fontsize=10, loc='best')
    plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
    x_ticks = np.linspace(min(T2), max(T2), num=5)
    x_ticks = np.append(x_ticks, 40)  # Ensure T2 = 40 is included in the x-ticks
    plt.xticks(x_ticks)
    ymax = np.max(f_rec_LR) * 2
    plt.ylim(0, ymax)

    # Plotting the second subplot
    plt.subplot(1, 3, 2)
    plt.plot(TE, A @ f_rec_LR, linestyle=':', linewidth=3, color='red', label= f'LocReg')
    plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
    plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
    plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
    plt.plot(TE, A @ f_rec_LS, linestyle='-.', linewidth=3, color='black', label='NNLS')
    plt.plot(TE, curr_data,linewidth=3,linestyle='solid',color='pink', label='Real Data')
    plt.legend(fontsize=10, loc='best')
    plt.xlabel('TE', fontsize=20, fontweight='bold')
    plt.ylabel('Intensity', fontsize=20, fontweight='bold')
    
    plt.subplot(1, 3, 3)
    plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
    plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
    plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
    plt.semilogy(T2, lambda_LR * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg')

    plt.legend(fontsize=10, loc='best')
    plt.xlabel('T2', fontsize=20, fontweight='bold')
    x_ticks = np.linspace(min(T2), max(T2), num=5)
    x_ticks = np.append(x_ticks, 40)  # Ensure T2 = 40 is included in the x-ticks
    plt.xticks(x_ticks)
    plt.ylabel('Lambda', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.legend()
    fig.subplots_adjust(top=0.85)  # Decrease the 'top' value to add space
    plt.savefig(f"{savepath}/px_xcoord{x_val}_ycoord{y_val}_filtered_{filtered}.png")
    print(f"savefig xcoord{x_val}_ycoord{y_val}")
    return

def plot_coord(mask, wasserstein_error_maps, title, savepath,xcoord=None,ycoord=None):
    """Helper method to plot and save a figure."""
    for method, error_map in wasserstein_error_maps.items():
        plt.figure()
        error_map = mask * error_map
        zoom_slice = error_map[79:228, 103:215]
        plt.title(title)
        # plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
        plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax=0.01)
        plt.xlabel('X Index (103 to 215)')
        plt.ylabel('Y Index (79 to 228)')
        plt.xticks(ticks=range(0, zoom_slice.shape[1], 25), labels=range(103, 216, 25))
        plt.yticks(ticks=range(0, zoom_slice.shape[0], 25), labels=range(79, 229, 25))
        plt.axis('on')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
        if xcoord is not None and ycoord is not None:
            plt.title(title)
            # Adjust for the zoom slice position (mapping the original coordinates to the zoomed coordinates)
            zoom_x = xcoord - 103# Adjusting xcoord for the zoom slice range
            zoom_y = ycoord - 79 # Adjusting ycoord for the zoom slice range
            plt.scatter(zoom_x, zoom_y, color='red', s=10, label="Target (Red Dot)")
        plt.savefig(f"{savepath}/Wass.Score_Err_Map_{method}_{title}_x_{xcoord}_y_{ycoord}.png")
    return 

for method in methods:
    for x_val, y_val, wassscore in pixel_wasserstein_scores[method]:
        wasserstein_error_maps[method][x_val, y_val] = wassscore

    for x_val, y_val, SADscore in pixel_SAD_scores[method]:
        SAD_maps[method][x_val, y_val] = SADscore

    for x_val, y_val, L2score in pixel_L2_scores[method]:
        L2_maps[method][x_val, y_val] = L2score

curve_plot(passing_x_coord, passing_y_coord, merged_data, wasserstein_error_maps, savepath = f"{pixelanalysisfilepath}", filtered = True)
curve_plot(passing_x_coord, passing_y_coord, merged_data, wasserstein_error_maps, savepath = f"{pixelanalysisfilepath}",  filtered = False)
plot_coord(BW, wasserstein_error_maps, "Wass. Score Error Map", savepath = f"{pixelanalysisfilepath}",xcoord=passing_x_coord,ycoord=passing_y_coord)
print("done")


