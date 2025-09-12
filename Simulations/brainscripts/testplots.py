import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import scipy
from datetime import date
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')

# Define file paths
# filtered_path = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_simexp117Jan25.pkl"
filtered_path = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/filteredbrain/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps29Jan25.pkl"
unfiltered_path = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_unfiltered_noNESMA21Jan25.pkl"
comparison_folder = f"/home/kimjosy/LocReg_Regularization-1/data/Brain/comparison_filtered_unfiltered_NESMA_{day}{month}{year}"

# addednoise = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_25Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_filtered_addnoise_SNR30025Jan25.pkl"
# nonoise = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_simexp117Jan25.pkl"
# comparison_folder = "/home/kimjosy/LocReg_Regularization-1/data/Brain/comparison_addednoiseSNR300_nonoise_NESMA"
# unfiltered_path = nonoise
# filtered_path = addednoise
# Ensure the comparison folder exists
os.makedirs(comparison_folder, exist_ok=True)

# Load data
with open(unfiltered_path, 'rb') as f:
    unfiltered_data = pickle.load(f)

with open(filtered_path, 'rb') as f:
    filtered_data = pickle.load(f)


# Assuming the data is a pandas DataFrame
column_names = unfiltered_data.columns.tolist()
print(column_names)

dTE = 11.3
n = 32
TE = dTE * np.linspace(1, n, n)
m = 150
T2 = np.linspace(10, 200, m)
A = np.zeros((n, m))
dT = T2[1] - T2[0]

def curve_plot(method, x_coord, y_coord, frec, curr_data, lambda_vals, curr_SNR, MWF, filepath):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # First plot: T2 vs f_rec
    axs[0].plot(T2, frec)
    axs[0].set_title('T2 vs f_rec')
    axs[0].set_xlabel('T2')
    axs[0].set_ylabel('f_rec')

    # Second plot: TE vs curr_data
    axs[1].plot(TE, curr_data)
    axs[1].set_title('TE vs Decay Data')
    axs[1].set_xlabel('TE')
    axs[1].set_ylabel('curr_data')

    # Third plot: T2 vs lambda
    axs[2].plot(T2, lambda_vals * np.ones(len(T2)))
    axs[2].set_title('T2 vs Lambda')
    axs[2].set_xlabel('T2')
    axs[2].set_ylabel('lambda')
    # Set the main title with curr_SNR and MWF value
    fig.suptitle(f'{method} Plots for x={x_coord}, y={y_coord} | SNR={curr_SNR}, MWF={MWF}', fontsize=16)
    # Save the figure
    plt.savefig(f"{filepath}/{method}_recon_xcoord{x_coord}_ycoord{y_coord}.png")
    print(f"savefig xcoord{x_coord}_ycoord{y_coord}")
    plt.close('all')
    return 

# curve_plot(method, x_coord, y_coord, frec, curr_data, lambda_vals, curr_SNR, MWF, filepath)


# filtered_T2 = filtered_data["GCV_estimate"][12225]
# plt.figure()
# <Figure size 640x480 with 0 Axes>
# plt.plot(unfiltered_T2)
# [<matplotlib.lines.Line2D object at 0x7f36cf9f82d0>]
# plt.plot(filtered_T2)
# [<matplotlib.lines.Line2D object at 0x7f36d3ad0690>]
# plt.legend(["unfiltered", "filtered"])

# Define methods to compare
methods = ['DP_estimate', 'LC_estimate', 'LR_estimate', 'GCV_estimate', 'LS_estimate']

# Initialize dictionaries to store the scores
wasserstein_scores = {}
SAD_scores = {}
L2_scores = {}
pixel_wasserstein_scores = {method: [] for method in methods}
pixel_SAD_scores = {method: [] for method in methods}
pixel_L2_scores = {method: [] for method in methods}

from scipy.ndimage import rotate
mask_filepath="/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat"
# mask_filepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/masks/mask_2.mat"
BW = scipy.io.loadmat(mask_filepath)["new_BW"]
# BW = scipy.io.loadmat(mask_filepath)["BW"]

wasserstein_error_maps = {method: np.zeros((313, 313)) for method in methods}
SAD_maps = {method: np.zeros((313, 313)) for method in methods}
L2_maps = {method: np.zeros((313, 313)) for method in methods}

def rotate_images(map):
    """Rotate the MWF images."""
    return rotate(map, 275, reshape=False)

# Loop through each pixel and calculate the scores for each method

# Merge the two DataFrames on X_val and Y_val to get the matching rows
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
        # pixel_wasserstein_scores[method].append((x_val, y_val, wassscore))
        # pixel_SAD_scores[method].append((x_val, y_val, SADscore))
        # pixel_L2_scores[method].append((x_val, y_val, L2score))
        # # curve_plot(method, x_val, y_val, filtered_estimate, curr_data, lambda_vals, curr_SNR, MWF, filepath)

        merged_data.at[index, f"{method}_wassscore"] = wassscore
        merged_data.at[index, f"{method}_SADscore"] = SADscore
        merged_data.at[index, f"{method}_L2score"] = L2score
        wasserstein_error_maps[method][x_val, y_val] = wassscore
        SAD_maps[method][x_val, y_val] = SADscore
        L2_maps[method][x_val, y_val] = L2score

#     # Append the calculated scores to the lists
#     wassscores.append(wassscore)
#     SADscores.append(SADscore)
print("merged_data", merged_data)
print("checkpoint")
#     L2scores.append(L2score)


# # Add the new columns to the merged DataFrame
# merged_data['Wasserstein_Score'] = wassscores
# merged_data['SAD_Score'] = SADscores
# merged_data['L2_Score'] = L2scores

# Inspect the result
# print(merged_data[['X_val', 'Y_val', 'Wasserstein_Score', 'SAD_Score', 'L2_Score']])



# for index, row in unfiltered_data.iterrows():
#     x_val = row['X_val']
#     y_val = row['Y_val']
#     unfilt_curr_data = row["Normalized Data"]
#     for method in methods:
#         unfiltered_estimate = unfiltered_data.loc[index, method]
#         filtered_estimate = filtered_data.loc[index, method]

#         score = wasserstein_distance(filtered_estimate, unfiltered_estimate)
#         SADscore = np.sum(np.abs(unfiltered_estimate - filtered_estimate)) / np.sum(unfiltered_estimate)
#         L2score = np.linalg.norm(unfiltered_estimate - filtered_estimate) / np.linalg.norm(unfiltered_estimate)

#         pixel_wasserstein_scores[method].append((x_val, y_val, score))
#         pixel_SAD_scores[method].append((x_val, y_val, SADscore))
#         pixel_L2_scores[method].append((x_val, y_val, L2score))
#         curve_plot(method, x_val, y_val, filtered_estimate, curr_data, lambda_vals, curr_SNR, MWF, filepath)



# Create brain maps for each method


# Populate the error maps
# for method in methods:
#     for x_val, y_val, wassscore in pixel_wasserstein_scores[method]:
#         wasserstein_error_maps[method][x_val, y_val] = wassscore

#     for x_val, y_val, SADscore in pixel_SAD_scores[method]:
#         SAD_maps[method][x_val, y_val] = SADscore

#     for x_val, y_val, L2score in pixel_L2_scores[method]:
#         L2_maps[method][x_val, y_val] = L2score

# Define file paths for saving plots
wass_file_path = f"{comparison_folder}/large_Wasserror"
SAD_file_path = f"{comparison_folder}/SAD"
L2_file_path = f"{comparison_folder}/L2norm"

# Ensure the directories exist
# os.makedirs(wass_file_path, exist_ok=True)
# os.makedirs(SAD_file_path, exist_ok=True)
# os.makedirs(L2_file_path, exist_ok=True)

# plt.figure()
# plt.imshow(BW ,cmap='viridis', vmin=0, vmax=0.008, interpolation='nearest')
# plt.savefig(f'{wass_file_path}/{method}_BW_map.png')

# Apply the rotate_images function to all maps
# wasserstein_error_maps = {method: map for method, map in wasserstein_error_maps.items()}
# SAD_maps = {method: map for method, map in SAD_maps.items()}
# L2_maps = {method: map for method, map in L2_maps.items()}

# Apply the mask to all maps
# wasserstein_error_maps = {method: BW * map  for method, map in wasserstein_error_maps.items()}
# SAD_maps = {method: BW * map for method, map in SAD_maps.items()}
# L2_maps = {method: BW * map  for method, map in L2_maps.items()}

# wasserstein_error_maps = {method: np.where(BW == 1, map, 0)   for method, map in wasserstein_error_maps.items()}
# SAD_maps = {method: np.where(BW == 1, map, 0) for method, map in SAD_maps.items()}
# L2_maps = {method: np.where(BW == 1, map, 0)  for method, map in L2_maps.items()}


# Define file paths for saving plots
wass_file_path = f"{comparison_folder}/large_Wasserror"
SAD_file_path = f"{comparison_folder}/SAD"
L2_file_path = f"{comparison_folder}/L2norm"

# Ensure the directories exist
os.makedirs(wass_file_path, exist_ok=True)
os.makedirs(SAD_file_path, exist_ok=True)
os.makedirs(L2_file_path, exist_ok=True)

title1 = f'Wasserstein Error Map \n Comparing NESMA-filtered {method} T2 estimates \n and Unfiltered {method} T2 estimates'
title2 = f'Wasserstein Error Map \n Comparing NESMA-filtered {method} without noise \n and added noise {method} SNR 300 T2 estimates'
# Plot and save the Wasserstein error maps
for method, error_map in wasserstein_error_maps.items():
    plt.figure()
    # error_map = BW @ rotate(error_map, 275, reshape = False)
    error_map = BW * error_map
    error_map = error_map[79:228, 103:215]
    plt.title(f'Wasserstein Error Map \n Comparing NESMA-filtered {method} T2 estimates \n and Unfiltered {method} T2 estimates')
    plt.imshow(error_map, cmap='viridis', vmin=0, vmax=0.01, interpolation='nearest')
    plt.colorbar(label='Wasserstein Distance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(f'{wass_file_path}/{method}_wasserstein_error_map.png')
    plt.close()

# # Plot and save the SAD maps
# title1= f'Sum of Absolute Differences (L1) \n Comparing NESMA-filtered {method} T2 estimates \n and Unfiltered {method} T2 estimates'
# title1= f'Sum of Absolute Differences (L1) \n Comparing NESMA-filtered {method} T2 estimates \n and Unfiltered {method} T2 estimates'

for method, error_map in SAD_maps.items():
    plt.figure()
    error_map = BW * error_map
    # error_map = BW @ error_map
    error_map = error_map[79:228, 103:215]
    plt.title(f'Sum of Absolute Differences (L1) \n Comparing NESMA-filtered {method} T2 estimates \n and Unfiltered {method} T2 estimates')
    plt.imshow(error_map, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(label='SAD')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(f'{SAD_file_path}/{method}_SADmap.png')
    plt.close()

# Plot and save the L2 norm maps
for method, error_map in L2_maps.items():
    plt.figure()
    error_map = BW * error_map
    # error_map = BW @ error_map
    error_map = error_map[79:228, 103:215]
    plt.title(f'Relative L2 Norm Error Map Comparing\nNESMA-filtered {method} T2 estimates \n and Unfiltered {method} T2 estimates')    
    plt.imshow(error_map,cmap='viridis', vmin=0, vmax=3, interpolation='nearest')
    plt.colorbar(label='Relative L2 Norm Error')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(f'{L2_file_path}/{method}_L2norm_map.png')
    plt.close()


for method in methods:
    wasserstein_error_maps[method] *= BW
    SAD_maps[method] *= BW
    L2_maps[method] *= BW

brain_wass = {method: wasserstein_error_maps[method][BW == 1] for method in methods}
brain_SAD = {method: SAD_maps[method][BW == 1] for method in methods}
brain_L2 = {method: L2_maps[method][BW == 1] for method in methods}

summary_statistics = {}
for method in methods:
    summary_statistics[method] = {
        'wasserstein': {
            'mean': np.nanmean(brain_wass[method]),
            'median': np.nanmedian(brain_wass[method]),
            'std': np.nanstd(brain_wass[method]),
            'min': np.nanmin(brain_wass[method]),
            'max': np.nanmax(brain_wass[method])
        },
        'SAD': {
            'mean': np.nanmean(brain_SAD[method]),
            'median': np.nanmedian(brain_SAD[method]),
            'std': np.nanstd(brain_SAD[method]),
            'min': np.nanmin(brain_SAD[method]),
            'max': np.nanmax(brain_SAD[method])
        },
        'L2': {
            'mean': np.nanmean(brain_L2[method]),
            'median': np.nanmedian(brain_L2[method]),
            'std': np.nanstd(brain_L2[method]),
            'min': np.nanmin(brain_L2[method]),
            'max': np.nanmax(brain_L2[method])
        }
    }
print("summary_statistics", summary_statistics)

# Rank the methods based on the median of Wasserstein distance (lowest is best)
ranked_methods = sorted(methods, key=lambda m: summary_statistics[m]['wasserstein']['median'])

# Save the summary statistics and ranking to a pickle file
summary_statistics_file = os.path.join(comparison_folder, 'summary_statistics.pkl')
with open(summary_statistics_file, 'wb') as f:
    pickle.dump({'summary_statistics': summary_statistics, 'ranking': ranked_methods}, f)

# Print the ranking
print("Ranking of methods based on median Wasserstein distance (lowest is best):")
for rank, method in enumerate(ranked_methods, start=1):
    print(f"{rank}. {method}")

# summary_statistics = {}
# for method in methods:
#     brain_wass = wasserstein_error_maps[method][BW == 1]
#     brain_SAD = SAD_maps[method][BW == 1]
#     brain_L2 = L2_maps[method][BW == 1]
    
#     summary_statistics[method] = {
#         'wasserstein': {
#             'mean': np.nanmean(brain_wass),
#             'median': np.nanmedian(brain_wass),
#             'std': np.nanstd(brain_wass),
#             'min': np.nanmin(brain_wass),
#             'max': np.nanmax(brain_wass)
#         }
#         ,
#         'SAD': {
#             'mean': np.nanmean(brain_SAD),
#             'median': np.nanmedian(brain_SAD),
#             'std': np.nanstd(brain_SAD),
#             'min': np.nanmin(brain_SAD),
#             'max': np.nanmax(brain_SAD)
#         },
#         'L2': {
#             'mean': np.nanmean(brain_L2),
#             'median': np.nanmedian(brain_L2),
#             'std': np.nanstd(brain_L2),
#             'min': np.nanmin(brain_L2),
#             'max': np.nanmax(brain_L2)
#         }
#     }
# print("summary_statistics", summary_statistics)


# # Rank the methods based on the median of Wasserstein distance (lowest is best)
# ranked_methods = sorted(methods, key=lambda m: summary_statistics[m]['wasserstein']['median'])

# # Save the summary statistics and ranking to a pickle file
# summary_statistics_file = os.path.join(comparison_folder, 'summary_statistics.pkl')
# with open(summary_statistics_file, 'wb') as f:
#     pickle.dump({'summary_statistics': summary_statistics, 'ranking': ranked_methods}, f)

# # Print the ranking
# print("Ranking of methods based on median Wasserstein distance (lowest is best):")
# for rank, method in enumerate(ranked_methods, start=1):
#     print(f"{rank}. {method}")
# # for rank, method in enumerate(ranked_methods2, start=1):
# #     print(f"{rank}. {method}")

# # for rank, method in enumerate(ranked_methods3, start=1):
# #     print(f"{rank}. {method}")

# print("Summary statistics and ranking saved to", summary_statistics_file)
# #unfiltered nad filtered L2; for each method (DP...etc) [supplementatl]
# #unfiltered nad filtered L1; for each method (DP...etc) [supplemental]

# #2) #ground truth GCV filtered...use filtred GCV for both for all ground truth; 
#     #computet he difference between results and filtered and unfiltered data against g.t.

# #3) SNR: 1000, 300 add gaussian noise to NESMA filtered data (makes it less effect of rician noise?) 
# #  bc rician amplifies the noise for unfiltered.

# # 4) SpanReg
#     # original brain data (filtered NESMA)
#     # collect T2 from spanreg
#     # then A * F(T2) = d, then add noise with uniform SNR
#     # recovery using that noisy data 
#     # (using DP, GCV, LocReg, LCurve, non-regular NNLS)
#     # 200, 800 SNR levels
