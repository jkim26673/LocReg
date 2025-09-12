import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from datetime import date
import scipy
# Define file paths (same as original)
unfiltered_path = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_unfiltered_noNESMA21Jan25.pkl"
# filtered_path = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/filteredbrain/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps29Jan25.pkl"
filtered_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14Apr25\est_table_xcoordlen_313_ycoordlen_313_filtered_noise_addition_uniform_noise_UPEN_LR1D2D14Apr25.pkl"
comparison_folder = "/home/kimjosy/LocReg_Regularization-1/data/Brain/comparison_filtered_unfiltered_NESMA_30Jan25"
mask_filepath="/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat"
BW = scipy.io.loadmat(mask_filepath)["new_BW"]

# Ensure the comparison folder exists
os.makedirs(comparison_folder, exist_ok=True)

# Load data
with open(unfiltered_path, 'rb') as f:
    unfiltered_data = pickle.load(f)

with open(filtered_path, 'rb') as f:
    filtered_data = pickle.load(f)

# Merge unfiltered and filtered data on X_val and Y_val
merged_data = pd.merge(unfiltered_data, filtered_data, on=['X_val', 'Y_val'], suffixes=('_unfiltered', '_filtered'))
merged_data.fillna(0, inplace=True)

# Define methods to compare
methods = ['DP_estimate', 'LC_estimate', 'LR_estimate', 'GCV_estimate', 'LS_estimate']

# Initialize dictionary to store scores
scores = {method: {'wasserstein': [], 'SAD': [], 'L2': []} for method in methods}
merged_data['BW'] = 0

# Loop through merged data and calculate error metrics
for index, row in merged_data.iterrows():
    x_val, y_val = row['X_val'], row['Y_val']
    merged_data.at[index, 'BW'] = BW[x_val, y_val]  # Note the order: BW[y_val, x_val]
    for method in methods:
        unfiltered_estimate = row[f"{method}_unfiltered"]
        filtered_estimate = row[f"{method}_filtered"]
        
        # Compute Wasserstein Distance (Wasserstein error)
        wassscore = wasserstein_distance(filtered_estimate, unfiltered_estimate)
        
        # Compute Sum of Absolute Differences (SAD)
        SADscore = np.sum(np.abs(unfiltered_estimate - filtered_estimate)) / np.sum(unfiltered_estimate)
        
        # Compute L2 score (Euclidean distance normalized)
        L2score = np.linalg.norm(unfiltered_estimate - filtered_estimate) / np.linalg.norm(unfiltered_estimate)
        
        # Store the results
        merged_data.at[index, f"{method}_wassscore"] = wassscore
        merged_data.at[index, f"{method}_SADscore"] = SADscore
        merged_data.at[index, f"{method}_L2score"] = L2score


# Filter rows where BW == 1
filtered_merged_data = merged_data[merged_data['BW'] == 1]

# Calculate median of wassscore for each method
median_wassscores = {}
for method in methods:
    median_wassscores[method] = filtered_merged_data[f"{method}_wassscore"].median()

print("Median Wasserstein scores for methods where BW == 1:", median_wassscores)
print("check")
import matplotlib.pyplot as plt
from scipy.stats import mode
from decimal import Decimal

# Filter rows where BW == 0 and BW == 1
excluded_values = merged_data[merged_data["BW"] == 0]
filtered_merged_data = merged_data[merged_data['BW'] == 1]

# Define methods
methods = ['DP_estimate', 'LC_estimate', 'LR_estimate', 'GCV_estimate', 'LS_estimate']

# Plot Wasserstein values for excluded_values
for method in methods:
    max_wassscore_excluded = excluded_values[f"{method}_wassscore"].max()
    min_wassscore_excluded = excluded_values[f"{method}_wassscore"].min()
    median_wassscore_excluded = excluded_values[f"{method}_wassscore"].median()
    mean_wassscore_excluded = excluded_values[f"{method}_wassscore"].mean()
    mode_wassscore_excluded = mode(excluded_values[f"{method}_wassscore"]).mode.item()
    
    max_count_excluded = int((excluded_values[f"{method}_wassscore"] == max_wassscore_excluded).sum())
    min_count_excluded = int((excluded_values[f"{method}_wassscore"] == min_wassscore_excluded).sum())
    mode_count_excluded = int((excluded_values[f"{method}_wassscore"] == mode_wassscore_excluded).sum())
    
    print(f"Excluded Values - {method}: Max Wasserstein Score = {max_wassscore_excluded}, Min Wasserstein Score = {min_wassscore_excluded}, Median Wasserstein Score = {median_wassscore_excluded}, Mean Wasserstein Score = {mean_wassscore_excluded}, Mode Wasserstein Score = {mode_wassscore_excluded}")
    print(f"Counts - Max: {max_count_excluded}, Min: {min_count_excluded}, Mode: {mode_count_excluded}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(excluded_values['X_val'], excluded_values['Y_val'], c=excluded_values[f"{method}_wassscore"], cmap='viridis', marker='o', edgecolor='yellow', label='Excluded Values')
    plt.colorbar(label=f"{method} Wasserstein Score")
    plt.title(f"Wasserstein Scores for {method} (Excluded Values)")
    plt.xlabel('X_val')
    plt.ylabel('Y_val')
    plt.grid(True)
    plt.legend()
    
    # Add text annotations for max, min, median, mean, and mode
    textstr = '\n'.join((
        f"Max: {Decimal(max_wassscore_excluded):.2E} (Count: {max_count_excluded})",
        f"Min: {Decimal(min_wassscore_excluded):.2E} (Count: {min_count_excluded})",
        f"Median: {Decimal(median_wassscore_excluded):.2E}",
        f"Mean: {Decimal(mean_wassscore_excluded):.2E}",
        f"Mode: {Decimal(mode_wassscore_excluded):.2E} (Count: {mode_count_excluded})"
    ))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))
    
    plt.savefig(os.path.join(comparison_folder, f"{method}_wassscore_excluded.png"))
    plt.close()

# Plot Wasserstein values for filtered_merged_data
for method in methods:
    max_wassscore_filtered = filtered_merged_data[f"{method}_wassscore"].max()
    min_wassscore_filtered = filtered_merged_data[f"{method}_wassscore"].min()
    median_wassscore_filtered = filtered_merged_data[f"{method}_wassscore"].median()
    mean_wassscore_filtered = filtered_merged_data[f"{method}_wassscore"].mean()
    mode_wassscore_filtered = mode(filtered_merged_data[f"{method}_wassscore"]).mode.item()
    
    max_count_filtered = int((filtered_merged_data[f"{method}_wassscore"] == max_wassscore_filtered).sum())
    min_count_filtered = int((filtered_merged_data[f"{method}_wassscore"] == min_wassscore_filtered).sum())
    mode_count_filtered = int((filtered_merged_data[f"{method}_wassscore"] == mode_wassscore_filtered).sum())
    
    print(f"Filtered Values - {method}: Max Wasserstein Score = {max_wassscore_filtered}, Min Wasserstein Score = {min_wassscore_filtered}, Median Wasserstein Score = {median_wassscore_filtered}, Mean Wasserstein Score = {mean_wassscore_filtered}, Mode Wasserstein Score = {mode_wassscore_filtered}")
    print(f"Counts - Max: {max_count_filtered}, Min: {min_count_filtered}, Mode: {mode_count_filtered}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_merged_data['X_val'], filtered_merged_data['Y_val'], c=filtered_merged_data[f"{method}_wassscore"], cmap='viridis', marker='o', edgecolor='blue', label='Filtered Values')
    plt.colorbar(label=f"{method} Wasserstein Score")
    plt.title(f"Wasserstein Scores for {method} (Filtered Values)")
    plt.xlabel('X_val')
    plt.ylabel('Y_val')
    plt.grid(True)
    plt.legend()
    
    # Add text annotations for max, min, median, mean, and mode
    textstr = '\n'.join((
        f"Max: {Decimal(max_wassscore_filtered):.2E} (Count: {max_count_filtered})",
        f"Min: {Decimal(min_wassscore_filtered):.2E} (Count: {min_count_filtered})",
        f"Median: {Decimal(median_wassscore_filtered):.2E}",
        f"Mean: {Decimal(mean_wassscore_filtered):.2E}",
        f"Mode: {Decimal(mode_wassscore_filtered):.2E} (Count: {mode_count_filtered})"
    ))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))
    
    plt.savefig(os.path.join(comparison_folder, f"{method}_wassscore_filtered.png"))
    plt.close()