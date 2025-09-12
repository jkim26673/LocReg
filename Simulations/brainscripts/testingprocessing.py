#TEST 4
import pandas as pd
import pickle
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import scipy.io  # Import scipy.io to load .mat files

# --- 1. File Paths ---
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noise_addition_recover04Feb25.pkl"
filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14Apr25\est_table_xcoordlen_313_ycoordlen_313_filtered_noise_addition_uniform_noise_UPEN_LR1D2D14Apr25.pkl"

filepath2 = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/filteredbrain/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps29Jan25.pkl"
filepath3 = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_unfiltered_noNESMA21Jan25.pkl"
# filepath3 = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/UnifSNR200/est_table_xcoordlen_313_ycoordlen_313_unfiltered_noise_addition_recover_SNR20017Feb25.pkl"
# filepath4 = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/UnifSNR200/est_table_xcoordlen_313_ycoordlen_313_unfiltered_noise_addition_recover_SNR20017Feb25.pkl"
filepath4 = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/UnifSNR800/est_table_xcoordlen_313_ycoordlen_313_unfiltered_noise_addition_recover_SNR80012Feb25.pkl"
# filepath4 = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/Curr_SNR/est_table_xcoordlen_313_ycoordlen_313_unfiltered_noise_addition_recover_curr_SNR_map12Feb25.pkl"


origfilepath2 = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"
noisefilepath2 = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14Apr25\est_table_xcoordlen_313_ycoordlen_313_filtered_noise_addition_uniform_noise_UPEN_LR1D2D14Apr25.pkl"

nametag = "1D2DLocReg"
noisyfilepath = noisefilepath2
origfilepath = origfilepath2
# mask_filepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat" # Path to your mask file
mask_filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\masks\new_mask.mat"
# nametag = "unfilteredNESMA_SNRvalue800"

# --- 2. Define MWF Column Names ---
# mwf_columns_xy = ['MWF_DP_x', 'MWF_DP_y',
#                     'MWF_LC_x', 'MWF_LC_y', 'MWF_LR_x', 'MWF_LR_y', 'MWF_GCV_x', 'MWF_GCV_y']
# mwf_columns_short = ['MWF_DP', 'MWF_LC', 'MWF_LR', 'MWF_GCV'] # Including MWF_LS for completeness, if needed later
mwf_columns_xy = ['MWF_DP_x', 'MWF_DP_y',
                    'MWF_LC_x', 'MWF_LC_y', 'MWF_LR_x', 'MWF_LR_y', 'MWF_GCV_x', 'MWF_GCV_y', 'MWF_LR2D_x', 'MWF_LR2D_y', 'MWF_UPEN_x', 'MWF_UPEN_y']
mwf_columns_short = ['MWF_DP', 'MWF_LC', 'MWF_LR', 'MWF_GCV', 'MWF_LR2D', 'MWF_UPEN'] # Including MWF_LS for completeness, if needed later

perc_change_columns = [f'perc_{mwf}' for mwf in mwf_columns_short]


# --- 3. Helper Functions ---
def calculate_percent_change(original, noisy):
    """Calculates percent change, handling cases where original is zero."""
    return np.where((original == 0) & (noisy == 0), 0, np.abs(original - noisy) / np.where(original == 0, noisy, original) * 100)

def remove_outliers_std_dev(df, columns, threshold=3):
    """Removes outliers based on standard deviation for specified columns."""
    df_filtered = df.copy() # To avoid modifying the original DataFrame
    for col in columns:
        mean = df_filtered[col].mean()
        std_dev = df_filtered[col].std()
        lower_bound = mean - threshold * std_dev
        upper_bound = mean + threshold * std_dev
        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]
    return df_filtered


# --- 4. Data Loading ---
print("--- 4. Data Loading ---")
with open(noisyfilepath, "rb") as file:
    noise_df = pickle.load(file)
print(f"Loaded noise_df with shape: {noise_df.shape}")

with open(origfilepath, "rb") as file:
    orig_df = pickle.load(file)
print(f"Loaded orig_df with shape: {orig_df.shape}")

# --- 4.1. Load Brain Mask ---
print("\n--- 4.1. Loading Brain Mask ---")
BW_mask_mat = scipy.io.loadmat(mask_filepath)
BW_mask = BW_mask_mat["new_BW"] # Assuming 'new_BW' is the key for your mask in the .mat file
print(f"Loaded brain mask with shape: {BW_mask.shape}")


# --- 5. Merging DataFrames ---
print("\n--- 5. Merging DataFrames ---")
merged_df = pd.merge(orig_df, noise_df, on=['X_val', 'Y_val'], how='outer', suffixes=('_x', '_y')) # Explicit suffixes
print(f"Merged DataFrame shape: {merged_df.shape}")


# --- 6. Preprocessing: NaN Fill, Thresholding, Zero-MWF Removal, Masking ---
print("\n--- 6. Preprocessing ---")
print("Filling NaN values with 0 in MWF columns...")
merged_df[mwf_columns_xy] = merged_df[mwf_columns_xy].fillna(0)

print("Applying threshold to MWF columns (threshold = 1e-2)...")
threshold_value = 1e-2
merged_df[mwf_columns_xy] = merged_df[mwf_columns_xy].where(merged_df[mwf_columns_xy] >= threshold_value, 0)

print("Applying brain mask to MWF values...")
for index, row in merged_df.iterrows():
    x_val = int(row['X_val'])
    y_val = int(row['Y_val'])
    if BW_mask[x_val, y_val] == 0: # If outside brain mask
        for mwf_col in mwf_columns_xy:
            merged_df.at[index, mwf_col] = 0  # Set MWF values to 0

print("Removing rows where all MWF columns are zero (after masking)...")
zero_mwf_condition = ~(merged_df[mwf_columns_xy] == 0).all(axis=1)
merged_df = merged_df[zero_mwf_condition]
print(f"DataFrame shape after preprocessing and masking: {merged_df.shape}")


# --- 7. Outlier Removal (MWF Values) ---
print("\n--- 7. Outlier Removal (MWF Values) ---")
print("Removing outliers from MWF columns using 3 standard deviations...")
df_filtered_mwf_outliers = remove_outliers_std_dev(merged_df, mwf_columns_xy, threshold=3)
print(f"DataFrame shape after MWF outlier removal: {df_filtered_mwf_outliers.shape}")

# --- 7.1. Find Max MWF Values in Filtered DataFrame ---
print("\n--- 7.1. Max MWF Values in Filtered DataFrame (after MWF outlier removal) ---")
max_mwf_values = df_filtered_mwf_outliers[mwf_columns_xy].max()
print("Maximum MWF values in filtered DataFrame (after MWF outlier removal):")
print(max_mwf_values)

# --- 8. Percent Change Calculation ---
print("\n--- 8. Percent Change Calculation ---")
for mwf in mwf_columns_short:
    original_col = f'{mwf}_x'
    noisy_col = f'{mwf}_y'
    perc_col_name = f'perc_{mwf}'
    print(f"Calculating percent change for {mwf}...")
    df_filtered_mwf_outliers[perc_col_name] = calculate_percent_change(
        df_filtered_mwf_outliers[original_col], df_filtered_mwf_outliers[noisy_col]
    )

# --- 9. NaN Handling in Percent Change ---
print("\n--- 9. NaN Handling in Percent Change ---")
print("Filling NaN values with 0 in percent change columns...")
df_filtered_mwf_outliers[perc_change_columns] = df_filtered_mwf_outliers[perc_change_columns].fillna(0)

# # --- 9.1. Max Percent Change Values BEFORE Outlier Removal ---
# print("\n--- 9.1. Max Percent Change Values BEFORE Outlier Removal ---")
# max_perc_change_before_outliers = df_filtered_mwf_outliers[perc_change_columns].max()
# print("Maximum Percent Change values (BEFORE outlier removal):")
# print(max_perc_change_before_outliers)

# # --- 10. Outlier Removal (Percent Change Values) ---
# print("\n--- 10. Outlier Removal (Percent Change Values) ---")
# print("Removing outliers from Percent Change columns using Z-score (threshold=3)...")
# df_before_perc_change_outliers = df_filtered_mwf_outliers.copy() # Keep a copy before removal
# z_scores_perc_change = df_filtered_mwf_outliers[perc_change_columns].apply(zscore)
# df_filtered_perc_change_outliers = df_filtered_mwf_outliers[(z_scores_perc_change.abs() < 3).all(axis=1)]
# print(f"DataFrame shape after Percent Change outlier removal: {df_filtered_perc_change_outliers.shape}")

# # --- 10.1. Max Percent Change Values AFTER Outlier Removal ---
# print("\n--- 10.1. Max Percent Change Values AFTER Outlier Removal ---")
# max_perc_change_after_outliers = df_filtered_perc_change_outliers[perc_change_columns].max()
# print("Maximum Percent Change values (AFTER outlier removal):")
# print(max_perc_change_after_outliers)


final_df_for_summary = df_filtered_mwf_outliers
# final_df_for_summary = max_perc_change_after_outliers

print("columns", final_df_for_summary.columns)
# --- 12. Result Summarization ---
print("\n--- 12. Result Summarization ---")
average_percent_changes = final_df_for_summary[perc_change_columns].median()
# average_percent_changes = final_df_for_summary[perc_change_columns].mean()
std_dev_percent_changes = final_df_for_summary[perc_change_columns].std()

print("\nAverage Percent Changes (Median):")
print(average_percent_changes)
print("\nStandard Deviation of Percent Changes:")
print(std_dev_percent_changes)

# --- 12.1. Publishable Table Output ---
print("\n--- 12.1. Publishable Table: Median and Standard Deviation of Percent Changes ---")
table_data = {
    'MWF Parameter': mwf_columns_short,
    'Median Percent Change': average_percent_changes.values,
    'Standard Deviation': std_dev_percent_changes.values
}
table_df = pd.DataFrame(table_data)
table_string = table_df.to_string(index=False, float_format="%.2f") # 2 decimal places
print("\n"+ table_string)
print("\n--- Copy the table above and paste into Google Slides or other document ---")


# --- 13. Box Plot Visualization ---
print("\n--- 13. Box Plot Visualization ---")
plt.figure(figsize=(10, 6))
final_df_for_summary[perc_change_columns].boxplot()
plt.title('Box Plots of Percent Changes') # Updated title
plt.ylabel('Percent Change')
plt.savefig(f"boxplot_perc_changes_{nametag}.png") # Updated filename to indicate masking
plt.close()
print("Box plot of percent changes saved as boxplot_perc_changes_masked.png")


# --- 14. Histogram Visualization ---
print("\n--- 14. Histogram Visualization ---")
for mwf in mwf_columns_short:
    perc_col_name = f'perc_{mwf}'
    plt.figure(figsize=(8, 6))
    plt.hist(final_df_for_summary[perc_col_name].dropna(), bins=30, edgecolor='black') # Handle potential NaNs in histogram data
    plt.title(f'Histogram of Percent Changes ({mwf})') # Updated title
    plt.xlabel('Percent Change')
    plt.ylabel('Frequency')
    plt.savefig(f"histogram_perc_{mwf}perc_{nametag}.png") # Updated filename to indicate masking
    plt.close()
    print(f"Histogram for {mwf} saved as histogram_perc_{mwf}_{nametag}.png")


print("\n--- Script execution completed successfully (with masking). ---")