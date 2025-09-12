import pandas as pd
import pickle
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import scipy.io  # Import scipy.io to load .mat files

# --- 1. File Paths ---
# filtNA_GCV_LR = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14Apr25\est_table_xcoordlen_313_ycoordlen_313_filtered_noise_addition_uniform_noise_UPEN_LR1D2D14Apr25.pkl"
# filtMM_GCV_LR = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"

# with open(filtMM_GCV_LR, 'rb') as file:
#     df1 = pickle.load(file)

# UPENfileMM = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_22Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps_justUPEN22Apr25.pkl"
# with open(UPENfileMM, 'rb') as file:
#     dfUPENMM = pickle.load(file)

# df1["MWF_UPEN"] = dfUPENMM["MWF_UPEN"]
# df1["UPEN_estimate"] = dfUPENMM["UPEN_estimate"]
# df1["Lam_UPEN"] = dfUPENMM["Lam_UPEN"]

# with open(filtNA_GCV_LR, 'rb') as file:
#     df2 = pickle.load(file)

# # UPENfileMMNA = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_22Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noiseaddition_justUPEN22Apr25.pkl"
# UPENfileMMNA = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_22Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noiseadditionUPEN22Apr25.pkl"
# with open(UPENfileMMNA, 'rb') as file:
#     dfUPENMMNA = pickle.load(file)

# df2["MWF_UPEN"] = dfUPENMMNA["MWF_UPEN"]
# df2["UPEN_estimate"] = dfUPENMMNA["UPEN_estimate"]
# df2["Lam_UPEN"] = dfUPENMMNA["Lam_UPEN"]
mmdf = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_26Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps_GCV_LR012_UPEN26Apr25.pkl"
# filepath2orig = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"
noisedf = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_29Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN29Apr25.pkl"

with open(mmdf, 'rb') as file:
    df1 = pickle.load(file)
with open(noisedf, 'rb') as file:
    df2 = pickle.load(file)
orig_df = df1
noise_df = df2

nametag = "1D2DLocReg"
mask_filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\masks\new_mask.mat"

orig_df_clean = orig_df.dropna(axis=1)

# Drop columns with any NaN values in noise_df
noise_df_clean = noise_df.dropna(axis=1)

# Ensure both DataFrames have the same columns (keep common columns)
common_columns = orig_df_clean.columns.intersection(noise_df_clean.columns)
orig_df = orig_df_clean[common_columns]
noise_df = noise_df_clean[common_columns]
print(noise_df.columns)
print(orig_df.columns)
# --- 2. Define MWF Column Names ---
mwf_columns_xy = ['MWF_DP_x', 'MWF_DP_y',
                    'MWF_LC_x', 'MWF_LC_y', 'MWF_LR_x', 'MWF_LR_y', 'MWF_GCV_x', 'MWF_GCV_y', 'MWF_LR2D_x', 'MWF_LR2D_y','MWF_LR1D_x', 'MWF_LR1D_y', 'MWF_UPEN_x', 'MWF_UPEN_y']
mwf_columns_short = ['MWF_DP', 'MWF_LC', 'MWF_LR', 'MWF_GCV', 'MWF_LR2D', 'MWF_LR1D', 'MWF_UPEN'] # Including MWF_LS for completeness, if needed later
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

# --- 4.1. Load Brain Mask ---
print("\n--- 4.1. Loading Brain Mask ---")
BW_mask_mat = scipy.io.loadmat(mask_filepath)
BW_mask = BW_mask_mat["new_BW"] # Assuming 'new_BW' is the key for your mask in the .mat file
print(f"Loaded brain mask with shape: {BW_mask.shape}")

# --- 5. Merging DataFrames ---
print("\n--- 5. Merging DataFrames ---")
merged_df = pd.merge(orig_df, noise_df, on=['X_val', 'Y_val'], how='outer', suffixes=('_x', '_y')) # Explicit suffixes
print(f"Merged DataFrame shape: {merged_df.shape}")

# # --- 6. Preprocessing: NaN Fill, Thresholding, Zero-MWF Removal, Masking ---
# print("\n--- 6. Preprocessing ---")
# print("Filling NaN values with 0 in MWF columns...")
# merged_df[mwf_columns_xy] = merged_df[mwf_columns_xy].fillna(0)

# print("Applying threshold to MWF columns (threshold = 1e-2)...")
# threshold_value = 1e-2
# merged_df[mwf_columns_xy] = merged_df[mwf_columns_xy].where(merged_df[mwf_columns_xy] >= threshold_value, 0)

# print("Applying brain mask to MWF values...")
# for index, row in merged_df.iterrows():
#     x_val = int(row['X_val'])
#     y_val = int(row['Y_val'])
#     if BW_mask[x_val, y_val] == 0: # If outside brain mask
#         for mwf_col in mwf_columns_xy:
#             merged_df.at[index, mwf_col] = 0  # Set MWF values to 0

# print("Removing rows where all MWF columns are zero (after masking)...")
# zero_mwf_condition = ~(merged_df[mwf_columns_xy] == 0).all(axis=1)
# merged_df = merged_df[zero_mwf_condition]
# print(f"DataFrame shape after preprocessing and masking: {merged_df.shape}")

# # --- 7. Outlier Removal (MWF Values) ---
# print("\n--- 7. Outlier Removal (MWF Values) ---")
# print("Removing outliers from MWF columns using 3 standard deviations...")
# df_filtered_mwf_outliers = remove_outliers_std_dev(merged_df, mwf_columns_xy, threshold=3)
# print(f"DataFrame shape after MWF outlier removal: {df_filtered_mwf_outliers.shape}")

# # --- 7.1. Find Max MWF Values in Filtered DataFrame ---
# print("\n--- 7.1. Max MWF Values in Filtered DataFrame (after MWF outlier removal) ---")
# max_mwf_values = df_filtered_mwf_outliers[mwf_columns_xy].max()
# print("Maximum MWF values in filtered DataFrame (after MWF outlier removal):")
# print(max_mwf_values)

# # --- 8. Percent Change Calculation ---
# print("\n--- 8. Percent Change Calculation ---")
# for mwf in mwf_columns_short:
#     original_col = f'{mwf}_x'
#     noisy_col = f'{mwf}_y'
#     perc_col_name = f'perc_{mwf}'
#     print(f"Calculating percent change for {mwf}...")
#     df_filtered_mwf_outliers[perc_col_name] = calculate_percent_change(
#         df_filtered_mwf_outliers[original_col], df_filtered_mwf_outliers[noisy_col]
#     )

# # --- 9. NaN Handling in Percent Change ---
# print("\n--- 9. NaN Handling in Percent Change ---")
# print("Filling NaN values with 0 in percent change columns...")
# df_filtered_mwf_outliers[perc_change_columns] = df_filtered_mwf_outliers[perc_change_columns].fillna(0)

# # --- 9.1. Max Percent Change Values BEFORE Outlier Removal ---
# print("\n--- 9.1. Max Percent Change Values BEFORE Outlier Removal ---")
# max_perc_change_before_outliers = df_filtered_mwf_outliers[perc_change_columns].max()
# print("Maximum Percent Change values (BEFORE outlier removal):")
# print(max_perc_change_before_outliers)


# # # --- 10. Outlier Removal (Percent Change Values) ---
# # print("\n--- 10. Outlier Removal (Percent Change Values) ---")
# # print("Removing outliers from Percent Change columns using Z-score (threshold=3)...")
# # df_before_perc_change_outliers = df_filtered_mwf_outliers.copy() # Keep a copy before removal
# # z_scores_perc_change = df_filtered_mwf_outliers[perc_change_columns].apply(zscore)
# # df_filtered_perc_change_outliers = df_filtered_mwf_outliers[(z_scores_perc_change.abs() < 3).all(axis=1)]
# # print(f"DataFrame shape after Percent Change outlier removal: {df_filtered_perc_change_outliers.shape}")

# #Step 1: Calculate IQR for each percent change column
# Q1 = df_filtered_mwf_outliers[perc_change_columns].quantile(0.25)
# Q3 = df_filtered_mwf_outliers[perc_change_columns].quantile(0.75)
# IQR = Q3 - Q1

# # Step 2: Create a mask for rows that are NOT outliers in any of the columns
# mask = ~((df_filtered_mwf_outliers[perc_change_columns] < (Q1 - 1.5 * IQR)) | 
#          (df_filtered_mwf_outliers[perc_change_columns] > (Q3 + 1.5 * IQR))).any(axis=1)

# # Step 3: Filter the DataFrame
# df_filtered_perc_change_outliers = df_filtered_mwf_outliers[mask]

# # Optional: Print the result
# print("\n--- After IQR Outlier Removal ---")
# print(f"Original shape: {df_filtered_mwf_outliers.shape}")
# print(f"New shape: {df_filtered_perc_change_outliers.shape}")
# # # --- 10.1. Max Percent Change Values AFTER Outlier Removal ---
# # print("\n--- 10.1. Max Percent Change Values AFTER Outlier Removal ---")
# # max_perc_change_after_outliers = df_filtered_perc_change_outliers[perc_change_columns].max()
# # print("Maximum Percent Change values (AFTER outlier removal):")
# # print(max_perc_change_after_outliers)

# final_df_for_summary = df_filtered_perc_change_outliers

# # --- 12. Result Summarization ---
# print("\n--- 12. Result Summarization ---")
# average_percent_changes = final_df_for_summary[perc_change_columns].median()
# std_dev_percent_changes = final_df_for_summary[perc_change_columns].std()

# print("\nAverage Percent Changes (Median):")
# print(average_percent_changes)
# print("\nStandard Deviation of Percent Changes:")
# print(std_dev_percent_changes)

# # --- 12.1. Publishable Table Output ---
# print("\n--- 12.1. Publishable Table: Median and Standard Deviation of Percent Changes ---")
# table_data = {
#     'MWF Parameter': mwf_columns_short,
#     'Median Percent Change': average_percent_changes.values,
#     'Standard Deviation': std_dev_percent_changes.values
# }
# table_df = pd.DataFrame(table_data)
# table_string = table_df.to_string(index=False, float_format="%.2f") # 2 decimal places
# print("\n"+ table_string)
# print("\n--- Copy the table above and paste into Google Slides or other document ---")


# # --- 13. Box Plot Visualization ---
# print("\n--- 13. Box Plot Visualization ---")
# plt.figure(figsize=(10, 6))
# final_df_for_summary[perc_change_columns].boxplot()
# plt.title('Box Plots of Percent Changes') # Updated title
# plt.ylabel('Percent Change')
# plt.savefig(f"boxplot_perc_changes_{nametag}.png") # Updated filename to indicate masking
# plt.close()
# print("Box plot of percent changes saved as boxplot_perc_changes_masked.png")


# # --- 14. Histogram Visualization ---
# print("\n--- 14. Histogram Visualization ---")
# for mwf in mwf_columns_short:
#     perc_col_name = f'perc_{mwf}'
#     plt.figure(figsize=(8, 6))
#     plt.hist(final_df_for_summary[perc_col_name].dropna(), bins=30, edgecolor='black') # Handle potential NaNs in histogram data
#     plt.title(f'Histogram of Percent Changes ({mwf})') # Updated title
#     plt.xlabel('Percent Change')
#     plt.ylabel('Frequency')
#     plt.savefig(f"histogram_perc_{mwf}perc_{nametag}.png") # Updated filename to indicate masking
#     plt.close()
#     print(f"Histogram for {mwf} saved as histogram_perc_{mwf}_{nametag}.png")


# print("\n--- Script execution completed successfully (with masking). ---")


import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import scipy.io
import pickle

# --- 3. Enhanced Helper Functions ---
def calculate_percent_change(original, noisy, epsilon=1e-6):
    """Calculates robust percent change with epsilon to prevent division by near-zero."""
    return np.where(
        (original == 0) & (noisy == 0), 0,
        np.abs(original - noisy) / np.where(np.abs(original) < epsilon, np.abs(noisy) + epsilon, original) * 100
    )

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """Removes outliers using the IQR method."""
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[columns] < (Q1 - multiplier * IQR)) |
             (df[columns] > (Q3 + multiplier * IQR))).any(axis=1)
    return df[mask]

# --- 6. Faster Masking ---
print("Applying brain mask to MWF values (vectorized)...")
x_vals = merged_df["X_val"].astype(int).values
y_vals = merged_df["Y_val"].astype(int).values
valid_mask = BW_mask[x_vals, y_vals] != 0
merged_df.loc[~valid_mask, mwf_columns_xy] = 0

# # --- 7. Outlier Removal (MWF) using IQR ---
print("\n--- 7. Outlier Removal (MWF Values using IQR) ---")
df_filtered_mwf_outliers = remove_outliers_iqr(merged_df, mwf_columns_xy, multiplier=1.5)
print(f"Shape after MWF IQR outlier removal: {df_filtered_mwf_outliers.shape}")
# df_filtered_mwf_outliers = merged_df
# # --- 8. Percent Change Calculation (with epsilon) ---
# print("\n--- 8. Percent Change Calculation ---")
# for mwf in mwf_columns_short:
#     original_col = f'{mwf}_x'
#     noisy_col = f'{mwf}_y'
#     perc_col_name = f'perc_{mwf}'
#     df_filtered_mwf_outliers[perc_col_name] = calculate_percent_change(
#         df_filtered_mwf_outliers[original_col], df_filtered_mwf_outliers[noisy_col]
#     )

# --- 8. Absolute Difference Calculation ---
# --- 8. Absolute Difference Calculation ---
for mwf in mwf_columns_short:
    original_col = f'{mwf}_x'
    noisy_col = f'{mwf}_y'
    diff_col_name = f'diff_{mwf}'
    print(f"Calculating absolute difference for {mwf}...")
    df_filtered_mwf_outliers.loc[:, diff_col_name] = np.abs(df_filtered_mwf_outliers[original_col] - df_filtered_mwf_outliers[noisy_col])

diff_columns = ["diff_MWF_LR", "diff_MWF_GCV", "diff_MWF_LR2D", "diff_MWF_LR1D", "diff_MWF_UPEN"]
# Drop rows where any of the diff_ columns have NaN
# Drop rows where any of the diff_ columns have NaN
df_clean = df_filtered_mwf_outliers.dropna(subset=diff_columns)

# Calculate mean and standard deviation for each diff_ column
means = df_clean[diff_columns].mean()
std_devs = df_clean[diff_columns].std()

# Calculate MSE for each diff_ column
mse_values = means**2 + std_devs**2

# Rank the MSEs (lowest MSE gets rank 1)
mse_ranks = mse_values.rank(method='min').astype(int)

# Print MSE values with scientific notation and ranks
print("\n--- Mean Squared Error (MSE) and Ranks ---")
for diff_col in diff_columns:
    mse_val = mse_values[diff_col]
    rank = mse_ranks[diff_col]
    print(f"{diff_col}: MSE = {mse_val:.2e}, Rank = {rank}")

# Create a figure with subplots (one histogram per diff_ column)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
axes = axes.flatten()  # Flatten for easier iteration

# Plot histogram for each diff_ column
for i, diff_col in enumerate(diff_columns):
    col_data = df_clean[diff_col]
    mse_val = mse_values[diff_col]
    rank = mse_ranks[diff_col]
    axes[i].hist(col_data, edgecolor='black', alpha=0.7)
    axes[i].set_title(
        f'Distribution of {diff_col}\nMSE = {mse_val:.2e}, Rank = {rank}'
    )
    axes[i].set_xlabel('MWF Absolute Difference')
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, alpha=0.3)

# Remove the extra subplot (since we have 5 columns but 6 subplots)
fig.delaxes(axes[-1])

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig("noiseaddition_histogram_mse_ranked.png", dpi=300, bbox_inches='tight')
plt.close()  # Close to free memory
 # Close to free memory

# --- 9. NaN Handling in Absolute Difference Columns ---
# print("\n--- 9. NaN Handling in Absolute Difference ---")
# # Update column names for absolute differences
# diff_columns = [f'diff_{mwf}' for mwf in mwf_columns_short]
# # Fill NaN values with 0 in absolute difference columns
# df_filtered_mwf_outliers[diff_columns] = df_filtered_mwf_outliers[diff_columns].fillna(0)

# # --- 9. Fill NaNs in Percent Change ---
# df_filtered_mwf_outliers[perc_change_columns] = df_filtered_mwf_outliers[perc_change_columns].fillna(0)

# --- 10. Outlier Removal (Percent Change using IQR) ---
# print("\n--- 10. Outlier Removal (Percent Change using IQR) ---")
# df_filtered_perc_change_outliers = remove_outliers_iqr(df_filtered_mwf_outliers, diff_columns, multiplier=1.5)
# print(f"Shape after Percent Change IQR outlier removal: {df_filtered_perc_change_outliers.shape}")

# --- 12. Result Summarization (Median + Standard Deviation) ---
print("\n--- 12. Result Summarization (Median + Standard Deviation) ---")
final_df_for_summary = df_clean

diff_columns = ["diff_MWF_LR", "diff_MWF_GCV", "diff_MWF_LR2D", "diff_MWF_LR1D", "diff_MWF_UPEN"]
mwf_columns_short = ['MWF_LR', 'MWF_GCV', 'MWF_LR2D', 'MWF_LR1D', 'MWF_UPEN'] # Including MWF_LS for completeness, if needed later

final_df_for_summary
# Calculate median
median_percent_changes = final_df_for_summary[diff_columns].median()

# Calculate standard deviation from median
std_dev_percent_changes = final_df_for_summary[diff_columns].std()

# Publishable Table
table_data = {
    'MWF Parameter': mwf_columns_short,
    'Median Absolute Difference': median_percent_changes.values,
    'Standard Deviation': std_dev_percent_changes.values
}

table_df = pd.DataFrame(table_data)
print("\n--- Median and Standard Deviation of Absolute Differences ---")
print(table_df.to_string(index=False))

# --- 13. Box Plot ---
print("\n--- 13. Box Plot Visualization ---")
plt.figure(figsize=(10, 6))
final_df_for_summary[diff_columns].boxplot()
plt.title('Box Plots of Absolute Differences')
plt.ylabel('Absolute Difference')
plt.savefig(f"boxplot_abs_diff_{nametag}.png")
plt.close()
print("Saved boxplot.")

# --- 14. Histograms ---
print("\n--- 14. Histogram Visualization ---")
for mwf in mwf_columns_short:
    col = f'diff_{mwf}'
    plt.figure(figsize=(8, 6))
    plt.hist(final_df_for_summary[col].dropna(), bins=30, edgecolor='black')
    plt.title(f'Histogram of Absolute Differences ({mwf})')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency')
    plt.savefig(f"histogram_abs_diff_{mwf}_{nametag}.png")
    plt.close()
    print(f"Saved histogram for {mwf}.")
