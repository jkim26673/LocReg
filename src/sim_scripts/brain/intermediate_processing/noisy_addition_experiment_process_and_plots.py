# filepath2 = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/filteredbrain/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps29Jan25.pkl"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noise_addition_recover04Feb25.pkl"
# import pandas as pd
# import pickle
# import numpy as np

# # with open (filepath, "rb") as file:
# #     noise_df = pickle.load(file)

# # with open (filepath2, "rb") as file:
# #     orig_df = pickle.load(file)

# # # Perform a full outer join
# # # merged_df = pd.merge(orig_df, noise_df, on=['X_val', 'Y_val'], how='outer')
# # merged_df = pd.merge(orig_df, noise_df, on=['X_val', 'Y_val'], how='outer')
# # mwf_columns = ['MWF_LS_x', 'MWF_LS_y', 'MWF_DP_x', 'MWF_DP_y', 
# #                'MWF_LC_x', "MWF_LC_y", 'MWF_LR_x', "MWF_LR_y", 'MWF_GCV_x', "MWF_GCV_y"]

# # # merged_df[mwf_columns] = merged_df[mwf_columns].fillna(0)
# # # Fill NaN values with 0 for the selected MWF columns
# # merged_df[mwf_columns] = merged_df[mwf_columns].fillna(0)

# # # Create a condition to filter out rows where both _x and _y are 0 for any MWF columns
# # condition = ~((merged_df['MWF_LS_x'] == 0) & (merged_df['MWF_LS_y'] == 0) &
# #               (merged_df['MWF_DP_x'] == 0) & (merged_df['MWF_DP_y'] == 0) &
# #               (merged_df['MWF_LC_x'] == 0) & (merged_df['MWF_LC_y'] == 0) &
# #               (merged_df['MWF_LR_x'] == 0) & (merged_df['MWF_LR_y'] == 0) &
# #               (merged_df['MWF_GCV_x'] == 0) & (merged_df['MWF_GCV_y'] == 0))

# # # Apply the condition to filter out the rows
# # merged_df = merged_df[condition]


# # # Replace NaN values in MWF columns with 0
# # mwf_columns = ['MWF_LS', 'MWF_DP', 'MWF_LC', 'MWF_LR', 'MWF_GCV']

# # # Create a function to calculate the percent change
# # def calculate_percent_change(x, y):
# #     # If both x and y are 0, return 0
# #     return np.where((x == 0) & (y == 0), 0, np.abs(x - y) / np.where(x == 0, y, x) * 100)

# # # List of the MWF columns to compare
# # mwf_columns = ['MWF_LS', 'MWF_DP', 'MWF_LC', 'MWF_LR', 'MWF_GCV']

# # # Loop through each pair of columns to calculate percent change and create new columns
# # for mwf in mwf_columns:
# #     # Create column names for x and y (from df and df2)
# #     col_x = f'{mwf}_x'
# #     col_y = f'{mwf}_y'
    
# #     # Calculate percent change and create new column in merged_df
# #     perc_col = f'perc_{mwf}'
# #     merged_df[perc_col] = calculate_percent_change(merged_df[col_x], merged_df[col_y])

# # from scipy.stats import zscore

# # # List of 'perc_MWF' columns
# # perc_columns = ['perc_MWF_LS', 'perc_MWF_DP', 'perc_MWF_LC', 'perc_MWF_LR', 'perc_MWF_GCV']

# # # Calculate Z-scores for each of the perc_MWF columns
# # merged_df[perc_columns] = merged_df[perc_columns].apply(zscore)

# # # Set a threshold to define outliers (commonly 3 or -3)
# # threshold = 3

# # # Filter out rows where the Z-score exceeds the threshold in any of the perc_MWF columns
# # merged_df = merged_df[(merged_df[perc_columns] < threshold).all(axis=1)]

# # # Define a function to remove outliers using IQR
# # def remove_outliers_iqr(df, columns):
# #     Q1 = df[columns].quantile(0.25)
# #     Q3 = df[columns].quantile(0.75)
# #     IQR = Q3 - Q1
    
# #     # Filter out rows where values are outside the IQR range
# #     condition = ~((df[columns] < (Q1 - 1.5 * IQR)) | (df[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
# #     return df[condition]

# # # Remove outliers using IQR for the perc_MWF columns
# # merged_df = remove_outliers_iqr(merged_df, perc_columns)

# # # Calculate the average percent change for each MWF across all pixels
# # average_percent_changes = merged_df[[f'perc_{mwf}' for mwf in mwf_columns]].mean()

# # print("done")
# # # Display the summary of average percent changes
# # print(average_percent_changes)


# import numpy as np
# import pandas as pd
# import pickle
# from scipy.stats import zscore
# import matplotlib.pyplot as plt

# # Load the data
# with open(filepath, "rb") as file:
#     noise_df = pickle.load(file)

# with open(filepath2, "rb") as file:
#     orig_df = pickle.load(file)

# # Perform a full outer join
# merged_df = pd.merge(orig_df, noise_df, on=['X_val', 'Y_val'], how='outer')

# # List of MWF columns
# mwf_columns = ['MWF_LS_x', 'MWF_LS_y', 'MWF_DP_x', 'MWF_DP_y', 
#                'MWF_LC_x', 'MWF_LC_y', 'MWF_LR_x', 'MWF_LR_y', 'MWF_GCV_x', 'MWF_GCV_y']

# # Fill NaN values in MWF columns with 0
# merged_df[mwf_columns] = merged_df[mwf_columns].fillna(0)

# threshold = 1e-2

# # List of MWF columns
# mwf_columns = ['MWF_LS_x', 'MWF_LS_y', 'MWF_DP_x', 'MWF_DP_y', 
#                'MWF_LC_x', 'MWF_LC_y', 'MWF_LR_x', 'MWF_LR_y', 'MWF_GCV_x', 'MWF_GCV_y']

# # Replace values below the threshold with 0
# merged_df[mwf_columns] = merged_df[mwf_columns].where(merged_df[mwf_columns] >= threshold, 0)

# # Create a condition to filter out rows where both _x and _y are 0 for any MWF columns
# condition = ~((merged_df['MWF_LS_x'] == 0) & (merged_df['MWF_LS_y'] == 0) &
#               (merged_df['MWF_DP_x'] == 0) & (merged_df['MWF_DP_y'] == 0) &
#               (merged_df['MWF_LC_x'] == 0) & (merged_df['MWF_LC_y'] == 0) &
#               (merged_df['MWF_LR_x'] == 0) & (merged_df['MWF_LR_y'] == 0) &
#               (merged_df['MWF_GCV_x'] == 0) & (merged_df['MWF_GCV_y'] == 0))

# # Apply the condition to filter out the rows
# merged_df = merged_df[condition]

# print("before merged_df.shape", merged_df.shape)

# def remove_outliers(df, columns):
#     for column in columns:
#         mean = df[column].mean()
#         std_dev = df[column].std()
        
#         # Define the outlier bounds using 3 standard deviations
#         lower_bound = mean - 3 * std_dev
#         upper_bound = mean + 3 * std_dev
        
#         # Filter out the outliers
#         df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
#     return df


# filtered_df = remove_outliers(merged_df, mwf_columns)

# # Find the removed values
# hashable_columns = [col for col in merged_df.columns if not isinstance(merged_df[col].iloc[0], np.ndarray)]
# removed_values = pd.concat([merged_df, filtered_df], axis=0, keys=['original', 'filtered'], names=['source']).reset_index(level='source')
# removed_values = removed_values[removed_values.duplicated(subset=hashable_columns, keep=False)]
# removed_values = removed_values[removed_values['source'] == 'original']

# print("merged_df after removing outliers", filtered_df)
# print("Removed values", removed_values)

# print("ignore after")
# # Remove outliers for the MWF columns using 3 standard deviations
# merged_df = remove_outliers(merged_df, mwf_columns)
# # print("merged_df",merged_df)
# new_merged = merged_df
# # Display the filtered DataFrame
# print(new_merged.shape)

# # Define a function to calculate the percent change
# def calculate_percent_change(x, y):
#     # If both x and y are 0, return 0
#     return np.where((x == 0) & (y == 0), 0, np.abs(x - y) / np.where(x == 0, y, x) * 100)

# # List of MWF columns to compare
# mwf_columns_short = ['MWF_DP', 'MWF_LC', 'MWF_LR', 'MWF_GCV']

# # Loop through each pair of columns to calculate percent change and create new columns
# for mwf in mwf_columns_short:
#     # Create column names for x and y (from df and df2)
#     col_x = f'{mwf}_x'
#     col_y = f'{mwf}_y'
    
#     # Calculate percent change and create new column in merged_df
#     perc_col = f'perc_{mwf}'
#     merged_df[perc_col] = calculate_percent_change(merged_df[col_x], merged_df[col_y])

# perc_columns = ['perc_MWF_DP', 'perc_MWF_LC', 'perc_MWF_LR', 'perc_MWF_GCV']

# # Check for any NaN values in the perc_MWF columns and remove them
# merged_df[perc_columns] = merged_df[perc_columns].fillna(0)

# # Apply Z-score to perc_MWF columns and filter out outliers (using Z-score threshold)
# perc_columns = [ 'perc_MWF_DP', 'perc_MWF_LC', 'perc_MWF_LR', 'perc_MWF_GCV']

# # Calculate Z-scores for each of the perc_MWF columns
# z_scores = merged_df[perc_columns].apply(zscore)

# # Set a threshold to define outliers (commonly 3 or -3)
# threshold = 3

# # Filter out rows where the Z-score exceeds the threshold in any of the perc_MWF columns
# merged_df = merged_df[(z_scores.abs() < threshold).all(axis=1)]

# # Alternatively, you could use IQR to filter out outliers (not both Z-score and IQR)
# # Uncomment below if you want to use IQR instead of Z-score
# # merged_df = remove_outliers_iqr(merged_df, perc_columns)

# # Calculate the average percent change for each MWF across all pixels
# average_percent_changes = merged_df[[f'perc_{mwf}' for mwf in mwf_columns_short]].median()

# std_dev_percent_changes = merged_df[[f'perc_{mwf}' for mwf in mwf_columns_short]].std()

# # Print the average percent changes and their standard deviations
# print("Average Percent Changes:")
# print(average_percent_changes)

# print("\nStandard Deviation of Percent Changes:")
# print(std_dev_percent_changes)

# testDP = np.array(merged_df["perc_MWF_DP"])
# plt.figure()
# plt.hist(testDP, bins=30, edgecolor='black')
# plt.title('Histogram of Percent Changes (MWF_DP)')
# plt.xlabel('Percent Change')
# plt.ylabel('Frequency')
# # Save the histogram as a PNG file
# plt.savefig("histogram_test_DP.png")
# plt.close()

# testLC = np.array(merged_df["perc_MWF_LC"])
# plt.figure()
# plt.hist(testLC, bins=30, edgecolor='black')
# plt.title('Histogram of Percent Changes (MWF_LC)')
# plt.xlabel('Percent Change')
# plt.ylabel('Frequency')
# # Save the histogram as a PNG file
# plt.savefig("histogram_test_LC.png")
# plt.close()

# testGCV = np.array(merged_df["perc_MWF_GCV"])
# plt.figure()
# plt.hist(testGCV, bins=30, edgecolor='black')
# plt.title('Histogram of Percent Changes (MWF_GCV)')
# plt.xlabel('Percent Change')
# plt.ylabel('Frequency')
# # Save the histogram as a PNG file
# plt.savefig("histogram_test_GCV.png")
# plt.close()

# testLR = np.array(merged_df["perc_MWF_LR"])
# plt.figure()
# plt.hist(testDP, bins=30, edgecolor='black')
# plt.title('Histogram of Percent Changes (MWF_LR)')
# plt.xlabel('Percent Change')
# plt.ylabel('Frequency')
# # Save the histogram as a PNG file
# plt.savefig("histogram_test_LR.png")
# plt.close()

#TEST 3

# import pandas as pd
# import pickle
# import numpy as np
# from scipy.stats import zscore
# import matplotlib.pyplot as plt

# # --- 1. File Paths ---
# filepath2 = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/filteredbrain/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps29Jan25.pkl"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noise_addition_recover04Feb25.pkl"


# filepath3 = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_unfiltered_noNESMA21Jan25.pkl"

# # --- 2. Define MWF Column Names ---
# mwf_columns_xy = ['MWF_DP_x', 'MWF_DP_y',
#                   'MWF_LC_x', 'MWF_LC_y', 'MWF_LR_x', 'MWF_LR_y', 'MWF_GCV_x', 'MWF_GCV_y']
# mwf_columns_short = ['MWF_DP', 'MWF_LC', 'MWF_LR', 'MWF_GCV'] # Including MWF_LS for completeness, if needed later
# perc_change_columns = [f'perc_{mwf}' for mwf in mwf_columns_short]


# # --- 3. Helper Functions ---
# def calculate_percent_change(original, noisy):
#     """Calculates percent change, handling cases where original is zero."""
#     return np.where((original == 0) & (noisy == 0), 0, np.abs(original - noisy) / np.where(original == 0, noisy, original) * 100)

# def remove_outliers_std_dev(df, columns, threshold=3):
#     """Removes outliers based on standard deviation for specified columns."""
#     df_filtered = df.copy() # To avoid modifying the original DataFrame
#     for col in columns:
#         mean = df_filtered[col].mean()
#         std_dev = df_filtered[col].std()
#         lower_bound = mean - threshold * std_dev
#         upper_bound = mean + threshold * std_dev
#         df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]
#     return df_filtered


# # --- 4. Data Loading ---
# print("--- 4. Data Loading ---")
# with open(filepath, "rb") as file:
#     noise_df = pickle.load(file)
# print(f"Loaded noise_df with shape: {noise_df.shape}")

# with open(filepath2, "rb") as file:
#     orig_df = pickle.load(file)
# print(f"Loaded orig_df with shape: {orig_df.shape}")


# # --- 5. Merging DataFrames ---
# print("\n--- 5. Merging DataFrames ---")
# merged_df = pd.merge(orig_df, noise_df, on=['X_val', 'Y_val'], how='outer', suffixes=('_x', '_y')) # Explicit suffixes
# print(f"Merged DataFrame shape: {merged_df.shape}")


# # --- 6. Preprocessing: NaN Fill, Thresholding, Zero-MWF Removal ---
# print("\n--- 6. Preprocessing ---")
# print("Filling NaN values with 0 in MWF columns...")
# merged_df[mwf_columns_xy] = merged_df[mwf_columns_xy].fillna(0)

# print("Applying threshold to MWF columns (threshold = 1e-2)...")
# threshold_value = 1e-2
# merged_df[mwf_columns_xy] = merged_df[mwf_columns_xy].where(merged_df[mwf_columns_xy] >= threshold_value, 0)

# print("Removing rows where all MWF columns are zero...")
# zero_mwf_condition = ~(merged_df[mwf_columns_xy] == 0).all(axis=1)
# merged_df = merged_df[zero_mwf_condition]
# print(f"DataFrame shape after preprocessing: {merged_df.shape}")


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


# # # --- 11. Filter Percent Changes >= 100% BEFORE Summarization ---
# # print("\n--- 11. Filter Percent Changes >= 100% ---")
# # df_before_100perc_filter = df_filtered_perc_change_outliers.copy() # Keep a copy before filtering
# # filter_100perc_condition = (df_filtered_perc_change_outliers[perc_change_columns] < 100).all(axis=1) # Condition: ALL perc_change cols < 100
# # df_filtered_100perc = df_filtered_perc_change_outliers[filter_100perc_condition] # Apply filter
# # print(f"DataFrame shape before 100% percent change filter: {df_before_100perc_filter.shape}")
# # print(f"DataFrame shape after 100% percent change filter: {df_filtered_100perc.shape}")
# # final_df_for_summary = df_filtered_100perc.copy() # Use this for summary and plots from now on
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
# plt.savefig("boxplot_perc_changes_perc.png") # Updated filename
# plt.close()
# print("Box plot of percent changes saved as boxplot_perc_changes.png")


# # --- 14. Histogram Visualization ---
# print("\n--- 14. Histogram Visualization ---")
# for mwf in mwf_columns_short:
#     perc_col_name = f'perc_{mwf}'
#     plt.figure(figsize=(8, 6))
#     plt.hist(final_df_for_summary[perc_col_name].dropna(), bins=30, edgecolor='black') # Handle potential NaNs in histogram data
#     plt.title(f'Histogram of Percent Changes ({mwf})') # Updated title
#     plt.xlabel('Percent Change')
#     plt.ylabel('Frequency')
#     plt.savefig(f"histogram_perc_{mwf}perc.png") # Updated filename
#     plt.close()
#     print(f"Histogram for {mwf} saved as histogram_perc_{mwf}.png")


# print("\n--- Script execution completed successfully. ---")

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

# filepath1orig = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14Apr25\est_table_xcoordlen_313_ycoordlen_313_filtered_noise_addition_uniform_noise_UPEN_LR1D2D14Apr25.pkl"
# filepath2orig = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"
filepath2orig = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_26Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps_GCV_LR012_UPEN26Apr25.pkl"
# filepath2orig = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"
filepath1orig = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_29Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN29Apr25.pkl"
with open(filepath2orig, 'rb') as file:
    df1 = pickle.load(file)

# UPENfileMM = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_22Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps_justUPEN22Apr25.pkl"
# with open(UPENfileMM, 'rb') as file:
#     dfUPENMM = pickle.load(file)

# df1["MWF_UPEN"] = dfUPENMM["MWF_UPEN"]
# df1["UPEN_estimate"] = dfUPENMM["UPEN_estimate"]
# df1["Lam_UPEN"] = dfUPENMM["Lam_UPEN"]

with open(filepath1orig, 'rb') as file:
    df2 = pickle.load(file)

# UPENfileMMNA = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_22Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noiseaddition_justUPEN22Apr25.pkl"
# UPENfileMMNA = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_22Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noiseadditionUPEN22Apr25.pkl"
# with open(UPENfileMMNA, 'rb') as file:
#     dfUPENMMNA = pickle.load(file)

# df2["MWF_UPEN"] = dfUPENMMNA["MWF_UPEN"]
# df2["UPEN_estimate"] = dfUPENMMNA["UPEN_estimate"]
# df2["Lam_UPEN"] = dfUPENMMNA["Lam_UPEN"]

orig_df = df1
noise_df = df2
# origfilepath2 = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_22Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps_justUPEN22Apr25.pkl"
# noisefilepath2 = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_22Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noiseaddition_justUPEN22Apr25.pkl"

nametag = "1D2DLocReg"
# noisyfilepath = noisefilepath2
# origfilepath = origfilepath2
# mask_filepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat" # Path to your mask file
mask_filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\masks\new_mask.mat"
# nametag = "unfilteredNESMA_SNRvalue800"

# --- 2. Define MWF Column Names ---
# mwf_columns_xy = ['MWF_DP_x', 'MWF_DP_y',
#                     'MWF_LC_x', 'MWF_LC_y', 'MWF_LR_x', 'MWF_LR_y', 'MWF_GCV_x', 'MWF_GCV_y']
# mwf_columns_short = ['MWF_DP', 'MWF_LC', 'MWF_LR', 'MWF_GCV'] # Including MWF_LS for completeness, if needed later
mwf_columns_xy = ['MWF_DP_x', 'MWF_DP_y',
                    'MWF_LC_x', 'MWF_LC_y', 'MWF_LR_x', 'MWF_LR_y', 'MWF_GCV_x', 'MWF_GCV_y', 'MWF_LR2D_x', 'MWF_LR2D_y', 'MWF_UPEN_x', 'MWF_UPEN_y', 'MWF_LR1D_x','MWF_LR1D_y']
# mwf_columns_xy = ['MWF_DP_x', 'MWF_DP_y',
#                     'MWF_LC_x', 'MWF_LC_y', 'MWF_LR_x', 'MWF_LR_y', 'MWF_GCV_x', 'MWF_GCV_y', 'MWF_UPEN_x', 'MWF_UPEN_y']
mwf_columns_short = ['MWF_DP', 'MWF_LC', 'MWF_LR', 'MWF_GCV', 'MWF_LR2D', 'MWF_UPEN','MWF_LR1D'] # Including MWF_LS for completeness, if needed later
# mwf_columns_short = ['MWF_DP', 'MWF_LC', 'MWF_LR', 'MWF_GCV', 'MWF_UPEN'] # Including MWF_LS for completeness, if needed later

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
# print("--- 4. Data Loading ---")
# with open(noisyfilepath, "rb") as file:
#     noise_df = pickle.load(file)
# print(f"Loaded noise_df with shape: {noise_df.shape}")

# with open(origfilepath, "rb") as file:
#     orig_df = pickle.load(file)
# print(f"Loaded orig_df with shape: {orig_df.shape}")

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

# --- 9.1. Max Percent Change Values BEFORE Outlier Removal ---
print("\n--- 9.1. Max Percent Change Values BEFORE Outlier Removal ---")
max_perc_change_before_outliers = df_filtered_mwf_outliers[perc_change_columns].max()
print("Maximum Percent Change values (BEFORE outlier removal):")
print(max_perc_change_before_outliers)

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

# --- 12. Result Summarization ---
print("\n--- 12. Result Summarization ---")
average_percent_changes = final_df_for_summary[perc_change_columns].median()
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






























# # Filter out rows where any perc_MWF value is less than 0 or greater than 100
# merged_df = merged_df[(merged_df[perc_columns] > 0).all(axis=1) & (merged_df[perc_columns] < 100).all(axis=1)]
# # Now, calculate the average and standard deviation on the filtered DataFrame
# average_percent_changes = merged_df[[f'perc_{mwf}' for mwf in mwf_columns_short]].median()

# std_dev_percent_changes = merged_df[[f'perc_{mwf}' for mwf in mwf_columns_short]].std()

# # Print the average percent changes and their standard deviations
# print("Average Percent Changes:")
# print(average_percent_changes)

# print("\nStandard Deviation of Percent Changes:")
# print(std_dev_percent_changes)

# testDP = np.array(merged_df["perc_MWF_DP"])
# plt.figure()
# plt.hist(testDP, bins=30, edgecolor='black')
# plt.title('Histogram of Percent Changes (MWF_DP)')
# plt.xlabel('Percent Change')
# plt.ylabel('Frequency')
# # Save the histogram as a PNG file
# plt.savefig("histogram_test_filtered_DP.png")
# plt.close()

# testLC = np.array(merged_df["perc_MWF_LC"])
# print("testLC std", np.std(testLC))
# plt.figure()
# plt.hist(testLC, bins=30, edgecolor='black')
# plt.title('Histogram of Percent Changes (MWF_LC)')
# plt.xlabel('Percent Change')
# plt.ylabel('Frequency')
# # Save the histogram as a PNG file
# plt.savefig("histogram_test_filtered_LC.png")
# plt.close()

# testGCV = np.array(merged_df["perc_MWF_GCV"])
# plt.figure()
# plt.hist(testGCV, bins=30, edgecolor='black')
# plt.title('Histogram of Percent Changes (MWF_GCV)')
# plt.xlabel('Percent Change')
# plt.ylabel('Frequency')
# # Save the histogram as a PNG file
# plt.savefig("histogram_test_filtered_GCV.png")
# plt.close()

# testLR = np.array(merged_df["perc_MWF_LR"])
# plt.figure()
# plt.hist(testDP, bins=30, edgecolor='black')
# plt.title('Histogram of Percent Changes (MWF_LR)')
# plt.xlabel('Percent Change')
# plt.ylabel('Frequency')
# # Save the histogram as a PNG file
# plt.savefig("histogram_test_filtered_LR.png")
# plt.close()