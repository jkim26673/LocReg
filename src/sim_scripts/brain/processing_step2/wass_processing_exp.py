
from scipy.ndimage import rotate
# from src.utils.load_imports.loading import *
# from src.utils.load_imports.load_regmethods import *
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# filepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25_wassscores.pkl"
filepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep2625/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN26Sep25_processed_wassscores.pkl"

with open(filepath, 'rb') as file:
    df1 = pickle.load(file)

mask_filepath = r"/Users/kimjosy/Downloads/LocReg/data/brain/masks/new_mask.mat"
# savepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925"
# savepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/results_06Jun25"
# savepath =r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep2625"
# mask_filepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/data/brain/masks/new_mask.mat"
BW_mask_mat = scipy.io.loadmat(mask_filepath)
BW_mask = BW_mask_mat["new_BW"]
print(f"Loaded brain mask with shape: {BW_mask.shape}")

# mwf_columns_short = ['MWF_Ref', 'MWF_DP', 'MWF_LC', 'MWF_LR', 'MWF_GCV', 'MWF_LR2D', 'MWF_UPEN', 'MWF_LR1D']
# mwf_columns_short = ['MWF_Ref', 'MWF_LR', 'MWF_GCV', 'MWF_LR2D', 'MWF_UPEN', 'MWF_LR1D']
mwf_columns_short = ['Wass_LR', 'Wass_GCV', 'Wass_LR2D', 'Wass_UPEN', 'Wass_LR1D']

# mwf_columns_short = ['sm_MWF_ref', 'sm_MWF_LR', 'sm_MWF_GCV', 'sm_MWF_LR2D', 'sm_MWF_UPEN', 'sm_MWF_LR1D']
# ref_mwf_tag = "sm_MWF_ref"
ref_mwf_tag = "MWF_Ref"

merged_df = df1


# --- 6. Preprocessing ---
# print("\n--- 6. Preprocessing ---")
# merged_df[mwf_columns_short] = merged_df[mwf_columns_short].fillna(0)
# threshold_value = 1e-2
# merged_df[mwf_columns_short] = merged_df[mwf_columns_short].where(merged_df[mwf_columns_short] >= threshold_value, 0)
# zero_mwf_condition = ~(merged_df[mwf_columns_short] == 0).all(axis=1)
# merged_df = merged_df[zero_mwf_condition]
# print(f"DataFrame shape after preprocessing and masking: {merged_df.shape}")

def remove_outliers_std_dev(df, columns, threshold=3):
    """Removes outliers based on standard deviation for specified columns."""
    df_filtered = df.copy()
    for col in columns:
        mean = df_filtered[col].mean()
        std_dev = df_filtered[col].std()
        lower_bound = mean - threshold * std_dev
        upper_bound = mean + threshold * std_dev
        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]
    return df_filtered

# # --- 7. Outlier Removal ---
print("\n--- 7. Outlier Removal ---")
# df_filtered_mwf_outliers = remove_outliers_std_dev(merged_df, mwf_columns_short, threshold=3)
# print(f"DataFrame shape after MWF outlier removal: {df_filtered_mwf_outliers.shape}")
df_filtered_mwf_outliers = merged_df
# --- 8. Absolute Difference Calculation ---
print("\n--- 8. Absolute Difference Calculation ---")
absdiff_columns = [f'{mwf}' for mwf in mwf_columns_short if mwf != ref_mwf_tag]
for mwf in mwf_columns_short:
    if mwf == ref_mwf_tag:
        continue
    # df_filtered_mwf_outliers[f'absdiff_{mwf}'] = np.abs(df_filtered_mwf_outliers['MWF_Ref'] - df_filtered_mwf_outliers[mwf])
    df_filtered_mwf_outliers[f'Median_{mwf}'] = np.median(df_filtered_mwf_outliers[mwf])
    df_filtered_mwf_outliers[f'Mean_{mwf}'] = np.mean(df_filtered_mwf_outliers[mwf])

    # df_filtered_mwf_outliers[f'absdiff_{mwf}'] = np.linalg.norm(df_filtered_mwf_outliers[ref_mwf_tag]- df_filtered_mwf_outliers[mwf], ord = 1)
# print(df_filtered_mwf_outliers)
# --- 8.1. MSE Calculation ---
# print("\n--- 8.1. MSE Calculation ---")
# mse_results = {}
# for mwf in mwf_columns_short:
#     if mwf == ref_mwf_tag:
#         continue
#     mse = np.mean((df_filtered_mwf_outliers[ref_mwf_tag] - df_filtered_mwf_outliers[mwf]) ** 2)
#     mse_results[mwf] = mse

# print("Mean Squared Error (MSE) for each MWF method vs MWF_Ref:")
# for method, mse_val in mse_results.items():
#     print(f"{method}: {mse_val:.10f}")

# --- 9. NaN Handling in Absolute Differences ---
print("\n--- 9. NaN Handling in Absolute Differences ---")
# df_filtered_mwf_outliers[absdiff_columns] = df_filtered_mwf_outliers[absdiff_columns].fillna(0)

print(df_filtered_mwf_outliers)
# --- 12. Result Summarization ---
print("\n--- 12. Result Summarization ---")
average_absdiffs = df_filtered_mwf_outliers[absdiff_columns].mean()
std_dev_absdiffs = df_filtered_mwf_outliers[absdiff_columns].std()
median_absdiffs = df_filtered_mwf_outliers[absdiff_columns].median()

# --- 12.1. Publishable Table Output ---
print("\n--- 12.1. Publishable Table ---")
table_data = {
    'MWF Parameter': [mwf for mwf in mwf_columns_short if mwf != ref_mwf_tag],
    'Mean': average_absdiffs.values,
    'Median': median_absdiffs.values,
    'Std Dev of Abs Diff': std_dev_absdiffs.values,
    # 'MSE vs MWF_Ref': [mse_results[mwf] for mwf in mwf_columns_short if mwf != ref_mwf_tag]
}
table_df = pd.DataFrame(table_data)
table_string = table_df.to_string(index=False, float_format="%.8f")
print("\n" + table_string)
print("\n--- Copy the table above and paste into Google Slides or another document ---")
