import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

nsigma = 5
nrps = 5
unif_sigma = np.linspace(2, 5, nsigma).T
rps = np.linspace(1.1, 4, 5).T
# File paths and parameters
modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000/est_table_SNR1000_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_18Oct24.pkl"
file_path_final = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000"

# Define values to keep
sigma_values_to_keep = [(2.0, 6.0), (2.75, 8.25), (3.5, 10.5)]
rps_values_to_keep = [1.100, 1.825, 2.550]

# Load the data
with open(modfile, 'rb') as file:
    errors = pickle.load(file)

# Drop unwanted columns and create DataFrame
errors = errors.drop(["DP_vect", "GCV_vect", "LR_vect", "oracle_vect", "LC_vect"], axis=1)
df = pd.DataFrame(errors)

# Convert Sigma values to tuples
df['Sigma'] = df['Sigma'].apply(tuple)

# Print unique RPS_val values in ascending order
# print("Unique RPS_val values in ascending order:")

# sigma_val = df["Sigma"].sort_values(ascending=True).unique()[0:3]
# print("sigma_val", sigma_val)
# rps_val = df["RPS_val"].sort_values(ascending=True).unique()[0:3]
# print("rps_val", rps_val)
# # Apply the filter to keep only the specified 
# # rps_values_to_keep = [1.100, 1.825, 2.550]
# # Sort the filtered DataFrame

df['Sigma'] = df['Sigma'].apply(tuple)
df_sorted = df.sort_values(by=['NR','Sigma', 'RPS_val'], ascending=[True, True, True])
print("df_sorted", df_sorted)

grouped = df_sorted.groupby(['Sigma', 'RPS_val']).agg({
    'err_DP': 'sum',
    'err_LC': 'sum',
    'err_LR': 'sum',
    'err_GCV': 'sum',
    'err_oracle': 'sum',
})
num_NRs = df_sorted['NR'].nunique()
# Average the errors
average_errors = grouped / num_NRs
errors = average_errors

errs_oracle = errors["err_oracle"].to_numpy().reshape(nsigma,nrps)
errs_oracle = np.array(errs_oracle)
errs_LC= errors["err_LC"].to_numpy().reshape(nsigma,nrps)
errs_LC = np.array(errs_LC)
errs_GCV = errors["err_GCV"].to_numpy().reshape(nsigma,nrps)
errs_GCV = np.array(errs_GCV)
errs_DP = errors["err_DP"].to_numpy().reshape(nsigma,nrps)
errs_DP = np.array(errs_DP)
errs_LR = errors["err_LR"].to_numpy().reshape(nsigma,nrps)
errs_LR = np.array(errs_LR)

compare_GCV = errs_LR - errs_GCV
compare_DP = errs_LR - errs_DP
compare_LC = errs_LR - errs_LC
compare_oracle = errs_LR - errs_oracle

compare_GCV = compare_GCV[:3, :3]
compare_DP = compare_DP[:3, :3]
compare_LC = compare_LC[:3, :3]
compare_oracle = compare_oracle[:3, :3]

# df_sorted = df_filtered.sort_values(by=['NR', 'Sigma', 'RPS_val'], ascending=[True, True, True])
# print("Sorted DataFrame:")
# print(df_sorted)

# Calculate the number of unique NR values and average the errors
# num_NRs = df_sorted['NR'].nunique()
# grouped = df_sorted.groupby(['Sigma', 'RPS_val']).agg({
#     'err_DP': 'sum',
#     'err_LC': 'sum',
#     'err_LR': 'sum',
#     'err_GCV': 'sum',
#     'err_oracle': 'sum'
# })

# # Average the errors
# average_errors = grouped / num_NRs
# errors = average_errors

# # Reshape error arrays
# errs_oracle = errors["err_oracle"].to_numpy().reshape(len(sigma_values_to_keep), len(rps_values_to_keep))
# errs_LC = errors["err_LC"].to_numpy().reshape(len(sigma_values_to_keep), len(rps_values_to_keep))
# errs_GCV = errors["err_GCV"].to_numpy().reshape(len(sigma_values_to_keep), len(rps_values_to_keep))
# errs_DP = errors["err_DP"].to_numpy().reshape(len(sigma_values_to_keep), len(rps_values_to_keep))
# errs_LR = errors["err_LR"].to_numpy().reshape(len(sigma_values_to_keep), len(rps_values_to_keep))

# # Compute comparisons
# compare_GCV = errs_LR - errs_GCV
# compare_DP = errs_LR - errs_DP
# compare_LC = errs_LR - errs_LC
# compare_oracle = errs_LR - errs_oracle

def compare_heatmap():
    fig, axs = plt.subplots(2, 2, sharey=True, figsize=(14, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # Define tick labels for each method
    tick_labels = [
        ['LocReg is better', 'Neutral', 'GCV is better'],
        ['LocReg is better', 'Neutral', 'DP is better'],
        ['LocReg is better', 'Neutral', 'L-Curve is better'],
        ['LocReg is better', 'Neutral', 'Oracle is better']
    ]
    axs = axs.flatten()
    x_ticks = rps[:3]
    y_ticks = unif_sigma[:3]

    def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
        im = sns.heatmap(data, cmap='jet', ax=ax, cbar=True, vmin=-0.5, vmax=0.5,
                          annot=True, fmt=".3f", annot_kws={"size": 12, "weight": "bold"},  
                          linewidths=0.5, linecolor='black', 
                          cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8}, xticklabels= 1, yticklabels= 1)

        ax.set_xlabel('Peak Separation', fontsize=20)
        ax.set_ylabel('Peak Width', fontsize=20)
        ax.set_title(title, fontsize=20, pad=20)
        x_ticks = np.round(x_ticks, 4)
        y_ticks = np.round(y_ticks, 4)
        # Set x and y ticks
        ax.set_xticklabels(x_ticks, rotation=-90, fontsize=14)
        ax.set_yticklabels(y_ticks, fontsize=14)
        # Ensure y-axis is visible
        ax.yaxis.set_visible(True)
        # Get the colorbar from the heatmap
        cbar = im.collections[0].colorbar
        cbar.set_ticks([-0.5, 0, 0.5])
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=16)

    # Add heatmaps for each method
    add_heatmap(axs[0], compare_GCV, tick_labels[0], 'LocReg Error - GCV Error (Rel. L2 Norm)', x_ticks, y_ticks)
    add_heatmap(axs[1], compare_DP, tick_labels[1], 'LocReg Error - DP Error (Rel. L2 Norm)', x_ticks, y_ticks)
    add_heatmap(axs[2], compare_LC, tick_labels[2], 'LocReg Error - L-Curve Error (Rel. L2 Norm)', x_ticks, y_ticks)
    add_heatmap(axs[3], compare_oracle, tick_labels[3], 'LocReg Error - Oracle Error (Rel. L2 Norm)', x_ticks, y_ticks)

    # Ensure y-axis labels show for all plots
    for ax in axs:
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        # if ax != axs[2] and ax != axs[3]:  # Only show y-ticks for the left plots
        # ax.set_yticks(np.arange(len(y_ticks)))  # Set the ticks for all axes
        # ax.set_yticklabels(y_ticks, fontsize=14)  # Set the tick labels for all axes
        ax.tick_params(labelleft=True)
        # else:
        #     ax.yaxis.set_visible(False)
    # Optimize layout to remove whitespace
    plt.tight_layout()
    plt.savefig(os.path.join(file_path_final, "compare_heatmap_poster3by3.png"))
    print("Saved Comparison Heatmap")
    plt.close()

compare_heatmap()
