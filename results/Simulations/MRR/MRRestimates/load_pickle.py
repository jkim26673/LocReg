import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#High SNR
datafile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-19_SNR_10_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_10_errtype_Wass. Score/est_table_SNR10_iter10_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_10_errtype_Wass. Score_19Nov24.pkl"
with open(datafile, 'rb') as file:
    errors = pickle.load(file)

if errors.isna().any().any():
    print("The DataFrame has NA values.")
else:
    print("The DataFrame does not have any NA values.")
#chuans version

datafile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/avg_error_vals_20240813_SNR_1000_lamini_LCurve_show_1.pkl"
with open(datafile, 'rb') as file:
    errors = pickle.load(file)

LRerr = errors["avg_MDL_err_LocReg_LC"]

picklefile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_20240814_SNR_1000_sol_struct_uneql2_noshow.pkl"
with open(picklefile, 'rb') as file:
    solstr = pickle.load(file)
    ndo = solstr["noisy_data"][0]

noisyfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_14Aug24noisydata.npy"
noisy = np.load(noisyfile)

noiselessfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_14Aug24noiselessdata.npy"
noiseless = np.load(noiselessfile)


# datafile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/avg_error_vals_20240813_SNR_1000_lamini_LCurve_show_1.pkl"

# datafile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter5_lamini_LCurve_dist_broadL_narrowR_21Aug24.pkl"
with open(datafile, 'rb') as file:
    errors = pickle.load(file)
    
errs_oracle = errors["avg_MDL_err_oracle"]
errs_LC= errors["avg_MDL_err_LC"]
errs_GCV = errors["avg_MDL_err_GCV"]
errs_DP = errors["avg_MDL_err_DP"]
errs_LR = errors["avg_MDL_err_LocReg_LC"]

avg_oracle = np.mean(errs_oracle)
avg_LC = np.mean(errs_LC)
avg_GCV = np.mean(errs_GCV)
avg_DP = np.mean(errs_DP)
avg_LR = np.mean(errs_LR)

print("avg_oracle",avg_oracle)
print("avg_LC",avg_LC)
print("avg_GCV",avg_GCV)
print("avg_DP",avg_DP)
print("avg_LR",avg_LR)

# datafile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_13Aug24.pkl"
# with open(datafile, 'rb') as file:
#     errors = pickle.load(file)

# datafile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter10_lamini_LCurve_dist_broadL_narrowR_15Aug24.pkl"
datafile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_15Aug24_modifiedalgo.pkl"
import pickle
import numpy as np
import pandas as pd

# origfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_orig_algo_15Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_mod_algo_15Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_ep_min_1e_minus1_15Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_longfeedback_80_15Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_gammainit_0.1_15Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_gammainit_0.001_15Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_gamma_init0.5_15Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR50_iter1_lamini_LCurve_dist_narrowL_broadR_15Aug24.pkl"

# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter10_lamini_LCurve_dist_broadL_narrowR_20Aug24.pkl"

# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter5_lamini_LCurve_dist_broadL_narrowR_21Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_narrowL_broadR_21Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter10_lamini_LCurve_dist_narrowL_broadR_22Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR200_iter10_lamini_LCurve_dist_narrowL_broadR_22Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR50_iter10_lamini_LCurve_dist_narrowL_broadR_22Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter10_lamini_LCurve_dist_broadL_narrowR_22Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_testingagain_23Aug24_everyiteration1.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_testingagain_23Aug24_every5.pkl"

# narrowL,broadR
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter10_lamini_LCurve_dist_narrowL_broadR_22Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR300_iter10_lamini_LCurve_dist_narrowL_broadR_26Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR50_iter10_lamini_LCurve_dist_narrowL_broadR_22Aug24.pkl"

#broadL, narrow R
# modfile = "SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter10_lamini_LCurve_dist_broadL_narrowR_22Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR300_iter10_lamini_LCurve_dist_broadL_narrowR_23Aug24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR50_iter10_lamini_LCurve_dist_broadL_narrowR_26Aug24.pkl"

# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR300_iter10_lamini_LCurve_dist_rightsingledist_03Sep24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_narrowL_broadR_11Sep24wass.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_narrowL_broadR_11Sep24L2.pkl"


# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR100_iter20_lamini_LCurve_dist_narrowL_broadR_fastLR_01Oct24L2.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR100_iter100_lamini_LCurve_dist_narrowL_broadR_18Sep24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR100_iter20_lamini_LCurve_dist_narrowL_broadR_fastLR_01Oct24wass.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter50_lamini_LCurve_dist_narrowL_broadR_SNR1000_final_09Oct24.pkl"

import pickle
import pandas as pd
modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000/est_table_SNR1000_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_18Oct24.pkl"

#3 noiseR
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter3_lamini_LCurve_dist_narrowL_broadR_testing_10Oct24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter2_lamini_LCurve_dist_narrowL_broadR_testing2_10Oct24.pkl"
# # modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_narrowL_broadR_testing_10Oct24.pkl"
# datafile = modfile
# with open(datafile, 'rb') as file:
#     errors = pickle.load(file)

# errors = errors.drop(["DP_vect", "GCV_vect","LR_vect", "oracle_vect", "LC_vect"], axis = 1)
# df = pd.DataFrame(errors)

# df['Sigma'] = df['Sigma'].apply(tuple)
# df_sorted = df.sort_values(by=['NR','Sigma', 'RPS_val'], ascending=[True, True, True])
# # df2 = df_sorted.drop(["peak_resol"], axis = 1)
# # df2 = df2.drop(["LR_vect"], axis = 1)
# # dfNR0 = df2.loc[df2["NR"] == 0]
# # dfNR1 = df2.loc[df2["NR"] == 1]


# errors = errors.drop(["LR_vect","LR_vect"], axis = 1)
# df = pd.DataFrame(errors)

# df['Sigma'] = df['Sigma'].apply(tuple)
# df_sorted = df.sort_values(by=['NR','Sigma', 'RPS_val'], ascending=[True, True, True])
# df2 = df_sorted.drop(["peak_resol"], axis = 1)
# df2 = df2.drop(["LR_vect"], axis = 1)
# dfNR0 = df2.loc[df2["NR"] == 0]
# dfNR1 = df2.loc[df2["NR"] == 1]
# # np.median(df["err_oracle"])
# grouped = df_sorted.groupby(['Sigma', 'RPS_val']).agg({
#     'err_DP': 'sum',
#     'err_LC': 'sum',
#     'err_LR': 'sum',
#     'err_GCV': 'sum',
#     'err_oracle': 'sum'
# })

import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
nsigma = 3
nrps = 3

# Load the data
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-26_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim2_SNR_1000_errtype_Wass. Scoretesting/est_table_SNR1000_iter2_lamini_LCurve_dist_narrowL_broadR_parallel_nsim2_SNR_1000_errtype_Wass. Scoretesting_26Oct24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-26_SNR_50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim1_SNR_50_errtype_Wass. Score/est_table_SNR50_iter1_lamini_LCurve_dist_broadL_narrowR_parallel_nsim1_SNR_50_errtype_Wass. Score_26Oct24.pkl"
modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-19_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_300_errtype_Wass. Score/est_table_SNR300_iter1_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_300_errtype_Wass. Score_19Nov24.pkl"
datafile = modfile
with open(datafile, 'rb') as file:
    errors = pickle.load(file)

# Drop unwanted columns
errors = errors.drop(["DP_vect", "GCV_vect", "LR_vect", "oracle_vect", "LC_vect"], axis=1)
df = pd.DataFrame(errors)

# Convert Sigma values to tuples
df['Sigma'] = df['Sigma'].apply(tuple)

# Define values to keep
sigma_values_to_keep = [(2.0, 6.0), (2.75, 8.25), (3.5, 10.5)]
rps_values_to_keep = [1.100, 1.825, 2.550]
unif_sigma = [2.0, 2.75, 3.5]
rps = [1.100, 1.825, 2.550]
# Apply the filter to keep only the specified values
df_filtered = df[df['Sigma'].isin(sigma_values_to_keep) & df['RPS_val'].isin(rps_values_to_keep)]

# Sort the filtered DataFrame
df_sorted = df_filtered.sort_values(by=['NR', 'Sigma', 'RPS_val'], ascending=[True, True, True])

# Optional: If you want to drop specific columns, uncomment these lines
# df_sorted = df_sorted.drop(["peak_resol"], axis=1)
# df_sorted = df_sorted.drop(["LR_vect"], axis=1)

# Filter for NR values if needed
# dfNR0 = df_sorted.loc[df_sorted["NR"] == 0]
# dfNR1 = df_sorted.loc[df_sorted["NR"] == 1]


# Calculate the number of unique NR values
num_NRs = df_sorted['NR'].nunique()
grouped = df_sorted.groupby(['Sigma', 'RPS_val']).agg({
    'err_DP': 'sum',
    'err_LC': 'sum',
    'err_LR': 'sum',
    'err_GCV': 'sum',
    'err_oracle': 'sum'
})
# Average the errors
average_errors = grouped / num_NRs

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

file_path_final = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000"
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
    x_ticks = rps
    y_ticks = unif_sigma

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
    # Save the figure
    plt.savefig(os.path.join(file_path_final, f"compare_heatmap_poster3by3.png"))
    print("Saved Comparison Heatmap")
    plt.close()
compare_heatmap()




test = average_errors
test["Error LR-DP"] = average_errors["err_LR"] - average_errors["err_DP"]
test["Error LR-oracle"] = average_errors["err_LR"] - average_errors["err_oracle"]
test["Error LR-GCV"] = average_errors["err_LR"] - average_errors["err_GCV"]
test["Error LR-LC"] = average_errors["err_LR"] - average_errors["err_LC"]

heatmaperror = test.drop(['err_LR', 'err_DP',"err_oracle", "err_GCV", "err_LC"], axis=1)

print("heatmap data_table")
print(heatmaperror)

errors = average_errors

# errors = df.groupby('Sigma').agg({
#     'NR': 'sum',
#     'err_DP': 'sum',
#     'err_LC': 'sum',
#     'err_LR': 'sum',
#     'err_GCV': 'sum',
#     'err_oracle': 'sum'
#     }).reset_index()

errs_oracle = errors["err_oracle"]
errs_LC= errors["err_LC"]
errs_GCV = errors["err_GCV"]
errs_DP = errors["err_DP"]
errs_LR = errors["err_LR"]
# nsim = errors["NR"].iloc[-1] + 1

# grouped = df.groupby('NR').sum()
# print(grouped)

avg_oracle = np.mean(errs_oracle)
avg_LC = np.mean(errs_LC)
avg_GCV = np.mean(errs_GCV)
avg_DP = np.mean(errs_DP)
avg_LR = np.mean(errs_LR)

print("avg_oracle",avg_oracle)
print("avg_LC",avg_LC)
print("avg_GCV",avg_GCV)
print("avg_DP",avg_DP)
print("avg_LR",avg_LR)



nsigma = 5
rps = np.linspace(1, 4, 5).T
# errors = average_errors

df['Sigma'] = df['Sigma'].apply(tuple)
df_sorted = df.sort_values(by=['NR','Sigma', 'RPS_val'], ascending=[True, True, True])
print("df_sorted", df_sorted)

grouped = df_sorted.groupby(['Sigma', 'RPS_val']).agg({
    'err_DP': 'sum',
    'err_LC': 'sum',
    'err_LR': 'sum',
    'err_GCV': 'sum',
    'err_oracle': 'sum',
    # "shift_err_DP": 'sum',
    # "shift_err_LC": 'sum',
    # "shift_err_LR": 'sum',
    # "shift_err_GCV": 'sum',
    # "shift_err_oracle": 'sum'
})
num_NRs = df_sorted['NR'].nunique()
# Average the errors
average_errors = grouped / num_NRs
errors = average_errors

print(grouped)

n = nsigma
m = len(rps)

# errors = df.groupby('Sigma').agg({
#     'NR': 'sum',
#     'err_DP': 'sum',
#     'err_LC': 'sum',
#     'err_LR': 'sum',
#     'err_GCV': 'sum',
#     'err_oracle': 'sum'
#     }).reset_index()


errs_oracle = errors["err_oracle"].to_numpy().reshape(n,m)
errs_oracle = np.array(errs_oracle)
errs_LC= errors["err_LC"].to_numpy().reshape(n,m)
errs_LC = np.array(errs_LC)
errs_GCV = errors["err_GCV"].to_numpy().reshape(n,m)
errs_GCV = np.array(errs_GCV)
errs_DP = errors["err_DP"].to_numpy().reshape(n,m)
errs_DP = np.array(errs_DP)
errs_LR = errors["err_LR"].to_numpy().reshape(n,m)
errs_LR = np.array(errs_LR)

# avg_oracle = np.mean(errs_oracle)
# avg_LC = np.mean(errs_LC)
# avg_GCV = np.mean(errs_GCV)
# avg_DP = np.mean(errs_DP)
# avg_LR = np.mean(errs_LR)
# print(errs_LR)
# compare_GCV = errs_LR - errs_GCV
# # print("compare_GCV", compare_GCV)
# compare_DP = errs_LR - errs_DP
# compare_LC = errs_LR - errs_LC
# compare_oracle = errs_LR - errs_oracle

# avg_oracle = np.mean(errs_oracle)
# avg_LC = np.mean(errs_LC)
# avg_GCV = np.mean(errs_GCV)
# avg_DP = np.mean(errs_DP)
# avg_LR = np.mean(errs_LR)
# print(errs_LR)
compare_GCV = errs_LR - errs_GCV
# print("compare_GCV", compare_GCV)
compare_DP = errs_LR - errs_DP
compare_LC = errs_LR - errs_LC
compare_oracle = errs_LR - errs_oracle

show = 1
nsigma = 5
unif_sigma = np.linspace(2, 5, nsigma).T
# file_name = "SNR100_iter100_lamini_LCurve_dist_narrowL_broadR_18Sep24"
# file_name = "SNR100_iter100_lamini_LCurve_dist_narrowL_broadR_18Sep24"
# file_name = "SNR100_iter20_lamini_LCurve_dist_narrowL_broadR_fastLR_01Oct24L2"
# file_name = "SNR100_iter20_lamini_LCurve_dist_narrowL_broadR_fastLR_01Oct24L2"
file_name = "SNR100_iter20_lamini_LCurve_dist_narrowL_broadR_fastLR_01Oct24wass"
def add_custom_colorbar(ax, im, title):
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(title, fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    # Define custom ticks and labels
    ticks = [-0.5, 0, 0.5]
    tick_labels = ['LocReg is better', 'Neutral', 'LocReg is worse']
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    return cbar
# feature_df['IdealModel_data'] = feature_df
y_min, y_max = unif_sigma[0], unif_sigma[-1]
if y_min == y_max:
    y_max += 0.01  # Small offset

plt.figure(figsize=(8, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.8)
plt.subplot(4, 1, 1)
plt.imshow(compare_GCV, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
plt.colorbar()
plt.xlabel('Ratio of Peak Separation', fontsize=18)
plt.ylabel('Gaussian Sigma', fontsize=18)
plt.title('LocReg L2 Error - GCV L2 Error', fontsize=18)
plt.clim(-0.5, 0.5)

plt.subplot(4, 1, 2)
plt.imshow(compare_DP, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
plt.colorbar()
plt.xlabel('Ratio of Peak Separation', fontsize=18)
plt.ylabel('Gaussian Sigma', fontsize=18)
plt.title('LocReg L2 Error - DP L2 Error', fontsize=18)
plt.clim(-0.5, 0.5)

plt.subplot(4, 1, 3)
plt.imshow(compare_LC, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
plt.colorbar()
plt.xlabel('Ratio of Peak Separation', fontsize=18)
plt.ylabel('Gaussian Sigma', fontsize=18)
plt.title('LocReg L2 Error - L-Curve L2 Error', fontsize=18)
plt.clim(-0.5, 0.5)

plt.subplot(4, 1, 4)
plt.imshow(compare_oracle, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
plt.colorbar()
plt.xlabel('Ratio of Peak Separation', fontsize=18)
plt.ylabel('Gaussian Sigma', fontsize=18)
plt.title('LocReg L2 Error - Oracle L2 Error', fontsize=18)
plt.clim(-0.5, 0.5)




plt.figure(figsize=(8, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.8)
plt.subplot(4, 1, 1)
plt.imshow(compare_GCV, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
plt.colorbar()
plt.xlabel('Ratio of Peak Separation', fontsize=18)
plt.ylabel('Gaussian Sigma', fontsize=18)
plt.title('LocReg Wassterstein Error - GCV Wassterstein Error', fontsize=13)
plt.clim(-0.5, 0.5)

plt.subplot(4, 1, 2)
plt.imshow(compare_DP, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
plt.colorbar()
plt.xlabel('Ratio of Peak Separation', fontsize=18)
plt.ylabel('Gaussian Sigma', fontsize=18)
plt.title('LocReg Wassterstein Error - DP Wassterstein Error', fontsize=13)
plt.clim(-0.5, 0.5)

plt.subplot(4, 1, 3)
plt.imshow(compare_LC, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
plt.colorbar()
plt.xlabel('Ratio of Peak Separation', fontsize=18)
plt.ylabel('Gaussian Sigma', fontsize=18)
plt.title('LocReg Wassterstein Error - L-Curve Wassterstein Error', fontsize=13)
plt.clim(-0.5, 0.5)

plt.subplot(4, 1, 4)
plt.imshow(compare_oracle, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
plt.colorbar()
plt.xlabel('Ratio of Peak Separation', fontsize=18)
plt.ylabel('Gaussian Sigma', fontsize=18)
plt.title('LocReg Wassterstein Error - Oracle Wassterstein Error', fontsize=13)
plt.clim(-0.5, 0.5)

# plt.subplot(4, 1, 4)
# plt.imshow(compare_LocReg, cmap='jet', aspect='auto', extent=[rps[0], rps[-1], unif_sigma[0], unif_sigma[-1]])
# plt.colorbar()
# plt.xlabel('Ratio of Peak Separation', fontsize=18)
# plt.ylabel('Gaussian Sigma', fontsize=18)
# plt.title('Comparison with LocReg using LC lam', fontsize=18)
# plt.clim(-0.5, 0.5)
# plt.savefig('heatmap.png')
string = "Heatmap"
plt.savefig(f"{file_name}_heatmap.png")
print(f"Saved Heatmap")
plt.close()


















#Shifted Lambdas:

modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_narrowL_broadR_11Sep24wass.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_narrowL_broadR_11Sep24L2.pkl"
datafile = modfile
with open(datafile, 'rb') as file:
    errors = pickle.load(file)

df = pd.DataFrame(errors)
df['Sigma'] = df['Sigma'].apply(tuple)
df_sorted = df.sort_values(by=['NR','Sigma', 'RPS_val'], ascending=[True, True, True])

grouped = df_sorted.groupby(['Sigma', 'RPS_val']).agg({
    'err_DP': 'sum',
    'err_LC': 'sum',
    'err_LR': 'sum',
    'err_GCV': 'sum',
    'err_oracle': 'sum',
    # "shift_err_DP": 'sum',
    # "shift_err_LC": 'sum',
    # "shift_err_LR": 'sum',
    # "shift_err_GCV": 'sum',
    # "shift_err_oracle": 'sum'
})

# grouped = df_sorted.groupby(['Sigma', 'RPS_val']).agg({
#     # 'err_DP': 'sum',
#     # 'err_LC': 'sum',
#     # 'err_LR': 'sum',
#     # 'err_GCV': 'sum',
#     # 'err_oracle': 'sum',
#     "shift_err_DP": 'sum',
#     "shift_err_LC": 'sum',
#     "shift_err_LR": 'sum',
#     "shift_err_GCV": 'sum',
#     "shift_err_oracle": 'sum'
# })

# grouped.rename(columns={'shift_err_DP': 'err_DP', 'shift_err_LC': 'err_LC', 
#                                   'shift_err_LR': 'err_LR', 'shift_err_GCV': 'err_GCV', 'shift_err_oracle': 'err_oracle'}, inplace=True)

# Calculate the number of unique NR values
num_NRs = df_sorted['NR'].nunique()

# Average the errors
average_errors = grouped / num_NRs

print(average_errors)

test = average_errors
test["Error LR-DP"] = average_errors["err_LR"] - average_errors["err_DP"]
test["Error LR-oracle"] = average_errors["err_LR"] - average_errors["err_oracle"]
test["Error LR-GCV"] = average_errors["err_LR"] - average_errors["err_GCV"]
test["Error LR-LC"] = average_errors["err_LR"] - average_errors["err_LC"]

heatmaperror = test.drop(['err_LR', 'err_DP',"err_oracle", "err_GCV", "err_LC"], axis=1)

print("heatmap data_table")
print(heatmaperror)

errors = average_errors

# errors = df.groupby('Sigma').agg({
#     'NR': 'sum',
#     'err_DP': 'sum',
#     'err_LC': 'sum',
#     'err_LR': 'sum',
#     'err_GCV': 'sum',
#     'err_oracle': 'sum'
#     }).reset_index()

errs_oracle = errors["err_oracle"]
errs_LC= errors["err_LC"]
errs_GCV = errors["err_GCV"]
errs_DP = errors["err_DP"]
errs_LR = errors["err_LR"]
# nsim = errors["NR"].iloc[-1] + 1

# grouped = df.groupby('NR').sum()
# print(grouped)

avg_oracle = np.mean(errs_oracle)
avg_LC = np.mean(errs_LC)
avg_GCV = np.mean(errs_GCV)
avg_DP = np.mean(errs_DP)
avg_LR = np.mean(errs_LR)

print("avg_oracle",avg_oracle)
print("avg_LC",avg_LC)
print("avg_GCV",avg_GCV)
print("avg_DP",avg_DP)
print("avg_LR",avg_LR)



z_oracle = np.abs(errs_oracle - np.mean(errs_oracle))/(np.std(errs_oracle))
z_LC= np.abs(errs_LC - np.mean(errs_LC))/(np.std(errs_LC))
z_GCV = np.abs(errs_GCV - np.mean(errs_GCV))/(np.std(errs_GCV))
z_DP = np.abs(errs_DP - np.mean(errs_DP))/(np.std(errs_DP))
z_LR= np.abs(errs_LR - np.mean(errs_LR))/(np.std(errs_LR))

zscorelist = [z_oracle, z_LC,z_GCV,z_DP,z_LR]

threshold = 3

maskpositions = []
# Iterate over each array in the zscorelist
for i, array in enumerate(zscorelist):
    # Convert to a numpy array if it's not already
    array = np.array(array)
    # Count how many values exceed the threshold
    count_exceeding = np.sum(array > threshold)

    mask = array > threshold
    maskpositions.append(mask)
    # Print the result
    print(f"Array {i} has {count_exceeding} outliers.")

errorlist = [errs_oracle,errs_LC,errs_GCV,errs_DP,errs_LR]
newerrlist = []
for i,array in enumerate(errorlist):
    mask = maskpositions[i]
    array = np.delete(array, mask)
    newerrlist.append(array)

neworacle = newerrlist[0]
newLC = newerrlist[1]
newGCV = newerrlist[2]
newDP = newerrlist[3]
newLR = newerrlist[4]

avg_oracle = np.median(neworacle)
avg_LC = np.median(newLC)
avg_GCV = np.median(newGCV)
avg_DP = np.median(newDP)
avg_LR = np.median(newLR)

print("avg_oracle",avg_oracle)
print("avg_LC",avg_LC)
print("avg_GCV",avg_GCV)
print("avg_DP",avg_DP)
print("avg_LR",avg_LR)



# with open("SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_06Aug24noise_arr.txt.npy", 'rb') as file:
#     errors = pickle.load(file)
# #low snr
# with open("/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/avg_error_vals_20240725_SNR_50_lamini_LCurve_show_1_20nrun.pkl", 'rb') as file:
#     errors = pickle.load(file)


#Low SNR
# import numpy as np
# arr = np.load("/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_13Aug24noise_arr.txt.npy")
# print(arr.shape)