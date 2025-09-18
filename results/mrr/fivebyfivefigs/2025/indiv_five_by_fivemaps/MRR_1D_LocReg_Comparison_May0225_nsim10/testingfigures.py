# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import itertools
# from scipy.stats import norm as normsci
# import pandas as pd
# filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\est_table_SNR1000_iter10_lamini_GCV_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_WassScore_compare1st2ndDeriv_UPEN_02May25.pkl"
# with open(filepath, 'rb') as file:
#     df = pickle.load(file)

# savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\savefolder"
# uniqueNRs = np.unique(df["NR"].tolist()) 
# uniquesig = np.array(sorted(set(tuple(sigma) for sigma in df["Sigma"])))
# uniqueRPS = np.unique(sorted(df["RPS_val"]))
# # solution_list= ["GCV_vect", "LR_vect", "LR_vect_1stDer", "LR_vect_2ndDer", "upen_vect"]
# # lam_list= ["GCV_lam", "LR_lam", "LR_lam_1stDer", "LR_lam_2ndDer", "upen_lam"]
# # err_list= ["err_GCV", "err_LR", "err_LR_1stDer", "err_LR_2ndDer", "err_upen"]

# solution_list= ["GCV_vect", "LR_vect", "upen_vect"]
# lam_list= ["GCV_lam", "LR_lam", "upen_lam"]
# err_list= ["err_GCV", "err_LR", "err_upen"]

# npeaks = 2
# nsigma = 5
# f_coef = np.ones(npeaks)
# rps = np.linspace(1.1, 4, nsigma).T
# nrps = len(rps)

# def calc_T2mu(rps):
#     mps = rps / 2
#     nrps = len(rps)
#     T2_left = 40 * np.ones(nrps)
#     T2_mid = T2_left * mps
#     T2_right = T2_left * rps
#     T2mu = np.column_stack((T2_left, T2_right))
#     return T2mu

# def calc_sigma_i(iter_i, diff_sigma):
#     sigma_i = diff_sigma[iter_i, :]
#     return sigma_i

# def calc_rps_val(iter_j, rps):
#     rps_val = rps[iter_j]
#     return rps_val

# def calc_diff_sigma(nsigma):
#     unif_sigma = np.linspace(2, 5, nsigma).T
#     diff_sigma = np.column_stack((unif_sigma, 3 *unif_sigma))
#     return unif_sigma, diff_sigma

# def load_Gaus(Gaus_info):
#     n, m = Gaus_info['A'].shape
#     # T2 = Gaus_info['T2'].flatten()
#     T2 = np.linspace(10,200,m)
#     TE = Gaus_info['TE'].flatten()
#     A = Gaus_info['A']
#     # Lambda = Gaus_info['Lambda'].reshape(-1,1)
#     # Lambda = np.append(0, Lambda)
#     # Lambda = np.append(0, np.logspace(-6,-1,20)).reshape(-1,1)
#     SNR = 1000
#     return T2, TE, A, m,  SNR

# def calc_dat_noisy(A, TE, IdealModel_weighted, SNR, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     dat_noiseless = A @ IdealModel_weighted  # Compute noiseless data
#     SD_noise = np.max(np.abs(dat_noiseless)) / SNR  # Standard deviation of noise
#     noise = np.random.normal(0, SD_noise, size=dat_noiseless.shape)  # Add noise
#     dat_noisy = dat_noiseless + noise
#     return dat_noisy, noise, SD_noise

# def get_IdealModel_weighted(iter_j, m, npeaks, T2, T2mu, sigma_i):
#     p = np.zeros((npeaks, m))
#     T2mu_sim = T2mu[iter_j, :]
#     p = np.array([normsci.pdf(T2, mu, sigma) for mu, sigma in zip(T2mu_sim, sigma_i)])
#     IdealModel_weighted = (p.T @ f_coef) / npeaks
#     return IdealModel_weighted

# import seaborn as sns
# def compare_heatmap(new_df, err_list, savepath, err_type="WassScore"):
#     # Dynamically determine the number of subplots based on the length of err_list
#     n_methods = len(err_list)
#     n_rows = (n_methods + 1) // 2  # Arrange in 2 columns
#     fig, axs = plt.subplots(n_rows, 2, sharey=True, figsize=(18, 6 * n_rows))
#     plt.subplots_adjust(wspace=0.3, hspace=0.4)

#     # Define tick labels dynamically
#     tick_labels = [
#         [f'LocReg is better', 'Neutral', f'{err.split("_")[1]} is better'] for err in err_list
#     ]
#     axs = axs.flatten()
#     x_ticksval = uniqueRPS
#     y_ticksval = unif_sigma

#     # Calculate vmax and vmin for color scaling
#     vmax1 = max(new_df[err].max() for err in err_list)
#     vmin1 = -vmax1

#     # Format for scientific notation
#     fmt1 = ".2e" if err_type == "WassScore" else ".3e"
#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         im = sns.heatmap(data, cmap='jet', ax=ax, cbar=True, vmin=vmin1, vmax=vmax1,
#                         annot=True, fmt=fmt1, annot_kws={"size": 12, "weight": "bold"},
#                         linewidths=0.5, linecolor='black',
#                         cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8}, xticklabels=1, yticklabels=1)

#         ax.set_xlabel('Peak Separation', fontsize=20)
#         ax.set_ylabel('Peak Width', fontsize=20)
#         ax.set_title(title, fontsize=20, pad=20)

#         # Dynamically adjust tick labels based on data shape
#         x_ticks = np.round(x_ticks[:data.shape[1]], 4)  # Match number of columns
#         y_ticks = np.round(y_ticks[:data.shape[0]], 4)  # Match number of rows

#         ax.set_xticklabels(x_ticks, rotation=-90, fontsize=14)
#         ax.set_yticklabels(y_ticks, fontsize=14)

#         # Set colorbar ticks and labels
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([vmin1, 0, vmax1])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)

#     # Generate heatmaps for each comparison
#     for idx, err in enumerate(err_list):
#         comparison_data = new_df["err_LR"] - new_df[err]  # Replace "err_LR" with LocReg error column
#         comparison_data = np.atleast_2d(comparison_data)  # Ensure comparison_data is 2D
#         add_heatmap(axs[idx], comparison_data, tick_labels[idx],
#                     title=f'LocReg Error - {err.split("_")[1]} Error ({err_type})', x_ticks=x_ticksval, y_ticks=y_ticksval)
#         # Hide unused subplots
#     for ax in axs[len(err_list):]:
#         ax.axis('off')

#     # Save the figure
#     plt.tight_layout()
#     plt.savefig(f"{savepath}/compare_heatmap.png")
#     print("Saved Comparison Heatmap")
#     plt.close()

# def indiv_heatmap(new_df, err_list, savepath, err_type="WassScore"):
#     # Dynamically determine the number of subplots based on the length of err_list
#     n_methods = len(err_list)
#     n_rows = (n_methods + 1) // 2  # Arrange in 2 columns
#     fig, axs = plt.subplots(n_rows, 2, sharey=True, figsize=(18, 6 * n_rows))
#     plt.subplots_adjust(wspace=0.3, hspace=0.3)

#     # Define tick labels dynamically
#     tick_labels = [
#         [f'Low {err.split("_")[1]} Error', f'High {err.split("_")[1]} Error'] for err in err_list
#     ]
#     axs = axs.flatten()
#     x_ticks = uniqueRPS
#     y_ticks = unif_sigma

#     # Calculate vmax for color scaling
#     vmax1 = max(new_df[err].max() for err in err_list)

#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         fmt1 = ".4f" if err_type == "WassScore" else ".3f"
#         im = sns.heatmap(data, cmap='jet', ax=ax, cbar=True, vmin=0, vmax=vmax1,
#                          annot=True, fmt=fmt1, annot_kws={"size": 12, "weight": "bold"},
#                          linewidths=0.5, linecolor='black',
#                          cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8}, xticklabels=1, yticklabels=1)

#         ax.set_xlabel('Peak Separation', fontsize=20)
#         ax.set_ylabel('Peak Width', fontsize=20)
#         ax.set_title(title, fontsize=20, pad=20)

#         # Dynamically adjust tick labels based on data shape
#         x_ticks = np.round(x_ticks[:data.shape[1]], 4) if data.shape[1] > 0 else []
#         y_ticks = np.round(y_ticks[:data.shape[0]], 4) if data.shape[0] > 0 else []

#         ax.set_xticklabels(x_ticks, rotation=-90, fontsize=14)
#         ax.set_yticklabels(y_ticks, fontsize=14)

#         # Set colorbar ticks and labels
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([0, vmax1])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)

#     # Generate heatmaps for each error
#     for idx, err in enumerate(err_list):
#         data = np.atleast_2d(new_df[err])  # Ensure data is 2D
#         add_heatmap(axs[idx], data, tick_labels[idx],
#                     title=f'{err.split("_")[1]} Error ({err_type})', x_ticks=x_ticks, y_ticks=y_ticks)

#     # Hide unused subplots
#     for ax in axs[len(err_list):]:
#         ax.axis('off')

#     # Save the figure
#     plt.tight_layout()
#     plt.savefig(f"{savepath}/indiv_heatmap.png")
#     print("Saved Individual Heatmap")
#     plt.close()

# unif_sigma, diff_sigma = calc_diff_sigma(nsigma)
# file_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\Simulations\num_of_basis_functions\lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_6_basis2_40110lmbda_min-6lmbda_max008Oct24.pkl"
# # Load the Gaussian data
# Gaus_info = np.load(file_path, allow_pickle=True)
# T2, TE, A, m,  SNR = load_Gaus(Gaus_info)
# print("TE",TE)
# n_sim = 10

# T2mu = calc_T2mu(rps)
# lis = []
# gtdf = pd.DataFrame()
# for i in range(n_sim):
#     for j in range(nsigma):
#         for k in range(nrps):
#             iter = (i,j,k)
#             iter_sim, iter_sigma, iter_rps = iter 
#             sigma_i = diff_sigma[iter_sigma, :]
#             rps_val = calc_rps_val(iter_rps, rps)
#             # Generate Ground Truth and add random noise or specific noise array
#             IdealModel_weighted = get_IdealModel_weighted(iter_rps, m, npeaks, T2, T2mu, sigma_i)
#             lis.append({
#                 "gt": IdealModel_weighted,
#                 "NR": iter_sim,
#                 "iter_sigma": iter_sigma,
#                 "iter_rps": iter_rps,
#                 "Sigma": sigma_i.tolist(),  # Convert to list for JSON compatibility
#                 "RPS_val": rps_val
#             })
# gtdf = pd.DataFrame(lis)

# sig_to_index = {tuple(sig): idx for idx, sig in enumerate(uniquesig)}
# rps_to_index = {(rps): idx for idx, rps in enumerate(uniqueRPS)}
# T2axis = np.linspace(10,200,150)
# for NR in uniqueNRs:
#     for (sig, RPS) in itertools.product(uniquesig, uniqueRPS):
#         filt_df = df[df["NR"] == 0]
#         gtdffilt = gtdf[gtdf["NR"] == 0]
#         newgtdf = gtdffilt.loc[(gtdffilt["RPS_val"] == RPS) & (gtdffilt["Sigma"].apply(lambda sigma: np.array_equal(sigma, sig)))]
#         new_df = filt_df.loc[(filt_df["RPS_val"] == RPS) & (filt_df["Sigma"].apply(lambda sigma: np.array_equal(sigma, sig)))]
#         plt.figure()
#         for str_idx, str in enumerate(solution_list):  # Use enumerate to get the index
#             label = (str.replace("vect", "")
#                         .replace("LR", "LocReg")
#                         .replace("1stDer", "1st Derivative")
#                         .replace("2ndDer", "2nd Derivative")
#                         .replace("_", " ")
#                         .replace('upen', "UPEN"))
#             # Attach the corresponding value from err_list
#             error_value = new_df[err_list[str_idx]].iloc[0]
#             formatted_error = f"{error_value:.2e}"  # Format in scientific notation (e.g., 1.23e-03)
#             # Check if this is the smallest error value
#             smallest_error = min(new_df[err].iloc[0] for err in err_list)
#             star = "*" if error_value == smallest_error else ""
#         # Attach the formatted error and star to the label
#             # label += f" ({formatted_error}{star})"            
#             plt.plot(T2axis, new_df[str].iloc[0], label = label)
#         plt.plot(T2axis,newgtdf["gt"].iloc[0], label = "Ground Truth", color = "black")
#         plt.legend()
#         plt.xlabel("T2 Axis (ms)")
#         plt.ylabel("Amplitude")
#         sigma_index = sig_to_index[tuple(sig)]
#         rps_index = rps_to_index[RPS]
#         plt.savefig(f"{savepath}\Simulation{0}_Sigma{sigma_index}_RPS{rps_index}.png")
#         plt.close()

# compare_heatmap(new_df, err_list, savepath, err_type="WassScore")
# indiv_heatmap(new_df, err_list, savepath, err_type="WassScore")

# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import itertools
# import os
# from scipy.stats import norm as normsci
# import pandas as pd
# import seaborn as sns

# # Debug flag
# debug = False

# # File paths
# filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\est_table_SNR1000_iter10_lamini_GCV_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_WassScore_compare1st2ndDeriv_UPEN_02May25.pkl"
# savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\savefolder"

# # Load the data
# with open(filepath, 'rb') as file:
#     df = pickle.load(file)

# # Extract unique values
# uniqueNRs = np.unique(df["NR"].tolist())
# uniquesig = np.array(sorted(set(tuple(sigma) for sigma in df["Sigma"])))
# uniqueRPS = np.unique(sorted(df["RPS_val"]))

# # Define lists for solutions, lambdas, and errors
# solution_list = ["GCV_vect", "LR_vect", "upen_vect"]
# lam_list = ["GCV_lam", "LR_lam", "upen_lam"]
# err_list = ["err_GCV", "err_LR", "err_upen"]

# # Simulation parameters
# npeaks = 2
# nsigma = 5
# f_coef = np.ones(npeaks)
# rps = np.linspace(1.1, 4, nsigma).T
# nrps = len(rps)

# def calc_T2mu(rps):
#     mps = rps / 2
#     nrps = len(rps)
#     T2_left = 40 * np.ones(nrps)
#     T2_mid = T2_left * mps
#     T2_right = T2_left * rps
#     T2mu = np.column_stack((T2_left, T2_right))
#     return T2mu

# def calc_rps_val(iter_j, rps):
#     return rps[iter_j]

# def calc_diff_sigma(nsigma):
#     unif_sigma = np.linspace(2, 5, nsigma).T
#     diff_sigma = np.column_stack((unif_sigma, 3 * unif_sigma))
#     return unif_sigma, diff_sigma

# def load_Gaus(Gaus_info):
#     n, m = Gaus_info['A'].shape
#     T2 = np.linspace(10, 200, m)
#     TE = Gaus_info['TE'].flatten()
#     A = Gaus_info['A']
#     SNR = 1000
#     return T2, TE, A, m, SNR

# def calc_dat_noisy(A, TE, IdealModel_weighted, SNR, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     dat_noiseless = A @ IdealModel_weighted
#     SD_noise = np.max(np.abs(dat_noiseless)) / SNR
#     noise = np.random.normal(0, SD_noise, size=dat_noiseless.shape)
#     dat_noisy = dat_noiseless + noise
#     return dat_noisy, noise, SD_noise

# def get_IdealModel_weighted(iter_j, m, npeaks, T2, T2mu, sigma_i):
#     T2mu_sim = T2mu[iter_j, :]
#     p = np.array([normsci.pdf(T2, mu, sigma) for mu, sigma in zip(T2mu_sim, sigma_i)])
#     IdealModel_weighted = (p.T @ f_coef) / npeaks
#     return IdealModel_weighted

# def preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list):
#     df_copy = df.copy()
#     df_copy['Sigma_tuple'] = df_copy['Sigma'].apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)
    
#     if debug:
#         print("Unique Sigma values:", df_copy['Sigma_tuple'].unique())

#     err_grids = {err: np.full((len(unif_sigma), len(uniqueRPS)), np.nan) for err in err_list}
#     sigma_to_index = {sig: idx for idx, sig in enumerate(unif_sigma)}
#     rps_to_index = {rps: idx for idx, rps in enumerate(uniqueRPS)}

#     grouped = df_copy.groupby(['Sigma_tuple', 'RPS_val'])[err_list].mean().reset_index()

#     if debug:
#         print("Grouped DataFrame shape:", grouped.shape)
#         print("Grouped DataFrame head:\n", grouped.head())

#     for _, row in grouped.iterrows():
#         sigma = row['Sigma_tuple'][0]
#         rps = row['RPS_val']
#         if sigma in sigma_to_index and rps in rps_to_index:
#             sigma_idx = sigma_to_index[sigma]
#             rps_idx = rps_to_index[rps]
#             for err in err_list:
#                 err_grids[err][sigma_idx, rps_idx] = row[err]

#     return err_grids

# def compare_heatmap(df, err_list, savepath, unif_sigma, uniqueRPS, err_type="WassScore"):
#     err_grids = preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list)
#     n_methods = len(err_list)
#     n_rows = (n_methods + 1) // 2
#     fig, axs = plt.subplots(n_rows, 2, sharey=True, figsize=(18, 6 * n_rows))
#     axs = axs.flatten()
#     plt.subplots_adjust(wspace=0.3, hspace=0.4)
#     tick_labels = [[f'LocReg is better', 'Neutral', f'{err.split("_")[1]} is better'] for err in err_list]
#     vmax1 = max(np.nanmax(err_grids[err]) for err in err_list)
#     vmin1 = -vmax1
#     fmt1 = ".2e" if err_type == "WassScore" else ".3e"

#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         im = sns.heatmap(
#             data, cmap='jet', ax=ax, cbar=True, vmin=vmin1, vmax=vmax1,
#             annot=True, fmt=fmt1, annot_kws={"size": 12, "weight": "bold"},
#             linewidths=0.5, linecolor='black',
#             cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8},
#             xticklabels=np.round(x_ticks, 4), yticklabels=np.round(y_ticks, 4)
#         )
#         ax.set_xlabel('Peak Separation', fontsize=20)
#         ax.set_ylabel('Peak Width', fontsize=20)
#         ax.set_title(title, fontsize=20, pad=20)
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=14)
#         ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([vmin1, 0, vmax1])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)

#     for idx, err in enumerate(err_list):
#         comparison_data = err_grids['err_LR'] - err_grids[err]
#         add_heatmap(
#             axs[idx], comparison_data, tick_labels[idx],
#             title=f'LocReg Error - {err.split("_")[1]} Error ({err_type})',
#             x_ticks=uniqueRPS, y_ticks=unif_sigma
#         )

#     for ax in axs[len(err_list):]:
#         ax.axis('off')

#     plt.tight_layout()
#     plt.savefig(os.path.join(savepath, "compare_heatmap.png"))
#     print("Saved Comparison Heatmap")
#     plt.close()

# def indiv_heatmap(df, err_list, savepath, unif_sigma, uniqueRPS, err_type="WassScore"):
#     err_grids = preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list)
#     n_methods = len(err_list)
#     n_rows = (n_methods + 1) // 2
#     fig, axs = plt.subplots(n_rows, 2, sharey=True, figsize=(18, 6 * n_rows))
#     axs = axs.flatten()
#     plt.subplots_adjust(wspace=0.3, hspace=0.3)
#     tick_labels = [[f'Low {err.split("_")[1]} Error', f'High {err.split("_")[1]} Error'] for err in err_list]
#     vmax1 = max(np.nanmax(err_grids[err]) for err in err_list)
#     fmt1 = ".4f" if err_type == "WassScore" else ".3f"

#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         im = sns.heatmap(
#             data, cmap='jet', ax=ax, cbar=True, vmin=0, vmax=vmax1,
#             annot=True, fmt=fmt1, annot_kws={"size": 12, "weight": "bold"},
#             linewidths=0.5, linecolor='black',
#             cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8},
#             xticklabels=np.round(x_ticks, 4), yticklabels=np.round(y_ticks, 4)
#         )
#         ax.set_xlabel('Peak Separation', fontsize=20)
#         ax.set_ylabel('Peak Width', fontsize=20)
#         ax.set_title(title, fontsize=20, pad=20)
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=14)
#         ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([0, vmax1])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)

#     for idx, err in enumerate(err_list):
#         add_heatmap(
#             axs[idx], err_grids[err], tick_labels[idx],
#             title=f'{err.split("_")[1]} Error ({err_type})',
#             x_ticks=uniqueRPS, y_ticks=unif_sigma
#         )

#     for ax in axs[len(err_list):]:
#         ax.axis('off')

#     plt.tight_layout()
#     plt.savefig(os.path.join(savepath, "indiv_heatmap.png"))
#     print("Saved Individual Heatmap")
#     plt.close()

# # Load Gaus data
# unif_sigma, diff_sigma = calc_diff_sigma(nsigma)
# file_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\Simulations\num_of_basis_functions\lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_6_basis2_40110lmbda_min-6lmbda_max008Oct24.pkl"
# Gaus_info = np.load(file_path, allow_pickle=True)
# T2, TE, A, m, SNR = load_Gaus(Gaus_info)

# # Ground truth simulation
# T2mu = calc_T2mu(rps)
# lis = []
# n_sim = 10

# for i in range(n_sim):
#     for j in range(nsigma):
#         for k in range(nrps):
#             sigma_i = diff_sigma[j, :]
#             rps_val = calc_rps_val(k, rps)
#             IdealModel_weighted = get_IdealModel_weighted(k, m, npeaks, T2, T2mu, sigma_i)
#             lis.append({
#                 "gt": IdealModel_weighted,
#                 "NR": i,
#                 "iter_sigma": j,
#                 "iter_rps": k,
#                 "Sigma": sigma_i.tolist(),
#                 "RPS_val": rps_val
#             })

# gtdf = pd.DataFrame(lis)

# # Plot comparisons
# sig_to_index = {tuple(sig): idx for idx, sig in enumerate(uniquesig)}
# rps_to_index = {rps: idx for idx, rps in enumerate(uniqueRPS)}
# T2axis = np.linspace(10, 200, 150)

# for NR in uniqueNRs:
#     for (sig, RPS) in itertools.product(uniquesig, uniqueRPS):
#         filt_df = df[df["NR"] == NR]
#         gtdffilt = gtdf[gtdf["NR"] == NR]
#         newgtdf = gtdffilt.loc[(gtdffilt["RPS_val"] == RPS) & (gtdffilt["Sigma"].apply(lambda sigma: np.allclose(sigma, sig)))]
#         new_df = filt_df.loc[(filt_df["RPS_val"] == RPS) & (filt_df["Sigma"].apply(lambda sigma: np.allclose(sigma, sig)))]
#         if new_df.empty or newgtdf.empty:
#             continue
#         plt.figure()
#         for str_idx, str in enumerate(solution_list):
#             label = str.replace("vect", "").replace("LR", "LocReg").replace("1stDer", "1st Derivative").replace("2ndDer", "2nd Derivative").replace("_", " ").replace("upen", "UPEN")
#             error_value = new_df[err_list[str_idx]].iloc[0]
#             plt.plot(T2axis, new_df[str].iloc[0], label=f"{label} ({error_value:.2e})")
#         plt.plot(T2axis, newgtdf["gt"].iloc[0], label="Ground Truth", color="black")
#         plt.legend()
#         plt.xlabel("T2 Axis (ms)")
#         plt.ylabel("Amplitude")
#         sigma_index = sig_to_index[tuple(sig)]
#         rps_index = rps_to_index[RPS]
#         plt.savefig(os.path.join(savepath, f"Simulation{NR}_Sigma{sigma_index}_RPS{rps_index}.png"))
#         plt.close()

# # Generate heatmaps
# compare_heatmap(df, err_list, savepath, unif_sigma, uniqueRPS, err_type="WassScore")
# indiv_heatmap(df, err_list, savepath, unif_sigma, uniqueRPS, err_type="WassScore")

import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import os
from scipy.stats import norm as normsci
import pandas as pd
import seaborn as sns

# Debug flag
debug = False

# File paths
filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\est_table_SNR1000_iter10_lamini_GCV_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_WassScore_compare1st2ndDeriv_UPEN_02May25.pkl"
# savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\savefolder_LocReg"
savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\savefolder_LocReg1D"
# savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\savefolder_LocReg2D"
# Load the data

filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May2025_nsim1\est_table_SNR1000_iter1_lamini_GCV_dist_broadL_narrowR_parallel_nsim1_SNR_1000_errtype_WassScore_compare1st2ndDeriv_UPEN_20May25.pkl"
# savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\savefolder_LocReg"
savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May2025_nsim1"
# savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\savefolder_LocReg2D"
# Load the data
filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May2025_nsim10\est_table_SNR1000_iter10_lamini_GCV_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_WassScore_compare1st2ndDeriv_UPEN_20May25.pkl"
savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May2025_nsim10"

filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-22_nsim1\est_table_SNR1000_iter1_22May25.pkl"
savepath =r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-22_nsim1"

filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-22_nsim2\est_table_SNR1000_iter2_22May25.pkl"
savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-22_nsim2"

filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-23_nsim2\est_table_SNR1000_iter2_23May25.pkl"
savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-23_nsim2"

filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-23_nsim10_SNR1000_1\est_table_SNR1000_iter10_23May25.pkl"
# savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10"
savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-23_nsim10_SNR1000_1"

# filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-23_nsim4_SNR1000\est_table_SNR1000_iter4_23May25.pkl"
# savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-23_nsim4_SNR1000"
with open(filepath, 'rb') as file:
    df = pickle.load(file)

# Extract unique values
uniqueNRs = np.unique(df["NR"].tolist())
uniquesig = np.array(sorted(set(tuple(sigma) for sigma in df["Sigma"])) )
uniqueRPS = np.unique(sorted(df["RPS_val"]))
n_sim = len(uniqueNRs)

# Define lists for solutions, lambdas, and errors
solution_list = ["GCV_vect", "LR_vect", "upen_vect", "upen_vect1D", "upen_vect0D"]
lam_list = ["GCV_lam", "LR_lam", "upen_lam", "upen_lam1D", "upen_lam0D"]
err_list = ["err_GCV", "err_LR", "err_upen", "err_upen1D","err_upen0D"]

# Simulation parameters
npeaks = 2
nsigma = 5
f_coef = np.ones(npeaks)
rps = np.linspace(1.1, 4, nsigma).T
nrps = len(rps)

def calc_T2mu(rps):
    mps = rps / 2
    nrps = len(rps)
    T2_left = 40 * np.ones(nrps)
    T2_mid = T2_left * mps
    T2_right = T2_left * rps
    T2mu = np.column_stack((T2_left, T2_right))
    return T2mu

def calc_rps_val(iter_j, rps):
    return rps[iter_j]

def calc_diff_sigma(nsigma):
    unif_sigma = np.linspace(2, 5, nsigma).T
    diff_sigma = np.column_stack((unif_sigma, 3 * unif_sigma))
    # diff_sigma = np.column_stack((3*unif_sigma, unif_sigma))
    return unif_sigma, diff_sigma

def load_Gaus(Gaus_info):
    n, m = Gaus_info['A'].shape
    T2 = np.linspace(10, 200, m)
    TE = Gaus_info['TE'].flatten()
    A = Gaus_info['A']
    SNR = 1000
    return T2, TE, A, m, SNR

def get_IdealModel_weighted(iter_j, m, npeaks, T2, T2mu, sigma_i):
    T2mu_sim = T2mu[iter_j, :]
    p = np.array([normsci.pdf(T2, mu, sigma) for mu, sigma in zip(T2mu_sim, sigma_i)])
    IdealModel_weighted = (p.T @ f_coef) / npeaks
    return IdealModel_weighted

def preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list):
    df_copy = df.copy()
    df_copy['Sigma_tuple'] = df_copy['Sigma'].apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)
    print("First few rows of the DataFrame:")
    print(df_copy.head())
    group_sizes = df_copy.groupby(['Sigma_tuple', 'RPS_val']).size()
    print("Group sizes (Sigma, RPS combinations and realizations count):")
    print(group_sizes)
    
    err_grids = {err: np.full((len(unif_sigma), len(uniqueRPS)), np.nan) for err in err_list}
    sigma_to_index = {sig: idx for idx, sig in enumerate(unif_sigma)}
    rps_to_index = {rps: idx for idx, rps in enumerate(uniqueRPS)}

    grouped = df_copy.groupby(['Sigma_tuple', 'RPS_val'])[err_list].mean().reset_index()
    print("Grouped data after averaging over noise realizations:")
    print(grouped.head())

    # Track noise realizations correctly
    noise_realizations_count = group_sizes.to_dict()

    for _, row in grouped.iterrows():
        sigma = row['Sigma_tuple'][0]
        rps = row['RPS_val']
        if sigma in sigma_to_index and rps in rps_to_index:
            sigma_idx = sigma_to_index[sigma]
            rps_idx = rps_to_index[rps]
            for err in err_list:
                err_grids[err][sigma_idx, rps_idx] = row[err]
        else:
            print(f"Warning: Sigma={sigma}, RPS={rps} not found in Sigma_to_index or RPS_to_index. Skipping.")

    print("Number of noise realizations being averaged for each (Sigma, RPS):")
    for (sigma, rps), count in noise_realizations_count.items():
        print(f"Sigma={sigma}, RPS={rps}: Averaging over {count} realizations")
    return err_grids


import pandas as pd
# filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May2025_nsim10\est_table_SNR1000_iter10_lamini_GCV_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_WassScore_compare1st2ndDeriv_UPEN_20May25.pkl"
# filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-22_nsim1\est_table_SNR1000_iter1_22May25.pkl"
# filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-22_nsim1\est_table_SNR1000_iter1_22May25.pkl"
# filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-22_nsim2\est_table_SNR1000_iter2_22May25.pkl"
with open(filepath, 'rb') as file:
    df = pickle.load(file)
solution_list = ["f_rec_GCV", "f_rec_LocReg1", "f_rec_upen", "f_rec_upen1D", "f_rec_upen0D"]
print("Null counts in solution columns:")
print(df[solution_list].isnull().sum())
for col in solution_list:
    invalid_rows = df[df[col].apply(lambda x: x is None or not isinstance(x, (list, np.ndarray)) or len(x) == 0)]
    if not invalid_rows.empty:
        print(f"\nInvalid rows for {col}:")
        print(invalid_rows[["NR", "Sigma", "RPS_val"]])
# solution_list = ["GCV_vect", "LR_vect", "upen_vect"]
# lam_list = ["GCV_lam", "LR_lam", "upen_lam"]
# err_list = ["err_GCV", "err_LR", "err_upen"]
# solution_list= ["GCV_vect", "LR_vect_1stDer", "upen_vect"]
# lam_list= ["GCV_lam", "LR_lam_1stDer", "upen_lam"]
# err_list= ["err_GCV","err_LR_1stDer", "err_upen"]
# solution_list= ["GCV_vect",  "LR_vect_2ndDer", "upen_vect"]
# lam_list= ["GCV_lam",  "LR_lam_2ndDer", "upen_lam"]
# err_list= ["err_GCV", "err_LR_2ndDer", "err_upen"]

solution_list = [
    "f_rec_GCV", "f_rec_oracle", 
    "f_rec_LocReg", "f_rec_LocReg1", "f_rec_LocReg2", 
    "f_rec_upen", "f_rec_upen1D", "f_rec_upen0D"
]

lam_list = [
    "lambda_GCV", "lambda_oracle", 
    "lambda_LR", "lambda_LR1", "lambda_LR2", 
    "lambda_upen", "lambda_upen1D", "lambda_upen0D"
]

err_list = [
    "err_GCV", "err_oracle", 
    "err_LocReg", "err_LocReg1", "err_LocReg2", 
    "err_upen", "err_upen1D", "err_upen0D"
]

methodname = [
    "GCV", "Oracle", 
    "LocReg", "LocReg 1st Derivative", "LocReg 2nd Derivative", 
    "UPEN 2nd Derivative", "UPEN 1st Derivative", "UPEN"
]

# def compare_heatmap(df, err_list, savepath, unif_sigma, uniqueRPS, err_type="WassScore"):
#     err_grids = preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list)
#     mainmethod = "LocReg 1st Derivative"
#     moderr_list = err_list.copy()

#     # Modify error list names for display
#     for idx, err in enumerate(moderr_list):
#         moderr_list[idx] = err.replace("err_", "").replace("upen", "UPEN")
#     moderr_list = [mainmethod if "LocReg" in err else err for err in moderr_list]

#     # Create a dictionary of methods excluding the main method
#     methodlist = {f"{errmeth}": idx for idx, errmeth in enumerate(moderr_list)}
#     methodlist.pop(mainmethod)  # Remove LocReg from the comparison

#     # Dynamically calculate the number of rows for subplots
#     n_methods = len(methodlist)
#     n_rows = (n_methods + 1) // 2
#     fig, axs = plt.subplots(n_rows, 2, sharey=True, figsize=(18, 6 * n_rows))
#     axs = axs.flatten()
#     plt.subplots_adjust(wspace=0.3, hspace=0.4)

#     # Define tick labels dynamically for each comparison
#     tick_labels = [[f'{mainmethod} is better', 'Neutral', f'{err} is better'] for err in methodlist]

#     # Calculate vmax and vmin for color scaling
#     vmax1 = max(np.nanmax(err_grids[err]) for err in err_list)
#     vmin1 = -vmax1
#     fmt1 = ".2e" if err_type == "WassScore" else ".3e"

#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         # Calculate the average error value
#         avg_error = np.nanmean(data)
#         title_with_avg = f"{title}\n(Average Error: {avg_error:.2e})"

#         im = sns.heatmap(
#             data, cmap='jet', ax=ax, cbar=True, vmin=vmin1, vmax=vmax1,
#             annot=True, fmt=fmt1, annot_kws={"size": 12, "weight": "bold"},
#             linewidths=0.5, linecolor='black',
#             cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8},
#             xticklabels=np.round(x_ticks, 4), yticklabels=np.round(y_ticks, 4)
#         )
        
#         ax.set_xlabel('Peak Separation', fontsize=20)
#         ax.set_ylabel('Peak Width', fontsize=20)
#         ax.set_title(title_with_avg, fontsize=20, pad=20)
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=14)
#         ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([vmin1, 0, vmax1])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)

#     # Generate heatmaps for each comparison (excluding LocReg vs. LocReg)
#     for idx, (err, err_idx) in enumerate(methodlist.items()):
#         comparison_data = err_grids['err_LocReg1'] - err_grids[err_list[err_idx]]
#         # comparison_data = err_grids['err_LR'] - err_grids[err_list[err_idx]]
#         # comparison_data = err_grids['err_LR_2ndDer'] - err_grids[err_list[err_idx]]
#         add_heatmap(
#             axs[idx], comparison_data, tick_labels[idx],
#             title=f'{mainmethod} Error - {err} Error ({err_type})',
#             x_ticks=uniqueRPS, y_ticks=unif_sigma
#         )

#     # Hide unused subplots
#     for ax in axs[len(methodlist):]:
#         ax.axis('off')

#     # Save the figure
#     plt.tight_layout()
#     plt.savefig(os.path.join(savepath, "compare_heatmap.png"))
#     print("Saved Comparison Heatmap")
#     plt.close()

def compare_heatmap(df, err_list, methodname, savepath, unif_sigma, uniqueRPS, err_type="WassScore"):
    err_grids = preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list)
    
    locreg_methods = {
        "LocReg": "err_LocReg",
        "LocReg 1st Derivative": "err_LocReg1",
        "LocReg 2nd Derivative": "err_LocReg2"
    }

    # Map err_list entries to methodname labels for titles
    err_to_name = dict(zip(err_list, methodname))

    for mainmethod, mainerr in locreg_methods.items():
        # Set of all methods excluding the current mainerr
        methodlist = {err_to_name[err_list[i]]: i for i in range(len(err_list)) if err_list[i] != mainerr}

        n_methods = len(methodlist)
        n_rows = (n_methods + 1) // 2
        fig, axs = plt.subplots(n_rows, 2, sharey=True, figsize=(24, 8 * n_rows))
        axs = axs.flatten()
        plt.subplots_adjust(wspace=0.35, hspace=0.4)

        # Determine vmax and vmin for color scale based on all involved errors in this plot
        vmax1 = max(np.nanmax(err_grids[mainerr] - err_grids[err_list[idx]]) for idx in methodlist.values())
        vmin1 = -vmax1

        fmt1 = ".2e" if err_type == "WassScore" else ".3e"

        def add_heatmap(ax, data, title, x_ticks, y_ticks, tick_labels):
            avg_error = np.nanmean(data)
            title_with_avg = f"{title}\n(Average Difference: {avg_error:.2e})"
            im = sns.heatmap(
                data, cmap='jet', ax=ax, cbar=True, vmin=vmin1, vmax=vmax1,
                annot=True, fmt=fmt1, annot_kws={"size": 14, "weight": "bold"},
                linewidths=0.7, linecolor='black',
                cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8},
                xticklabels=np.round(x_ticks, 4), yticklabels=np.round(y_ticks, 4)
            )
            ax.set_xlabel('Peak Separation', fontsize=22)
            ax.set_ylabel('Peak Width', fontsize=22)
            ax.set_title(title_with_avg, fontsize=24, pad=25)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=16)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

            cbar = im.collections[0].colorbar
            cbar.set_ticks([vmin1, 0, vmax1])
            cbar.set_ticklabels([f'{mainmethod} better', 'Neutral', f'{title.split(" - ")[1].split()[0]} better'])
            cbar.ax.tick_params(labelsize=16)

        for idx, (method_name, err_idx) in enumerate(methodlist.items()):
            comparison_data = err_grids[mainerr] - err_grids[err_list[err_idx]]
            add_heatmap(
                axs[idx], comparison_data,
                title=f'{mainmethod} Error - {method_name} Error ({err_type})',
                x_ticks=uniqueRPS, y_ticks=unif_sigma,
                tick_labels=[f'{mainmethod} better', 'Neutral', f'{method_name} better']
            )

        # Turn off any unused subplots
        for ax in axs[n_methods:]:
            ax.axis('off')

        plt.tight_layout()
        # plotsavepath = savepath + "\moreNRs"
        filename = f"compare_heatmap_{mainmethod.replace(' ', '_')}.png"
        plt.savefig(os.path.join(savepath, filename))
        print(f"Saved Comparison Heatmap: {filename}")
        plt.close()


# def indiv_heatmap(df, err_list, savepath, unif_sigma, uniqueRPS, err_type="WassScore"):
#     err_grids = preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list)
#     n_methods = len(err_list)
#     n_rows = (n_methods + 1) // 2
#     fig, axs = plt.subplots(n_rows, 2, sharey=True, figsize=(18, 6 * n_rows))
#     axs = axs.flatten()
#     plt.subplots_adjust(wspace=0.3, hspace=0.3)
#     tick_labels = [[f'Low {err.split("_")[1]} Error', f'High {err.split("_")[1]} Error'] for err in err_list]
#     vmax1 = max(np.nanmax(err_grids[err]) for err in err_list)
#     fmt1 = ".4f" if err_type == "WassScore" else ".3f"

#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         # Calculate the average error value
#         avg_error = np.nanmean(data)
#         title_with_avg = f"{title}\n(Average Error: {avg_error:.2e})"

#         im = sns.heatmap(
#             data, cmap='jet', ax=ax, cbar=True, vmin=0, vmax=vmax1,
#             annot=True, fmt=fmt1, annot_kws={"size": 12, "weight": "bold"},
#             linewidths=0.5, linecolor='black',
#             cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8},
#             xticklabels=np.round(x_ticks, 4), yticklabels=np.round(y_ticks, 4)
#         )
#         ax.set_xlabel('Peak Separation', fontsize=20)
#         ax.set_ylabel('Peak Width', fontsize=20)
#         ax.set_title(title_with_avg, fontsize=20, pad=20)
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=14)
#         ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([0, vmax1])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)

#     for idx, err in enumerate(err_list):
#         add_heatmap(
#             axs[idx], err_grids[err], tick_labels[idx],
#             title=f'{err.split("_")[1]} Error ({err_type})',
#             x_ticks=uniqueRPS, y_ticks=unif_sigma
#         )

#     for ax in axs[len(err_list):]:
#         ax.axis('off')

#     plt.tight_layout()
#     plt.savefig(os.path.join(savepath, "indiv_heatmap.png"))
#     print("Saved Individual Heatmap")
#     plt.close()
def indiv_heatmap(df, err_list, methodname, savepath, unif_sigma, uniqueRPS, err_type="WassScore"):
    err_grids = preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list)

    # Groupings by derivative order, matching err_list keys
    groups = {
        "0th Derivative": ["err_GCV", "err_oracle", "err_LocReg", "err_upen0D"],
        "1st Derivative": ["err_LocReg1", "err_upen1D"],
        "2nd Derivative": ["err_LocReg2", "err_upen"]  # Confirm if err_upen0D is 2nd derivative
    }

    fmt1 = ".4f" if err_type == "WassScore" else ".3f"

    # Map err_list entries to methodname labels for consistent titles
    err_to_name = dict(zip(err_list, methodname))

    def add_heatmap(ax, data, title, x_ticks, y_ticks):
        avg_error = np.nanmean(data)
        title_with_avg = f"{title}\n(Average Error: {avg_error:.2e})"
        vmax1 = np.nanmax(data)

        im = sns.heatmap(
            data, cmap='jet', ax=ax, cbar=True, vmin=0, vmax=vmax1,
            annot=True, fmt=fmt1, annot_kws={"size": 14, "weight": "bold"},
            linewidths=0.7, linecolor='black',
            cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8},
            xticklabels=np.round(x_ticks, 4), yticklabels=np.round(y_ticks, 4)
        )
        ax.set_xlabel('Peak Separation', fontsize=22)
        ax.set_ylabel('Peak Width', fontsize=22)
        ax.set_title(title_with_avg, fontsize=24, pad=25)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

        # Adjust colorbar ticks & labels
        cbar = im.collections[0].colorbar
        cbar.set_ticks([0, vmax1])
        cbar.set_ticklabels(['Low Error', 'High Error'])
        cbar.ax.tick_params(labelsize=16)

    for deriv_order, methods in groups.items():
        # Keep only methods present in err_list to avoid errors
        active_methods = [m for m in methods if m in err_list]
        n_methods = len(active_methods)
        if n_methods == 0:
            continue  # skip if no methods found in this group

        n_rows = (n_methods + 1) // 2
        fig, axs = plt.subplots(n_rows, 2, sharey=True, figsize=(24, 8 * n_rows))
        axs = axs.flatten()
        plt.subplots_adjust(wspace=0.35, hspace=0.35)

        for idx, err_key in enumerate(active_methods):
            method_label = err_to_name.get(err_key, err_key)
            add_heatmap(
                axs[idx], err_grids[err_key],
                title=f"{method_label} ({err_type})",
                x_ticks=uniqueRPS, y_ticks=unif_sigma
            )

        # Turn off unused axes if any
        for ax in axs[n_methods:]:
            ax.axis('off')

        plt.tight_layout()
        filename = f"indiv_heatmap_{deriv_order.replace(' ', '_')}.png"
        plt.savefig(os.path.join(savepath, filename))
        print(f"Saved Individual Heatmap: {filename}")
        plt.close()


# Load Gaus data
unif_sigma, diff_sigma = calc_diff_sigma(nsigma)
gauspath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\Simulations\num_of_basis_functions\lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_6_basis2_40110lmbda_min-6lmbda_max008Oct24.pkl"
Gaus_info = np.load(gauspath, allow_pickle=True)
T2, TE, A, m, SNR = load_Gaus(Gaus_info)

# Ground truth simulation
T2mu = calc_T2mu(rps)
lis = []

for i in range(n_sim):
    for j in range(nsigma):
        for k in range(nrps):
            sigma_i = diff_sigma[j, :]
            rps_val = calc_rps_val(k, rps)
            IdealModel_weighted = get_IdealModel_weighted(k, m, npeaks, T2, T2mu, sigma_i)
            lis.append({
                "gt": IdealModel_weighted,
                "NR": i,
                "iter_sigma": j,
                "iter_rps": k,
                "Sigma": sigma_i.tolist(),
                "RPS_val": rps_val
            })

gtdf = pd.DataFrame(lis)

# Plot comparisons
# Define groups: [Oracle, GCV, LocReg variant, UPEN variant]
# Define plot groups: [Oracle, GCV, LocReg variant, UPEN variant]
groups = {
    "0th Derivative": [1, 0, 2, 7],   # Oracle, GCV, LocReg, UPEN
    "1st Derivative": [1, 0, 3, 6],   # Oracle, GCV, LocReg 1st Derivative, UPEN 1st Derivative
    "2nd Derivative": [1, 0, 4, 5],   # Oracle, GCV, LocReg 2nd Derivative, UPEN 2nd Derivative
}

# Map Sigma and RPS to indices for filenames
sig_to_index = {tuple(sig): idx for idx, sig in enumerate(uniquesig)}
rps_to_index = {rps: idx for idx, rps in enumerate(uniqueRPS)}
T2axis = np.linspace(10, 200, 150)

# # Loop over NR, Sigma, RPS
# for NR in uniqueNRs:
#     for (sig, RPS) in itertools.product(uniquesig, uniqueRPS):
#         filt_df = df[df["NR"] == NR]
#         gtdffilt = gtdf[gtdf["NR"] == NR]
#         newgtdf = gtdffilt.loc[
#             (gtdffilt["RPS_val"] == RPS) &
#             (gtdffilt["Sigma"].apply(lambda sigma: np.allclose(sigma, sig)))
#         ]
#         new_df = filt_df.loc[
#             (filt_df["RPS_val"] == RPS) &
#             (filt_df["Sigma"].apply(lambda sigma: np.allclose(sigma, sig)))
#         ]

#         if new_df.empty or newgtdf.empty:
#             print(f"Skipping NR={NR}, Sigma={sig}, RPS={RPS}: new_df.empty={new_df.empty}, newgtdf.empty={newgtdf.empty}")
#             continue
#         # if NR == 0:
#         if NR <= 5:
#             for group_name, indices in groups.items():
#                 plt.figure(figsize=(10, 6))
#                 for idx in indices:
#                     str = solution_list[idx]
#                     label = methodname[idx]
#                     error_value = new_df[err_list[idx]].iloc[0]
#                     data_to_plot = new_df[str].iloc[0]

#                     if data_to_plot is None or not isinstance(data_to_plot, (list, np.ndarray)) or len(data_to_plot) == 0:
#                         print(f"Invalid data for NR={NR}, Sigma={sig}, RPS={RPS}, Solution={str}")
#                         continue
#                     if len(data_to_plot) != len(T2axis):
#                         print(f"Length mismatch for NR={NR}, Sigma={sig}, RPS={RPS}, Solution={str}: data length={len(data_to_plot)}, T2axis length={len(T2axis)}")
#                         continue

#                     plt.plot(
#                         T2axis, data_to_plot, linewidth=3,
#                         label=f"{label} ({error_value:.2e})"
#                     )

#                 # Plot ground truth (black dashed line)
#                 gt_data = newgtdf["gt"].iloc[0]
#                 if gt_data is not None and len(gt_data) == len(T2axis):
#                     plt.plot(
#                         T2axis, gt_data,
#                         label="Ground Truth", color="black", linewidth=3, linestyle = "--"
#                     )

#                 # Keep original plot settings
#                 plt.legend(fontsize=12)
#                 plt.xlabel("T2 Axis (ms)", fontsize=14)
#                 plt.ylabel("Amplitude", fontsize=14)
#                 plt.ylim(0, np.max(gt_data) * 1.5)
#                 # plt.title(f"NR={NR}, Sigma={sig}, RPS={RPS} — {group_name}", fontsize=16)
#                 # plt.grid(True)

#                 sigma_index = sig_to_index[tuple(sig)]
#                 rps_index = rps_to_index[RPS]
#                 filename = f"Simulation{NR}_Sigma{sigma_index}_RPS{rps_index}_{group_name.replace(' ', '_')}.png"
#                 plotsavepath = savepath + "\moreNRs"
#                 plt.savefig(os.path.join(plotsavepath, filename))
#                 plt.close()
#         else:
#             pass
# Loop over NR, Sigma, RPS
for NR in uniqueNRs:
    for (sig, RPS) in itertools.product(uniquesig, uniqueRPS):

        print(f"\n---\nChecking NR={NR}, Sigma={sig}, RPS={RPS}")

        # Filter DataFrames
        filt_df = df[df["NR"] == NR]
        gtdffilt = gtdf[gtdf["NR"] == NR]

        print(f"→ Rows with NR={NR}: df={len(filt_df)}, gtdf={len(gtdffilt)}")

        # Apply Sigma and RPS filters
        newgtdf = gtdffilt.loc[
            (gtdffilt["RPS_val"] == RPS) &
            (gtdffilt["Sigma"].apply(lambda sigma: np.allclose(np.array(sigma), np.array(sig))))
        ]
        new_df = filt_df.loc[
            (filt_df["RPS_val"] == RPS) &
            (filt_df["Sigma"].apply(lambda sigma: np.allclose(np.array(sigma), np.array(sig))))
        ]

        print(f"→ After filtering: new_df={len(new_df)}, newgtdf={len(newgtdf)}")

        if new_df.empty or newgtdf.empty:
            print(f"⚠️ Skipping NR={NR}, Sigma={sig}, RPS={RPS} — Empty filtered data")
            continue

        if NR <= 5:
            for group_name, indices in groups.items():
                plt.figure(figsize=(10, 6))
                for idx in indices:
                    sol_key = solution_list[idx]
                    label = methodname[idx]

                    try:
                        error_value = new_df[err_list[idx]].iloc[0]
                        data_to_plot = new_df[sol_key].iloc[0]
                    except Exception as e:
                        print(f"⚠️ Error accessing solution for {sol_key}: {e}")
                        continue

                    if data_to_plot is None or not isinstance(data_to_plot, (list, np.ndarray)) or len(data_to_plot) == 0:
                        print(f"⚠️ Invalid data for NR={NR}, Sigma={sig}, RPS={RPS}, Solution={sol_key}")
                        continue
                    if len(data_to_plot) != len(T2axis):
                        print(f"⚠️ Length mismatch for {sol_key}: got {len(data_to_plot)}, expected {len(T2axis)}")
                        continue

                    plt.plot(T2axis, data_to_plot, linewidth=3, label=f"{label} ({error_value:.2e})")

                # Plot ground truth
                gt_data = newgtdf["gt"].iloc[0]
                if gt_data is not None and len(gt_data) == len(T2axis):
                    plt.plot(T2axis, gt_data, label="Ground Truth", color="black", linewidth=3, linestyle="--")

                # Final plot settings
                plt.legend(fontsize=12)
                plt.xlabel("T2 Axis (ms)", fontsize=14)
                plt.ylabel("Amplitude", fontsize=14)
                plt.ylim(0, np.max(gt_data) * 1.5)

                sigma_index = sig_to_index[tuple(sig)]
                rps_index = rps_to_index[RPS]
                filename = f"Simulation{NR}_Sigma{sigma_index}_RPS{rps_index}_{group_name.replace(' ', '_')}.png"
                plotsavepath = savepath + "\moreNRs"
                full_path = os.path.join(plotsavepath, filename)

                print(f"✅ Saving plot to {full_path}")
                plt.savefig(full_path)
                plt.close()

# Step 1: Normalize Sigma to tuple form for grouping
df['Sigma'] = df['Sigma'].apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)

# Step 2: Sort the DataFrame
df_sorted = df.sort_values(by=['NR', 'Sigma', 'RPS_val'], ascending=True)

# Plot heatmaps
# compare_heatmap(df_sorted, err_list, savepath, unif_sigma, uniqueRPS)
# indiv_heatmap(df_sorted, err_list, savepath, unif_sigma, uniqueRPS)
compare_heatmap(df_sorted, err_list, methodname, savepath, unif_sigma, uniqueRPS)
indiv_heatmap(df_sorted, err_list, methodname, savepath, unif_sigma, uniqueRPS)


# def preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list):
#     df_copy = df.copy()
#     df_copy['Sigma_tuple'] = df_copy['Sigma'].apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)

#     err_grids = {err: np.full((len(unif_sigma), len(uniqueRPS)), np.nan) for err in err_list}
#     sigma_to_index = {sig: idx for idx, sig in enumerate(unif_sigma)}
#     rps_to_index = {rps: idx for idx, rps in enumerate(uniqueRPS)}

#     grouped = df_copy.groupby(['Sigma_tuple', 'RPS_val'])[err_list].mean().reset_index()
#     print("grouped",grouped)
#     for _, row in grouped.iterrows():
#         sigma = row['Sigma_tuple'][0]
#         rps = row['RPS_val']
#         if sigma in sigma_to_index and rps in rps_to_index:
#             sigma_idx = sigma_to_index[sigma]
#             rps_idx = rps_to_index[rps]
#             for err in err_list:
#                 err_grids[err][sigma_idx, rps_idx] = row[err]

#     return err_grids


# def preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list):
#     # Create a copy of the DataFrame to avoid altering the original
#     df_copy = df.copy()
#     # Convert Sigma column to tuples for grouping
#     df_copy['Sigma_tuple'] = df_copy['Sigma'].apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)
#     # Debugging: Print the first few rows to verify the structure
#     print("First few rows of the DataFrame:")
#     print(df_copy.head())
#     # Debugging: Check if we have 10 noise realizations per Sigma and RPS combination
#     group_sizes = df_copy.groupby(['Sigma_tuple', 'RPS_val']).size()
#     print("Group sizes (Sigma, RPS combinations and realizations count):")
#     print(group_sizes)
#     # Initialize error grids to store averaged errors for each method
#     err_grids = {err: np.full((len(unif_sigma), len(uniqueRPS)), np.nan) for err in err_list}

#     # Create dictionaries for indexing based on Sigma and RPS values
#     sigma_to_index = {sig: idx for idx, sig in enumerate(unif_sigma)}
#     rps_to_index = {rps: idx for idx, rps in enumerate(uniqueRPS)}

#     # Ensure the noise realizations are being properly grouped
#     # Group by Sigma_tuple, RPS_val, and NR (Noise Realization)
#     # test = df_copy.groupby(['Sigma_tuple', 'RPS_val', 'NR'])

#     grouped = df_copy.groupby(['Sigma_tuple', 'RPS_val'])[err_list].mean().reset_index()
# # , 'NR'
#     # Debugging: Check the groupings after aggregation
#     print("Grouped data after averaging over noise realizations:")
#     print(grouped.head())

#     # Track the number of noise realizations being averaged
#     noise_realizations_count = {}

#     # Fill the error grids with averaged error values and print out additional details
#     for _, row in grouped.iterrows():
#         sigma = row['Sigma_tuple'][0]  # Extract sigma value
#         rps = row['RPS_val']  # Extract rps value

#         # Ensure we only fill the grid if the sigma and rps values are valid
#         if sigma in sigma_to_index and rps in rps_to_index:
#             sigma_idx = sigma_to_index[sigma]
#             rps_idx = rps_to_index[rps]

#             # For each error type, assign the averaged value to the corresponding location in the grid
#             for err in err_list:
#                 err_grids[err][sigma_idx, rps_idx] = row[err]

#             # Track the number of noise realizations being averaged
#             if (sigma, rps) not in noise_realizations_count:
#                 noise_realizations_count[(sigma, rps)] = 0
#             noise_realizations_count[(sigma, rps)] += 1
#         else:
#             # Print out the combinations of Sigma and RPS that are not being averaged
#             print(f"Warning: Sigma={sigma}, RPS={rps} not found in Sigma_to_index or RPS_to_index. Skipping.")
    
#     # Print out the number of noise realizations for each Sigma and RPS combination
#     print("Number of noise realizations being averaged for each (Sigma, RPS):")
#     for (sigma, rps), count in noise_realizations_count.items():
#         print(f"Sigma={sigma}, RPS={rps}: Averaging over {count} realizations")
#     return err_grids


# def compare_heatmap(df, err_list, savepath, unif_sigma, uniqueRPS, err_type="WassScore"):
#     err_grids = preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list)
#     # mainmethod = "LocReg"  # Define the main method for comparison
#     mainmethod = "LocReg 1st Derivative"
#     moderr_list = err_list.copy()

#     # Modify error list names for display
#     for idx, err in enumerate(moderr_list):
#         moderr_list[idx] = err.replace("err_", "").replace("upen", "UPEN")
#     moderr_list = [mainmethod if "LR" in err else err for err in moderr_list]

#     # Create a dictionary of methods excluding the main method
#     methodlist = {f"{errmeth}": idx for idx, errmeth in enumerate(moderr_list)}
#     methodlist.pop(mainmethod)  # Remove LocReg from the comparison

#     # Dynamically calculate the number of rows for subplots
#     n_methods = len(methodlist)
#     n_rows = (n_methods + 1) // 2
#     fig, axs = plt.subplots(n_rows, 2, sharey=True, figsize=(18, 6 * n_rows))
#     axs = axs.flatten()
#     plt.subplots_adjust(wspace=0.3, hspace=0.4)

#     # Define tick labels dynamically for each comparison
#     tick_labels = [[f'{mainmethod} is better', 'Neutral', f'{err} is better'] for err in methodlist]

#     # Calculate vmax and vmin for color scaling
#     vmax1 = max(np.nanmax(err_grids[err]) for err in err_list)
#     vmin1 = -vmax1
#     fmt1 = ".2e" if err_type == "WassScore" else ".3e"

#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         im = sns.heatmap(
#             data, cmap='jet', ax=ax, cbar=True, vmin=vmin1, vmax=vmax1,
#             annot=True, fmt=fmt1, annot_kws={"size": 12, "weight": "bold"},
#             linewidths=0.5, linecolor='black',
#             cbar_kws= {"orientation": "horizontal", "pad": 0.2, "shrink": 0.8},
#             xticklabels=np.round(x_ticks, 4), yticklabels=np.round(y_ticks, 4)
#         )
        
#         ax.set_xlabel('Peak Separation', fontsize=20)
#         ax.set_ylabel('Peak Width', fontsize=20)
#         ax.set_title(title, fontsize=20, pad=20)
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=14)
#         ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([vmin1, 0, vmax1])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)

#     # Generate heatmaps for each comparison (excluding LocReg vs. LocReg)
#     for idx, (err, err_idx) in enumerate(methodlist.items()):
#         # comparison_data = err_grids['err_LR'] - err_grids[err_list[err_idx]]
#         comparison_data = err_grids['err_LR_1stDer'] - err_grids[err_list[err_idx]]
#         add_heatmap(
#             axs[idx], comparison_data, tick_labels[idx],
#             title=f'{mainmethod} Error - {err} Error ({err_type})',
#             x_ticks=uniqueRPS, y_ticks=unif_sigma
#         )

#     # Hide unused subplots
#     for ax in axs[len(methodlist):]:
#         ax.axis('off')

#     # Save the figure
#     plt.tight_layout()
#     plt.savefig(os.path.join(savepath, "compare_heatmap.png"))
#     print("Saved Comparison Heatmap")
#     plt.close()

# def indiv_heatmap(df, err_list, savepath, unif_sigma, uniqueRPS, err_type="WassScore"):
#     err_grids = preprocess_heatmap_data(df, unif_sigma, uniqueRPS, err_list)
#     n_methods = len(err_list)
#     n_rows = (n_methods + 1) // 2
#     fig, axs = plt.subplots(n_rows, 2, sharey=True, figsize=(18, 6 * n_rows))
#     axs = axs.flatten()
#     plt.subplots_adjust(wspace=0.3, hspace=0.3)
#     tick_labels = [[f'Low {err.split("_")[1]} Error', f'High {err.split("_")[1]} Error'] for err in err_list]
#     vmax1 = max(np.nanmax(err_grids[err]) for err in err_list)
#     fmt1 = ".4f" if err_type == "WassScore" else ".3f"

#     def add_heatmap(ax, data, tick_labels, title, x_ticks, y_ticks):
#         im = sns.heatmap(
#             data, cmap='jet', ax=ax, cbar=True, vmin=0, vmax=vmax1,
#             annot=True, fmt=fmt1, annot_kws={"size": 12, "weight": "bold"},
#             linewidths=0.5, linecolor='black',
#             cbar_kws={"orientation": "horizontal", "pad": 0.2, "shrink": 0.8},
#             xticklabels=np.round(x_ticks, 4), yticklabels=np.round(y_ticks, 4)
#         )
#         ax.set_xlabel('Peak Separation', fontsize=20)
#         ax.set_ylabel('Peak Width', fontsize=20)
#         ax.set_title(title, fontsize=20, pad=20)
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, fontsize=14)
#         ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
#         cbar = im.collections[0].colorbar
#         cbar.set_ticks([0, vmax1])
#         cbar.set_ticklabels(tick_labels)
#         cbar.ax.tick_params(labelsize=16)

#     for idx, err in enumerate(err_list):
#         add_heatmap(
#             axs[idx], err_grids[err], tick_labels[idx],
#             title=f'{err.split("_")[1]} Error ({err_type})',
#             x_ticks=uniqueRPS, y_ticks=unif_sigma
#         )

#     for ax in axs[len(err_list):]:
#         ax.axis('off')

#     plt.tight_layout()
#     plt.savefig(os.path.join(savepath, "indiv_heatmap.png"))
#     print("Saved Individual Heatmap")
#     plt.close()

# for NR in uniqueNRs:
#     for (sig, RPS) in itertools.product(uniquesig, uniqueRPS):
#         filt_df = df[df["NR"] == NR]
#         gtdffilt = gtdf[gtdf["NR"] == NR]
#         newgtdf = gtdffilt.loc[(gtdffilt["RPS_val"] == RPS) & (gtdffilt["Sigma"].apply(lambda sigma: np.allclose(sigma, sig)))]
#         new_df = filt_df.loc[(filt_df["RPS_val"] == RPS) & (filt_df["Sigma"].apply(lambda sigma: np.allclose(sigma, sig)))]
#         if new_df.empty or newgtdf.empty:
#             continue
#         plt.figure()
#         for str_idx, str in enumerate(solution_list):
#             label = str.replace("vect", "").replace("LR", "LocReg").replace("1stDer", "1st Derivative").replace("2ndDer", "2nd Derivative").replace("_", " ").replace("upen", "UPEN").replace("0D","")
#             error_value = new_df[err_list[str_idx]].iloc[0]
#             plt.plot(T2axis, new_df[str].iloc[0], linewidth=4,label=f"{label} ({error_value:.2e})")
#         plt.plot(T2axis, newgtdf["gt"].iloc[0], label="Ground Truth", color="black", linewidth=2)
#         plt.legend()
#         plt.xlabel("T2 Axis (ms)")
#         plt.ylabel("Amplitude")
#         plt.ylim(0, np.max(newgtdf["gt"].iloc[0])*1.5)
#         sigma_index = sig_to_index[tuple(sig)]
#         rps_index = rps_to_index[RPS]
#         plt.savefig(os.path.join(savepath, f"Simulation{NR}_Sigma{sigma_index}_RPS{rps_index}.png"))
#         plt.close()