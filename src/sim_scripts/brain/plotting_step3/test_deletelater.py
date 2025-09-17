import pickle
import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd
import matplotlib.pyplot as plt

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

# Load the data
# esttable = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_simexp117Jan25.pkl"
esttable = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_unfiltered_noNESMA21Jan25.pkl"
with open(esttable, 'rb') as file:
    data = pickle.load(file)

# Assuming the data is a pandas DataFrame
column_names = data.columns.tolist()
print(column_names)

print("LS_estimates:")
print(data['LS_estimate'][0])

dTE = 11.3
n = 32
TE = dTE * np.linspace(1, n, n)
m = 150
T2 = np.linspace(10, 200, m)
A = np.zeros((n, m))
dT = T2[1] - T2[0]


# Set the ground truth as GCV_estimate
data['ground_truth'] = data['GCV_estimate']

# Calculate the Wasserstein score between DP_estimate, LC_estimate, LR_estimate, and LS_estimate against the GCV_estimate
methods = ['DP_estimate', 'LC_estimate', 'LR_estimate', 'LS_estimate']
wasserstein_scores = {}
SAD_scores = {}
L2_scores = {}
# Initialize a dictionary to store the Wasserstein scores for each pixel
pixel_wasserstein_scores = {method: [] for method in methods}
pixel_SAD_scores = {method: [] for method in methods}
pixel_L2_scores = {method: [] for method in methods}

# Loop through each pixel and calculate the Wasserstein score for each method
for index, row in data.iterrows():
    x_val = row['X_val']
    y_val = row['Y_val']
    ground_truth = row['ground_truth']
    
    for method in methods:
        estimate = row[method]
        score = wasserstein_distance(ground_truth, estimate)
        SADscore = np.sum(np.abs(ground_truth - estimate)) / np.sum(ground_truth)
        L2score = np.linalg.norm(ground_truth - estimate) / np.linalg.norm(ground_truth)    
        pixel_wasserstein_scores[method].append((x_val, y_val, score))
        pixel_SAD_scores[method].append((x_val, y_val, SADscore))
        pixel_L2_scores[method].append((x_val, y_val, L2score))

# Create a brain map of Wasserstein errors for each method
wasserstein_error_maps = {method: np.zeros((313, 313)) for method in methods}
SAD_maps = {method: np.zeros((313, 313)) for method in methods}
L2_maps = {method: np.zeros((313, 313)) for method in methods}

# Populate the Wasserstein error maps
for method in methods:
    for x_val, y_val, score in pixel_wasserstein_scores[method]:
        wasserstein_error_maps[method][x_val, y_val] = score

for method in methods:
    for x_val, y_val, SADscore in pixel_SAD_scores[method]:
        SAD_maps[method][x_val, y_val] = SADscore

for method in methods:
    for x_val, y_val, L2score in pixel_L2_scores[method]:
        L2_maps[method][x_val, y_val] = L2score


wass_file_path = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/large_Wasserror"
# Plot and save the Wasserstein error maps
for method, error_map in wasserstein_error_maps.items():
    plt.figure()
    plt.title(f'Wasserstein Error Map for {method}')
    plt.imshow(error_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Wasserstein Distance')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(f'{wass_file_path}/{method}_wasserstein_error_map.png')
    plt.close()

SAD_file_path = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/SAD"
# Plot and save the Wasserstein error maps
for method, error_map in SAD_maps.items():
    plt.figure()
    plt.title(f'Sum of Absolute Differences (SAD) Map for {method}')
    plt.imshow(error_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='SAD')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(f'{SAD_file_path}/{method}_SADmap.png')
    plt.close()

L2_file_path = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/L2norm"
# Plot and save the Wasserstein error maps
for method, error_map in L2_maps.items():
    plt.figure()
    plt.title(f'Relative L2 Norm Error Map for {method}')
    plt.imshow(error_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Relative L2 Norm Error')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(f'{L2_file_path}/{method}_L2norm_map.png')
    plt.close()


#1)#unfiltered nad filtered wass; for each method (DP...etc)

#unfiltered nad filtered L2; for each method (DP...etc) [supplementatl]
#unfiltered nad filtered L1; for each method (DP...etc) [supplemental]

#2) #ground truth GCV filtered...use filtred GCV for both for all ground truth; 
    #computet he difference between results and filtered and unfiltered data against g.t.

#3) SNR: 1000, 300 add gaussian noise to NESMA filtered data (makes it less effect of rician noise?) 
#  bc rician amplifies the noise for unfiltered.

# 4) SpanReg
    # original brain data (filtered NESMA)
    # collect T2 from spanreg
    # then A * F(T2) = d, then add noise with uniform SNR
    # recovery using that noisy data 
    # (using DP, GCV, LocReg, LCurve, non-regular NNLS)
    # 200, 800 SNR levels

# # Define a threshold for low Wasserstein errors
# error_threshold = 0.01

# # Filter out pixels with very low Wasserstein errors
# filtered_pixel_wasserstein_scores = {method: [] for method in methods}
# for method in methods:
#     for x_val, y_val, score in pixel_wasserstein_scores[method]:
#         if score > error_threshold:
#             filtered_pixel_wasserstein_scores[method].append((x_val, y_val, score))

# # Create a brain map of filtered Wasserstein errors for each method
# filtered_wasserstein_error_maps = {method: np.zeros((313, 313)) for method in methods}

# # Populate the filtered Wasserstein error maps
# for method in methods:
#     for x_val, y_val, score in filtered_pixel_wasserstein_scores[method]:
#         filtered_wasserstein_error_maps[method][x_val, y_val] = score

# # Plot and save the filtered Wasserstein error maps
# for method, error_map in filtered_wasserstein_error_maps.items():
#     plt.figure()
#     plt.title(f'Filtered Wasserstein Error Map for {method}')
#     plt.imshow(error_map, cmap='viridis', interpolation='nearest')
#     plt.colorbar(label='Wasserstein Distance')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.savefig(f'/home/kimjosy/LocReg_Regularization-1/Simulations/brainscripts/large_Wasserror/{method}_filtered_wasserstein_error_map.png')
#     plt.close()

# Print out the Wasserstein scores by x_val and y_val
for method in methods:
    print(f"Wasserstein scores for {method}:")
    for x_val, y_val, score in pixel_wasserstein_scores[method]:
        print(f"x: {x_val}, y: {y_val}, score: {score}")
# # Identify pixels with large noticeable differences in MWF
# threshold = 0.1  # Define a threshold for noticeable differences

# # Identify pixels with large Wasserstein errors
# large_error_pixels = data[data[methods].apply(lambda row: any(abs(row[method] - row['ground_truth']) > threshold for method in methods), axis=1)]

# # Loop through flagged pixels and plot the curves
# for index, row in large_error_pixels.iterrows():
#     x_coord = row['x_coord']
#     y_coord = row['y_coord']
#     curr_SNR = row['SNR']
#     MWF = row['MWF']
#     curr_data = row['curr_data']
#     lambda_vals = row['lambda_vals']
#     frec = row['frec']
    
#     for method in methods:
#         curve_plot(method, x_coord, y_coord, frec, curr_data, lambda_vals, curr_SNR, MWF, '/home/kimjosy/LocReg_Regularization-1/Simulations/brainscripts/large_Wasserror')

# # Plot the curves for each method
# # Identify pixels with large MWF differences
# mwf_threshold = 0.1  # Define a threshold for large MWF differences
# large_mwf_diff_pixels = data[data['MWF_diff'] > mwf_threshold]

# # Loop through pixels with large MWF differences and plot the curves
# for index, row in large_mwf_diff_pixels.iterrows():
#     x_coord = row['x_coord']
#     y_coord = row['y_coord']
#     curr_SNR = row['SNR']
#     MWF = row['MWF']
#     curr_data = row['curr_data']
#     lambda_vals = row['lambda_vals']
#     frec = row['frec']
    
#     for method in methods:
#         curve_plot(method, x_coord, y_coord, frec, curr_data, lambda_vals, curr_SNR, MWF, '/home/kimjosy/LocReg_Regularization-1/Simulations/brainscripts/large_MWFdiff')

#         # Print out the LS_estimates and all the ones that wasserstein_distance is inputting
