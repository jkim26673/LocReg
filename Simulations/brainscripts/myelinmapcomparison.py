
# filetag = "origbrainfig_unfiltered"
filetag = "NESMAfiltered"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/comparingSNR180noisetoorig"
# filepath = data_path
# filetag = "Unfiltered"

# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_15Jan25"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25/brainfigs"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/brainfigs_unfiltered"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/filteredbrain"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/unfilterdbrain_1_21_25"
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def load_brain_data(brain_data_filepath, mask_filepath, estimates_filepath):
    # Load the brain data
    brain_data = scipy.io.loadmat(brain_data_filepath)["final_data_2"]
    
    # Load the mask
    BW = scipy.io.loadmat(mask_filepath)["new_BW"]
    
    # Load the estimates dataframe
    with open(estimates_filepath, "rb") as file:
        df = pickle.load(file)
    
    return brain_data, BW, df


def initialize_MWF_arrays():
    MWF_DP = np.zeros((313, 313))
    MWF_LC = np.zeros((313, 313))
    MWF_LR = np.zeros((313, 313))
    MWF_GCV = np.zeros((313, 313))
    MWF_OR = np.zeros((313, 313))
    MWF_LS = np.zeros((313, 313))
    
    return MWF_DP, MWF_LC, MWF_LR, MWF_GCV, MWF_OR, MWF_LS


def load_MWF_values(df, MWF_DP, MWF_LC, MWF_LR, MWF_GCV, MWF_OR, MWF_LS):
    for i, row in df.iterrows():
        x = row['X_val']  # Adjust for 0-based index
        y = row['Y_val']  # Adjust for 0-based index
        
        MWF_DP[x, y] = row['MWF_DP']
        MWF_LR[x, y] = row['MWF_LR']
        MWF_LC[x, y] = row['MWF_LC']
        MWF_GCV[x, y] = row['MWF_GCV']
        # MWF_OR[x, y] = row['MWF_OR']
        MWF_LS[x, y] = row['MWF_LS']
        
    return MWF_DP, MWF_LC, MWF_LR, MWF_GCV, MWF_OR, MWF_LS


def rotate_images(BW, MWF_GT, MWF_LR, MWF_LC, MWF_DP, MWF_GCV, MWF_LS):
    brain_MWF_GT = BW * MWF_GT
    brain_MWF_LR = BW * MWF_LR
    brain_MWF_LC = BW * MWF_LC
    brain_MWF_DP = BW * MWF_DP
    brain_MWF_GCV = BW * MWF_GCV
    brain_MWF_LS = BW * MWF_LS

    return brain_MWF_GT, brain_MWF_LR, brain_MWF_LC, brain_MWF_DP, brain_MWF_GCV, brain_MWF_LS

def compute_norms(MWF_filt_pixel, MWF_unfilt_slice):
    norm = np.linalg.norm(MWF_filt_slice - MWF_unfilt_slice) / np.linalg.norm(MWF_unfilt_slice)
    return norm

def compute_sum_abs_diff(MWF_filt_slice, MWF_unfilt_slice, BW):
    sum_abs_diff = np.sum(np.abs(MWF_filt_slice - MWF_unfilt_slice)) / np.sum(BW)
    return sum_abs_diff

def compute_pixelwise_l2_norm(slice1, slice2):
    """
    Compute the pixel-wise L2 norm error between two slices.
    :param slice1: First MWF slice (2D array).
    :param slice2: Second MWF slice (2D array).
    :return: 2D array of pixel-wise L2 norm errors.
    """
    # Ensure both slices have the same shape
    if slice1.shape != slice2.shape:
        raise ValueError("Slices must have the same shape.")
    
    # Compute the pixel-wise L2 norm error (sqrt((slice1 - slice2)^2))
    l2_norm_error_map = np.sqrt((slice1 - slice2) ** 2)
    
    return l2_norm_error_map

def plot_l2_norm_error(slice1, slice2, BW, filepath, slice1str, slice2str, title="Pixel-wise L2 Norm Error Map"):
    """
    Plot and save the pixel-wise L2 norm error map.
    :param error_map: 2D array of pixel-wise L2 norm errors.
    :param filepath: Path to save the plot.
    :param title: Title for the plot.
    """
    slice1 = BW * slice1
    slice2 = BW * slice2

    # Crop the slices for better visualization
    slice1 = slice1[79:228, 103:215]
    slice2 = slice2[79:228, 103:215]
    l2_norm_error_map = np.sqrt((slice1 - slice2) ** 2)
    title = f"L2 Norm Error: {slice2str} - {slice1str}"
    savepath = f"{filepath}/{slice2str}_minus_{slice1str}_L2norm_error_filetag.png"
    plt.figure()
    plt.title(title)
    plt.imshow(l2_norm_error_map, cmap='viridis', vmin=0, vmax=0.2)
    plt.xlabel('X Index (103 to 215)')
    plt.ylabel('Y Index (79 to 228)')
    plt.xticks(ticks=range(0, slice1.shape[1], 25), labels=range(103, 216, 25))
    plt.yticks(ticks=range(0, slice1.shape[0], 25), labels=range(79, 229, 25))
    plt.axis('on')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    # Save the plot
    plt.savefig(filepath)
    plt.close()

def compute_pixelwise_sad(slice1, slice2):
    """
    Compute the pixel-wise Sum of Absolute Differences (SAD) between two slices.
    :param slice1: First MWF slice (2D array).
    :param slice2: Second MWF slice (2D array).
    :return: 2D array of pixel-wise SAD.
    """
    # Ensure both slices have the same shape
    if slice1.shape != slice2.shape:
        raise ValueError("Slices must have the same shape.")
    
    # Compute the pixel-wise Sum of Absolute Differences (SAD)
    sad_map = np.abs(slice1 - slice2)
    
    return sad_map



def print_comparison_results(norm_DP_GT, norm_LR_GT, norm_LC_GT, norm_LS_GT, norm_GCV_GT, 
                             sum_abs_diff_DP_GT, sum_abs_diff_LR_GT, sum_abs_diff_LC_GT, 
                             sum_abs_diff_LS_GT, sum_abs_diff_GCV_GT):
    print("Norm based comparisons:")
    print(f"DP vs GT: {norm_DP_GT}")
    print(f"LR vs GT: {norm_LR_GT}")
    print(f"LC vs GT: {norm_LC_GT}")
    print(f"LS vs GT: {norm_LS_GT}")
    print(f"GCV vs GT: {norm_GCV_GT}")

    print("\nSum of absolute differences divided by BW:")
    print(f"DP vs GT: {sum_abs_diff_DP_GT}")
    print(f"LR vs GT: {sum_abs_diff_LR_GT}")
    print(f"LC vs GT: {sum_abs_diff_LC_GT}")
    print(f"LS vs GT: {sum_abs_diff_LS_GT}")
    print(f"GCV vs GT: {sum_abs_diff_GCV_GT}")

def plot_and_save(MWF_slice, BW, filepath, title, xcoord=None, ycoord=None):
    zoom_slice = BW * MWF_slice
    zoom_slice = MWF_slice[79:228, 103:215]
    plt.figure()
    plt.title(title)
    plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
    plt.xlabel('X Index (103 to 215)')
    plt.ylabel('Y Index (79 to 228)')
    plt.xticks(ticks=range(0, zoom_slice.shape[1], 25), labels=range(103, 216, 25))
    plt.yticks(ticks=range(0, zoom_slice.shape[0], 25), labels=range(79, 229, 25))
    plt.axis('on')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()

    if xcoord is not None and ycoord is not None:
        zoom_x = xcoord - 103
        zoom_y = ycoord - 79
        plt.scatter(zoom_x, zoom_y, color='red', s=10, label="Target (Red Dot)")

    plt.savefig(f"{filepath}/{title}_filetag.png")



def plot_difference(slice1, slice2, BW, filepath, slice1str, slice2str, xcoord=None, ycoord=None):
    # Apply the mask (BW)
    slice1 = BW * slice1
    slice2 = BW * slice2

    # Crop the slices for better visualization
    slice1 = slice1[79:228, 103:215]
    slice2 = slice2[79:228, 103:215]

    title = f"L2 Norm Error: {slice2str} - {slice1str}"
    savepath = f"{filepath}/{slice2str}_minus_{slice1str}_L2norm_error_filetag.png"
    plt.figure()
    plt.title(title)
    plt.imshow(l2_norm_error, cmap='viridis', vmin=0, vmax=0.2)
    plt.xlabel('X Index (103 to 215)')
    plt.ylabel('Y Index (79 to 228)')
    plt.xticks(ticks=range(0, slice1.shape[1], 25), labels=range(103, 216, 25))
    plt.yticks(ticks=range(0, slice1.shape[0], 25), labels=range(79, 229, 25))
    plt.axis('on')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    # Compute the difference between the two slices
    # Compute L2 Norm Error map
    l2_norm_error = compute_norms(slice1, slice2) # L2 norm error is the absolute difference for visualization purposes

    # Compute SAD map
    sad_map = compute_sum_abs_diff(slice1, slice2, BW) # SAD map is simply the absolute difference

    # Plot and save the difference (L2 Norm Error map)
    title = f"L2 Norm Error: {slice2str} - {slice1str}"
    savepath = f"{filepath}/{slice2str}_minus_{slice1str}_L2norm_error_filetag.png"
    plt.figure()
    plt.title(title)
    plt.imshow(l2_norm_error, cmap='viridis', vmin=0, vmax=0.2)
    plt.xlabel('X Index (103 to 215)')
    plt.ylabel('Y Index (79 to 228)')
    plt.xticks(ticks=range(0, slice1.shape[1], 25), labels=range(103, 216, 25))
    plt.yticks(ticks=range(0, slice1.shape[0], 25), labels=range(79, 229, 25))
    plt.axis('on')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()

    if xcoord is not None and ycoord is not None:
        plt.scatter(xcoord - 103, ycoord - 79, color='red', s=20, label="Target (Red Dot)")
        savepath = f"{filepath}/{slice2str}_minus_{slice1str}_L2norm_error_xcoord{xcoord}_ycoord{ycoord}_filetag.png"
    
    plt.savefig(savepath)
    plt.close()


def plot_sad_map(slice1, slice2, BW, filepath, slice1str, slice2str, title="Pixel-wise SAD Map"):
    """
    Plot and save the pixel-wise SAD map.
    :param sad_map: 2D array of pixel-wise SAD values.
    :param filepath: Path to save the plot.
    :param title: Title for the plot.
    """
    # Apply the mask (BW)
    slice1 = BW * slice1
    slice2 = BW * slice2

    # Crop the slices for better visualization
    slice1 = slice1[79:228, 103:215]
    slice2 = slice2[79:228, 103:215]
    sad_map = np.abs(slice1 - slice2)
    # Plot and save the SAD map
    title = f"SAD: {slice2str} - {slice1str}"
    savepath = f"{filepath}/{slice2str}_minus_{slice1str}_SAD_error.png"
    plt.figure()
    plt.title(title)
    plt.imshow(sad_map, cmap='viridis', vmin=0, vmax=0.2)
    plt.xlabel('X Index (103 to 215)')
    plt.ylabel('Y Index (79 to 228)')
    plt.xticks(ticks=range(0, slice1.shape[1], 25), labels=range(103, 216, 25))
    plt.yticks(ticks=range(0, slice1.shape[0], 25), labels=range(79, 229, 25))
    plt.axis('on')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig(savepath)
    plt.close()



def plot_l2_norm_error(slice1, slice2, BW, filepath, slice1str, slice2str, title="Pixel-wise L2 Norm Error Map"):
    """
    Plot and save the pixel-wise L2 norm error map.
    :param error_map: 2D array of pixel-wise L2 norm errors.
    :param filepath: Path to save the plot.
    :param title: Title for the plot.
    """
    slice1 = BW * slice1
    slice2 = BW * slice2
    # Crop the slices for better visualization
    slice1 = slice1[79:228, 103:215]
    slice2 = slice2[79:228, 103:215]
    l2_norm_error_map = np.sqrt((slice1 - slice2) ** 2)
    title = f"L2 Norm Error: {slice2str} - {slice1str}"
    savepath = f"{filepath}/{slice2str}_minus_{slice1str}_L2norm_error.png"
    plt.figure()
    plt.title(title)
    plt.imshow(l2_norm_error_map, cmap='viridis', vmin=0, vmax=0.2)
    plt.xlabel('X Index (103 to 215)')
    plt.ylabel('Y Index (79 to 228)')
    plt.xticks(ticks=range(0, slice1.shape[1], 25), labels=range(103, 216, 25))
    plt.yticks(ticks=range(0, slice1.shape[0], 25), labels=range(79, 229, 25))
    plt.axis('on')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    # Save the plot
    plt.savefig(filepath)
    plt.close()


# Example usage
brain_data_filepath = "/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat"
mask_filepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat"
filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/filteredbrain/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps29Jan25.pkl"
unfiltered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_unfiltered_noNESMA21Jan25.pkl"
filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25"


brain_data, BW, filtered_df = load_brain_data(brain_data_filepath, mask_filepath, filtered_estimates_filepath)
_, _, unfiltered_df = load_brain_data(brain_data_filepath, mask_filepath, unfiltered_estimates_filepath)

MWF_DP_filt, MWF_LC_filt, MWF_LR_filt, MWF_GCV_filt, MWF_OR_filt, MWF_LS_filt = initialize_MWF_arrays()
MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt = initialize_MWF_arrays()

MWF_DP_filt, MWF_LC_filt, MWF_LR_filt, MWF_GCV_filt, MWF_OR_filt, MWF_LS_filt = load_MWF_values(filtered_df, MWF_DP_filt, MWF_LC_filt, MWF_LR_filt, MWF_GCV_filt, MWF_OR_filt, MWF_LS_filt)
MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt = load_MWF_values(unfiltered_df, MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt)

brain_MWF_GT_filt, brain_MWF_LR_filt, brain_MWF_LC_filt, brain_MWF_DP_filt, brain_MWF_GCV_filt, brain_MWF_LS_filt = rotate_images(BW, MWF_DP_filt, MWF_LC_filt, MWF_LR_filt, MWF_GCV_filt, MWF_OR_filt, MWF_LS_filt)
brain_MWF_GT_unfilt, brain_MWF_LR_unfilt, brain_MWF_LC_unfilt, brain_MWF_DP_unfilt, brain_MWF_GCV_unfilt, brain_MWF_LS_unfilt = rotate_images(BW, MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt)

# Assuming MWF_filt_slices and MWF_unfilt_slices are lists of brain slices
MWF_filt_slices = [brain_MWF_LR_filt, brain_MWF_LC_filt, brain_MWF_DP_filt, brain_MWF_GCV_filt, brain_MWF_LS_filt]
MWF_unfilt_slices = [brain_MWF_LR_unfilt, brain_MWF_LC_unfilt, brain_MWF_DP_unfilt, brain_MWF_GCV_unfilt, brain_MWF_LS_unfilt]

slicename = ["LocReg", "L-Curve", "DP", "GCV", "NNLS"]
slicename_unfiltered = list(map(lambda name: f"Unfiltered_{name}", slicename))

# Initialize lists to store the results
norms = []
sum_abs_diffs = []

# Iterate over the corresponding elements of both lists
for i, (MWF_filt_slice, MWF_unfilt_slice) in enumerate(zip(MWF_filt_slices, MWF_unfilt_slices)):
    # Get the filtered and unfiltered slice names
    slice1str = slicename[i]
    slice2str = slicename_unfiltered[i]
    
    # # Call the compute_norms function with filtered and unfiltered slices
    # norm = compute_norms(MWF_filt_slice, MWF_unfilt_slice)
    
    # # Call the compute_sum_abs_diff function with filtered and unfiltered slices
    # sad = compute_sum_abs_diff(MWF_filt_slice, MWF_unfilt_slice, BW)
    
    # # Store the results in the lists
    # norms.append(norm)
    # sum_abs_diffs.append(sad)
    
    # Plot the difference between filtered and unfiltered slices with appropriate labels
    # Compute the pixel-wise SAD
    # sad_map = compute_pixelwise_sad(MWF_filt_slice, MWF_unfilt_slice)
    # Plot and save the SAD map
    # plot_sad_map(sad_map, "pixelwise_sad_map.png", title="Pixel-wise SAD Map")
    # plot_sad_map(MWF_filt_slice, MWF_unfilt_slice, BW, filepath, slice1str, slice2str, title="Pixel-wise SAD Map")
    plot_l2_norm_error(MWF_filt_slice, MWF_unfilt_slice, BW, filepath, slice1str, slice2str, title="Pixel-wise L2 Norm Error Map")
    # plot_difference(MWF_filt_slice, MWF_unfilt_slice, BW, filepath, slice1str, slice2str, xcoord=None, ycoord=None)



