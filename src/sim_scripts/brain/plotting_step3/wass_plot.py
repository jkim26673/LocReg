import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import os
import glob
import pandas as pd

filetag = "NESMAfiltered"

def load_brain_data(brain_data_filepath, mask_filepath, estimates_filepath):
    # Load the brain data
    brain_data = scipy.io.loadmat(brain_data_filepath)["final_data_2"]
    
    # Load the mask
    BW = scipy.io.loadmat(mask_filepath)["new_BW"]

    df_list = []
    # filepath2 = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25.pkl"
    # filepath2 = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25_wassscores.pkl"
    filepath2 = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/results_06Jun25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN06Jun25_processed_wassscores.pkl"

    with open(filepath2, 'rb') as file2:
        filtered_df = pickle.load(file2)
    df_list.append(filtered_df)
    df = pd.concat(df_list, ignore_index=True)
    # df.to_pickle(f'{estimates_filepath}\combinedestimates.pkl')
    return brain_data, BW, df


def initialize_MWF_arrays():
    MWF_Ref = np.zeros((313, 313))
    MWF_GCV = np.zeros((313, 313))
    MWF_LR = np.zeros((313, 313))
    MWF_LR1D = np.zeros((313, 313))
    MWF_LR2D = np.zeros((313, 313))
    MWF_UPEN = np.zeros((313, 313))
    if ref == True:
        return [MWF_Ref, MWF_GCV, MWF_LR, MWF_LR1D, MWF_LR2D, MWF_UPEN]
    else:
        return [MWF_GCV, MWF_LR, MWF_LR1D, MWF_LR2D, MWF_UPEN]

def load_MWF_values(df, MWF_list, ref = True):
    if ref == True:
        for i, row in df.iterrows():
            x = row['X_val']  # Adjust for 0-based index
            y = row['Y_val']  # Adjust for 0-based index
            MWF_list[0][x, y] = row['MWF_Ref']
            MWF_list[1][x, y] = row['MWF_GCV']
            MWF_list[2][x, y] = row['MWF_LR']
            MWF_list[3][x, y] = row['MWF_LR1D']
            MWF_list[4][x, y] = row['MWF_LR2D']
            MWF_list[5][x, y] = row['MWF_UPEN']
    else:
        for i, row in df.iterrows():
            x = row['X_val']  # Adjust for 0-based index
            y = row['Y_val']  # Adjust for 0-based index
            MWF_list[0][x, y] = row['Wass_GCV']
            MWF_list[1][x, y] = row['Wass_LR']
            MWF_list[2][x, y] = row['Wass_LR1D']
            MWF_list[3][x, y] = row['Wass_LR2D']
            MWF_list[4][x, y] = row['Wass_UPEN']
    return MWF_list

def rotate_images(BW, MWF_list):
    MWF_list = [BW * MWF for MWF in MWF_list]
    return MWF_list


# Example usage

brain_data_filepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/data/brain/processed/cleaned_brain_data.mat"
mask_filepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/data/brain/masks/new_mask.mat"

# savepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/Sep1925/wass_score_plot"
savepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/results_06Jun25/wass_score_plot"

import os
import pickle
import pandas as pd

# folder_path = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/Sep1925/wass_score_plot"

folder_path = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/results_06Jun25/wass_score_plot"

brain_data, BW, filtered_df = load_brain_data(brain_data_filepath, mask_filepath, folder_path)
# _, _, unfiltered_df = load_brain_data(brain_data_filepath, mask_filepath, unfiltered_estimates_filepath)
# ref = True
ref = False
# MWF_DP_filt, MWF_LC_filt, MWF_LR_filt, MWF_GCV_filt, MWF_LS_filt, MWF_ref_filt

MWF_list = initialize_MWF_arrays()
# MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt = initialize_MWF_arrays()

MWF_list = load_MWF_values(filtered_df, MWF_list, ref)
# MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt = load_MWF_values(unfiltered_df, MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt)

MWF_list = rotate_images(BW, MWF_list)
# brain_MWF_GT_unfilt, brain_MWF_LR_unfilt, brain_MWF_LC_unfilt, brain_MWF_DP_unfilt, brain_MWF_GCV_unfilt, brain_MWF_LS_unfilt = rotate_images(BW, MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt)

# Assuming MWF_filt_slices and MWF_unfilt_slices are lists of brain slices
MWF_filt_slices = MWF_list
# MWF_unfilt_slices = [brain_MWF_LR_unfilt, brain_MWF_LC_unfilt, brain_MWF_DP_unfilt, brain_MWF_GCV_unfilt, brain_MWF_LS_unfilt]
if ref == True:
    slicename = [ "Reference",  "GCV", "LocReg", "LocReg 1st Derivative", "LocReg 2nd Derivative", "UPEN"]
    slicename_unfiltered = list(map(lambda name: f"{name}", slicename))
else:
    slicename = ["GCV", "LocReg", "LocReg 1st Derivative", "LocReg 2nd Derivative", "UPEN"]
    slicename_unfiltered = list(map(lambda name: f"{name}", slicename))

# Initialize lists to store the results
norms = []
sum_abs_diffs = []

def plot_and_save2(MWF_slice, BW, savepath, str, xcoord=None, ycoord=None):
    zoom_slice = BW * MWF_slice
    # zoom_slice = MWF_slice[79:228, 103:215]
    xinit =75
    xfin =245    
    yinit =103
    yfin =215
    zoom_slice = MWF_slice[xinit:xfin, yinit:yfin]
    plt.figure()
    plt.title(f"{str}")
    plt.imshow(zoom_slice, cmap='viridis', vmin=0)
    plt.xlabel(f'X Index ({yinit} to {yfin})')
    plt.ylabel(f'Y Index ({xinit} to {xfin})')
    plt.xticks(ticks=range(0, zoom_slice.shape[1], 25), labels=range(yinit, yfin, 25))
    plt.yticks(ticks=range(0, zoom_slice.shape[0], 25), labels=range(xinit, xfin, 25))
    plt.axis('on')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()

    if xcoord is not None and ycoord is not None:
        zoom_x = xcoord - yinit
        zoom_y = ycoord - xinit
        plt.scatter(zoom_x, zoom_y, color='red', s=10, label="Target (Red Dot)")

    plt.savefig(f"{savepath}/{str}.png")

# Iterate over the corresponding elements of both lists
# for i, (MWF_filt_slice, MWF_unfilt_slice) in enumerate(zip(MWF_filt_slices, MWF_unfilt_slices)):
for i, MWF_filt_slice in enumerate(MWF_filt_slices):
    # Get the filtered and unfiltered slice names
    slice1str = slicename[i]
    # slice2str = slicename_unfiltered[i]
    plot_and_save2(MWF_filt_slice, BW, savepath, slice1str, xcoord=None, ycoord=None)


