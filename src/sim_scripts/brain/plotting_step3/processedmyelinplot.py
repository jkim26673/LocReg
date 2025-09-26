import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

import os
import glob
import pandas as pd



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
    plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax = 0.20)
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

# def load_brain_data(brain_data_filepath, mask_filepath, estimates_filepath):
#     # Load the brain data
#     brain_data = scipy.io.loadmat(brain_data_filepath)["final_data_2"]
    
#     # Load the mask
#     BW = scipy.io.loadmat(mask_filepath)["new_BW"]
    
#     # Load the estimates dataframe
#     with open(estimates_filepath, "rb") as file:
#         df = pickle.load(file)
#     print(df)
#     return brain_data, BW, df
def load_brain_data(brain_data_filepath, mask_filepath):
    # Load the brain data
    brain_data = scipy.io.loadmat(brain_data_filepath)["final_data_2"]
    # Load the mask
    BW = scipy.io.loadmat(mask_filepath)["new_BW"]
    # Load the estimates dataframe
    # with open(estimates_filepath, "rb") as file:
    #     df = pickle.load(file)
    # print(df)
    return brain_data, BW

def initialize_MWF_arrays():
    processed_slice = np.zeros((313, 313))
    return [processed_slice]

def load_MWF_values(df, MWF_list, ref = True):
    for i, row in df.iterrows():
        x = row['X_val']  # Adjust for 0-based index
        y = row['Y_val']  # Adjust for 0-based index
        MWF_list[0][x, y] = 1
    return MWF_list

def rotate_images(BW, MWF_list):
    MWF_list = [BW * MWF for MWF in MWF_list]
    return MWF_list

brain_data_filepath = r"/Users/kimjosy/Downloads/LocReg/data/brain/processed/cleaned_brain_data.mat"
mask_filepath = r"/Users/kimjosy/Downloads/LocReg/data/brain/masks/new_mask.mat"
savepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925"

# filtered_estimates_filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_06Jun25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN06Jun25.pkl"

brain_data, BW = load_brain_data(brain_data_filepath, mask_filepath)

folder = "/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925"

# Find all pickle files that start with "temp_checkpoint"
pkl_files = glob.glob(os.path.join(folder, "temp_checkpoint*.pkl"))

# Load each pickle, convert list of dicts to DataFrame
dfs = [pd.DataFrame(pd.read_pickle(f)) for f in pkl_files]

# Concatenate all DataFrames
filtered_df = pd.concat(dfs, ignore_index=True)

print(f"Combined shape: {filtered_df.shape}")

ref = True
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
    slicename = [ "Processed Slice"]
    slicename_unfiltered = list(map(lambda name: f"{name}", slicename))

for i, MWF_filt_slice in enumerate(MWF_filt_slices):
    # Get the filtered and unfiltered slice names
    slice1str = slicename[i]
    # slice2str = slicename_unfiltered[i]
    plot_and_save2(MWF_filt_slice, BW, savepath, slice1str, xcoord=None, ycoord=None)
    