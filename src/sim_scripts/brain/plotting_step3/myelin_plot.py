
# # filetag = "origbrainfig_unfiltered"
# import pickle
# import numpy as np
# import scipy
# import matplotlib.pyplot as plt
# from scipy.stats import wasserstein_distance

# filetag = "NESMAfiltered"
# # filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_SpanReg_curr_SNR_addition04Feb25.pkl"


# def load_brain_data(brain_data_filepath, mask_filepath, estimates_filepath):
#     # Load the brain data
#     brain_data = scipy.io.loadmat(brain_data_filepath)["final_data_2"]
    
#     # Load the mask
#     BW = scipy.io.loadmat(mask_filepath)["new_BW"]
    
#     # Load the estimates dataframe
#     with open(estimates_filepath, "rb") as file:
#         df = pickle.load(file)
    
#     return brain_data, BW, df


# def initialize_MWF_arrays():
#     MWF_DP = np.zeros((313, 313))
#     MWF_LC = np.zeros((313, 313))
#     MWF_LR = np.zeros((313, 313))
#     MWF_GCV = np.zeros((313, 313))
#     MWF_LS = np.zeros((313, 313))
#     ref_MWF = np.zeros((313, 313))
#     if ref == True:
#         return [MWF_DP, MWF_LC, MWF_LR, MWF_GCV, MWF_LS, ref_MWF]
#     else:
#         return [MWF_DP, MWF_LC, MWF_LR, MWF_GCV, MWF_LS]


# def load_MWF_values(df, MWF_list, ref = True):
#     if ref == True:
#         for i, row in df.iterrows():
#             x = row['X_val']  # Adjust for 0-based index
#             y = row['Y_val']  # Adjust for 0-based index
#             MWF_list[0][x, y] = row['MWF_DP']
#             MWF_list[1][x, y] = row['MWF_LC']
#             MWF_list[2][x, y] = row['MWF_LR']
#             MWF_list[3][x, y] = row['MWF_GCV']
#             MWF_list[4][x, y] = row['MWF_LS']
#             MWF_list[5][x, y] = row['ref_MWF']
#     else:
#         for i, row in df.iterrows():
#             x = row['X_val']  # Adjust for 0-based index
#             y = row['Y_val']  # Adjust for 0-based index
#             MWF_list[0][x, y] = row['MWF_DP']
#             MWF_list[1][x, y] = row['MWF_LC']
#             MWF_list[2][x, y] = row['MWF_LR']
#             MWF_list[3][x, y] = row['MWF_GCV']
#             MWF_list[4][x, y] = row['MWF_LS']
#     return MWF_list

# def rotate_images(BW, MWF_list):
#     MWF_list = [BW * MWF for MWF in MWF_list]
#     return MWF_list

# def plot_and_save2(MWF_slice, BW, filepath, strslice, xcoord=None, ycoord=None):
#     zoom_slice = BW * MWF_slice
#     zoom_slice = MWF_slice[79:228, 103:215]
#     plt.figure()
#     plt.title(f"strslice")
#     plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
#     plt.xlabel('X Index (103 to 215)')
#     plt.ylabel('Y Index (79 to 228)')
#     plt.xticks(ticks=range(0, zoom_slice.shape[1], 25), labels=range(103, 216, 25))
#     plt.yticks(ticks=range(0, zoom_slice.shape[0], 25), labels=range(79, 229, 25))
#     plt.axis('on')
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     if xcoord is not None and ycoord is not None:
#         zoom_x = xcoord - 103
#         zoom_y = ycoord - 79
#         plt.scatter(zoom_x, zoom_y, color='red', s=10, label="Target (Red Dot)")
#     plt.savefig(f"{filepath}/{strslice}.png")
#     plt.close()


# # Example usage
# brain_data_filepath = "/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat"
# mask_filepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat"
# # filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_SpanReg_curr_SNR_addition04Feb25.pkl"
# # filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_11Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_SpanReg_curr_SNR_addition11Feb25.pkl"

# # filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/Curr_SNR/est_table_xcoordlen_313_ycoordlen_313_unfiltered_noise_addition_recover_curr_SNR_map12Feb25.pkl"
# # filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/UnifSNR800/est_table_xcoordlen_313_ycoordlen_313_unfiltered_noise_addition_recover_SNR80012Feb25.pkl"
# # filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_16Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noise_addition_uniform_noise16Feb25.pkl"
# # filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_16Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noise_addition_uniform_noise16Feb25.pkl"
# filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/UnifSNR200/est_table_xcoordlen_313_ycoordlen_313_unfiltered_noise_addition_recover_SNR20017Feb25.pkl"
# # filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noise_addition_recover04Feb25.pkl"
# # unfiltered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_unfiltered_noNESMA21Jan25.pkl"
# # filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25"
# # filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/spanreg"
# # filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/Curr_SNR"
# # filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/UnifSNR800"
# # filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_16Feb25"
# # filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/testing"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/UnifSNR200"
# # filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/noiseaddition"
# brain_data, BW, filtered_df = load_brain_data(brain_data_filepath, mask_filepath, filtered_estimates_filepath)
# # _, _, unfiltered_df = load_brain_data(brain_data_filepath, mask_filepath, unfiltered_estimates_filepath)
# # ref = True
# ref= False
# # MWF_DP_filt, MWF_LC_filt, MWF_LR_filt, MWF_GCV_filt, MWF_LS_filt, MWF_ref_filt

# MWF_list = initialize_MWF_arrays()
# # MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt = initialize_MWF_arrays()

# MWF_list = load_MWF_values(filtered_df, MWF_list, ref)
# # MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt = load_MWF_values(unfiltered_df, MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt)

# MWF_list = rotate_images(BW, MWF_list)
# # brain_MWF_GT_unfilt, brain_MWF_LR_unfilt, brain_MWF_LC_unfilt, brain_MWF_DP_unfilt, brain_MWF_GCV_unfilt, brain_MWF_LS_unfilt = rotate_images(BW, MWF_DP_unfilt, MWF_LC_unfilt, MWF_LR_unfilt, MWF_GCV_unfilt, MWF_OR_unfilt, MWF_LS_unfilt)

# # Assuming MWF_filt_slices and MWF_unfilt_slices are lists of brain slices
# MWF_filt_slices = MWF_list
# # MWF_unfilt_slices = [brain_MWF_LR_unfilt, brain_MWF_LC_unfilt, brain_MWF_DP_unfilt, brain_MWF_GCV_unfilt, brain_MWF_LS_unfilt]
# if ref == True:
#     slicename = [ "DP",  "L-Curve", "LocReg", "GCV", "NNLS", "Reference"]
#     slicename_unfiltered = list(map(lambda name: f"Unfiltered_{name}", slicename))
# else:
#     slicename = [ "DP",  "L-Curve", "LocReg", "GCV", "NNLS"]
#     slicename_unfiltered = list(map(lambda name: f"Unfiltered_{name}", slicename))

# # Initialize lists to store the results
# norms = []
# sum_abs_diffs = []

# def plot_and_save2(MWF_slice, BW, filepath, str, xcoord=None, ycoord=None):
#     zoom_slice = BW * MWF_slice
#     zoom_slice = MWF_slice[79:228, 103:215]
#     plt.figure()
#     plt.title(f"{str}")
#     plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
#     plt.xlabel('X Index (103 to 215)')
#     plt.ylabel('Y Index (79 to 228)')
#     plt.xticks(ticks=range(0, zoom_slice.shape[1], 25), labels=range(103, 216, 25))
#     plt.yticks(ticks=range(0, zoom_slice.shape[0], 25), labels=range(79, 229, 25))
#     plt.axis('on')
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()

#     if xcoord is not None and ycoord is not None:
#         zoom_x = xcoord - 103
#         zoom_y = ycoord - 79
#         plt.scatter(zoom_x, zoom_y, color='red', s=10, label="Target (Red Dot)")

#     plt.savefig(f"{filepath}/{str}.png")

# # Iterate over the corresponding elements of both lists
# # for i, (MWF_filt_slice, MWF_unfilt_slice) in enumerate(zip(MWF_filt_slices, MWF_unfilt_slices)):
# for i, MWF_filt_slice in enumerate(MWF_filt_slices):
#     # Get the filtered and unfiltered slice names
#     slice1str = slicename[i]
#     # slice2str = slicename_unfiltered[i]
#     plot_and_save2(MWF_filt_slice, BW, filepath, slice1str, xcoord=None, ycoord=None)
    
#     # # Call the compute_norms function with filtered and unfiltered slices
#     # norm = compute_norms(MWF_filt_slice, MWF_unfilt_slice)
    
#     # # Call the compute_sum_abs_diff function with filtered and unfiltered slices
#     # sad = compute_sum_abs_diff(MWF_filt_slice, MWF_unfilt_slice, BW)
    
#     # # Store the results in the lists
#     # norms.append(norm)
#     # sum_abs_diffs.append(sad)
    
#     # Plot the difference between filtered and unfiltered slices with appropriate labels
#     # Compute the pixel-wise SAD
#     # sad_map = compute_pixelwise_sad(MWF_filt_slice, MWF_unfilt_slice)
#     # Plot and save the SAD map
#     # plot_sad_map(sad_map, "pixelwise_sad_map.png", title="Pixel-wise SAD Map")
#     # plot_sad_map(MWF_filt_slice, MWF_unfilt_slice, BW, filepath, slice1str, slice2str, title="Pixel-wise SAD Map")
#     # plot_l2_norm_error(MWF_filt_slice, MWF_unfilt_slice, BW, filepath, slice1str, slice2str, title="Pixel-wise L2 Norm Error Map")
#     # plot_difference(MWF_filt_slice, MWF_unfilt_slice, BW, filepath, slice1str, slice2str, xcoord=None, ycoord=None)





# filetag = "origbrainfig_unfiltered"
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import os
import glob
import pandas as pd

filetag = "NESMAfiltered"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_SpanReg_curr_SNR_addition04Feb25.pkl"


def load_brain_data(brain_data_filepath, mask_filepath, estimates_filepath):
    # Load the brain data
    brain_data = scipy.io.loadmat(brain_data_filepath)["final_data_2"]
    
    # Load the mask
    BW = scipy.io.loadmat(mask_filepath)["new_BW"]
    
    # Load the estimates dataframe
    # with open(estimates_filepath, "rb") as file:
    #     df = pickle.load(file)
    # print(df)
    # df = pd.DataFrame(df)

    df_list = []

    for filename in os.listdir(estimates_filepath):
        if filename.startswith('temp_checkpoint_') and filename.endswith('.pkl'):
            file_path = os.path.join(estimates_filepath, filename)
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)  # <- assumes dictionary
                df = pd.DataFrame(data_dict)  
                df_list.append(df)
    # filepath2 = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_08Jun25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN08Jun25.pkl"
    # with open(filepath2, 'rb') as file2:
    #     df2 = pickle.load(file2)

    # folder = "/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925"

    # # Find all pickle files that start with "temp_checkpoint"
    # pkl_files = glob.glob(os.path.join(folder, "temp_checkpoint*.pkl"))

    # # Load each pickle, convert list of dicts to DataFrame
    # dfs = [pd.DataFrame(pd.read_pickle(f)) for f in pkl_files]

    # # Concatenate all DataFrames
    # filtered_df = pd.concat(dfs, ignore_index=True)

    # print(f"Combined shape: {filtered_df.shape}")
    # df_list.append(filtered_df)
    # df = pd.concat(df_list, ignore_index=True)
    # df.to_pickle(f'{estimates_filepath}\combinedestimates.pkl')

    # filepath2 = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25.pkl"
    # filepath2 = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25_wassscore_newMWF.pkl"
    # filepath2 = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/results_06Jun25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN06Jun25_processed_wassscores_newMWF.pkl"
    # filepath2 = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep2625/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN26Sep25.pkl"
    filepath2 = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25_modifiedMWF.pkl"
    with open(filepath2, 'rb') as file2:
        df = pickle.load(file2)

    # folder = "/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925"

    # # Find all pickle files that start with "temp_checkpoint"
    # pkl_files = glob.glob(os.path.join(folder, "temp_checkpoint*.pkl"))

    # # Load each pickle, convert list of dicts to DataFrame
    # dfs = [pd.DataFrame(pd.read_pickle(f)) for f in pkl_files]

    # # Concatenate all DataFrames
    # filtered_df = pd.concat(dfs, ignore_index=True)

    # print(f"Combined shape: {filtered_df.shape}")
    # df_list.append(filtered_df)
    # df = pd.concat(df_list, ignore_index=True)
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
            # MWF_list[0][x, y] = row['MWF_Ref']
            # MWF_list[1][x, y] = row['MWF_GCV']
            # MWF_list[2][x, y] = row['MWF_LR']
            # MWF_list[3][x, y] = row['MWF_LR1D']
            # MWF_list[4][x, y] = row['MWF_LR2D']
            # MWF_list[5][x, y] = row['MWF_UPEN']
            MWF_list[0][x, y] = row['sm_MWF_ref']
            MWF_list[1][x, y] = row['sm_MWF_GCV']
            MWF_list[2][x, y] = row['sm_MWF_LR']
            MWF_list[3][x, y] = row['sm_MWF_LR1D']
            MWF_list[4][x, y] = row['sm_MWF_LR2D']
            MWF_list[5][x, y] = row['sm_MWF_UPEN']
    else:
        for i, row in df.iterrows():
            x = row['X_val']  # Adjust for 0-based index
            y = row['Y_val']  # Adjust for 0-based index
            # MWF_list[0][x, y] = row['MWF_GCV']
            # MWF_list[1][x, y] = row['MWF_LR']
            # MWF_list[2][x, y] = row['MWF_LR1D']
            # MWF_list[3][x, y] = row['MWF_LR2D']
            # MWF_list[4][x, y] = row['MWF_UPEN']
            MWF_list[0][x, y] = row['sm_MWF_GCV']
            MWF_list[1][x, y] = row['sm_MWF_LR']
            MWF_list[2][x, y] = row['sm_MWF_LR1D']
            MWF_list[3][x, y] = row['sm_MWF_LR2D']
            MWF_list[4][x, y] = row['sm_MWF_UPEN']
    return MWF_list

def rotate_images(BW, MWF_list):
    MWF_list = [BW * MWF for MWF in MWF_list]
    return MWF_list

# def plot_and_save2(MWF_slice, BW, filepath, strslice, xcoord=None, ycoord=None):
#     zoom_slice = BW * MWF_slice
#     zoom_slice = MWF_slice[79:228, 103:215]
#     plt.figure()
#     plt.title(f"strslice")
#     plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
#     plt.xlabel('X Index (103 to 215)')
#     plt.ylabel('Y Index (79 to 228)')
#     plt.xticks(ticks=range(0, zoom_slice.shape[1], 25), labels=range(103, 216, 25))
#     plt.yticks(ticks=range(0, zoom_slice.shape[0], 25), labels=range(79, 229, 25))
#     plt.axis('on')
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     if xcoord is not None and ycoord is not None:
#         zoom_x = xcoord - 103
#         zoom_y = ycoord - 79
#         plt.scatter(zoom_x, zoom_y, color='red', s=10, label="Target (Red Dot)")
#     plt.savefig(f"{filepath}/{strslice}.png")
#     plt.close()


# Example usage

brain_data_filepath = r"/Users/kimjosy/Downloads/LocReg/data/brain/processed/cleaned_brain_data.mat"
mask_filepath = r"/Users/kimjosy/Downloads/LocReg/data/brain/masks/new_mask.mat"
# brain_data_filepath = r"/Users/kimjosy/Downloads/LocReg/data/brain/processed/cleaned_brain_data.mat"
# mask_filepath = r"/Users/kimjosy/Downloads/LocReg/data/brain/masks/new_mask.mat"
# brain_data_filepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/data/Brain/processed/cleaned_brain_data.mat"
# mask_filepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/data/Brain/masks/new_mask.mat"
# filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_SpanReg_curr_SNR_addition04Feb25.pkl"
# filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_11Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_SpanReg_curr_SNR_addition11Feb25.pkl"

# filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/Curr_SNR/est_table_xcoordlen_313_ycoordlen_313_unfiltered_noise_addition_recover_curr_SNR_map12Feb25.pkl"
# filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/UnifSNR800/est_table_xcoordlen_313_ycoordlen_313_unfiltered_noise_addition_recover_SNR80012Feb25.pkl"
# filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_16Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noise_addition_uniform_noise16Feb25.pkl"
# filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_16Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noise_addition_uniform_noise16Feb25.pkl"
# filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/UnifSNR200/est_table_xcoordlen_313_ycoordlen_313_unfiltered_noise_addition_recover_SNR20017Feb25.pkl"
# filtered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_noise_addition_recover04Feb25.pkl"
# unfiltered_estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_unfiltered_noNESMA21Jan25.pkl"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/spanreg"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/Curr_SNR"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_12Feb25/unfiltered/UnifSNR800"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_16Feb25"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/testing"
# savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14May25"
# savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_22May25"
# savepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_30May25"
# savepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925/new_MWF_plot"
# savepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/results_06Jun25/new_MWF_plot"
# savepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep2625/MWF_plot"
savepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925/modified_MWF_plot"
# savepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925/orig_MWF_plot"
# savepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/Sep1925"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_04Feb25/noiseaddition"
# filtered_estimates_filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_02May25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN02May25.pkl"
# filtered_estimates_filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14May25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN14May25.pkl"
# filtered_estimates_filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_22May25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN22May25.pkl"
# filtered_estimates_filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_30May25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN30May25.pkl"
# filtered_estimates_filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_08Jun25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN08Jun25.pkl"

# filtered_estimates_filepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_06Jun25\temp_checkpoint_7.pkl"

import os
import pickle
import pandas as pd

# folder_path = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925"
# folder_path = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/Sep1925"
folder_path = savepath

brain_data, BW, filtered_df = load_brain_data(brain_data_filepath, mask_filepath, folder_path)
# _, _, unfiltered_df = load_brain_data(brain_data_filepath, mask_filepath, unfiltered_estimates_filepath)
# ref = True
ref = True
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
    plt.title(f"{str} Myelin Map")
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

# Iterate over the corresponding elements of both lists
# for i, (MWF_filt_slice, MWF_unfilt_slice) in enumerate(zip(MWF_filt_slices, MWF_unfilt_slices)):
for i, MWF_filt_slice in enumerate(MWF_filt_slices):
    # Get the filtered and unfiltered slice names
    slice1str = slicename[i]
    # slice2str = slicename_unfiltered[i]
    plot_and_save2(MWF_filt_slice, BW, savepath, slice1str, xcoord=None, ycoord=None)
    
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
    # plot_l2_norm_error(MWF_filt_slice, MWF_unfilt_slice, BW, filepath, slice1str, slice2str, title="Pixel-wise L2 Norm Error Map")
    # plot_difference(MWF_filt_slice, MWF_unfilt_slice, BW, filepath, slice1str, slice2str, xcoord=None, ycoord=None)



