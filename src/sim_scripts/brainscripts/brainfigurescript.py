# import pickle
# import scipy.io
# from scipy.ndimage import rotate
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import wasserstein_distance as wass

# brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat")["final_data_2"]

# maskfilepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/masks/mask_2.mat"
# # estimatesfilepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results/est_table_xcoordlen_313_ycoordlen_31328Nov24.pkl"
# estimatesfilepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results/est_table_xcoordlen_313_ycoordlen_313_withLS_OR30Nov24.pkl"
# with open (estimatesfilepath, "rb") as file:
#     df = pickle.load(file)
# # MWF_DP = df["MWF_DP"]
# # MWF_LR = df["MWF_LR"]
# # MWF_LC = df["MWF_LC"]
# # MWF_GCV = df["MWF_GCV"]
# # MWF_GT = MWF_GCV
# BW = scipy.io.loadmat(maskfilepath)["BW"]

# MWF_DP = np.zeros((313,313))
# MWF_LC = np.zeros((313,313))
# MWF_GCV = np.zeros((313,313))
# MWF_LR = np.zeros((313,313))
# MWF_OR = np.zeros((313,313))
# MWF_LS = np.zeros((313,313))

# for i, row in df.iterrows():
#     x = row['X_val'] # Adjust for 0-based index
#     y = row['Y_val'] # Adjust for 0-based index
#     MWF_DP_val = row['MWF_DP']  # Directly use the numeric MWF_DP value
#     MWF_LC_val = row['MWF_LC']  # Directly use the numeric MWF_DP value
#     MWF_LR_val = row['MWF_LR']  # Directly use the numeric MWF_DP value
#     MWF_GCV_val = row['MWF_GCV']  # Directly use the numeric MWF_DP value
#     MWF_OR_val = row['MWF_OR']  # Directly use the numeric MWF_DP value
#     MWF_LS_val = row['MWF_LS']  # Directly use the numeric MWF_DP value

#     # Assign the MWF_DP value to the grid
#     MWF_DP[x, y] = MWF_DP_val
#     MWF_LR[x, y] = MWF_LR_val
#     MWF_LC[x, y] = MWF_LC_val
#     MWF_GCV[x, y] = MWF_GCV_val
#     MWF_OR[x, y] = MWF_OR_val
#     MWF_LS[x, y] = MWF_LS_val

# MWF_GT = MWF_GCV
# # Assuming MWF_GT is a NumPy array representing the image
# brain_MWF_GT = BW @ rotate(MWF_GT, 275, reshape=False)
# brain_MWF_LR = BW @ rotate(MWF_LR, 275, reshape=False)
# brain_MWF_LC = BW @ rotate(MWF_LC, 275, reshape=False)
# brain_MWF_DP = BW @ rotate(MWF_DP, 275, reshape=False)
# brain_MWF_GCV = BW @ rotate(MWF_GCV, 275, reshape=False)
# brain_MWF_LS = BW @ rotate(MWF_LS, 275, reshape=False)
# # brain_MWF_OR = BW @ rotate(MWF_OR, 275, reshape=False)

# norm_DP_GT = np.linalg.norm(brain_MWF_DP - brain_MWF_GT) / np.linalg.norm(brain_MWF_GT)
# norm_LR_GT = np.linalg.norm(brain_MWF_LR - brain_MWF_GT) / np.linalg.norm(brain_MWF_GT)
# norm_LC_GT = np.linalg.norm(brain_MWF_LC - brain_MWF_GT) / np.linalg.norm(brain_MWF_GT)
# norm_LS_GT = np.linalg.norm(brain_MWF_LS - brain_MWF_GT) / np.linalg.norm(brain_MWF_GT)
# # norm_OR_GT = np.linalg.norm(brain_MWF_OR - brain_MWF_GT) / np.linalg.norm(brain_MWF_GT)
# norm_GCV_GT = np.linalg.norm(brain_MWF_GCV - brain_MWF_GT) / np.linalg.norm(brain_MWF_GT)

# # Sum of absolute differences divided by the sum of BW
# sum_abs_diff_DP_GT = np.sum(np.abs(brain_MWF_DP - brain_MWF_GT)) / np.sum(BW)
# sum_abs_diff_LR_GT = np.sum(np.abs(brain_MWF_LR - brain_MWF_GT)) / np.sum(BW)
# sum_abs_diff_LC_GT = np.sum(np.abs(brain_MWF_LC - brain_MWF_GT)) / np.sum(BW)
# sum_abs_diff_LS_GT = np.sum(np.abs(brain_MWF_LS - brain_MWF_GT)) / np.sum(BW)
# # sum_abs_diff_OR_GT = np.sum(np.abs(brain_MWF_OR - brain_MWF_GT)) / np.sum(BW)
# sum_abs_diff_GCV_GT = np.sum(np.abs(brain_MWF_GCV - brain_MWF_GT)) / np.sum(BW)

# # Print the results
# print("Norm based comparisons:")
# print(f"DP vs GT: {norm_DP_GT}")
# print(f"LR vs GT: {norm_LR_GT}")
# print(f"LC vs GT: {norm_LC_GT}")
# print(f"LS vs GT: {norm_LS_GT}")
# # print(f"OR vs GT: {norm_OR_GT}")
# print(f"GCV vs GT: {norm_GCV_GT}")

# print("\nSum of absolute differences divided by BW:")
# print(f"DP vs GT: {sum_abs_diff_DP_GT}")
# print(f"LR vs GT: {sum_abs_diff_LR_GT}")
# print(f"LC vs GT: {sum_abs_diff_LC_GT}")
# print(f"LS vs GT: {sum_abs_diff_LS_GT}")
# # print(f"OR vs GT: {sum_abs_diff_OR_GT}")
# print(f"GCV vs GT: {sum_abs_diff_GCV_GT}")

# filepath= "/home/kimjosy/LocReg_Regularization-1/data/Brain/figures"
# # Assuming DP_zoom is a NumPy array
# # You can slice the array as in MATLAB
# DP_zoom_slice = MWF_DP[79:228, 103:215]  # Python is 0-indexed, so adjust the indices
# plt.figure()
# plt.title("DP")
# plt.imshow(DP_zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
# plt.xlabel('X Index (103 to 215)')
# plt.ylabel('Y Index (79 to 228)')
# plt.xticks(ticks=range(0, DP_zoom_slice.shape[1], 25), labels=range(103, 216, 25))
# plt.yticks(ticks=range(0, DP_zoom_slice.shape[0], 25), labels=range(79, 229, 25))
# plt.axis('on')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar()
# plt.savefig(f"{filepath}/DPbrainfig")

# LR_zoom_slice = MWF_LR[79:228, 103:215]  # Python is 0-indexed, so adjust the indices
# plt.figure()
# plt.title("LocReg")
# plt.imshow(LR_zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
# plt.xlabel('X Index (103 to 215)')
# plt.ylabel('Y Index (79 to 228)')
# plt.xticks(ticks=range(0, LR_zoom_slice.shape[1], 25), labels=range(103, 216, 25))
# plt.yticks(ticks=range(0, LR_zoom_slice.shape[0], 25), labels=range(79, 229, 25))
# plt.axis('on')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar()
# plt.savefig(f"{filepath}/LRbrainfig")

# LC_zoom_slice = MWF_LC[79:228, 103:215]  # Python is 0-indexed, so adjust the indices
# plt.figure()
# plt.title("L-Curve")
# plt.imshow(LC_zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
# plt.xlabel('X Index (103 to 215)')
# plt.ylabel('Y Index (79 to 228)')
# plt.xticks(ticks=range(0, LC_zoom_slice.shape[1], 25), labels=range(103, 216, 25))
# plt.yticks(ticks=range(0, LC_zoom_slice.shape[0], 25), labels=range(79, 229, 25))
# plt.axis('on')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar()
# plt.savefig(f"{filepath}/LCbrainfig")

# GCV_zoom_slice = MWF_GCV[79:228, 103:215]  # Python is 0-indexed, so adjust the indices
# plt.figure()
# plt.title("GCV")
# plt.imshow(GCV_zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
# plt.xlabel('X Index (103 to 215)')
# plt.ylabel('Y Index (79 to 228)')
# plt.xticks(ticks=range(0, GCV_zoom_slice.shape[1], 25), labels=range(103, 216, 25))
# plt.yticks(ticks=range(0, GCV_zoom_slice.shape[0], 25), labels=range(79, 229, 25))
# plt.axis('on')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar()
# plt.savefig(f"{filepath}/GCVbrainfig")

# LS_zoom_slice = MWF_LS[79:228, 103:215]  # Python is 0-indexed, so adjust the indices
# plt.figure()
# plt.title("Non-Negative Least Squares")
# plt.imshow(LS_zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
# plt.xlabel('X Index (103 to 215)')
# plt.ylabel('Y Index (79 to 228)')
# plt.xticks(ticks=range(0, LS_zoom_slice.shape[1], 25), labels=range(103, 216, 25))
# plt.yticks(ticks=range(0, LS_zoom_slice.shape[0], 25), labels=range(79, 229, 25))
# plt.axis('on')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar()
# plt.savefig(f"{filepath}/LSbrainfig")

# slice2 = DP_zoom_slice
# slice1 = LR_zoom_slice
# diff = slice2 - slice1
# slice2str = "DP"
# slice1str = "LocReg"
# title = f"{slice2str} - {slice1str}"

# plt.figure()
# plt.title(title)
# plt.imshow(diff, cmap='viridis', vmin=0, vmax=0.2)
# plt.xlabel('X Index (103 to 215)')
# plt.ylabel('Y Index (79 to 228)')
# plt.xticks(ticks=range(0, slice1.shape[1], 25), labels=range(103, 216, 25))
# plt.yticks(ticks=range(0, slice1.shape[0], 25), labels=range(79, 229, 25))
# plt.axis('on')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar()
# plt.savefig(f"{filepath}/{slice2str}_minus_{slice1str}_brainfig")
# # diffslice = 


#UNCOMMETN HERE

# import pickle
# import scipy.io
# from scipy.ndimage import rotate
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import wasserstein_distance as wass

# filetag = "noisybrainfigSNR300"
# class NoisyBrainAnalysis:
#     def __init__(self, brain_data_filepath, mask_filepath, estimates_filepath):
#         # Load the brain data
#         self.brain_data = scipy.io.loadmat(brain_data_filepath)["final_data_2"]
        
#         # Load the mask
#         self.BW = scipy.io.loadmat(mask_filepath)["BW"]
        
#         # Load the estimates dataframe
#         with open(estimates_filepath, "rb") as file:
#             self.df = pickle.load(file)
        
#         # Initialize MWF arrays
#         self.MWF_DP = np.zeros((313, 313))
#         self.MWF_LC = np.zeros((313, 313))
#         self.MWF_LR = np.zeros((313, 313))
#         self.MWF_GCV = np.zeros((313, 313))
#         self.MWF_OR = np.zeros((313, 313))
#         self.MWF_LS = np.zeros((313, 313))

#         self.load_MWF_values()

#         # Assume MWF_GT is the MWF_GCV by default
#         self.MWF_GT = self.MWF_GCV
        
#         # Perform rotations
#         self.rotate_images()

#     def load_MWF_values(self):
#         """Load MWF values into respective arrays."""
#         for i, row in self.df.iterrows():
#             x = row['X_val']  # Adjust for 0-based index
#             y = row['Y_val']  # Adjust for 0-based index
            
#             # # Assign the MWF values to the grid
#             # self.MWF_DP[x, y] = row['MWF_DP']
#             # self.MWF_LR[x, y] = row['MWF_LR']
#             # self.MWF_LC[x, y] = row['MWF_LC']
#             # self.MWF_GCV[x, y] = row['MWF_GCV']
#             # self.MWF_OR[x, y] = row['MWF_OR']
#             # self.MWF_LS[x, y] = row['MWF_LS']
#             # Assign the MWF values to the grid
#             self.MWF_DP[x, y] = row['noisy_MWF_DP']
#             self.MWF_LR[x, y] = row['noisy_MWF_LR']
#             self.MWF_LC[x, y] = row['noisy_MWF_LC']
#             self.MWF_GCV[x, y] = row['noisy_MWF_GCV']
#             # self.MWF_OR[x, y] = row['noisy_MWF_OR']
#             self.MWF_LS[x, y] = row['noisy_MWF_LS']


#     def rotate_images(self):
#         """Rotate the MWF images."""
#         self.brain_MWF_GT = self.BW @ rotate(self.MWF_GT, 275, reshape=False)
#         self.brain_MWF_LR = self.BW @ rotate(self.MWF_LR, 275, reshape=False)
#         self.brain_MWF_LC = self.BW @ rotate(self.MWF_LC, 275, reshape=False)
#         self.brain_MWF_DP = self.BW @ rotate(self.MWF_DP, 275, reshape=False)
#         self.brain_MWF_GCV = self.BW @ rotate(self.MWF_GCV, 275, reshape=False)
#         self.brain_MWF_LS = self.BW @ rotate(self.MWF_LS, 275, reshape=False)

#     def compute_norms(self):
#         """Compute the normalized differences between MWF values and ground truth (MWF_GT)."""
#         self.norm_DP_GT = np.linalg.norm(self.brain_MWF_DP - self.brain_MWF_GT) / np.linalg.norm(self.brain_MWF_GT)
#         self.norm_LR_GT = np.linalg.norm(self.brain_MWF_LR - self.brain_MWF_GT) / np.linalg.norm(self.brain_MWF_GT)
#         self.norm_LC_GT = np.linalg.norm(self.brain_MWF_LC - self.brain_MWF_GT) / np.linalg.norm(self.brain_MWF_GT)
#         self.norm_LS_GT = np.linalg.norm(self.brain_MWF_LS - self.brain_MWF_GT) / np.linalg.norm(self.brain_MWF_GT)
#         self.norm_GCV_GT = np.linalg.norm(self.brain_MWF_GCV - self.brain_MWF_GT) / np.linalg.norm(self.brain_MWF_GT)

#     def compute_sum_abs_diff(self):
#         """Compute the sum of absolute differences divided by BW."""
#         self.sum_abs_diff_DP_GT = np.sum(np.abs(self.brain_MWF_DP - self.brain_MWF_GT)) / np.sum(self.BW)
#         self.sum_abs_diff_LR_GT = np.sum(np.abs(self.brain_MWF_LR - self.brain_MWF_GT)) / np.sum(self.BW)
#         self.sum_abs_diff_LC_GT = np.sum(np.abs(self.brain_MWF_LC - self.brain_MWF_GT)) / np.sum(self.BW)
#         self.sum_abs_diff_LS_GT = np.sum(np.abs(self.brain_MWF_LS - self.brain_MWF_GT)) / np.sum(self.BW)
#         self.sum_abs_diff_GCV_GT = np.sum(np.abs(self.brain_MWF_GCV - self.brain_MWF_GT)) / np.sum(self.BW)

#     def print_comparison_results(self):
#         """Print the computed norms and absolute differences."""
#         print("Norm based comparisons:")
#         print(f"DP vs GT: {self.norm_DP_GT}")
#         print(f"LR vs GT: {self.norm_LR_GT}")
#         print(f"LC vs GT: {self.norm_LC_GT}")
#         print(f"LS vs GT: {self.norm_LS_GT}")
#         print(f"GCV vs GT: {self.norm_GCV_GT}")

#         print("\nSum of absolute differences divided by BW:")
#         print(f"DP vs GT: {self.sum_abs_diff_DP_GT}")
#         print(f"LR vs GT: {self.sum_abs_diff_LR_GT}")
#         print(f"LC vs GT: {self.sum_abs_diff_LC_GT}")
#         print(f"LS vs GT: {self.sum_abs_diff_LS_GT}")
#         print(f"GCV vs GT: {self.sum_abs_diff_GCV_GT}")

#     def plot_and_save(self, filepath):
#         """Plot and save figures for different MWF values."""
#         self._plot_save(self.MWF_DP, "DP", filepath)
#         self._plot_save(self.MWF_LR, "LocReg", filepath)
#         self._plot_save(self.MWF_LC, "L-Curve", filepath)
#         self._plot_save(self.MWF_GCV, "GCV", filepath)
#         self._plot_save(self.MWF_LS, "Non-Negative Least Squares", filepath)

#     def _plot_save(self, MWF_slice, title, filepath,xcoord=None,ycoord=None):
#         """Helper method to plot and save a figure."""
#         zoom_slice = MWF_slice[79:228, 103:215]
#         plt.figure()
#         plt.title(title)
#         # plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
#         plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
#         # plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax=0.02)
#         plt.xlabel('X Index (103 to 215)')
#         plt.ylabel('Y Index (79 to 228)')
#         plt.xticks(ticks=range(0, zoom_slice.shape[1], 25), labels=range(103, 216, 25))
#         plt.yticks(ticks=range(0, zoom_slice.shape[0], 25), labels=range(79, 229, 25))
#         plt.axis('on')
#         plt.gca().set_aspect('equal', adjustable='box')
#         plt.colorbar()
#         if xcoord is not None and ycoord is not None:
#             plt.title(title)
#             # Adjust for the zoom slice position (mapping the original coordinates to the zoomed coordinates)
#             zoom_x = xcoord - 103# Adjusting xcoord for the zoom slice range
#             zoom_y = ycoord - 79 # Adjusting ycoord for the zoom slice range
#             plt.scatter(zoom_x, zoom_y, color='red', s=10, label="Target (Red Dot)")

#         plt.savefig(f"{filepath}/{title}_{filetag}")

#     def plot_difference(self, slice1, slice2, slice1str, slice2str, filepath,xcoord=None,ycoord=None):
#         """Plot and save the difference between two slices."""
#         slice2 = slice2[79:228, 103:215]
#         slice1 = slice1[79:228, 103:215]
#         diff = slice2 - slice1
#         title = f"{slice2str} - {slice1str}"
#         savepath = f"{filepath}/{slice2str}_minus_{slice1str}_{filetag}"
#         plt.figure()
#         plt.title(title)
#         # plt.imshow(diff, cmap='viridis', vmin=0, vmax=0.2)
#         plt.imshow(diff, cmap='viridis', vmin=0, vmax=0.2)
#         # plt.imshow(diff, cmap='viridis', vmin=0, vmax=0.02)

#         plt.xlabel('X Index (103 to 215)')
#         plt.ylabel('Y Index (79 to 228)')
#         plt.xticks(ticks=range(0, slice1.shape[1], 25), labels=range(103, 216, 25))
#         plt.yticks(ticks=range(0, slice1.shape[0], 25), labels=range(79, 229, 25))
#         plt.axis('on')
#         plt.gca().set_aspect('equal', adjustable='box')
#         plt.colorbar()
#         if xcoord is not None and ycoord is not None:
#             title = f"{slice2str} - {slice1str} at Voxel X{xcoord} Y{ycoord}"
#             plt.title(title)
#             # Adjust for the zoom slice position (mapping the original coordinates to the zoomed coordinates)
#             zoom_x = xcoord  - 103 # Adjusting xcoord for the zoom slice range
#             zoom_y = ycoord  - 79 # Adjusting ycoord for the zoom slice range
#             plt.scatter(zoom_x, zoom_y, color='red', s=20, label="Target (Red Dot)")
#             savepath = f"{filepath}/{slice2str}_minus_{slice1str}_xcoord{xcoord}_ycoord{ycoord}_{filetag}"
#         plt.savefig(savepath)

#UNCOMMETN ABOVE HERE




    # def SNR_plot(self, slice, filepath, xcoord=None,ycoord=None):
    #     """Plot and save the difference between two slices."""
    #     # slice2 = slice2[79:228, 103:215]
    #     slice1 = slice1[79:228, 103:215]
    #     title = "SNR_map"
    #     savepath = f"{filepath}/{slice2str}_minus_{slice1str}_brainfig"
    #     plt.figure()
    #     plt.title(title)
    #     plt.imshow(diff, cmap='viridis', vmin=0, vmax=0.2)
    #     plt.xlabel('X Index (103 to 215)')
    #     plt.ylabel('Y Index (79 to 228)')
    #     plt.xticks(ticks=range(0, slice1.shape[1], 25), labels=range(103, 216, 25))
    #     plt.yticks(ticks=range(0, slice1.shape[0], 25), labels=range(79, 229, 25))
    #     plt.axis('on')
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     plt.colorbar()
    #     if xcoord is not None and ycoord is not None:
    #         title = f"{slice2str} - {slice1str} at Voxel X{xcoord} Y{ycoord}"
    #         plt.title(title)
    #         # Adjust for the zoom slice position (mapping the original coordinates to the zoomed coordinates)
    #         zoom_x = xcoord  - 103 # Adjusting xcoord for the zoom slice range
    #         zoom_y = ycoord  - 79 # Adjusting ycoord for the zoom slice range
    #         plt.scatter(zoom_x, zoom_y, color='red', s=20, label="Target (Red Dot)")
    #         savepath = f"{filepath}/{slice2str}_minus_{slice1str}_xcoord{xcoord}_ycoord{ycoord}_brainfig"
    #     plt.savefig(savepath)



    # def SNR_plot(self, slice, filepath, xcoord=None,ycoord=None):
    #     """Plot and save the difference between two slices."""
    #     # slice2 = slice2[79:228, 103:215]
    #     slice1 = slice1[79:228, 103:215]
    #     title = "SNR_map"
    #     savepath = f"{filepath}/{slice2str}_minus_{slice1str}_brainfig"
    #     plt.figure()
    #     plt.title(title)
    #     plt.imshow(diff, cmap='viridis', vmin=0, vmax=0.2)
    #     plt.xlabel('X Index (103 to 215)')
    #     plt.ylabel('Y Index (79 to 228)')
    #     plt.xticks(ticks=range(0, slice1.shape[1], 25), labels=range(103, 216, 25))
    #     plt.yticks(ticks=range(0, slice1.shape[0], 25), labels=range(79, 229, 25))
    #     plt.axis('on')
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     plt.colorbar()
    #     if xcoord is not None and ycoord is not None:
    #         title = f"{slice2str} - {slice1str} at Voxel X{xcoord} Y{ycoord}"
    #         plt.title(title)
    #         # Adjust for the zoom slice position (mapping the original coordinates to the zoomed coordinates)
    #         zoom_x = xcoord  - 103 # Adjusting xcoord for the zoom slice range
    #         zoom_y = ycoord  - 79 # Adjusting ycoord for the zoom slice range
    #         plt.scatter(zoom_x, zoom_y, color='red', s=20, label="Target (Red Dot)")
    #         savepath = f"{filepath}/{slice2str}_minus_{slice1str}_xcoord{xcoord}_ycoord{ycoord}_brainfig"
    #     plt.savefig(savepath)
# Example usage:

# estimates_file_path = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_13Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_18013Jan25.pkl"
# estimates_file_path = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_15Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_50015Jan25.pkl"
# data_path ="/home/kimjosy/LocReg_Regularization-1/data/Brain/results_13Jan25"
# data_path ="/home/kimjosy/LocReg_Regularization-1/data/Brain/results_13Jan252"

# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_15Jan25"



# filepath="/home/kimjosy/LocReg_Regularization-1/data/brain/testfigures2"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/comparingSNR180noise"
# filepath = data_path 

# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_15Jan25"


#uncomment and run
# estimates_file_path = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_simexp117Jan25.pkl"

# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25"
# noisy_brain_analysis = NoisyBrainAnalysis(
#     brain_data_filepath="/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat",
#     mask_filepath="/home/kimjosy/LocReg_Regularization-1/data/brain/masks/mask_2.mat",
#     # estimates_filepath="/home/kimjosy/LocReg_Regularization-1/data/brain/results/est_table_xcoordlen_313_ycoordlen_313_withLS_OR30Nov24.pkl"
#     # estimates_filepath= "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_noiseadd_recovery/est_table_xcoordlen_313_ycoordlen_313_withLS_OR10Dec24.pkl"
#     estimates_filepath= estimates_file_path
# )

# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25"

# noisy_brain_analysis.compute_norms()
# noisy_brain_analysis.compute_sum_abs_diff()
# noisy_brain_analysis.print_comparison_results()
# noisy_brain_analysis.plot_and_save(filepath)
# noisy_brain_analysis.plot_difference(noisy_brain_analysis.MWF_LR, noisy_brain_analysis.MWF_LS, "LocReg", "NNLS", filepath)
# noisy_brain_analysis.plot_difference(noisy_brain_analysis.MWF_LR, noisy_brain_analysis.MWF_LC, "LocReg", "L-Curve", filepath)
# noisy_brain_analysis.plot_difference(noisy_brain_analysis.MWF_LR, noisy_brain_analysis.MWF_GCV, "LocReg", "GCV", filepath)
# noisy_brain_analysis.plot_difference(noisy_brain_analysis.MWF_LR, noisy_brain_analysis.MWF_DP, "LocReg", "DP", filepath, xcoord=115, ycoord=124)
# noisy_brain_analysis.plot_difference(noisy_brain_analysis.MWF_LR, noisy_brain_analysis.MWF_DP, "LocReg", "GCV", filepath, xcoord=115, ycoord=124)

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
filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/filteredbrain"
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/unfilterdbrain_1_21_25"
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


class OrigBrainAnalysis:
    def __init__(self, brain_data_filepath, mask_filepath, estimates_filepath):
        # Load the brain data
        self.brain_data = scipy.io.loadmat(brain_data_filepath)["final_data_2"]
        # Load the mask
        # self.BW = scipy.io.loadmat(mask_filepath)["BW"]
        self.BW = scipy.io.loadmat(mask_filepath)["new_BW"]
        # Load the estimates dataframe
        with open(estimates_filepath, "rb") as file:
            self.df = pickle.load(file)
        
        # Initialize MWF arrays
        self.MWF_DP = np.zeros((313, 313))
        self.MWF_LC = np.zeros((313, 313))
        self.MWF_LR = np.zeros((313, 313))
        self.MWF_GCV = np.zeros((313, 313))
        self.MWF_OR = np.zeros((313, 313))
        self.MWF_LS = np.zeros((313, 313))

        self.load_MWF_values()
        # Assume MWF_GT is the MWF_GCV by default
        self.MWF_GT = self.MWF_GCV
        # Perform rotations
        self.rotate_images()

    def load_MWF_values(self):
        """Load MWF values into respective arrays."""
        for i, row in self.df.iterrows():
            x = row['X_val']  # Adjust for 0-based index
            y = row['Y_val']  # Adjust for 0-based index
            
            # # Assign the MWF values to the grid
            self.MWF_DP[x, y] = row['MWF_DP']
            self.MWF_LR[x, y] = row['MWF_LR']
            self.MWF_LC[x, y] = row['MWF_LC']
            self.MWF_GCV[x, y] = row['MWF_GCV']
            # self.MWF_OR[x, y] = row['MWF_OR']
            self.MWF_LS[x, y] = row['MWF_LS']
            # Assign the MWF values to the grid
            # self.MWF_DP[x, y] = row['noisy_MWF_DP']
            # self.MWF_LR[x, y] = row['noisy_MWF_LR']
            # self.MWF_LC[x, y] = row['noisy_MWF_LC']
            # self.MWF_GCV[x, y] = row['noisy_MWF_GCV']
            # # self.MWF_OR[x, y] = row['noisy_MWF_OR']
            # self.MWF_LS[x, y] = row['noisy_MWF_LS']


    def rotate_images(self):
        """Rotate the MWF images."""
        # self.brain_MWF_GT = self.BW @ rotate(self.MWF_GT, 275, reshape=False)
        # self.brain_MWF_LR = self.BW @ rotate(self.MWF_LR, 275, reshape=False)
        # self.brain_MWF_LC = self.BW @ rotate(self.MWF_LC, 275, reshape=False)
        # self.brain_MWF_DP = self.BW @ rotate(self.MWF_DP, 275, reshape=False)
        # self.brain_MWF_GCV = self.BW @ rotate(self.MWF_GCV, 275, reshape=False)
        # self.brain_MWF_LS = self.BW @ rotate(self.MWF_LS, 275, reshape=False)
        self.brain_MWF_GT = self.BW * self.MWF_GT
        self.brain_MWF_LR = self.BW * self.MWF_LR
        self.brain_MWF_LC = self.BW * self.MWF_LC
        self.brain_MWF_DP = self.BW * self.MWF_DP
        self.brain_MWF_GCV = self.BW * self.MWF_GCV
        self.brain_MWF_LS = self.BW * self.MWF_LS

    def compute_norms(self):
        """Compute the normalized differences between MWF values and ground truth (MWF_GT)."""
        self.norm_DP_GT = np.linalg.norm(self.brain_MWF_DP - self.brain_MWF_GT) / np.linalg.norm(self.brain_MWF_GT)
        self.norm_LR_GT = np.linalg.norm(self.brain_MWF_LR - self.brain_MWF_GT) / np.linalg.norm(self.brain_MWF_GT)
        self.norm_LC_GT = np.linalg.norm(self.brain_MWF_LC - self.brain_MWF_GT) / np.linalg.norm(self.brain_MWF_GT)
        self.norm_LS_GT = np.linalg.norm(self.brain_MWF_LS - self.brain_MWF_GT) / np.linalg.norm(self.brain_MWF_GT)
        self.norm_GCV_GT = np.linalg.norm(self.brain_MWF_GCV - self.brain_MWF_GT) / np.linalg.norm(self.brain_MWF_GT)

    def compute_sum_abs_diff(self):
        """Compute the sum of absolute differences divided by BW."""
        self.sum_abs_diff_DP_GT = np.sum(np.abs(self.brain_MWF_DP - self.brain_MWF_GT)) / np.sum(self.BW)
        self.sum_abs_diff_LR_GT = np.sum(np.abs(self.brain_MWF_LR - self.brain_MWF_GT)) / np.sum(self.BW)
        self.sum_abs_diff_LC_GT = np.sum(np.abs(self.brain_MWF_LC - self.brain_MWF_GT)) / np.sum(self.BW)
        self.sum_abs_diff_LS_GT = np.sum(np.abs(self.brain_MWF_LS - self.brain_MWF_GT)) / np.sum(self.BW)
        self.sum_abs_diff_GCV_GT = np.sum(np.abs(self.brain_MWF_GCV - self.brain_MWF_GT)) / np.sum(self.BW)

    def print_comparison_results(self):
        """Print the computed norms and absolute differences."""
        print("Norm based comparisons:")
        print(f"DP vs GT: {self.norm_DP_GT}")
        print(f"LR vs GT: {self.norm_LR_GT}")
        print(f"LC vs GT: {self.norm_LC_GT}")
        print(f"LS vs GT: {self.norm_LS_GT}")
        print(f"GCV vs GT: {self.norm_GCV_GT}")

        print("\nSum of absolute differences divided by BW:")
        print(f"DP vs GT: {self.sum_abs_diff_DP_GT}")
        print(f"LR vs GT: {self.sum_abs_diff_LR_GT}")
        print(f"LC vs GT: {self.sum_abs_diff_LC_GT}")
        print(f"LS vs GT: {self.sum_abs_diff_LS_GT}")
        print(f"GCV vs GT: {self.sum_abs_diff_GCV_GT}")

    def plot_and_save(self, filepath):
        """Plot and save figures for different MWF values."""
        self._plot_save(self.BW * self.MWF_DP, "DP", filepath)
        self._plot_save(self.BW * self.MWF_LR, "LocReg", filepath)
        self._plot_save( self.BW *self.MWF_LC, "L-Curve", filepath)
        self._plot_save(self.BW * self.MWF_GCV, "GCV", filepath)
        self._plot_save(self.BW * self.MWF_LS, "Non-Negative Least Squares", filepath)

    def _plot_save(self, MWF_slice, title, filepath,xcoord=None,ycoord=None):
        """Helper method to plot and save a figure."""
        zoom_slice = self.BW * MWF_slice
        zoom_slice = MWF_slice[79:228, 103:215]
        plt.figure()
        plt.title(title)
        # plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
        plt.imshow(zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
        plt.xlabel('X Index (103 to 215)')
        plt.ylabel('Y Index (79 to 228)')
        plt.xticks(ticks=range(0, zoom_slice.shape[1], 25), labels=range(103, 216, 25))
        plt.yticks(ticks=range(0, zoom_slice.shape[0], 25), labels=range(79, 229, 25))
        plt.axis('on')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
        if xcoord is not None and ycoord is not None:
            plt.title(title)
            # Adjust for the zoom slice position (mapping the original coordinates to the zoomed coordinates)
            zoom_x = xcoord - 103# Adjusting xcoord for the zoom slice range
            zoom_y = ycoord - 79 # Adjusting ycoord for the zoom slice range
            plt.scatter(zoom_x, zoom_y, color='red', s=10, label="Target (Red Dot)")

        plt.savefig(f"{filepath}/{title}_{filetag}")

    def plot_difference(self, slice1, slice2, slice1str, slice2str, filepath,xcoord=None,ycoord=None):
        """Plot and save the difference between two slices."""
        slice1 = self.BW * slice1
        slice2 = self.BW * slice2
        slice2 = slice2[79:228, 103:215]
        slice1 = slice1[79:228, 103:215]
        diff = slice2 - slice1
        title = f"{slice2str} - {slice1str}"
        savepath = f"{filepath}/{slice2str}_minus_{slice1str}_{filetag}"
        plt.figure()
        plt.title(title)
        plt.imshow(diff, cmap='viridis', vmin=0, vmax=0.2)
        # plt.imshow(diff, cmap='viridis', vmin=0, vmax=0.05)
        plt.xlabel('X Index (103 to 215)')
        plt.ylabel('Y Index (79 to 228)')
        plt.xticks(ticks=range(0, slice1.shape[1], 25), labels=range(103, 216, 25))
        plt.yticks(ticks=range(0, slice1.shape[0], 25), labels=range(79, 229, 25))
        plt.axis('on')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
        if xcoord is not None and ycoord is not None:
            title = f"{slice2str} - {slice1str} at Voxel X{xcoord} Y{ycoord}"
            plt.title(title)
            # Adjust for the zoom slice position (mapping the original coordinates to the zoomed coordinates)
            zoom_x = xcoord  - 103 # Adjusting xcoord for the zoom slice range
            zoom_y = ycoord  - 79 # Adjusting ycoord for the zoom slice range
            plt.scatter(zoom_x, zoom_y, color='red', s=20, label="Target (Red Dot)")
            savepath = f"{filepath}/{slice2str}_minus_{slice1str}_xcoord{xcoord}_ycoord{ycoord}_{filetag}"
        plt.savefig(savepath)

orig_brain_analysis = OrigBrainAnalysis(
    brain_data_filepath="/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat",
    # mask_filepath="/home/kimjosy/LocReg_Regularization-1/data/brain/masks/mask_2.mat",
    mask_filepath="/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat",
    # estimates_filepath="/home/kimjosy/LocReg_Regularization-1/data/brain/results/est_table_xcoordlen_313_ycoordlen_313_withLS_OR30Nov24.pkl"
    # estimates_filepath = "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_simexp117Jan25.pkl"
    # estimates_filepath= "/home/kimjosy/LocReg_Regularization-1/data/Brain/est_table_xcoordlen_313_ycoordlen_313_withLS_OR10Dec24.pkl"
    # estimates_filepath="/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_simexp117Jan25.pkl"
    # estimates_filepath= "/home/kimjosy/LocReg_Regularization-1/data/Brain/results_17Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_simexp117Jan25.pkl"
    # estimates_filepath="/home/kimjosy/LocReg_Regularization-1/data/Brain/results_21Jan25/est_table_xcoordlen_313_ycoordlen_313_SNR_300_unfiltered_noNESMA21Jan25.pkl"
    estimates_filepath="/home/kimjosy/LocReg_Regularization-1/data/Brain/results_29Jan25/filteredbrain/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps29Jan25.pkl"
    # estimates_filepath=estimates_file_path
)

orig_brain_analysis.compute_norms()
orig_brain_analysis.compute_sum_abs_diff()
orig_brain_analysis.print_comparison_results()
orig_brain_analysis.plot_and_save(filepath)

# orig_brain_analysis.plot_difference(noisy_brain_analysis.MWF_LR, orig_brain_analysis.MWF_LR, "LocRegNoisy", "LocReg_orig", filepath)
# orig_brain_analysis.plot_difference(noisy_brain_analysis.MWF_GCV, orig_brain_analysis.MWF_GCV, "GCVNoisy", "GCV_orig", filepath)
# orig_brain_analysis.plot_difference(noisy_brain_analysis.MWF_LC, orig_brain_analysis.MWF_LC, "LCNoisy", "LC_orig", filepath)
# orig_brain_analysis.plot_difference(noisy_brain_analysis.MWF_DP, orig_brain_analysis.MWF_DP, "DPNoisy", "DP_orig", filepath)
# orig_brain_analysis.plot_difference(noisy_brain_analysis.MWF_LS, orig_brain_analysis.MWF_LS, "LSNoisy", "LS_orig", filepath)


# SNR_map:

#given some diff slice; put a red dot somewhere

#1.) real brain data without tests; tune hyperparameters for LocReg find the pixels where NNLS fills in well, but Locreg doesnt...;
#best way is to See actual decay and fitted decay with diff. methods with actual noise decay to see underreg (find out what techniques that can address it) or overreg(know the techniques that can address it); pick one or two pixels. normalization? rician?
#see the fit curve and see if its above or below the actual decay; plot lambda distributions with reconstructions;
#confrim whether the bad pixels normalized data is close to 1. if normalized data max value is 0.7 its a problem...; 
# hpyeratmeter: stopping criteria; figure it out.; check if 1st step of locreg is good or not; if two step method 
#       (usually relaxed method for 1st step (higher tol)) to leave more room for error in 2nd step (same tol);


#tell to do real brain data to get a better T2 distribution before fix as gt for test1 and test2


#2.) test1: stabilization across low and high SNR levels 180 and 400; uniform scenario; sensitivity and accuracy in diff. noise levels;
#arbitratily use locreg/gcv as gt.;#do unsupervised learning/naive bayes to find the two clusters of SNR_map;

#3.) test2 for last: SNR map from real data; acrooding to each pixel and then recovery; deviation; in real scenario; across number of SNR
# arbitratily use locreg/gcv as gt.; some SNR = 0; so we skip it where we assign MWF=0; 





import scipy
SNR_map = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/SNR_map.mat")["SNR_map"]
SNR_map = SNR_map[SNR_map!= 0]
# SNR_map = SNR_map[80,160]
# print("SNR_map.shape", SNR_map.shape)
# #create mask for newSNRmap
# newSNRmap = np.zeros((313,313)) @ SNR_map
# # print("newSNRmap", newSNRmap.shape)
# newSNRmap[80,160] = SNR_map[80,160]
# SNR_map = newSNRmap
print("max SNR_map",np.max(SNR_map))
print("min SNR_map",np.min(SNR_map))
filepath= "/home/kimjosy/LocReg_Regularization-1/data/brain/figures"
SNR_map = SNR_map.flatten()
# SNR_map = SNR_map  # Python is 0-indexed, so adjust the indices
plt.figure()
plt.title("SNR map")
plt.hist(SNR_map, bins = 100)
# plt.imshow(SNR_map, cmap='viridis', vmin=0, vmax=0.2)
# plt.axis('off')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar()
plt.savefig(f"{filepath}/SNRmapbrainfig")
# OR_zoom_slice = MWF_OR[79:228, 103:215]  # Python is 0-indexed, so adjust the indices
# plt.figure()
# plt.title("Oracle")
# plt.imshow(OR_zoom_slice, cmap='viridis', vmin=0, vmax=0.2)
# plt.axis('off')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar()
# plt.savefig(f"{filepath}/ORbrainfig")