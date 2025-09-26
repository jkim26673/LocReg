from scipy.ndimage import rotate
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
# filepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25.pkl"
# filepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/results_06Jun25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN06Jun25_processed.pkl"
filepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/results_06Jun25/combinedestimates.pkl"
# filepath =r"/Users/joshuakim/Downloads/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN06Jun25_processed_copy.pkl"
with open(filepath, 'rb') as file:
    df = pickle.load(file)

# Apply Wasserstein distance calculations
df["Wass_GCV"] = df.apply(lambda row: wasserstein_distance(row["GCV_estimate"], row["ref_estimate"]), axis=1)
df["Wass_LR"] = df.apply(lambda row: wasserstein_distance(row["LR_estimate"], row["ref_estimate"]), axis=1)
df["Wass_LR1D"] = df.apply(lambda row: wasserstein_distance(row["LR1D_estimate"], row["ref_estimate"]), axis=1)
df["Wass_LR2D"] = df.apply(lambda row: wasserstein_distance(row["LR2D_estimate"], row["ref_estimate"]), axis=1)
df["Wass_UPEN"] = df.apply(lambda row: wasserstein_distance(row["UPEN_estimate"], row["ref_estimate"]), axis=1)

# Save updated DataFrame
# df.to_pickle('/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25_wassscores.pkl')
df.to_pickle('/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/results_06Jun25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN06Jun25_processed_wassscores.pkl')