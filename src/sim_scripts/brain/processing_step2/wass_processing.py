from scipy.ndimage import rotate
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
# filepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25.pkl"
# filepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/results_06Jun25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN06Jun25_processed.pkl"
# filepath = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/results_06Jun25/combinedestimates.pkl"
# filepath =r"/Users/joshuakim/Downloads/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN06Jun25_processed_copy.pkl"
# filepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25.pkl"
# filepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/results_06Jun25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN06Jun25_processed_wassscores.pkl"
filepath = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep2625/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN26Sep25.pkl"
with open(filepath, 'rb') as file:
    df = pickle.load(file)

# Apply Wasserstein distance calculations
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
import pickle
import numpy as np
import pandas as pd

#pixel800; 2993 (artefact beginning); 869 upped T2 lower bound from 15 to 20; change the round of lr_MWF 2.8247151662451206e-05 to 3rd decimal point...for all MWF values...
#June 6: voxel 8929, 800, 4346 gcv wass score is much smaller than locreg
#June 6: voxel 2163 smallest locreg wass score
#june 6;  highest wass score LR: 656,1259,7038
#june 6: greatest diff in wass scoer LR and gcv: 5108
# 1)round MWF to 2-3 decimal points
# 2)sept19;869,934 voxel is interesting...reconstruction of locreg and ref is similar...gcv is flatine...but mwf of gcv and ref is closer.; ask chuan if we can just get rid of peaks
# 3)sept19; 5012; just poor reconstrcutions...locreg is flipped version of ref.

#sept19;869 voxel is interesting...reconstruction of locreg and ref is similar...gcv is flatine...but mwf of gcv and ref is closer.
ref_est = df.iloc[5108]["ref_estimate"]
gcv_est = df.iloc[5108]["GCV_estimate"]
lr_est = df.iloc[5108]["LR_estimate"]

dTE = 11.3
n = 32
TE = dTE * np.linspace(1, n, n)
m = 150
T2 = np.linspace(10, 200, m)
A = np.zeros((n, m))
dT = T2[1] - T2[0]
dT = 1.275167785234899
gcv_norm = gcv_est/(np.sum(gcv_est)* dT)
ref_norm = ref_est/(np.sum(ref_est) * dT)
lr_norm = lr_est/(np.sum(lr_est) * dT)
# row = df.loc[df["Wass_GCV"] < df["Wass_LR"]]
# row.loc[(row["Wass_LR"] - row["Wass_GCV"]).idxmax()]
# smooth_dist = gaussian_filter1d(dist, sigma=1)
# mwf = np.trapz(smooth_dist[t2_values <= 40], t2_values[t2_values <= 40])
Myelin_idx = np.where((T2 >= 20) & (T2 <= 40))
mask_wass = np.where((T2 < 20) | (T2 > 165))
gcv_norm[mask_wass] = 0
ref_norm[mask_wass] = 0
lr_norm[mask_wass] = 0
# print(wasserstein_distance(gcv_norm, ref_norm))
# print(wasserstein_distance(lr_norm, ref_norm))

# plt.plot(T2,gcv_norm,label = "GCV");plt.plot(T2,lr_norm,label='LR');plt.plot(T2,ref_norm,label="ref");plt.legend();plt.show()

#put artefact blockers... T2>20, T2<185; when calculating wass score
total_MWF = np.cumsum(gcv_norm)
gcv_MWF = total_MWF[Myelin_idx[-1][-1]]
total_MWF = np.cumsum(ref_norm)
ref_MWF = total_MWF[Myelin_idx[-1][-1]]
total_MWF = np.cumsum(lr_norm)
lr_MWF = total_MWF[Myelin_idx[-1][-1]]

def calc_new_MWF(f_rec):
    Myelin_idx = np.where((T2 >= 15) & (T2 <= 40))
    total_MWF = np.cumsum(f_rec)
    MWF = total_MWF[Myelin_idx[-1][-1]]
    return MWF

# wasserstein_distance(gcv_norm,ref_norm)
# wasserstein_distance(lr_norm,ref_norm)

df["Wass_GCV"] = df.apply(lambda row: wasserstein_distance(row["GCV_estimate"], row["ref_estimate"]), axis=1)
df["Wass_LR"] = df.apply(lambda row: wasserstein_distance(row["LR_estimate"], row["ref_estimate"]), axis=1)
df["Wass_LR1D"] = df.apply(lambda row: wasserstein_distance(row["LR1D_estimate"], row["ref_estimate"]), axis=1)
df["Wass_LR2D"] = df.apply(lambda row: wasserstein_distance(row["LR2D_estimate"], row["ref_estimate"]), axis=1)
df["Wass_UPEN"] = df.apply(lambda row: wasserstein_distance(row["UPEN_estimate"], row["ref_estimate"]), axis=1)

# df["sm_MWF_GCV"] = df.apply(lambda row: calc_new_MWF(row["GCV_estimate"]), axis=1)
# df["sm_MWF_LR"] = df.apply(lambda row: calc_new_MWF(row["LR_estimate"]), axis=1)
# df["sm_MWF_LR1D"] = df.apply(lambda row: calc_new_MWF(row["LR1D_estimate"]), axis=1)
# df["sm_MWF_LR2D"] = df.apply(lambda row: calc_new_MWF(row["LR2D_estimate"]), axis=1)
# df["sm_MWF_UPEN"] = df.apply(lambda row: calc_new_MWF(row["UPEN_estimate"]), axis=1)
# df["sm_MWF_ref"] = df.apply(lambda row: calc_new_MWF(row["ref_estimate"]), axis=1)

# Save updated DataFrame
# df.to_pickle('/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25_wassscores.pkl')
# df.to_pickle('/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp/results_06Jun25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN06Jun25_processed_wassscores.pkl')
# df.to_pickle('/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN19Sep25_wassscore_newMWF.pkl')
# df.to_pickle('/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/results_06Jun25/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN06Jun25_processed_wassscores_newMWF.pkl')
df.to_pickle("/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep2625/est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_NA_GCV_LR012_UPEN26Sep25_processed_wassscores.pkl")