import numpy as np
import os
import pickle
import sys
# sys.path.append(r'/home/kimjosy/LocReg_Regularization-1')  # Replace this path with the actual path to the parent directory of Utilities_functions
sys.path.append(".")
from Utilities_functions.generate_gaussian_regs_L2_old import generate_gaussian_regs_L2_old
# from Simulations.heatmap_unequal_width_All import heatmap_unequal_width_All
from Simulations.gen_spanreg_heatmap_copyreference import heatmap_unequal_width_All
import pickle
import scipy.io
from tqdm import tqdm
import pandas as pd
import multiprocess as mp
from multiprocessing import Pool, freeze_support
from multiprocessing import set_start_method
from datetime import date
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
import functools

#Changing Hyperparameters
show = 1
n_sim = 1

# target_iterator = [(a) for a in n_sim]

# num_cpus_avail = np.min([len(target_iterator),60])

#### Important for Naming
# date = date.today()
# day = date.strftime('%d')
# month = date.strftime('%B')[0:3]
# year = date.strftime('%y')

# Setting up kernel matrix
# n = 150
# m = 200
n = 100
m = 150
# TE = np.linspace(0.3, 400, n) # change to 10 instead of 0.3, 400 etc.; spanreg brain (Data points TE range, T2 range); 
TE = np.linspace(10, 400, n) # change to 10 instead of 0.3, 400 etc.; spanreg brain (Data points TE range, T2 range); 
T2 = np.linspace(1, 200, m).T #
A = np.zeros((n, m))
dT = T2[1] - T2[0]

for i in range(n):
    for j in range(m):
        A[i, j] = np.exp(-TE[i] / T2[j]) * dT  # set up Kernel matrix

# offline computation
SNR = 1000

#if n_run is >= 20 its empircally stable
# n_run = 20
n_run = 20
#reg_param_lb = -8
reg_param_lb = -6
reg_param_ub = 0
#N_reg = 16
N_reg = 16

#Nc = np.array([20,80, 140])
# Nc = np.array([20, 50, 80])

#Nc is number of centers: vector of 2; 2 gaussians (centered 50 and centered at 80)
#10 guassiasn = 10 Nc values.
Nc = np.array([40, 110])

# Nc = np.array([100, 20])
nsigma = 2

#cmin and cmax 
cmin = T2[0]
cmax = T2[-1]
#original:

#left most has sigma 2, rightmost is 4; 10
# sigma_min = 2
# sigma_max = 4
sigma_min = 2
sigma_max = 6

#2 gaussian: left is borad, right is narrow; 

#3 gaussian is cube of heatmap

date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')

# sigma_min = 4
# sigma_max = 2
run = 1
if run == 1:
    # Replace the function 'generate_gaussian_regs_L2_old' with its Python implementation.
    # Gaus_info = generate_gaussian_regs_L2_old(A, T2, TE, SNR, n_run, reg_param_lb, reg_param_ub, N_reg, Nc, cmin, cmax,
    #                                          sigma_min, sigma_max)
    print("Starting computing Gaus_info")

    Gaus_info = generate_gaussian_regs_L2_old(A, T2, TE, SNR, n_run, reg_param_lb, reg_param_ub, N_reg, Nc, cmin, cmax,
                                             sigma_min, sigma_max)
    print("Finished computing Gaus_info")
    # Save file
    Ncstr = ''.join(str(elem) for elem in Nc)
    #FileName = f"lambda_{N_reg}_SNR_{SNR}_nrun_{n_run}_sigma_min_{sigma_min}_sigma_max_{sigma_max}_basis2_{Ncstr}lmbda_min{reg_param_lb}lmbda_max{reg_param_ub}"
    FileName = f"lambda_{N_reg}_SNR_{SNR}_nrun_{n_run}_sigma_min_{sigma_min}_sigma_max_{sigma_max}_basis2_{Ncstr}lmbda_min{reg_param_lb}lmbda_max{reg_param_ub}{day}{month}{year}"

    # Construct the directory path
    directory_path = os.path.join(os.getcwd(), "Simulations", "num_of_basis_functions")
    os.makedirs(directory_path, exist_ok=True)

    # Construct the full file path using os.path.join()
    file_path = os.path.join(directory_path, FileName + '.pkl')
     # Save file
    with open(file_path, 'wb') as file:
        pickle.dump(Gaus_info, file)
    print(f"File saved at: {file_path}")

# file_path ="/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_8_SNR_200_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-4lmbda_max-1.pkl"

# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_8_SNR_50_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-4lmbda_max-1.pkl"

# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_200_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max0_73124.pkl"


# #Uncomment
# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max0.pkl"
# Gaus_info = np.load(file_path, allow_pickle=True)
# print(f"File loaded from: {file_path}")

# #11/21/23 Simulation: n_sim = 20

# # Replace the function 'heatmap_unequal_width_All' with its Python implementation.
# sol_strct_uneql = heatmap_unequal_width_All(Gaus_info, show, n_sim)

# ### Looping through Iterations of the brain - applying parallel processing to improve the speed
# if __name__ == '__main__':
#     freeze_support()

#     print("Finished Assignments...")  

#     lis = []

#     with mp.Pool(processes = num_cpus_avail) as pool:

#         with tqdm(total = len(target_iterator)) as pbar:
#             for estimates_dataframe in pool.imap_unordered(heatmap_unequal_width_All, range(len(target_iterator))):
            
#                 lis.append(estimates_dataframe)

#                 pbar.update()

#         pool.close()
#         pool.join()
    

#     print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
#     df = pd.concat(lis, ignore_index= True)

#     df.to_pickle(data_folder + f'/' + data_tag +'.pkl')     

# # ############## Save General Code Code ################

# # hprParams = {
# #     "TI1g_indices": TI1g_indices,         #first iterator
# #     "TI2g_indices": TI2g_indices,         #second iterator
# #     "true_params": true_params,
# #     "TI_STANDARD": TI_STANDARD,
# #     "spacing": sp,
# #     "num_TE": ext,
# #     'multi_start': multi_starts_obj,
# #     'run_number': run_number,
# #     'norm_factor': norm_factor,
# #     'n_noise_realizations': var_reps
# # }

# # f = open(f'{data_folder}/{front_label}hprParameter_run{run_number}_{day}{month}{year}.pkl','wb')
# # pickle.dump(hprParams,f)
# # f.close()