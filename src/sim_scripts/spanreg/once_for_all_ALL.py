import numpy as np
import os
import pickle
import sys
sys.path.append('.')  # Replace this path with the actual path to the parent directory of Utilities_functions
from src.regularization.reg_methods.spanreg.generate_gaussian_regs_L2_old import generate_gaussian_regs_L2_old
from src.sim_scripts.spanreg.heatmap_unequal_width_All import heatmap_unequal_width_All
import pickle
import scipy.io
# Setting up kernel matrix
n = 150
m = 200
TE = np.linspace(0.3, 400, n)
T2 = np.linspace(1, 200, m).T
A = np.zeros((n, m))
dT = T2[1] - T2[0]

for i in range(n):
    for j in range(m):
        A[i, j] = np.exp(-TE[i] / T2[j]) * dT  # set up Kernel matrix


# T2: 10 to 200 ms for 150
# TE: 10 to 400 ms for 100
# offline computation
SNR = 500
n_run = 20
#reg_param_lb = -8
reg_param_lb = -4
reg_param_ub = -1
#N_reg = 16
N_reg = 8

#Nc = np.array([20,80, 140])
Nc = np.array([20, 50, 80])
#Nc = np.array([50, 80])
nsigma = len(Nc)
cmin = T2[0]
cmax = T2[-1]
#original:
sigma_min = 2
sigma_max = 4
# sigma_min = 4
# sigma_max = 2
run = 1
if run == 1:
    # Replace the function 'generate_gaussian_regs_L2_old' with its Python implementation.
    # Gaus_info = generate_gaussian_regs_L2_old(A, T2, TE, SNR, n_run, reg_param_lb, reg_param_ub, N_reg, Nc, cmin, cmax,
    #                                          sigma_min, sigma_max)
    Gaus_info = generate_gaussian_regs_L2_old(A, T2, TE, SNR, n_run, reg_param_lb, reg_param_ub, N_reg, Nc, cmin, cmax,
                                             sigma_min, sigma_max)
    print("Finished computing Gaus_info")
    # Save file
    Ncstr = ''.join(str(elem) for elem in Nc)
    #FileName = f"lambda_{N_reg}_SNR_{SNR}_nrun_{n_run}_sigma_min_{sigma_min}_sigma_max_{sigma_max}_basis2_{Ncstr}lmbda_min{reg_param_lb}lmbda_max{reg_param_ub}"
    FileName = f"lambda_{N_reg}_SNR_{SNR}_nrun_{n_run}_sigma_min_{sigma_min}_sigma_max_{sigma_max}_basis2_{Ncstr}lmbda_min{reg_param_lb}lmbda_max{reg_param_ub}"

    # Construct the directory path
    directory_path = os.path.join(os.getcwd(), "Simulations", "num_of_basis_functions")
    os.makedirs(directory_path, exist_ok=True)

    # Construct the full file path using os.path.join()
    file_path = os.path.join(directory_path, FileName + '.pkl')
     # Save file
    with open(file_path, 'wb') as file:
        pickle.dump(Gaus_info, file)
    print(f"File saved at: {file_path}")

# Load previously saved file
file_path = "lambda_16_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_1205030lmbda_min-8lmbda_max-1.npy"

#file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_16_SNR_500_nrun_1_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-8lmbda_max-1.pkl"
#file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_16_SNR_500_nrun_1_sigma_min_4_sigma_max_2_basis2_5080lmbda_min-8lmbda_max-1modifiedGaussian.pkl"
#file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_16_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-8lmbda_max-1samenoisereal.pkl"
file_path = '/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_16_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-8lmbda_max-18_30_23.pkl'

#1023
file_path = '/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_16_SNR_500_nrun_1_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-8lmbda_max-1.pkl'


file_path = '/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_8_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_5020120lmbda_min-4lmbda_max-1.pkl'

#2peak:
file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_8_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-4lmbda_max-1.pkl"

#3 peak:
file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_8_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_205080lmbda_min-4lmbda_max-1.pkl"


#2 peak on 10_23_23
file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_16_SNR_500_nrun_1_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-8lmbda_max-1.pkl"

#2peak on 11/21/23:
file_path ="/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_8_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-4lmbda_max-1.pkl"

file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_200_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max0_73124.pkl"
import pickle
import scipy.io
# mat = scipy.io.loadmat('file.mat')
Gaus_info = np.load(file_path, allow_pickle=True)
print(f"File loaded from: {file_path}")

# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# scipy.io.savemat('Gaus_info_python.mat', {'data': data})


# import scipy.io
# Gaus_info = scipy.io.loadmat('/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_16_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-8lmbda_max-1.mat')
# Gaus_info = scipy.io.loadmat('/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/num_of_basis_functions/lambda_8_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_5020lmbda_min-4lmbda_max-109_04_23.mat')
# Gaus_info = Gaus_info['Gaus_info'][0][0]
# print(f"File loaded from: {Gaus_info}")

show = 1
n_sim = 1

#11/21/23 Simulation: n_sim = 20

# Replace the function 'heatmap_unequal_width_All' with its Python implementation.
sol_strct_uneql = heatmap_unequal_width_All(Gaus_info, show, n_sim)

# Additional calculations, uncomment if needed
# difference = avg_MDL_err - avg_MDL_err_GCV
# MultiReg_MDL = np.mean(np.mean(avg_MDL_err))
# DP_MDL = np.mean(np.mean(avg_MDL_err_GCV))
# np.mean(np.mean(difference))
