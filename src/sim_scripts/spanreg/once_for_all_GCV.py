from src.utils.load_imports.loading import *
from src.regularization.reg_methods.spanreg.generate_gaussian_regs_L2_old import generate_gaussian_regs_L2_old
from src.sim_scripts.spanreg.heatmap_unequal_width_All import heatmap_unequal_width_All

# Setting up kernel matrix
n = 150
m = 200
TE = np.linspace(0.3, 400, n)
T2 = np.linspace(1, 200, m)
A = np.zeros((n, m))
dT = T2[1] - T2[0]

for i in range(n):
    for j in range(m):
        A[i, j] = np.exp(-TE[i] / T2[j]) * dT  # set up Kernel matrix

# offline computation
SNR = 500
n_run = 16
reg_param_lb = -8
reg_param_ub = -1
N_reg = 16
ngen = 0

Nc = np.array([120, 50, 30])
nsigma = len(Nc)
cmin = T2[0]
cmax = T2[-1]
sigma_min = 2
sigma_max = 4
run = 0
if run == 1:
    # Replace the function 'generate_gaussian_regs_L2_old' with its Python implementation.
    Gaus_info = generate_gaussian_regs_L2_old(A, T2, TE, SNR, n_run, reg_param_lb, reg_param_ub, N_reg, Nc, cmin, cmax,
                                              sigma_min, sigma_max)

    # Save file
    Ncstr = ''.join(str(Nc).split())
    FileName = f"lambda_{N_reg}_SNR_{SNR}_nrun_{n_run}_sigma_min_{sigma_min}_sigma_max_{sigma_max}_basis2_{Ncstr}lmbda_min{reg_param_lb}lmbda_max{reg_param_ub}"
    matfile = os.path.join('num_of_basis_functions', FileName + '.npy')
    np.save(matfile, Gaus_info)

# Load previously saved file
file_path = "lambda_16_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_1205030lmbda_min-8lmbda_max-1.npy"
Gaus_info = np.load(file_path)

show = 1
n_sim = 1

# Replace the function 'heatmap_unequal_width_GCV' with its Python implementation.
avg_MDL_err, avg_MDL_err_GCV = heatmap_unequal_width_GCV(Gaus_info, show, n_sim)

difference = avg_MDL_err - avg_MDL_err_GCV
MultiReg_MDL = np.mean(np.mean(avg_MDL_err))
DP_MDL = np.mean(np.mean(avg_MDL_err_GCV))
mean_diff = np.mean(np.mean(difference))

print("Mean Difference:", mean_diff)
print("MultiReg MDL:", MultiReg_MDL)
print("DP MDL:", DP_MDL)
