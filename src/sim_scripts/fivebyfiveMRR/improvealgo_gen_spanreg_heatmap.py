from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

# os.environ["MSK_IPAR_OPTIMIZER"] = 'MSK_OPTIMIZER_INTPNT'
# os.environ["MSK_IPAR_BI_MAX_ITERATIONS "] = "1000"

parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
cwd = os.getcwd()

cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

pat_tag = "MRR"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
series_tag = "SpanRegFig"
simulation_save_folder = f"SimulationSets/{pat_tag}/{series_tag}"
# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + simulation_save_folder 

#Hyperparameters and Global Parameters
preset_noise = True
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_15Aug24noise_arr_modifiedalgo.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_test_21Aug24noise_arr.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_13Aug24noise_arr.txt.npy"
noise_file_path="/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_improve_algo_SNR300_iter1_lamini_gcv_dist_narrowL_broadR_improvealgo_27Aug24noise_arr.npy"
testing = False
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_15Aug24noise_arr_modifiedalgo.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_13Aug24noise_arr.txt.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_06Aug24noise_arr.txt.npy"

#Number of simulations:
n_sim = 1
SNR_value = 50

###LocReg hyperparameters
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
exp = 0.5
feedback = True
lam_ini_val = "gcv"
dist_type = "semicircle"
# gamma_init = 5
#gamma_init = 0.5 is best
gamma_init = 0.5

###Plotting hyperparameters
npeaks = 1
# npeaks = 3
nsigma = 5
f_coef = np.ones(npeaks)
rps = np.linspace(1, 4, 5).T

#orig/2bump:
#npeak = 2
# rps = np.linspace(1, 4, 5).T
# nsigma = 5
#single/hat:
#npeak = 1
# rps = np.linspace(1, 1, 5).T
# nsigma = 5

#threebump:
# npeaks = 3
# rps = np.linspace(4, 4, 1).T



# Lambda = Gaus_info['Lambda'].reshape(-1,1)
# Alpha_vec = np.logspace(-6, 0,16)
# Alpha_vec = Gaus_info['Lambda'].reshape(-1,1)
nrps = len(rps)

###SNR Values to Evaluate
# SNR_value = 50
# SNR_value = 200
# SNR_value = 500
# SNR_value = 1000

#Load Data File
# #narrow left, broad right, SNR 1000
# file_path ="/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max0.pkl"

# #narrow left, broad right, SNR 50
# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_50_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max015Aug24.pkl"

#broad left, narrow right, SNR 1000
# file_path ="/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_1000_nrun_20_sigma_min_4_sigma_max_2_basis2_5080lmbda_min-6lmbda_max015Aug24.pkl"
file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_1000_nrun_1_sigma_min_2_sigma_max_6_basis2_10040lmbda_min-6lmbda_max016Aug24.pkl"

# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_1000_nrun_10_sigma_min_2_sigma_max_6_basis2_5020lmbda_min-6lmbda_max019Aug24.pkl"


# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_50_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max031Jul24.pkl"
# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_200_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max0_73124.pkl"
# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max1_073124.pkl"

Gaus_info = np.load(file_path, allow_pickle=True)
print(f"File loaded from: {file_path}")
A = Gaus_info["A"]
n, m = Gaus_info['A'].shape

# print("Gaus_info['A'].shape", Gaus_info['A'].shape)
###Number of noisy realizations; 20 NR is enough to until they ask for more noise realizations

#Naming for Data Folder
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')
data_path = "SimulationsSets/MRR/SpanRegFig"
add_tag = ""
data_head = "est_table"
data_tag = (f"{data_head}_SNR{SNR_value}_iter{n_sim}_lamini_{lam_ini_val}_dist_{dist_type}_{add_tag}{day}{month}{year}")
data_folder = (os.getcwd() + f'/{data_path}')
os.makedirs(data_folder, exist_ok = True)

#Number of tasks to execute
target_iterator = [(a,b,c) for a in range(n_sim) for b in range(nsigma) for c in range(nrps)]
num_cpus_avail = np.min([len(target_iterator),100])

noisy_data = np.zeros((n_sim, nsigma, nrps, n))
noiseless_data = np.zeros((n_sim, nsigma, nrps, n))

if preset_noise == True:
    noise_arr = np.load(noise_file_path, allow_pickle=True)
    noise_list = []
else:
    noise_arr = np.zeros((n_sim, nsigma, nrps, A.shape[0]))
    noise_list = []

#Functions
def create_result_folder(string, SNR, lam_ini_val, dist_type):
    folder_name = f"{cwd_full}/{string}_{date}_SNR_{SNR}_lamini_{lam_ini_val}_dist_{dist_type}_modularized"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# def minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g):
#     OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
#     OP_rhos = np.zeros((len(Alpha_vec)))
#     for j in (range(len(Alpha_vec))):
#         try:
#             # Fallback to nonnegtik_hnorm
#             sol, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j], '0', nargin=4)
#         except Exception as e:
#             print(f"Error in nonnegtik_hnorm: {e}")
#             # If both methods fail, solve using cvxpy
#             lam_vec = Alpha_vec[j] * np.ones(G.shape[1])
#             A = (G.T @ G + np.diag(lam_vec))
#             eps = 1e-2
#             ep4 = np.ones(A.shape[1]) * eps
#             b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vec
#             y = cp.Variable(G.shape[1])
#             cost = cp.norm(A @ y - b, 2)**2
#             constraints = [y >= 0]
#             problem = cp.Problem(cp.Minimize(cost), constraints)
#             problem.solve(solver=cp.MOSEK, verbose = False)
#             sol = y.value
#             sol = sol - eps
#             sol = np.maximum(sol, 0)
#         OP_x_lc_vec[:, j] = sol
#         OP_rhos[j] = np.linalg.norm(OP_x_lc_vec[:,j] - g, 2)**2

#     OP_log_err_norm = np.log10(OP_rhos)
#     min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)
#     min_x = Alpha_vec[min_index[0]]
#     min_z = np.min(OP_log_err_norm)
#     OP_min_alpha1 = min_x
#     OP_min_alpha1_ind = min_index[0]
#     f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
#     return f_rec_OP_grid, OP_min_alpha1

def minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g):
    OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
    OP_rhos = np.zeros((len(Alpha_vec)))
    for j in (range(len(Alpha_vec))):
        try:
            # Fallback to nonnegtik_hnorm
            sol, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j], '0', nargin=4)
        except Exception as e:
            print(f"Error in nonnegtik_hnorm: {e}")
            # If both methods fail, solve using cvxpy
            lam_vec = Alpha_vec[j] * np.ones(G.shape[1])
            A = (G.T @ G + np.diag(lam_vec))
            eps = 1e-2
            ep4 = np.ones(A.shape[1]) * eps
            b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vec
            
            y = cp.Variable(G.shape[1])
            cost = cp.norm(A @ y - b, 2)**2
            constraints = [y >= 0]
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.MOSEK, verbose=False)
            
            sol = y.value
            sol = sol - eps
            sol = np.maximum(sol, 0)

        OP_x_lc_vec[:, j] = sol
        OP_rhos[j] = np.linalg.norm(OP_x_lc_vec[:,j] - g, 2)**2

    OP_log_err_norm = np.log10(OP_rhos)
    min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)
    min_x = Alpha_vec[min_index[0]]
    min_z = np.min(OP_log_err_norm)
    OP_min_alpha1 = min_x
    OP_min_alpha1_ind = min_index[0]
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    # OP_min_alpha1 = np.sqrt(OP_min_alpha1)
    return f_rec_OP_grid, OP_min_alpha1

def calc_T2mu(rps):
    #30 change 30 to 40
    mps = rps / 2
    #singlepeak:
    #mps = rps / 1
    #T2_left = 100 * np.ones(nrps)
    nrps = len(rps)
    T2_left = 100 * np.ones(nrps)
    T2_mid = T2_left * mps
    T2_right = T2_left * rps

    #twopeak
    T2mu = np.column_stack((T2_right))
    #threepeak
    # T2mu = np.column_stack((T2_left, T2_mid,  T2_right))
    # T2mu = T2_left

    return T2mu

def calc_sigma_i(iter_i, diff_sigma):
    sigma_i = diff_sigma[iter_i, :]
    return sigma_i

def calc_rps_val(iter_j, rps):
    rps_val = rps[iter_j]
    return rps_val

def calc_diff_sigma(nsigma):
    # unif_sigma = np.linspace(2, 5, nsigma).T
    unif_sigma = np.linspace(2, 4, nsigma).T
    # diff_sigma = np.column_stack((unif_sigma, 3 * unif_sigma))
    # diff_sigma = np.column_stack((3 * unif_sigma, unif_sigma, unif_sigma))
    # diff_sigma = np.column_stack((3 * unif_sigma, unif_sigma))

    #hat/single peak dist
    # unif_sigma = np.linspace(2, 4, nsigma).T
    diff_sigma = np.column_stack((unif_sigma, unif_sigma))
    return unif_sigma, diff_sigma

def load_Gaus(Gaus_info):
    T2 = Gaus_info['T2'].flatten()
    TE = Gaus_info['TE'].flatten()
    A = Gaus_info['A']
    Lambda = Gaus_info['Lambda'].reshape(-1,1)
    n, m = Gaus_info['A'].shape
    # SNR = Gaus_info['SNR']
    SNR = SNR_value
    return T2, TE, Lambda, A, m,  SNR

def calc_dat_noisy(A, TE, IdealModel_weighted, SNR):
    dat_noiseless = A @ IdealModel_weighted
    noise = np.column_stack([np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE), 1)]) 
    noise  = np.ravel(noise)
    dat_noisy = dat_noiseless + np.ravel(noise)
    return dat_noisy, noise

def semicircle_distribution(T2):
    # Calculate the center and radius of the semicircle
    start = T2[0]
    end = T2[-1]
    center = (start + end) / 2
    radius = min(center - start, end - center)

    # Define the semicircle distribution function
    def f(t):
        if center - radius <= t <= center + radius:
            return np.sqrt(radius**2 - (t - center)**2)
        else:
            return 0

    # Calculate the raw distribution values
    raw_distribution = np.array([f(t) for t in T2])

    # Find the maximum value in the raw distribution
    max_value = np.max(raw_distribution)

    # Normalize the distribution to have a maximum value of 0.5
    normalized_distribution = raw_distribution / max_value * 0.01

    return normalized_distribution

def get_IdealModel_weighted(iter_j, m, npeaks, T2, T2mu, sigma_i):
    p = np.zeros((npeaks, m))
    T2mu_sim = T2mu[iter_j, :]
    p = np.array([normsci.pdf(T2, mu, sigma) for mu, sigma in zip(T2mu_sim, sigma_i)])
    IdealModel_weighted = p.T @ f_coef / npeaks
    return IdealModel_weighted

def error(IdealModel,reconstr):
    true_norm = linalg_norm(IdealModel)
    err = linalg_norm(IdealModel - reconstr) / true_norm
    return err

# def wasserstein
# def 

# def profile_function(func, *args, **kwargs):
#     profiler = cProfile.Profile()
#     profiler.enable()
#     result = func(*args, **kwargs)
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
#     stats.print_stats(10)
#     return result

def generate_estimates(i_param_combo):
    def plot(iter_sim, iter_sigma, iter_rps):
        plt.figure(figsize=(12.06, 4.2))
        # Plotting the first subplot
        plt.subplot(1, 3, 1) 
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} (Error: {round(err_LR,3)})')
        # plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle (Error: {round(err_oracle,3)})')
        # plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {round(err_DP,3)})')
        plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {round(err_GCV,3)})')
        # plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Error: {round(err_LC,3)})')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
        plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
        ymax = np.max(IdealModel_weighted) * 1.15
        plt.ylim(0, ymax)

        # Plotting the second subplot
        plt.subplot(1, 3, 2)
        plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label= f'LocReg {lam_ini_val}')
        # plt.plot(TE, A @ f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle')
        # plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
        plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
        # plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('TE', fontsize=20, fontweight='bold')
        plt.ylabel('Intensity', fontsize=20, fontweight='bold')
        
        plt.subplot(1, 3, 3)
        # plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
        plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
        # plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
        plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')
        # plt.semilogy(T2, lambda_oracle * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')

        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2', fontsize=20, fontweight='bold')
        plt.ylabel('Lambda', fontsize=20, fontweight='bold')

        plt.tight_layout()
        string = "MRR_1D_LocReg_Comparison"
        file_path = create_result_folder(string, SNR, lam_ini_val, dist_type)
        plt.savefig(os.path.join(file_path, f"Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}.png"))
        plt.close() 

    iter_sim, iter_sigma, iter_rps = target_iterator[i_param_combo]
    L = np.eye(A.shape[1])
    sigma_i = diff_sigma[iter_sigma, :]
    rps_val = calc_rps_val(iter_rps, rps)

    # Profile the function calls individually
    # IdealModel_weighted = get_IdealModel_weighted(iter_rps, m, npeaks, T2, T2mu, sigma_i)
    IdealModel_weighted = semicircle_distribution(T2)
    if preset_noise == False:
        dat_noisy,noise = calc_dat_noisy( A, TE, IdealModel_weighted, SNR)
    else:
        dat_noiseless = A @ IdealModel_weighted
        # print("noise_arr.shape", noise_arr.shape)
        # print("iter_sim", iter_sim)
        # print("iter_sigma", iter_sigma)
        # print("iter_rps", iter_rps)

        # noise_arr = noise_arr.flatten()
        # noise_arr = noise_arr.reshape(z,y)
        # print("dat_noiseless.shape", dat_noiseless.shape)
        # print("noise_iter shape",noise_arr[])
        noise = noise_arr[iter_sim, iter_sigma, iter_rps,:]
        # print("noise", noise.shape)
        # print("noise.shape", noise.shape)
        dat_noisy = dat_noiseless + np.ravel(noise)
        # print("dat_noiseless.shape", dat_noiseless.shape)

        noisy_data[iter_sim, iter_sigma, iter_rps,:] = dat_noisy
        noiseless_data[iter_sim, iter_sigma, iter_rps,:] = dat_noiseless

        # noise = noise_arr.tolist()
        # noise_arr = np.zeros((n_sim, nsigma, nrps))
    # f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)
    # f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
    f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
    f_rec_GCV = f_rec_GCV[:, 0]
    lambda_GCV = np.squeeze(lambda_GCV)

    # if lam_ini_val == "LCurve" or lam_ini_val == "L-Curve":
    #     LRIto_ini_lam = lambda_LC
    # elif lam_ini_val == "GCV" or lam_ini_val == "gcv":
    #     LRIto_ini_lam = lambda_GCV
    # elif lam_ini_val == "DP" or lam_ini_val == "dp":
    #     LRIto_ini_lam = lambda_DP

    if lam_ini_val == "GCV" or lam_ini_val == "gcv":
        LRIto_ini_lam = lambda_GCV

    # LRIto_ini_lam = lambda_LC
    # maxiter = 75
    maxiter = 200
    # maxiter = 600
    # f_rec_LocReg_LC, lambda_locreg_LC = profile_function(LocReg_Ito_mod, dat_noisy, A, lambda_LC, gamma_init, maxiter)
    # f_rec_oracle, lambda_oracle = profile_function(minimize_OP, Alpha_vec, L, dat_noisy, A, len(T2), IdealModel_weighted)
    f_rec_LocReg_LC, lambda_locreg_LC, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(dat_noisy, A, LRIto_ini_lam, gamma_init, maxiter)
  
    
    if testing == True and iter_sigma == 0 and iter_rps == 0:
        meanfrec1 = np.mean(test_frec1)
        meanlam1 = np.mean(test_lam1)
        plt.figure(figsize=(10, 5))  # Create a new figure
        
        # First subplot
        plt.subplot(1, 2, 1) 
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, test_frec1, linestyle=':', linewidth=3, color='red', label=f'frec1 gamma init {gamma_init:.2f}')
        plt.legend(fontsize=10, loc='best')
        
        # Second subplot
        plt.subplot(1, 2, 2)  # Changed to 2 for the second subplot
        plt.semilogy(T2, test_lam1 * np.ones(len(T2)), linewidth=3, color='green', label=f'test_lam1 with meanlam1 {meanlam1:.2f}')
        plt.legend(fontsize=10, loc='best')
        
        # Print information
        print(f"{numiterate} iters for Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}")
        
        # Show plots
        plt.savefig(f'plot_output_gamma_init{gamma_init}.png')
    else:
        pass

    # f_rec_oracle, lambda_oracle = minimize_OP(Lambda, L, dat_noisy, A, len(T2), IdealModel_weighted)

    #normalization
    sum_x = np.sum(f_rec_LocReg_LC)
    f_rec_LocReg_LC = f_rec_LocReg_LC / sum_x
    # sum_oracle = np.sum(f_rec_oracle)
    # f_rec_oracle = f_rec_oracle / sum_oracle
    sum_GCV = np.sum(f_rec_GCV)
    f_rec_GCV = f_rec_GCV / sum_GCV
    # sum_LC = np.sum(f_rec_LC)
    # f_rec_LC = f_rec_LC / sum_LC
    # sum_DP = np.sum(f_rec_DP)
    # f_rec_DP = f_rec_DP / sum_DP

    # Flatten results
    f_rec_GCV = f_rec_GCV.flatten()
    # f_rec_DP = f_rec_DP.flatten()
    # f_rec_LC = f_rec_LC.flatten()
    f_rec_LocReg_LC = f_rec_LocReg_LC.flatten()
    # f_rec_oracle = f_rec_oracle.flatten()'

    # Calculate errors
    # err_LC = error(IdealModel_weighted, f_rec_LC)
    # err_DP = error(IdealModel_weighted, f_rec_DP)
    err_GCV = error(IdealModel_weighted, f_rec_GCV)
    # err_oracle = error( IdealModel_weighted, f_rec_oracle)
    err_LR = error( IdealModel_weighted, f_rec_LocReg_LC)

    # if iter_rps == 0 and iter_sigma == 0:
    #     true_norm = linalg_norm(IdealModel_weighted)
    #     print("true_norm", linalg_norm(IdealModel_weighted))
    #     print("locreg error",err_LR)
    #     print("gcv error", err_GCV)
    #     print("oracle error", err_oracle)

    # Plot a random simulation
    # random_sim = random.randint(0, n_sim)
    if iter_sim == 0:
        plot(iter_sim, iter_sigma, iter_rps)
        print(f"Finished Plots for iteration {iter_sim} sigma {iter_sigma} rps {iter_rps}")
    else:
        pass

    # Create DataFrame
    # sol_strct_uneql = {}
    # sol_strct_uneql['noiseless_data'] = noiseless_data
    # sol_strct_uneql['noisy_data'] = noisy_data  

    feature_df = pd.DataFrame(columns=["NR", 'Sigma', 'RPS_val', "err_LR", "err_GCV"])
    feature_df["NR"] = [iter_sim]
    feature_df["Sigma"] = [sigma_i]
    feature_df["RPS_val"] = [rps_val]
    # feature_df["err_DP"] = [err_DP]
    # feature_df["err_LC"] = [err_LC]
    feature_df["err_LR"] = [err_LR]
    feature_df["err_GCV"] = [err_GCV]
    # feature_df["err_oracle"] = [err_oracle]

    # noise_df = pd.DataFrame(columns= ["Noise"])
    return feature_df, noise
# def generate_estimates(i_param_combo):
#     def plot(iter_sim, iter_sigma,iter_rps):
#         #IdealModel_weighted, f_rec_oracle, f_rec_LocReg_LC, f_rec_DP,f_rec_GCV, f_rec_LC, oracleerr, locregerr, dperr, gcverr, lcerr, lambda_oracle,lambda_locreg_LC, lambda_DP, lambda_GCV, lambda_LC
#         plt.figure(figsize=(12.06, 4.2))
#         # Plotting the first subplot
#         plt.subplot(1, 3, 1) 
#         plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
#         plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg LCurve (Error: {round(err_LR,3)})')
#         plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle (Error: {round(err_oracle,3)})')
#         plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {round(err_DP,3)})')
#         plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {round(err_GCV,3)})')
#         plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Error: {round(err_LC,3)})')
#         plt.legend(fontsize=10, loc='best')
#         plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
#         plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
#         ymax = np.max(IdealModel_weighted) * 1.15
#         plt.ylim(0, ymax)

#         # Plotting the second subplot
#         plt.subplot(1, 3, 2)
#         plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
#         plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label='LocReg LCurve')
#         plt.plot(TE, A @ f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle')
#         plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
#         plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
#         plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
#         plt.legend(fontsize=10, loc='best')
#         plt.xlabel('TE', fontsize=20, fontweight='bold')
#         plt.ylabel('Intensity', fontsize=20, fontweight='bold')
        
#         plt.subplot(1, 3, 3)
#         plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
#         plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
#         plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
#         plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label='LocReg LCurve')
#         plt.semilogy(T2, lambda_oracle * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')
        
#         plt.legend(fontsize=10, loc='best')
#         plt.xlabel('T2', fontsize=20, fontweight='bold')
#         plt.ylabel('Lambda', fontsize=20, fontweight='bold')

#         plt.tight_layout()
#         string = "MRR_1D_LocReg_Comparison"
#         file_path = create_result_folder(string, SNR_value, lam_ini_val)
#         plt.savefig(os.path.join(file_path, f"simulation_{iter_sim}_sigmaval{iter_sigma}_rpsval{iter_rps}.png"))
#         print(f"Saved Comparison Plot for Sigmaval{iter_sigma}_rpsval{iter_rps}")
#         plt.close()  
#     # target_iterator = [(a,b,c,d) for a in range(n_sim) for b in range(nsigma) for c in range(nrps) for d in range(m)]
#     iter_sim, iter_sigma, iter_rps = target_iterator[i_param_combo]
#     L = np.eye(A.shape[1])
#     #Get the  sigma_i values
#     sigma_i = diff_sigma[iter_sigma,:]
#     #Get the possible rps valute
#     rps_val = calc_rps_val(iter_rps, rps)
#     ###Get the reconstructions per iteration combo
#     #Generate the ground truth based on RPS and Sigma Value
#     IdealModel_weighted = get_IdealModel_weighted(iter_rps, m, npeaks, T2, T2mu, sigma_i)
#     #Generate the data noisy based on RPS and Sigma Value
#     dat_noisy = calc_dat_noisy(A, TE, IdealModel_weighted, SNR)
#     #Get the parameter-selection methods reconstructions per iteration combo
#     f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)
#     f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
#     f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
#     f_rec_GCV = f_rec_GCV[:,0]
#     lambda_GCV = np.squeeze(lambda_GCV)
#     LRIto_ini_lam = lambda_LC
#     gamma_init = 5
#     maxiter = 75
#     f_rec_LocReg_LC, lambda_locreg_LC = LocReg_Ito_mod(dat_noisy, A, lambda_LC, gamma_init, maxiter)
#     # print("finished LocReg")
#     f_rec_oracle, lambda_oracle = minimize_OP(Alpha_vec, L, dat_noisy, A, len(T2), IdealModel_weighted)
#     # print("finished Oracle")
#     f_rec_GCV = f_rec_GCV.flatten()
#     f_rec_DP = f_rec_DP.flatten()
#     f_rec_LC = f_rec_LC.flatten()
#     f_rec_LocReg_LC = f_rec_LocReg_LC.flatten()
#     f_rec_oracle = f_rec_oracle.flatten()

#     #Generate Errors for each iteration combo
#     err_LC = error(IdealModel_weighted,f_rec_LC)
#     err_DP = error(IdealModel_weighted,f_rec_DP)
#     err_GCV = error(IdealModel_weighted,f_rec_GCV)
#     err_oracle = error(IdealModel_weighted,f_rec_oracle)
#     err_LR = error(IdealModel_weighted,f_rec_LocReg_LC)

#     #Plot a random simulation
#     random_sim = random.randint(0, n_sim)
#     if iter_sim == 1:
#         plot(iter_sim, iter_sigma, iter_rps)
#         print("Finished Plot")
#     else:
#         pass

#     feature_df = pd.DataFrame(columns = ["NR", 'Sigma', 'RPS_val', 'err_DP',"err_LC", "err_LR", "err_GCV", "err_oracle" ])
#     feature_df["NR"] = [iter_sim]
#     feature_df["Sigma"] = [sigma_i]
#     feature_df["RPS_val"] = [rps_val]
#     feature_df["err_DP"] = [err_DP]
#     feature_df["err_LC"] = [err_LC]
#     feature_df["err_LR"] = [err_LR]
#     feature_df["err_GCV"] = [err_GCV]
#     feature_df["err_oracle"] = [err_oracle]
#     return feature_df

if __name__ == '__main__':
    freeze_support()
    unif_sigma, diff_sigma = calc_diff_sigma(nsigma)
    T2, TE, Lambda, A, m,  SNR = load_Gaus(Gaus_info)
    T2mu = calc_T2mu(rps)

    print("Finished Assignments...")  

    lis = []
    #insert noise_arr:
    # profiler = cProfile.Profile()
    # profiler.enable()

    if preset_noise == False:
        with mp.Pool(processes = num_cpus_avail) as pool:
            with tqdm(total = len(target_iterator)) as pbar:
                for estimates_dataframe, noisereal in pool.imap_unordered(generate_estimates, range(len(target_iterator))):
                    lis.append(estimates_dataframe)
                    noise_list.append(noisereal)
                    pbar.update()
            pool.close()
            pool.join()
        noise_arr = np.array(noise_list)
        noise_arr = noise_arr.reshape(n_sim, nsigma, nrps, A.shape[0])
    else:
        with mp.Pool(processes = num_cpus_avail) as pool:
            with tqdm(total = len(target_iterator)) as pbar:
                for estimates_dataframe, noisereal in pool.imap_unordered(generate_estimates, range(len(target_iterator))):
                    lis.append(estimates_dataframe)
                    # print("target_iterator", target_iterator)
                    noise_list.append(noisereal)
                    pbar.update()
            pool.close()
            pool.join()
        noise_arr = np.array(noise_list)
        noise_arr = noise_arr.reshape(n_sim, nsigma, nrps, A.shape[0])
    # lis.append(generate_estimates(target_iterator[0]))

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(10) 

    print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
    df = pd.concat(lis, ignore_index= True)

    df.to_pickle(data_folder + f'/' + data_tag +'.pkl')    
    if preset_noise == False:
        np.save(data_folder + f'/' + data_tag + "noise_arr", noise_arr)
        print("noise array saved")
    else:
        print("Used preset noise array")
        # np.save(data_folder + f'/' + data_tag + "noiselessdata", noiseless_data)
        # np.save(data_folder + f'/' + data_tag + "noisydata", noisy_data)
        # print("test noise array saved")
        pass
