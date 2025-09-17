from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
cwd = os.getcwd()

cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

pat_tag = "MRR"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
series_tag = "LambdaPerturb"
simulation_save_folder = f"SimulationSets/{pat_tag}/{series_tag}"
# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + simulation_save_folder 

#Hyperparameters and Global Parameters

###LocReg hyperparameters
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
exp = 0.5
feedback = False
lam_ini_val = "LCurve"

###Plotting hyperparameters
npeaks = 2
nsigma = 5
f_coef = np.ones(npeaks)
rps = np.linspace(1, 4, 5).T
# Lambda = Gaus_info['Lambda'].reshape(-1,1)
# Alpha_vec = np.logspace(-6, 0,16)
# Alpha_vec = Gaus_info['Lambda'].reshape(-1,1)
nrps = len(rps)

###SNR Values to Evaluate
# SNR_value = 50
# SNR_value = 200
# SNR_value = 500
# SNR_value = 1000
SNR_value = 200

#Load Data File
# file_path ="/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max0.pkl"
# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_50_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max031Jul24.pkl"
file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_200_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max0_73124.pkl"
# file_path = "/home/kimjosy/LocReg_Regularization-1/Simulations/num_of_basis_functions/lambda_16_SNR_500_nrun_20_sigma_min_2_sigma_max_4_basis2_5080lmbda_min-6lmbda_max1_073124.pkl"

Gaus_info = np.load(file_path, allow_pickle=True)
print(f"File loaded from: {file_path}")


###Number of noisy realizations; 20 NR is enough to until they ask for more noise realizations
n_sim = 1

#Naming for Data Folder
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')
data_path = "SimulationsSets/MRR/SpanRegFig"
add_tag = ""
data_head = "est_table"
data_tag = (f"{data_head}_SNR{SNR_value}_iter{n_sim}_{add_tag}{day}{month}{year}")
data_folder = (os.getcwd() + f'/{data_path}')
os.makedirs(data_folder, exist_ok = True)

#Number of tasks to execute
target_iterator = [(a,b,c) for a in range(n_sim) for b in range(nsigma) for c in range(nrps)]
num_cpus_avail = np.min([len(target_iterator),100])

#Functions
def create_result_folder(string, SNR,lam_ini_val):
    folder_name = f"{cwd_full}/{string}_{date}_SNR_{SNR}_lamini_{lam_ini_val}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

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
    return f_rec_OP_grid, OP_min_alpha1

def calc_T2mu(rps):
    mps = rps / 2
    nrps = len(rps)
    T2_left = 30 * np.ones(nrps)
    T2_mid = T2_left * mps
    T2_right = T2_left * rps
    T2mu = np.column_stack((T2_left,  T2_right))
    return T2mu

def calc_sigma_i(iter_i, diff_sigma):
    sigma_i = diff_sigma[iter_i, :]
    return sigma_i

def calc_rps_val(iter_j, rps):
    rps_val = rps[iter_j]
    return rps_val

def calc_diff_sigma(nsigma):
    unif_sigma = np.linspace(2, 5, nsigma).T
    diff_sigma = np.column_stack((unif_sigma, 3 * unif_sigma))
    return unif_sigma, diff_sigma

def load_Gaus(Gaus_info):
    T2 = Gaus_info['T2'].flatten()
    TE = Gaus_info['TE'].flatten()
    A = Gaus_info['A']
    Lambda = Gaus_info['Lambda'].reshape(-1,1)
    n, m = Gaus_info['A'].shape
    SNR = Gaus_info['SNR']
    return T2, TE, Lambda, A, m,  SNR

def calc_dat_noisy(A, TE, IdealModel_weighted, SNR):
    dat_noiseless = A @ IdealModel_weighted
    noise = np.column_stack([np.max(np.abs(dat_noiseless)) / SNR * np.random.randn(len(TE), 1)]) 
    dat_noisy = dat_noiseless + np.ravel(noise)
    return dat_noisy

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

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)
    return result

def generate_estimates(i_param_combo):
    def plot(iter_sim, iter_sigma, iter_rps):
        plt.figure(figsize=(12.06, 4.2))
        # Plotting the first subplot
        plt.subplot(1, 3, 1) 
        plt.plot(T2, IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        # plt.plot(T2, f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label=f'LocReg LCurve (Error: {round(err_LR,3)})')
        # plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle (Error: {round(err_oracle,3)})')
        plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {round(err_DP,3)})')
        plt.plot(T2, f_rec_pet,  linewidth=3, color='red', label=f'Perturb (Error: {round(err_DP_pet,3)})')
        # plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {round(err_GCV,3)})')
        # plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Error: {round(err_LC,3)})')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
        plt.ylabel('Amplitude', fontsize=20, fontweight='bold')
        ymax = np.max(IdealModel_weighted) * 1.15
        plt.ylim(0, ymax)

        # Plotting the second subplot
        plt.subplot(1, 3, 2)
        plt.plot(TE, A @ IdealModel_weighted, linewidth=3, color='black', label='Ground Truth')
        # plt.plot(TE, A @ f_rec_LocReg_LC, linestyle=':', linewidth=3, color='red', label='LocReg LCurve')
        # plt.plot(TE, A @ f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle')
        plt.plot(TE, A @ f_rec_DP, linewidth=3, color='green', label='DP')
        plt.plot(TE, A @ f_rec_pet, linewidth=3, color='red', label=f'DP Perturb')

        # plt.plot(TE, A @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
        # plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('TE', fontsize=20, fontweight='bold')
        plt.ylabel('Intensity', fontsize=20, fontweight='bold')
        
        plt.subplot(1, 3, 3)
        plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
        plt.semilogy(T2, lambda_pet * np.ones(len(T2)), linewidth=3, color='red', label='DP Perturb')

        # plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
        # plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
        # plt.semilogy(T2, lambda_locreg_LC * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label='LocReg LCurve')
        # plt.semilogy(T2, lambda_oracle * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')

        plt.legend(fontsize=10, loc='best')
        plt.xlabel('T2', fontsize=20, fontweight='bold')
        plt.ylabel('Lambda', fontsize=20, fontweight='bold')

        plt.tight_layout()
        string = "MRR_1D_LocReg_ComparisonLambdaPerturb"
        file_path = create_result_folder(string, SNR, lam_ini_val)
        plt.savefig(os.path.join(file_path, f"Simulation{iter_sim}_Sigma{iter_sigma}_RPS{iter_rps}.png"))
        plt.close() 

    iter_sim, iter_sigma, iter_rps = target_iterator[i_param_combo]
    L = np.eye(A.shape[1])
    sigma_i = diff_sigma[iter_sigma, :]
    rps_val = calc_rps_val(iter_rps, rps)

    # Profile the function calls individually
    IdealModel_weighted = get_IdealModel_weighted(iter_rps, m, npeaks, T2, T2mu, sigma_i)
    dat_noisy = calc_dat_noisy( A, TE, IdealModel_weighted, SNR)
    f_rec_DP, lambda_DP = discrep_L2(dat_noisy, A, SNR, Lambda)

    lambda_pet = lambda_DP * np.ones(len(T2))
    lambda_pet[100:104] = lambda_pet[100:104] + 0.1
    f_rec_pet, _, _ = nonnegtik_hnorm(A, dat_noisy, lambda_pet, '0', nargin =4)


    # f_rec_LC, lambda_LC = Lcurve(dat_noisy, A, Lambda)
    # f_rec_GCV, lambda_GCV = GCV_NNLS(dat_noisy, A, Lambda)
    # f_rec_GCV = f_rec_GCV[:, 0]
    # lambda_GCV = np.squeeze(lambda_GCV)
    # LRIto_ini_lam = lambda_LC
    gamma_init = 5
    maxiter = 75
    # f_rec_LocReg_LC, lambda_locreg_LC = profile_function(LocReg_Ito_mod, dat_noisy, A, lambda_LC, gamma_init, maxiter)
    # f_rec_oracle, lambda_oracle = profile_function(minimize_OP, Alpha_vec, L, dat_noisy, A, len(T2), IdealModel_weighted)
    # f_rec_LocReg_LC, lambda_locreg_LC = LocReg_Ito_mod(dat_noisy, A, lambda_LC, gamma_init, maxiter)
    # f_rec_oracle, lambda_oracle = minimize_OP(Lambda, L, dat_noisy, A, len(T2), IdealModel_weighted)

    # Flatten results
    # f_rec_GCV = f_rec_GCV.flatten()
    f_rec_DP = f_rec_DP.flatten()
    # f_rec_LC = f_rec_LC.flatten()
    # f_rec_LocReg_LC = f_rec_LocReg_LC.flatten()
    # f_rec_oracle = f_rec_oracle.flatten()

    # Calculate errors
    # err_LC = error(IdealModel_weighted, f_rec_LC)
    err_DP = error(IdealModel_weighted, f_rec_DP)
    err_DP_pet = error(IdealModel_weighted, f_rec_pet)

    # err_GCV = error(IdealModel_weighted, f_rec_GCV)
    # err_oracle = error(IdealModel_weighted, f_rec_oracle)
    # err_LR = error(IdealModel_weighted, f_rec_LocReg_LC)

    # Plot a random simulation
    # random_sim = random.randint(0, n_sim)
    plot(iter_sim, iter_sigma, iter_rps)
    print("Finished Plot")


    # Create DataFrame
    # feature_df = pd.DataFrame(columns=["NR", 'Sigma', 'RPS_val', 'err_DP', "err_LC", "err_LR", "err_GCV", "err_oracle"])
    # feature_df["NR"] = [iter_sim]
    # feature_df["Sigma"] = [sigma_i]
    # feature_df["RPS_val"] = [rps_val]
    # feature_df["err_DP"] = [err_DP]
    # feature_df["err_LC"] = [err_LC]
    # feature_df["err_LR"] = [err_LR]
    # feature_df["err_GCV"] = [err_GCV]
    # feature_df["err_oracle"] = [err_oracle]
    return 


if __name__ == '__main__':
    freeze_support()
    unif_sigma, diff_sigma = calc_diff_sigma(nsigma)
    T2, TE, Lambda, A, m,  SNR = load_Gaus(Gaus_info)
    T2mu = calc_T2mu(rps)

    print("Finished Assignments...")  

    lis = []

    # profiler = cProfile.Profile()
    # profiler.enable()

    with mp.Pool(processes = num_cpus_avail) as pool:
        with tqdm(total = len(target_iterator)) as pbar:
            for estimates_dataframe in pool.imap_unordered(generate_estimates, range(len(target_iterator))):
                lis.append(estimates_dataframe)
                pbar.update()
        pool.close()
        pool.join()
    # lis.append(generate_estimates(target_iterator[0]))

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(10) 

    print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
    # df = pd.concat(lis, ignore_index= True)

    # df.to_pickle(data_folder + f'/' + data_tag +'.pkl')    
