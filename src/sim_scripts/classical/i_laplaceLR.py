from src.utils.load_imports.load_classical import *
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
cwd = os.getcwd()

cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

pat_tag = "MRR"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
series_tag = "classical_prob"
simulation_save_folder = f"SimulationSets/{pat_tag}/{series_tag}"
# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + simulation_save_folder 

#Hyperparameters and Global Parameters
preset_noise = False
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_15Aug24noise_arr_modifiedalgo.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_test_21Aug24noise_arr.npy"
noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_13Aug24noise_arr.txt.npy"
testing = False
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_lamini_LCurve_dist_broadL_narrowR_15Aug24noise_arr_modifiedalgo.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_13Aug24noise_arr.txt.npy"
# noise_file_path = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter1_06Aug24noise_arr.txt.npy"

#Number of simulations:
n_sim = 10
SNR_value = 300

###LocReg hyperparameters
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
exp = 0.5
feedback = True
lam_ini_val = "LCurve"
dist_type = "i_laplace"
# gamma_init = 5
#gamma_init = 0.5 is best
gamma_init = 0.5


def create_result_folder(string, SNR, lam_ini_val, dist_type):
    folder_name = f"{cwd_full}/{string}_{date}_SNR_{SNR}_lamini_{lam_ini_val}_dist_{dist_type}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# def create_result_folder(string, SNR):
#     # Create a folder based on the current date and time
#     date = datetime.now().strftime("%Y%m%d")
#     #folder_name = f"c:/Users/kimjosy/LocReg_Regularization/{string}_{date}_SNR_{SNR}"
#     folder_name = f"{cwd_full}/{string}_{date}_SNR_{SNR}"
#     # folder_name = f"/Volumes/Lexar/NIH/Experiments/GridSearch/{string}_{date}_Run"
#     # Create the folder if it doesn't exist
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#     return folder_name

def minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g):
    OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
    OP_rhos = np.zeros((len(Alpha_vec)))
    for j in (range(len(Alpha_vec))):
        # A  = G.T @ G + Alpha_vec[j]**2 * L.T @ L 
        # b =  G.T @ data_noisy
        # exp = np.linalg.inv(A) @ b
        # exp = iterative_algorithm(G, data_noisy, B_1, B_2, Alpha_vec[j], Alpha_vec2[j], max_iter=1000, tol=1e-6)
        # exp, _ = nnls(A, b, maxiter = 400)
        # sol = np.linalg.lstsq(A,b, rcond=None)[0]
        # exp = np.linalg.solve(A,b)
        exp, _ = tikhonov_vec(U, s, V, data_noisy, Alpha_vec[j], x_0 = None, nargin = 5)

        # exp = fe.fnnls(A, b) 
        # print(np.linalg.cond(A))
        # exp = nnls(A, b, maxiter = 1e30)[0]
        # x = cp.Variable(nT2)
        # cost = cp.sum_squares(A @ x - b)
        # cost = cp.norm(A @ x - b, 2)**2
        # constraints = [x >= 0]
        # problem = cp.Problem(cp.Minimize(cost), constraints)
        # problem.solve(solver=cp.MOSEK, verbose=False)
        # exp,_ = fnnls(A,b)
        OP_x_lc_vec[:, j] = exp
        # print("L_x_lc_vec[:,j,k]", L_x_lc_vec[:,j,k])
        OP_rhos[j] = norm(OP_x_lc_vec[:,j] - g, 2)**2
    
    OP_log_err_norm = np.log10(OP_rhos)
    min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)

    min_x = Alpha_vec[min_index[0]]
    min_z = np.min(OP_log_err_norm)

    OP_min_alpha1 = min_x
    OP_min_alpha1_ind = min_index[0]
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    return f_rec_OP_grid, OP_min_alpha1


# n = 1000
# n = 100
# nT2 = n
# T2 = np.linspace(-np.pi/2,np.pi/2,n)
# TE = T2

# G, data_noiseless, g = wing(n, nargin=1, nargout=3)
# _,_,g = baart(n)
n = 100
nT2 = n
T2 = np.linspace(0,10,n)
TE = T2

G, data_noiseless, g, _ = i_laplace(n, example = 1, nargin=1)
# G, data_noiseless, g, _ = i_laplace(n, example = 1)
# print("G.shape", G.shape)
# print("G", G)

U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 1000
SD_noise= 1/SNR*max(abs(data_noiseless))
Lambda_vec = np.logspace(-6,-1,16)
nLambda = len(Lambda_vec)

nrun = 1
com_vec_DP = np.zeros(nrun)
com_vec_GCV = np.zeros(nrun)
com_vec_LC = np.zeros(nrun)
com_vec_ItoLR = np.zeros(nrun)
com_vec_ItoLR2 = np.zeros(nrun)
com_vec_ItoLR3 = np.zeros(nrun)
com_vec_ItoLR4 = np.zeros(nrun)
com_vec_oracle = np.zeros(nrun)
com_vec_ItoLR_feed = np.zeros(nrun)
com_vec_ItoLR2_feed = np.zeros(nrun)
com_vec_ItoLR3_feed = np.zeros(nrun)
com_vec_ItoLR4_feed = np.zeros(nrun)

# eps1 = 1e-3
# ep_min = 1e-2
# eps_cut = 10
# eps_floor = 1e-4
# eps1 = 1e-2
# ep_min = 1e-2
# eps_cut = 1.2
# eps_floor = 1e-4
# feedback = False
# exp = 0.5
# exp = 2/3
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
exp = 0.5
feedback = True
lam_ini_val = "LCurve"
dist_type = "i_laplace"
gamma_init = 0.5

def error(IdealModel,reconstr):
    true_norm = linalg_norm(IdealModel)
    err = linalg_norm(IdealModel - reconstr) / true_norm
    return err

for i in tqdm(range(nrun)):

    SD_noise= 1/SNR*max(abs(data_noiseless))

    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    #noise_arr[:,i] = noise.reshape(1,-1)
    #data_noisy = data_noiseless + noise[:,i]
    data_noisy = data_noiseless + noise
    lambda_LC,rho,eta,_ = l_curve(U,s,data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
    f_rec_LC,lambda_LC = tikhonov_vec(U, s, V, data_noisy, (lambda_LC), x_0 = None, nargin = 5)
    # f_rec_LC,_,_ = tikhonov(U,s,V,data_noisy,lambda_LC, nargin=5, nargout=1)
    com_vec_LC[i] = linalg_norm(g - f_rec_LC)

    delta1 = linalg_norm(noise)*1.05
    # x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta1, x_0= None, nargin = 5)
    # f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)
    
    x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta1, x_0= None, nargin = 5)
    # f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)
    L = np.eye(G.shape[1])
    x_true = None
    # f_rec_DP, lambda_DP = Tikhonov(G, data_noisy, L, x_true, regparam = 'DP', delta = delta1)
    f_rec_DP, lambda_DP = tikhonov_vec(U, s, V, data_noisy, lambda_DP, x_0 = None, nargin = 5)
    # f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)

    com_vec_DP[i] = linalg_norm(g - f_rec_DP)

    L = np.eye(G.shape[1])
    x_true = None
    # f_rec_GCV, reg_min = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')
    lambda_GCV,_,reg_param = gcv(U,s,data_noisy, method = 'Tikh', nargin = 3, nargout = 3)
    # f_rec_GCV, lambda_GCV = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')
    f_rec_GCV, lambda_GCV = tikhonov_vec(U, s, V, data_noisy, lambda_GCV, x_0 = None, nargin = 5)
    # f_rec_GCV,_,_ = tikhonov(U,s,V,data_noisy,lambda_GCV, nargin=5, nargout=1)

    # lambda_GCV,_,reg_param = gcv(U,s,data_noisy, method = 'Tikh', nargin = 3, nargout = 3)
    # f_rec_GCV,_,_ = tikhonov(U,s,V,data_noisy,lambda_GCV, nargin=5, nargout=1)
    com_vec_GCV[i] = linalg_norm(g - f_rec_GCV)

    # x0_ini = f_rec_LC
    # ep1 = 1e-8
    # ep2 = 1e-1
    # ep3 = 1e-3 
    # ep4 = 1e-4 
    # f_rec_Chuan, lambda_Chuan = LocReg_unconstrained(data_noisy, G, x0_ini, lambda_LC, ep1, ep2, ep3)

    if lam_ini_val == "LCurve" or lam_ini_val == "L-Curve":
        LRIto_ini_lam = lambda_LC
    elif lam_ini_val == "GCV" or lam_ini_val == "gcv":
        LRIto_ini_lam = lambda_GCV
    elif lam_ini_val == "DP" or lam_ini_val == "dp":
        LRIto_ini_lam = lambda_DP

    # LRIto_ini_lam = lambda_LC
    # maxiter = 75
    maxiter = 200
    #Deriv
    # f_rec_LocReg, lambda_locreg, test_frec1, test_lam1, numiterate = LocReg_Ito_classical(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter)
    f_rec_LocReg = f_rec_DP
    lambda_locreg = lambda_DP
    LR_Ito_lams = lambda_locreg
    com_vec_ItoLR[i] = linalg_norm(g - f_rec_LocReg)

    f_rec_oracle, oracle_lam = minimize_OP(Lambda_vec, L, data_noisy, G, nT2, g)
    com_vec_oracle[i] = linalg_norm(g - f_rec_oracle)

    #normalization:
    sum_x = np.sum(f_rec_LocReg)
    f_rec_LocReg = f_rec_LocReg / sum_x
    sum_oracle = np.sum(f_rec_oracle)
    f_rec_oracle = f_rec_oracle / sum_oracle
    sum_GCV = np.sum(f_rec_GCV)
    f_rec_GCV = f_rec_GCV / sum_GCV
    sum_LC = np.sum(f_rec_LC)
    f_rec_LC = f_rec_LC / sum_LC
    sum_DP = np.sum(f_rec_DP)
    f_rec_DP = f_rec_DP / sum_DP

    f_rec_GCV = f_rec_GCV.flatten()
    f_rec_DP = f_rec_DP.flatten()
    f_rec_LC = f_rec_LC.flatten()
    f_rec_LocReg = f_rec_LocReg.flatten()
    f_rec_oracle = f_rec_oracle.flatten()


    err_LC = error(g, f_rec_LC)
    err_DP = error(g, f_rec_DP)
    err_GCV = error(g, f_rec_GCV)
    err_oracle = error( g, f_rec_oracle)
    err_LR = error( g, f_rec_LocReg)
    #add 1/n for formalism
    if i % 1 == 0:
        #Plot the curves
        plt.figure(figsize=(12.06, 4.2))
        # Plotting the first subplot
        plt.subplot(1, 3, 1) 
        plt.plot(T2, g, linewidth=3, color='black', label='Ground Truth')
        plt.plot(T2, f_rec_LocReg, linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val} (Error: {round(err_LR,3)})')
        plt.plot(T2, f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label=f'Oracle (Error: {round(err_oracle,3)})')
        plt.plot(T2, f_rec_DP, linewidth=3, color='green', label=f'DP (Error: {round(err_DP,3)})')
        plt.plot(T2, f_rec_GCV, linestyle='--', linewidth=3, color='blue', label=f'GCV (Error: {round(err_GCV,3)})')
        plt.plot(T2, f_rec_LC, linestyle='-.', linewidth=3, color='purple', label=f'L-curve (Error: {round(err_LC,3)})')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('t', fontsize=20, fontweight='bold')
        plt.ylabel('f(t)', fontsize=20, fontweight='bold')
        ymax = np.max(g) * 1.15
        plt.ylim(0, ymax)

        # Plotting the second subplot
        plt.subplot(1, 3, 2)
        plt.plot(TE, G @ g, linewidth=3, color='black', label='Ground Truth')
        plt.plot(TE, G @ f_rec_LocReg, linestyle=':', linewidth=3, color='red', label= f'LocReg {lam_ini_val}')
        plt.plot(TE, G @ f_rec_oracle, linestyle='-.', linewidth=3, color='gold', label='Oracle')
        plt.plot(TE, G @ f_rec_DP, linewidth=3, color='green', label='DP')
        plt.plot(TE, G @ f_rec_GCV, linestyle='--', linewidth=3, color='blue', label='GCV')
        plt.plot(TE, G @ f_rec_LC, linestyle='-.', linewidth=3, color='purple', label='L-curve')
        plt.legend(fontsize=10, loc='best')
        plt.xlabel('s', fontsize=20, fontweight='bold')
        plt.ylabel('g(s)', fontsize=20, fontweight='bold')
        
        plt.subplot(1, 3, 3)
        plt.semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color='green', label='DP')
        plt.semilogy(T2, lambda_GCV * np.ones(len(T2)), linestyle=':', linewidth=3, color='blue', label='GCV')
        plt.semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='purple', label='L-curve')
        plt.semilogy(T2, lambda_locreg * np.ones(len(T2)), linestyle=':', linewidth=3, color='red', label=f'LocReg {lam_ini_val}')
        plt.semilogy(T2, oracle_lam * np.ones(len(T2)), linestyle='-.', linewidth=3, color='gold', label='Oracle')

        plt.legend(fontsize=10, loc='best')
        plt.xlabel('t', fontsize=20, fontweight='bold')
        plt.ylabel('Lambda', fontsize=20, fontweight='bold')

        plt.tight_layout()
        string = "Classical_LocReg_Comparison"
        file_path = create_result_folder(string, SNR, lam_ini_val, dist_type)
        plt.savefig(os.path.join(file_path, f"Simulation{i}_{dist_type}.png"))
        plt.close() 