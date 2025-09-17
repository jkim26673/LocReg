# #Packages
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
cwd = os.getcwd()

cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

pat_tag = "MRR"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
series_tag = "oldtest"
output_folder = f"SimulationSets/{pat_tag}/{series_tag}"

# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + output_folder 


def create_result_folder(string, SNR):
    # Create a folder based on the current date and time
    date = datetime.now().strftime("%Y%m%d")
    #folder_name = f"c:/Users/kimjosy/LocReg_Regularization/{string}_{date}_SNR_{SNR}"
    folder_name = f"{cwd_full}/{string}_{date}_SNR_{SNR}"
    # folder_name = f"/Volumes/Lexar/NIH/Experiments/GridSearch/{string}_{date}_Run"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

#Create N number of Penalty Diagonal Matrices
def generate_B_matrices(N, M):
    B_matrices = []

    if M % N != 0:
        ValueError("M must be divisible by N")
    ones_per_matrix = M // N  # Calculate number of ones per matrix

    for i in range(N):
        start = ones_per_matrix * i  # Calculate the starting index for ones in this matrix
        end = min(ones_per_matrix * (i + 1), M)  # Calculate the ending index for ones in this matrix
        diagonal_values = np.zeros(M)
        diagonal_values[start:end] = 1  # Set ones in the appropriate range
        B_matrices.append(np.diag(diagonal_values))
    return B_matrices

def minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g):
    OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
    OP_rhos = np.zeros((len(Alpha_vec)))
    for j in (range(len(Alpha_vec))):
        X, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j]**2, '0', nargin = 4)
        # A  = G.T @ G + Alpha_vec[j]**2 * L.T @ L 
        # b =  G.T @ data_noisy
        # val = Alpha_vec[j]**2 * np.ones(G.shape[1])
        # eps = 1e-2
        # A = (G.T @ G + np.diag(val))
        # ep4 = np.ones(G.shape[1]) * eps
        # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * (val)
        # # exp = np.linalg.inv(A) @ b
        # # exp = iterative_algorithm(G, data_noisy, B_1, B_2, Alpha_vec[j], Alpha_vec2[j], max_iter=1000, tol=1e-6)
        # try:
        #     # Solve the problem using nnls
        #     sol = nnls(A, b, maxiter=1000)[0]
        #     sol = sol - eps
        #     # Ensure non-negative solution
        #     sol = np.array(sol)
        #     sol[sol < 0] = 0
            
        # except Exception as e:
        #     # Handle exceptions if nnls fails
        #     y = cp.Variable(G.shape[1])
        #     cost = cp.norm(A @ y - b, 2)**2
        #     constraints = [y >= 0]
        #     problem = cp.Problem(cp.Minimize(cost), constraints)
        #     try:
        #         problem.solve(solver=cp.MOSEK, verbose=False)
        #         #you can try a different solver if you don't have the license for MOSEK doesn't work (should be free for one year)
        #     except Exception as e:
        #         print(e)
        #     # reconst = y.value
        #     # reconst,_ = nnls(A,b, maxiter = 10000)
        #     sol = y.value
        #     # print(f"An error occurred during nnls optimization, using MOSEK instead: {e}")
        #     sol = sol - eps
        #     # Ensure non-negative solution
        #     sol = np.array(sol)
        #     sol[sol < 0] = 0
        # print("exp shape", exp.shape)
        # print("g.shape", g.shape)
        # sol = np.linalg.lstsq(A,b, rcond=None)[0]
        # exp = np.linalg.solve(A,b)
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
        OP_x_lc_vec[:, j] = X
        # print("L_x_lc_vec[:,j,k]", L_x_lc_vec[:,j,k])
        OP_rhos[j] = np.linalg.norm(OP_x_lc_vec[:,j] - g, 2)**2
    
    OP_log_err_norm = np.log10(OP_rhos)
    min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)

    min_x = Alpha_vec[min_index[0]]
    min_z = np.min(OP_log_err_norm)

    OP_min_alpha1 = min_x
    OP_min_alpha1_ind = min_index[0]
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    return f_rec_OP_grid, OP_min_alpha1
# def minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g):
#     OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
#     OP_rhos = np.zeros((len(Alpha_vec)))
#     for j in tqdm(range(len(Alpha_vec))):
#         A  = G.T @ G + Alpha_vec[j]**2 * L.T @ L 
#         b =  G.T @ data_noisy
#         # exp = np.linalg.inv(A) @ b
#         # exp = iterative_algorithm(G, data_noisy, B_1, B_2, Alpha_vec[j], Alpha_vec2[j], max_iter=1000, tol=1e-6)
#         # exp, _ = nnls(A, b, maxiter = 1000, atol = 1e-6)
#         x = cp.Variable(G.shape[1])
#         # cost = cp.sum_squares(A @ x - b)
#         cost = cp.norm(A @ x - b, 2)**2
#         constraints = [x >= 0]
#         problem = cp.Problem(cp.Minimize(cost), constraints)
#         problem.solve(solver=cp.MOSEK, verbose=False)
#         exp = x.value
#         # exp = np.linalg.solve(A,b)
#         # exp = fe.fnnls(A, b) 
#         # print(np.linalg.cond(A))
#         # exp = nnls(A, b, maxiter = 1e30)[0]
#         # x = cp.Variable(nT2)
#         # cost = cp.sum_squares(A @ x - b)
#         # cost = cp.norm(A @ x - b, 2)**2
#         # constraints = [x >= 0]
#         # problem = cp.Problem(cp.Minimize(cost), constraints)
#         # problem.solve(solver=cp.MOSEK, verbose=False)
#         # exp,_ = fnnls(A,b)
#         OP_x_lc_vec[:, j] = exp
#         # print("L_x_lc_vec[:,j,k]", L_x_lc_vec[:,j,k])
#         OP_rhos[j] = norm(OP_x_lc_vec[:,j] - g, 2)**2
    
#     OP_log_err_norm = np.log10(OP_rhos)
#     min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)

#     min_x = Alpha_vec[min_index[0]]
#     min_z = np.min(OP_log_err_norm)

#     OP_min_alpha1 = min_x
#     OP_min_alpha1_ind = min_index[0]
#     f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
#     return f_rec_OP_grid, OP_min_alpha1
# sigma1 = 10
# mu1 = 80
# sigma2 = 3
# mu2 = 100
# SNR = 1000

def minimize(lam_vector, G, nT2, data_noisy):
    machine_eps = np.finfo(float).eps
    # ep = machine_eps
    # A = (g_mat.T @ g_mat + np.diag(lam_vector))
    # b = g_mat.T @ data_noisy + (g_mat.T @ g_mat * ep) + ep*lam_vector
    # b = g_mat.T @ data_noisy
    # ep = machine_eps
    eps = 1e-2
    A = (G.T @ G + np.diag(lam_vector))
    # b = G.T @ data_noisy + (G.T @ G * eps) @ np.ones(nT2) + eps*lam_vector
    ep4 = np.ones(G.shape[1]) * eps
    b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vector
    # b = (G.T @ data_noisy)
    sol = nnls(A, b, maxiter=1000)[0]
    sol = sol - eps
    # sol2 = np.linalg.lstsq(A,b)[0]
    # sol3 = np.linalg.solve(A,b)
    sol = np.array(sol)
    sol[sol < 0] = 0
    # machine_eps = np.finfo(float).eps
    # print(type(sol))

    # sol[sol < 0] = machine_eps

    return sol

#Initialize the MRR Problem
#Generate the TE values/ time
def run_all(sigma1, mu1, sigma2, mu2, SNR, threebump):
    TE = np.arange(1,512,4).T
    #Generate the T2 values
    T2 = np.arange(1,201).T
    #Generate G_matrix
    G = np.zeros((len(TE),len(T2)))
    #For every column in each row, fill in the e^(-TE(i))
    for i in range(len(TE)):
        for j in range(len(T2)):
            G[i,j] = np.exp(-TE[i]/T2[j])
    nTE = len(TE)
    nT2 = len(T2)

    if threebump==True:
        sigma3 = 5
        mu3 = 160
        g1 = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
        g2 = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
        g3 = (1 / (np.sqrt(2 * np.pi) * sigma3)) * np.exp(-((T2 - mu3) ** 2) / (2 * sigma3 ** 2))
        g  = (g1 + g2 + g3)/(3)
    else:
        #Create ground truth
        g = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
        g = g + (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
        g = g/2

    #Generate 2 Penalty Matrices for the 2P Ito Problem:
    # N = 2
    # M = nT2
    # B_matrices = generate_B_matrices(N, M)

    #Generate noisy data
    data_noiseless = np.dot(G, g)
    U, s, V = csvd(G, tst = None, nargin = 1, nargout = 3)
    # s = np.diag(s)

    SD_noise = 1 / SNR
    # noise = np.random.normal(0, SD_noise, size=data_noiseless.shape)
    # # data_noisy = data_noiseless + noise

    # # #Do DP and LC
    # delta = np.linalg.norm(noise)*1.05
    # x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta, x_0= None, nargin = 5)
    # reg_corner,rho,eta,_ = l_curve(U,s,data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
    # # print("finished l-curve")
    # Lambda = np.ones(len(T2)) * reg_corner
    # Lambda = Lambda.reshape(-1,1)
    # f_rec_LC, lambda_LC = Lcurve(data_noisy, G, Lambda)
    # # print("finished LCurve")
    # # f_rec_LC,_,_ = tikhonov(U,s,V,data_noisy,reg_corner, nargin=5, nargout=1)
    # # print("finished tikhnov")

    # L = np.eye(G.shape[1])
    # x_true = None
    # f_rec_GCV, lambda_GCV = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')

    # f_rec_GCV = np.ones(G.shape[1])

    #Chuan's Algorithm
    # maxiter = 400
    # f_rec_Chuan, lambda_Chuan = Chuan_LR(data_noisy, G, f_rec_LC, maxiter)
    # print("finished Chuan")

    #Ito_LocReg(this is the Ito algorithm for 2 parameter problem)

    #Initialize gamma
    # lam_ini = lambda_LC
    # LR_ini_lam = lam_ini * np.ones(nT2)

    #Setup for 2P Ito Problem
    # gamma_init = 5
    # param_num = 2
    # maxiter = 50
    # B_mats = B_matrices
    # LRIto_ini_lam = lambda_LC
    # best_f_rec, fin_etas = Ito_LocReg(data_noisy, G, LRIto_ini_lam, gamma_init, param_num, B_mats, maxiter)
    # print("Completed 2P Ito")

    #Construct 2P Lambdas for Plotting
    # Ito_lams_2P = fin_etas[0] * np.diag(B_mats[0]) + fin_etas[1] * np.diag(B_mats[1])


    # Run LocReg_Ito_mod (This is the Ito Problem for NP or N parameters)
    # # print("Starting NP Ito")
    # LR_mod_rec, fin_lam, c_array, lam_arr_fin, sol_arr_fin = LocReg_Ito_mod(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter)
    # LR_Ito_lams = fin_lam
    Alpha_vec = np.logspace(-8,-1,250)
    # f_rec_OP_grid, oracle_lam = minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g)
    # plt.show()
    # plt.close()
    # plt.plot(f_rec_ito)
    # plt.plot(g)
    # plt.legend(["Ito", "Ground Truth"])
    # plt.show()
    # plt.close()

    #Plot the solutions and lambdas for the 5th and 6th iteration
    # fig, axs = plt.subplots(2,1, figsize=(6, 6))
    # axs[0].semilogy(T2, lam_arr_fin[6], label = "Iteration 5")
    # axs[0].semilogy(T2, lam_arr_fin[7], label = "Iteration 6")
    # axs[1].plot(T2, sol_arr_fin[6], label = "Iteration 5")
    # axs[1].plot(T2, sol_arr_fin[7], label = "Iteration 6")
    # axs[0].legend()
    # axs[1].legend()
    # plt.tight_layout()
    # plt.show()

    # print("sol_arr_fin[6] - sol_arr_fin[7]", np.linalg.norm(sol_arr_fin[6] - sol_arr_fin[7]))



    # Uncomment below to run this Ito Problem for 100 simulations against LC/othermethods

    n = nT2
    nrun = 20
    com_vec_DP = np.zeros(nrun)
    com_vec_GCV = np.zeros(nrun)
    com_vec_LC = np.zeros(nrun)
    com_vec_locreg_DP = np.zeros(nrun)
    com_vec_locreg_LC = np.zeros(nrun)
    com_vec_locreg_GCV = np.zeros(nrun)
    com_vec_ItoLR = np.zeros(nrun)
    com_vec_ItoLR2P = np.zeros(nrun)
    com_vec_oracle = np.zeros(nrun)

    # com_vec_locreg_ne = np.zeros(nrun)

    res_vec_DP = np.zeros((n,nrun))
    res_vec_GCV = np.zeros((n,nrun))
    res_vec_LC = np.zeros((n,nrun))
    res_vec_locreg_DP = np.zeros((n,nrun))
    res_vec_locreg_GCV = np.zeros((n,nrun))
    res_vec_locreg_LC = np.zeros((n,nrun))
    # res_vec_locreg_ne = np.zeros((n,nrun))
    res_vec_ItoLR = np.zeros((n, nrun))
    res_vec_ItoLR2P = np.zeros((n, nrun))

    noise_size = int(data_noiseless.shape[0])
    noise_arr = np.zeros((noise_size, nrun))
    lam_DP = np.zeros((n,nrun))
    lam_LC = np.zeros((n,nrun))
    lam_GCV = np.zeros((n,nrun))
    lam_LocReg_DP = np.zeros((n,nrun))
    lam_LocReg_LC = np.zeros((n,nrun))
    lam_LocReg_GCV = np.zeros((n,nrun))
    lam_ItoLR = np.zeros((n,nrun))
    lam_ItoLR2P = np.zeros((n,nrun))

    # lam_LocReg_ne = np.zeros((n,nrun))

    ep1 = 1e-8
        # % 1/(|x|+ep1)
    ep2 = 1e-1
        # % norm(dx)/norm(x)
    ep3 = 1e-3
    # % norm(x_(k-1) - x_k)/norm(x_(k-1))
    ep4 = 1e-4


    for i in tqdm(range(nrun)):
        # data_noisy = data_noiseless + noise_arr[:,i]
        noise = np.random.normal(0,SD_noise, data_noiseless.shape)
        # noise_arr[:,i] = noise.reshape(1,-1)
        # #data_noisy = data_noiseless + noise[:,i]
        data_noisy = data_noiseless + noise
        delta1 = np.linalg.norm(noise)*1.05
        # delta1 = norm(noise)
        safety_fact = 1.05
        # delta1 = np.sqrt(np.abs(nTE - nT2)) * SD_noise * safety_fact
        # delta1 = safety_fact * np.sqrt(nT2) * np.max(data_noisy) / SNR
        # delta1 = np
        # x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta1, x_0= None, nargin = 5)
        # Lambda = np.ones(len(T2)) * lambda_DP
        # Lambda = Lambda.reshape(-1,1)
        L = np.eye(G.shape[1])
        x_true = None
        # f_rec_DP, lambda_DP = discrep_L2(data_noisy, G, SNR, Lambda)
        # f_rec_DP, lambda_DP = Tikhonov(G, data_noisy, L, x_true, regparam = 'DP', delta = delta1)
        # f_rec_DP, lambda_DP = tikhonov_vec(U, s, V, data_noisy, np.sqrt(lambda_DP), x_0 = None, nargin = 5)
        Lambda = np.ones(len(T2))* lambda_DP

        #discrep_L2 assumes lambda**2
        f_rec_DP,_ = discrep_L2(data_noisy, G, SNR, Lambda)
        f_rec_DP = f_rec_DP.flatten()
        lambda_DP = np.sqrt(lambda_DP)
        # plt.plot(f_rec_DP, label = "gazzola")
        # plt.plot(test_DP, label = "tikhonov")
        # plt.legend()
        # plt.show()
        # plt.close()

        # lambda_DP = np.sqrt(lambda_DP)
        # l_curve returns single lambda
        reg_corner,rho,eta,_ = l_curve(U,s,data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
        Lambda = np.ones(len(T2)) * reg_corner**2
        Lambda = Lambda.reshape(-1,1)
        #Lcurve calculates assumes an initial input of lambda**2
        f_rec_LC, lambda_LC = Lcurve(data_noisy, G, (Lambda))
        # f_rec_LC2 = np.linalg.solve(G.T@G + lambda_LC*L.T@L, G.T@data_noisy)
        lambda_LC = np.sqrt(lambda_LC)
        # GCV
        # f_rec_GCV, lambda_GCV = GCV_NNLS(data_noisy, G, Lambda)
        f_rec_GCV, lambda_GCV = Tikhonov(G, data_noisy, L, x_true, regparam = 'gcv')
        Lambda = np.ones(len(T2))* (lambda_GCV)

        #GCV_NNLS assumes Lambda**2
        f_rec,_ = GCV_NNLS(data_noisy, G, Lambda)
        f_rec_GCV = f_rec[:,0]
        lambda_GCV = np.sqrt(lambda_GCV)
        # f_rec_GCV, lambda_GCV = tikhonov_vec(U, s, V, data_noisy, np.sqrt(lambda_GCV), x_0 = None, nargin = 5)
        # lambda_GCV = np.sqrt(lambda_GCV)
        # f_rec_GCV = np.ones(G.shape[1])
        LRIto_ini_lam = lambda_LC
        # gamma_init = 10
        gamma_init = 5
        maxiter = 75
        # best_f_rec, fin_etas = Ito_LocReg(data_noisy, G, LRIto_ini_lam, gamma_init, param_num, B_mats, maxiter)
        # # print("Completed 2P Ito")
        # com_vec_ItoLR2P[i] = norm(g - best_f_rec)
        # lam_ItoLR2P[:,i] = fin_etas

        # res_vec_ItoLR2P[:,i] =

        # print("Starting NP Ito")

        LR_mod_rec, fin_lam = LocReg_Ito_mod(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter)
        # fin_lam = np.sqrt(fin_lam)
        com_vec_ItoLR[i] = norm(g - LR_mod_rec)
        lam_ItoLR[:,i] = fin_lam
        # fin_lam = np.sqrt(fin_lam)
 
        # f_test = minimize(fin_lam, G, nT2, data_noisy)

        # Alpha_vec = np.sqrt(Alpha_vec)
        f_rec_OP_grid, oracle_lam = minimize_OP(Alpha_vec, L, data_noisy, G, nT2, g)
        # f_test = np.linalg.solve(G.T@G + np.diag(fin_lam), G.T@data_noisy)
        # oracle_lam = np.sqrt(oracle_lam)
        # print("oracle lam:", oracle_lam)
        # f_test = nnls(G.T @ G + oracle_lam * L.T @ L, G.T @ data_noisy)[0]
        # f_test = nnls(G.T @ G + np.diag(fin_lam), G.T @ data_noisy)[0]

        com_vec_oracle[i] = norm(g - f_rec_OP_grid)
    #Calculate the errors
        err_DP = np.linalg.norm(g - f_rec_DP)
        err_DP = np.linalg.norm(g - f_rec_DP)

        err_Lcurve = np.linalg.norm(g - f_rec_LC)
        # err_Ito2P = np.linalg.norm(g - best_f_rec)
        err_LR_Ito = np.linalg.norm(g - LR_mod_rec)
        # err_Chuan = np.linalg.norm(g - f_rec_Chuan)
        err_GCV = np.linalg.norm(g - f_rec_GCV)
        err_oracle = np.linalg.norm(g - f_rec_OP_grid)
        # err_test = np.linalg.norm(g - f_test)
        # print("f_rec_GCV", f_rec_GCV.shape)
        #Plot the curves
        if i % 1 == 0:
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            # plt.subplots_adjust(wspace=0.3)

            # Plotting the first subplot
            # plt.subplot(1, 3, 1)
            ymax = np.max(g) * 1.15
            axs[0, 0].plot(T2, g, color = "black",  label = "Ground Truth")
            axs[0, 0].plot(T2, f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
            axs[0, 0].plot(T2, LR_mod_rec, color = "blue",  label = "Ito NP")
            axs[0, 0].plot(T2, f_rec_LC, color = "orange", label = "LC 1P")
            # axs[0, 0].plot(T2, f_test, color = "pink", label = "test")
            axs[0, 0].plot(T2, f_rec_GCV, color = "green", label = "GCV 1P")
            axs[0, 0].plot(T2, f_rec_DP, color = "red", label = "DP 1P")

            # axs[0, 0].plot(T2, f_rec_Chuan, color = "red", label = "Chuan")
            axs[0, 0].set_xlabel('T2 Relaxation Time', fontsize=20, fontweight='bold')
            axs[0, 0].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
            axs[0, 0].legend(fontsize=10, loc='best')
            axs[0, 0].set_ylim(0, ymax)

            # Plotting the second subplot
            # plt.subplot(1, 3, 2)
            axs[0, 1].plot(TE, G @ g, linewidth=3, color='black', label='Ground Truth')
            axs[0, 1].plot(TE, G @ f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
            axs[0, 1].plot(TE, G @ LR_mod_rec, color = "blue",  label = "Ito NP")
            axs[0, 1].plot(TE, G @ f_rec_LC, color = "orange", label = "LC 1P")
            axs[0, 1].plot(TE, G @ f_rec_GCV, color = "green", label = "GCV 1P")
            axs[0, 1].plot(TE, G @ f_rec_DP, color = "red", label = "DP 1P")

            # axs[0, 1].plot(TE, G @ f_rec_Chuan, color = "red", label = "Chuan")
            axs[0, 1].legend(fontsize=10, loc='best')
            axs[0, 1].set_xlabel('TE', fontsize=20, fontweight='bold')
            axs[0, 1].set_ylabel('Intensity', fontsize=20, fontweight='bold')

            # plt.subplot(1, 3, 3)
            axs[1, 0].semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='orange', label='LC 1P')
            axs[1, 0].semilogy(T2, oracle_lam * np.ones(len(T2)), linewidth=3, color='purple', label='Oracle 1P')
            axs[1, 0].semilogy(T2, fin_lam, color = "blue", linewidth=3,  label = "Ito NP")
            # axs[1,0].semilogy(T2, lambda_Chuan, color = "red",  label = "Chuan")
            axs[1, 0].semilogy(T2, lambda_GCV * np.ones(len(T2)),linewidth=3,  color = "green", label = "GCV 1P")
            axs[1, 0].semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color = "red", label = "DP 1P")
            axs[1, 0].legend(fontsize=10, loc='best')
            axs[1, 0].set_xlabel('T2', fontsize=20, fontweight='bold')
            axs[1, 0].set_ylabel('Lambda', fontsize=20, fontweight='bold')
            # ymax2 = 1.5 * np.max(lambda_LC)
            # axs[1, 0].set_ylim(0, ymax2)

            table_ax = axs[1, 1]
            table_ax.axis('off')

            # Define the data for the table (This is part of the plot)
            data = [
                # ["L-Curve Lambda", lambda_LC.item()],
                # ["Initial Lambda for Ito", LRIto_ini_lam],
                # ["Initial Eta2 for Ito", round(lam_ini, 4)],
                # ["Initial Eta2 for Ito", LRIto_ini_lam],
                # ["Final Eta1 for Ito", fin_etas[0].item()],
                # ["Final Eta2 for Ito", fin_etas[1].item()],
                ["1P Conventional DP", err_DP.item()],
                ["1P Conventional L-Curve", err_Lcurve.item()],
                ["1P Conventional GCV", err_GCV.item()],
                ["Ito NP", err_LR_Ito.item()],
                # ["error Ito 2P", err_Ito2P.item()],
                ["1P Oracle", err_oracle.item()],
                # ["error test", err_test.item()],
                # ["error Chuan", err_Chuan.item()],
                # ["SNR", SNR]

                # ["Initial Lambdas for Ito Loc", LR_ini_lam],
                # ["Final Lambdas for Ito Loc", LR_Ito_lams]
            ]

            table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Parameter Selection Method', 'Median MSE'])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.2)
            table_ax.set_title(f'Laplace Problem Table SNR {SNR}', fontsize=16, fontweight='bold', y=0.7)  # Adjust the y position

            # Create the table
            # table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
            # table.auto_set_font_size(False)
            # table.set_fontsize(12)
            # table.scale(1.2, 1.2)

            #Save the results in the save results folder
            plt.tight_layout()
            string = "comparison"
            file_path = create_result_folder(string, SNR)
            plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve_{i}"))
            print(f"Saving comparison plot is complete")
        # Multi_Reg_Gaussian_Sum1
        # f_rec, alpha_L2, F_info, C_L2 = Multi_Reg_Gaussian_Sum1(dat_noisy, Gaus_info)
        # LocReg
        #maxiter = 400
        # ep1 = 1e-2; # 1/(|x|+ep1)
        # ep2 = 1e-2; # norm(dx)/norm(x)
        # ep3 = 1e-2; # norm(x_(k-1) - x_k)/norm(x_(k-1))
        # ep4 = 1e-4; # lb for ep1
        # maxiter = 400
        # f_rec_LocReg_LC, lambda_locreg_LC = Chuan_LR(data_noisy, G, f_rec_LC, maxiter)
        # f_rec_LocReg_DP, lambda_locreg_DP = LocReg(data_noisy, G, f_rec_DP, maxiter)
        # f_rec_LocReg_GCV, lambda_locreg_GCV = LocReg(data_noisy, G, f_rec_GCV.ravel(), maxiter)
        # # L-curve
        #reg_corner,rho,eta = l_curve(U,s,data_noisy)
        # reg_corner,rho,eta,_ = l_curve(U,s,data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
        lam_LC[:,i] = lambda_LC
        # f_rec_LC,_,_ = tikhonov(U,s,V,data_noisy,reg_corner, nargin=5, nargout=1)
        # com_vec_LC[i] = norm(g - f_rec_LC)
        com_vec_LC[i] = np.mean(norm(g - f_rec_LC)**2)
        # res_vec_LC[:,i] = f_rec_LC
        # %% GCV
        # reg_min,_,reg_param = gcv(U,s,data_noisy, method = None, nargin = 3, nargout = 3)
        # lam_GCV[:,i] = lambda_GCV
        # f_rec_GCV,_,_ = tikhonov(U,s,V,data_noisy,reg_min, nargin=5, nargout=1)
        # com_vec_GCV[i] = norm(g - f_rec_GCV)
        com_vec_GCV[i] = np.mean(norm(g - f_rec_GCV)**2)

        # res_vec_GCV[:,i] = f_rec_GCV
        # f_rec_GCV = res_vec_GCV[:,i]
        # reg_min = lam_GCV[:,i]
        # %% DP
        #delta = norm(noise[:,i])*1.05
        delta = norm(noise)*1.05
        # x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta, x_0= None, nargin = 5)
        lam_DP[:,i] = lambda_DP
        # f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)
        # com_vec_DP[i] = norm(g - f_rec_DP)
        com_vec_DP[i] = np.mean(norm(g - f_rec_DP)**2)
        # res_vec_DP[:,i] = f_rec_DP

        # %% locreg
        # x0_ini = f_rec_GCV
        # # %     lambda_ini = reg_corner;
        # ep1 = 1e-8
        # # % 1/(|x|+ep1)
        # ep2 = 1e-1
        # # % norm(dx)/norm(x)
        # ep3 = 1e-3
        # # % norm(x_(k-1) - x_k)/norm(x_(k-1))
        # ep4 = 1e-4
        # # % lb for ep1

        # f_rec_locreg, lambda_locreg = LocReg_unconstrained_NE(data_noisy, G, x0_ini, lambda_DP, ep1, ep2, ep3)
        # lam_LocReg_ne[:,i] = lambda_locreg
        # com_vec_locreg_ne[i] = norm(g - f_rec_locreg)
        # res_vec_locreg_ne[:,i] = f_rec_locreg
        # f_rec_locreg, lambda_locreg = LocReg_unconstrained(data_noisy, G, x0_ini, reg_corner, ep1, ep2, ep3)
        # lam_LocReg_LC[:,i] = lambda_locreg
        # com_vec_locreg_LC[i] = norm(g - f_rec_locreg)
        # res_vec_locreg_LC[:,i] = f_rec_locreg

        # #Breaks down if not the right initial vector added
        # x0_ini = f_rec_DP
        # f_rec_locreg, lambda_locreg = LocReg_unconstrained(data_noisy, G, x0_ini, lambda_DP, ep1, ep2, ep3)
        # lam_LocReg_DP[:,i] = lambda_locreg
        # com_vec_locreg_DP[i] = norm(g - f_rec_locreg)
        # res_vec_locreg_DP[:,i] = f_rec_locreg

        # x0_ini = f_rec_GCV
        # f_rec_locreg, lambda_locreg = LocReg_unconstrained(data_noisy, G, x0_ini, reg_min, ep1, ep2, ep3)
        # lam_LocReg_LC[:,i] = lambda_locreg_LC
        # com_vec_locreg_LC[i] = norm(g - f_rec_LocReg_LC)
        # res_vec_locreg_LC[:,i] = lambda_locreg_LC


    vecs = np.asarray([com_vec_DP, com_vec_LC, com_vec_GCV, com_vec_locreg_LC, com_vec_ItoLR])

    # threshold = 3 * np.median(com_vec_GCV)  # Adjust as needed

    # Identify outliers by comparing with the threshold
    # outliers = com_vec_GCV > threshold

    # Replace outliers with NaN (Not a Number) or any other value of your choice
    # com_vec_GCV[outliers] = np.median(com_vec_GCV)
    print("sigma1",sigma1)
    print("mu1",mu1)
    print("sigma2",sigma2)
    print("mu2",mu2)
    print("SNR",SNR)
    print("Median DP:", np.median(com_vec_DP))
    print("Median LC:", np.median(com_vec_LC))
    print("Median GCV:", np.median(com_vec_GCV))
    print("Median Oracle:", np.median(com_vec_oracle))
    # print("Median Chuan LR:", np.median(com_vec_locreg_LC))
    # print("Median 2P Ito:", np.median(com_vec_ItoLR2P))
    print("Median NP Ito:", np.median(com_vec_ItoLR))
    print("Noise Realizations:", nrun)
    print("Gamma Initial:", gamma_init)

    import pickle
    hyp = {'dp_err': com_vec_DP,
            'lc_err': com_vec_LC,
            'gcv_err': com_vec_GCV,
            'oracle_err': com_vec_oracle,
            'ito_gamma': com_vec_ItoLR,}

    hyp2 = {'snr': SNR, 
            'dp_err': com_vec_DP,
            'lc_err': com_vec_LC,
            'gcv_err': com_vec_GCV,
            'oracle_err': com_vec_oracle,
            'ito_gamma': com_vec_ItoLR,
            'noise_real': nrun,
            'gamma_init': gamma_init,
            # 'gamma_new_d_f': gamma_new4}
    }

    # Calculate the medians
    medians = {key: np.median(values) for key, values in hyp.items()}

    # Rank the medians
    sorted_keys = sorted(medians, key=medians.get)
    ranks = {key: rank + 1 for rank, key in enumerate(sorted_keys)}

    # Display the results
    print("Medians:")
    for key, median in medians.items():
        print(f"{key}: {median}")
    print("\nRanks:")
    for key, rank in ranks.items():
        print(f"{key}: {rank}")

    results = {'medians': medians, 'ranks': ranks}
    try:
        with open(f'{file_path}/{string}_rankings_SNR_{SNR}.pkl', 'wb') as file:
            pickle.dump(results, file)
        print(f'Saved {string}_rankings_SNR_{SNR}.pkl')
    except Exception as e:
        print(f'Failed to save {string}_rankings_SNR_{SNR}.pkl: {e}')

    # Save hyperparameters data
    try:
        with open(f'{file_path}/{string}_hyp_data_SNR_{SNR}.pkl', 'wb') as file:
            pickle.dump(hyp2, file)
        print(f'Saved {string}_hyp_data_SNR_{SNR}.pkl')
    except Exception as e:
        print(f'Failed to save {string}_hyp_data_SNR_{SNR}.pkl: {e}')

    return vecs

date = datetime.now().strftime("%Y%m%d")


sigma1 = 3
mu1 = 40
sigma2 = 10
mu2 = 160
SNR = 200

orig_200 = run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=False)
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito_{date}_orig_200.csv', orig_200 , delimiter=',')



# sigma1 = 10
# mu1 = 40
# sigma2 = 3
# mu2 = 160
# SNR = 200

# flip_200 =run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=False)
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito_{date}_flip_200.csv', flip_200 , delimiter=',')

# sigma1 = 3
# mu1 = 40
# sigma2 = 10
# mu2 = 60
# SNR = 200

# merge_200 = run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=False)
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito_{date}_merge_200.csv', merge_200 , delimiter=',')


#Flipped L/R
# sigma1 = 3
# mu1 = 40
# sigma2 = 10
# mu2 = 160
# SNR = 50

# orig_50 =run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=False)
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito_{date}_orig_50.csv', orig_50 , delimiter=',')

# sigma1 = 3
# mu1 = 40
# sigma2 = 10
# mu2 = 160
# SNR = 1000

# orig_1000 = run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=False)
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito_{date}_orig_1000.csv', orig_1000 , delimiter=',')


# #Flipped R/L

# sigma1 = 10
# mu1 = 40
# sigma2 = 3
# mu2 = 160
# SNR = 50

# flip_50 =run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=False)
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito_{date}_flip_50.csv', flip_50 , delimiter=',')

# sigma1 = 10
# mu1 = 40
# sigma2 = 3
# mu2 = 160
# SNR = 1000

# flip_1000 =run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=False)
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito_{date}_flip_1000.csv', flip_1000 , delimiter=',')

# sigma1 = 3
# mu1 = 40
# sigma2 = 10
# mu2 = 60
# SNR = 200

# merge_200 = run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=False)
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito_{date}_merge_200.csv', merge_200 , delimiter=',')



# #Flipped mixed

# sigma1 = 3
# mu1 = 60
# sigma2 = 10
# mu2 = 110
# SNR = 50

# run_all(sigma1, mu1, sigma2, mu2, SNR)

# sigma1 = 3
# mu1 = 60
# sigma2 = 10
# mu2 = 110
# SNR = 200

# run_all(sigma1, mu1, sigma2, mu2, SNR)

# sigma1 = 3
# mu1 = 60
# sigma2 = 10
# mu2 = 110
# SNR = 1000

# run_all(sigma1, mu1, sigma2, mu2, SNR)


#Merged

# sigma1 = 3
# mu1 = 40
# sigma2 = 10
# mu2 = 60
# SNR = 50

# merge_50 = run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=False)
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito_{date}_merge_50.csv', merge_50 , delimiter=',')

# sigma1 = 3
# mu1 = 40
# sigma2 = 10
# mu2 = 60
# SNR = 1000

# merge_1000 = run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=False)
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito_{date}_merge_1000.csv', merge_1000 , delimiter=',')

# #3 Bump
# sigma1 = 3
# mu1 = 40
# sigma2 = 6
# mu2 = 100
# SNR = 1000

# thrbump_1000 = run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=True)
# np.savetxt(f'/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito_{date}_thrbump_1000.csv', thrbump_1000 , delimiter=',')






# #3 Bump
# sigma1 = 3
# mu1 = 40
# sigma2 = 6
# mu2 = 100
# SNR = 1000

# thrbump_1000 = run_all(sigma1, mu1, sigma2, mu2, SNR, threebump=True)


#TEST GCV 
# sigma1 = 3
# mu1 = 60
# sigma2 = 10
# mu2 = 120
# SNR = 50

# run_all(sigma1, mu1, sigma2, mu2, SNR)

# sigma1 = 3
# mu1 = 80
# sigma2 = 10
# mu2 = 100
# SNR = 200

# run_all(sigma1, mu1, sigma2, mu2, SNR)

# sigma1 = 3
# mu1 = 80
# sigma2 = 10
# mu2 = 100
# SNR = 1000

# run_all(sigma1, mu1, sigma2, mu2, SNR)


#Save the vectors of values as csv files

# # # np.savetxt('baart_vec_1023.csv', vecs , delimiter=',')
# # # np.savetxt('baart_vec_1121.csv', vecs , delimiter=',')
# np.savetxt('/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/Testing_Ito/Ito_LR_04_27_24.csv', vecs , delimiter=',')



###Ignore code below

# maxiter = 50
# blur_n = 50
# gamma_init = 1
# LRIto_ini_lam = 1e-3

# GravIto_ini_lam = LRIto_ini_lam
# grav_n = 1000

# n = grav_n
# t = np.linspace(-np.pi/2,np.pi/2,n**2)
# t = np.linspace(-np.pi/2,np.pi/2,n)
# # G, data_noiseless, g = blur(N = n)
# # G = G.toarray()

# G, data_noiseless, g = gravity(n, nargin=1)
# # U,s,V = csvd(G, tst = None, nargin = 1, nargout = 3)
# SNR = 200
# SD_noise = 1 / SNR
# noise = np.random.normal(0, SD_noise, size=data_noiseless.shape)
# noise = 5e-2
# data_noisy = data_noiseless + noise
# # blur_rec, fin_lam, c_array, lam_arr_fin, sol_arr_fin = blur_ito(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter)
# blur_rec, fin_lam, c_array, lam_arr_fin, sol_arr_fin = grav_ito(data_noisy, G, GravIto_ini_lam, gamma_init, maxiter)

# has_negative_values = np.any(blur_rec < 0)

# if has_negative_values:
#     print("blur_rec has negative values.")
# else:
#     print("blur_rec does not have any negative values.")

# blur_rec[blur_rec < 0] = 0
# err_ItoBlur = np.linalg.norm(g - blur_rec, 2)

# fig, axs = plt.subplots(2, 2, figsize=(6, 6))
# # plt.subplots_adjust(wspace=0.3)

# # Plotting the first subplot
# # plt.subplot(1, 3, 1)
# ymax = np.max(g) * 1.15
# axs[0, 0].plot(t, g, color = "black",  label = "Ground Truth")
# axs[0, 0].plot(t, blur_rec, color = "purple",  label = "Ito Gravity")
# axs[0, 0].set_xlabel('T', fontsize=20, fontweight='bold')
# axs[0, 0].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
# axs[0, 0].legend(fontsize=10, loc='best')
# axs[0, 0].set_ylim(0, ymax)

# # Plotting the second subplot
# # plt.subplot(1, 3, 2)
# # axs[0, 1].plot(t, G @ g, linewidth=3, color='black', label='Ground Truth')
# # # plt.plot(TE, A @ f_rec_LocReg, linestyle=':', linewidth=3, color='magenta', label='LocReg')
# # axs[0, 1].plot(t, G @ blur_rec, color = "purple",  label = "Ito Blur")

# # # plt.plot(TE, A @ f_rec_LC, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
# # # plt.plot(TE, A @ f_rec, linestyle='-.', linewidth=3, color='red', label='SpanReg')
# # axs[0, 1].legend(fontsize=10, loc='best')
# # axs[0, 1].set_xlabel('t', fontsize=20, fontweight='bold')
# # axs[0, 1].set_ylabel('Intensity', fontsize=20, fontweight='bold')

# # plt.subplot(1, 3, 3)
# # axs[1, 0].semilogy(t, fin_lam * np.ones(len(t)), linewidth=3, color='purple', label='Ito Blur')
# # axs[1, 0].legend(fontsize=10, loc='best')
# # axs[1, 0].set_xlabel('T2', fontsize=20, fontweight='bold')
# # axs[1, 0].set_ylabel('Lambda', fontsize=20, fontweight='bold')

# table_ax = axs[1, 1]
# table_ax.axis('off')

# # Define the data for the table
# data = [
#     ["Initial Eta for Ito Gravity", GravIto_ini_lam],
#     ["Final Eta for Ito Gravity", fin_lam],
#     ["error Ito Gravity", err_ItoBlur]
#     # ["Initial Lambdas for Ito Loc", LR_ini_lam],
#     # ["Final Lambdas for Ito Loc", LR_Ito_lams]
# ]

# # Create the table
# table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.scale(1.2, 1.2)

# plt.tight_layout()
# string = "comparison"
# file_path = create_result_folder(string)
# plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve"))
# print(f"Saving comparison plot is complete")
# plt.show()
# plt.close()

# #Ito_LocReg
# lam_ini = lambda_LC
# gamma_init = 10
# param_num = 2
# maxiter = 50
# B_mats = B_matrices

# LR_ini_lam = lam_ini * np.ones(nT2)

# LRIto_ini_lam = 1e-3
# best_f_rec, fin_etas = Ito_LocReg(data_noisy, G, LRIto_ini_lam, gamma_init, param_num, B_mats, maxiter)
# print("Completed 2P Ito")

# print("Starting NP Ito")
# LR_mod_rec, fin_lam, c_array = LocReg_Ito_mod(data_noisy, G, LRIto_ini_lam, gamma_init, maxiter)

#

# # print("c array shape:", c_array.shape)
# # print("c array shape:", c_array)

# from tqdm import tqdm
# from Utilities_functions.LocReg import LocReg
# from Utilities_functions.LocReg_unconstrainedB import LocReg_unconstrainedB
# from Utilities_functions.discrep_L2 import discrep_L2
# from Utilities_functions.GCV_NNLS import GCV_NNLS
# from Utilities_functions.Lcurve import Lcurve
# from Utilities_functions.Multi_Reg_Gaussian_Sum1 import Multi_Reg_Gaussian_Sum1
# from Utilities_functions.LocReg_NEW_NNLS import LocReg_NEW_NNLS
# from numpy.linalg import norm
# from regu.gravity import gravity


# n = 1000
# nT2 = n
# T2 = np.linspace(-np.pi/2,np.pi/2,n)
# TE = T2

# G, data_noiseless, g = gravity(n, nargin=1)

# print("lam_arr_fin[0] shape:", lam_arr_fin[0].shape)
# print("lam_arr_fin[0]:", lam_arr_fin[0])