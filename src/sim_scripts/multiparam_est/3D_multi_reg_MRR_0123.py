#This script is an extension of the 2d two parameter grid search problem for the MRR T2 problem
# proposed in "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Simulations/2D_multi_reg_MRR_0116.py"

#This script extends the grid search to a 3 dimensional grid search problem
#Our diagonal matrices B1, B2, B3 represent the index positions of our T2 axis, where ones represent where along the T2 axis 
#we want our regularization to be applied and zeros represent where we do not apply regularization.

#In this case we split our T2 axis into three equal parts and apply regularization accordingly. We seek to obtain the optimal alpha1 value for 
#the positions in the T2 axis covered by the B1 matrix. We seek to obtain the optimal alpha2 value for the positions in the T2 axis
#covered by the B2 matrix. We seek to obtain the optimal alpha3 value for the positions in the T2 axis
#covered by the B3 matrix

#The goal is to find the optimal alpha1,  alpha2, and alpha3 regularization parameters assuming we know the ground truth. 

#We compare with One Parameter (OP) Grid Search representing the best conventional tikhonov regularization parameter.

#Packages
from tools.trips_py.new_gcv_pasha import generalized_crossvalidation
from pylops import LinearOperator, Identity
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
import math
from mpl_toolkits.mplot3d import Axes3D
import h5py
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import sympy as sp
from autograd import grad, elementwise_grad, jacobian, hessian, holomorphic_grad
from mpl_toolkits import mplot3d

########################################################################
#Define functions and classes
class three_dim_MRR_grid_search():
    def __init__(self, T2, TE, SNR, nalpha1, nalpha2, nalpha3, mu1, mu2, mu3, sigma1, sigma2, sigma3, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, alp_3_lb,alp_3_ub):
        self.T2 = T2
        self.TE = TE

        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3

        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3

        self.alp_1_lb = alp_1_lb
        self.alp_1_ub = alp_1_ub
        self.alp_2_lb = alp_2_lb
        self.alp_2_ub = alp_2_ub
        self.alp_3_lb = alp_3_lb
        self.alp_3_ub = alp_3_ub

        self.nT2 = len(T2)
        self.nTE = len(TE)
        self.SNR = SNR
        self.nalpha1 = nalpha1
        self.nalpha2 = nalpha2
        self.nalpha3 = nalpha3

        self.three_param_init_params()
        self.one_param_init_params()
        self.one_param_conv_methods_init_params()

    def three_param_init_params(self):
        self.G = None
        self.g = None
        self.noise = None
        self.data_noisy = None
        self.B_1 = None
        self.B_2 = None
        self.B_3= None

        self.f_rec_TP_grid = None
        self.com_rec_TP_grid = None
        self.TP_x_lc_vec = None
        self.TP_rhos = None
        self.TP_min_alpha1 = None
        self.TP_min_alpha2 = None
        self.TP_min_alpha3 = None
        self.TP_min_alpha1_ind = None
        self.TP_min_alpha2_ind = None
        self.TP_min_alpha3_ind = None

        self.TP_fig = None

    def one_param_init_params(self):
        #op Parameters
        self.f_rec_OP_grid = None
        self.com_rec_OP_grid = None
        self.OP_rhos = None
        self.OP_x_lc_vec = None
        self.OP_min_alpha1 = None
        self.OP_min_alpha1_ind = None

    def one_param_conv_methods_init_params(self):
        #Decompose G
        self.U = None
        self.s = None
        self.V = None
        #DP
        self.lam_DP_OP = None
        self.f_rec_DP_OP = None
        self.com_rec_DP = None
        #LC
        self.lam_LC_OP = None
        self.f_rec_LC_OP = None
        self.com_rec_LC = None
        #GCV
        self.lam_GCV_OP = None
        self.f_rec_GCV_OP = None
        self.com_rec_GCV = None

    def get_G(self):
        G_mat = np.zeros((self.nTE, self.nT2))  
        for i in range(self.nTE):
            for j in range(self.nT2):
                G_mat[i,j] = np.exp(-self.TE[i]/self.T2[j])
        self.G = G_mat
    
    def get_g(self):
        g1 = (1 / (np.sqrt(2 * np.pi) * self.sigma1)) * np.exp(-((self.T2 - self.mu1) ** 2) / (2 * self.sigma1 ** 2))
        g2 = (1 / (np.sqrt(2 * np.pi) * self.sigma2)) * np.exp(-((self.T2 - self.mu2) ** 2) / (2 * self.sigma2 ** 2))
        g3 = (1 / (np.sqrt(2 * np.pi) * self.sigma3)) * np.exp(-((self.T2 - self.mu3) ** 2) / (2 * self.sigma3 ** 2))
        self.g  = (g1 + g2 + g3)/(3)
    
    def decompose_G(self):
        U, s, V = csvd(self.G, tst = None, nargin = 1, nargout = 3)
        self.U = U
        self.s = s
        self.V = V

    def get_noisy_data(self):
        data_noiseless = np.dot(self.G, self.g)
        SD_noise= 1/self.SNR*max(abs(data_noiseless))
        self.noise = np.random.normal(0,SD_noise, data_noiseless.shape)
        self.data_noisy = data_noiseless + self.noise

    def get_diag_matrices(self):
        div_nT2 = int(self.nT2/3)
        div_mat = np.eye(div_nT2)
        self.B_1 = np.zeros((self.nT2, self.nT2))
        self.B_2 = np.zeros((self.nT2, self.nT2))
        self.B_3 = np.zeros((self.nT2, self.nT2))

        self.B_1[:div_nT2, :div_nT2] = div_mat
        self.B_2[div_nT2:self.nT2-div_nT2, div_nT2:self.nT2-div_nT2] = div_mat
        self.B_3[self.nT2-div_nT2:, self.nT2-div_nT2:] = div_mat

    ###### Objective Functions

    def minimize_TP_objective(self, Alpha_vec, Alpha_vec2, Alpha_vec3):
        self.TP_x_lc_vec = np.zeros((self.nT2, len(Alpha_vec), len(Alpha_vec2), len(Alpha_vec3)))
        self.TP_rhos = np.zeros((len(Alpha_vec), len(Alpha_vec2), len(Alpha_vec3)))

        for j, k, l in (product(range(len(Alpha_vec)), range(len(Alpha_vec2)), range(len(Alpha_vec3)))):
            x = cp.Variable(self.nT2)
            # objective = cp.Minimize(cp.norm(G.T @ G @ x - G.T @ data_noisy, 2) +
            #                         Alpha_vec[j]**2 * cp.norm(B_1 @ x, 2)**2 +
            #                         Alpha_vec2[k]**2 * cp.norm(B_2 @ x, 2)**2)
            objective = cp.Minimize(cp.norm(self.G @ x - self.data_noisy, 2)**2 +
                                    Alpha_vec[j]**2 * cp.norm(self.B_1 @ x, 2)**2 +
                                    Alpha_vec2[k]**2 * cp.norm(self.B_2 @ x, 2)**2 +
                                    Alpha_vec3[l]**2 * cp.norm(self.B_3 @ x, 2)**2)
            constraints = [x >= 0]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK, verbose=False)
            if x.value is not None:
                self.TP_x_lc_vec[:, j, k, l] = x.value.flatten()
                self.TP_rhos[j, k, l] = norm(self.TP_x_lc_vec[:, j, k, l] - self.g, 2)**2
            else:
                print("x.value == None")

    def minimize_OP_objective(self, Alpha_vec):
        self.OP_x_lc_vec = np.zeros((self.nT2, len(Alpha_vec)))
        self.OP_rhos = np.zeros((len(Alpha_vec)))
        for j in (range(len(Alpha_vec))):
            x = cp.Variable(self.nT2)
            objective = cp.Minimize(cp.norm(self.G @ x - self.data_noisy, 2)**2 +
                                    Alpha_vec[j]**2 * cp.norm(self.B_1 @ x, 2)**2 +
                                    Alpha_vec[j]**2 * cp.norm(self.B_2 @ x, 2)**2 +
                                    Alpha_vec[j]**2 * cp.norm(self.B_3 @ x, 2)**2)
            constraints = [x >= 0]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.MOSEK, verbose=False)
            if x.value is not None:
                self.OP_x_lc_vec[:, j] = x.value.flatten()
                self.OP_rhos[j] = norm(self.OP_x_lc_vec[:,j] - self.g, 2)**2
            else:
                print("x.value == None")

    def minimize_DP(self, SNR):
        delta = norm(self.noise)*1.05
        x_delta,lambda_OP = discrep(self.U,self.s,self.V,self.data_noisy,delta, x_0= None, nargin = 5)
        self.lam_DP_OP = lambda_OP
        Lambda = np.ones(len(self.T2))* lambda_OP
        f_rec_DP_OP,_ = discrep_L2(self.data_noisy, self.G, SNR, Lambda)
        self.f_rec_DP_OP = f_rec_DP_OP.flatten()
        # Lambda = np.ones(len(self.T2))* self.OP_min_alpha1
        # f_rec,self.lam_DP_OP = discrep_L2(self.data_noisy, self.G, SNR, Lambda)
        # self.f_rec_DP_OP = f_rec.flatten()

    def minimize_LC(self):
        reg_corner,rho,eta,_ = l_curve(self.U,self.s,self.data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
        self.lam_LC_OP = reg_corner
        Lambda =np.ones(len(self.T2))* reg_corner
        Lambda = Lambda.reshape(-1,1)
        f_rec_LC_OP,_ = Lcurve(self.data_noisy, self.G, Lambda)
        self.f_rec_LC_OP = f_rec_LC_OP.flatten()
        # Lambda =np.ones(len(self.T2))* self.OP_min_alpha1
        # Lambda = Lambda.reshape(-1,1)
        # f_rec, lam = Lcurve(self.data_noisy, self.G, Lambda)
        # self.f_rec_LC_OP = f_rec.flatten()
        # self.lam_LC_OP = lam[0]

    def minimize_GCV(self):
        reg_min,_,_ = gcv(self.U,self.s,self.data_noisy, method = 'Tikh', nargin = 3, nargout = 3)
        self.lam_GCV_OP = reg_min
        Lambda = np.ones(len(self.T2))* reg_min
        f_rec,_ = GCV_NNLS(self.data_noisy, self.G, Lambda)
        self.f_rec_GCV_OP = f_rec[:,0]
        # f_rec_GCV_OP,_,_ = tikhonov(self.U,self.s,self.V,self.data_noisy,reg_min, nargin=5, nargout=1)
        # Lambda = np.ones(len(self.T2))* self.OP_min_alpha1
        # f_rec, lam = GCV_NNLS(self.data_noisy, self.G, Lambda)
        # self.f_rec_GCV_OP = f_rec[:,0]
        # self.lam_GCV_OP = lam[0]

    ###### Plotting Grid Search Surfaces

    def plot_TP_surface(self, Alpha_vec, Alpha_vec2, rhos, iter_num, which_xalpha, which_yalpha):

        # Create meshgrid
        plot1, plot2 = np.meshgrid(Alpha_vec, Alpha_vec2)
        
        if which_xalpha ==1 and which_yalpha == 2:
            TP_log_err_norm = np.log10(rhos[:,:,:].sum(axis=2))
        elif which_xalpha ==1 and which_yalpha ==3:
            TP_log_err_norm = np.log10(rhos[:,:,:].sum(axis=1))
        elif which_xalpha ==2 and which_yalpha ==3:
            TP_log_err_norm = np.log10(rhos[:,:,:].sum(axis=0))

        # Find indices of minimum value
        min_index = np.unravel_index(np.argmin(TP_log_err_norm), TP_log_err_norm.shape)

        min_x = Alpha_vec[min_index[0]]
        min_y = Alpha_vec2[min_index[1]]
        min_z = np.min(TP_log_err_norm)

        # Create a 3D surface plot
        fig = go.Figure(data=[go.Surface(x=Alpha_vec, y=Alpha_vec2, z=TP_log_err_norm.T)])

        # Add a scatter plot for the minimum point
        fig.add_trace(go.Scatter3d(
            x=np.array([min_x]),
            y=np.array([min_y]),
            z=np.array([min_z]),
            mode='markers',
            marker=dict(
                size=5,
                color='gold',
                symbol='circle'
            ),
            name='Minimum Point'
        ))

        # Configure layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(type='log', title=f'Alpha_{which_xalpha} Values 10^{self.alp_1_lb} to 10^{self.alp_1_ub}'),
                yaxis=dict(type='log', title=f'Alpha_{which_yalpha} Values 10^{self.alp_2_lb} to 10^{self.alp_2_ub}'),
                zaxis=dict(title='log(norm(p_a1,a2 - p_true)**2)')  # Assuming z-axis is log scale
            ),
            title=f"Surface of Grid Search on the Alpha_{which_xalpha} and Alpha_{which_yalpha} Plane" + f"\n for NR {iter_num}",
            annotations=[
                dict(
                    text=f"Optimal Alpha_{which_xalpha}  value: {min_x}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1.08
                ),
                dict(
                    text=f"Optimal Alpha_{which_yalpha}  value: {min_y}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1.05
                )
            ]
        )
        self.TP_fig = fig

    def plot_OP_surface(self, Alpha_vec, rhos, iter_num):
        OP_log_err_norm = np.log10(rhos)
        min_OP_len = np.min(OP_log_err_norm)
        min_OP_len_ind = np.argmin(OP_log_err_norm)
        corresponding_alpha = Alpha_vec[min_OP_len_ind]
        g_fig = plt.figure()
        plt.plot(Alpha_vec,OP_log_err_norm)
        plt.scatter(corresponding_alpha, min_OP_len, color='red', marker='o', label='Minimum Value')
        plt.xlabel('Alpha Value')
        plt.ylabel('log(norm(p_a1 - p_true)**2)')
        plt.title(f"Plot for One Parameter Grid Search for NR {iter_num}")
        plt.suptitle(f"Optimal Alpha Value: {corresponding_alpha}")
        plt.legend()
        plt.savefig("".join([file_path, f"One_Parameter_Surface_Plot_{date}_NR_{iter_num}_mu1_{self.mu1}_mu2_{self.mu2}_mu3_{self.mu3}_sigma1_{self.sigma1}_sigma2_{self.sigma2}_sigma3_{self.sigma3}"]))
        print(f"Saved One Parameter Surface Plot for NR {iter_num}")
        plt.close()

    ##### Get Optimal Alphas

    def get_opt_TP_alphas(self, Alpha_vec, Alpha_vec2, Alpha_vec3, TP_rhos):
        TP_log_err_norm = np.log10(TP_rhos)
        # min_indices = np.where(Z == np.min(Z))
        min_index = np.unravel_index(np.argmin(TP_log_err_norm), TP_log_err_norm.shape)
        
        min_x = Alpha_vec[min_index[0]]
        min_y = Alpha_vec2[min_index[1]]
        min_z = Alpha_vec3[min_index[2]]
        min_gamma = np.min(TP_log_err_norm)

        self.TP_min_alpha1 = min_x
        self.TP_min_alpha1_ind = min_index[0]
        self.TP_min_alpha2 = min_y
        self.TP_min_alpha2_ind = min_index[1]
        self.TP_min_alpha3 = min_z
        self.TP_min_alpha3_ind = min_index[2]

    def get_opt_OP_alphas(self, Alpha_vec):
        OP_log_err_norm = np.log10(self.OP_rhos)
        min_index = np.unravel_index(np.argmin(OP_log_err_norm), OP_log_err_norm.shape)

        min_x = Alpha_vec[min_index[0]]
        min_z = np.min(OP_log_err_norm)

        self.OP_min_alpha1 = min_x
        self.OP_min_alpha1_ind = min_index[0]

    ##### Get Reconstructions Using Optimal Alphas
    def get_f_rec_TP_grid(self):
        alpha1 = self.TP_min_alpha1
        alpha2 = self.TP_min_alpha2
        alpha3 = self.TP_min_alpha3
        self.f_rec_TP_grid = self.TP_x_lc_vec[:, self.TP_min_alpha1_ind, self.TP_min_alpha2_ind, self.TP_min_alpha3_ind]
        
    def get_f_rec_OP_grid(self):
        self.f_rec_OP_grid = self.OP_x_lc_vec[:,self.OP_min_alpha1_ind]

    ##### Get L2 Errors of the TP and OP Reconstructions Against Ground Truth
    def get_error(self):
        self.com_rec_OP_grid = norm(self.g - self.f_rec_OP_grid)
        self.com_rec_TP_grid = norm(self.g - self.f_rec_TP_grid.flatten())
        self.com_rec_DP = norm(self.g - self.f_rec_DP_OP)
        self.com_rec_LC = norm(self.g - self.f_rec_LC_OP)
        self.com_rec_GCV = norm(self.g - self.f_rec_GCV_OP)

    # def plot_gt_and_comparison(self, file_path, date, compare, num_real):
    #     if compare != True:
    #         g_fig = plt.figure()
    #         plt.plot(self.g)
    #         plt.xlabel("T2")
    #         plt.ylabel("Amplitude")
    #         plt.title("MRR Problem Ground Truth")
    #         plt.savefig("".join([file_path, f"ground_truth_{date}_mu1_{self.mu1}_mu2_{self.mu2}_mu3_{self.mu3}_sigma1_{self.sigma1}_sigma2_{self.sigma2}_sigma3_{self.sigma3}"]))
    #         print(f"Saving ground truth plot is complete")
    #         np.save("".join([file_path, f"ground_truth_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_{self.nalpha3}_alpha3_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}_alpha3log_{self.alp_3_lb}_{self.alp_3_ub}"]),self.g)
    #         print(f"Saving ground truth data  is complete") 
    #     else:
    #         fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    #         # Subplot 1: Reconstructions
    #         axs[0].plot(self.g, label="Ground Truth")
    #         axs[0].plot(self.f_rec_OP_grid, label="Conventional One Parameter Grid Search")
    #         axs[0].plot(self.f_rec_TP_grid, label="Three Parameter Grid Search")
    #         axs[0].set_xlabel("T2", fontsize = 10)
    #         axs[0].set_ylabel("Amplitude", fontsize = 10)
    #         axs[0].set_title(f"MRR Problem NR{num_real + 1}")
    #         axs[0].legend(["Ground Truth", "Conventional One Parameter Grid Search", "Three Parameter Grid Search"])
    #         fig.suptitle(f"error_OP = {round(self.com_rec_OP_grid.item(), 5)}; error_ThP = {round(self.com_rec_TP_grid.item(), 5)}" + f"\n ThP_alpha_1 = {round(self.TP_min_alpha1.item(), 5)}; ThP_alpha_2 = {round(self.TP_min_alpha2.item(), 5)} ThP_alpha_3 = {round(self.TP_min_alpha3.item(), 5)} "+ f"\n OP_alpha = {self.OP_min_alpha1}", fontsize=8)

    #         # Subplot 2: Alpha Plots
    #         line = self.TP_min_alpha1.item() * np.diag(self.B_1) + self.TP_min_alpha2.item() * np.diag(self.B_2) + self.TP_min_alpha3.item() * np.diag(self.B_3)
    #         axs[1].semilogy(self.T2, self.OP_min_alpha1.item() * np.ones(len(self.T2)), linestyle=':', linewidth=3, color='orange', label='Conventional One Parameter Grid Search')
    #         axs[1].semilogy(self.T2, line, linewidth = 3, color = "green", label = "Three Parameter Grid Search Lambdas")
    #         handles, labels = axs[1].get_legend_handles_labels()
    #         dict_of_labels = dict(zip(labels, handles))
    #         axs[1].legend(dict_of_labels.values(), dict_of_labels.keys(), fontsize=10, loc='best')
    #         axs[1].set_xlabel('T2', fontsize=10)
    #         axs[1].set_ylabel('Alpha', fontsize=10)
    #         # Adjust layout for better spacing
    #         plt.tight_layout()

    def plot_gt_and_comparison(self, file_path, date, compare, num_real):
        if compare != True:
            g_fig = plt.figure()
            plt.plot(self.g)
            plt.xlabel("T2")
            plt.ylabel("Amplitude")
            plt.title("MRR Problem Ground Truth", fontsize = 15)
            plt.savefig(os.path.join(file_path, f"ground_truth_{date}_mu1_{self.mu1}_mu2_{self.mu2}_sigma1_{self.sigma1}_sigma2_{self.sigma2}"))
            print(f"Saving ground truth plot is complete")
            np.save(os.path.join(file_path, f"ground_truth_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"),self.g)
            print(f"Saving ground truth data  is complete") 
        else:
            fig, axs = plt.subplots(1, 3, figsize=(24, 10))
            # Subplot 1: Reconstructions
            axs[0].plot(self.g, label="Ground Truth", linewidth=6.0)
            axs[0].plot(self.f_rec_OP_grid, label="Conventional One Parameter Grid Search", linewidth=3.0)
            axs[0].plot(self.f_rec_TP_grid, label="Three Parameter Grid_Search", linewidth=3.0)
            # axs[0].plot(self.f_rec_DP_OP, label = "DP")
            # axs[0].plot(self.f_rec_LC_OP, label = "LC")
            # axs[0].plot(self.f_rec_GCV_OP, label = "GCV")
            axs[0].set_xlabel("T2", fontsize = 20, fontweight='bold')
            axs[0].set_ylabel("Amplitude", fontsize = 20, fontweight='bold')
            axs[0].set_title(f"MRR Problem NR{num_real + 1}")
            axs[0].legend(["Ground Truth", "Conventional One Parameter Grid Search", "Three Parameter Grid Search", "DP", "LC", "GCV"])
            OP_errors = [self.com_rec_OP_grid, self.com_rec_DP, self.com_rec_LC, self.com_rec_GCV]
            algo_names = ["Oracle", "DP", "LC", "GCV"]
            name_ind = np.argwhere(OP_errors == np.min(OP_errors))[0][0]
            minimum_algo = algo_names[name_ind]
            minimum_error = np.min(OP_errors)
            lamDP = self.lam_DP_OP[0]
            err_ratio = self.com_rec_OP_grid/self.com_rec_TP_grid
            fig.suptitle(r"$\bf{L2\ Norm\ Error\ Values:}$" + f"\nerror_OP_GS = {round(self.com_rec_OP_grid.item(), 4)}; error_TP_GS = {round(self.com_rec_TP_grid.item(),4)}; error_DP = {round(self.com_rec_DP.item(), 4)} ; error_LC = {round(self.com_rec_LC.item(),4)} ; error_GCV = {round(self.com_rec_GCV.item(),4)}" 
                         + f"\n The lowest error for the One Parameter Algorithms is {minimum_algo} with error of {round(minimum_error.item(),4)}\n" + f"\n OP to TP Error Ratio = {round(err_ratio.item(),4)}\n" + f"\n SNR = {self.SNR} \n"
                         + r"$\bf{Lambda/Alpha\ Values}$" + f"\nTP_alpha_1 = {round(self.TP_min_alpha1.item(),4)}; TP_alpha_2 = {round(self.TP_min_alpha2.item(),4)}; TP_alpha_3 = {round(self.TP_min_alpha3.item(),4)}" + f"\n OP_alpha = {round(self.OP_min_alpha1.item(),4)}" 
                         + f"\n DP_lam = {round(lamDP.item(),4)}; LC_lam = {round(self.lam_LC_OP.item(),4)} ; GCV_lam = {round(self.lam_GCV_OP.item(),4)}", fontsize=12)

            # Subplot 2: Alpha Plots
            TP_Alphas = self.TP_min_alpha1.item() * np.diag(self.B_1) + self.TP_min_alpha2.item() * np.diag(self.B_2) + self.TP_min_alpha3.item() * np.diag(self.B_3)
            # TP_Alphas = self.TP_min_alpha1.item() * np.diag(self.B_1) + self.TP_min_alpha2.item() * np.diag(self.B_2) 

            #Print out the OP Alpha
            axs[1].semilogy(self.T2, self.OP_min_alpha1.item() * np.ones(len(self.T2)), linewidth=3, color='orange', label='Conventional One Parameter Grid Search')
            
            #Print out the TP Alpha
            # print(self.f_rec_DP_OP)
            # print(self.f_rec_DP_OP.shape)
            # print(self.f_rec_LC_OP)
            # print(self.f_rec_LC_OP.shape)
            # print(self.f_rec_GCV_OP)
            # print(self.f_rec_GCV_OP.shape)
            axs[1].semilogy(self.T2, TP_Alphas, linewidth = 3, color = "green", label = "Three Parameter Grid Search")
            axs[1].semilogy(self.T2, self.lam_DP_OP * np.ones(len(self.T2)), linewidth = 3, color = "red", label = "DP NNLS")
            axs[1].semilogy(self.T2, self.lam_LC_OP * np.ones(len(self.T2)), linewidth = 3, color = "purple", label = "LCurve NNLS")
            axs[1].semilogy(self.T2, self.lam_GCV_OP * np.ones(len(self.T2)), linewidth = 3, color = "gray", label = "GCV NNLS")

            handles, labels = axs[1].get_legend_handles_labels()
            dict_of_labels = dict(zip(labels, handles))
            axs[1].legend(dict_of_labels.values(), dict_of_labels.keys(), fontsize=10, loc='best')
            axs[1].set_xlabel('T2', fontsize=20, fontweight='bold')
            axs[1].set_ylabel('Alpha', fontsize=20, fontweight='bold')

            #Subplot 3: Decay Curves
            axs[2].plot(self.TE, self.G @ self.g, linewidth = 3, color = "blue", label = "Ground Truth")
            axs[2].plot(self.TE, self.G @ self.f_rec_TP_grid, linewidth=3, color='green', label='Three Parameter Grid Search Oracle')
            axs[2].plot(self.TE, self.G @ self.f_rec_OP_grid, linewidth=3, color= "orange", label='One Parameter Grid Search Oracle')
            axs[2].scatter(self.TE, self.data_noisy, s=30, color="black", label='Data Points')

            # axs[2].plot(self.TE, self.G @ self.f_rec_DP_OP, linewidth=3, color='brown', label='DP')
            # axs[2].plot(self.TE, self.G @ self.f_rec_LC_OP, linestyle='-.', linewidth=3, color='cyan', label='L-curve')
            # axs[2].plot(self.TE, self.G @ self.f_rec_GCV_OP, linestyle='--', linewidth=3, color='blue', label='GCV')
            axs[2].legend(fontsize=10, loc='best')
            axs[2].set_xlabel('TE', fontsize=20, fontweight='bold')
            axs[2].set_ylabel('Intensity', fontsize=20, fontweight='bold')
            plt.tight_layout()
            # Save the figure
            plt.savefig(os.path.join(file_path, f"compare_gt_convtikhreg_multireg_{date}_NR_{num_real + 1}_mu1_{self.mu1}_mu2_{self.mu2}_mu3_{self.mu3}_sigma1_{self.sigma1}_sigma2_{self.sigma2}_sigma3_{self.sigma3}"))
            print("Saving comparison plot is complete")

    #     self.fig = fig
    def run_all(self, Alpha_vec, Alpha_vec2, Alpha_vec3, num_real, file_path, date):
        opt_TP_alphas = []
        opt_TP_alphas_ind = []

        opt_OP_alpha =[]
        opt_OP_alpha_ind = []

        f_rec_TP_grids = []
        f_rec_OP_grids = []

        errors_TP_grid = []
        errors_OP_grid = []

        for i in tqdm(range(num_real)):
            file_path = create_result_folder()

            #Construct and Save the Noisy Data
            self.get_noisy_data()
            np.save(os.path.join(file_path, f"noisy_data_for_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_{self.nalpha3}_alpha3_discretization_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}_alpha3log_{self.alp_3_lb}_{self.alp_3_ub}"), self.data_noisy)
            print(f"Saved Noisy Data for NR {i+1}")

            #Minimize the Objective Functions to Find the Surface and Save

            #######Minimize and Save the TP Objective
            self.minimize_TP_objective(Alpha_vec, Alpha_vec2, Alpha_vec3)
            print(f"Minimized the Two Parameter Objective Function for NR {i+1}")
            np.save(os.path.join(file_path, f"TP_rhos_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_{self.nalpha3}_alpha3_discretization_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}_alpha3log_{self.alp_3_lb}_{self.alp_3_ub}"),self.TP_rhos)
            print(f"Saved TP_rhos for NR {i+1}")
            np.save(os.path.join(file_path, f"TP_x_lc_vec_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_{self.nalpha3}_alpha3_discretization_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}_alpha3log_{self.alp_3_lb}_{self.alp_3_ub}"),self.TP_x_lc_vec)
            print(f"Saved TP_x_lc_vec for NR {i+1}")

            #######Minimize and Save the OP Objective 
            self.minimize_OP_objective(Alpha_vec)
            print(f"Minimized the One Parameter Objective Function for NR {i+1}")
            np.save(os.path.join(file_path, f"OP_rhos_NR{i+1}_{date}_{self.nalpha1}_alpha1_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}"),self.OP_rhos)
            print(f"Saved OP_rhos for NR {i+1}")
            np.save(os.path.join(file_path, f"OP_x_lc_vec_NR{i+1}_{date}_{self.nalpha1}_alpha1_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}"),self.OP_x_lc_vec)
            print(f"Saved OP_x_lc_vec for NR {i+1}")

            #Plot the Surfaces and Save

            ######Plot the TP Surface and Save
            self.plot_TP_surface(Alpha_vec, Alpha_vec2,self.TP_rhos, i+1, 1, 2)
            print(f"Plotted Two Parameter Surface Projection onto Alpha 1 and Alpha 2 Plane for NR {i+1}")
            pio.write_html(self.TP_fig, file= os.path.join(file_path, f"TP_surface_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_{self.nalpha3}_discretization_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}.html"))
            print(f"Saved Two Parameter Surface Plot for NR {i+1}")
            self.plot_TP_surface(Alpha_vec, Alpha_vec2,self.TP_rhos, i+1, 1, 3)
            print(f"Plotted Two Parameter Surface Projection onto Alpha 1 and Alpha 3 Plane for NR {i+1}")
            pio.write_html(self.TP_fig, file= os.path.join(file_path, f"TP_surface_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha3}_alpha3_discretization_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha3log_{self.alp_3_lb}_{self.alp_3_ub}.html"))
            print(f"Saved Two Parameter Surface Plot for NR {i+1}")
            self.plot_TP_surface(Alpha_vec, Alpha_vec2,self.TP_rhos, i+1, 2, 3)
            print(f"Plotted Two Parameter Surface Projection onto Alpha 2 and Alpha 3 Plane for NR {i+1}")
            pio.write_html(self.TP_fig, file= os.path.join(file_path, f"TP_surface_NR{i+1}_{date}_{self.nalpha2}_alpha2_{self.nalpha3}_alpha3_discretization_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}_alpha3log_{self.alp_3_lb}_{self.alp_3_ub}.html"))
            print(f"Saved Two Parameter Surface Plot for NR {i+1}")


            #######Plot the OP Surface and Save
            self.plot_OP_surface(Alpha_vec,self.OP_rhos, i+1)
            print(f"Saved One Parameter Surface Plot for NR {i+1}")

            #Get the Optimal Alpha Values and Save

            #######Retrieve Optimal TP Alphas

            self.get_opt_TP_alphas(Alpha_vec, Alpha_vec2, Alpha_vec3, self.TP_rhos)
            print(f"Retrieved Optimal Two Parameter Alphas for NR {i+1}")
            opt_TP_alphas.append((self.TP_min_alpha1, self.TP_min_alpha2,self.TP_min_alpha3))
            opt_TP_alphas_ind.append((self.TP_min_alpha1_ind, self.TP_min_alpha2_ind, self.TP_min_alpha3_ind))

            #######Retrieve Optimal OP Alphas
            self.get_opt_OP_alphas(Alpha_vec)
            print(f"Retrieved Optimal One Parameter Alphas for NR {i+1}")
            opt_OP_alpha.append(self.OP_min_alpha1)
            opt_OP_alpha_ind.append(self.OP_min_alpha1_ind)

            self.minimize_DP(SNR)
            print(f"Calculated Conventional DP solution for NR {i+1}")
            self.minimize_GCV()
            print(f"Calculated Conventional GCV solution for NR {i+1}")
            self.minimize_LC()
            print(f"Calculated Conventional LC solution for NR {i+1}")

            #Get the Optimal Reconstructions and Save

            #######Retrieve Optimal TP Reconstructions
            self.get_f_rec_TP_grid()
            print(f"Retrieved Best Two Parameter Reconstruction for NR {i+1}")
            f_rec_TP_grids.append(self.f_rec_TP_grid)

            #######Retrieve Optimal OP Reconstructions
            self.get_f_rec_OP_grid()
            print(f"Retrieved Best One Parameter Reconstruction for NR {i+1}")
            f_rec_OP_grids.append(self.f_rec_OP_grid)

            #Get Reconstructions Errors of OP and TP Grid Search against the Ground Truth
            self.get_error()
            print(f"Calculated the Reconstruction Errors for NR {i+1}")
            errors_OP_grid.append(self.com_rec_OP_grid)
            errors_TP_grid.append(self.com_rec_TP_grid)

            #Plot Comparison of Reconstructions with GT
            compare_val = True
            self.plot_gt_and_comparison(file_path, date, compare_val, i)
            print(f"Plotted the Comparison Plots for NR {i+1}")

        #Plot Comparison of Reconstructions with GT
        compare_val = False
        self.plot_gt_and_comparison(file_path, date, compare_val, num_real)
        print(f"Plotted the Ground Truth")

        #Save the TP optimal alphas and their indices
        np.save(os.path.join(file_path, f"opt_TP_alphas_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_{self.nalpha3}_alpha3_discretization_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}_alpha3log_{self.alp_3_lb}_{self.alp_3_ub}"),opt_TP_alphas)
        print(f"Saved opt_TP_alphas") 
        np.save(os.path.join(file_path, f"opt_TP_alphas_ind_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_{self.nalpha3}_alpha3_discretization_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}_alpha3log_{self.alp_3_lb}_{self.alp_3_ub}"),opt_TP_alphas_ind)
        print(f"Saved opt_TP_alphas_ind") 

        #Save the OP optimal alphas and their indices
        np.save(os.path.join(file_path, f"opt_OP_alphas_{date}_{self.nalpha1}_alpha1_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}"),opt_OP_alpha)
        print(f"Saved opt_alp_OPs") 
        np.save(os.path.join(file_path, f"opt_OP_alphas_ind_{date}_{self.nalpha1}_alpha1_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}"),opt_OP_alpha_ind)
        print(f"Saved opt_alp_OP_inds") 

        #Save the Reconstructions
        np.save(os.path.join(file_path, f"f_rec_TP_grids_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_{self.nalpha3}_alpha3_discretization_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}_alpha3log_{self.alp_3_lb}_{self.alp_3_ub}"),f_rec_TP_grids)
        print(f"Saved f_rec_TP_grids") 
        np.save(os.path.join(file_path, f"f_rec_OP_grids_{date}_{self.nalpha1}_alpha1_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}"),f_rec_OP_grids)
        print(f"Saved f_rec_OP_grids") 

        #Save the Error between TP and OP Reconstructions and Ground Truth
        np.save(os.path.join(file_path, f"errors_TP_grid_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_{self.nalpha3}_alpha3_discretization_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}_alpha3log_{self.alp_3_lb}_{self.alp_3_ub}"),errors_TP_grid)
        print(f"Saved errors_TP_grid") 
        np.save(os.path.join(file_path, f"errors_OP_grid_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_{self.nalpha3}_alpha3_discretization_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}_alpha3log_{self.alp_3_lb}_{self.alp_3_ub}"),errors_OP_grid)
        print(f"Saved errors_OP_grid") 

def create_result_folder():
    # Create a folder based on the current date and time
    date = datetime.now().strftime("%Y%m%d")
    folder_name = f"/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/3D_Grid_Search/{date}_Run"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return folder_name


#############################################################################################
#Initial Parameters
if __name__ == '__main__':
    
    print("Running 3D_multi_reg_MRR_0123.py script")
    file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/"
    # Get today's date
    today_date = datetime.today()
    # Format the date as "month_day_lasttwodigits"
    date = today_date.strftime("%m_%d_%y")

    T2 = np.arange(1,202).T
    TE = np.arange(1,512,4).T
    nalpha1 = 15
    nalpha2 = 15
    nalpha3 = 15
    #minus6_0
    # alp_1_lb = -6
    # alp_1_ub = 0
    # alp_2_lb = -6
    # alp_2_ub = 0
    # alp_3_lb = -6
    # alp_3_ub = 0

    alp_1_lb = -3
    alp_1_ub = -1
    alp_2_lb = 0
    alp_2_ub = 0
    alp_3_lb = 0
    alp_3_ub = 0

    SNR = 200

    #-4 to -1 for line of solutions

    mu1 = 30
    mu2 = 110
    mu3 = 165
    sigma1 = 9
    sigma2 = 5
    sigma3 = 2

    MRR_inst = three_dim_MRR_grid_search(T2, TE, SNR, nalpha1, nalpha2, nalpha3, mu1, mu2, mu3, sigma1, sigma2, sigma3, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, alp_3_lb, alp_3_ub)
    MRR_inst.get_G()
    MRR_inst.get_g()
    MRR_inst.decompose_G()
    # MRR_inst.get_noisy_data()
    MRR_inst.get_diag_matrices()

    Alpha_vec = np.logspace(alp_1_lb, alp_1_ub, nalpha1)
    Alpha_vec2 = np.linspace(alp_2_lb, alp_2_ub, nalpha2)
    Alpha_vec3 = np.linspace(alp_3_lb, alp_3_ub, nalpha3)

    num_real=3
    MRR_inst.run_all(Alpha_vec, Alpha_vec2,Alpha_vec3, num_real, file_path, date)
    print("Finished 3D_multi_reg_MRR_0123.py script")
