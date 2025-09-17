"""#This is the Two Parameter (TP) grid search problem inspired by Lu paper and this is applied onto the MRR simulated problem

#Our diagonal matrices B1 and B2 represent the index positions of our T2 axis, where ones represent where along the T2 axis 
#we want our regularization to be applied and zeros represent where we do not apply regularization.

#In this case we split our T2 axis into two equal parts and apply regularization accordingly. We seek to obtain the optimal alpha1 value for 
#the positions in the T2 axis covered by the B1 matrix, and we seek to obtain the optimal alpha2 value for the positions in the T2 axis
#covered by the B2 matrix

#The goal is to find the optimal alpha1 and alpha2 regularization parameters assuming we know the ground truth. This is essentially the oracle lambda approach.

i
#We compare with One Parameter (OP) Grid Search representing the best conventional tikhonov regularization parameter.
"""
# #Packages
from tools.trips_py.new_gcv_pasha import generalized_crossvalidation
from pylops import LinearOperator, Identity
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
from src.utils.load_imports.load_classical import *
from mpl_toolkits.mplot3d import Axes3D
import h5py
import plotly.graph_objects as go
import plotly.io as pio
import sympy as sp
from autograd import grad, elementwise_grad, jacobian, hessian, holomorphic_grad
from mpl_toolkits import mplot3d
########################################################################
#Define functions and classes



class MRR_grid_search():
    def __init__(self, T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub):
        self.T2 = T2
        self.TE = TE
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alp_1_lb = alp_1_lb
        self.alp_1_ub = alp_1_ub
        self.alp_2_lb = alp_2_lb
        self.alp_2_ub = alp_2_ub
        self.nT2 = len(T2)
        self.nTE = len(TE)
        self.SNR = SNR
        self.nalpha1 = nalpha1
        self.nalpha2 = nalpha2
        self.two_param_init_params()
        self.one_param_grid_init_params()
        self.one_param_conv_methods_init_params()

    def two_param_init_params(self):
        self.G = None
        self.g = None
        self.noise = None
        self.data_noisy = None
        self.B_1 = None
        self.B_2 = None

        self.f_rec_TP_grid = None
        self.com_rec_TP_grid = None
        self.TP_x_lc_vec = None
        self.TP_rhos = None
        self.TP_resid = None
        self.com_rec_DP_TP = None
        self.f_rec_TP_DP = None
        self.TP_DP_alpha1 = None
        self.TP_DP_alpha2 = None
        self.TP_min_alpha1 = None
        self.TP_min_alpha2 = None
        self.TP_min_alpha1_ind = None
        self.TP_min_alpha2_ind = None
        self.TP_DP_alpha2_ind = None
        self.TP_DP_alpha1_ind = None
        self.TP_HS_alp1_ind = None
        self.TP_HS_alp2_ind = None
        self.f_rec_TP_HS = None
        self.com_rec_HS_TP = None
        self.TP_fig = None

    def one_param_grid_init_params(self):
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
    
    def decompose_G(self):
        U, s, V = csvd(self.G, tst = None, nargin = 1, nargout = 3)
        self.U = U
        self.s = s
        self.V = V

    def get_g(self):
        g1 = (1 / (np.sqrt(2 * np.pi) * self.sigma1)) * np.exp(-((self.T2 - self.mu1) ** 2) / (2 * self.sigma1 ** 2))
        g2 = (1 / (np.sqrt(2 * np.pi) * self.sigma2)) * np.exp(-((self.T2 - self.mu2) ** 2) / (2 * self.sigma2 ** 2))
        self.g  = 0.5 * (g1 + g2)
    
    def get_noisy_data(self):
        data_noiseless = np.dot(self.G, self.g)
        SD_noise= 1/self.SNR*max(abs(data_noiseless))
        self.noise = np.random.normal(0,SD_noise, data_noiseless.shape)
        self.data_noisy = data_noiseless + self.noise

    def generate_B_matrices(self, N, M):
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
    
    def get_diag_matrices(self):
        half_nT2 = int(self.nT2/2) 
        half_mat1 = np.eye(half_nT2)
        # half_mat2 = np.eye(half_nT2 - 1)
        self.B_1 = np.zeros((self.nT2, self.nT2))
        self.B_2 = np.zeros((self.nT2, self.nT2))
        # self.B_1[:half_nT2 + 1, :half_nT2 + 1] = half_mat1[:half_nT2 + 1, :half_nT2 + 1]
        # self.B_2[-(half_nT2 - 1):, -(half_nT2 - 1):] = half_mat2[-(half_nT2 - 1):, -(half_nT2 - 1):]
        self.B_1[:half_nT2, :half_nT2] = half_mat1[:half_nT2, :half_nT2]
        self.B_2[-(half_nT2):, -(half_nT2):] = half_mat1[-(half_nT2):, -(half_nT2):]
    
    ###### Objective Functions
    def minimize_TP_objective(self, Alpha_vec, Alpha_vec2):
        def sigmoidBigLeft(x, a, b):
            return 1 / (1 + np.exp(-(x - a) / b))
        def sigmoidBigRight(x, a, b):
            return -1 / (1 + np.exp(-(x - a) / b))
        half_nT2 = int(self.nT2/2) 
        self.TP_x_lc_vec = np.zeros((self.nT2, len(Alpha_vec), len(Alpha_vec2)))
        self.TP_rhos = np.zeros((len(Alpha_vec), len(Alpha_vec2)))
        self.TP_resid = np.zeros((len(Alpha_vec), len(Alpha_vec2)))
        for j, k in (product(range(len(Alpha_vec)), range(len(Alpha_vec2)))):
            alp_mat = (Alpha_vec[j])* self.B_1.T @ self.B_1 + (Alpha_vec2[k]) * self.B_2.T @ self.B_2
            A = np.linalg.inv(self.G.T @ self.G + Alpha_vec[j] * self.B_1.T @ self.B_1 + Alpha_vec2[k] * self.B_2.T @ self.B_2)
            x = np.diag(alp_mat)
            if Alpha_vec[j] > Alpha_vec[k]:
                smoothed_function = sigmoidBigRight(np.arange(len(x)), a=half_nT2, b=30)
            else: 
                smoothed_function = sigmoidBigLeft(np.arange(len(x)), a=half_nT2, b=30)
            # Rescale the smoothed function to match alp1 and alp2
            scaled_alp_mat = (smoothed_function - np.min(smoothed_function)) * (Alpha_vec[k] - Alpha_vec[j]) / (np.max(smoothed_function) - np.min(smoothed_function)) + Alpha_vec[j]
            # # plt.plot(scaled_alp_mat, label = "resclaed")
            # # plt.plot(np.diag(alp_mat), label = "orig")
            # # plt.show()
            # # plt.close()
            A = self.G.T @ self.G + np.diag(scaled_alp_mat)
            A = self.G.T @ self.G + alp_mat

            b =  self.G.T @ self.data_noisy
            invA = np.linalg.inv((A))
            print("Condition number:",np.linalg.cond(invA))

            # Save the array to a text file
            # np.savetxt(f"/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/2D_grid_search_20240308_Run/InverseA_alp1{j}_alp2{k}.txt",invA)
            string = "InverseA"
            file_path = create_result_folder(string)
            np.savetxt(f"{file_path}/InverseA_alp1{j}_alp2{k}.txt",invA)
            # plt.plot(invA)
            # plt.imshow(invA, cmap='viridis', aspect='auto', origin='lower')
            # plt.colorbar()  # Add colorbar
            # plt.show()

            # print(invA[98:102,:])

            # exp = np.linalg.inv(A) @ b
            # exp = self.iterative_algorithm(self.G, self.data_noisy, self.B_1, self.B_2, Alpha_vec[j], Alpha_vec2[k], max_iter=1000, tol=1e-6)
            # exp = nnls(A, b, maxiter = 1e30)[0]
            x = cp.Variable(self.nT2)
            cost = cp.norm(A @ x - b, 2)**2
            constraints = [x >= 0]
            problem = cp.Problem(cp.Minimize(cost), constraints)
            # exp,_ = fnnls(A,b)
            problem.solve(solver=cp.MOSEK, verbose=False)
            self.TP_x_lc_vec[:, j, k] = x.value
            # print("L_x_lc_vec[:,j,k]", L_x_lc_vec[:,j,k])
            # print("exp", exp)
            self.TP_rhos[j,k]  = norm(self.TP_x_lc_vec[:, j, k] - self.g, 2)**2
            self.TP_resid[j,k] = norm(self.G @ self.TP_x_lc_vec[:,j,k] - self.data_noisy, 2)
            print(f"finished Alpha{j}, Alpha{k}")
            # x = cp.Variable(self.nT2)
            # objective = cp.Minimize(cp.norm(self.G @ x - self.data_noisy, 2)**2 +
            #                         Alpha_vec[j] * cp.norm(self.B_1 @ x, 2)**2 +
            #                         Alpha_vec2[k] * cp.norm(self.B_2 @ x, 2)**2)
            # constraints = [x >= 0]
            # problem = cp.Problem(objective, constraints)
            # problem.solve(solver=cp.MOSEK, verbose=False)
            # if x.value is not None:
            #     self.TP_x_lc_vec[:, j, k] = x.value.flatten()
            #     self.TP_rhos[j, k] = norm(self.TP_x_lc_vec[:, j, k] - self.g, 2)**2
            #     self.TP_resid[j,k] = norm(self.G @ self.TP_x_lc_vec[:,j,k] - self.data_noisy, 2)
            # else:
            #     print("x.value == None")

    def projection_onto_C(self,x):
        return np.maximum(x, 0)

    def iterative_algorithm(self, A, b, B_1, B_2, lambd1, lambd2, max_iter=1000, tol=1e-6):
        x0 = np.ones(A.shape[1])
        x = x0.copy()  # Initialize x with initial guess
        ATA = A.T @ A
        LTL1 = B_1.T @ B_1
        LTL2 = B_2.T @ B_2
        alpha = 1
        I = np.eye(len(x))
        
        for _ in range(max_iter):
            x_new = self.projection_onto_C(x - alpha * ((ATA + lambd1/2 * LTL1 + lambd2/2 * LTL2) @ x - A.T @ b))
            
            # Check convergence
            if np.linalg.norm(x_new - x) < tol:
                break
            
            x = x_new
        
        return x

    def minimize_OP_objective(self, Alpha_vec):
        self.OP_x_lc_vec = np.zeros((self.nT2, len(Alpha_vec)))
        self.OP_rhos = np.zeros((len(Alpha_vec)))
        for j in (range(len(Alpha_vec))):
            A  = self.G.T @ self.G + Alpha_vec[j] * self.B_1.T @ self.B_1 + Alpha_vec2[j] * self.B_2.T @ self.B_2
            b =  self.G.T @ self.data_noisy
            # exp = np.linalg.inv(A) @ b
            # exp = self.iterative_algorithm(self.G, self.data_noisy, self.B_1, self.B_2, Alpha_vec[j], Alpha_vec2[j], max_iter=1000, tol=1e-6)
            # exp, _ = nnls(A, b, maxiter = 1e10)
            # exp = fe.fnnls(A, b) 
            # print(np.linalg.cond(A))
            # exp = nnls(A, b, maxiter = 1e30)[0]
            x = cp.Variable(self.nT2)
            # cost = cp.sum_squares(A @ x - b)
            cost = cp.norm(A @ x - b, 2)**2
            constraints = [x >= 0]
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.MOSEK, verbose=False)
            # exp,_ = fnnls(A,b)
            self.OP_x_lc_vec[:, j] = x.value
            # print("L_x_lc_vec[:,j,k]", L_x_lc_vec[:,j,k])
            self.OP_rhos[j] = norm(self.OP_x_lc_vec[:,j] - self.g, 2)**2
            # self.TP_resid[j,k] = norm(self.G @ self.TP_x_lc_vec[:, j, k] - self.data_noisy)
            # x = cp.Variable(self.nT2)
            # objective = cp.Minimize(cp.norm(self.G @ x - self.data_noisy, 2)**2 +
            #                         Alpha_vec[j] * cp.norm(self.B_1 @ x, 2)**2 +
            #                         Alpha_vec[j] * cp.norm(self.B_2 @ x, 2)**2)
            # constraints = [x >= 0]
            # problem = cp.Problem(objective, constraints)
            # problem.solve(solver=cp.MOSEK, verbose=False)
            # if x.value is not None:
            #     self.OP_x_lc_vec[:, j] = x.value.flatten()
            #     self.OP_rhos[j] = norm(self.OP_x_lc_vec[:,j] - self.g, 2)**2
            # else:
            #     print("x.value == None")
    
    def minimize_DP(self, SNR):
        delta = norm(self.noise)*1.05 
        # delta = norm(self.noise)**2
        x_delta,lambda_OP = discrep(self.U,self.s,self.V,self.data_noisy,delta, x_0= None, nargin = 5)
        self.lam_DP_OP = lambda_OP
        Lambda = np.ones(len(self.T2))* lambda_OP

        f_rec_DP_OP,_ = discrep_L2(self.data_noisy, self.G, SNR, Lambda)
        self.f_rec_DP_OP = f_rec_DP_OP.flatten()

        # f_rec_DP_OP,_,_ = tikhonov(self.U,self.s,self.V,self.data_noisy,lambda_OP, nargin=5, nargout=1)
        # self.f_rec_DP_OP = f_rec_DP_OP.flatten()

        """NNLS
        # f_rec_DP_OP,_ = discrep_L2(self.data_noisy, self.G, SNR, Lambda)
        # self.f_rec_DP_OP = f_rec_DP_OP.flatten()
        """
        # Lambda = np.ones(len(self.T2))* self.OP_min_alpha1
        # f_rec,self.lam_DP_OP = discrep_L2(self.data_noisy, self.G, SNR, Lambda)
        # self.f_rec_DP_OP = f_rec.flatten()

    def minimize_LC(self):
        reg_corner,rho,eta,_ = l_curve(self.U,self.s,self.data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
        self.lam_LC_OP = reg_corner
        # f_rec_LC_OP,_,_ = tikhonov(self.U,self.s,self.V,self.data_noisy,reg_corner, nargin=5, nargout=1)
        # self.f_rec_LC_OP = f_rec_LC_OP.flatten()

        """NNLS
        """
        Lambda =np.ones(len(self.T2))* reg_corner
        Lambda = Lambda.reshape(-1,1)
        f_rec_LC_OP,_ = Lcurve(self.data_noisy, self.G, Lambda)
        self.f_rec_LC_OP = f_rec_LC_OP.flatten()

        # Lambda =np.ones(len(self.T2))* self.OP_min_alpha1
        # Lambda = Lambda.reshape(-1,1)
        # f_rec, lam = Lcurve(self.data_noisy, self.G, Lambda)
        # self.f_rec_LC_OP = f_rec.flatten()
        # self.lam_LC_OP = lam[0]

    def get_DP_vals(self, L_residual, noise):
        data_noiseless = np.dot(self.G, self.g)
        SD_noise= 1/self.SNR*max(abs(data_noiseless))
        m = data_noiseless.shape[0]
        print("m", m)
        # delta = norm(self.noise)
        delta = np.sqrt(m) * SD_noise
        print("delta", delta)
        ind = np.unravel_index(np.argmin(np.abs(L_residual - delta)), L_residual.shape)
        DP_val = L_residual[ind]
        return DP_val, ind

    def minimize_GCV(self):
        reg_min,_,_ = gcv(self.U,self.s,self.data_noisy, method = 'Tikh', nargin = 3, nargout = 3)
        self.lam_GCV_OP = reg_min
        # f_rec_GCV_OP,_,_ = tikhonov(self.U,self.s,self.V,self.data_noisy,reg_min, nargin=5, nargout=1)
        # self.f_rec_GCV_OP = f_rec_GCV_OP
        Lambda = np.ones(len(self.T2))* reg_min
        f_rec,_ = GCV_NNLS(self.data_noisy, self.G, Lambda)
        self.f_rec_GCV_OP = f_rec[:,0]

        """NNLS
        Lambda = np.ones(len(self.T2))* reg_min
        f_rec,_ = GCV_NNLS(self.data_noisy, self.G, Lambda)
        self.f_rec_GCV_OP = f_rec[:,0]
        """
        # f_rec_GCV_OP,_,_ = tikhonov(self.U,self.s,self.V,self.data_noisy,reg_min, nargin=5, nargout=1)
        # Lambda = np.ones(len(self.T2))* self.OP_min_alpha1
        # f_rec, lam = GCV_NNLS(self.data_noisy, self.G, Lambda)
        # self.f_rec_GCV_OP = f_rec[:,0]
        # self.lam_GCV_OP = lam[0]

    ###### Plotting Grid Search Surfaces

    def plot_TP_surface(self, Alpha_vec, Alpha_vec2, error_val, curvature_cond, resid_cond, iter_num):

        # Assuming Lambda_vec, Lambda_vec2, and self.rhos are defined
        # Create meshgrid
        plot1, plot2 = np.meshgrid(Alpha_vec, Alpha_vec2)
        min_index = np.unravel_index(np.argmin(error_val), error_val.shape)
        max_index = np.unravel_index(np.argmax(error_val), error_val.shape)

        if curvature_cond == True:
            TP_zval = error_val
            Alpha_vector = np.log10(Alpha_vec)
            Alpha_vector2 = np.log10(Alpha_vec2)
            z_val_title = "Curvature"
            min_x = Alpha_vector[max_index[0]]
            min_y = Alpha_vector2[max_index[1]]
            min_z = np.max(error_val)
            goal_name = "Maximum Curvature"
            typescale = 'linear'
        else:
            if resid_cond == False:
                TP_zval = TP_log_err_norm = np.log10(error_val)
                z_val_title = 'Log(norm(p_a1,a2 - p_true)**2)'
            else:
                TP_zval = error_val
                z_val_title = 'Residual Norm'
            # Alpha_vector = np.log10(Alpha_vec)
            Alpha_vector = (Alpha_vec)
            # print(Alpha_vec10.shape)
            # Alpha_vector2 = np.log10(Alpha_vec2)
            Alpha_vector2 = (Alpha_vec2)
            # print(Alpha_vec210.shape)
            min_x = Alpha_vector[min_index[0]]
            min_y = Alpha_vector2[min_index[1]]
            min_z = np.min(error_val)
            goal_name = "Minimum Point"
            typescale = 'linear'
        # Find indices of minimum value

        # Create a 3D surface plot
        fig = go.Figure(data=[go.Surface(x=Alpha_vector, y=Alpha_vector2, z= error_val.T)])

        # Add a scatter plot for the minimum point
        fig.add_trace(go.Scatter3d(
            x=np.array([min_x]),
            y=np.array([min_y]),
            z=np.array([min_z]),
            mode='markers',
            marker=dict(
                size=5,
                color='orange',
                symbol='circle'
            ),
            name= goal_name
        ))

        if resid_cond == True:
            m = 128
            data_noiseless = np.dot(self.G, self.g)
            SD_noise= 1/self.SNR*max(abs(data_noiseless))
            # delta = np.sqrt(m) * SD_noise
            delta = norm(self.noise)
            closest_indices = np.unravel_index(np.argmin(np.abs(error_val - delta)), error_val.shape)
            ind_x = closest_indices[0]
            ind_y = closest_indices[1]

            fig.add_trace(go.Surface(
            x=Alpha_vector,
            y=Alpha_vector2,
            z=[[delta] * len(Alpha_vector)] * len(Alpha_vector2),  # Creates a plane at z=delta
            colorscale='Viridis',  # Adjust the colorscale as needed
            opacity=0.8,  # Adjust the opacity as needed
            showscale=False,  # Hide the color scale
            name='Delta Plane'
            ))
            fig.add_trace(go.Scatter3d(
            x=[Alpha_vector[ind_x]],
            y=[Alpha_vector2[ind_y]],
            z=[error_val[ind_x, ind_y]],
            mode='markers',
            marker=dict(
                size=5,
                color='red',
                symbol='circle'
            ),
            name='Closest Point to Delta'
            ))

        # Configure layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(type=typescale, title=f'Alpha_1 Values 10^{self.alp_1_lb} to 10^{self.alp_1_ub}'),
                yaxis=dict(type=typescale, title=f'Alpha_2 Values 10^{self.alp_2_lb} to 10^{self.alp_2_ub}'),
                zaxis=dict(title= z_val_title)  # Assuming z-axis is log scale
            ),
            title=f"Surface of Grid Search for NR {iter_num}",
            annotations=[
                dict(
                    text=f"Optimal Alpha_1 value: {min_x}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1.08
                ),
                dict(
                    text=f"Optimal Alpha_2 value: {min_y}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=1.05
                )
            ]
        )
        return fig

    def plot_OP_surface(self, Alpha_vec, rhos, file_path, iter_num):
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
        plt.savefig(os.path.join(file_path, f"One_Parameter_Surface_Plot_{date}_NR_{iter_num}_mu1_{self.mu1}_mu2_{self.mu2}_sigma1_{self.sigma1}_sigma2_{self.sigma2}.png"))
        print(f"Saved One Parameter Surface Plot for NR {iter_num}")
        plt.close()
    
    ##### Get Optimal Alphas

    def get_opt_TP_alphas(self, Alpha_vec, Alpha_vec2):
        TP_log_err_norm = np.log10(self.TP_rhos)
        # min_indices = np.where(Z == np.min(Z))
        min_index = np.unravel_index(np.argmin(TP_log_err_norm), TP_log_err_norm.shape)
        # delta = norm(self.noise)
        delta = norm(self.noise)
        closest_indices = np.unravel_index(np.argmin(np.abs(self.TP_resid - delta)), self.TP_resid.shape)
        ind_x = closest_indices[0]
        ind_y = closest_indices[1]
        self.TP_DP_alpha1_ind = ind_x
        self.TP_DP_alpha2_ind = ind_y
        self.TP_DP_alpha1 = Alpha_vec[ind_x]
        self.TP_DP_alpha2 = Alpha_vec2[ind_y]

        min_x = Alpha_vec[min_index[0]]
        min_y = Alpha_vec2[min_index[1]]
        min_z = np.min(TP_log_err_norm)

        self.TP_min_alpha1 = min_x
        self.TP_min_alpha1_ind = min_index[0]
        self.TP_min_alpha2 = min_y
        self.TP_min_alpha2_ind = min_index[1]

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
        self.f_rec_TP_grid = self.TP_x_lc_vec[:, self.TP_min_alpha1_ind, self.TP_min_alpha2_ind]
        self.f_rec_TP_DP = self.TP_x_lc_vec[:, self.TP_DP_alpha1_ind, self.TP_DP_alpha2_ind]
        
    def get_f_rec_OP_grid(self):
        self.f_rec_OP_grid = self.OP_x_lc_vec[:,self.OP_min_alpha1_ind]

    ##### Get L2 Errors of the TP and OP Reconstructions Against Ground Truth
    def get_error(self):
        self.com_rec_OP_grid = norm(self.g - self.f_rec_OP_grid)
        self.com_rec_TP_grid = norm(self.g - self.f_rec_TP_grid.flatten())
        self.com_rec_DP = norm(self.g - self.f_rec_DP_OP)
        self.com_rec_LC = norm(self.g - self.f_rec_LC_OP)
        self.com_rec_GCV = norm(self.g - self.f_rec_GCV_OP)
        self.com_rec_DP_TP = norm(self.g - self.f_rec_TP_DP)

    ##### Plot the Ground Truth with and without the TP and OP Reconstructions

    def plot_gt_and_comparison(self, file_path, date, compare, num_real):
        if compare != True:
            g_fig = plt.figure()
            plt.plot(MRR_inst.g)
            plt.xlabel("T2")
            plt.ylabel("Amplitude")
            plt.title("MRR Problem Ground Truth", fontsize = 15)
            plt.savefig(os.path.join(file_path, f"ground_truth_{date}_mu1_{self.mu1}_mu2_{self.mu2}_sigma1_{self.sigma1}_sigma2_{self.sigma2}"))
            print(f"Saving ground truth plot is complete")
            np.save(os.path.join(file_path, f"ground_truth_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"),self.g)
            print(f"Saving ground truth data  is complete") 
        else:
            # fig, axs = plt.subplots(1, 3, figsize=(24, 10))
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))

            # Subplot 1: Reconstructions
    
            axs[0, 0].plot(MRR_inst.g, label="Ground Truth",  color='black', linewidth=6.0)
            # axs[0].plot(self.f_rec_OP_grid, label="1P Oracle",  color='orange',linewidth=3.0)
            # axs[0].plot(self.f_rec_TP_grid, label="2P Oracle", color = "green", linewidth=3.0)
            axs[0, 0].plot(self.f_rec_TP_DP, label = "Two Parameter DP LR", color = "blue", linewidth = 3.0)
            # axs[0].plot(self.f_rec_TP_HS, label = "2P HyperSurface", color = "blue", linewidth = 3.0)
            # self.TP_HS_alp1_ind = inds[0]
            # self.TP_HS_alp2_ind = inds[1]
            # self.f_rec_TP_HS = f_star_alphaL
            axs[0, 0].plot(self.f_rec_DP_OP, label = "DP", color = "orange", linewidth = 3.0)
            axs[0, 0].plot(self.f_rec_LC_OP, label = "LC", color = "green", linewidth = 3.0)
            axs[0, 0].plot(self.f_rec_GCV_OP, label = "GCV", color = "red", linewidth = 3.0)

            axs[0, 0].set_xlabel("T2", fontsize = 20, fontweight='bold')
            axs[0, 0].set_ylabel("Amplitude", fontsize = 20, fontweight='bold')
            axs[0, 0].set_title(f"MRR Problem NR{num_real + 1}")
            # axs[0].legend(["Ground Truth", "1P Oracle", "2P Oracle", "2P HS", "1P DP", "1P LC", "1P GCV"])
            # axs[0].legend(["Ground Truth", "2P HS", "1P DP", "1P LC", "1P GCV"])
            axs[0, 0].legend(["Ground Truth", "2P DP", "1P DP", "1P LC", "1P GCV"])

            # axs[0].legend(["Ground Truth", "Conventional One Parameter Grid Search", "Two Parameter Grid Search", "Hypersurface", "GCV"])
            # self.com_rec_OP_grid
            OP_errors = [self.com_rec_DP_TP, self.com_rec_DP, self.com_rec_LC, self.com_rec_GCV]
            algo_names = ["2P DP", "1P DP", "1P LC", "1P GCV"]
            name_ind = np.argwhere(OP_errors == np.min(OP_errors))[0][0]
            minimum_algo = algo_names[name_ind]
            minimum_error = np.min(OP_errors)
            lamDP = self.lam_DP_OP[0]
            err_ratio = self.com_rec_OP_grid/self.com_rec_TP_grid
            # fig.suptitle(r"$\bf{L2\ Norm\ Error\ Values:}$" + f"\nE_1P_Oracle = {round(self.com_rec_OP_grid.item(), 4)}; E_2P_Oracle = {round(self.com_rec_TP_grid.item(), 4)};  E_2P_HS = {round(self.com_rec_HS_TP.item(),4)};E_1P_DP = {round(self.com_rec_DP.item(), 4)} ; E_1P_LC = {round(self.com_rec_LC.item(),4)} ; E_1P_GCV = {round(self.com_rec_GCV.item(),4)}" 
            #              + f"\n The lowest error for the One Parameter Algorithms is {minimum_algo} with error of {round(minimum_error.item(),4)}\n" + f"\n 1P to 2P Error Ratio = {round(err_ratio.item(),4)}\n" + f"SNR = {self.SNR}\n" + f"\n"
            #              + r"$\bf{Lambda/Alpha\ Values}$" + f"\n2P_Oracle_alpha_1 = {round(self.TP_min_alpha1.item(),4)}; 2P_Oracle_alpha_2 = {round(self.TP_min_alpha2.item(),4)}" +   f"\n2P_HS_alpha_1 = {round(Alpha_vec[self.TP_HS_alp1_ind].item(),4)}; 2P_HS_alpha_2 = {round(Alpha_vec2[self.TP_HS_alp2_ind].item(),4)}" + f"\n 1P_Oracle_alpha = {round(self.OP_min_alpha1.item(),4)}" 
            #              + f"\n 1P_DP_alpha = {round(lamDP,4)}; 1P_LC_alpha = {round(self.lam_LC_OP,4)} ; 1P_GCV_alpha = {round(self.lam_GCV_OP,4)}", fontsize=12)
            # fig.suptitle(r"$\bf{L2\ Norm\ Error\ Values:}$" + f"\nE_1P_Oracle = {round(self.com_rec_OP_grid.item(), 4)}; E_2P_Oracle = {round(self.com_rec_TP_grid.item(), 4)};  E_2P_DP = {round(self.com_rec_DP_TP.item(),4)};E_1P_DP = {round(self.com_rec_DP.item(), 4)} ; E_1P_LC = {round(self.com_rec_LC.item(),4)} ; E_1P_GCV = {round(self.com_rec_GCV.item(),4)}" 
            #              + f"\n The lowest error for the One Parameter Algorithms is {minimum_algo} with error of {round(minimum_error.item(),4)}\n" + f"\n 1P to 2P Error Ratio = {round(err_ratio.item(),4)}\n" + f"SNR = {self.SNR}\n" + f"\n"
            #              + r"$\bf{Lambda/Alpha\ Values}$" + f"\n2P_Oracle_alpha_1 = {round(self.TP_min_alpha1.item(),4)}; 2P_Oracle_alpha_2 = {round(self.TP_min_alpha2.item(),4)}" +   f"\n2P_DP_alpha_1 = {round(self.TP_DP_alpha1.item(),4)}; 2P_DP_alpha_2 = {round(self.TP_DP_alpha2.item(),4)}" + f"\n 1P_Oracle_alpha = {round(self.OP_min_alpha1.item(),4)}" 
            #              + f"\n 1P_DP_alpha = {round(lamDP,4)}; 1P_LC_alpha = {round(self.lam_LC_OP,4)} ; 1P_GCV_alpha = {round(self.lam_GCV_OP,4)}", fontsize=12)

            # Adjust subplot settings
            # Subplot 2: Alpha Plots
            def sigmoidBigLeft(x, a, b):
                return 1 / (1 + np.exp(-(x - a) / b))
            def sigmoidBigRight(x, a, b):
                return -1 / (1 + np.exp(-(x - a) / b))
            half_nT2 = int(self.nT2/2) 

            def which_smooth(alp1,alp2):
                alp_mat = (alp1)* self.B_1.T @ self.B_1 + (alp2) * self.B_2.T @ self.B_2
                # A = np.linalg.inv(self.G.T @ self.G + Alpha_vec[j] * self.B_1.T @ self.B_1 + Alpha_vec2[k] * self.B_2.T @ self.B_2)
                x = np.diag(alp_mat)
                if alp1 > alp2:
                    smoothed_function = sigmoidBigRight(np.arange(len(x)), a=half_nT2, b=30)
                else: 
                    smoothed_function = sigmoidBigLeft(np.arange(len(x)), a=half_nT2, b=30)
                # Rescale the smoothed function to match alp1 and alp2
                scaled_alp_mat = (smoothed_function - np.min(smoothed_function)) * (alp2 - alp1) / (np.max(smoothed_function) - np.min(smoothed_function)) + alp1
                return scaled_alp_mat
            TP_Alphas = which_smooth(self.TP_min_alpha1.item(), self.TP_min_alpha2.item())
            TP_DP_Alphas = which_smooth(self.TP_DP_alpha1.item(), self.TP_DP_alpha2.item())
            # TP_Alphas = self.TP_min_alpha1.item() * np.diag(self.B_1) + self.TP_min_alpha2.item() * np.diag(self.B_2) 
            # TP_DP_Alphas = self.TP_DP_alpha1.item() * np.diag(self.B_1) + self.TP_DP_alpha2.item() * np.diag(self.B_2) 
            # TP_HS_Alphas = Alpha_vec[self.TP_HS_alp1_ind].item() * np.diag(self.B_1) + Alpha_vec2[self.TP_HS_alp2_ind].item() * np.diag(self.B_2)
            #Print out the OP Alpha
            # axs[1].semilogy(self.T2, self.OP_min_alpha1.item() * np.ones(len(self.T2)), linewidth=3, color='orange', label='OP Oracle')
            axs[0,1].semilogy(self.T2, TP_Alphas, linewidth = 3, color = "black", label = "2P Oracle")

            axs[0, 1].semilogy(self.T2, TP_DP_Alphas, linewidth = 3, color = "blue", label = "2P DP")
            # axs[1].semilogy(self.T2, TP_HS_Alphas, linewidth = 3, color = "blue", label = "2P HyperSurface")
            axs[0, 1].semilogy(self.T2, self.lam_DP_OP * np.ones(len(self.T2)), linewidth = 3, color = "orange", label = "1P DP")
            axs[0, 1].semilogy(self.T2, self.lam_LC_OP * np.ones(len(self.T2)), linewidth = 3, color = "green", label = "1P LCurve")
            axs[0, 1].semilogy(self.T2, self.lam_GCV_OP * np.ones(len(self.T2)), linewidth = 3, color = "red", label = "1P GCV")

            handles, labels = axs[0, 1].get_legend_handles_labels()
            dict_of_labels = dict(zip(labels, handles))
            axs[0, 1].legend(dict_of_labels.values(), dict_of_labels.keys(), fontsize=10, loc='best')
            axs[0, 1].set_xlabel('T2', fontsize=20, fontweight='bold')
            axs[0, 1].set_ylabel('Alpha', fontsize=20, fontweight='bold')

            #Subplot 3: Decay Curves; #Plot Noisy Decay Curve
            axs[1, 0].plot(self.TE, self.G @ MRR_inst.g, linewidth = 3, color = "black", label = "Ground Truth")
            # axs[2].plot(self.TE, self.G @ self.f_rec_TP_grid, linewidth=3, color='green', label='2P Oracle')
            axs[1, 0].plot(self.TE, self.G @ self.f_rec_TP_DP, linewidth=3, color='blue', label='2P DP')
            # axs[2].plot(self.TE, self.G @ self.f_rec_TP_HS, linewidth=3, color='blue', label='2P HS')
            # axs[2].plot(self.TE, self.G @ self.f_rec_OP_grid, linewidth=3, color= "orange", label='1P Oracle')
            axs[1, 0].scatter(self.TE, self.data_noisy, s=30, color="black", label='Data Points')
            axs[1, 0].plot(self.TE, self.G @ self.f_rec_DP_OP, linewidth=3, color='orange', label='1P DP')
            axs[1, 0].plot(self.TE, self.G @ self.f_rec_LC_OP, linestyle='-.', linewidth=3, color='green', label='1P LCurve')
            axs[1, 0].plot(self.TE, self.G @ self.f_rec_GCV_OP, linestyle='--', linewidth=3, color='red', label='1P GCV')
            axs[1, 0].legend(fontsize=10, loc='best')
            axs[1, 0].set_xlabel('TE', fontsize=20, fontweight='bold')
            axs[1, 0].set_ylabel('Intensity', fontsize=20, fontweight='bold')

            table_ax = axs[1, 1]
            table_ax.axis('off')

            # Define the data for the table
            data = [
                ["E_1P_Oracle", round(self.com_rec_OP_grid.item(), 4)],
                ["E_2P_Oracle", round(self.com_rec_TP_grid.item(), 4)],
                ["E_2P_DP", round(self.com_rec_DP_TP.item(), 4)],
                ["E_1P_DP", round(self.com_rec_DP.item(), 4)],
                ["E_1P_LC", round(self.com_rec_LC.item(), 4)],
                ["E_1P_GCV", round(self.com_rec_GCV.item(), 4)],
                ["Lowest Error Algorithm", minimum_algo],
                ["Error of Lowest Algorithm", round(minimum_error.item(), 4)],
                ["1P to 2P Error Ratio", round(err_ratio.item(), 4)],
                ["SNR", self.SNR],
                ["2P_Oracle_alpha_1", round(self.TP_min_alpha1.item(), 4)],
                ["2P_Oracle_alpha_2", round(self.TP_min_alpha2.item(), 4)],
                ["2P_DP_alpha_1", round(self.TP_DP_alpha1.item(), 4)],
                ["2P_DP_alpha_2", round(self.TP_DP_alpha2.item(), 4)],
                ["1P_Oracle_alpha", round(self.OP_min_alpha1.item(), 4)],
                ["1P_DP_alpha", round(lamDP, 4)],
                ["1P_LC_alpha", round(self.lam_LC_OP, 4)],
                ["1P_GCV_alpha", round(self.lam_GCV_OP, 4)]
            ]

            # Create the table
            table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.2)

            # Adjust layout for better spacing
            # Adjust layout for better spacing
            plt.tight_layout()
            plt.savefig(os.path.join(file_path, f"compare_gt_convtikhreg_multireg_{date}_NR_{num_real + 1}_mu1_{self.mu1}_mu2_{self.mu2}_sigma1_{self.sigma1}_sigma2_{self.sigma2}"))
            print(f"Saving comparison plot is complete")
            plt.close()

    def run_all(self, Alpha_vec, Alpha_vec2, num_real, date):

        #2D Grid Search
        opt_TP_alphas = []
        opt_TP_alphas_ind = []
        opt_TP_DP_alphas_ind = []

        opt_OP_alpha =[]
        opt_OP_alpha_ind = []

        f_rec_TP_grids = []
        f_rec_OP_grids = []

        errors_TP_grid = []
        errors_OP_grid = []


        f_star_alps_NR = []
        residual_norms_NR = []
        B_1_constr_norms_NR = []
        B_2_constr_norms_NR = []
        first_partial_deriv_B1s_NR = []
        second_partial_deriv_B2s_NR = []
        pmats_NR= []
        determinants_NR = []
        w_sqrs_NR = []
        w_m_plus_1s_NR = []
        curvatures_NR = []
        #L_hypersurface
        # f_star_alps = []
        # both_constraint_norms_list = []
        # residual_norms = []
        f_star_alphaLs = []
        efficiency_vals = []
        error_norm_min_inds = []
        error_norm_min_alp1 = []
        error_norm_min_alp2 = []
        curvature_min_inds = []
        curvature_min_alp1 = []
        curvature_min_alp2 = []
        # all_first_partials_list = []
        # p_mat_list = []
        # determinants = []
        # w_sqrs = []
        # w_m_plus_1s = []
        # curvatures = []

        for i in tqdm(range(num_real)):
            string = "2D_grid_search"
            file_path = create_result_folder(string)

            #Construct and Save the Noisy Data
            self.get_noisy_data()
            np.save(os.path.join(file_path, f"noisy_data_for_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretization_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"), self.data_noisy)
            print(f"Saved Noisy Data for NR {i+1}")

            # print("B_1", np.where(self.B_1 == 1))
            # print("B_1", self.B_1.shape)
            # print("B_2", np.where(self.B_2 == 1))
            # print("B_2", self.B_2.shape)
            # print("B_1.T @ B_1", (self.B_1.T @ self.B_1).shape)
            # print("B_2.T @ B_2", (self.B_2.T @ self.B_2).shape)
            # print("Alpha_vec1", Alpha_vec.shape)
            # print("Alpha_vec2", Alpha_vec2.shape)

            #Minimize the Objective Functions to Find the Surface and Save

            ######Minimize and Save the TP Objective
            self.minimize_TP_objective(Alpha_vec, Alpha_vec2)
            print(f"Minimized the Two Parameter Objective Function for NR {i+1}")
            np.save(os.path.join(file_path, f"TP_rhos_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"),self.TP_rhos)
            print(f"Saved TP_rhos for NR {i+1}")
            np.save(os.path.join(file_path, f"TP_x_lc_vec_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"),self.TP_x_lc_vec)
            print(f"Saved TP_x_lc_vec for NR {i+1}")

            # DP_val, ind = self.get_DP_vals(self.TP_resid, self.noise) 
            # print("DP_val", DP_val)
            # print("ind", ind)
            #######Minimize and Save the OP Objective 
            self.minimize_OP_objective(Alpha_vec)
            print(f"Minimized the One Parameter Objective Function for NR {i+1}")
            np.save(os.path.join(file_path, f"OP_rhos_NR{i+1}_{date}_{self.nalpha1}_alpha1_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}"),self.OP_rhos)
            print(f"Saved OP_rhos for NR {i+1}")
            np.save(os.path.join(file_path, f"OP_x_lc_vec_NR{i+1}_{date}_{self.nalpha1}_alpha1_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}"),self.OP_x_lc_vec)
            print(f"Saved OP_x_lc_vec for NR {i+1}")

            #Plot the Surfaces and Save

            ######Plot the TP Surface and Save
            curvature_cond = False
            resid_cond = False
            self.TP_fig = self.plot_TP_surface(Alpha_vec, Alpha_vec2, self.TP_rhos, curvature_cond, resid_cond, i+1)
            print(f"Plotted the Surface for the Two Parameter Grid Search for NR {i+1}")
            pio.write_html(self.TP_fig, file= os.path.join(file_path, f"TP_rhos_surface_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}.html"))
            print(f"Saved Two Parameter Surface Plot for NR {i+1}")

            curvature_cond = False
            resid_cond = True
            self.TP_fig = self.plot_TP_surface(Alpha_vec, Alpha_vec2, self.TP_resid, curvature_cond, resid_cond, i+1)
            print(f"Plotted the Surface for the Two Parameter Grid Search for NR {i+1}")
            pio.write_html(self.TP_fig, file= os.path.join(file_path, f"TP_resid_surface_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}.html"))
            print(f"Saved Two Parameter Surface Plot for NR {i+1}")
            #######Plot the OP Surface and Save
            self.plot_OP_surface(Alpha_vec,self.OP_rhos,file_path, i+1)
            print(f"Saved One Parameter Surface Plot for NR {i+1}")

            #Get the Optimal Alpha Values and Save

            #######Retrieve Optimal TP Alphas

            self.get_opt_TP_alphas(Alpha_vec, Alpha_vec2)
            print(f"Retrieved Optimal Two Parameter Alphas for NR {i+1}")
            opt_TP_alphas.append((self.TP_min_alpha1, self.TP_min_alpha2))
            opt_TP_alphas_ind.append((self.TP_min_alpha1_ind, self.TP_min_alpha2_ind))
            opt_TP_DP_alphas_ind.append((self.TP_DP_alpha1_ind, self.TP_DP_alpha2_ind))

            # #######Retrieve Optimal OP Alphas
            self.get_opt_OP_alphas(Alpha_vec)
            print(f"Retrieved Optimal One Parameter Alphas for NR {i+1}")
            opt_OP_alpha.append(self.OP_min_alpha1)
            opt_OP_alpha_ind.append(self.OP_min_alpha1_ind)

            #Get the Optimal Reconstructions and Save

            #Performing DP, GCV, LC conventional solution.
            self.minimize_DP(SNR)
            print(f"Calculated Conventional DP solution for NR {i+1}")
            self.minimize_GCV()
            print(f"Calculated Conventional GCV solution for NR {i+1}")
            self.minimize_LC()
            print(f"Calculated Conventional LC solution for NR {i+1}")

            #######Retrieve Optimal TP Reconstructions
            self.get_f_rec_TP_grid()
            print(f"Retrieved Best Two Parameter Reconstruction for NR {i+1}")
            f_rec_TP_grids.append(self.f_rec_TP_grid)

            # #######Retrieve Optimal OP Reconstructions
            self.get_f_rec_OP_grid()
            print(f"Retrieved Best One Parameter Reconstruction for NR {i+1}")
            f_rec_OP_grids.append(self.f_rec_OP_grid)



            #Get Reconstructions Errors of OP and TP Grid Search against the Ground Truth
            self.get_error()
            print(f"Calculated the Reconstruction Errors for NR {i+1}")
            errors_OP_grid.append(self.com_rec_OP_grid)
            errors_TP_grid.append(self.com_rec_TP_grid)

            gt_dumb = MRR_inst.g
            #Plot Comparison of Reconstructions with GT

            #Run hypersurface
            # print("Starting L_hypersurface")
            # string = "L_hypersurface"
            # file_path = create_result_folder(string)
            # # for i in tqdm(range(num_real)):
            # l_surf = L_hypersurface(T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num = 2)
            
            # # Curvatures = np.zeros((self.nT2,nalpha1,nalpha2))
            # znorms = []
            # xnorms = []
            # partial_first = []
            # p_mats = []
            # determinants = []
            # w_sqrs = []
            # w_m_plus_1s = []
            # curvatures = np.zeros((nalpha1,nalpha2))

            # # optalp1 = self.TP_min_alpha1
            # # optalp2 = self.TP_min_alpha2
            # L_rhos, L_x_lc_vec = l_surf.minimize_obj(Alpha_vec, Alpha_vec2)
            L_rhos = self.TP_rhos
            L_x_lc_vec = self.TP_x_lc_vec
            # print(L_x_lc_vec[:,0,2])


            # x_j_alp(self,B_mats,j,alphas)
            #  z_alp(self, alphas)
            #Append values
            # f_star_alps.append(f_star_alp)
            # f_star_alps[:,i] = f_star_alp

            # # residual_norms.append(residual_norm)
            # residual_norms[i] = residual_norm

            # B_1_constr_norms[i] = B_1_constraint_norm
            # B_2_constr_norms[i] = B_2_constraint_norm

                # both_constraint_norms_list.append(both_constraint_norms)
            # both_constraint_norms = [B_1_constraint_norm, B_2_constraint_norm]
                #Calculate curvature val for a given alpha1 and alpha2
            # print("Starting Curvature")
            # curvatures = np.zeros((len(Alpha_vec), len(Alpha_vec2)))

            # # # Create a single instance of Curvature
            # curvature_instance = Curvature(T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num=2)

            # Compute alphas2 array outside the loop
            # alphas2 = np.array(list(product(Alpha_vec, Alpha_vec2)))
            # test = l_surf.z_alp(alphas2)
            # print("test",test.shape)
            # d_alphas = curvature_instance.d_alphas(l_surf.z_alp, alphas2)
            # jacobian_mat = curvature_instance.jacobian_x(l_surf.x_alp, alphas2)
            # dz_dxs_val = curvature_instance.dz_dxs(d_alphas, jacobian_mat, l_surf.x_alp(alphas2))
            # P_matrix = curvature_instance.get_P_mat(d_alphas, l_surf.z_alp, jacobian_mat, l_surf.x_alp, alphas2)
            # determinant = curvature_instance.eval_determinant(P_matrix)
            # w_sqr = curvature_instance.get_w_sqr(dz_dxs_val)
            # w_m_plus_1 = curvature_instance.get_w_M_plus_1(w_sqr)
            # curvatures = curvature_instance.get_curvature(determinant, w_m_plus_1)
            # const_norms1 = np.zeros(len(Alpha_vec))
            # const_norms2 = np.zeros(len(Alpha_vec2))
            # residual_norms = np.zeros((len(Alpha_vec), len(Alpha_vec2)))

            # for j, k in (product(range(len(Alpha_vec)), range(len(Alpha_vec2)))):
            #     curvature_instance = Curvature(T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num = 2)
            #     alphas2 = np.array([Alpha_vec[j],Alpha_vec2[k]])
            #     # f_star_alp = l_surf.F_star(alphas2)
                # print(f_star_alp)
                # residual_norm = l_surf.get_residual_norm(f_star_alp)  # Assuming this is available in the L_hypersurface instance
                # const_norm1 = l_surf.get_constraint_norm(MRR_inst.B_1, f_star_alp)
                # const_norm2 = l_surf.get_constraint_norm(MRR_inst.B_2, f_star_alp)
                # const_norms1[j] = const_norm1
                # const_norms2[k] = const_norm2
                # residual_norms[j,k] = residual_norm
            
            # curvfig, inds = curvature_instance.get_curvatures(const_norms1, const_norms2, residual_norms)
            # print("Finished curvature plot")
            # print(f"Plotted the Curvature Surface for the Two Parameter Grid Search for NR {i+1}")
            # pio.write_html(curvfig, file= os.path.join(file_path, f"curvature_plot_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}.html"))
            # print(f"Saved Two Parameter Curvature Surface Plot for NR {i+1}")

                # # inv = curvature_instance.inverse(alphas2)
                # d_alphas = curvature_instance.d_alphas(l_surf.z_alp, alphas2)
                # # d_alphas_analyt = [dx,dy] = l_surf.dF_analytic(alphas2[0],alphas2[1]) 
                
                # # print("d_alphas", d_alphas)
                # # print("x grad" , 2 * alphas2[0])
                # # print("y grad", 3 * alphas2[1]**2)
                # # print("d_alphas_analyt", d_alphas_analyt)

                # jacobian_mat = curvature_instance.jacobian_x(l_surf.x_alp, alphas2)
                # # print("jacobian_mat", jacobian_mat)
                # dz_dxs_val = curvature_instance.dz_dxs(d_alphas,jacobian_mat, l_surf.x_alp(alphas2))
                # # print("dz_dxs_val", dz_dxs_val)
                # P_matrix = curvature_instance.get_P_mat(d_alphas, l_surf.z_alp, jacobian_mat, l_surf.x_alp, alphas2)
                # # print("finish p_mat", P_matrix)
                # determinant = curvature_instance.eval_determinant(P_matrix)

                # w_sqr = curvature_instance.get_w_sqr(dz_dxs_val)
                # # print("w_sqr:", w_sqr)

                # w_m_plus_1 = curvature_instance.get_w_M_plus_1(w_sqr)
                # # print("w_m_plus_1:", w_m_plus_1)
                # # w_m_plus_1s[i,j] = w_m_plus_1

                # curvature_val = curvature_instance.get_curvature(determinant, w_m_plus_1)
                # # print("curvature:", curvature)
                # curvatures[j,k] = curvature_val
                # print(f"Finished curvatures[{j},{k}]")
            # #     #Append values
            # #     # all_first_partials_list.append(all_first_partials)
            # #     # p_mat_list.append(p_mat)
            # #     # determinants.append(determinant)
            # #     # w_sqrs.append(w_sqr)
            # #     # w_m_plus_1s.append(w_m_plus_1)
            # #     # curvatures.append(curvature)
            # #     # curvatures[i,j] = curvature
            # #     # print("Finished Curvature Calculatoin")

            # # # #SAVE Loop results for a given noise realization
            # # # f_star_alps_NR.append(f_star_alps)
            # # # residual_norms_NR.append(residual_norms) 
            # # # B_1_constr_norms_NR.append(B_1_constr_norms) 
            # # # B_2_constr_norms_NR.append(B_2_constr_norms) 
            # # # first_partial_deriv_B1s_NR.append(first_partial_deriv_B1s) 
            # # # second_partial_deriv_B2s_NR.append(first_partial_deriv_B2s)
            # # # pmats_NR.append(p_mats)
            # # # determinants_NR.append(determinants) 
            # # # w_sqrs_NR.append(w_sqrs)
            # # # w_m_plus_1s_NR.append(w_m_plus_1s)
            # # curvatures_NR.append(curvatures)

            # # Plot curvature plot
            # plot_instance = Plot(T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num = 2)
            # curvature_condition = True
            # print("curvature", curvatures)
            # resid_cond = False
            # curvature_plot = plot_instance.get_plot(Alpha_vec, Alpha_vec2, curvatures, curvature_condition, resid_cond,  i+1)
            # print("Finished curvature plot")
            # print(f"Plotted the Curvature Surface for the Two Parameter Grid Search for NR {i+1}")
            # pio.write_html(plot_instance.curvature_plot, file= os.path.join(file_path, f"curvature_plot_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}.html"))
            # print(f"Saved Two Parameter Curvature Surface Plot for NR {i+1}")

            # #Plot error_norm plot
            # curvature_condition = False
            # resid_cond = False
            # error_plot = plot_instance.get_plot(Alpha_vec, Alpha_vec2, L_rhos, curvature_condition, resid_cond, i+1)
            # print(f"Plotted the Error Norm Surface for the Two Parameter Grid Search for NR {i+1}")
            # pio.write_html(error_plot, file= os.path.join(file_path, f"error_norm_plot_NR{i+1}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}.html"))
            # print(f"Saved Two Parameter Error Norm Surface Plot for NR {i+1}")

            # # #Get curvative minimum alphas

            # print("Starting to Select Curvature Alphas")
            # select_alpha_instance = Select_Alpha(T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num = 2)
            # curvature_condition = True
            # min_index, min_x, min_y, _, _  = select_alpha_instance.get_minimum_params(Alpha_vec, Alpha_vec2,curvatures, curvature_condition)

            # #Append values
            # curvature_min_inds.append(min_index)
            # curvature_min_alp1.append(min_x)
            # curvature_min_alp2.append(min_y)

            # print("Finished Select Curvature Alphas")

            # print("Starting to Select Error Norm Alphas")

            #Get error norm minimum alphas
            # curvature_condition = False
            # min_index, min_x, min_y, _, _  =select_alpha_instance.get_minimum_params(Alpha_vec, Alpha_vec2,l_surf.L_rhos, curvature_condition)

            # #Append values
            # error_norm_min_inds.append(min_index)
            # error_norm_min_alp1.append(min_x)
            # error_norm_min_alp2.append(min_y)

            # print("Finished Select Error Norm Alphas")

            # print("Starting Metric calculations")
            # # #Get the right metrics based on error_norm
            # metric_instance = Metrics(T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num = 2)
            # # print(min_index)
            # # f_star_alphaL = metric_instance.get_f_star_alphaL(inds[0], inds[1], L_x_lc_vec)
            # f_star_alphaL = metric_instance.get_f_star_alphaL(min_index[0], min_index[1], L_x_lc_vec)
            # # print("f_star_alphaL shape", f_star_alphaL.shape)
            # # self.TP_HS_alp1_ind = inds[0]
            # # self.TP_HS_alp2_ind = inds[1]
            # self.TP_HS_alp1_ind = min_index[0]
            # self.TP_HS_alp2_ind = min_index[1]
            # self.f_rec_TP_HS = f_star_alphaL
            # self.com_rec_HS_TP = norm(gt_dumb - self.f_rec_TP_HS)

            # print("L_rhos shape", l_surf.L_rhos.shape)
            # print("g.shape", self.g.shape)
            # efficiency_val = metric_instance.get_efficiency(l_surf.L_rhos, self.g, f_star_alphaL)

        
            # #Append values
            # f_star_alphaLs.append(f_star_alphaL)
            # efficiency_vals.append(efficiency_val)
            compare_val = True
            self.plot_gt_and_comparison(file_path, date, compare_val, i)
            print(f"Plotted the Comparison Plots for NR {i+1}")
            print("Finished Metric calculations")
            print("Finished L_hypersurface\n")

        #Plot Just GT
        string = "2D_grid_search"
        file_path = create_result_folder(string)

        compare_val = False
        self.plot_gt_and_comparison(file_path, date, compare_val, num_real)
        print(f"Plotted the Ground Truth")

        #Save the TP optimal alphas and their indices
        np.save(os.path.join(file_path, f"opt_TP_alphas_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"), opt_TP_alphas)
        print(f"Saved opt_TP_alphas") 
        np.save(os.path.join(file_path, f"opt_TP_alphas_ind_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"),opt_TP_alphas_ind)
        print(f"Saved opt_TP_alphas_ind") 
        np.save(os.path.join(file_path, f"opt_TP_alphas_ind_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"), opt_TP_DP_alphas_ind)
        print(f"Saved opt_TP_DP_alphas_ind") 


        # #Save the OP optimal alphas and their indices
        np.save(os.path.join(file_path, f"opt_OP_alphas_{date}_{self.nalpha1}_alpha1_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}"),opt_OP_alpha)
        print(f"Saved opt_alp_OPs") 
        np.save(os.path.join(file_path, f"opt_OP_alphas_ind_{date}_{self.nalpha1}_alpha1_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}"),opt_OP_alpha_ind)
        print(f"Saved opt_alp_OP_inds") 

        #Save the Reconstructions
        np.save(os.path.join(file_path, f"f_rec_TP_grids_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"),f_rec_TP_grids)
        print(f"Saved f_rec_TP_grids") 
        np.save(os.path.join(file_path, f"f_rec_OP_grids_{date}_{self.nalpha1}_alpha1_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}"),f_rec_OP_grids)
        print(f"Saved f_rec_OP_grids") 

        # #Save the Error between TP and OP Reconstructions and Ground Truth
        np.save(os.path.join(file_path, f"errors_TP_grid_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"),errors_TP_grid)
        print(f"Saved errors_TP_grid") 
        np.save(os.path.join(file_path, f"errors_OP_grid_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"),errors_OP_grid)
        print(f"Saved errors_OP_grid") 

        #Save Hypersurface values
        string = "L_hypersurface"
        file_path = create_result_folder(string)

        var_list = [f_star_alps_NR, residual_norms_NR, B_1_constr_norms_NR, B_2_constr_norms_NR, first_partial_deriv_B1s_NR, second_partial_deriv_B2s_NR, pmats_NR,
                      determinants_NR, w_sqrs_NR, w_m_plus_1s_NR, curvatures_NR]
        for var in var_list:
            np.save(os.path.join(file_path, f"{var}_{date}_{self.nalpha1}_alpha1_{self.nalpha2}_alpha2_discretizations_alpha1log_{self.alp_1_lb}_{self.alp_1_ub}_alpha2log_{self.alp_2_lb}_{self.alp_2_ub}"), var)
            print(f"Saved {var}") 

### 2 Parameter L_hypersurface 

class L_hypersurface(MRR_grid_search):
    #naming conventions is based on "Simultaneous multiple regularization parameter selection by 
    #means of the L-hypersurface with applications to linear inverse problems posed in the wavelet transform domain" by Berge, Kilmer, and Miller

    def __init__(self, T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num):
        # MRR_grid_search.__init__(self, T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub)
        self.M = param_num
        self.f_star_alp = None 
        self.residual_norm = None
        self.constraint_norm = None
        self.both_constraint_norms = None
        self.param_list = np.zeros(self.M)
        self.nalpha1 =MRR_inst.nalpha1
        self.nalpha2 =MRR_inst.nalpha2
        self.TP_rhos = MRR_inst.TP_rhos
        self.g = MRR_inst.g
        self.G = MRR_inst.G
        self.B_1 = MRR_inst.B_1
        self.B_2 = MRR_inst.B_2
        self.data_noisy = MRR_inst.data_noisy
        self.nT2 = MRR_inst.nT2
        self.L_rhos = None
        self.L_x_lc_vec =None
        self.L_residual =None

    # def get_f_star(self, alpha1_ind, alpha2_ind):
    #     # self.minimize_TP_objective(Alpha_vec, Alpha_vec2)
    #     #Take the minimized TP objective value
    #     self.f_star_alp = MRR_inst.TP_x_lc_vec[:, alpha1_ind, alpha2_ind]
    #     f_star_alp = MRR_inst.TP_x_lc_vec[:, alpha1_ind, alpha2_ind]
    #     print("f_star_alp", f_star_alp)
    #     print("f_star_alp.shape", f_star_alp.shape)
    #     print("B_1.shape",  MRR_inst.B_1.shape)

    #     return f_star_alp
    def minimize_obj(self, Alpha_vec, Alpha_vec2):
        L_x_lc_vec = np.zeros((self.nT2, len(Alpha_vec), len(Alpha_vec2)))
        L_rhos = np.zeros((len(Alpha_vec), len(Alpha_vec2)))
        L_residual = np.zeros((len(Alpha_vec), len(Alpha_vec2)))
        for j, k in (product(range(len(Alpha_vec)), range(len(Alpha_vec2)))):
            inv = np.linalg.inv(self.G.T @ self.G + Alpha_vec[j] * self.B_1.T @ self.B_1 + Alpha_vec2[k] * self.B_2.T @ self.B_2)
            exp =  inv @ self.G.T @ self.data_noisy
            L_x_lc_vec[:,j,k] = exp
            # print("L_x_lc_vec[:,j,k]", L_x_lc_vec[:,j,k])
            L_rhos[j,k] = norm(L_x_lc_vec[:, j, k] - self.g, 2)**2
            L_residual[j,k] = norm(self.G @ L_x_lc_vec[:, j, k] - self.data_noisy)
        self.L_rhos = L_rhos
        self.L_residual = L_residual
        self.L_x_lc_vec = L_x_lc_vec
        return L_rhos, L_x_lc_vec

    def get_f_star(self, L_rhos):
        TP_log_err_norm = np.log10(L_rhos)
        # min_indices = np.where(Z == np.min(Z))
        min_index = np.unravel_index(np.argmin(TP_log_err_norm), TP_log_err_norm.shape)
        min_x = Alpha_vec[min_index[0]]
        min_y = Alpha_vec2[min_index[1]]
        # self.minimize_TP_objective(Alpha_vec, Alpha_vec2)
        #Take the minimized TP objective value
        self.f_star_alp = L_rhos[:, min_x, min_y]
        f_star_alp = L_rhos[:, min_x, min_y]
        # print("f_star_alp", f_star_alp)
        # print("f_star_alp.shape", f_star_alp.shape)
        # print("B_1.shape",  MRR_inst.B_1.shape)
        return f_star_alp

    def F_star(self, alphas):
        exp = np.linalg.inv(self.G.T @ self.G + alphas[0] * self.B_1.T @ self.B_1 + alphas[1] * self.B_2.T @ self.B_2) @ self.G.T @ self.data_noisy
        return exp

    def inverse(self, alphas):
        import autograd.numpy as np
        import autograd.numpy.linalg as la
        inv = (la.inv(self.G.T @ self.G + (alphas[0]) * self.B_1.T @ self.B_1 + (alphas[1]) * self.B_2.T @ self.B_2) @ self.G.T @ self.data_noisy)
        # inv = alphas[0]**2 + alphas[1]**3
        #cosine(x)*cosine(y); see if that 
        return inv
    def z_alp(self,alphas):
        import autograd.numpy as np
        import autograd.numpy.linalg as la
        inv = self.inverse(alphas)
        # term1 = self.data_noisy - self.G @ (la.inv(self.G.T @ self.G + alphas[0]**2 * self.B_1.T @ self.B_1 + (alphas[1])**2 * self.B_2.T @ self.B_2) @ self.G.T @ self.data_noisy)
        # norm_squared = np.linalg.norm(term1, axis=1)**2
        # log_norm_squared = np.log10(norm_squared)
        # return log_norm_squared
        # inv = (la.inv(self.G.T @ self.G + (alphas[0])**2 * self.B_1.T @ self.B_1 + (alphas[1])**2 * self.B_2.T @ self.B_2) @ self.G.T @ self.data_noisy)
        # residual_norm  = np.log10(la.norm(MRR_inst.data_noisy - MRR_inst.G @ inv , 2)**2)
        # self.residual_norm = residual_norm
        return np.log10(la.norm(self.data_noisy - self.G @ inv, 2)**2)
        # return inv

    def x_alp(self,alphas):
        import autograd.numpy as np
        import autograd.numpy.linalg as la
        inv = self.inverse(alphas)
        x1 = np.log10(la.norm(MRR_inst.B_1 @ inv, 2)**2)
        x2 = np.log10(la.norm(MRR_inst.B_2 @ inv, 2)**2)
        return np.array([x1,x2])
    # def x_1_alp(self,alphas):
    #     import autograd.numpy as np
    #     import autograd.numpy.linalg as la
    #     return np.log10(norm(MRR_inst.B_1 @ ((la.inv(self.G.T @ self.G + (alphas[0])**2 * self.B_1.T @ self.B_1 + (alphas[1])**2 * self.B_2.T @ self.B_2) @ self.G.T @ self.data_noisy)), 2)**2)

    # def x_2_alp(self,alphas):
    #     import autograd.numpy as np
    #     import autograd.numpy.linalg as la
    #     return np.log10(norm(MRR_inst.B_2 @ (la.inv(self.G.T @ self.G + (alphas[0])**2 * self.B_1.T @ self.B_1 + (alphas[1])**2 * self.B_2.T @ self.B_2) @ self.G.T @ self.data_noisy), 2)**2)


    # def dF_analytic(self,x,y):
    #     '''Analytic derivative (to obtain the exact value)'''
    #     Q = self.G.T @ self.G + x**2 * self.B_1.T @ self.B_1 + y**2 * self.B_2.T @ self.B_2
    #     dQ_dx = - np.linalg.inv(Q) @ (2*x*self.B_1.T @ self.B_1) @ np.linalg.inv(Q) @ self.G.T @ self.data_noisy 
    #     dQ_dy = - np.linalg.inv(Q) @ (2*y*self.B_2.T @ self.B_2) @ np.linalg.inv(Q) @ self.G.T @ self.data_noisy 
    #     return dQ_dx, dQ_dy

    # def x_j_alp(self,B_mats,j,alphas):
    #     constraint_norm = np.log10(norm(B_mats[j-1] @ self.F_star(alphas), 2)**2)
    #     self.constraint_norm = constraint_norm
    #     return constraint_norm
    
    def get_residual_norm(self, f_star_alp):
        residual_norm  = np.log10(norm(MRR_inst.data_noisy - MRR_inst.G @ f_star_alp, 2)**2)
        self.residual_norm = residual_norm
        return residual_norm
    
    def get_constraint_norm(self, B_mat, f_star_alp):
        constraint_norm = np.log10(norm(B_mat @ f_star_alp, 2)**2)
        self.constraint_norm = constraint_norm
        return constraint_norm
    
    # def get_both_constraint_norms(self, f_star_alp):
    #     both_constraint_norms = [self.get_constraint_norm(MRR_inst.B_1, f_star_alp), self.get_constraint_norm(MRR_inst.B_2, f_star_alp)]
    #     self.both_constraint_norms = both_constraint_norms
    #     return both_constraint_norms

class Curvature(L_hypersurface):
    def __init__(self, T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num):
        # L_hypersurface.__init__(self, T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num)
        # self.constraint_norms = self.both_constraint_norms
        self.determinant = None
        self.all_first_partials = None
        self.P_mat = None
        self.w_sqr = None
        self.curvature = None
        self.g = MRR_inst.g
        self.G = MRR_inst.G
        self.B_1 = MRR_inst.B_1
        self.B_2 = MRR_inst.B_2
        self.data_noisy = MRR_inst.data_noisy
        self.nT2 = MRR_inst.nT2
        self.M = param_num
    
    def F_star(self, alphas):
        #solved using the method of finite differences
        F = np.linalg.inv(self.G.T @ self.G + (alphas[0]) * self.B_1.T @ self.B_1 + (alphas[1]) * self.B_2.T @ self.B_2) @ self.G.T @ self.data_noisy
        return F

    # def d_alphas(self,znorm_func,alphas):
    #     """This finds the gradient operator for znorm_func"""
    #     grad_alphas = elementwise_grad(znorm_func)
    #     # print("znorms", znorm_func)
    #     d_alphas = grad_alphas(alphas) 
    #     # print("d_alphas", d_alphas)
    #     return d_alphas

    def d_alphas(self,alphas):
        Ainv = self.inverse(alphas)
        dfda1 = Ainv @ (self.B_1 @ self.F_star(alphas))
        dfda2 = Ainv @ (self.B_2 @ self.F_star(alphas))
        arr = [dfda1,dfda2]
        return arr
    
    def dx_dalp(self,xnorms,alphas):
        u1 = (self.B_1 @ self.F_star(alphas))
        u2 = (self.B_2 @ self.F_star(alphas))
        Ainv = self.inverse(alphas)
        dx1da1 = 2 * np.exp(-xnorms[0]) * ((self.B_1 @ self.F_star(alphas)) @ self.B_1 @ (Ainv @ u1))
        dx1da2 = 2 * np.exp(-xnorms[0]) * ((self.B_1 @ self.F_star(alphas)) @ self.B_1 @ (Ainv @ u2))
        dx2da1 = 2 * np.exp(-xnorms[1]) * ((self.B_2 @ self.F_star(alphas)) @ self.B_2 @ (Ainv @ u1))
        dx2da2 = 2 * np.exp(-xnorms[1]) * ((self.B_2 @ self.F_star(alphas)) @ self.B_2 @ (Ainv @ u2))
        jac = np.array([[dx1da1, dx1da2], [dx2da1, dx2da2]])
        return jac

    def hessian(self,xnorms,alphas):
        u1 = (self.B_1 @ self.F_star(alphas))
        u2 = (self.B_2 @ self.F_star(alphas))
        Ainv = self.inverse(alphas)
        # dx1da1 = 2 * np.exp(-xnorms[0]) * ((self.B_1 @ self.F_star(alphas)) @ self.B_1 @ (Ainv @ u1))
        dfda1da2 = -1* Ainv @ (self.B_1 @ (Ainv @ u2) + self.B_2 @ (Ainv @ u1))
        dfda2da1 = -1* Ainv @ (self.B_1 @ (Ainv @ u1) + self.B_2 @ (Ainv @ u2))
        dzdx1dx2
        dzdx2dx1
        dzdx1dx1
        dzdx2dx2
        jac = np.array([[dx1da1, dx1da2], [dx2da1, dx2da2]])
        return jac

    # def jacobian_x(self,xnorms,alphas):
    #     """This finds the Jacobian matrix for xnorms"""
    #     jacsx = jacobian(xnorms)
    #     # print("xnorm", xnorms)
    #     jac_mat = jacsx(alphas)
    #     # print("jac_mat:", jac_mat)
    #     return jac_mat

    def e_i(self,i, n):
        """Returns the unit vector with 1 at the ith position given length n"""
        return np.eye(n)[i-1]

    def dz_dxs(self,dalphas,jacobian, xnorms):
        import autograd.numpy as np
        import autograd.numpy.linalg as la
        """Returns partial z given each partial xnorm """
        M = len(xnorms)
        # print(M)
        dz_dxs = np.zeros(M)
        # print("la.inv(jacobian)", jacobian.shape)
        for i in range(M):
            dz_dxs[i] = dalphas.T @ la.inv(jacobian) @ self.e_i(i,M)
        return dz_dxs

    def get_P_mat(self,dalphas,znorm_func,jacobian, xnorms_funcs, alphas):
        import autograd.numpy as np
        import autograd.numpy.linalg as la
        """Returns hessian matrix given a value/values given to it """
        def hessian_mat(vals,alphas):
            hessians = hessian(vals)
            hessian_mat = hessians(alphas)
            return hessian_mat

        # print("printing mat")
        x_mat = hessian_mat(xnorms_funcs,alphas)
        # print("hessian xnorms", mat)
        # print("hessian xnorms", mat.shape)
        z_mat = hessian_mat(znorm_func,alphas)
        # print("hessian znorms", z_mat)
        M = len(alphas)
        dz_dxi_dxj = np.zeros((M,M))
        jac_inv = la.inv(jacobian)

        arr = []
        for i in range(M):
            matrix = jac_inv.T @ x_mat[i] 
            arr.append(matrix)
        print(arr)

        # print("Arr", arr)
        # print("Arr shape", len(arr))

        arr = np.array(arr)
        for i in range(M):
            for j in range(M):
                dz_dxi_dxj[i,j] = self.e_i(j,M).T @ jac_inv.T @ hessian_mat(znorm_func,alphas) @ jac_inv @ self.e_i(i,M) - dalphas.T @ jac_inv @ (self.e_i(j,M).T @ arr) @ jac_inv @ self.e_i(i,M)

        P_mat = dz_dxi_dxj
        return P_mat

    # def min_dist_point(self,``)

    def eval_determinant(self, P_mat):
        determinant = np.linalg.det(P_mat)
        self.determinant = determinant
        return determinant

    def get_w_sqr(self, dz_dxs):
        w_sqr = 1 + np.sum((dz_dxs)**2)
        self.w_sqr = w_sqr
        return w_sqr
    
    def get_w_M_plus_1(self, w_sqr):
        power_factor = (self.M + 1) // 2 
        if self.M % 2 == 0:
            w = math.sqrt(w_sqr)
            w_M_plus_1 = w_sqr**power_factor * w
        else:
            w_M_plus_1 = w_sqr**power_factor
        # w_M_plus_1 = w_sqr**2
        self.w_M_plus_1 = w_M_plus_1
        return w_M_plus_1

    def get_curvature(self, determinant, w_M_plus_1):
        curvature = ((-1)**self.M) * determinant / w_M_plus_1
        print("M",self.M)
        print("curvature", curvature)
        print("determinant",determinant)
        print("w_M_plus_1",w_M_plus_1 )
        self.curvature = curvature 
        return curvature

    # def get_curvatures(self, cons_norm1, cons_norm2, resid_norm):    # Example surface data
    #     import numpy as np
    #     x, y = np.meshgrid(cons_norm1, cons_norm2)
    #     z = resid_norm
    #     # Function to compute gradients
    #     def compute_gradients(z, dx, dy):
    #         # First derivatives
    #         dzdx = np.gradient(z, axis=1) / dx
    #         dzdy = np.gradient(z, axis=0) / dy

    #         # Second derivatives
    #         d2zdx2 = np.gradient(dzdx, axis=1) / dx
    #         d2zdy2 = np.gradient(dzdy, axis=0) / dy
    #         d2zdxdy = np.gradient(dzdx, axis=0) / dy

    #         return dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy

    #     # Compute gradients
    #     dx = x[0, 1] - x[0, 0]
        
    #     dy = y[1, 0] - y[0, 0]
    #     dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy = compute_gradients(z, dx, dy)

    #     # Mean curvature (H) and Gaussian curvature (K)
    #     H = 0.5 * (d2zdx2 * (1 + dzdy**2) - 2 * dzdx * dzdy * d2zdxdy + d2zdy2 * (1 + dzdx**2)) / (1 + dzdx**2 + dzdy**2)**1.5
    #     K = (d2zdx2 * d2zdy2 - d2zdxdy**2) / (1 + dzdx**2 + dzdy**2)**2

    #     # For visualization, you might use matplotlib
    #     import plotly.graph_objects as go
    #     # Assuming you have defined Alpha_vec, Alpha_vec2, and K
    #     # If not, please define them before proceeding

    #     Vec1 = np.log10(Alpha_vec)
    #     Vec2 = np.log10(Alpha_vec2)
    #     x, y = np.meshgrid(Vec1,Vec2)

    #     fig = go.Figure(data=[go.Surface(z=K.T, x=x, y=y)])
    #     fig.update_layout(scene=dict(xaxis_title='log10(Alpha_vec)',
    #                                 yaxis_title='log10(Alpha_vec2)',
    #                                 zaxis_title='Curvature'))
        
    #     max_index = np.unravel_index(np.argmax(K), K.shape)
    #     z_val_title = "Point of Maximum Curvature"
    #     min_x = Vec1[max_index[0]]
    #     min_y = Vec2[max_index[1]]
    #     min_z = np.max(K)

    #     # Add a scatter plot for the minimum point
    #     fig.add_trace(go.Scatter3d(
    #         x=np.array([min_x]),
    #         y=np.array([min_y]),
    #         z=np.array([min_z]),
    #         mode='markers',
    #         marker=dict(
    #             size=5,
    #             color='gold',
    #             symbol='circle'
    #         ),
    #         name= z_val_title
    #     ))
    #     ind = [max_index[0],max_index[1]]

    #     # plt.figure()
    #     # plt.title("Mean Curvature")
    #     # plt.contourf(x, y, H, cmap='viridis')
    #     # plt.colorbar()

    #     # plt.figure()
    #     # plt.title("Gaussian Curvature")
    #     # plt.contourf(x, y, K, cmap='viridis')
    #     # plt.colorbar()
    #     return fig,ind


class Plot(L_hypersurface):
    def __init__(self,T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num):
        # MRR_inst.__init__(T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub)
        self.alp_1_lb = MRR_inst.alp_1_lb
        self.alp_1_ub = MRR_inst.alp_1_ub
        self.alp_2_lb = MRR_inst.alp_2_lb
        self.alp_2_ub = MRR_inst.alp_2_ub
        self.curvature_plot = None
        self.errornorm_plot = None
        self.histogram = None
        # self.alp_1_lb = MRR_inst.
    
    def get_plot(self, Alpha_vec, Alpha_vec2, TP_zval, curvature_cond, resid_cond, iter_num):
        if curvature_cond == True:
            fig = self.plot_TP_surface(Alpha_vec, Alpha_vec2, TP_zval,curvature_cond, resid_cond,iter_num)
            self.curvature_plot = fig
        else:
            fig = self.plot_TP_surface(Alpha_vec, Alpha_vec2, TP_zval,curvature_cond, resid_cond,iter_num)
            self.errornorm_plot = fig
        return fig

    def get_histogram(self, efficiencies):
        fig = plt.figure(12,6)
        sns.histplot(efficiencies, bins=30, kde=True, color='lightgreen', edgecolor='red')
        plt.title("Efficiency Histogram")
        plt.xlabel("Efficiencies", fontsize = 12)
        plt.ylabel("Count", fontsize =12 )
        self.histogram = fig
        return fig

class Select_Alpha(L_hypersurface):
    def __init__(self,T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num):
        self.L_min_x = None
        self.L_min_y = None
        self.L_min_z = None
        self.L_min_ind = None
        self.L_TP_zval = None

    def get_minimum_params(self, Alpha_vec, Alpha_vec2, z_val, curvature_cond):
        plot1, plot2 = np.meshgrid(Alpha_vec, Alpha_vec2)
        if curvature_cond == False:
            TP_zval = TP_log_err_norm = np.log10(z_val)
        else:
            TP_zval = TP_curvature = z_val

        self.L_min_index = min_index = np.unravel_index(np.argmin(TP_zval), TP_zval.shape)
        self.L_min_x = min_x = Alpha_vec[min_index[0]]
        self.L_min_y = min_y = Alpha_vec2[min_index[1]]
        self.L_min_z = min_z = np.min(TP_zval)
        self.L_TP_zval = TP_zval
        return min_index, min_x, min_y, min_z, TP_zval

class Metrics(L_hypersurface):
    def __init__(self,T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub, param_num):
        # MRR_inst.__init__(T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_2_ub)
        self.efficiency = None
        self.error_norm = None
        self.f_star_alphaL = None
        self.g = MRR_inst.g

    def get_f_star_alphaL(self, L_min_x, L_min_y, TP_x_lc_vec):
        alpha1_ind = L_min_x
        alpha2_ind = L_min_y
        self.f_star_alphaL = f_star_alphaL = TP_x_lc_vec[:, L_min_x, L_min_y]
        return f_star_alphaL
    
    def get_efficiency(self, com_rec_TP_grid, gt, f_star_alphaL):
        efficiency = (com_rec_TP_grid)/(norm(f_star_alphaL - gt, 2))
        self.efficiency = efficiency
        return efficiency


def create_result_folder(string):
    # Create a folder based on the current date and time
    date = datetime.now().strftime("%Y%m%d")
    folder_name = f"/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/{string}_{date}_Run"
    # folder_name = f"/Volumes/Lexar/NIH/Experiments/GridSearch/{string}_{date}_Run"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return folder_name


#############################################################################################
#Initial Parameters
if __name__ == '__main__':
    
    print("Running 2D_multi_reg_MRR_0116.py script")
    # file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/"
    file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/"
    # Get today's date
    today_date = datetime.today()
    # Format the date as "month_day_lasttwodigits"
    date = today_date.strftime("%m_%d_%y")

    T2 = np.arange(1,201)
    TE = np.arange(1,512,4)
    nalpha1 = 100
    nalpha2 = 100
    alp_1_lb = -9
    alp_1_ub = -1
    alp_2_lb = -9
    alp_2_ub = -1
    SNR = 200

    #-4 to -1 for line of solutions
    mu1 = 40
    mu2 = 150
    sigma1 = 4
    sigma2 = 20
    # sigma1 = 4
    # sigma2 = 15
    MRR_inst = MRR_grid_search(T2, TE, SNR, nalpha1, nalpha2, mu1, mu2, sigma1, sigma2, alp_1_lb, alp_1_ub, alp_2_lb, alp_1_ub)
    MRR_inst.get_G()
    MRR_inst.get_g()
    MRR_inst.get_diag_matrices()
    MRR_inst.decompose_G()
    Alpha_vec = np.logspace(alp_1_lb, alp_1_ub, nalpha1)
    Alpha_vec2 = np.logspace(alp_2_lb, alp_2_ub, nalpha2)
    num_real=3
    curvature_cond = False
    MRR_inst.run_all(Alpha_vec, Alpha_vec2, num_real, date)
    print("Finished 2D_multi_reg_MRR_0116.py  script")