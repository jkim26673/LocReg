#multi_reg_garbage_code
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
from src.utils.load_imports.load_classical import *

# from regu.Lcurve import Lcurve
# from regu.gravity import gravity
# from regu.baart import baart
# from regu.heat import heat
# from regu.csvd import csvd
# from regu.l_curve import l_curve
# from regu.tikhonov import tikhonov
# from regu.gcv import gcv
# from regu.discrep import discrep
# from Utilities_functions.LocReg_unconstrained import LocReg_unconstrained
# from Utilities_functions.TwoParam_LR import Multi_Param_LR
# from regu.shaw import shaw
# from regu.tikhonov_multi_param import tikhonov_multi_param
# from regu.i_laplace import i_laplace
# from lsqnonneg import lsqnonneg
# #from Simulations.lcurve_functions import l_cuve,csvd,l_corner
# from Simulations.l_curve_corner import l_curve_corner
# from regu.csvd import csvd
#Find the relative error:
def min_max(arr):
    minimum = np.min(arr)
    maximum = np.max(arr)
    mean = np.mean(arr)
    median = np.median(arr)
    std_dev = np.std(arr)
    return mean, median, std_dev


def minimize(T2, Lambda_vec, Lambda_vec2, nalpha1, nalpha2, B_1, B_2, G, g):
    x_lc_vec = np.zeros((len(T2), len(Lambda_vec), len(Lambda_vec2)))
    rhos = np.zeros((len(Lambda_vec), len(Lambda_vec2)))
    error = np.zeros((nalpha1**2, len(Lambda_vec), len(Lambda_vec2)))

    for j, k in (product(range(len(Lambda_vec)), range(len(Lambda_vec2)))):
        x = cp.Variable(nT2)
        # objective = cp.Minimize(cp.norm(G.T @ G @ x - G.T @ data_noisy, 2) +
        #                         Lambda_vec[j]**2 * cp.norm(B_1 @ x, 2)**2 +
        #                         Lambda_vec2[k]**2 * cp.norm(B_2 @ x, 2)**2)
        objective = cp.Minimize(cp.norm(G @ x - data_noisy, 2)**2 +
                                Lambda_vec[j]**2 * cp.norm(B_1 @ x, 2)**2 +
                                Lambda_vec2[k]**2 * cp.norm(B_2 @ x, 2)**2)
        problem = cp.Problem(objective)
        problem.solve(solver=cp.SCS, verbose=False)

        x_lc_vec[:, j, k] = x.value.flatten()

        rhos[j,k] = norm(x_lc_vec[:,j,k] - g,2)**2

    return x_lc_vec, rhos

def plot(lam_vec1, lam_vec2, rhos, iter_num):
    plot1 = lam_vec1
    plot2 = lam_vec2
    plot1, plot2 = np.meshgrid(plot1, plot1)
    Z = np.log((rhos))
    sb.set_style('whitegrid')

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')

    surface = axes.plot_surface(plot1, plot2, Z, rstride=1, cstride=1, cmap = "inferno")

    axes.set_xlabel('Alpha_1 Values')
    axes.set_ylabel('Alpha_2 Values')
    axes.set_zlabel('log(norm(p_a1,a2 - p_true)**2)')

    colorbar = fig.colorbar(surface, ax=axes, shrink=0.5, aspect= 20)
    plt.title(f"Surface of Grid Search for NR {iter_num}")
    return fig

def G(TE,T2):
    G = np.zeros((len(TE),len(T2)))
#For every column in each row, fill in the e^(-TE(i))
    for i in range(len(TE)):
        for j in range(len(T2)):
            G[i,j] = np.exp(-TE[i]/T2[j])
    nTE = len(TE)
    return G

mu1 = 40
mu2 = 150
sigma1 = 4
sigma2 = 12

def g(mu1,mu2,sigma1,sigma2):
    g = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
    g = g + (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
    g = g/2
    return g

def diag_matrices(T2):
    nT2 = len(T2)
    half_nT2 = int(len(T2)/2)
    half_mat = np.eye(half_nT2)
    L_1 = np.zeros((nT2, nT2))
    L_2 = np.zeros((nT2, nT2))
    L_1[:half_nT2, :half_nT2] = half_mat
    L_2[half_nT2:, half_nT2:] = half_mat
    B_1 = L_1
    B_2 = L_2
    return B_1,B_2



file_path = ""

def minimize_and_plot(T2, Lambda_vec, Lambda_vec2, nalpha1, nalpha2, B_1, B_2, rhos, G, g, num_real, file_path):
    for i in tqdm(range(num_real)):
        x_lc_vec,rhos = minimize(T2, Lambda_vec, Lambda_vec2, nalpha1, nalpha2, B_1, B_2, G, g)
        fig = plot(Lambda_vec, Lambda_vec2, rhos, i+1)
        #Save figure, save x_lc_vec, save rhos
    

#Generate the TE values/ time
#Generate the T2 values

Lambda_vec = np.logspace(-10,5,40)
nLambda = len(Lambda_vec)

alpha_1 = 0.2
alpha_2 = 0.2
beta = 0.1
c = 1
ep = 1e-8
omega = 0.5
delta = 0.01 * norm(data_noiseless)
nrun = 100

B_1 = (D.T @ D)**0.5
B_2 = (D_tilde.T @ D_tilde)**0.5

B_1[np.isnan(B_1)] = -1
B_2[np.isnan(B_2)] = 1


data_noisy = data_noiseless + delta * noise 

rel_error_beta = np.zeros(nrun)
rel_error_alpha1 = np.zeros(nrun)
rel_error_alpha2 = np.zeros(nrun)
rel_error_three_param = np.zeros(nrun)
final_alpha_1s = np.zeros(nrun)
final_alpha_2s = np.zeros(nrun)
final_betas = np.zeros(nrun)
final_xdelta = np.zeros((nrun,len(TE)))
for i in range(nrun):
    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    data_noisy = data_noiseless + noise
    # _, _, _, _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, 0, beta, omega, ep, c, delta)
    # rel_error_beta[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, 0, 0, omega, ep, c, delta)
    # rel_error_alpha1[i] = rel_error
    # _, _, _,  _, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, 0, alpha_2, 0, omega, ep, c, delta)
    # rel_error_alpha2[i] = rel_error
    final_alpha_1, final_alpha_2, final_beta,  x_delta, rel_error = Multi_Param_LR(data_noisy, G, g, B_1, B_2, alpha_1, alpha_2, beta, omega, ep, c, delta)
    final_alpha_1s[i] = final_alpha_1
    final_alpha_2s[i] = final_alpha_2
    final_betas[i] = final_beta
    final_xdelta[i] = x_delta
    rel_error_three_param[i] = rel_error


test_alpha1 = min_max(final_alpha_1s)[0]
test_alpha2 = min_max(final_alpha_2s)[0]
test_beta = min_max(final_betas)[0]
test_alpha1
test_alpha2
test_beta
test = np.linalg.inv(G.T @ G + test_alpha1 * B_1.T @ B_1 + test_alpha2 * B_2.T @ B_2 + test_beta * np.eye(data_noisy.shape[0])) @ G.T @ data_noisy

np.linalg.cond(G.T @ G + test_alpha1 * (B_1.T @ B_1)**2 + test_alpha2 * (B_2.T @ B_2)**2 + test_beta * np.eye(data_noisy.shape[0]))

Josh_rel_error = norm(g - test)/norm(g)
Josh_rel_error
plt.plot(g)
plt.plot(test)
plt.plot(x_delta)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.legend(['Ground Truth', 'Replicated Version', 'Lu Paper'])
plt.title("Inverse Laplace Problem 1% Noise")
plt.show()

plt.plot(g)
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.title("Inverse Laplace Problem Ground Truth")
plt.show()

#Get the parameters for the Replicated Results
min_max(final_alpha_1s)
min_max(final_alpha_2s)
min_max(final_betas)
min_max(rel_error_three_param)


U, s, V = csvd(G, tst = None, nargin = 1, nargout = 3)
s = np.diag(s)

plt.plot(g)
plt.title("Ground Truth")
plt.xlabel("T2")
plt.ylabel("Amplitude")
plt.show()

    # def plot_grid_search(self,Lambda_vec,Lambda_vec2, iter_num):
    #     plot1 = Lambda_vec
    #     plot2 = Lambda_vec2
    #     plot1, plot2 = np.meshgrid(plot1, plot1)
    #     Z = np.log(self.rhos)
    #     sb.set_style('whitegrid')

    #     fig = go.Figure()
    #     surface = go.Surface(x=plot1, y=plot2, z=Z)

    #     fig.update_layout(
    #         scene=dict(
    #             xaxis_title=f'Alpha_1 Values log10^{self.lam_1_lb} to log10^{self.lam_1_ub} ',
    #             yaxis_title=f'Alpha_2 Values log10^{self.lam_2_lb} to log10^{self.lam_2_ub}',
    #             zaxis_title='log(norm(p_a1,a2 - p_true)**2)'
    #         ),
    #         title=f"Surface of Grid Search for NR {iter_num}"
    #     )
    #     self.fig = fig
    #     # fig = plt.figure()
    #     # axes = fig.add_subplot(111, projection='3d')

    #     # surface = axes.plot_surface(plot1, plot2, Z, rstride=1, cstride=1, cmap = "inferno")

    #     # axes.set_xlabel('Alpha_1 Values')
    #     # axes.set_ylabel('Alpha_2 Values')
    #     # axes.set_zlabel('log(norm(p_a1,a2 - p_true)**2)')

    #     # colorbar = fig.colorbar(surface, ax=axes, shrink=0.5, aspect= 20)
    #     # plt.title(f"Surface of Grid Search for NR {iter_num}")
    #     # self.fig = fig
    # def plot_grid_search(self, Lambda_vec, Lambda_vec2, iter_num):
    #     plot1, plot2 = np.meshgrid(Lambda_vec, Lambda_vec2)
    #     Z = np.log(self.rhos)
    #     sb.set_style('whitegrid')

    #     # Find all indices where the surface has the minimum value
    #     min_indices = np.where(self.rhos == np.min(self.rhos))
        
    #     min_x = Lambda_vec[min_indices[0]]
    #     min_y = Lambda_vec2[min_indices[1]]
    #     min_z = np.min(self.rhos)

    #     fig = go.Figure()
    #     surface = go.Surface(x=plot1, y=plot2, z=Z)

    #     fig.add_trace(surface)

    #     fig.add_trace(go.Scatter3d(
    #         x=min_x,
    #         y=min_y,
    #         z=min_z * np.ones_like(min_x),  # Use the minimum z value for all points
    #         mode='markers',
    #         marker=dict(
    #             size=5,
    #             color='red',
    #             symbol='circle'
    #         ),
    #         name='Minimum Points'
    #     ))

    #     fig.update_layout(
    #         scene=dict(
    #             xaxis_title=f'Alpha_1 Values log10^{self.lam_1_lb} to log10^{self.lam_1_ub}',
    #             yaxis_title=f'Alpha_2 Values log10^{self.lam_2_lb} to log10^{self.lam_2_ub}',
    #             zaxis_title='log(norm(p_a1,a2 - p_true)**2)'
    #         ),
    #         title=f"Surface of Grid Search for NR {iter_num}"
    #     )

    #     self.fig = fig


    # def plot_grid_search(self, Lambda_vec, Lambda_vec2, Lambda_vec3, iter_num):
        # import sys
        # import numpy as np
        # nalpha1 = 5
        # nalpha2 = 5
        # nalpha3 = 5

        # lam_1_lb = -4
        # lam_1_ub = 0
        # lam_2_lb = -4
        # lam_2_ub = 0
        # lam_3_lb = -4
        # lam_3_ub = 0
        # Lambda_vec = np.logspace(lam_1_lb, lam_1_ub, nalpha1)
        # Lambda_vec2 = np.logspace(lam_2_lb, lam_2_ub, nalpha2)
        # Lambda_vec3 = np.logspace(lam_3_lb, lam_3_ub, nalpha3)

    #     sys.path.insert(0, "/Users/steveh/Downloads/ChartDirector/lib")
    #     from pychartdir import *
    #     sys.path.insert(0, "/Users/steveh/Downloads/ChartDirector/pythondemo")
    #     from surface4d_josh_modified import createChart
    #     rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/rhos_NR1_01_24_24_5_alpha1_5_alpha2_5_alpha3_discretization_alpha1log_-4_0_alpha2log_-4_0_alpha3log_-4_0.npy")
    #     fig = createChart(1,Lambda_vec,Lambda_vec2,Lambda_vec3,rhos)
    #     from surface4d import createChart

    # def plot_grid_search(self, Lambda_vec, Lambda_vec2, Lambda_vec3, iter_num):

    #     import numpy as np
    #     import plotly.graph_objects as go
    #     import plotly.express as px
    #     from scipy.spatial import Delaunay

        

    #     # Assuming Lambda_vec, Lambda_vec2, and self.rhos are defined

    #     # Create meshgrid
    #     plot1, plot2, plot3 = np.meshgrid(Lambda_vec, Lambda_vec2, Lambda_vec3, indexing='ij')
    #     Z = np.log10(self.rhos)

    #     # Find indices of minimum value
    #     min_index = np.unravel_index(np.argmin(Z), Z.shape)

    #     min_x = Lambda_vec[min_index[0]]
    #     min_y = Lambda_vec2[min_index[1]]
    #     min_z = Lambda_vec3[min_index[2]]

    #     # min_x_mesh, min_y_mesh, min_z_mesh = np.meshgrid(min_x,min_y,min_z)
        
    #     flat_X = plot1.flatten()
    #     flat_Y = plot2.flatten()
    #     flat_Z = plot3.flatten()
    #     flat_color = Z.flatten()
    #     min_gamma = np.min(Z)

    #     # Reshape to a 25 by 2 array
    #     # combinations = np.column_stack((grid1.ravel(), grid2.ravel()))
    #     # combinations = np.column_stack((flat_X, flat_Y))
    #     # tri = Delaunay(combinations) #triangulate the 2d points
    #     # I, J, K = (tri.simplices).T
    #     # # print("simplices:", "\n", tri.simplices)
    #     # fig=go.Figure(go.Mesh3d(x=flat_X, y=flat_Y, z=flat_Z,
    #     #                         i=I, j=J, k=K, 
    #     #                     intensity=flat_color, colorscale="Viridis",  colorbar=dict(title='log(norm(p_a1,a2 - p_true)**2)'))) #these two last attributes, intensity and colorscale, assign a color according to snow height


    #     # Add a marker for the minimum point
    #     # fig.add_trace(go.Scatter3d(
    #     #     x=[min_x],
    #     #     y=[min_y],
    #     #     z=[min_z],
    #     #     mode='markers',
    #     #     marker=dict(color='red', size=10),
    #     #     name='Minimum Point'
    #     # ))

    #     # Configure layout
    #     # fig.update_layout(
    #     #     scene=dict(
    #     #         xaxis=dict(title='Alpha_1'),
    #     #         yaxis=dict(title='Alpha_2'),
    #     #         zaxis=dict(title='Alpha_3'),
    #     #     ),
    #     #     title='Surface Plot with Minimum Point',
    #     #     margin=dict(l=0, r=0, b=0, t=40)
    #     # )
    #     # fig = go.Figure(data=[go.Surface(x=plot1, y=plot2, z=plot3, surfacecolor= Z, colorscale="ice", opacity=0.8)])
    #     # surface_trace = go.Surface(x=Lambda_vec, y=Lambda_vec2, z=Lambda_vec3,surfacecolor= Z , colorscale='Viridis')
    #     # fig.add_trace(fig)
        


    #     # # points2D = np.vstack([Lambda_vec,Lambda_vec2]).T
    #     # tri = Delaunay(combinations) #triangulate the 2d points
    #     # I, J, K = (tri.simplices).T
    #     # print("simplices:", "\n", tri.simplices)
    #     # fig=go.Figure(go.Mesh3d(x=Lambda_vec, y=Lambda_vec2, z=Lambda_vec3,
    #     #                         i=I, j=J, k=K, 
    #     #                     intensity=, colorscale="ice" )) #these two last attributes, intensity and colorscale, assign a color according to snow height

    #     #SCATTERPLOT
    #     # fig = px.scatter_3d(x=flat_X, y=flat_Y, z=flat_Z, color=flat_color,
    #     #             labels={'x': 'Alpha_1', 'y': 'Alpha_2', 'z': 'Alpha_3', 'color': 'log(norm(p_a1,a2 - p_true)**2)'},
    #     #             color_continuous_scale='Viridis')
    #     # fig.add_trace(go.Scatter3d(x=[min_x], y=[min_y], z=[min_z], mode='markers', marker=dict(color="red", size=20), name='Minimum Point'))

    #     # fig.update_layout(
    #     #     coloraxis_colorbar=dict(yanchor="bottom", y=-0.1, x=0.9, title='log(norm(p_a1,a2 - p_true)**2)'),
    #     #     coloraxis=dict(colorbar=dict(yanchor="bottom", y=-0.1))
    #     # )

    #     # # fig.add_trace(px.scatter_3d(x=[min_x], y=[min_y], z=[min_z], color= "gold", size=[8]).data[0])
    #     # fig.update_layout(scene=dict(xaxis_type='log', yaxis_type='log', zaxis_type='log'))


    #     # fig = go.Figure()

    #     # surface_trace = go.Surface(x=Lambda_vec, y=Lambda_vec2, z=Lambda_vec3, colorscale='Viridis', cmin=min_gamma, cmax=np.max(Z), colorbar=dict(title='log(norm(p_a1,a2 - p_true)**2)'))
    #     # surface_trace = go.Surface(x=Lambda_vec, y=Lambda_vec2, z=Lambda_vec3, surfacecolor=Z.T, colorscale='Viridis')
    #     # fig.add_trace(surface_trace)
    #     # # Add surface plot
    #     # fig.add_trace(surface_trace)

    #     # # Add scatter plot for the minimum point
    #     # fig.add_trace(go.Scatter3d(
    #     #     x=[min_x],
    #     #     y=[min_y],
    #     #     z=[min_z],
    #     #     mode='markers',
    #     #     marker=dict(
    #     #         size=5,
    #     #         color=min_gamma,
    #     #         colorscale='Viridis',
    #     #         symbol='circle'
    #     #     ),
    #     #     name='Minimum Point'
    #     # ))

    #     # # Create a 3D surface plot
    #     # fig = go.Figure(data=[go.Surface(x=Lambda_vec, y=Lambda_vec2,z=Lambda_vec3, colorbar=dict(title='Colorbar Title'),
    #     #                          surfacecolor=Z.T, colorscale='Viridis', cmin=Z.min(), cmax=Z.max())])

    #     # # Add a scatter plot for the minimum point
    #     # fig.add_trace(go.Scatter3d(
    #     #     x=np.array([min_x]),
    #     #     y=np.array([min_y]),
    #     #     z=np.array([min_z]),
    #     #     mode='markers',
    #     #     marker=dict(
    #     #         size=5,
    #     #         color='gold',
    #     #         symbol='circle'
    #     #     ),
    #     #     name='Minimum Point'
    #     # ))
    #     # fig = px.scatter_3d(x=Lambda_vec, y=Lambda_vec2, z=Lambda_vec3, color=Z.flatten(),
    #     #                     labels={'x': f'Alpha_1 Values 10^{self.lam_1_lb} to 10^{self.lam_1_ub}',
    #     #                             'y': f'Alpha_2 Values 10^{self.lam_2_lb} to 10^{self.lam_2_ub}',
    #     #                             'z': f'Alpha_3 Values 10^{self.lam_3_lb} to 10^{self.lam_3_ub}'},
    #     #                     title=f"Surface of Grid Search for NR {iter_num}",
    #     #                     color_continuous_scale='Viridis')

    #     # Add a scatter plot for the minimum point
    #     # fig.add_trace(go.Scatter3d(
    #     #     x=np.array([min_x]),
    #     #     y=np.array([min_y]),
    #     #     z=np.array([min_z]),
    #     #     mode='markers',
    #     #     marker=dict(
    #     #         size=5,
    #     #         color='gold',
    #     #         symbol='circle'
    #     #     ),
    #     #     name='Minimum Point'
    #     # ))

    #     # Configure layout
    #     fig.update_layout(
    #         scene=dict(
    #             xaxis=dict(type='log', title=f'Alpha_1 Values 10^{self.lam_1_lb} to 10^{self.lam_1_ub}'),
    #             yaxis=dict(type='log', title=f'Alpha_2 Values 10^{self.lam_2_lb} to 10^{self.lam_2_ub}'),
    #             zaxis=dict(type='log', title=f'Alpha_3 Values 10^{self.lam_3_lb} to 10^{self.lam_3_ub}'),
    #                                     ),
    #         title=f"Surface of Grid Search for NR {iter_num}",
    #         annotations=[
    #             dict(
    #                 text=f"Optimal Alpha_1 value: {min_x}",
    #                 showarrow=False,
    #                 xref="paper",
    #                 yref="paper",
    #                 x=0,
    #                 y=1.02
    #             ),
    #             dict(
    #                 text=f"Optimal Alpha_2 value: {min_y}",
    #                 showarrow=False,
    #                 xref="paper",
    #                 yref="paper",
    #                 x=0,
    #                 y=0.99
    #             ),
    #             dict(
    #                 text=f"Optimal Alpha_3 value: {min_z}",
    #                 showarrow=False,
    #                 xref="paper",
    #                 yref="paper",
    #                 x=0,
    #                 y=0.96
    #             )
    #         ]
    #     )
    #     self.fig = fig




        # for j, k in (product(range(len(Lambda_vec)), range(len(Lambda_vec2)))):
        #     x = cp.Variable(self.nT2)
        #     objective = cp.Minimize(cp.norm(self.G @ x - self.data_noisy, 2)**2 +
        #                             Lambda_vec[j]**2 * cp.norm(self.B_1 @ x, 2)**2 +
        #                             Lambda_vec2[k]**2 * cp.norm(self.B_2 @ x, 2)**2)
        #     # objective = cp.Minimize(cp.norm(self.G @ x - self.data_noisy, 2)**2 +
        #     #                         Lambda_vec[j]**2 * cp.norm(x, 2)**2)
        #     constraints = [x >= 0]
        #     problem = cp.Problem(objective, constraints)
        #     problem.solve(solver=cp.MOSEK, verbose=False)
        #     if x.value is not None:
        #         self.OP_lc_vec[:, j, k] = x.value.flatten()
        #         if norm(self.G @ x.value.flatten() - self.data_noisy) - self.delta <= 0:
        #             x_value.flatten()
        #         print(OP_ind)
        #         self.rhos[j, k] = norm(self.OP_lc_vec[:, j, k] - self.g, 2)**2
        #     else:
        #         print("x.value == None")

        # U, s, V = csvd(self.G, tst = None, nargin = 1, nargout = 3)
        # x_delta,lambda_OP = discrep(U,s,V,self.data_noisy,self.delta, x_0= None, nargin = 5)
        # self.lam_OP = lambda_OP
        # self.f_rec_OP,_,_ = tikhonov(U,s,V,self.data_noisy,lambda_OP, nargin=5, nargout=1)

            # g_fig = plt.figure()
            # plt.plot(self.g, label="Ground Truth")
            # plt.plot(self.f_rec_OP, label="OP")
            # plt.plot(self.f_rec_grid, label="Grid_Search")
            # plt.xlabel("T2")
            # plt.ylabel("Amplitude")
            # plt.title(f"MRR Problem NR{num_real + 1}")
            # plt.legend(["Ground Truth", "OP", "Grid_Search"])


    # def plot_L_hypersurface(self, Lambda_vec, Lambda_vec2, iter_num):
    
    #     plot1, plot2 = np.meshgrid(Lambda_vec, Lambda_vec2)
    #     Z = np.log(self.rhos)
    #     sb.set_style('whitegrid')

    #     # Find all indices where the surface has the minimum value
    #     min_indices = np.where(Z == np.min(Z))
        
    #     min_x = Lambda_vec[min_indices[0]]
    #     min_y = Lambda_vec2[min_indices[1]]
    #     min_z = np.min(Z)

    #     fig = go.Figure()
    #     surface = go.Surface(x=plot1, y=plot2, z=Z)

    #     fig.add_trace(surface)

    #     fig.add_trace(go.Scatter3d(
    #         x=min_x,
    #         y=min_y,
    #         z=min_z * np.ones_like(min_x),  # Use the minimum z value for all points
    #         mode='markers',
    #         marker=dict(
    #             size=8,
    #             color='gold',
    #             symbol='circle'
    #         ),
    #         name='Minimum Points'
    #     ))

    #     fig.update_layout(
    #         scene=dict(
    #             xaxis_title=f'Alpha_1 Values 10^{self.alp_1_lb} to 10^{self.alp_1_ub}',
    #             yaxis_title=f'Alpha_2 Values 10^{self.alp_2_lb} to 10^{self.alp_2_ub}',
    #             zaxis_title='log(norm(p_a1,a2 - p_true)**2)'
    #         ),
    #         title=f"Surface of Grid Search for NR {iter_num}",
    #         annotations=[
    #             dict(
    #                 text=f"Optimal Alpha_1 value: {min_x}",
    #                 showarrow=False,
    #                 xref="paper",
    #                 yref="paper",
    #                 x=0,
    #                 y=1.08
    #             ),
    #             dict(
    #                 text=f"Optimal Alpha_2 value: {min_y}",
    #                 showarrow=False,
    #                 xref="paper",
    #                 yref="paper",
    #                 x=0,
    #                 y=1.05
    #             )
    #         ]
    #     )



# NR8_rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/rhos_NR8_01_19_24_10_Lambda_discretizations.npy")
# np.where(NR8_rhos == np.min(NR8_rhos))
# NR7_rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/rhos_NR7_01_19_24_10_Lambda_discretizations.npy")
# np.where(NR7_rhos == np.min(NR7_rhos))
# NR5_rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/rhos_NR5_01_19_24_10_Lambda_discretizations.npy")
# np.where(NR5_rhos == np.min(NR5_rhos))
    
# rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/rhos_NR2_01_19_24_20_Lambda1_20_Lambda2_discretizations_lam1log_-3_-1_lam2log_-3_-1.npy")
# np.where(rhos == np.min(rhos))
# np.min(rhos)

# rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/rhos_NR1_01_19_24_20_Lambda1_20_Lambda2_discretizations_lam1log_-3_-1_lam2log_-3_-1.npy")
# np.where(np.log(rhos) == np.min(np.log(rhos)))
# np.min(np.log(rhos))

# alp_OP = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/opt_alp_OPs_01_19_24_25_alpha1_25_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")
# opt_alpha = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/opt_alphas_01_19_24_25_alpha1_25_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")

# nT2 = len(T2)
# half_nT2 = int(nT2/2)
# half_mat = np.eye(half_nT2)
# B_1 = np.zeros((nT2, nT2))
# B_2 = np.zeros((nT2, nT2))
# B_1[:half_nT2, :half_nT2] = half_mat
# B_2[half_nT2:, half_nT2:] = half_mat

# G = np.zeros((len(TE), len(T2)))
# for i in range(len(TE)):
#         for j in range(len(T2)):
#             G[i,j] = np.exp(-TE[i]/T2[j])


# data_noisy = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/noisy_data_for_NR3_01_20_24_20_alpha1_20_alpha2_discretization_alpha1log_-3_-1_alpha2log_-3_-1.npy")
# rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/rhos_NR3_01_20_24_20_alpha1_20_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")
# opt_alpha = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/opt_alphas_01_20_24_20_alpha1_20_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")
# x_lc_vec = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/x_lc_vec_NR3_01_20_24_20_alpha1_20_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")

# opt_alpha_3 = opt_alpha[2]

# x = cp.Variable(len(T2))
#             # objective = cp.Minimize(cp.norm(G.T @ G @ x - G.T @ data_noisy, 2) +
#             #                         Lambda_vec[j]**2 * cp.norm(B_1 @ x, 2)**2 +
#             #                         Lambda_vec2[k]**2 * cp.norm(B_2 @ x, 2)**2)
# objective = cp.Minimize(cp.norm(G @ x - data_noisy, 2)**2 +
#                         opt_alpha_3[0]**2 * cp.norm(B_1 @ x, 2)**2 +
#                         opt_alpha_3[1]**2 * cp.norm(B_2 @ x, 2)**2)
# constraints = [x >= 0]
# problem = cp.Problem(objective, constraints)
# problem.solve(solver=cp.ECOS, verbose=False)
# x.value.flatten()
    
# error_grid = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/errors_grid_01_22_24_20_alpha1_20_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")
# g1 = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
# g2 = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
# g  = 0.5 * (g1 + g2)

# grid_search = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/x_lc_vec_NR1_01_22_24_20_alpha1_20_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")
    
# g = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/ground_truth_01_22_24_20_alpha1_20_alpha2_discretizations_alpha1log_-6_-1_alpha2log_-6_-1.npy")
# alpha_ind = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/opt_alphas_ind_01_22_24_20_alpha1_20_alpha2_discretizations_alpha1log_-6_-1_alpha2log_-6_-1.npy")
# x_lec_vec = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/x_lc_vec_NR1_01_22_24_20_alpha1_20_alpha2_discretizations_alpha1log_-6_-1_alpha2log_-6_-1.npy")
# x_ind = alpha_ind[0][0]
# y_ind = alpha_ind[0][1]
# f_rec = x_lec_vec[:,x_ind,y_ind].flatten()
# plt.plot(g)
# plt.plot(f_rec)
# plt.show()


            # fig = plt.subplot(2)
            # g_fig = plt.figure()
            # plt.plot(self.g, label="Ground Truth")
            # plt.plot(self.f_rec_DP, label="DP")
            # plt.plot(self.f_rec_grid, label="Grid_Search")
            # plt.xlabel("T2")
            # plt.ylabel("Amplitude")
            # plt.title(f"MRR Problem Ground Truth NR{num_real + 1}")
            # plt.legend(["Ground Truth", "DP", "3D Grid Search"])
            ## plt.suptitle(f"error_DP = {round(self.com_rec_DP,3)}; error_grid = {round(self.com_rec_grid,3)}" + f"\n alpha_1 = {self.min_alpha1}; alpha_2 = {round(self.min_alpha2,3)} ; alpha_3 = {round(self.min_alpha3,3)} ; alp_DP = {round(self.alp_DP,3)}", fontsize=8)
            # plt.suptitle(f"error_DP = {round(self.com_rec_DP.item(), 5)}; error_grid = {round(self.com_rec_grid.item(), 5)}" + f"\n alpha_1 = {round(self.min_alpha1.item(), 5)}; alpha_2 = {round(self.min_alpha2.item(), 5)} ; alpha_3 = {round(self.min_alpha3.item(), 5)} ; alp_DP = {round(self.alp_DP.item(), 5)}", fontsize=8)
            # plt.savefig("".join([file_path, f"compare_gt_DP_multireg_{date}_NR_{num_real + 1}_mu1_{self.mu1}_mu2_{self.mu2}_mu3_{self.mu3}_sigma1_{self.sigma1}_sigma2_{self.sigma2}_sigma3_{self.sigma3}"]))
            # print(f"Saving comparison plot is complete")



# NR8_TP_rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/TP_rhos_NR8_01_19_24_10_Alpha_discretizations.npy")
# np.where(NR8_TP_rhos == np.min(NR8_TP_rhos))
# NR7_TP_rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/TP_rhos_NR7_01_19_24_10_Alpha_discretizations.npy")
# np.where(NR7_TP_rhos == np.min(NR7_TP_rhos))
# NR5_TP_rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/TP_rhos_NR5_01_19_24_10_Alpha_discretizations.npy")
# np.where(NR5_TP_rhos == np.min(NR5_TP_rhos))
    
# TP_rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/TP_rhos_NR2_01_19_24_20_Lambda1_20_Lambda2_discretizations_lam1log_-3_-1_lam2log_-3_-1.npy")
# np.where(TP_rhos == np.min(TP_rhos))
# np.min(TP_rhos)

# TP_rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/TP_rhos_NR1_01_19_24_20_Lambda1_20_Lambda2_discretizations_lam1log_-3_-1_lam2log_-3_-1.npy")
# np.where(np.log(TP_rhos) == np.min(np.log(TP_rhos)))
# np.min(np.log(TP_rhos))

# alp_DP = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/opt_alp_DPs_01_19_24_25_alpha1_25_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")
# opt_alpha = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/opt_alphas_01_19_24_25_alpha1_25_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")

# nT2 = len(T2)
# half_nT2 = int(nT2/2)
# half_mat = np.eye(half_nT2)
# B_1 = np.zeros((nT2, nT2))
# B_2 = np.zeros((nT2, nT2))
# B_1[:half_nT2, :half_nT2] = half_mat
# B_2[half_nT2:, half_nT2:] = half_mat

# G = np.zeros((len(TE), len(T2)))
# for i in range(len(TE)):
#         for j in range(len(T2)):
#             G[i,j] = np.exp(-TE[i]/T2[j])


# data_noisy = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/noisy_data_for_NR3_01_20_24_20_alpha1_20_alpha2_discretization_alpha1log_-3_-1_alpha2log_-3_-1.npy")
# TP_rhos = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/TP_rhos_NR3_01_20_24_20_alpha1_20_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")
# opt_alpha = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/opt_alphas_01_20_24_20_alpha1_20_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")
# TP_x_lc_vec = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/TP_x_lc_vec_NR3_01_20_24_20_alpha1_20_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")

# opt_alpha_3 = opt_alpha[2]

# x = cp.Variable(len(T2))
#             # objective = cp.Minimize(cp.norm(G.T @ G @ x - G.T @ data_noisy, 2) +
#             #                         Alpha_vec[j]**2 * cp.norm(B_1 @ x, 2)**2 +
#             #                         Alpha_vec2[k]**2 * cp.norm(B_2 @ x, 2)**2)
# objective = cp.Minimize(cp.norm(G @ x - data_noisy, 2)**2 +
#                         opt_alpha_3[0]**2 * cp.norm(B_1 @ x, 2)**2 +
#                         opt_alpha_3[1]**2 * cp.norm(B_2 @ x, 2)**2)
# constraints = [x >= 0]
# problem = cp.Problem(objective, constraints)
# problem.solve(solver=cp.ECOS, verbose=False)
# x.value.flatten()
    
# error_grid = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/errors_grid_01_22_24_20_alpha1_20_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")
# g1 = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
# g2 = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
# g  = 0.5 * (g1 + g2)

# grid_search = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/TP_x_lc_vec_NR1_01_22_24_20_alpha1_20_alpha2_discretizations_alpha1log_-3_-1_alpha2log_-3_-1.npy")
    
# g = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/ground_truth_01_22_24_20_alpha1_20_alpha2_discretizations_alpha1log_-6_-1_alpha2log_-6_-1.npy")
# alpha_ind = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/opt_alphas_ind_01_22_24_20_alpha1_20_alpha2_discretizations_alpha1log_-6_-1_alpha2log_-6_-1.npy")
# x_lec_vec = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/TP_x_lc_vec_NR1_01_22_24_20_alpha1_20_alpha2_discretizations_alpha1log_-6_-1_alpha2log_-6_-1.npy")
# x_ind = alpha_ind[0][0]
# y_ind = alpha_ind[0][1]
# f_rec = x_lec_vec[:,x_ind,y_ind].flatten()
# plt.plot(g)
# plt.plot(f_rec)
# plt.show()
 