#Curvature Check
import numpy as np
import sys
sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
from numpy.linalg import norm
from regu.csvd import csvd
from regu.tikhonov import tikhonov
from regu.discrep import discrep
from regu.l_curve import l_curve
from fnnls import fnnls
from regu.gcv import gcv
from Utilities_functions.discrep_L2 import discrep_L2
from Utilities_functions.GCV_NNLS import GCV_NNLS
from Utilities_functions.Lcurve import Lcurve
from regu.nonnegtik_hnorm import nonnegtik_hnorm
from scipy.optimize import nnls
from itertools import product
import cvxpy as cp
from tqdm import tqdm
import scipy
import matplotlib.pyplot as plt
import math
import time
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
from datetime import datetime
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import mosek
import sympy as sp
# import fnnlsEigen as fe 
# import numdifftools.nd_statsmodels as nd
from autograd import grad, elementwise_grad, jacobian, hessian, holomorphic_grad
import os
mosek_lic_path = "/Users/steveh/Downloads/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_lic_path


def plot(x,y):
    # alp1 = np.log10(x)
    # alp2 = np.log10(y)
    alp1 = x
    alp2 = y
    plot1, plot2 = np.meshgrid(x, y)

    z = np.cos(plot1) + np.sin(plot2)
    min_index = np.unravel_index(np.argmin(z), z.shape)
    max_index = np.unravel_index(np.argmax(z), z.shape)
    max_z = np.max(z)
    # print("max_index", max_index)
    fig = go.Figure(data=[go.Surface(x=alp1, y=alp2, z= z.T)])
    typescale = 'linear'

    xval = x[max_index[0]]
    yval = y[max_index[1]]
    fig.add_trace(go.Scatter3d(
        x=np.array([xval]),
        y=np.array([yval]),
        z=np.array([max_z]),
        mode='markers',
        marker=dict(
            size=5,
            color='orange',
            symbol='circle'
        ),
        name= "Z"
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(type=typescale, title=f'Alpha_1 Values 10^{alp_1_lb} to 10^{alp_1_ub}'),
            yaxis=dict(type=typescale, title=f'Alpha_2 Values 10^{alp_2_lb} to 10^{alp_2_ub}'),
            zaxis=dict(title= "Z")  # Assuming z-axis is log scale
        ),
        title=f"Surface of Grid Search",
        annotations=[
            dict(
                text=f"Optimal Alpha_1 value: {xval}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=1.08
            ),
            dict(
                text=f"Optimal Alpha_2 value: {yval}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=1.05
            )
        ]
    )
    return fig


from __future__ import division
import random
import math

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.9       # constant inertia weight (how much to weigh the previous velocity)
        c1=2        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
        global num_dimensions

        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        # print final results
        print 'FINAL:'
        print pos_best_g
        print err_best_g

if __name__ == "__PSO__":
    main()

#--- RUN ----------------------------------------------------------------------+

initial=[5,5]               # initial starting location [x1,x2...]
bounds=[(-10,10),(-10,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
PSO(func1,initial,bounds,num_particles=15,maxiter=30)
#--- RUN ----------------------------------------------------------------------+

initial=[5,5]               # initial starting location [x1,x2...]
bounds=[(-10,10),(-10,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
PSO(func1,initial,bounds,num_particles=15,maxiter=30)

print("Running 2D_multi_reg_MRR_0116.py script")
file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/"
# Get today's date
today_date = datetime.today()
# Format the date as "month_day_lasttwodigits"
date = today_date.strftime("%m_%d_%y")

T2 = np.arange(1,201)
TE = np.arange(1,512,4)
nalpha1 = 50
nalpha2 = 50
alp_1_lb = -4
alp_1_ub = -1
alp_2_lb = -4
alp_2_ub = -1
SNR = 200

Alpha_vec = np.logspace(alp_1_lb, alp_1_ub, nalpha1)
Alpha_vec2 = np.logspace(alp_2_lb, alp_2_ub, nalpha2)

# def curv_plot(x,y):
#     # alp1 = np.log10(x)
#     # alp2 = np.log10(y)
#     alp1 = x
#     alp2 = y
#     plot1, plot2 = np.meshgrid(x, y)

#     # z = np.cos(plot1) + np.sin(plot2)
#     z = (np.cos(plot1)*np.sin(plot2))/((1 - np.sin(plot1) + np.cos(plot2))**2)
#     min_index = np.unravel_index(np.argmin(z), z.shape)
#     max_index = np.unravel_index(np.argmax(z), z.shape)
#     max_z = np.max(z)
#     # print("max_index", max_index)
#     fig = go.Figure(data=[go.Surface(x=alp1, y=alp2, z= z.T)])
#     typescale = 'linear'

#     xval = x[max_index[0]]
#     yval = y[max_index[1]]
#     fig.add_trace(go.Scatter3d(
#         x=np.array([xval]),
#         y=np.array([yval]),
#         z=np.array([max_z]),
#         mode='markers',
#         marker=dict(
#             size=5,
#             color='orange',
#             symbol='circle'
#         ),
#         name= "Maximum Curvature"
#     ))
#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(type=typescale, title=f'Alpha_1 Values 10^{alp_1_lb} to 10^{alp_1_ub}'),
#             yaxis=dict(type=typescale, title=f'Alpha_2 Values 10^{alp_2_lb} to 10^{alp_2_ub}'),
#             zaxis=dict(title= "Curvature")  # Assuming z-axis is log scale
#         ),
#         title=f"Surface of Grid Search",
#         annotations=[
#             dict(
#                 text=f"Optimal Alpha_1 value: {xval}",
#                 showarrow=False,
#                 xref="paper",
#                 yref="paper",
#                 x=0,
#                 y=1.08
#             ),
#             dict(
#                 text=f"Optimal Alpha_2 value: {yval}",
#                 showarrow=False,
#                 xref="paper",
#                 yref="paper",
#                 x=0,
#                 y=1.05
#             )
#         ]
#     )
#     return fig

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def curv_plot(x, y):
#     alp1 = x
#     alp2 = y
#     plot1, plot2 = np.meshgrid(x, y)

#     z = (np.cos(plot1) * np.sin(plot2)) / ((1 - np.sin(plot1) + np.cos(plot2)) ** 2)
#     min_index = np.unravel_index(np.argmin(z), z.shape)
#     max_index = np.unravel_index(np.argmax(z), z.shape)
#     max_z = np.max(z)

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(plot1, plot2, z)

#     ax.scatter(x[max_index[0]], y[max_index[1]], max_z, color='orange', label="Maximum Curvature")

#     ax.set_xlabel('Alpha_1')
#     ax.set_ylabel('Alpha_2')
#     ax.set_zlabel('Curvature')

#     plt.title("Surface of Grid Search")
#     plt.legend()
#     plt.show()

import numpy as np
# from mayavi import mlab

# def curv_plot(x, y):
#     alp1 = x
#     alp2 = y
#     plot1, plot2 = np.meshgrid(x, y)

#     z = (np.cos(plot1) * np.sin(plot2)) / ((1 - np.sin(plot1) + np.cos(plot2)) ** 2)

#     # Create a Mayavi figure
#     fig = mlab.figure(size=(800, 600), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

#     # Plot the surface
#     surf = mlab.surf(plot1, plot2, z, colormap='viridis')

#     # Add a color bar
#     mlab.colorbar(surf, orientation='vertical')

#     # Plot the maximum curvature point
#     max_index = np.unravel_index(np.argmax(z), z.shape)
#     mlab.points3d(alp1[max_index[0]], alp2[max_index[1]], z[max_index], color=(1, 0, 0), scale_factor=0.5)

#     # Set labels and title
#     mlab.xlabel('Alpha_1')
#     mlab.ylabel('Alpha_2')
#     mlab.zlabel('Curvature')
#     mlab.title('Surface of Grid Search')

#     # Show the plot
#     mlab.show()



if __name__ == '__main__':

    print("Running 2D_multi_reg_MRR_0116.py script")
    file_path = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/"
    # Get today's date
    today_date = datetime.today()
    # Format the date as "month_day_lasttwodigits"
    date = today_date.strftime("%m_%d_%y")

    T2 = np.arange(1,201)
    TE = np.arange(1,512,4)
    nalpha1 = 50
    nalpha2 = 50
    alp_1_lb = -4
    alp_1_ub = -1
    alp_2_lb = -4
    alp_2_ub = -1
    SNR = 200

    #-4 to -1 for line of solutions
    mu1 = 40
    mu2 = 150
    sigma1 = 4
    sigma2 = 20
    # Alpha_vec = np.logspace(alp_1_lb, alp_1_ub, nalpha1)
    # Alpha_vec2 = np.logspace(alp_2_lb, alp_2_ub, nalpha2)
    num_real=3

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    fig = plot(x, y)
    pio.write_html(fig, file= os.path.join(file_path, f"Curvature_Check_Plot_1_{date}_{nalpha1}_alpha1_{nalpha2}_alpha2_discretizations_alpha1log_{alp_1_lb}_{alp_1_ub}_alpha2log_{alp_2_lb}_{alp_2_ub}.html"))
    print("Figure Printed")
    curvfig = curv_plot(x, y)
    # Example usage:
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    curv_plot(x, y)
    print("Figure Printed")


# def plot_TP_surface(Alpha_vec, Alpha_vec2, error_val, curvature_cond, resid_cond, iter_num):

#     # Assuming Lambda_vec, Lambda_vec2, and self.rhos are defined
#     # Create meshgrid
#     plot1, plot2 = np.meshgrid(Alpha_vec, Alpha_vec2)
#     min_index = np.unravel_index(np.argmin(error_val), error_val.shape)
#     max_index = np.unravel_index(np.argmax(error_val), error_val.shape)

#     if curvature_cond == True:
#         TP_zval = error_val
#         Alpha_vector = Alpha_vec
#         Alpha_vector2 = Alpha_vec2
#         z_val_title = "Curvature"
#         min_x = Alpha_vector[max_index[0]]
#         min_y = Alpha_vector2[max_index[1]]
#         min_z = np.max(error_val)
#         goal_name = "Maximum Curvature"
#         typescale = 'linear'
#     else:
#         if resid_cond == False:
#             TP_zval = TP_log_err_norm = np.log10(error_val)
#             z_val_title = 'Log(norm(p_a1,a2 - p_true)**2)'
#         else:
#             TP_zval = error_val
#             z_val_title = 'Residual Norm'
#         # Alpha_vector = np.log10(Alpha_vec)
#         Alpha_vector = Alpha_vec
#         # print(Alpha_vec10.shape)
#         # Alpha_vector2 = np.log10(Alpha_vec2)
#         Alpha_vector2 = Alpha_vec2
#         # print(Alpha_vec210.shape)
#         min_x = Alpha_vector[min_index[0]]
#         min_y = Alpha_vector2[min_index[1]]
#         min_z = np.min(error_val)
#         goal_name = "Minimum Point"
#         typescale = 'log'
#     # Find indices of minimum value

#     # Create a 3D surface plot
#     fig = go.Figure(data=[go.Surface(x=Alpha_vector, y=Alpha_vector2, z= error_val.T)])

#     # Add a scatter plot for the minimum point
#     fig.add_trace(go.Scatter3d(
#         x=np.array([min_x]),
#         y=np.array([min_y]),
#         z=np.array([min_z]),
#         mode='markers',
#         marker=dict(
#             size=5,
#             color='orange',
#             symbol='circle'
#         ),
#         name= goal_name
#     ))

#     if resid_cond == True:
#         delta = 1.05 * norm(noise)
#         closest_indices = np.unravel_index(np.argmin(np.abs(error_val - delta)), error_val.shape)
#         ind_x = closest_indices[0]
#         ind_y = closest_indices[1]

#         fig.add_trace(go.Surface(
#         x=Alpha_vector,
#         y=Alpha_vector2,
#         z=[[delta] * len(Alpha_vector)] * len(Alpha_vector2),  # Creates a plane at z=delta
#         colorscale='Viridis',  # Adjust the colorscale as needed
#         opacity=0.8,  # Adjust the opacity as needed
#         showscale=False,  # Hide the color scale
#         name='Delta Plane'
#         ))
#         fig.add_trace(go.Scatter3d(
#         x=[Alpha_vector[ind_x]],
#         y=[Alpha_vector2[ind_y]],
#         z=[error_val[ind_x, ind_y]],
#         mode='markers',
#         marker=dict(
#             size=5,
#             color='red',
#             symbol='circle'
#         ),
#         name='Closest Point to Delta'
#         ))

#     # Configure layout
#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(type=typescale, title=f'Alpha_1 Values 10^{alp_1_lb} to 10^{alp_1_ub}'),
#             yaxis=dict(type=typescale, title=f'Alpha_2 Values 10^{alp_2_lb} to 10^{alp_2_ub}'),
#             zaxis=dict(title= z_val_title)  # Assuming z-axis is log scale
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
#     return fig