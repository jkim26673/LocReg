import numpy as np
import os
import cvxpy as cp
import matplotlib.pyplot as plt
mosek_lic_path = "/Users/steveh/Downloads/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_lic_path
# N Parameter Ito problem


def LocReg_Ito_mod(data_noisy, g_mat, lam_ini, gamma_init, maxiter):
    # Initialize the MRR Problem
    te = np.arange(1, 512, 4).T
    # Generate the T2 values
    t2 = np.arange(1, 201).T
    dt2 = t2[1] - t2[0]
    # Generate G_matrix
    g_mat = np.zeros((len(te), len(t2)))
    # For every column in each row, fill in the e^(-TE(i))
    for i in range(len(te)):
        for j in range(len(t2)):
            g_mat[i, j] = np.exp(-te[i] / t2[j]) * dt2
    # sigma1 >= 3; cover as many T2 points
    sigma1 = 3
    mu1 = 40
    sigma2 = 10
    mu2 = 160
    # Create ground truth
    g = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((t2 - mu1) ** 2) / (2 * sigma1 ** 2))
    g = g + (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((t2 - mu2) ** 2) / (2 * sigma2 ** 2))
    g = g / 2
    # Generalize, add G and data_noisy in the input; take outside
    # make sure lam_vec is squared??

    def minimize(lam_vector):
        A = (g_mat.T @ g_mat + np.diag(lam_vector))
        b = g_mat.T @ data_noisy
        y = cp.Variable(g_mat.shape[1])
        cost = cp.norm(A @ y - b, 2) ** 2
        constraints = [y >= 0]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.MOSEK, verbose=False)
            # you can try a different solver if you don't have the license
            # for MOSEK doesn't work (should be free for one year)
        except Exception as e:
            print(e)
        sol = y.value
        # reconst,_ = nnls(A,b, maxiter = 10000)
        return sol

    def phi_resid(kern_matrix, param_vec):
        return np.linalg.norm(kern_matrix @ param_vec - data_noisy, 2) ** 2

    def fixed_point_algo(gamma, lam_vector, check):
        """
        gamma: gamma val
        lam_vec: vector of lambdas
        """
        # lam_curr = np.sqrt(lam_vec)
        lam_curr = lam_vector
        k = 1

        ep = 1e-3
        # ep_min = ep
        # epscond = False
        # ini_f_rec = minimize(lam_curr, ep_min, epscond)
        f_old = np.ones(g_mat.shape[1])

        c_arr = []
        lam_arr = []
        sol_arr = []

        # fig, axs = plt.subplots(3, 1, figsize=(6, 6))
        #
        # # # Show the initial plot
        # plt.tight_layout()
        # plt.ion()  # Turn on interactive mode

        # Uncomment the code below to run the Fixed Point Algo (FPA) using while loop until convergence

        while True:
            #Initial L-Curve solution
            try:
                # curr_f_rec = minimize(lam_curr, ep_min, epscond)
                curr_f_rec = minimize(lam_curr)
                if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
                    print(f"curr_f_rec is None after minimization for iteration {k}")
                else:
                    pass
            except Exception as e:
                print("An error occurred during minimization:", e)


            # if (np.linalg.norm(psi_lam/phi_new)) < 1e-4:
            #     print("condition passed")
            #     print("np.linalg.norm(psi_lam/ phi_new)", np.linalg.norm(psi_lam/ phi_new))
            #     psi_lam = psi_lam + ep_min
            # psi_lam = list(psi_lam)


            #Get new solution with new lambda vector

            if check == True:
                axs[0].plot(t2,g, color = "black", label = "ground truth")
                axs[0].plot(t2, curr_f_rec, label = "reconstruction")
                axs[1].semilogy(t2, lam_curr, label = "lambdas")
                # Redraw the plot
                plt.draw()
                plt.tight_layout()
                plt.pause(0.01)


            #Update lambda: then check
            #New Lambda find the new residual and the new penalty
            phi_new = np.linalg.norm(data_noisy - np.dot(g_mat,curr_f_rec), 2)**2
            psi_lam = [lam_curr[i] * curr_f_rec[i] for i in range(len(lam_curr))]
            #define scaling factor;
            c = 1/gamma
            # c = np.std(data_noisy - - np.dot(G,curr_f_rec))/gamma
            c_arr.append(c)

            #STEP 4
            #redefine new lam
            lam_new = c * (phi_new / psi_lam)
            # print("Lam_new.shape", lam_new.shape)
            # machine_eps = np.finfo(float).eps
            # cs = c * np.ones(len(psi_lam))
            psi_lam = np.array(psi_lam)
            print("np.linalg.norm(phi_new/psi_lam):", np.linalg.norm(phi_new/psi_lam))

            #If doesnt converge; update f


            #Step4: Check stopping criteria based on relative change of regularization parameter eta
            #or the  inverse solution
            #update criteria of lambda
            if (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old)) + (np.linalg.norm(lam_new-lam_curr)/np.linalg.norm(lam_curr)) < ep or k >= maxiter:
                # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",
                (np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old))
                # print("ep value: ", ep)
                c_arr_fin = np.array(c_arr)
                lam_arr_fin = np.array(lam_arr)
                sol_arr_fin = np.array(sol_arr)
                plt.ioff()
                plt.show()
                print(f"Total of {k} iterations")
                return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
            else:
                # ep_min = ep_min / 1.2
                # if ep_min <= 1e-4:
                #     ep_min = 1e-4
                # print(f"Finished Iteration {k}")
                lam_curr = lam_new
                f_old = curr_f_rec
                k = k + 1
                lam_arr.append(lam_new)
                sol_arr.append(curr_f_rec)
        #
        # # Running the FPA iteration by iteration
        # testiter = 5
        # for k in range(testiter):
        #     try:
        #         # curr_f_rec = minimize(lam_curr, ep_min, epscond)
        #         curr_f_rec = minimize(lam_curr)
        #         if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
        #             print(f"curr_f_rec is None after minimization for iteration {k}")
        #             break
        #         else:
        #             pass
        #     except Exception as e:
        #         print("An error occurred during minimization:", e)
        #
        #     # Get new solution with new lambda vector
        #
        #     # Update lambda: then check
        #     # New Lambda find the new residual and the new penalty
        #     phi_new = np.linalg.norm(data_noisy - np.dot(g_mat, curr_f_rec), 2) ** 2
        #     psi_lam = [curr_f_rec[i] * lam_curr[i] for i in range(len(lam_curr))]
        #     # define scaling factor;
        #     # c = 1 / (1 + gamma)
        #     c = ((gamma ** gamma) / ((1 + gamma) ** (1 + gamma)))
        #     # c = np.std(data_noisy - - np.dot(G,curr_f_rec))/gamma
        #     # c = ((gamma**gamma)/((1+gamma)**(1+gamma)))/gamma
        #     c_arr.append(c)
        #     # STEP 4
        #     # redefine new lam
        #     lam_new = c * (phi_new / psi_lam)
        #     # Make terms into arrays
        #     psi_lam = np.array(psi_lam)
        #     lam_new = np.array(lam_new)
        #     machine_eps = np.finfo(float).eps
        #
        #     # Try Yvonne's idea
        #     if np.any(psi_lam / phi_new) < machine_eps:
        #         print("condition satisfied")
        #
        #     # If doesnt converge; update f
        #
        #     # Plot iteration by iteration
        #     if check==True:
        #         axs[0].plot(t2, g, color="black", label="ground truth")
        #         axs[0].plot(t2, curr_f_rec, label="reconstruction")
        #         axs[1].semilogy(t2, lam_curr, label="lambdas")
        #         axs[2].semilogy(t2, lam_new, label="new lambda")
        #         # axs[3].semilogy(T2, test, label="lambda_new")
        #         # axs[4].semilogy(T2, np.array(psi_lam2), label="lambda_new * reconstruction")
        #
        #         # Redraw the plot
        #         plt.draw()
        #         # axs[0].legend()
        #         # axs[1].legend()
        #         # axs[2].legend()
        #         # axs[3].legend()
        #         # axs[4].legend()
        #         plt.tight_layout()
        #         plt.pause(0.001)
        #     else:
        #         pass
        #
        #     # Step4: Check stopping criteria based on relative change of regularization parameter eta
        #     # or the  inverse solution
        #     # update criteria of lambda
        #     if ((np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) +
        #             (np.linalg.norm(lam_new - lam_curr) / np.linalg.norm(
        #             lam_curr)) < ep or k == maxiter - 1 or k >= maxiter):
        #         # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old))
        #         # print("ep value: ", ep)
        #         c_arr_fin = np.array(c_arr)
        #         lam_arr_fin = np.array(lam_arr)
        #         sol_arr_fin = np.array(sol_arr)
        #         plt.ioff()
        #         plt.show()
        #         print(f"Total of {k} iterations")
        #         return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
        #     else:
        #         # ep_min = ep_min / 1.2
        #         # if ep_min <= 1e-4:
        #         #     ep_min = 1e-4
        #         # print(f"Finished Iteration {k}")
        #         k += 1
        #         lam_curr = lam_new
        #         f_old = curr_f_rec
        #         lam_arr.append(lam_new)
        #         sol_arr.append(curr_f_rec)

    # MAIN CODE FOR ITO LR:

    # Step 1: Initialize gamma and lambda as lam_vec
    lam_vec = lam_ini * np.ones(g_mat.shape[1])

    # Step 2:Run FPA until convergence
    best_f_rec1, fin_lam1, c_arr_fin1, lam_arr_fin, sol_arr_fin = fixed_point_algo(gamma_init, lam_vec, check=False)
    # print("first FPA is done")

    # Step 3: Calculate new noise level (phi_resid)
    new_resid = phi_resid(g_mat, best_f_rec1)

    # Step 4: Calculate and update new gamma:
    zero_vec = np.zeros(len(best_f_rec1))
    zero_resid = phi_resid(g_mat, zero_vec)

    # If residual is L2:
    gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25
    # If residual is L1:
    # gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.5
    # Step 4: Perform fixed point algo with new gamma value
    # check = True ; if want to print iteration by ieration




    best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin, sol_arr_fin = fixed_point_algo(gamma_new, fin_lam1, check=False)

    return best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin, sol_arr_fin


te = np.arange(1, 512, 4).T
# Generate the T2 values
t2 = np.arange(1, 201).T
dt2 = t2[1] - t2[0]
# Generate G_matrix
g_mat = np.zeros((len(te), len(t2)))
# For every column in each row, fill in the e^(-TE(i))
for i in range(len(te)):
    for j in range(len(t2)):
        g_mat[i, j] = np.exp(-te[i] / t2[j]) * dt2

# sigma1 >= 3; cover as many T2 points
sigma1 = 3
mu1 = 40
sigma2 = 10
mu2 = 160
# Create ground truth
g = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((t2 - mu1) ** 2) / (2 * sigma1 ** 2))
g = g + (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((t2 - mu2) ** 2) / (2 * sigma2 ** 2))
g = g / 2
# Generalize, add G and data_noisy in the input; take outside
# make sure lam_vec is squared??
data_noiseless = np.dot(g_mat, g)
SNR = 200
SD_noise = 1 / SNR
noise = np.random.normal(0, SD_noise, size=data_noiseless.shape)
data_noisy = data_noiseless + noise
def minimize(lam_vector):
    A = (g_mat.T @ g_mat + np.diag(lam_vector))
    b = g_mat.T @ data_noisy
    y = cp.Variable(g_mat.shape[1])
    cost = cp.norm(A @ y - b, 2) ** 2
    constraints = [y >= 0]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    try:
        problem.solve(solver=cp.MOSEK, verbose=False)
        # you can try a different solver if you don't have the license
        # for MOSEK doesn't work (should be free for one year)
    except Exception as e:
        print(e)
    sol = y.value
    # reconst,_ = nnls(A,b, maxiter = 10000)
    return sol

def phi_resid(kern_matrix, param_vec):
    return np.linalg.norm(kern_matrix @ param_vec - data_noisy, 2) ** 2

def fixed_point_algo(gamma, lam_vector, check):
    """
    gamma: gamma val
    lam_vec: vector of lambdas
    """
    # lam_curr = np.sqrt(lam_vec)
    lam_curr = lam_vector
    k = 1

    ep = 1e-3
    # ep_min = ep
    # epscond = False
    # ini_f_rec = minimize(lam_curr, ep_min, epscond)
    f_old = np.ones(g_mat.shape[1])

    c_arr = []
    lam_arr = []
    sol_arr = []

    # fig, axs = plt.subplots(3, 1, figsize=(6, 6))
    #
    # # # Show the initial plot
    # plt.tight_layout()
    # plt.ion()  # Turn on interactive mode
    # Running the FPA iteration by iteration
    testiter = 5
    for k in range(testiter):
        try:
            # curr_f_rec = minimize(lam_curr, ep_min, epscond)
            curr_f_rec = minimize(lam_curr)
            if curr_f_rec is None or any(elem is None for elem in curr_f_rec):
                print(f"curr_f_rec is None after minimization for iteration {k}")
                break
            else:
                pass
        except Exception as e:
            print("An error occurred during minimization:", e)

        # Get new solution with new lambda vector

        # Update lambda: then check
        # New Lambda find the new residual and the new penalty
        phi_new = np.linalg.norm(data_noisy - np.dot(g_mat, curr_f_rec), 2) ** 2
        psi_lam = [curr_f_rec[i] * lam_curr[i] for i in range(len(lam_curr))]
        # define scaling factor;
        # c = 1 / (1 + gamma)
        c = ((gamma ** gamma) / ((1 + gamma) ** (1 + gamma)))
        # c = np.std(data_noisy - - np.dot(G,curr_f_rec))/gamma
        # c = ((gamma**gamma)/((1+gamma)**(1+gamma)))/gamma
        c_arr.append(c)
        # STEP 4
        # redefine new lam
        lam_new = c * (phi_new / psi_lam)
        # Make terms into arrays
        psi_lam = np.array(psi_lam)
        lam_new = np.array(lam_new)
        machine_eps = np.finfo(float).eps

        # Try Yvonne's idea
        if np.any(psi_lam / phi_new) < machine_eps:
            print("condition satisfied")

        # If doesnt converge; update f

        # Plot iteration by iteration
        if check==True:
            axs[0].plot(t2, g, color="black", label="ground truth")
            axs[0].plot(t2, curr_f_rec, label="reconstruction")
            axs[1].semilogy(t2, lam_curr, label="lambdas")
            axs[2].semilogy(t2, lam_new, label="new lambda")
            # axs[3].semilogy(T2, test, label="lambda_new")
            # axs[4].semilogy(T2, np.array(psi_lam2), label="lambda_new * reconstruction")
            # Redraw the plot
            plt.draw()
            # axs[0].legend()
            # axs[1].legend()
            # axs[2].legend()
            # axs[3].legend()
            # axs[4].legend()
            plt.tight_layout()
            plt.pause(0.001)
        else:
            pass

        # Step4: Check stopping criteria based on relative change of regularization parameter eta
        # or the  inverse solution
        # update criteria of lambda
        if ((np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) +
                (np.linalg.norm(lam_new - lam_curr) / np.linalg.norm(
                lam_curr)) < ep or k == maxiter - 1 or k >= maxiter):
            # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old))
            # print("ep value: ", ep)
            c_arr_fin = np.array(c_arr)
            lam_arr_fin = np.array(lam_arr)
            sol_arr_fin = np.array(sol_arr)
            plt.ioff()
            plt.show()
            print(f"Total of {k} iterations")
            return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
        else:
            # ep_min = ep_min / 1.2
            # if ep_min <= 1e-4:
            #     ep_min = 1e-4
            # print(f"Finished Iteration {k}")
            k += 1
            lam_curr = lam_new
            f_old = curr_f_rec
            lam_arr.append(lam_new)
            sol_arr.append(curr_f_rec)

# MAIN CODE FOR ITO LR:

# Step 1: Initialize gamma and lambda as lam_vec
lam_ini = 1e-3
lam_vec = lam_ini * np.ones(g_mat.shape[1])
maxiter = 50

# Step 2:Run FPA until convergence
gamma_init = 10
best_f_rec1, fin_lam1, c_arr_fin1, lam_arr_fin, sol_arr_fin = fixed_point_algo(gamma_init, lam_vec, check=False)
# print("first FPA is done")

# Step 3: Calculate new noise level (phi_resid)
new_resid = phi_resid(g_mat, best_f_rec1)

# Step 4: Calculate and update new gamma:
zero_vec = np.zeros(len(best_f_rec1))
zero_resid = phi_resid(g_mat, zero_vec)

# If residual is L2:
gamma_new = gamma_init * (new_resid / (0.05 * zero_resid)) ** 0.25
# If residual is L1:
# gamma_new = gamma_init * (new_resid/ (0.05 * zero_resid))**0.5
# Step 4: Perform fixed point algo with new gamma value
# check = True ; if want to print iteration by ieration
# lam_curr = np.sqrt(lam_vec)







lam_vector = fin_lam1
test_gam = gamma_new
lam_curr = lam_vector
k = 1

ep = 1e-3
# ep_min = ep
# epscond = False
# ini_f_rec = minimize(lam_curr, ep_min, epscond)
f_old = np.ones(g_mat.shape[1])

c_arr = []
lam_arr = []
sol_arr = []

curr_f_rec = minimize(lam_curr)

# Get new solution with new lambda vector
phi_new = np.linalg.norm(data_noisy - np.dot(g_mat, curr_f_rec), 2) ** 2
psi_lam = [curr_f_rec[i] * lam_curr[i] for i in range(len(lam_curr))]

c = ((gamma ** gamma) / ((1 + gamma) ** (1 + gamma)))
c_arr.append(c)
# STEP 4
# redefine new lam
lam_new = c * (phi_new / psi_lam)
# Make terms into arrays
psi_lam = np.array(psi_lam)
lam_new = np.array(lam_new)
machine_eps = np.finfo(float).eps

# If doesnt converge; update f

# Plot iteration by iteration
if check == True:
    axs[0].plot(t2, g, color="black", label="ground truth")
    axs[0].plot(t2, curr_f_rec, label="reconstruction")
    axs[1].semilogy(t2, lam_curr, label="lambdas")
    axs[2].semilogy(t2, lam_new, label="new lambda")
    # axs[3].semilogy(T2, test, label="lambda_new")
    # axs[4].semilogy(T2, np.array(psi_lam2), label="lambda_new * reconstruction")
    # Redraw the plot
    plt.draw()
    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    # axs[3].legend()
    # axs[4].legend()
    plt.tight_layout()
    plt.pause(0.001)
else:
    pass

    # Step4: Check stopping criteria based on relative change of regularization parameter eta
    # or the  inverse solution
    # update criteria of lambda
if ((np.linalg.norm(curr_f_rec - f_old) / np.linalg.norm(f_old)) +
        (np.linalg.norm(lam_new - lam_curr) / np.linalg.norm(
            lam_curr)) < ep or k == maxiter - 1 or k >= maxiter):
    # print("(np.linalg.norm(new_f_rec-curr_f_rec)/np.linalg.norm(curr_f_rec)): ",(np.linalg.norm(curr_f_rec-f_old)/np.linalg.norm(f_old))
    # print("ep value: ", ep)
    c_arr_fin = np.array(c_arr)
    lam_arr_fin = np.array(lam_arr)
    sol_arr_fin = np.array(sol_arr)
    plt.ioff()
    plt.show()
    print(f"Total of {k} iterations")
    return curr_f_rec, lam_new, c_arr_fin, lam_arr_fin, sol_arr_fin
else:
    # ep_min = ep_min / 1.2
    # if ep_min <= 1e-4:
    #     ep_min = 1e-4
    # print(f"Finished Iteration {k}")
    k += 1
    lam_curr = lam_new
    f_old = curr_f_rec
    lam_arr.append(lam_new)
    sol_arr.append(curr_f_rec)

# best_f_rec2, fin_lam2, c_arr_fin2, lam_arr_fin, sol_arr_fin = fixed_point_algo(gamma_new, fin_lam1, check=False)