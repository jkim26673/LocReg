from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

def regularized_deconvolution(G, T2, data_noisy, Jacobian, Lambda_vec_l1, Lambda_vec, ep=1e-3):
    """
    Perform regularized deconvolution using locreg, L1 and L2 regularization techniques.

    Parameters:
        G (ndarray): The G matrix representing the convolution kernel.
        data_noisy (ndarray): Noisy data obtained from the convolution.
        Jacobian (ndarray): The Jacobian matrix used in the deconvolution.
        Lambda_vec_l1 (ndarray): A vector of L1 regularization parameters.
        Lambda_vec (ndarray): A vector of L2 regularization parameters.
        ep (float): A small value used for iterative regularization.

    Returns:
        ndarray: x0_ini, the optimized solution obtained from iterative localized regularization.
        ndarray: x0_ini_l1, the optimized solution obtained from L1 regularization.
        ndarray: x_lc_vec[:, ireg_corner], the optimized solution obtained from L2 regularization.
    """

    # L1 regularization
    x_lc_vec_l1 = np.zeros((len(T2), len(Lambda_vec_l1)))
    rhos_l1 = np.zeros(len(Lambda_vec_l1)).T
    etas_l1 = np.zeros(len(Lambda_vec_l1)).T

    # for j, lambda_val in enumerate(Lambda_vec_l1):
    #     x = cp.Variable(len(T2))
    #     objective = cp.Minimize(cp.norm(Jacobian @ x - G.T @ data_noisy, 2) + lambda_val * cp.norm(x, 1))
    #     constraints = [x >= 0]
    #     problem = cp.Problem(objective, constraints)
    #     problem.solve(solver=cp.MOSEK, verbose=False)
    #     # try:
    #     #     # Try using the 'MOSEK' solver
    #     #     problem.solve(solver=cp.MOSEK, verbose=False)
    #     # except cp.SolverError:
    #     #     # If 'MOSEK' fails, try using the 'ECOS' solver
    #     #     problem.solve(solver=cp.ECOS, verbose=False)
    #     # except cp.SolverError:
    #     #     problem.solve(solver=cp.SCS, verbose=False)

    #     x_lc_vec_l1[:, j] = x.value.flatten()
    #     rhos_l1[j] = np.linalg.norm(data_noisy - G @ x_lc_vec_l1[:, j]) ** 2
    #     etas_l1[j] = np.linalg.norm(x_lc_vec_l1[:, j], 1)
    for j, lambda_val in enumerate(Lambda_vec_l1):
        x = cp.Variable(len(T2))
        objective = cp.Minimize(cp.norm(Jacobian @ x - G.T @ data_noisy, 2) + lambda_val * cp.norm(x, 1))
        constraints = [x >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=False)
        #ECOS
        #x_lc_vec_l1[:, j] = x.value.flatten()
        x_lc_vec_l1[:, j] = x.value.flatten()
        rhos_l1[j] = np.linalg.norm(data_noisy - G @ x_lc_vec_l1[:, j]) ** 2
        etas_l1[j] = np.linalg.norm(x_lc_vec_l1[:, j], 1)

    ireg_corner_l1 = l_curve_corner(rhos_l1, etas_l1, Lambda_vec_l1)[1]
    x0_ini_l1 = x_lc_vec_l1[:, ireg_corner_l1]

    # L2 regularization
    x_lc_vec = np.zeros((len(T2), len(Lambda_vec)))
    rhos = np.zeros(len(Lambda_vec)).T
    etas = np.zeros(len(Lambda_vec)).T

    for j, lambda_val in enumerate(Lambda_vec):
        x_lc_vec[:, j] = nnls(Jacobian + (lambda_val ** 2) * np.eye(len(T2)), G.T @ data_noisy)[0]
        rhos[j] = np.linalg.norm(data_noisy - G @ x_lc_vec[:, j]) ** 2
        etas[j] = np.linalg.norm(x_lc_vec[:, j]) ** 2

    ireg_corner = l_curve_corner(rhos, etas, Lambda_vec)[1]
    x0_ini = x_lc_vec[:, ireg_corner]

    x0_LS,_,estimated_noise = lsqnonneg(G, data_noisy)

    # Iterative regularization
    x0 = x0_ini

    n = 1
    prev_x = x0_ini
    estimated_noise_std = np.std(estimated_noise)
    track = []

    while True:
        lambda_val = np.std(estimated_noise) / (np.abs(x0_ini) + ep)
        LHS = Jacobian + np.diag(lambda_val)
        RHS = G.T @ data_noisy + (Jacobian * ep) @ np.ones(len(T2)) + ep * lambda_val
        x0_ini = lsqnonneg(LHS, RHS)[0]
        x0_ini = x0_ini - ep
        x0_ini[x0_ini < 0] = 0

        curr_noise = (G @ x0_ini) - data_noisy
        delta_p = np.linalg.solve(LHS, G.T @ curr_noise)
        prev = np.linalg.norm(delta_p)
        LHS_temp = LHS.copy()

        while True:
            x0_ini = x0_ini - delta_p
            x0_ini[x0_ini < 0] = 0
            curr_noise = np.dot(G,x0_ini) - data_noisy
            delta_p = np.linalg.solve(LHS_temp, G.T @ curr_noise)
            if np.abs(np.linalg.norm(delta_p) / prev - 1) < 1e-3:
                break
            prev = np.linalg.norm(delta_p)

        x0_ini = x0_ini - delta_p
        x0_ini[x0_ini < 0] = 0
        # ax.plot(T2, x0_ini)
        # plt.draw()

        track.append(np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x))

        if (np.linalg.norm(x0_ini - prev_x) / np.linalg.norm(prev_x)) < 1e-2:
            break
        
        # ax.plot(T2, x0_ini)
        # plt.draw()

        ep = ep / 1.1
        if ep <= 1e-6:
            ep = 1e-6
        n = n + 1
        prev_x = x0_ini

    return x0_ini, x0_ini_l1, x_lc_vec[:, ireg_corner]

def l2_rms(locreg, l2, l1, g):
    """
    Calculate the normalized root-mean-square (RMS) errors for Loc-Reg (Iterative Regularization),
    L2 L-curve, and L1 L-curve solutions, compared to the ground truth g.

    Parameters:
        locreg (ndarray): Loc-Reg solution.
        l2 (ndarray): L2 L-curve solution.
        l1 (ndarray): L1 L-curve solution.
        g (ndarray): Ground truth.

    Returns:
        ndarray: Array containing the normalized L2 RMS errors for Loc-Reg, L2 L-curve, and L1 L-curve.
    """
    g_truth = np.linalg.norm(g)
    def error (exp,truth):
        return np.linalg.norm(exp-truth)
    #normalized locreg error
    locreg_err = error(locreg, g)/g_truth
    #normalized L2 error
    l2_error = error(l2,g)/g_truth
        # normalized L1 error
    l1_error = error(l1,g)/g_truth
    return np.array([locreg_err,l2_error,l1_error])

def kl_div(locreg,l2,l1,g):
    """
    Calculate the Kullback-Leibler divergence for Loc-Reg, L2 L-curve, and L1 L-curve solutions,
    compared to the ground truth g.

    Parameters:
        locreg (ndarray): Loc-Reg solution.
        l2 (ndarray): L2 L-curve solution.
        l1 (ndarray): L1 L-curve solution.
        g (ndarray): Ground truth.

    Returns:
        ndarray: Array containing the KL divergence for Loc-Reg, L2 L-curve, and L1 L-curve.
    """
    #compare locreg with ground truth
    locreg_err = entropy(locreg, g)
    #compare L2 with ground truth
    l2_error = entropy(l2, g)
    #compare L1 with ground truth
    l1_error = entropy(l1, g)
    return np.array([locreg_err,l2_error,l1_error])

def wass_m(T2, gamma, gnd_truth_dist, locreg, l2, l1):
    emd_locreg = wasserstein_distance(T2, T2+gamma, u_weights=gnd_truth_dist, v_weights=locreg)
    emd_l2 = wasserstein_distance(T2, T2+gamma, u_weights=gnd_truth_dist, v_weights=l2)
    emd_l1 = wasserstein_distance(T2, T2+gamma, u_weights=gnd_truth_dist, v_weights=l1)
    return np.array([emd_locreg,emd_l2,emd_l1])

def get_plots(opt_gamma_list, T2, g, locreg, l1, l2, kl=True, l2_rms = True, wass = True):
    shifted_locreg = np.interp(T2 + opt_gamma_list[0], T2, locreg)
    shifted_l2 = np.interp(T2 + opt_gamma_list[1], T2, l2)
    shifted_l1 = np.interp(T2 + opt_gamma_list[2], T2, l1)
    fig, ax = plt.subplots()
    ax.plot(T2, g, linewidth=2)
    ax.plot(T2, shifted_locreg, linewidth=2)
    ax.plot(T2, shifted_l2, linewidth=2)
    ax.plot(T2, shifted_l1, linewidth=2)
    ax.legend(['True', 'Loc-Reg', 'l2 L-curve', 'l1 L-curve'], fontsize=14)
    ax.set_xlabel('T2', fontsize=18)
    ax.set_ylabel('Amplitude', fontsize=18)
    if kl == True:
        ax.set_title("KL-divergence" +
                    f"\nOpt. Gamma for LocReg = {opt_gamma_list[0]}" +
                    f"\nOpt. Gamma for L2 = {opt_gamma_list[1]}" +
                    f"\nOpt. Gamma for L1 = {opt_gamma_list[2]}", fontsize=8, y=1.0)
    elif l2_rms == True:
        ax.set_title("L2_RMS" +
                    f"\nOpt. Gamma for LocReg = {opt_gamma_list[0]}" +
                    f"\nOpt. Gamma for L2 = {opt_gamma_list[1]}" +
                    f"\nOpt. Gamma for L1 = {opt_gamma_list[2]}", fontsize=8, y=1.0)
    elif wass == True:
        ax.set_title("Wassterstein Score" +
                    f"\nOpt. Gamma for LocReg = {opt_gamma_list[0]}" +
                    f"\nOpt. Gamma for L2 = {opt_gamma_list[1]}" +
                    f"\nOpt. Gamma for L1 = {opt_gamma_list[2]}", fontsize=8, y=1.0)
    return fig

def get_scores(gamma, T2, g, locreg, l1, l2):
    kl_scores_list = []
    l2_rmsscores_list = []
    wass_scores_list = []

    shifted_locreg = np.interp(T2 + gamma, T2, locreg)
    shifted_l2 = np.interp(T2 + gamma, T2, l2)
    shifted_l1 = np.interp(T2 + gamma, T2, l1)
    # fig, ax = plt.subplots()
    # ax.plot(T2, g, linewidth=2)
    #ax.plot(T2, locreg, linewidth=2)
    #ax.plot(T2, l2, linewidth=2)
    #ax.plot(T2, l1, linewidth=2)
    
    # Calculate the kl_score and l2_rmsscore for this gamma value
    kl_scores_gamma = kl_div(shifted_locreg, shifted_l2, shifted_l1, g)
    l2_rmsscores_gamma = l2_rms(shifted_locreg, shifted_l2, shifted_l1, g)
    wass_scores_gamma = wass_m(T2, gamma, g, locreg, l2, l1)

    # Add the kl_score and l2_rmsscore as text annotations on the plot
    # ax.text(0.7, 0.9, f"KL Score:\nLoc-Reg: {kl_scores_gamma[0]:.4f}\nl2 L-curve: {kl_scores_gamma[1]:.4f}\nl1 L-curve: {kl_scores_gamma[2]:.4f}",
    #         transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # ax.text(0.7, 0.6, f"L2 RMS Score:\nLoc-Reg: {l2_rmsscores_gamma[0]:.4f}\nl2 L-curve: {l2_rmsscores_gamma[1]:.4f}\nl1 L-curve: {l2_rmsscores_gamma[2]:.4f}",
    #         transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # ax.text(0.7, 0.3, f"Wass. Score:\nLoc-Reg: {wass_scores_gamma[0]:.4f}\nl2 L-curve: {wass_scores_gamma[1]:.4f}\nl1 L-curve: {wass_scores_gamma[2]:.4f}",
    #         transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # # Set the legend size
    # ax.legend(['True', 'Loc-Reg', 'l2 L-curve', 'l1 L-curve'], fontsize=10, loc = "upper left")

    # # Adjust the title position
    # ax.set_title(f"Gamma = {gamma}", fontsize=14, y=1.02)
    #fig
    return kl_scores_gamma, l2_rmsscores_gamma, wass_scores_gamma


# def find_min_gamma(gamma_list, metric_list):
#     locreg =  [arr[0] for arr in metric_list]
#     l2 = [arr[1] for arr in metric_list]
#     l1 = [arr[2] for arr in metric_list]
#     locreg_ind = np.argmin(locreg)
#     opt_gam_locreg = gamma_list[locreg_ind]
#     l2_ind = np.argmin(l2)
#     opt_gam_l2 = gamma_list[l2_ind]
#     l1_ind = np.argmin(l1)
#     opt_gam_l1 = gamma_list[l1_ind]
#     array = np.array([opt_gam_locreg,opt_gam_l2,opt_gam_l1])
#     return array

import numpy as np

def find_min_gamma(gamma_list, metric_list):
    try:
        locreg = [arr[0] for arr in metric_list]
        l2 = [arr[1] for arr in metric_list]
        l1 = [arr[2] for arr in metric_list]
    except IndexError as e:
        raise ValueError("Each element in 'metric_list' must be a list or array with at least three elements.") from e

    locreg_ind = np.argmin(locreg)
    opt_gam_locreg = gamma_list[locreg_ind]

    l2_ind = np.argmin(l2)
    opt_gam_l2 = gamma_list[l2_ind]

    l1_ind = np.argmin(l1)
    opt_gam_l1 = gamma_list[l1_ind]

    array = np.array([opt_gam_locreg, opt_gam_l2, opt_gam_l1])
    return array


def gen_opt_gamma_plot(gammas_list, kl_scores_list, l2_rmsscores_list, wass_scores_list):
    methods = ['Loc-Reg', 'l2 L-curve', 'l1 L-curve']
    num_methods = len(methods)
    
    # Calculate the scores for each method outside the loop
    kl_scores = [[kl_scores[i] for kl_scores in kl_scores_list] for i in range(num_methods)]
    l2_rmsscores = [[l2_rmsscores[i] for l2_rmsscores in l2_rmsscores_list] for i in range(num_methods)]
    wass_scores = [[wass_scores[i] for wass_scores in wass_scores_list] for i in range(num_methods)]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot KL Scores
    for i, method in enumerate(methods):
        axes[0].scatter(gammas_list, kl_scores[i], label=method, s=50)

    # Add legend, labels, and title
    axes[0].legend(fontsize=14)
    axes[0].set_xlabel('Gamma', fontsize=18)
    axes[0].set_ylabel('KL Score', fontsize=18)
    axes[0].set_title('KL Score vs. Gamma', fontsize=18)
    axes[0].grid(True)

    # Plot L2 RMS Scores
    for i, method in enumerate(methods):
        axes[1].scatter(gammas_list, l2_rmsscores[i], label=method, s=50)

    # Add legend, labels, and title
    axes[1].legend(fontsize=14)
    axes[1].set_xlabel('Gamma', fontsize=18)
    axes[1].set_ylabel('L2 RMS Score', fontsize=18)
    axes[1].set_title('L2 RMS Score vs. Gamma', fontsize=18)
    axes[1].grid(True)

    # Plot Wasserstein Scores
    for i, method in enumerate(methods):
        axes[2].scatter(gammas_list, wass_scores[i], label=method, s=50)

    # Add legend, labels, and title
    axes[2].legend(fontsize=14)
    axes[2].set_xlabel('Gamma', fontsize=18)
    axes[2].set_ylabel('Wasserstein Score', fontsize=18)
    axes[2].set_title('Wasserstein Score vs. Gamma', fontsize=18)
    axes[2].grid(True)

    return fig

# Example usage:
# Assuming you have the necessary data for gammas_list, kl_scores_list, l2_rmsscores_list, and wass_scores_list
# fig, axes = gen_opt_gamma_plot(gammas_list, kl_scores_list, l2_rmsscores_list, wass_scores_list)
# plt.show()  # Display the plots (optional)



def main():
    # Generate the TE values/time
    TE = np.arange(1, 512, 4).T
    # Generate the T2 values
    T2 = np.arange(1, 201).T
    # Generate G_matrix
    G = np.exp(-TE[:, np.newaxis] / T2)

    np.random.seed(34)
    #need to shift T2 - gamma

    nTE = len(TE)
    nT2 = len(T2)
    sigma1 = 2
    mu1 = 40
    sigma2 = 6
    mu2 = 100

    #Create ground truth
    g = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((T2 - mu1) ** 2) / (2 * sigma1 ** 2))
    g = g + (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((T2 - mu2) ** 2) / (2 * sigma2 ** 2))
    g = g / 2

    data_noiseless = G @ g

    Jacobian = G.T @ G

    U, s, V = svd(G, full_matrices=False)
    s = np.diag(s)

    SNR = 1000
    SD_noise = 1 / SNR
    Lambda_vec_l1 = np.logspace(-5, 2, 20)
    Lambda_vec = np.logspace(-2, 2, 10)

    noise = np.random.normal(0, SD_noise, size=data_noiseless.shape)
    data_noisy = data_noiseless + noise
    # List to store plots
    plot_list = []
    kl_scores_list = []
    l2_rmsscores_list = []
    wass_scores_list = []

    locreg, l1, l2 = regularized_deconvolution(G, T2, data_noisy, Jacobian, Lambda_vec_l1, Lambda_vec)

    #gammas = np.linspace(0,40,10)
    gammas_list = np.linspace(-40,40,200)

    def process_gamma(gamma):
        try:
            kl_scores, l2_rmsscores, wass_scores = get_scores(gamma, T2, g, locreg, l1, l2)
            #get optimum lambda
            return kl_scores, l2_rmsscores, wass_scores
        
        except Exception as e:
            print(f"Error occurred for gamma = {gamma}. Skipping plot generation. Error: {str(e)}")
            return None, None, None

    # locreg, l1, l2 = regularized_deconvolution(G, T2, data_noisy, Jacobian, Lambda_vec_l1, Lambda_vec)
    # kl_scores, l2_rmsscores, wass_scores = get_scores(gamma_test, T2, g, locreg, l1, l2)

    # opt_kl = find_min_gamma(gammas_list, kl_scores)
    # opt_l2_rms = find_min_gamma(gammas_list, l2_rmsscores)
    # opt_wass = find_min_gamma(gammas_list, wass_scores)
    # kl_plot = get_plots(opt_kl, T2, g, locreg, l1, l2, kl=True, l2_rms=False, wass = False)
    # l2_rms_plot = get_plots(opt_l2_rms, T2, g, locreg, l1, l2, kl=False, l2_rms=True, wass = False)
    # wass_plot = get_plots(opt_wass, T2, g, locreg, l1, l2, kl=False, l2_rms=False, wass = True)
    # for gamma in tqdm(gammas):
    #     kl_scores_list.append(kl_scores)
    #     l2_rmsscores_list.append(l2_rmsscores)
    #     wass_scores_list.append(wass_scores)


    # Use parallel processing to generate plots for all gamma values
    try:
        import concurrent.futures

        # Set the number of threads based on the number of gamma values to process
        num_threads = min(gammas_list.size, os.cpu_count())

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Process all gamma values in parallel
            results = list(tqdm(executor.map(process_gamma, gammas_list), total=gammas_list.size))

            for kl_scores, l2_rmsscores, wass_scores in results:
                kl_scores_list.append(kl_scores)
                l2_rmsscores_list.append(l2_rmsscores)
                wass_scores_list.append(wass_scores)
            # for kl_scores, l2_rmsscores, wass_scores in results:
            #     kl_scores_list.append(kl_scores)
            #     l2_rmsscores_list.append(l2_rmsscores)
            #     wass_scores_list.append(wass_scores)
                # if figure is not None:
                #     plot_list.append(figure)
                #     kl_scores_list.append(kl_scores)
                #     l2_rmsscores_list.append(l2_rmsscores)
                #     wass_scores_list.append(wass_scores)
    except ImportError:
        # Use sequential processing if concurrent.futures is not available (Python < 3.2)
        for gamma in tqdm(gammas_list):
            kl_scores, l2_rmsscores, wass_scores = process_gamma(gamma)
            kl_scores_list.append(kl_scores)
            l2_rmsscores_list.append(l2_rmsscores)
            wass_scores_list.append(wass_scores)

            # if figure is not None:
            #     plot_list.append(figure)
            #     kl_scores_list.append(kl_scores)
            #     l2_rmsscores_list.append(l2_rmsscores)
            #     wass_scores_list.append(wass_scores)
    opt_kl = find_min_gamma(gammas_list, kl_scores_list)
    opt_l2_rms = find_min_gamma(gammas_list, l2_rmsscores_list)
    opt_wass = find_min_gamma(gammas_list, wass_scores_list)

    #get plots of optimum lambda parameters
    kl_plot = get_plots(opt_kl, T2, g, locreg, l1, l2, kl=True, l2_rms=False, wass = False)
    l2_rms_plot = get_plots(opt_l2_rms, T2, g, locreg, l1, l2, kl=False, l2_rms=True, wass = False)
    wass_plot = get_plots(opt_wass, T2, g, locreg, l1, l2, kl=False, l2_rms=False, wass = True)
    gamma_opt_plot = gen_opt_gamma_plot(gammas_list, kl_scores_list, l2_rmsscores_list, wass_scores_list)

    if kl_plot is not None:
        plot_list.append(kl_plot)
    if l2_rms_plot is not None:
        plot_list.append(l2_rms_plot)
    if wass_plot is not None:
        plot_list.append(wass_plot)
    if gamma_opt_plot is not None:
        plot_list.append(gamma_opt_plot)

    # # Save all plots in a single PDF
    pdf_filename = "Opt_gamma_for_deconvolution_plots.pdf"
    with PdfPages(pdf_filename) as pdf:
        # Generate the plots and save them to the PDF
        for plt_obj in plot_list:
            if plt_obj is not None:
                pdf.savefig(plt_obj)

    # Print the number of plots generated
    # print("Number of plots:", len(plot_list))
    # print(kl_scores_list)
    # print(l2_rmsscores_list)
    # print(wass_scores_list)

    # print(f"Optimal gamma value for kl: {find_min_gamma(gammas,kl_scores_list)}")
    # print(f"Optimal gamma value for l2 norm: {find_min_gamma(gammas,l2_rmsscores_list)}")
    # print(f"Optimal gamma value for wassterstein: {find_min_gamma(gammas,wass_scores_list)}")
    return kl_scores_list, l2_rmsscores_list, wass_scores_list
if __name__ == "__main__":
    kl_scores_list, l2_rmsscores_list, wass_scores_list = main()
    # import matplotlib.pyplot as plt
    # matplotlib.use('TkAgg')
