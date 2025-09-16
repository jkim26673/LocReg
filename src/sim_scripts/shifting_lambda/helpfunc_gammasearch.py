# import numpy as np
# from scipy.stats import wasserstein_distance, entropy
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

def l2_rms(T2, gamma, data, g):
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
    shifted_data = np.interp(T2 + gamma, T2, data)

    g_truth = np.linalg.norm(g)
    def error (exp,truth):
        return np.linalg.norm(exp-truth)
    #normalized locreg error
    err = error(shifted_data, g)/g_truth
    return np.array([err])


def kl_div(T2, gamma, data, g):
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
    shifted = np.interp(T2 + gamma, T2, data)
    err = entropy(shifted, g)
    return np.array([err])

def wass_m(T2, gamma,data, gnd_truth_dist):
    emd = wasserstein_distance(T2, T2+gamma, u_weights=gnd_truth_dist, v_weights=data)
    # emd_l2 = wasserstein_distance(T2, T2+gamma, u_weights=gnd_truth_dist, v_weights=l2)
    # emd_l1 = wasserstein_distance(T2, T2+gamma, u_weights=gnd_truth_dist, v_weights=l1)
    return np.array([emd])


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

    return kl_scores_gamma, l2_rmsscores_gamma, wass_scores_gamma

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
    return array, locreg_ind, l2_ind, l1_ind


def process_gamma(gamma):
    try:
        kl_scores, l2_rmsscores, wass_scores = get_scores(gamma, T2, IdealModel_weighted, f_rec_LocReg_LC, f_rec_DP, f_rec_LC)
        #get optimum lambda
        return kl_scores, l2_rmsscores, wass_scores
    except Exception as e:
        print(f"Error occurred for gamma = {gamma}. Skipping plot generation. Error: {str(e)}")
        return None, None, None