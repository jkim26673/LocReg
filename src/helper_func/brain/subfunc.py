from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

#Brain Processing
def gen_maskeddata(brain_data:np.ndarray, mask:np.ndarray):
    _,_,s = brain_data.shape
    ones_array = np.ones(s)
    expanded_mask = mask[:, :, np.newaxis] * ones_array
    masked_data = expanded_mask * brain_data
    return masked_data

#Errors Functions
def wass_error(IdealModel,reconstr):
    err = wasserstein_distance(IdealModel,reconstr)
    return err

def l2_error(IdealModel,reconstr):
    true_norm = linalg_norm(IdealModel)
    err = linalg_norm(IdealModel - reconstr) / true_norm
    return err

def cvxpy_tikhreg(Lambda, G, data_noisy):
    """
    Alternative way of performing non-negative Tikhonov regularization.
    """
    return nonnegtik_hnorm

def add_noise(data, SNR):
    SD_noise = np.max(np.abs(data)) / SNR  # Standard deviation of noise
    noise = np.random.normal(0, SD_noise, size=data.shape)  # Add noise
    dat_noisy = data + noise
    return dat_noisy, noise

def choose_error(pdf1, pdf2,err_type):
    """
    choose_error: Helps select error type based on input
    err_type(string): "Wassterstein" or "Wass. Score"
    vec1(np.array): first probability density function (PDF) you wish to compare to.
    vec2(np.array): second probability density funciton (PDF) you wish to compare to vec1.
    
    typically we select pdf1 to be the ground truth, and we select pdf2 to be our reconstruction method from regularization.
    ex. pdf1 = gt
        pdf2 = GCV solution (f_rec_GCV)
    """
    if err_type == "Wass. Score" or "Wassterstein":
        err = wass_error(pdf1, pdf2)
    else:
        err = l2_error(pdf1, pdf2)
    return err

def compute_MWF(f_rec, T2, Myelin_idx):
    """
    Normalize f_rec and compute MWF.
    Inputs: 
    1. f_rec (np.array type): the reconstruction generated from Tikhonov regularization 
    2. T2 (np.array): the transverse relaxation time constant vector in ms
    3. Myelin_idx (np.array):the indices of T2 where myelin is present (e.g. T2 < 40 ms)
    
    Outputs:
    1. MWF (float): the myelin water fraction
    2. f_rec_normalized (np.array type): the normalized reconstruction generated from Tikhonov regularization. 
        Normalized to 1.
    
    Test Example:
    1) f_rec = np.array([1,2,3,4,5,6])
    2) np.trapz(f_rec_normalized, T2) ~ 1.
    """
    # f_rec_normalized = f_rec / np.trapz(f_rec, T2)
    # print("np.sum(f_rec)*dTE",np.sum(f_rec)*dT)
    # f_rec_normalized = f_rec / (np.sum(f_rec)*dT)
    try:
        f_rec_normalized = f_rec / (np.sum(f_rec) * dT)
    except ZeroDivisionError:
        epsilon = 0.0001
        f_rec_normalized = f_rec / (epsilon)
        print("Division by zero encountered, using epsilon:", epsilon)
    total_MWF = np.cumsum(f_rec_normalized)
    MWF = total_MWF[Myelin_idx[-1][-1]]
    return f_rec_normalized, MWF

def get_signals(coord_pairs, mask_array, unfiltered_arr, A, dT):
    """
    Extracts signals from the brain data at the specified coordinates,
    normalizes them using NNLS, and returns the signals list.
    
    Args:
        coord_pairs (list of tuple): List of coordinates to extract signals from.
        mask_array (numpy.ndarray): Mask array.
        unfiltered_arr (numpy.ndarray): Unfiltered brain data array.
        A (numpy.ndarray): The matrix A.
        dT (float): Time step.
    
    Returns:
        list: List of normalized signals.
    """
    signals = []
    SNRs = []
    for (x_coord, y_coord) in coord_pairs:
        mask_value = mask_array[x_coord, y_coord]
        signal = unfiltered_arr[x_coord, y_coord, :]
        SNR = SNR_map[x_coord,y_coord]
        # sol1 = nnls(A, signal)[0]
        # factor = np.sum(sol1) * dT
        # signal = signal / factor
        signals.append(signal)
        SNRs.append(SNR)
        print(f"Coordinate: ({x_coord}, {y_coord}), Mask value: {mask_value}")
    return np.array(signals), np.array(SNR)


def calculate_noise(signals):
    """
    Calculates the noise by computing the root sum of squared deviations
    between each signal and the mean signal, and randomly flipping signs.
    
    Args:
        signals (numpy.ndarray): Array of normalized signals.
    
    Returns:
        numpy.ndarray: The noise array with random signs.
    """
    # normalized_signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)
    # signals = normalized_signals
    # normalized_signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)
    mean_sig = np.mean(signals, axis=0)
    # squared_diff = (signals - mean_sig) ** 2
    # deviations = signals - mean_sig
    # sum_squared_diff = np.sum(squared_diff, axis=0)
    # noise = np.sqrt(sum_squared_diff)
    # random_signs = np.random.choice([-1, 1], size=noise.shape)
    # noise_stddev = np.std(deviations, axis=0)
    squared_diff = (signals - mean_sig) ** 2
    sum_squared_diff = np.sum(squared_diff, axis=0)
    noise_stddev = np.sqrt(sum_squared_diff)[0]
    mean_sig_val = mean_sig[0]
    return  mean_sig_val, noise_stddev 

def filter_and_compute_MWF(reconstr, tol = 1e-6):
    all_close_to_zero = np.all(np.abs(reconstr) < tol)
    if np.all(reconstr[:-1] == 0) or all_close_to_zero:
        noisy_MWF = 0
        noisy_f_rec = reconstr
        Flag_Val = 1
    else:
        noisy_f_rec, noisy_MWF = compute_MWF(reconstr, T2, Myelin_idx)
        Flag_Val = 0
    return noisy_f_rec, noisy_MWF, Flag_Val

tolerance = 1e-6