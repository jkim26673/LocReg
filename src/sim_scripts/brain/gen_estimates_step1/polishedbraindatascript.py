from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
from src.utils.load_imports.load_classical import *
from src.helper_func.brain.subfunc import *

# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.addpath(r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\ZamaUPEN\1D_test', nargout=0)

#Load data
mask = scipy.io.loadmat(paths.MASK_PATH)["new_BW"]
SNR_map = scipy.io.loadmat(paths.FILT_SNR_MAP_PATH)["SNR_MAP"]
# unfilt_SNR_map = scipy.io.loadmat(paths.UNFILT_SNR_MAP_PATH)["new_BW"]
clean_data = scipy.io.loadmat(paths.CLEAN_BR_DATA_PATH)["final_data_2"]
raw_data = scipy.io.loadmat(paths.RAW_BR_DATA_PATH)["brain_data"]

#Load masked data
brain_data = gen_maskeddata(brain_data=clean_data, mask = mask)
p,q,s = brain_data.shape
#Naming file
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')
# cwd_full = path_funcs.gen_results_dir(paths.ROOT_DIR, "noise_addition_exp",f"{month}{day}{year}")
# data_path = os.path.join(r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/results/brain/noise_addition_exp", f"{month}{day}{year}")
data_path = os.path.join(r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp", f"{month}{day}{year}")

add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_filtered_NA_GCV_LR012_UPEN"
data_head = "est_table"
data_tag = (f"{data_head}_{add_tag}{day}{month}{year}")
# data_folder = (os.getcwd() + f'/{data_path}')
data_folder = os.path.join(os.getcwd(), f"{data_path}")
os.makedirs(data_folder, exist_ok = True)
logging.info(f"Save Folder for Final Estimates Table {data_folder})")
# import matlab.engine

# Ensure log directory exists
log_dir = data_folder
try:
    os.makedirs(log_dir, exist_ok=True)
except Exception as e:
    print(f"Error creating log directory: {e}")
    sys.exit(1)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers = []  # Clear existing handlers

# File handler for braindatascript.log
brain_log_path = os.path.join(log_dir, 'braindatascript.log')
try:
    file_handler = logging.FileHandler(brain_log_path)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error setting up braindatascript.log: {e}")

# File handler for app.log
app_log_path = os.path.join(log_dir, 'app.log')
try:
    app_handler = logging.FileHandler(app_log_path)
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(file_format)
    logger.addHandler(app_handler)
except Exception as e:
    print(f"Error setting up app.log: {e}")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(file_format)
logger.addHandler(console_handler)

logging.info("Logging initialized successfully")
logging.basicConfig(
    filename='braindatascript.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Log messages to a file named app.log
        logging.StreamHandler()  # Output log to console
    ]
)
# Create a logger
logger = logging.getLogger()
# Set up logging to file
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Set up logging to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(file_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

addingNoise = True

# SpanReg_level = 200

target_iterator = [(a,b) for a in range(p) for b in range(q)]
# target_iterator = [(80,160)]
print(target_iterator)
logging.debug(f'Target Iterator Length len({target_iterator})')

#Parallelization Switch
parallel = False
num_cpus_avail = os.cpu_count()
#LocReg Hyperparameters
eps1 = 1e-2
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
exp = 0.5
feedback = True
lam_ini_val = "LCurve"
gamma_init = 0.5
maxiter = 500

#CVXPY global parameters
eps = 1e-2

#Error Metric
err_type = "Wass. Score"

#Lambda Space
Lambda = np.logspace(-6,1,50).reshape(-1,1)
# curr_SNR = 1

preset_noise = False
unif_noise = True

#Key Functions
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

def minimize_OP(Lambda, data_noisy, G, nT2, g):
    """
    Calculates the oracle lambda solution. Iterates over many lambdas in Alpha_vec
    """
    OP_x_lc_vec = np.zeros((nT2, len(Lambda)))
    OP_rhos = np.zeros((len(Lambda)))
    for j in (range(len(Lambda))):
        sol = nonnegtik_hnorm(G, data_noisy, Lambda[j], '0', nargin=4)[0]
        OP_x_lc_vec[:, j] = sol
        # Calculate the error (rho)
        OP_rhos[j] = choose_error(g, OP_x_lc_vec[:, j], err_type = "Wass. Score")
    #Find the minimum value of errors, its index, and the corresponding lambda and reconstruction
    min_rhos = min(OP_rhos)
    min_index = np.argmin(OP_rhos)
    min_x = Lambda[min_index][0]
    OP_min_alpha1 = min_x
    OP_min_alpha1_ind = min_index
    f_rec_OP_grid = OP_x_lc_vec[:, OP_min_alpha1_ind]
    return f_rec_OP_grid, OP_min_alpha1, min_rhos , min_index

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
    try:
        f_rec_normalized = f_rec / (np.sum(f_rec) * dT)
    except ZeroDivisionError:
        epsilon = 0.0001
        f_rec_normalized = f_rec / (epsilon)
        print("Division by zero encountered, using epsilon:", epsilon)
    total_MWF = np.cumsum(f_rec_normalized)
    MWF = total_MWF[Myelin_idx[-1][-1]]
    return f_rec_normalized, MWF


def curve_plot(method, x_coord, y_coord, frec, curr_data, lambda_vals, curr_SNR, MWF, filepath):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # First plot: T2 vs f_rec
    axs[0].plot(T2, frec)
    axs[0].set_title('T2 vs f_rec')
    axs[0].set_xlabel('T2')
    axs[0].set_ylabel('f_rec')

    # Second plot: TE vs curr_data
    axs[1].plot(TE, curr_data)
    axs[1].set_title('TE vs Decay Data')
    axs[1].set_xlabel('TE')
    axs[1].set_ylabel('curr_data')

    # Third plot: T2 vs lambda
    axs[2].plot(T2, lambda_vals * np.ones(len(T2)))
    axs[2].set_title('T2 vs Lambda')
    axs[2].set_xlabel('T2')
    axs[2].set_ylabel('lambda')
    # Set the main title with curr_SNR and MWF value
    fig.suptitle(f'{method} Plots for x={x_coord}, y={y_coord} | SNR={curr_SNR}, MWF={MWF}', fontsize=16)
    # Save the figure
    plt.savefig(f"{filepath}/{method}_recon_xcoord{x_coord}_ycoord{y_coord}.png")
    print(f"savefig xcoord{x_coord}_ycoord{y_coord}")
    plt.close('all')
    return 
#We set GCV as ground truth
filepath = data_folder


def calculate_noise(signals):
    """
    Calculates the noise by computing the root sum of squared deviations
    between each signal and the mean signal, and randomly flipping signs.
    
    Args:
        signals (numpy.ndarray): Array of normalized signals.
    
    Returns:
        numpy.ndarray: The noise array with random signs.
    """
    mean_sig = np.mean(signals, axis=0)
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
        signals.append(signal)
        SNRs.append(SNR)
        print(f"Coordinate: ({x_coord}, {y_coord}), Mask value: {mask_value}")
    return np.array(signals), np.array(SNR)

def generate_spanregbrain(i_param_combo, seed=None):
    # Instead of creating a DataFrame, we'll return a dictionary
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
        # x_coord = 153
        # y_coord = 105
    
    # eliminate voxels with pure noise
    if brain_data[x_coord, y_coord][0] < 50:
        print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
        return {}  # Return empty dict instead of empty DataFrame
    else:
        # normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
        passing_x_coord = x_coord
        passing_y_coord = y_coord

        # passing_y_coord = 105
        curr_data = brain_data[passing_x_coord, passing_y_coord, :]
        curr_SNR = SNR_map[passing_x_coord, passing_y_coord]
        sol1 = nnls(A, curr_data)[0]
        factor = np.sum(sol1) * dT
        curr_data = curr_data/factor
        # logging.info(f"Processed voxel sol1 {sol1}")
        ref_rec, ref_lamb = GCV_NNLS(curr_data, A, Lambda)
        ref_rec = ref_rec[:, 0]
        ref_lamb = np.squeeze(ref_lamb)
        # ref_rec, ref_MWF, ref_Flag_Val = filter_and_compute_MWF(ref_rec, tol=1e-6)

        LRIto_ini_lam = ref_lamb
        f_rec_ini = ref_rec
        ref_rec, ref_lamb, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter=50)
        ref_rec, ref_MWF, ref_Flag_Val = filter_and_compute_MWF(ref_rec, tol=1e-6)

        # ref_rec, ref_lamb = GCV_NNLS(curr_data, A, Lambda)
        # ref_rec = ref_rec[:, 0]
        # ref_lamb = np.squeeze(ref_lamb)

        # LRIto_ini_lam = ref_lamb
        # f_rec_ini = ref_rec
        # ref_rec, ref_lamb, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter=50)
        # ref_rec, ref_lamb, LR_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LR, tol=1e-6)

        # ref_rec, ref_lamb = GCV_NNLS(curr_data, A, Lambda)
        # ref_rec = ref_rec[:, 0]
        # ref_lamb = np.squeeze(ref_lamb)
        # ref_rec, ref_MWF, ref_Flag_Val = filter_and_compute_MWF(ref_rec, tol=1e-6)
        data_gt = A @ ref_rec
        unif_noise_val = np.random.normal(0, tail_std, size=32)
        curr_data = data_gt + unif_noise_val

        # After normalizing, do regularization techniques
        # LS reconstruction and MWF calculation

        noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
        noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV, tol=1e-6)

        # print("noisy_MWF_GCV", noisy_MWF_GCV)
        # LocReg reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        noisy_f_rec_LR, noisy_lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter=50)
        noisy_f_rec_LR, noisy_MWF_LR, LR_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LR, tol=1e-6)
        gt = noisy_f_rec_GCV

        # #LocReg1stDeriv reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        f_rec_LR1D, noisy_lambda_LR1D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv(curr_data, A, LRIto_ini_lam, gamma_init, maxiter=50)
        noisy_f_rec_LR1D, noisy_MWF_LR1D, LR_Flag_Val1D = filter_and_compute_MWF(f_rec_LR1D, tol=1e-6)

        # LocReg2ndDeriv reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        f_rec_LR2D, noisy_lambda_LR2D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv2(curr_data, A, LRIto_ini_lam, gamma_init, maxiter=50)
        noisy_f_rec_LR2D, noisy_MWF_LR2D, LR_Flag_Val2D = filter_and_compute_MWF(f_rec_LR2D, tol=1e-6)

        # UPEN reconstruction and MWF calculation
        # std_dev = np.std(curr_data[len(curr_data)-5:])
        # SNR_est = np.max(np.abs(curr_data))/std_dev
        threshold = 1.05 * np.sqrt(A.shape[0]) * np.max(curr_data) / SNR
        noise_norm = threshold
        xex = noisy_f_rec_GCV
        Kmax = 50
        beta_0 = 1e-3
        tol_lam = 1e-5
        noisy_f_rec_UPEN, noisy_lambda_UPEN = UPEN_Zama(A, curr_data, xex, noise_norm, beta_0, Kmax, tol_lam)
        noisy_f_rec_UPEN, noisy_MWF_UPEN, UPEN_Flag_Val = filter_and_compute_MWF(noisy_f_rec_UPEN, tol=1e-6)

        # Return a dictionary instead of DataFrame
        result_dict = {
            "X_val": passing_x_coord,
            'Y_val': passing_y_coord,
            "curr_data": curr_data,
            "noise": unif_noise_val,
            "curr_SNR": curr_SNR,
            "ref_estimate": ref_rec,
            "LR_estimate": noisy_f_rec_LR,
            "LR1D_estimate": noisy_f_rec_LR1D,
            "LR2D_estimate": noisy_f_rec_LR2D,
            "GCV_estimate": noisy_f_rec_GCV,
            "UPEN_estimate": noisy_f_rec_UPEN,
            "MWF_Ref": ref_MWF,
            "MWF_LR": noisy_MWF_LR,
            "MWF_LR1D": noisy_MWF_LR1D,
            "MWF_LR2D": noisy_MWF_LR2D,
            "MWF_GCV": noisy_MWF_GCV,
            "MWF_UPEN": noisy_MWF_UPEN,
            "Lam_LR": noisy_lambda_LR,
            "Lam_LR1D": noisy_lambda_LR1D,
            "Lam_LR2D": noisy_lambda_LR2D,
            "Lam_GCV": noisy_lambda_GCV,
            "Lam_UPEN": noisy_lambda_UPEN,
            "Lam_Ref": ref_lamb,
            "GCV_Flag_Val": GCV_Flag_Val,
            "LR_Flag_Val": LR_Flag_Val,
            "LR1D_Flag_Val": LR_Flag_Val1D,
            "LR2D_Flag_Val": LR_Flag_Val2D,
            "UPEN_Flag_Val": UPEN_Flag_Val,
            "Ref_Flag_Val": ref_Flag_Val
        }
        
        print(f"completed processing for x {passing_x_coord} and y {passing_y_coord}")
        return result_dict

def generate_spanregbrain_simple(i_param_combo, seed=None):
    """
    Simplified version that only extracts basic data without solving inverse problems.
    Returns a dictionary with coordinate, data, noise, SNR, and processed flag.
    """
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
    
    # eliminate voxels with pure noise
    if brain_data[x_coord, y_coord][0] < 50:
        print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
        return {}  # Return empty dict for pixels that don't meet criteria
    else:
        # normalize data
        passing_x_coord = x_coord
        passing_y_coord = y_coord
        
        curr_data = brain_data[passing_x_coord, passing_y_coord, :]
        curr_SNR = SNR_map[passing_x_coord, passing_y_coord]
        
        # Basic normalization
        sol1 = nnls(A, curr_data)[0]
        factor = np.sum(sol1) * dT
        curr_data = curr_data/factor
        logging.info(f"Processed voxel basic normalization for ({passing_x_coord}, {passing_y_coord})")
        
        # Generate noise (if needed)
        unif_noise_val = np.random.normal(0, tail_std, size=32)
        
        # Return simplified dictionary with only essential data
        result_dict = {
            "X_val": passing_x_coord,
            'Y_val': passing_y_coord,
            "curr_data": curr_data,
            "noise": unif_noise_val,
            "curr_SNR": curr_SNR,
            "processed_val": 1  # Flag indicating this pixel was processed
        }
        
        print(f"completed simple processing for x {passing_x_coord} and y {passing_y_coord}")
        return result_dict

def save_dict_list_to_pickle(dict_list, filepath):
    """Helper function to save list of dictionaries to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(dict_list, f)


def load_dict_list_from_pickle(filepath):
    """Helper function to load list of dictionaries from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def convert_dict_list_to_dataframe(dict_list):
    """Convert list of dictionaries to DataFrame when needed for final output"""
    if not dict_list:
        return pd.DataFrame()
    return pd.DataFrame(dict_list)


unif_noise = True
preset_noise = True

if __name__ == "__main__":
    logging.info("Script started.")
    freeze_support()
    dTE = 11.3
    n = 32
    TE = dTE * np.linspace(1, n, n)
    m = 150
    T2 = np.linspace(10, 200, m)
    A = np.zeros((n, m))
    dT = T2[1] - T2[0]
    logging.info(f"T2 range is from 10ms to 200ms with {m} discretizations")
    logging.info(f"dT is {dT}")
    logging.info(f"TE range is {TE}")
    for i in range(n):
        for j in range(m):
            A[i, j] = np.exp(-TE[i]/T2[j]) * dT
    unif_noise = True
    if unif_noise == True:
        num_signals = 1000
        coord_pairs = set()
        brain_data = clean_data 
        while len(coord_pairs) < num_signals:
            x = random.randint(130, 200)
            y = random.randint(100, 200)
            mask_value = mask[x, y]
            if mask_value == 0:
                coord_pairs.add((x, y))
        coord_pairs = list(coord_pairs)
        print("Length", len(coord_pairs))
        signals, SNRs = get_signals(coord_pairs, mask, brain_data, A, dT)
        mean_sig = np.mean(signals, axis=0)
        sol2 = nnls(A, mean_sig)[0]
        factor2 = np.sum(sol2) * dT
        mean_sig = mean_sig / factor2
        tail_length = 3
        tail = mean_sig[-tail_length:]
        SNR = 100
        tail_std = np.abs(np.max(mean_sig))/SNR
        # tail_std = np.std(tail)
    else:
        pass

    logging.info(f"Kernel matrix is size {A.shape} and is form np.exp(-TE[i]/T2[j]) * dT")
    LS_estimates = np.zeros((p, q, m))
    MWF_LS = np.zeros((p, q))
    LR_estimates = np.zeros((p, q, m))
    MWF_LR = np.zeros((p, q))
    LC_estimates = np.zeros((p, q, m))
    MWF_LC = np.zeros((p, q))    
    GCV_estimates = np.zeros((p, q, m))
    MWF_GCV = np.zeros((p, q))
    DP_estimates = np.zeros((p, q, m))
    MWF_DP = np.zeros((p, q))
    OR_estimates = np.zeros((p, q, m))
    MWF_OR = np.zeros((p, q))
    Myelin_idx = np.where(T2 <= 40)
    logging.info("We define myelin index to be less than 40 ms.")
    logging.info("Since this is experimental patient brain data, we do not have the ground truth. Set the ground truth to be GCV.")
    lis = []  # Now this will be a list of dictionaries

    # === Checkpoint Setup ===
    # checkpoint_file = f"{data_folder}/checkpoint.pkl"
    # checkpoint_file = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_06Jun25\checkpoint.pkl"
    checkpoint_file = r"/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925/checkpoint.pkl"
    temp_checkpoint_prefix = f"{data_folder}/temp_checkpoint_"
    checkpoint_interval = 1000
    checkpoint_time_interval = 900     # You can adjust to e.g., 60 seconds
    last_checkpoint_time = time.time()
    temp_checkpoint_count = 0

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            lis = checkpoint_data['lis']
            start_j = checkpoint_data['j']
            start_k = checkpoint_data['k']
            temp_checkpoint_count = checkpoint_data['temp_checkpoint_count']
            logging.info(f"Resumed from checkpoint at j={start_j}, k={start_k}")
    else:
        start_j = 0
        start_k = 0
        logging.info("No checkpoint found, starting fresh.")

    # Assuming the main processing loop is here, like processing rows and columns
    start_time = time.time()  # Start timing

    # Track the overall progress
    total_iterations = p * q  # Assuming a rectangular grid (p rows x q columns)
    completed_iterations = 0  # Initialize the counter for completed iterations

    all_coords = [(j, k) for j in range(p) for k in range(q)]

    try:
        for j in tqdm(range(start_j, p), desc="Processing rows"):
            for k in tqdm(range(start_k if j == start_j else 0, q), desc=f"Cols in row {j}", leave=False):
                result_dict = generate_spanregbrain((j, k))
                if result_dict:  # Check if dictionary is not empty
                    lis.append(result_dict)
                    logging.info(f"Processed voxel ({j}, {k}), total processed: {len(lis)}")

            # Save checkpoint at end of each row
            if lis:
                # Save list of dictionaries directly
                temp_checkpoint_file = f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl"
                save_dict_list_to_pickle(lis, temp_checkpoint_file)
                logging.info(f"Saved temp checkpoint after row {j}: {temp_checkpoint_file}")
                temp_checkpoint_count += 1
                lis = []

            checkpoint_data = {
                'lis': [],  # Already saved in temp file
                'j': j + 1,  # Resume from the next row
                'k': 0,
                'temp_checkpoint_count': temp_checkpoint_count
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            last_checkpoint_time = time.time()

        # Final save - combine all dictionaries and convert to DataFrame only at the end
        final_dict_lists = []
        for i in range(temp_checkpoint_count):
            temp_file = f"{temp_checkpoint_prefix}{i}.pkl"
            if os.path.exists(temp_file):
                dict_list = load_dict_list_from_pickle(temp_file)
                final_dict_lists.extend(dict_list)  # Use extend to flatten the list
                os.remove(temp_file)
                logging.info(f"Cleaned up temp checkpoint {temp_file}")
        
        if lis:
            final_dict_lists.extend(lis)

        if final_dict_lists:
            # Convert to DataFrame only for final save
            df = convert_dict_list_to_dataframe(final_dict_lists)
            df.to_pickle(os.path.join(data_folder, f"{data_tag}.pkl"))
            logging.info(f"Final results saved to {data_folder}/{data_tag}.pkl")

        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logging.info("Removed main checkpoint after successful completion.")

    except Exception as e:
        logging.error(f"Error: {e}")
        if lis:
            # Save dictionaries directly
            save_dict_list_to_pickle(lis, f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'lis': [],
                'j': j,
                'k': k,
                'temp_checkpoint_count': temp_checkpoint_count
            }, f)
        raise


# def generate_spanregbrain(i_param_combo, seed= None):
#     feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "curr_data", "noise", "curr_SNR", 
#                                     "LS_estimate", "MWF_LS", "DP_estimate", "LC_estimate", "LR_estimate", "GCV_estimate", 
#                                     "MWF_DP", "MWF_LC", "MWF_LR", "MWF_GCV", "Lam_DP", "Lam_LC", "Lam_LR", "Lam_GCV"])
#     if parallel == True:
#         x_coord, y_coord = target_iterator[i_param_combo]
#         pass
#     else:
#         x_coord, y_coord = i_param_combo
#         # x_coord = 153
#         # y_coord = 105
#     #eliminate voxels with pure noise
#     if brain_data[x_coord, y_coord][0] < 50:
#         print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
#         return feature_df
#     else:
#         #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
#         passing_x_coord = x_coord
#         passing_y_coord = y_coord

#         # passing_y_coord = 105
#         curr_data = brain_data[passing_x_coord,passing_y_coord,:]
#         curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
#         sol1 = nnls(A, curr_data)[0]
#         factor = np.sum(sol1) * dT
#         curr_data = curr_data/factor
#         logging.info(f"Processed voxel sol1 {sol1}")
#         ref_rec, ref_lamb = GCV_NNLS(curr_data, A, Lambda)
#         ref_rec = ref_rec[:, 0]
#         ref_lamb = np.squeeze(ref_lamb)
#         ref_rec, ref_MWF, ref_Flag_Val = filter_and_compute_MWF(ref_rec, tol = 1e-6)
#         data_gt = A @ ref_rec
#         unif_noise_val = np.random.normal(0, tail_std, size=32)
#         curr_data = data_gt + unif_noise_val

#         #After normalizaing, do regularization techniques
#         #LS reconstruction and MWF calculation

#         noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
#         noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
#         noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
#         noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV, tol = 1e-6)

#         print("noisy_MWF_GCV", noisy_MWF_GCV)
#         #LocReg reconstruction and MWF calculation
#         LRIto_ini_lam = noisy_lambda_GCV
#         f_rec_ini = noisy_f_rec_GCV
#         noisy_f_rec_LR, noisy_lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
#         noisy_f_rec_LR, noisy_MWF_LR, LR_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LR, tol = 1e-6)
#         gt = noisy_f_rec_GCV

#         # #LocReg1stDeriv reconstruction and MWF calculation
#         LRIto_ini_lam = noisy_lambda_GCV
#         f_rec_ini = noisy_f_rec_GCV
#         f_rec_LR1D, noisy_lambda_LR1D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
#         noisy_f_rec_LR1D, noisy_MWF_LR1D, LR_Flag_Val1D = filter_and_compute_MWF(f_rec_LR1D, tol = 1e-6)

#         #LocReg2ndDeriv reconstruction and MWF calculation
#         LRIto_ini_lam = noisy_lambda_GCV
#         f_rec_ini = noisy_f_rec_GCV
#         f_rec_LR2D, noisy_lambda_LR2D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv2(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
#         noisy_f_rec_LR2D, noisy_MWF_LR2D, LR_Flag_Val2D = filter_and_compute_MWF(f_rec_LR2D, tol = 1e-6)

#         #UPEN reconstruction and MWF calculation
#         std_dev = np.std(curr_data[len(curr_data)-5:])
#         SNR_est = np.max(np.abs(curr_data))/std_dev
#         threshold = 1.05 * np.sqrt(A.shape[0]) * np.max(curr_data) / SNR_est
#         noise_norm = threshold
#         xex = noisy_f_rec_GCV
#         Kmax = 50
#         beta_0 = 1e-3
#         tol_lam = 1e-5
#         noisy_f_rec_UPEN, noisy_lambda_UPEN = UPEN_Zama(A, curr_data, xex, noise_norm, beta_0, Kmax, tol_lam)
#         noisy_f_rec_UPEN, noisy_MWF_UPEN, UPEN_Flag_Val = filter_and_compute_MWF(noisy_f_rec_UPEN, tol = 1e-6)

#         feature_df["X_val"] = [passing_x_coord]
#         feature_df["Y_val"] = [passing_y_coord]
#         feature_df["curr_data"] = [curr_data]
#         feature_df["noise"] = [unif_noise_val]
#         feature_df["curr_SNR"] = [curr_SNR]
#         feature_df["ref_estimate"] = [ref_rec]
#         feature_df["LR_estimate"] = [noisy_f_rec_LR]
#         feature_df["LR1D_estimate"] = [noisy_f_rec_LR1D]
#         feature_df["LR2D_estimate"] = [noisy_f_rec_LR2D]
#         feature_df["GCV_estimate"] = [noisy_f_rec_GCV]
#         feature_df["UPEN_estimate"] = [noisy_f_rec_UPEN]
#         feature_df["MWF_Ref"] = [ref_MWF]
#         feature_df["MWF_LR"] = [noisy_MWF_LR]
#         feature_df["MWF_LR1D"] = [noisy_MWF_LR1D]
#         feature_df["MWF_LR2D"] = [noisy_MWF_LR2D]
#         feature_df["MWF_GCV"] = [noisy_MWF_GCV]
#         feature_df["MWF_UPEN"] = [noisy_MWF_UPEN]
#         feature_df["Lam_LR"] = [noisy_lambda_LR]
#         feature_df["Lam_LR1D"] = [noisy_lambda_LR1D]
#         feature_df["Lam_LR2D"] = [noisy_lambda_LR2D]
#         feature_df["Lam_GCV"] = [noisy_lambda_GCV]
#         feature_df["Lam_UPEN"] = [noisy_lambda_UPEN]
#         feature_df["Lam_Ref"] = [ref_lamb]
#         feature_df["GCV_Flag_Val"] = [GCV_Flag_Val]
#         feature_df["LR_Flag_Val"] = [LR_Flag_Val]
#         feature_df["LR1D_Flag_Val"] = [LR_Flag_Val1D]
#         feature_df["LR2D_Flag_Val"] = [LR_Flag_Val2D]
#         feature_df["UPEN_Flag_Val"] = [UPEN_Flag_Val]
#         feature_df["Ref_Flag_Val"] = [ref_Flag_Val]
#         print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
#         return feature_df

# unif_noise = True
# preset_noise = True
# presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"

# if __name__ == "__main__":
#     logging.info("Script started.")
#     freeze_support()
#     dTE = 11.3
#     n = 32
#     TE = dTE * np.linspace(1,n,n)
#     m = 150
#     T2 = np.linspace(10,200,m)
#     A = np.zeros((n,m))
#     dT = T2[1] - T2[0]
#     logging.info(f"T2 range is from 10ms to 200ms with {m} discretizations")
#     logging.info(f"dT is {dT}")
#     logging.info(f"TE range is {TE}")
#     for i in range(n):
#         for j in range(m):
#             A[i,j] = np.exp(-TE[i]/T2[j]) * dT
#     unif_noise = True
#     if unif_noise == True:
#         num_signals = 1000
#         coord_pairs = set()
#         mask_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\masks\new_mask.mat" 
#         mask = scipy.io.loadmat(mask_path)["new_BW"]
#         brain_data = scipy.io.loadmat(r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\braindata\cleaned_brain_data (1).mat")["final_data_2"]
#         while len(coord_pairs) < num_signals:
#             x = random.randint(130, 200)
#             y = random.randint(100, 200)
#             mask_value = mask[x, y]
#             if mask_value == 0:
#                 coord_pairs.add((x, y))
#         coord_pairs = list(coord_pairs)
#         print("Length", len(coord_pairs))
#         signals, SNRs = get_signals(coord_pairs, mask, brain_data, A, dT)
#         mean_sig = np.mean(signals, axis=0)
#         sol2 = nnls(A, mean_sig)[0]
#         factor2 = np.sum(sol2) * dT
#         mean_sig = mean_sig / factor2
#         tail_length = 3
#         tail = mean_sig[-tail_length:]
#         tail_std = np.std(tail)
#     else:
#         pass

#     logging.info(f"Kernel matrix is size {A.shape} and is form np.exp(-TE[i]/T2[j]) * dT")
#     LS_estimates = np.zeros((p,q,m))
#     MWF_LS = np.zeros((p,q))
#     LR_estimates = np.zeros((p,q,m))
#     MWF_LR = np.zeros((p,q))
#     LC_estimates = np.zeros((p,q,m))
#     MWF_LC = np.zeros((p,q))    
#     GCV_estimates = np.zeros((p,q,m))
#     MWF_GCV = np.zeros((p,q))
#     DP_estimates = np.zeros((p,q,m))
#     MWF_DP = np.zeros((p,q))
#     OR_estimates = np.zeros((p,q,m))
#     MWF_OR = np.zeros((p,q))
#     Myelin_idx = np.where(T2<=40)
#     logging.info("We define myelin index to be less than 40 ms.")
#     logging.info("Since this is experimental patient brain data, we do not have the ground truth. Set the ground truth to be GCV.")
#     lis = []

#     # === Checkpoint Setup ===
#     checkpoint_file = f"C:/Users/kimjosy/Downloads/LocReg_Regularization-1/data/Brain/results_{day}{month}{year}/checkpoint.pkl"
#     # checkpoint_file = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_28May25\checkpoint.pkl"
#     temp_checkpoint_prefix = f"{data_folder}/temp_checkpoint_"
#     checkpoint_interval = 1000
#     checkpoint_time_interval = 900     # You can adjust to e.g., 60 seconds
#     last_checkpoint_time = time.time()
#     temp_checkpoint_count = 0

#     if os.path.exists(checkpoint_file):
#         with open(checkpoint_file, 'rb') as f:
#             checkpoint_data = pickle.load(f)
#             lis = checkpoint_data['lis']
#             start_j = checkpoint_data['j']
#             start_k = checkpoint_data['k']
#             temp_checkpoint_count = checkpoint_data['temp_checkpoint_count']
#             logging.info(f"Resumed from checkpoint at j={start_j}, k={start_k}")
#     else:
#         start_j = 0
#         start_k = 0
#         logging.info("No checkpoint found, starting fresh.")

#     # Assuming the main processing loop is here, like processing rows and columns
#     start_time = time.time()  # Start timing

#     # Track the overall progress
#     total_iterations = p * q  # Assuming a rectangular grid (p rows x q columns)
#     completed_iterations = 0  # Initialize the counter for completed iterations

#     all_coords = [(j, k) for j in range(p) for k in range(q)]
#     # sample_size = 1000
#     # if len(all_coords) < sample_size:
#     #     raise ValueError("Not enough pixels to sample from!")
#     # random_coords = random.sample(all_coords, sample_size)

#     try:
#         for j in tqdm(range(start_j, p), desc="Processing rows"):
#             for k in tqdm(range(start_k if j == start_j else 0, q), desc=f"Cols in row {j}", leave=False):
#                 estimates_dataframe = generate_spanregbrain((j, k))
#                 if not estimates_dataframe.empty:
#                     lis.append(estimates_dataframe)
#                     logging.info(f"Processed voxel ({j}, {k}), total processed: {len(lis)}")

#             # Save checkpoint at end of each row
#             if lis:
#                 temp_df = pd.concat(lis, ignore_index=True)
#                 temp_checkpoint_file = f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl"
#                 temp_df.to_pickle(temp_checkpoint_file)
#                 logging.info(f"Saved temp checkpoint after row {j}: {temp_checkpoint_file}")
#                 temp_checkpoint_count += 1
#                 lis = []

#             checkpoint_data = {
#                 'lis': [],  # Already saved in temp file
#                 'j': j + 1,  # Resume from the next row
#                 'k': 0,
#                 'temp_checkpoint_count': temp_checkpoint_count
#             }
#             with open(checkpoint_file, 'wb') as f:
#                 pickle.dump(checkpoint_data, f)
#             last_checkpoint_time = time.time()

#         # Final save
#         final_dfs = []
#         for i in range(temp_checkpoint_count):
#             temp_file = f"{temp_checkpoint_prefix}{i}.pkl"
#             if os.path.exists(temp_file):
#                 final_dfs.append(pd.read_pickle(temp_file))
#                 os.remove(temp_file)
#                 logging.info(f"Cleaned up temp checkpoint {temp_file}")
#         if lis:
#             final_dfs.append(pd.concat(lis, ignore_index=True))

#         if final_dfs:
#             df = pd.concat(final_dfs, ignore_index=True)
#             df.to_pickle(os.path.join(data_folder, f"{data_tag}.pkl"))
#             logging.info(f"Final results saved to {data_folder}/{data_tag}.pkl")

#         if os.path.exists(checkpoint_file):
#             os.remove(checkpoint_file)
#             logging.info("Removed main checkpoint after successful completion.")

#     except Exception as e:
#         logging.error(f"Error: {e}")
#         if lis:
#             temp_df = pd.concat(lis, ignore_index=True)
#             temp_df.to_pickle(f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl")
#         with open(checkpoint_file, 'wb') as f:
#             pickle.dump({
#                 'lis': [],
#                 'j': j,
#                 'k': k,
#                 'temp_checkpoint_count': temp_checkpoint_count
#             }, f)
#         raise




#     # try:
#     #     # for j in tqdm(range(start_j, p), desc="Processing rows"):
#     #     #     for k in tqdm(range(start_k if j == start_j else 0, q), desc=f"Cols in row {j}", leave=False):
#     #     #         estimates_dataframe = generate_spanregbrain((j, k))
#     #     #         if not estimates_dataframe.empty:
#     #     #             lis.append(estimates_dataframe)
#     #     #             logging.info(f"Processed voxel ({j}, {k}), total processed: {len(lis)}")
#     #     for j, k in tqdm(all_coords, desc="Processing random pixels"):
#     #         estimates_dataframe = generate_spanregbrain((j, k))
#     #         #Use dictionary to save memory instead of dataframe; 
#     #         #list of dictionary...
#     #         if not estimates_dataframe.empty:
#     #             lis.append(estimates_dataframe)
#     #             logging.info(f"Processed voxel ({j}, {k}), total processed: {len(lis)}")

#     #             completed_iterations += 1
                
#     #             # Calculate the elapsed time and estimate the remaining time
#     #             elapsed_time = time.time() - start_time
#     #             estimated_time_remaining = (elapsed_time / completed_iterations) * (total_iterations - completed_iterations)
                
#     #             # Log the progress and estimated time left
#     #             logging.info(f"Processed row {j}/{p}, column {k}/{q}, "
#     #                         f"elapsed time: {elapsed_time / 60:.2f} minutes, "
#     #                         f"estimated time left: {estimated_time_remaining / 60:.2f} minutes.")
                
#     #             # Save periodically
#     #             should_save = (len(lis) >= checkpoint_interval or
#     #                            (time.time() - last_checkpoint_time) >= checkpoint_time_interval)

#     #             if should_save:
#     #                 if lis:
#     #                     temp_df = pd.concat(lis, ignore_index=True)
#     #                     temp_checkpoint_file = f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl"
#     #                     temp_df.to_pickle(temp_checkpoint_file)
#     #                     logging.info(f"Saved temp checkpoint: {temp_checkpoint_file}")
#     #                     temp_checkpoint_count += 1
#     #                     lis = []

#     #                 checkpoint_data = {
#     #                     'lis': lis,
#     #                     'j': j,
#     #                     'k': k,
#     #                     'temp_checkpoint_count': temp_checkpoint_count
#     #                 }
#     #                 with open(checkpoint_file, 'wb') as f:
#     #                     pickle.dump(checkpoint_data, f)
#     #                 last_checkpoint_time = time.time()

#     #         start_k = 0  # Reset column counter after first row

#     #     # Final Save
#     #     final_dfs = []
#     #     for i in range(temp_checkpoint_count):
#     #         temp_file = f"{temp_checkpoint_prefix}{i}.pkl"
#     #         if os.path.exists(temp_file):
#     #             final_dfs.append(pd.read_pickle(temp_file))
#     #             os.remove(temp_file)
#     #             logging.info(f"Cleaned up temp checkpoint {temp_file}")
#     #     if lis:
#     #         final_dfs.append(pd.concat(lis, ignore_index=True))

#     #     if final_dfs:
#     #         df = pd.concat(final_dfs, ignore_index=True)
#     #         df.to_pickle(os.path.join(data_folder, f"{data_tag}.pkl"))
#     #         logging.info(f"Final results saved to {data_folder}/{data_tag}.pkl")

#     #     if os.path.exists(checkpoint_file):
#     #         os.remove(checkpoint_file)
#     #         logging.info("Removed main checkpoint after successful completion.")

#     # except Exception as e:
#     #     logging.error(f"Error: {e}")
#     #     if lis:
#     #         temp_df = pd.concat(lis, ignore_index=True)
#     #         temp_df.to_pickle(f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl")
#     #     with open(checkpoint_file, 'wb') as f:
#     #         pickle.dump({
#     #             'lis': lis,
#     #             'j': j,
#     #             'k': k,
#     #             'temp_checkpoint_count': temp_checkpoint_count
#     #         }, f)
#     #     raise  # 