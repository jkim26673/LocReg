from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

log_dir = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1"
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
# import matlab.engine
# print("successfully")
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

print("setting license path")
logging.info("setting license path")
mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
logging.info(f'MOSEK License Set from {mosek_license_path}')

#Naming Convention for Save Folder for Images:
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

pat_tag = "MRR"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
series_tag = "BrainData_Images"
simulation_save_folder = f"SimulationSets/{pat_tag}/{series_tag}"
# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + simulation_save_folder 
logging.info(f"Save Folder for Brain Images {cwd_full})")

#Simulation Test 1
#Simulation Test 2
#Simulation Test 3
addingNoise = True
#Load Brain data and SNR map from Chuan
# brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization-1/data/brain/braindata/mew_cleaned_brain_data_unfiltered.mat")["brain_data"]
# brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat")["final_data_2"]
brain_data = scipy.io.loadmat(r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\braindata\cleaned_brain_data (1).mat")["final_data_2"]
# SNR_map = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/SNR_map.mat")["SNR_map"]
# brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat")["new_BW"]
# SNR_map = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization-1/data/brain/SNRmap/new_SNR_Map.mat")["SNR_MAP"]
SNR_map =scipy.io.loadmat(r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\SNRmap\new_SNR_Map.mat")["SNR_MAP"]
logging.info(f"brain_data shape {brain_data.shape}")
logging.info(f"SNR_map shape {SNR_map.shape}")
logging.info(f"brain_data and SNR_map from Chuan have been successfully loaded")
print("SNR_map.shape",SNR_map.shape)

#Iterator for Parallel Processing

# brain_data = brain_data[80,160,:]
# brain_data2 = brain_data[80,160,:]
p,q,s = brain_data.shape
# print("p",p)
# print("q",q)
# print("s",s)
# p = 1
# q = 1
# s = 32
#replace minimum SNR of 0 to be 1
SNR_map = np.where(SNR_map == 0, 1, SNR_map)
print("max SNR_map",np.max(SNR_map))
print("min SNR_map",np.min(SNR_map))
#set a minimum SNR of <10; keep the pixel and set MWF to 0.

# brain_data = brain_data2
# print("brain_data.shape",brain_data.shape)
# print("SNR_map.shape",SNR_map.shape)
# SNR = 300
SpanReg_level = 800
# SpanReg_level = 200

target_iterator = [(a,b) for a in range(p) for b in range(q)]
# target_iterator = [(80,160)]
print(target_iterator)
logging.debug(f'Target Iterator Length len({target_iterator})')

#Naming for Saving Data Collected from Script Folder
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')
data_path = f"data/Brain/results_{day}{month}{year}"
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_filtered_myelinmaps_GCV_LR012_UPEN"
add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_filtered_NA_GCV_LR012_UPEN"
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_nofilt_myelinmaps_GCV_LR012_UPEN"
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_nofilt_NA_GCV_LR012_UPEN"

# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_filtered_noiseadditionUPEN"
data_head = "est_table"
data_tag = (f"{data_head}_{add_tag}{day}{month}{year}")
data_folder = (os.getcwd() + f'/{data_path}')
os.makedirs(data_folder, exist_ok = True)
logging.info(f"Save Folder for Final Estimates Table {data_folder})")

#Parallelization Switch
# parallel = False
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
    # lam_vec = Lambda * np.ones(G.shape[1])
    # A = (G.T @ G + np.diag(lam_vec))
    # ep4 = np.ones(A.shape[1]) * eps
    # b = (G.T @ data_noisy) + (G.T @ G @ ep4) + ep4 * lam_vec
    # y = cp.Variable(G.shape[1])
    # cost = cp.norm(A @ y - b, 2)**2
    # constraints = [y >= 0]
    # problem = cp.Problem(cp.Minimize(cost), constraints)
    # problem.solve(solver=cp.MOSEK, verbose=False)
    # sol = y.value
    # sol = sol - eps
    # sol = np.maximum(sol, 0)

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
        # try:Lambda
        #     # Fallback to nonnegtik_hnorm
        #     sol, rho, trash = nonnegtik_hnorm(G, data_noisy, Alpha_vec[j], '0', nargin=4)
        #     if np.all(sol == 0):
        #         logging.debug("Solution from nonnegtik_hnorm is a zero vector, switching to CVXPY solver.")
        #         print(f"Solution from nonnegtik_hnorm is a zero vector, switching to CVXPY solver.")
        #         raise ValueError("Zero vector detected, switching to CVXPY.")
        # except Exception as e:
            #CVXPY solution
            # logging.error("Error in nonnegtik_hnorm, using CVXPY")
            # print(f"Error in nonnegtik_hnorm: {e}")
        # sol = cvxpy_tikhreg(Lambda[j], G, data_noisy)
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
    # f_rec_normalized = f_rec / np.trapz(f_rec, T2)
    # print("np.sum(f_rec)*dTE",np.sum(f_rec)*dT)
    # f_rec_normalized = f_rec / (np.sum(f_rec)*dT)
    try:
        f_rec_normalized = f_rec / (np.sum(f_rec) * dT)
        # f_rec_normalized = f_rec / np.trapz(f_rec, T2)
        # f_rec_normalized = f_rec / simpson(y = f_rec, x = T2)
    except ZeroDivisionError:
        epsilon = 0.0001
        f_rec_normalized = f_rec / (epsilon)
        print("Division by zero encountered, using epsilon:", epsilon)
    total_MWF = np.cumsum(f_rec_normalized)
    MWF = total_MWF[Myelin_idx[-1][-1]]
    # MWF = total_MWF[Myelin_idx[-1]]
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

# from Simulations.upencode import upen_param_setup, upen_setup
from Simulations.upenzama import UPEN_Zama
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\ZamaUPEN\1D_test', nargout=0)
preset_noise = False
# presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14Apr25\est_table_xcoordlen_313_ycoordlen_313_filtered_noise_addition_uniform_noise_UPEN_LR1D2D14Apr25 copy.pkl"
presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"
# presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14Apr25\est_table_xcoordlen_313_ycoordlen_313_filtered_noise_addition_uniform_noise_UPEN_LR1D2D14Apr25.pkl"
unif_noise = True
def generate_spanregbrain(i_param_combo, seed= None):
    # print(f"Processing {i_param_combo}") 
    feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "curr_data", "noise", "curr_SNR", 
                                    "LS_estimate", "MWF_LS", "DP_estimate", "LC_estimate", "LR_estimate", "GCV_estimate", 
                                    "MWF_DP", "MWF_LC", "MWF_LR", "MWF_GCV", "Lam_DP", "Lam_LC", "Lam_LR", "Lam_GCV"])
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
        x_coord = 153
        y_coord = 105
    #eliminate voxels with pure noise
    if brain_data[x_coord, y_coord][0] < 50:
        print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
        return feature_df
    else:
        #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
        passing_x_coord = x_coord
        passing_y_coord = y_coord
        curr_data = brain_data[passing_x_coord,passing_y_coord,:]
        curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        
        # if unif_noise == True:
        #     unnormalized_data = curr_data
        #     if preset_noise == True:
        #         with open(presetfilepath, 'rb') as file:
        #             df = pickle.load(file)
        #         unif_noise_val = df["noise"][0]
        #         noise = unif_noise_val
        #         try:
        #             curr_data = df[(df["X_val"] == passing_x_coord) & (df["Y_val"] == passing_y_coord)]["curr_data"].tolist()[0]
        #         except:
        #             sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        #             X = sol1
        #             tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        #             all_close_to_zero = np.all(np.abs(X) < tolerance)
        #             factor = np.sum(sol1) * dT
        #             curr_data = curr_data/factor
        #             curr_data = curr_data + unif_noise_val
        #         try:
        #             noisy_f_rec_GCV = df[(df["X_val"] == passing_x_coord) & (df["Y_val"] == passing_y_coord)]["GCV_estimate"].tolist()[0]
        #         except:
        #             noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        #             noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        #             noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
        #             noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV, tol = 1e-6)
        #     else:
        #         noise = unif_noise_val
        #         curr_data = curr_data + unif_noise_val
            # try:
            #     sol1 = nnls(A, curr_data, maxiter=1e6)[0]
            # except:
            #     try:
            #         sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            #     except:
            #         print("need to skip, cannot find solution to LS solutions for normalizaiton")
            #         return feature_df
            # X = sol1
            # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
            # all_close_to_zero = np.all(np.abs(X) < tolerance)
            # factor = np.sum(sol1) * dT
            # curr_data = curr_data/factor
        # try:
        #     sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        # except:
        #     print("need to skip, cannot find solution to LS solutions for normalizaiton")
        #     return feature_df
        # sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        sol1 = nnls(A, curr_data)[0]
        # X = sol1
        # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
        # all_close_to_zero = np.all(np.abs(X) < tolerance)
        factor = np.sum(sol1) * dT
        curr_data = curr_data/factor
        logging.info(f"Processed voxel sol1 {sol1}")


        ref_rec, ref_lamb = GCV_NNLS(curr_data, A, Lambda)
        ref_rec = ref_rec[:, 0]
        ref_lamb = np.squeeze(ref_lamb)
        ref_rec, ref_MWF, ref_Flag_Val = filter_and_compute_MWF(ref_rec, tol = 1e-6)
        data_gt = A @ ref_rec
        # std_dev = np.std(data_gt[len(data_gt)-5:])
        # SNR_noadd = np.max(np.abs(data_gt))/std_dev
        curr_data = data_gt + unif_noise_val
        # curr_data, noise = add_noise(data_gt, SNR = SpanReg_level)
        # with open(presetfilepath, 'rb') as file:
        #     df = pickle.load(file)
        # unif_noise_val = df["noise"][0]
        # noise = unif_noise_val
        #After normalizaing, do regularization techniques
        #LS reconstruction and MWF calculation

        # try:
        #     noisy_f_rec_LS = nnls(A, curr_data, maxiter=1e6)[0]
        # except:
        #     try:
        #         noisy_f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        #     except:
        #         noisy_f_rec_LS = np.zeros(len(T2))
        #         Flag_Val = 2
        # noisy_f_rec_LS, noisy_MWF_LS, LS_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LS, tol = 1e-6)
        
        # #LCurve reconstruction and MWF calculation
        # noisy_f_rec_LC, noisy_lambda_LC = Lcurve(curr_data, A, Lambda)
        # noisy_f_rec_LC, noisy_MWF_LC, LC_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LC, tol = 1e-6)

        # #GCV reconstruction and MWF calculation

        # #DP reconstruction and MWF calculation
        # noisy_f_rec_DP, noisy_lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
        # noisy_f_rec_DP, noisy_MWF_DP, DP_Flag_Val = filter_and_compute_MWF(noisy_f_rec_DP, tol = 1e-6)

        noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
        noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV, tol = 1e-6)

        #LocReg reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        noisy_f_rec_LR, noisy_lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        noisy_f_rec_LR, noisy_MWF_LR, LR_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LR, tol = 1e-6)
        gt = noisy_f_rec_GCV

        # #LocReg1stDeriv reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        f_rec_LR1D, noisy_lambda_LR1D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        noisy_f_rec_LR1D, noisy_MWF_LR1D, LR_Flag_Val1D = filter_and_compute_MWF(f_rec_LR1D, tol = 1e-6)

        #LocReg2ndDeriv reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        f_rec_LR2D, noisy_lambda_LR2D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv2(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        noisy_f_rec_LR2D, noisy_MWF_LR2D, LR_Flag_Val2D = filter_and_compute_MWF(f_rec_LR2D, tol = 1e-6)

        #UPEN reconstruction and MWF calculation
        # result = upen_param_setup(TE, T2, A, curr_data)
        # noisy_f_rec_UPEN, _ ,_ , noisy_lambda_UPEN= upen_setup(result, curr_data, LRIto_ini_lam, True)
        # noise = unif_noise_val
        # noise_norm = np.linalg.norm(unif_noise_val)
        std_dev = np.std(curr_data[len(curr_data)-5:])
        SNR_est = np.max(np.abs(curr_data))/std_dev
        threshold = 1.05 * np.sqrt(A.shape[0]) * np.max(curr_data) / SNR_est
        # noisenorm = np.linalg.norm(unif_noise_val)
        # noise_norm = np.linalg.norm(noise)
        noise_norm = threshold
        # noise_norm = noisenorm
        xex = noisy_f_rec_GCV
        Kmax = 50
        beta_0 = 1e-3
        tol_lam = 1e-5
        noisy_f_rec_UPEN, noisy_lambda_UPEN = UPEN_Zama(A, curr_data, xex, noise_norm, beta_0, Kmax, tol_lam)
        noisy_f_rec_UPEN, noisy_MWF_UPEN, UPEN_Flag_Val = filter_and_compute_MWF(noisy_f_rec_UPEN, tol = 1e-6)

        feature_df["X_val"] = [passing_x_coord]
        feature_df["Y_val"] = [passing_y_coord]
        feature_df["curr_data"] = [curr_data]
        feature_df["noise"] = [unif_noise_val]
        feature_df["curr_SNR"] = [curr_SNR]
        # feature_df["LS_estimate"] = [noisy_f_rec_LS]
        # feature_df["MWF_LS"] = [noisy_MWF_LS]
        # feature_df["DP_estimate"] = [noisy_f_rec_DP]
        feature_df["ref_estimate"] = [ref_rec]
        feature_df["LR_estimate"] = [noisy_f_rec_LR]
        feature_df["LR1D_estimate"] = [noisy_f_rec_LR1D]
        feature_df["LR2D_estimate"] = [noisy_f_rec_LR2D]
        feature_df["GCV_estimate"] = [noisy_f_rec_GCV]
        feature_df["UPEN_estimate"] = [noisy_f_rec_UPEN]
        # feature_df["MWF_DP"] = [noisy_MWF_DP]
        # feature_df["MWF_LC"] = [noisy_MWF_LC]
        feature_df["MWF_Ref"] = [ref_MWF]
        feature_df["MWF_LR"] = [noisy_MWF_LR]
        feature_df["MWF_LR1D"] = [noisy_MWF_LR1D]
        feature_df["MWF_LR2D"] = [noisy_MWF_LR2D]
        feature_df["MWF_GCV"] = [noisy_MWF_GCV]
        feature_df["MWF_UPEN"] = [noisy_MWF_UPEN]
        # feature_df["Lam_DP"] = [noisy_lambda_DP]
        # feature_df["Lam_LC"] = [noisy_lambda_LC]
        feature_df["Lam_LR"] = [noisy_lambda_LR]
        feature_df["Lam_LR1D"] = [noisy_lambda_LR1D]
        feature_df["Lam_LR2D"] = [noisy_lambda_LR2D]
        feature_df["Lam_GCV"] = [noisy_lambda_GCV]
        feature_df["Lam_UPEN"] = [noisy_lambda_UPEN]
        feature_df["Lam_Ref"] = [ref_lamb]
        # feature_df["LS_Flag_Val"] = [LS_Flag_Val]
        # feature_df["LC_Flag_Val"] = [LC_Flag_Val]
        feature_df["GCV_Flag_Val"] = [GCV_Flag_Val]
        # feature_df["DP_Flag_Val"] = [DP_Flag_Val]
        feature_df["LR_Flag_Val"] = [LR_Flag_Val]
        feature_df["LR1D_Flag_Val"] = [LR_Flag_Val1D]
        feature_df["LR2D_Flag_Val"] = [LR_Flag_Val2D]
        feature_df["UPEN_Flag_Val"] = [UPEN_Flag_Val]
        feature_df["Ref_Flag_Val"] = [ref_Flag_Val]
        print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
        return feature_df


# def generate_brain_estimates2(i_param_combo, seed= None):
#     # print(f"Processing {i_param_combo}") 
#     feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "curr_data", "noise", "curr_SNR", 
#                                     "LS_estimate", "MWF_LS", "DP_estimate", "LC_estimate", "LR_estimate", "GCV_estimate", 
#                                     "MWF_DP", "MWF_LC", "MWF_LR", "MWF_GCV", "Lam_DP", "Lam_LC", "Lam_LR", "Lam_GCV"])
#     if parallel == True:
#         x_coord, y_coord = target_iterator[i_param_combo]
#         pass
#     else:
#         x_coord, y_coord = i_param_combo
#     #eliminate voxels with pure noise
#     if brain_data[x_coord, y_coord][0] < 50:
#         print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
#         return feature_df
#     else:
#         #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
#         passing_x_coord = x_coord
#         passing_y_coord = y_coord
#         curr_data = brain_data[passing_x_coord,passing_y_coord,:]
#         curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
#         # if unif_noise == True:
#         #     unnormalized_data = curr_data
#         #     if preset_noise == True:
#         #         with open(presetfilepath, 'rb') as file:
#         #             df = pickle.load(file)
#         #         unif_noise_val = df["noise"][0]
#         #         noise = unif_noise_val
#         #         try:
#         #             curr_data = df[(df["X_val"] == passing_x_coord) & (df["Y_val"] == passing_y_coord)]["curr_data"].tolist()[0]
#         #         except:
#         #             sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
#         #             X = sol1
#         #             tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
#         #             all_close_to_zero = np.all(np.abs(X) < tolerance)
#         #             factor = np.sum(sol1) * dT
#         #             curr_data = curr_data/factor
#         #             curr_data = curr_data + unif_noise_val
#         #         try:
#         #             noisy_f_rec_GCV = df[(df["X_val"] == passing_x_coord) & (df["Y_val"] == passing_y_coord)]["GCV_estimate"].tolist()[0]
#         #         except:
#         #             noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
#         #             noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
#         #             noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
#         #             noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV, tol = 1e-6)
#         #     else:
#         #         noise = unif_noise_val
#         #         curr_data = curr_data + unif_noise_val
#             # try:
#             #     sol1 = nnls(A, curr_data, maxiter=1e6)[0]
#             # except:
#             #     try:
#             #         sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
#             #     except:
#             #         print("need to skip, cannot find solution to LS solutions for normalizaiton")
#             #         return feature_df
#             # X = sol1
#             # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
#             # all_close_to_zero = np.all(np.abs(X) < tolerance)
#             # factor = np.sum(sol1) * dT
#             # curr_data = curr_data/factor

#         try:
#             sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
#         except:
#                 print("need to skip, cannot find solution to LS solutions for normalizaiton")
#                 return feature_df
#         X = sol1
#         tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
#         all_close_to_zero = np.all(np.abs(X) < tolerance)
#         factor = np.sum(sol1) * dT
#         curr_data = curr_data/factor

#         # with open(presetfilepath, 'rb') as file:
#         #     df = pickle.load(file)
#         # unif_noise_val = df["noise"][0]
#         # noise = unif_noise_val
#         #After normalizaing, do regularization techniques
#         #LS reconstruction and MWF calculation

#         # try:
#         #     noisy_f_rec_LS = nnls(A, curr_data, maxiter=1e6)[0]
#         # except:
#         #     try:
#         #         noisy_f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
#         #     except:
#         #         noisy_f_rec_LS = np.zeros(len(T2))
#         #         Flag_Val = 2
#         # noisy_f_rec_LS, noisy_MWF_LS, LS_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LS, tol = 1e-6)
        
#         # #LCurve reconstruction and MWF calculation
#         # noisy_f_rec_LC, noisy_lambda_LC = Lcurve(curr_data, A, Lambda)
#         # noisy_f_rec_LC, noisy_MWF_LC, LC_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LC, tol = 1e-6)

#         # #GCV reconstruction and MWF calculation

#         # #DP reconstruction and MWF calculation
#         # noisy_f_rec_DP, noisy_lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
#         # noisy_f_rec_DP, noisy_MWF_DP, DP_Flag_Val = filter_and_compute_MWF(noisy_f_rec_DP, tol = 1e-6)
#         noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
#         noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
#         noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
#         noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV, tol = 1e-6)

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
#         # result = upen_param_setup(TE, T2, A, curr_data)
#         # noisy_f_rec_UPEN, _ ,_ , noisy_lambda_UPEN= upen_setup(result, curr_data, LRIto_ini_lam, True)
#         # noise = unif_noise_val
#         # noise_norm = np.linalg.norm(unif_noise_val)
#         # threshold = 1.1 * np.sqrt(A.shape[0]) * np.max(curr_data) / curr_SNR
#         std_dev = np.std(curr_data[len(curr_data)-5:])
#         SNR_est = np.max(np.abs(curr_data))/std_dev
#         threshold = 1.05 * np.sqrt(A.shape[0]) * np.max(curr_data) / SNR_est
#         # noise_norm = np.linalg.norm(noise)
#         noise_norm = threshold
#         xex = noisy_f_rec_GCV
#         Kmax = 50
#         beta_0 = 1e-7
#         tol_lam = 1e-5
#         noisy_f_rec_UPEN, noisy_lambda_UPEN = UPEN_Zama(A, curr_data, xex, noise_norm, beta_0, Kmax, tol_lam)
#         noisy_f_rec_UPEN, noisy_MWF_UPEN, UPEN_Flag_Val = filter_and_compute_MWF(noisy_f_rec_UPEN, tol = 1e-6)

#         feature_df["X_val"] = [passing_x_coord]
#         feature_df["Y_val"] = [passing_y_coord]
#         feature_df["curr_data"] = [curr_data]
#         # feature_df["noise"] = [noise]
#         feature_df["curr_SNR"] = [curr_SNR]
#         # feature_df["LS_estimate"] = [noisy_f_rec_LS]
#         # feature_df["MWF_LS"] = [noisy_MWF_LS]
#         # feature_df["DP_estimate"] = [noisy_f_rec_DP]
#         # feature_df["LC_estimate"] = [noisy_f_rec_LC]
#         feature_df["LR_estimate"] = [noisy_f_rec_LR]
#         feature_df["LR1D_estimate"] = [noisy_f_rec_LR1D]
#         feature_df["LR2D_estimate"] = [noisy_f_rec_LR2D]
#         feature_df["GCV_estimate"] = [noisy_f_rec_GCV]
#         feature_df["UPEN_estimate"] = [noisy_f_rec_UPEN]
#         # feature_df["MWF_DP"] = [noisy_MWF_DP]
#         # feature_df["MWF_LC"] = [noisy_MWF_LC]
#         feature_df["MWF_LR"] = [noisy_MWF_LR]
#         feature_df["MWF_LR1D"] = [noisy_MWF_LR1D]
#         feature_df["MWF_LR2D"] = [noisy_MWF_LR2D]
#         feature_df["MWF_GCV"] = [noisy_MWF_GCV]
#         feature_df["MWF_UPEN"] = [noisy_MWF_UPEN]
#         # feature_df["Lam_DP"] = [noisy_lambda_DP]
#         # feature_df["Lam_LC"] = [noisy_lambda_LC]
#         feature_df["Lam_LR"] = [noisy_lambda_LR]
#         feature_df["Lam_LR1D"] = [noisy_lambda_LR1D]
#         feature_df["Lam_LR2D"] = [noisy_lambda_LR2D]
#         feature_df["Lam_GCV"] = [noisy_lambda_GCV]
#         feature_df["Lam_UPEN"] = [noisy_lambda_UPEN]
#         # feature_df["LS_Flag_Val"] = [LS_Flag_Val]
#         # feature_df["LC_Flag_Val"] = [LC_Flag_Val]
#         feature_df["GCV_Flag_Val"] = [GCV_Flag_Val]
#         # feature_df["DP_Flag_Val"] = [DP_Flag_Val]
#         feature_df["LR_Flag_Val"] = [LR_Flag_Val]
#         feature_df["LR1D_Flag_Val"] = [LR_Flag_Val1D]
#         feature_df["LR2D_Flag_Val"] = [LR_Flag_Val2D]
#         feature_df["UPEN_Flag_Val"] = [UPEN_Flag_Val]
#         print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
#         return feature_df

unif_noise = True
preset_noise = True
presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"
# def generate_brain_estimatesNA(i_param_combo, seed= None):
#     # print(f"Processing {i_param_combo}") 
#     feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "curr_data", "noise", "curr_SNR", 
#                                     "LS_estimate", "MWF_LS", "DP_estimate", "LC_estimate", "LR_estimate", "GCV_estimate", 
#                                     "MWF_DP", "MWF_LC", "MWF_LR", "MWF_GCV", "Lam_DP", "Lam_LC", "Lam_LR", "Lam_GCV"])
#     if parallel == True:
#         x_coord, y_coord = target_iterator[i_param_combo]
#         pass
#     else:
#         x_coord, y_coord = i_param_combo
#     #eliminate voxels with pure noise
#     if brain_data[x_coord, y_coord][0] < 50:
#         print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
#         return feature_df
#     else:
#         #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
#         passing_x_coord = x_coord
#         passing_y_coord = y_coord
#         curr_data = brain_data[passing_x_coord,passing_y_coord,:]
#         curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
#         if unif_noise == True:
#             unnormalized_data = curr_data
#             if preset_noise == True:
#                 with open(presetfilepath, 'rb') as file:
#                     df = pickle.load(file)
#                 unif_noise_val = df["noise"][0]
#                 noise = unif_noise_val
#                 curr_data = curr_data + unif_noise_val
#             # try:
#             #     sol1 = nnls(A, curr_data, maxiter=1e6)[0]
#             # except:
#             #     try:
#             #         sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
#             #     except:
#             #         print("need to skip, cannot find solution to LS solutions for normalizaiton")
#             #         return feature_df
#             # X = sol1
#             # tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
#             # all_close_to_zero = np.all(np.abs(X) < tolerance)
#             # factor = np.sum(sol1) * dT
#             # curr_data = curr_data/factor

#         sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
#         # sol1 = nnls(A, curr_data)[0]
#         # try:
#         #     sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
#         # except:
#         #         print("need to skip, cannot find solution to LS solutions for normalizaiton")
#         #         return feature_df
#         logging.info(f"Processed voxel sol1 {sol1}")
#         X = sol1
#         tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
#         all_close_to_zero = np.all(np.abs(X) < tolerance)
#         factor = np.sum(sol1) * dT
#         curr_data = curr_data/factor

#         # with open(presetfilepath, 'rb') as file:
#         #     df = pickle.load(file)
#         # unif_noise_val = df["noise"][0]
#         # noise = unif_noise_val
#         #After normalizaing, do regularization techniques
#         #LS reconstruction and MWF calculation

#         # try:
#         #     noisy_f_rec_LS = nnls(A, curr_data, maxiter=1e6)[0]
#         # except:
#         #     try:
#         #         noisy_f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
#         #     except:
#         #         noisy_f_rec_LS = np.zeros(len(T2))
#         #         Flag_Val = 2
#         # noisy_f_rec_LS, noisy_MWF_LS, LS_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LS, tol = 1e-6)
        
#         # #LCurve reconstruction and MWF calculation
#         # noisy_f_rec_LC, noisy_lambda_LC = Lcurve(curr_data, A, Lambda)
#         # noisy_f_rec_LC, noisy_MWF_LC, LC_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LC, tol = 1e-6)

#         # #GCV reconstruction and MWF calculation

#         # #DP reconstruction and MWF calculation
#         # noisy_f_rec_DP, noisy_lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
#         # noisy_f_rec_DP, noisy_MWF_DP, DP_Flag_Val = filter_and_compute_MWF(noisy_f_rec_DP, tol = 1e-6)
#         noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
#         noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
#         noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
#         noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV, tol = 1e-6)

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
#         # result = upen_param_setup(TE, T2, A, curr_data)
#         # noisy_f_rec_UPEN, _ ,_ , noisy_lambda_UPEN= upen_setup(result, curr_data, LRIto_ini_lam, True)
#         # noise = unif_noise_val
#         # noise_norm = np.linalg.norm(unif_noise_val)
#         # threshold = 1.1 * np.sqrt(A.shape[0]) * np.max(curr_data) / curr_SNR
#         std_dev = np.std(curr_data[len(curr_data)-5:])
#         SNR_est = np.max(np.abs(curr_data))/std_dev
#         threshold = 1.05 * np.sqrt(A.shape[0]) * np.max(curr_data) / SNR_est
#         # noise_norm = np.linalg.norm(noise)
#         noise_norm = threshold
#         xex = noisy_f_rec_GCV
#         Kmax = 50
#         beta_0 = 1e-7
#         tol_lam = 1e-5
#         noisy_f_rec_UPEN, noisy_lambda_UPEN = UPEN_Zama(A, curr_data, xex, noise_norm, beta_0, Kmax, tol_lam)
#         noisy_f_rec_UPEN, noisy_MWF_UPEN, UPEN_Flag_Val = filter_and_compute_MWF(noisy_f_rec_UPEN, tol = 1e-6)

#         feature_df["X_val"] = [passing_x_coord]
#         feature_df["Y_val"] = [passing_y_coord]
#         feature_df["curr_data"] = [curr_data]
#         feature_df["noise"] = [noise]
#         feature_df["curr_SNR"] = [curr_SNR]
#         # feature_df["LS_estimate"] = [noisy_f_rec_LS]
#         # feature_df["MWF_LS"] = [noisy_MWF_LS]
#         # feature_df["DP_estimate"] = [noisy_f_rec_DP]
#         # feature_df["LC_estimate"] = [noisy_f_rec_LC]
#         feature_df["LR_estimate"] = [noisy_f_rec_LR]
#         feature_df["LR1D_estimate"] = [noisy_f_rec_LR1D]
#         feature_df["LR2D_estimate"] = [noisy_f_rec_LR2D]
#         feature_df["GCV_estimate"] = [noisy_f_rec_GCV]
#         feature_df["UPEN_estimate"] = [noisy_f_rec_UPEN]
#         # feature_df["MWF_DP"] = [noisy_MWF_DP]
#         # feature_df["MWF_LC"] = [noisy_MWF_LC]
#         feature_df["MWF_LR"] = [noisy_MWF_LR]
#         feature_df["MWF_LR1D"] = [noisy_MWF_LR1D]
#         feature_df["MWF_LR2D"] = [noisy_MWF_LR2D]
#         feature_df["MWF_GCV"] = [noisy_MWF_GCV]
#         feature_df["MWF_UPEN"] = [noisy_MWF_UPEN]
#         # feature_df["Lam_DP"] = [noisy_lambda_DP]
#         # feature_df["Lam_LC"] = [noisy_lambda_LC]
#         feature_df["Lam_LR"] = [noisy_lambda_LR]
#         feature_df["Lam_LR1D"] = [noisy_lambda_LR1D]
#         feature_df["Lam_LR2D"] = [noisy_lambda_LR2D]
#         feature_df["Lam_GCV"] = [noisy_lambda_GCV]
#         feature_df["Lam_UPEN"] = [noisy_lambda_UPEN]
#         # feature_df["LS_Flag_Val"] = [LS_Flag_Val]
#         # feature_df["LC_Flag_Val"] = [LC_Flag_Val]
#         feature_df["GCV_Flag_Val"] = [GCV_Flag_Val]
#         # feature_df["DP_Flag_Val"] = [DP_Flag_Val]
#         feature_df["LR_Flag_Val"] = [LR_Flag_Val]
#         feature_df["LR1D_Flag_Val"] = [LR_Flag_Val1D]
#         feature_df["LR2D_Flag_Val"] = [LR_Flag_Val2D]
#         feature_df["UPEN_Flag_Val"] = [UPEN_Flag_Val]
#         print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
#         return feature_df



if __name__ == "__main__":
    logging.info("Script started.")
    freeze_support()
    dTE = 11.3
    n = 32
    TE = dTE * np.linspace(1,n,n)
    m = 150
    T2 = np.linspace(10,200,m)
    A = np.zeros((n,m))
    dT = T2[1] - T2[0]
    logging.info(f"T2 range is from 10ms to 200ms with {m} discretizations")
    logging.info(f"dT is {dT}")
    logging.info(f"TE range is {TE}")
    for i in range(n):
        for j in range(m):
            A[i,j] = np.exp(-TE[i]/T2[j]) * dT
    unif_noise = True
    if unif_noise == True:
        num_signals = 1000
        coord_pairs = set()
        mask_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\masks\new_mask.mat" 
        mask = scipy.io.loadmat(mask_path)["new_BW"]
        brain_data = scipy.io.loadmat(r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\braindata\cleaned_brain_data (1).mat")["final_data_2"]
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
        tail_std = np.std(tail)
        # print("tail_std", tail_std)
        unif_noise_val = np.random.normal(0, tail_std, size=32)
        plt.figure()
        print("unif_noise_val", unif_noise_val)
        plt.plot(unif_noise_val)
        plt.xlabel('Time/Index')
        plt.ylabel('Noise Standard Deviation')
        plt.title('Noise Standard Deviation Across Signals')
        plt.grid(True)
        plt.savefig("testfignoise.png")
        plt.close()
        plt.figure()
        plt.plot(mean_sig)
        plt.xlabel('TE')
        plt.ylabel('Amplitude')
        plt.title('Mean Signal')
        plt.grid(True)
        plt.savefig("testfigmeansig.png")
        plt.close()
    else:
        pass

    logging.info(f"Kernel matrix is size {A.shape} and is form np.exp(-TE[i]/T2[j]) * dT")
    LS_estimates = np.zeros((p,q,m))
    MWF_LS = np.zeros((p,q))
    LR_estimates = np.zeros((p,q,m))
    MWF_LR = np.zeros((p,q))
    LC_estimates = np.zeros((p,q,m))
    MWF_LC = np.zeros((p,q))    
    GCV_estimates = np.zeros((p,q,m))
    MWF_GCV = np.zeros((p,q))
    DP_estimates = np.zeros((p,q,m))
    MWF_DP = np.zeros((p,q))
    OR_estimates = np.zeros((p,q,m))
    MWF_OR = np.zeros((p,q))
    Myelin_idx = np.where(T2<=40)
    logging.info("We define myelin index to be less than 40 ms.")
    logging.info("Since this is experimental patient brain data, we do not have the ground truth. Set the ground truth to be GCV.")
    lis = []
    # checkpoint_file = f"C:/Users/kimjosy/Downloads/LocReg_Regularization-1/data/Brain/results_{day}{month}{year}/checkpoint.pkl"
    # temp_checkpoint_prefix = f"{data_folder}/temp_checkpoint_"
    # checkpoint_interval = 1000  # Save every 1000 valid voxels
    # checkpoint_time_interval = 3600*30  # Save every 5 seconds for testing
    # last_checkpoint_time = time.time()
    # temp_checkpoint_count = 0

    # # Load checkpoint if exists
    # if os.path.exists(checkpoint_file):
    #     with open(checkpoint_file, 'rb') as f:
    #         checkpoint_data = pickle.load(f)
    #         lis = checkpoint_data['lis']
    #         start_j = checkpoint_data['j']
    #         start_k = checkpoint_data['k']
    #         temp_checkpoint_count = checkpoint_data['temp_checkpoint_count']
    #         logging.info(f"Resumed from checkpoint at j={start_j}, k={start_k}, with {len(lis)} voxels processed, temp_checkpoint_count={temp_checkpoint_count}")
    # else:
    #     start_j = 0
    #     start_k = 0
    #     logging.info("No checkpoint found, starting from beginning")

    # try:
    #     for j in tqdm(range(start_j, p), desc="Processing rows", unit="row"):
    #         for k in tqdm(range(start_k if j == start_j else 0, q), desc="Processing columns", unit="col", leave=False):
    #             iteration = (j, k)
    #             estimates_dataframe = generate_brain_estimates2(iteration)
    #             if not estimates_dataframe.empty:
    #                 lis.append(estimates_dataframe)
    #                 logging.info(f"Processed voxel ({j}, {k}), total processed: {len(lis)}")

    #             # Save checkpoint every checkpoint_interval voxels or every checkpoint_time_interval seconds
    #             if len(lis) >= checkpoint_interval or (time.time() - last_checkpoint_time) >= checkpoint_time_interval:
    #                 # Save lis to a temporary pickle and clear it
    #                 if lis:
    #                     temp_df = pd.concat(lis, ignore_index=True)
    #                     temp_checkpoint_file = f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl"
    #                     temp_df.to_pickle(temp_checkpoint_file)
    #                     logging.info(f"Saved temporary checkpoint to {temp_checkpoint_file}, {len(lis)} voxels")
    #                     temp_checkpoint_count += 1
    #                     lis = []  # Clear lis to free memory

    #                 # Save main checkpoint with progress
    #                 checkpoint_data = {
    #                     'lis': lis,
    #                     'j': j,
    #                     'k': k,
    #                     'temp_checkpoint_count': temp_checkpoint_count
    #                 }
    #                 with open(checkpoint_file, 'wb') as f:
    #                     pickle.dump(checkpoint_data, f)
    #                 logging.info(f"Main checkpoint saved at j={j}, k={k}, temp_checkpoint_count={temp_checkpoint_count}")
    #                 last_checkpoint_time = time.time()

    #         # Reset start_k after the first resumed row
    #         start_k = 0

    #     # Save final results
    #     print(f"Completed processing, total voxels: {len(lis)}")
    #     # Combine all temporary checkpoints and remaining lis
    #     final_dfs = []
    #     for i in range(temp_checkpoint_count):
    #         temp_file = f"{temp_checkpoint_prefix}{i}.pkl"
    #         if os.path.exists(temp_file):
    #             temp_df = pd.read_pickle(temp_file)
    #             final_dfs.append(temp_df)
    #             os.remove(temp_file)  # Clean up
    #             logging.info(f"Loaded and removed temporary checkpoint {temp_file}")
    #     if lis:
    #         final_dfs.append(pd.concat(lis, ignore_index=True))

    #     if final_dfs:
    #         df = pd.concat(final_dfs, ignore_index=True)
    #         df.to_pickle(data_folder + f'/' + data_tag + '.pkl')
    #         logging.info(f"Final results saved to {data_folder}/{data_tag}.pkl")
    #     else:
    #         logging.warning("No data to save in final results")

    #     # Remove main checkpoint on completion
    #     if os.path.exists(checkpoint_file):
    #         os.remove(checkpoint_file)
    #         logging.info("Main checkpoint file removed after completion")

    # except Exception as e:
    #     logging.error(f"Error during processing: {e}")
    #     # Save temporary checkpoint and main checkpoint before exiting
    #     if lis:
    #         temp_df = pd.concat(lis, ignore_index=True)
    #         temp_checkpoint_file = f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl"
    #         temp_df.to_pickle(temp_checkpoint_file)
    #         logging.info(f"Saved emergency temporary checkpoint to {temp_checkpoint_file}, {len(lis)} voxels")
    #         temp_checkpoint_count += 1
    #     checkpoint_data = {
    #         'lis': lis,
    #         'j': j,
    #         'k': k,
    #         'temp_checkpoint_count': temp_checkpoint_count
    #     }
    #     with open(checkpoint_file, 'wb') as f:
    #         pickle.dump(checkpoint_data, f)
    #     logging.info(f"Emergency main checkpoint saved at j={j}, k={k}, temp_checkpoint_count={temp_checkpoint_count}")
    #     raise  # Re-raise for debugging

    # === Checkpoint Setup ===
    checkpoint_file = f"C:/Users/kimjosy/Downloads/LocReg_Regularization-1/data/Brain/results_{day}{month}{year}/checkpoint.pkl"
    # checkpoint_file = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_28May25\checkpoint.pkl"
    # checkpoint_file = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_16May25\checkpoint.pkl"
    # checkpoint_file = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_23Apr25\checkpoint.pkl"
    # checkpoint_file = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_26Apr25\checkpoint.pkl"
    # checkpoint_file = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_12May25\checkpoint.pkl"
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

    try:
        for j in tqdm(range(start_j, p), desc="Processing rows"):
            for k in tqdm(range(start_k if j == start_j else 0, q), desc=f"Cols in row {j}", leave=False):
                # estimates_dataframe = generate_brain_estimates2((j, k))
                # estimates_dataframe = generate_brain_estimatesNA((j, k))
                estimates_dataframe = generate_spanregbrain((j, k))
                if not estimates_dataframe.empty:
                    lis.append(estimates_dataframe)
                    logging.info(f"Processed voxel ({j}, {k}), total processed: {len(lis)}")

                completed_iterations += 1
                
                # Calculate the elapsed time and estimate the remaining time
                elapsed_time = time.time() - start_time
                estimated_time_remaining = (elapsed_time / completed_iterations) * (total_iterations - completed_iterations)
                
                # Log the progress and estimated time left
                logging.info(f"Processed row {j}/{p}, column {k}/{q}, "
                            f"elapsed time: {elapsed_time / 60:.2f} minutes, "
                            f"estimated time left: {estimated_time_remaining / 60:.2f} minutes.")
                
                # Save periodically
                should_save = (len(lis) >= checkpoint_interval or
                               (time.time() - last_checkpoint_time) >= checkpoint_time_interval)

                if should_save:
                    if lis:
                        temp_df = pd.concat(lis, ignore_index=True)
                        temp_checkpoint_file = f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl"
                        temp_df.to_pickle(temp_checkpoint_file)
                        logging.info(f"Saved temp checkpoint: {temp_checkpoint_file}")
                        temp_checkpoint_count += 1
                        lis = []

                    checkpoint_data = {
                        'lis': lis,
                        'j': j,
                        'k': k,
                        'temp_checkpoint_count': temp_checkpoint_count
                    }
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    last_checkpoint_time = time.time()

            start_k = 0  # Reset column counter after first row

        # Final Save
        final_dfs = []
        for i in range(temp_checkpoint_count):
            temp_file = f"{temp_checkpoint_prefix}{i}.pkl"
            if os.path.exists(temp_file):
                final_dfs.append(pd.read_pickle(temp_file))
                os.remove(temp_file)
                logging.info(f"Cleaned up temp checkpoint {temp_file}")
        if lis:
            final_dfs.append(pd.concat(lis, ignore_index=True))

        if final_dfs:
            df = pd.concat(final_dfs, ignore_index=True)
            df.to_pickle(os.path.join(data_folder, f"{data_tag}.pkl"))
            logging.info(f"Final results saved to {data_folder}/{data_tag}.pkl")

        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logging.info("Removed main checkpoint after successful completion.")

    except Exception as e:
        logging.error(f"Error: {e}")
        if lis:
            temp_df = pd.concat(lis, ignore_index=True)
            temp_df.to_pickle(f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'lis': lis,
                'j': j,
                'k': k,
                'temp_checkpoint_count': temp_checkpoint_count
            }, f)
        raise  # 