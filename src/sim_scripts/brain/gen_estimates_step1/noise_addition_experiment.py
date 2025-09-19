from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
from src.utils.load_imports.load_classical import *

# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.addpath(r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\ZamaUPEN\1D_test', nargout=0)


#Reconfigure the path script to all files for respective OS system.
# mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
mosek_license_path = r"C:\Users\kimjosy\Downloads\mosek\mosek.lic"
# unfilt_brain_data_path = "/home/kimjosy/LocReg_Regularization-1/data/brain/braindata/mew_cleaned_brain_data_unfiltered.mat"
# brain_data_path = "/home/kimjosy/LocReg_Regularization-1/data/brain/braindata/mew_cleaned_brain_data_unfiltered.mat"
# brain_data_path = "/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat"
# mask_path = "/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat" 
# SNR_map_path = "/home/kimjosy/LocReg_Regularization-1/data/brain/SNRmap/new_SNR_Map.mat"
# unfiltered_SNR_map_path = "/home/kimjosy/LocReg_Regularization-1/data/brain/SNRmap/new_SNR_Map_unfiltered.mat"
unfilt_brain_data_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\braindata\mew_cleaned_brain_data_unfiltered.mat"
brain_data_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\braindata\mew_cleaned_brain_data_unfiltered.mat"
# brain_data_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\braindata\cleaned_brain_data (1).mat"
mask_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\masks\new_mask.mat" 
SNR_map_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\SNRmap\new_SNR_Map.mat"
unfiltered_SNR_map_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\SNRmap\new_SNR_Map_unfiltered.mat"

os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Naming Convention for Save Folder for Images:
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

pat_tag = "MRR"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
series_tag = "BrainData_Images"
simulation_save_folder = os.path.join("SimulationSets", pat_tag, series_tag)
# cwd_full = cwd_cut + output_folder + lam_ini
cwd_full = cwd_cut + simulation_save_folder 


addingNoise = True
unif_noise = True
#Load Brain data and SNR map from Chuan
unfilt_brain_data = scipy.io.loadmat(unfilt_brain_data_path)["brain_data"]
# brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat")["final_data_2"]
brain_data = scipy.io.loadmat(brain_data_path)["brain_data"]
_,_,s = brain_data.shape
# SNR_map = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/SNR_map.mat")["SNR_map"]
mask = scipy.io.loadmat(mask_path)["new_BW"]
#filtered
SNR_map = scipy.io.loadmat(SNR_map_path)["SNR_MAP"]
#unfiltered
# SNR_map =scipy.io.loadmat("/home/kimjosy/LocReg_Regularization-1/data/brain/SNRmap/new_SNR_Map_unfiltered.mat")["SNR_MAP"]
ones_array = np.ones(s)
expanded_mask = mask[:, :, np.newaxis] * ones_array
brain_data = expanded_mask * brain_data

logging.info(f"brain_data shape {brain_data.shape}")
logging.info(f"SNR_map shape {SNR_map.shape}")
logging.info(f"brain_data and SNR_map from Chuan have been successfully loaded")
print("SNR_map.shape",SNR_map.shape)

#Iterator for Parallel Processing
p,q,s = brain_data.shape
#replace minimum SNR of 0 to be 1
SNR_map = np.where(SNR_map == 0, 1, SNR_map)
target_iterator = [(a,b) for a in range(p) for b in range(q)]
# print(target_iterator)
logging.debug(f'Target Iterator Length len({target_iterator})')

#Naming for Saving Data Collected from Script Folder
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')
# data_path = f"data/Brain/results_{day}{month}{year}"
data_path = os.path.join("data","Brain",f"results_{day}{month}{year}")
SNR_unif_level = 200
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_unfiltered_noise_addition_recover_SNR{SNR_unif_level}"
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_filtered_noise_addition_uniform_noise"
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_unfiltered_noise_addition_uniform_noise_UPEN"
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_unfiltered_noise_addition_recover_SNR{SNR_unif_level}"
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_unfiltered_noise_addition_recover_curr_SNR_map"
add_tag = f"xcoordlen_{p}_ycoordlen_{q}_filtered_noise_addition_uniform_noise_UPEN_LR1D2D"

data_head = "est_table"
data_tag = (f"{data_head}_{add_tag}{day}{month}{year}")
# data_folder = (os.getcwd() + f'/{data_path}')
data_folder = os.path.join(os.getcwd(), f"{data_path}")
os.makedirs(data_folder, exist_ok = True)
logging.info(f"Save Folder for Final Estimates Table {data_folder})")

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
lam_ini_val = "GCV"
gamma_init = 0.5
maxiter = 500

#CVXPY global parameters
eps = 1e-2

#Error Metric
err_type = "Wassterstein"

#Lambda Space
Lambda = np.logspace(-6,1,50).reshape(-1,1)
# curr_SNR = 1

Kmax = 500
beta_0 = 1e-7
tol_lam=1e-5

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

def generate_noisy_estimates(i_param_combo, seed= None):
    # print(f"Processing {i_param_combo}") 
    feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "curr_data", "noise", "curr_SNR", 
                                    "LS_estimate", "MWF_LS", "DP_estimate", "LC_estimate", "LR_estimate", "GCV_estimate", 
                                    "MWF_DP", "MWF_LC", "MWF_LR", "MWF_GCV", "Lam_DP", "Lam_LC", "Lam_LR", "Lam_GCV"])
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
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
        if unif_noise == True:
            unnormalized_data = curr_data
            noise = unif_noise_val
            curr_data = curr_data + unif_noise_val
            try:
                sol1 = nnls(A, curr_data, maxiter=1e6)[0]
            except:
                try:
                    sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
                except:
                    print("need to skip, cannot find solution to LS solutions for normalizaiton")
                    return feature_df
            X = sol1
            tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
            all_close_to_zero = np.all(np.abs(X) < tolerance)
            factor = np.sum(sol1) * dT
            curr_data = curr_data/factor
            plt.figure()
            plt.plot( curr_data)
            plt.savefig("testdata.png")
            plt.close()
        else:
            try:
                sol1 = nnls(A, curr_data, maxiter=1e6)[0]
            except:
                try:
                    sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
                except:
                    print("need to skip, cannot find solution to LS solutions for normalizaiton")
                    return feature_df
            X = sol1
            tolerance = 1e-6  # Adjust tolerance based on what you consider "close to zero"
            all_close_to_zero = np.all(np.abs(X) < tolerance)
            factor = np.sum(sol1) * dT
            curr_data = curr_data/factor
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
        noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
        noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV, tol = 1e-6)

        # #DP reconstruction and MWF calculation
        # noisy_f_rec_DP, noisy_lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
        # noisy_f_rec_DP, noisy_MWF_DP, DP_Flag_Val = filter_and_compute_MWF(noisy_f_rec_DP, tol = 1e-6)

        #LocReg reconstruction and MWF calculation
        # LRIto_ini_lam = noisy_lambda_GCV
        # f_rec_ini = noisy_f_rec_GCV
        # noisy_f_rec_LR, noisy_lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        # noisy_f_rec_LR, noisy_MWF_LR, LR_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LR, tol = 1e-6)
        # gt = noisy_f_rec_GCV

        # #LocReg1stDeriv reconstruction and MWF calculation
        # LRIto_ini_lam = noisy_lambda_GCV
        # f_rec_ini = noisy_f_rec_GCV
        # f_rec_LR1D, noisy_lambda_LR1D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        # noisy_f_rec_LR1D, noisy_MWF_LR1D, LR_Flag_Val1D = filter_and_compute_MWF(f_rec_LR1D, tol = 1e-6)

        #LocReg2ndDeriv reconstruction and MWF calculation
        LRIto_ini_lam = noisy_lambda_GCV
        f_rec_ini = noisy_f_rec_GCV
        f_rec_LR2D, noisy_lambda_LR2D, test_frec1, test_lam1, numiterate = LocReg_Ito_mod_deriv2(curr_data, A, LRIto_ini_lam, gamma_init, maxiter = 50)
        noisy_f_rec_LR2D, noisy_MWF_LR2D, LR_Flag_Val2D = filter_and_compute_MWF(f_rec_LR2D, tol = 1e-6)

        #UPEN reconstruction and MWF calculation
        # result = upen_param_setup(TE, T2, A, curr_data)
        # noisy_f_rec_UPEN, _ ,_ , noisy_lambda_UPEN= upen_setup(result, curr_data, LRIto_ini_lam, True)
        noise_norm = np.linalg.norm(unif_noise_val)
        xex = noisy_f_rec_GCV
        Kmax = 50
        noisy_f_rec_UPEN, noisy_lambda_UPEN = UPEN_Zama(A, curr_data, xex, noise_norm, beta_0, Kmax, tol_lam)
        noisy_f_rec_UPEN, noisy_MWF_UPEN, UPEN_Flag_Val = filter_and_compute_MWF(noisy_f_rec_UPEN, tol = 1e-6)

        feature_df["X_val"] = [passing_x_coord]
        feature_df["Y_val"] = [passing_y_coord]
        feature_df["curr_data"] = [curr_data]
        feature_df["noise"] = [noise]
        feature_df["curr_SNR"] = [curr_SNR]
        # feature_df["LS_estimate"] = [noisy_f_rec_LS]
        # feature_df["MWF_LS"] = [noisy_MWF_LS]
        # feature_df["DP_estimate"] = [noisy_f_rec_DP]
        # feature_df["LC_estimate"] = [noisy_f_rec_LC]
        # feature_df["LR_estimate"] = [noisy_f_rec_LR]
        # feature_df["LR1D_estimate"] = [noisy_f_rec_LR1D]
        feature_df["LR2D_estimate"] = [noisy_f_rec_LR2D]
        feature_df["GCV_estimate"] = [noisy_f_rec_GCV]
        feature_df["UPEN_estimate"] = [noisy_f_rec_UPEN]
        # feature_df["MWF_DP"] = [noisy_MWF_DP]
        # feature_df["MWF_LC"] = [noisy_MWF_LC]
        # feature_df["MWF_LR"] = [noisy_MWF_LR]
        # feature_df["MWF_LR1D"] = [noisy_MWF_LR1D]
        feature_df["MWF_LR2D"] = [noisy_MWF_LR2D]
        feature_df["MWF_GCV"] = [noisy_MWF_GCV]
        feature_df["MWF_UPEN"] = [noisy_MWF_UPEN]
        # feature_df["Lam_DP"] = [noisy_lambda_DP]
        # feature_df["Lam_LC"] = [noisy_lambda_LC]
        # feature_df["Lam_LR"] = [noisy_lambda_LR]
        # feature_df["Lam_LR1D"] = [noisy_lambda_LR1D]
        feature_df["Lam_LR2D"] = [noisy_lambda_LR2D]
        feature_df["Lam_GCV"] = [noisy_lambda_GCV]
        feature_df["Lam_UPEN"] = [noisy_lambda_UPEN]
        # feature_df["LS_Flag_Val"] = [LS_Flag_Val]
        # feature_df["LC_Flag_Val"] = [LC_Flag_Val]
        feature_df["GCV_Flag_Val"] = [GCV_Flag_Val]
        # feature_df["DP_Flag_Val"] = [DP_Flag_Val]
        # feature_df["LR_Flag_Val"] = [LR_Flag_Val]
        # feature_df["LR1D_Flag_Val"] = [LR_Flag_Val1D]
        feature_df["LR2D_Flag_Val"] = [LR_Flag_Val2D]
        feature_df["UPEN_Flag_Val"] = [UPEN_Flag_Val]
        print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
        return feature_df

#We set GCV as ground truth
filepath = data_folder

def worker_init():
    # Use current_process()._identity to get a unique worker ID for each worker
    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    np.random.seed(worker_id)  # Set a random seed for each worker


def parallel_processed(func, shift=True):
    with mp.Pool(processes=num_cpus_avail, initializer=worker_init) as pool:
        with tqdm(total=len(target_iterator)) as pbar:
            for estimates_dataframe in pool.imap_unordered(func, range(len(target_iterator))):  # Pass target_iterator directly
                lis.append(estimates_dataframe)
                pbar.update()
        pool.close()
        pool.join()
    return estimates_dataframe

#Unit Tests
if __name__ == "__main__":
    logging.info("Script started.")
    freeze_support()
    dTE = 11.3
    n = 32
    TE = dTE * np.linspace(1,n,n)
    m = 150
    T2 = np.linspace(10,200,m)
    A= np.zeros((n,m))
    dT = T2[1] - T2[0]
    for i in range(n):
        for j in range(m):
            A[i,j] = np.exp(-TE[i]/T2[j]) * dT
    if unif_noise == True:
        num_signals = 1000
        coord_pairs = set()
        
        # Generate random coordinates
        # for i in range(num_signals):
        #     x = random.randint(155, 160)
        #     y = random.randint(150, 160)
        #     mask_value = mask[x, y]
        #     coord_pairs.add((x, y))
        while len(coord_pairs) < num_signals:
            # Generate random coordinates
            x = random.randint(130, 200)
            y = random.randint(100, 200)
            # Get the mask_value at the coordinate
            mask_value = mask[x, y]
            
            # Check if the mask_value is 0 and add the coordinate to the set
            if mask_value == 0:
                coord_pairs.add((x, y))
        coord_pairs = list(coord_pairs)
        print("Length", len(coord_pairs))
        
        signals, SNRs = get_signals(coord_pairs, mask , unfilt_brain_data, A, dT)
        # mean_sig, unif_noise_val = calculate_noise(signals)
        # SNR_mean = np.mean(SNRs)
        # normalized_signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)
        # signals = normalized_signals
        mean_sig = np.mean(signals, axis=0)
        tail_length = 2  # Example length, adjust as needed
        # # Get the tail end of the signals
        tail = mean_sig[-tail_length:]
        # # Calculate the standard deviation of the tail ends
        tail_std = np.std(tail)
        # _, unif_noise_val = add_noise(mean_sig, SNR = SNR_mean)
        print("tail_std", tail_std)
        # _, unif_noise_val = add_noise(mean_sig, SNR = 1)
        unif_noise_val = np.random.normal(0, tail_std, size=32)  # Add noise
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
    logging.info(f"T2 range is from 10ms to 200ms with {m} discretizations")
    logging.info(f"dT is {dT}")
    logging.info(f"TE range is {TE}")
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
    LR1D_estimates = np.zeros((p,q,m))
    MWF_LR1D = np.zeros((p,q))
    LR2D_estimates = np.zeros((p,q,m))
    MWF_LR2D = np.zeros((p,q))
    UPEN_estimates = np.zeros((p,q,m))
    MWF_UPEN = np.zeros((p,q))
    Myelin_idx = np.where(T2<=40)
    logging.info("We define myelin index to be less than 40 ms.")
    logging.info("Since this is experimental patient brain data, we do not have the ground truth. Set the ground truth to be GCV.")
    lis = []

    if parallel == True:
        estimates_dataframe = parallel_processed(generate_noisy_estimates, shift = True)
        #Save file
        print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
        df = pd.concat(lis, ignore_index= True)
        finalpath = os.path.join(data_folder, f"{data_tag}.pkl")
        # df.to_pickle(data_folder + f'/' + data_tag +'.pkl')
        df.to_pickle(finalpath)
    else:
        print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
        lis = []
        logging.info("Generating Brain Estimates Without Parallel Processing.")
        for j in tqdm(range(p), desc="Processing rows", unit="row"):
            for k in tqdm(range(q), desc="Processing columns", unit="col", leave=False):  # leave=False to not overwrite the row progress
                iteration = (j, k)
                estimates_dataframe = generate_noisy_estimates(iteration) 
                lis.append(estimates_dataframe)
        #Save file
        print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
        df = pd.concat(lis, ignore_index= True)
        finalpath = os.path.join(data_folder, f"{data_tag}.pkl")
        df.to_pickle(finalpath)
        print(f"Fle saved at {finalpath}")


