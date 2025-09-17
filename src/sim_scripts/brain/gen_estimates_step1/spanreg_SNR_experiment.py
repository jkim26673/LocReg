from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
try:
    with mosek.Env() as env:
        print("MOSEK environment initialized successfully.")
except mosek.MosekError as e:
    print("MOSEK Error:", e)

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

addingNoise = True
#Load Brain data and SNR map from Chuan
brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization-1/data/brain/braindata/mew_cleaned_brain_data_unfiltered.mat")["brain_data"]
# brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat")["final_data_2"]
_,_,s = brain_data.shape
# SNR_map = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/SNR_map.mat")["SNR_map"]
mask = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat")["new_BW"]
SNR_map = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization-1/data/brain/SNRmap/new_SNR_Map.mat")["SNR_MAP"]

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
print(target_iterator)
logging.debug(f'Target Iterator Length len({target_iterator})')

#Naming for Saving Data Collected from Script Folder
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')
data_path = f"data/Brain/results_{day}{month}{year}"
add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_filtered_SpanReg_curr_SNR_addition"
data_head = "est_table"
data_tag = (f"{data_head}_{add_tag}{day}{month}{year}")
data_folder = (os.getcwd() + f'/{data_path}')
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

tolerance = 1e-6
def spanreg_curr_SNR(i_param_combo, seed= None):
    # print(f"Processing {i_param_combo}") 
    feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "ref_estimate", "ref_MWF", "ref_Lam", "curr_data", "noise", "curr_SNR", 
                                    "LS_estimate", "MWF_LS", "DP_estimate", "LC_estimate", "LR_estimate", "GCV_estimate", 
                                    "MWF_DP", "MWF_LC", "MWF_LR", "MWF_GCV", "Lam_DP", "Lam_LC", "Lam_LR", "Lam_GCV", "Flagged"])
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
        passing_x_coord = x_coord
        passing_y_coord = y_coord
        print(f"Starting {x_coord} and {y_coord}")
        curr_data = brain_data[passing_x_coord,passing_y_coord,:]
        curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;
        #LC
        noisy_f_rec_LC, noisy_lambda_LC = Lcurve(curr_data, A, Lambda)
        # noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        # noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        # ref_lambda_GCV = np.squeeze(noisy_lambda_GCV)
        # reference_estimate, reference_MWF = compute_MWF(noisy_f_rec_GCV, T2, Myelin_idx)
        #LR
        LRIto_ini_lam = noisy_lambda_LC
        f_rec_ini = noisy_f_rec_LC
        # print("curr_data", curr_data)
        reference_estimate, ref_lam, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter)
        X = reference_estimate
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(reference_estimate[:-1] == 0) or all_close_to_zero:
            print("near zero reference value, passing")
            pass
            # print(f"Passed {x_coord} and {y_coord}")
            # return feature_df
        reference_estimate, ref_MWF = compute_MWF(reference_estimate, T2, Myelin_idx)
        data_noiseless = A @ reference_estimate
        factor = np.sum(reference_estimate) * dT
        data_noiseless = data_noiseless/factor
        curr_data, noise = add_noise(data_noiseless, SNR = curr_SNR)
        # curr_SNR = SNR_map[passing_x_coord,passing_y_coord]
        #LS:
        try:
            noisy_f_rec_LS = nnls(A, curr_data, maxiter=1e6)[0]
        except:
            try:
                noisy_f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
            except:
                noisy_f_rec_LS = np.zeros(len(T2))
                Flag_Val = 2
        # f_rec_LS = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
        X = noisy_f_rec_LS
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_LS[:-1] == 0) or all_close_to_zero:
            noisy_MWF_LS = 0
            method = "LS"
            Flag_Val = 1
        else:
            noisy_f_rec_LS, noisy_MWF_LS = compute_MWF(noisy_f_rec_LS, T2, Myelin_idx)
            Flag_Val = 0
        #LC
        noisy_f_rec_LC, noisy_lambda_LC = Lcurve(curr_data, A, Lambda)
        X = noisy_f_rec_LC

        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_LC[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LC is 0 or all_close_to_zero")
            noisy_MWF_LC = 0
            method = "LC"
            Flag_Val = 1
        else:
            noisy_f_rec_LC, noisy_MWF_LC = compute_MWF(noisy_f_rec_LC, T2, Myelin_idx)
            Flag_Val = 0
        #GCV
        noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
        noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
        noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)

        X = noisy_f_rec_GCV
        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_GCV[:-1] == 0) or all_close_to_zero:
            noisy_MWF_GCV = 0
            method = "GCV"
            Flag_Val = 1
        else:
            noisy_f_rec_GCV, noisy_MWF_GCV = compute_MWF(noisy_f_rec_GCV, T2, Myelin_idx)
            Flag_Val = 0
        noisy_f_rec_DP, noisy_lambda_DP = discrep_L2_brain(curr_data, A, curr_SNR, Lambda, noise = True)
        X = noisy_f_rec_DP

        all_close_to_zero = np.all(np.abs(X) < tolerance)
        if np.all(noisy_f_rec_DP[:-1] == 0) or all_close_to_zero:
            # print("f_rec_DP is 0 or all_close_to_zero")
            noisy_MWF_DP = 0
            method = "DP"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_DP, curr_data, lambda_DP, curr_SNR, MWF_DP, filepath)
        else:
            noisy_f_rec_DP, noisy_MWF_DP = compute_MWF(noisy_f_rec_DP, T2, Myelin_idx)
            Flag_Val = 0
        # print("MWF_DP", MWF_DP)
        #LR
        LRIto_ini_lam = noisy_lambda_LC
        f_rec_ini = noisy_f_rec_LC
        # print("curr_data", curr_data)
        noisy_f_rec_LR, noisy_lambda_LR, test_frec1, test_lam1, numiterate = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter)
        X = noisy_f_rec_LR

        all_close_to_zero = np.all(np.abs(X) < tolerance)

        if np.all(noisy_f_rec_LR[:-1] == 0) or all_close_to_zero:
            # print("f_rec_LR is 0")
            noisy_MWF_LR = 0
            method = "LocReg"
            Flag_Val = 1
            # curve_plot(method, x_coord, y_coord, f_rec_LR, curr_data, lambda_LR, curr_SNR, MWF_LR, filepath)
        else:
            noisy_f_rec_LR, noisy_MWF_LR = compute_MWF(noisy_f_rec_LR, T2, Myelin_idx)
            Flag_Val = 0
        # print("MWF_LR", MWF_LR)
        gt = noisy_f_rec_GCV

        feature_df["X_val"] = [passing_x_coord]
        feature_df["Y_val"] = [passing_y_coord]
        feature_df["ref_estimate"] = [reference_estimate]
        feature_df["ref_MWF"] = [ref_MWF]
        feature_df["ref_Lam"] = [ref_lam]
        feature_df["curr_data"] = [curr_data]
        feature_df["noise"] = [noise]
        feature_df["curr_SNR"] = [curr_SNR]
        # real one 1/3/25
        feature_df["LS_estimate"] = [noisy_f_rec_LS]
        feature_df["MWF_LS"] = [noisy_MWF_LS]
        feature_df["DP_estimate"] = [noisy_f_rec_DP]
        feature_df["LC_estimate"] = [noisy_f_rec_LC]
        feature_df["LR_estimate"] = [noisy_f_rec_LR]
        feature_df["GCV_estimate"] = [noisy_f_rec_GCV]
        feature_df["MWF_DP"] = [noisy_MWF_DP]
        feature_df["MWF_LC"] = [noisy_MWF_LC]
        feature_df["MWF_LR"] = [noisy_MWF_LR]
        feature_df["MWF_GCV"] = [noisy_MWF_GCV]
        feature_df["Lam_DP"] = [noisy_lambda_DP]
        feature_df["Lam_LC"] = [noisy_lambda_LC]
        feature_df["Lam_LR"] = [noisy_lambda_LR]
        feature_df["Lam_GCV"] = [noisy_lambda_GCV]
        feature_df["Flagged"] = [Flag_Val]
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
    logging.info(f"T2 range is from 10ms to 200ms with {m} discretizations")
    logging.info(f"dT is {dT}")
    logging.info(f"TE range is {TE}")
    for i in range(n):
        for j in range(m):
            A[i,j] = np.exp(-TE[i]/T2[j]) * dT
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

    if parallel == True:
        logging.info("Generating Brain Estimates Using Parallel Processing.")
        estimates_dataframe = parallel_processed(spanreg_curr_SNR, shift = True)
        print("Saved estimated_data")
        print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
        df = pd.concat(lis, ignore_index= True)
        df.to_pickle(data_folder + f'/' + data_tag +'.pkl')
    else:
        logging.info("Generating Brain Estimates Without Parallel Processing.")
        for j in tqdm(range(p), desc="Processing rows", unit="row"):
            for k in tqdm(range(q), desc="Processing columns", unit="col", leave=False):  # leave=False to not overwrite the row progress
                iteration = (j, k)
                estimates_dataframe = spanreg_curr_SNR(iteration) 
                lis.append(estimates_dataframe)
        print(f"Completed {len(lis)} of {len(target_iterator)} voxels")
        df = pd.concat(lis, ignore_index= True)
        df.to_pickle(data_folder + f'/' + data_tag +'.pkl')
    #Save file
