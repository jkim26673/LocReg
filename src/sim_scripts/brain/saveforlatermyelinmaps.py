from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

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
brain_data = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/cleaned_brain_data.mat")["final_data_2"]
SNR_map = scipy.io.loadmat("/home/kimjosy/LocReg_Regularization/SNR_map.mat")["SNR_map"]
logging.info(f"brain_data shape {brain_data.shape}")
logging.info(f"SNR_map shape {SNR_map.shape}")
logging.info(f"brain_data and SNR_map from Chuan have been successfully loaded")


#Iterator for Parallel Processing

p,q,s = brain_data.shape

SNR_map = SNR_map[SNR_map!= 0]

print("max SNR_map",np.max(SNR_map))
print("min SNR_map",np.min(SNR_map))


target_iterator = [(a,b) for a in range(p) for b in range(q)]
# target_iterator = [(80,160)]
print(target_iterator)
logging.debug(f'Target Iterator Length len({target_iterator})')

#Naming for Saving Data Collected from Script Folder
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')
data_path = "data/Brain/results"
add_tag = f"xcoordlen_{p}_ycoordlen_{q}DPfix"
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

SNR = 180

#CVXPY global parameters
eps = 1e-2

#Error Metric
err_type = "Wass. Score"

#Lambda Space
Lambda = np.logspace(-6,1,50).reshape(-1,1)
#Key Functions


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
    f_rec_normalized = f_rec / np.trapz(f_rec, T2)
    total_MWF = np.cumsum(f_rec_normalized)
    MWF = total_MWF[Myelin_idx[-1][-1]]
    return f_rec_normalized, MWF

def generate_noisy_brain_estimates(i_param_combo, seed= None):
    # print(f"Processing {i_param_combo}") 
    feature_df = pd.DataFrame(columns=["X_val", 'Y_val', "DP_estimate", "LC_estimate", 
                                       "LR_estimate", "GCV_estimate", "OR_estimate", "LS_estimate", "MWF_DP", "MWF_LC", 
                                       "MWF_LR", "MWF_GCV", "MWF_OR", "MWF_LS", "Lam_DP", "Lam_LC", "Lam_GCV", "Lam_OR", "Lam_LR", "Flagged"])
    
    if parallel == True:
        x_coord, y_coord = target_iterator[i_param_combo]
        pass
    else:
        x_coord, y_coord = i_param_combo
        # print("x_coord", x_coord) 
        # print("y_coord", y_coord) 

    #eliminate voxels with pure noise
    if brain_data[x_coord, y_coord][0] < 50:
    # if brain_data[0] < 50:
        print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
        return feature_df
    # if SNR_map[x_coord, y_coord] == 0:
    #     print(f"SNR_map is 0 and pure noise for {x_coord} and {y_coord}")
    #     return feature_df
    else:
        passing_x_coord = x_coord
        passing_y_coord = y_coord
        # curr_data = brain_data[passing_x_coord,passing_y_coord,:]
         df[(df["X_val"] == passing_x_coord) & (df["Y_val"] == y_coord)]
        # curr_data = brain_data
        #normalize data; check if normalization 1 after identifying weird pixels; seaborn plots;

        #real one 1/3/25
        noisy_f_rec_DP, noisy_lambda_DP = discrep_L2(noisy_curr_data, A, curr_SNR, Lambda, noise = True)
        noisy_f_rec_DP, noisy_MWF_DP = compute_MWF(noisy_f_rec_DP, T2, Myelin_idx)
        # print("MWF_DP", MWF_DP)

        # gt = f_rec_GCV
        
        def curve_plot(x_coord, y_coord, frec):
            plt.plot(T2, frec)
            plt.savefig(f"LR_recon_xcoord{x_coord}_ycoord{y_coord}.png")
            print(f"savefig xcoord{x_coord}_ycoord{y_coord}")
        
        feature_df["X_val"] = [passing_x_coord]
        feature_df["Y_val"] = [passing_y_coord]
        feature_df["curr_data"] = [curr_data]
        feature_df["noisy_curr_data"] = [noisy_curr_data]
        feature_df["noise"] = [noise]

        # real one 1/3/25
        feature_df["noisy_LS_estimate"] = [noisy_f_rec_LS]
        feature_df["noisy_DP_estimate"] = [noisy_f_rec_DP]
        feature_df["noisy_LC_estimate"] = [noisy_f_rec_LC]
        feature_df["noisy_LR_estimate"] = [noisy_f_rec_LR]
        feature_df["noisy_GCV_estimate"] = [noisy_f_rec_GCV]
        feature_df["noisy_MWF_DP"] = [noisy_MWF_DP]
        feature_df["noisy_MWF_LC"] = [noisy_MWF_LC]
        feature_df["noisy_MWF_LR"] = [noisy_MWF_LR]
        feature_df["noisy_MWF_GCV"] = [noisy_MWF_GCV]
        feature_df["noisy_MWF_LS"] = [noisy_MWF_LS]
        feature_df["noisy_Lam_DP"] = [noisy_lambda_DP]
        feature_df["noisy_Lam_LC"] = [noisy_lambda_LC]
        feature_df["noisy_Lam_LR"] = [noisy_lambda_LR]
        feature_df["noisy_Lam_GCV"] = [noisy_lambda_GCV]

        print(f"completed dataframe for x {passing_x_coord} and y {passing_y_coord}")
        return feature_df

noisy_f_rec_DP, noisy_lambda_DP = discrep_L2(noisy_curr_data, A, curr_SNR, Lambda, noise = True)
noisy_f_rec_DP, noisy_MWF_DP = compute_MWF(noisy_f_rec_DP, T2, Myelin_idx)