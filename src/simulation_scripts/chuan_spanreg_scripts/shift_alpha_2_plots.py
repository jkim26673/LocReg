#This function takes the results from the multi_reg_MRR_0116.py simulations and attempts to show that
#selecting alpha2 despite its small regularization values is not trivial.

#The function varies alpha2 values by index and plots the reconstruction compared to the ground truth and the optimal grid 
#search reconstruction composed of optimal alpha1 and alpha2 values. This script attempts to show that there are major reconstruction differences
#when you vary alpha2.


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def vary_a2_plots(x_lc_vec,opt_alpha_ind,ground_truth,save_pathname, date, which_MRR_run, index_offset, save):
    opt_alpha_1 = opt_alpha_ind[0][0]
    opt_alpha_2 = opt_alpha_ind[0][1]
    f_rec_grid_opt = x_lc_vec[:, opt_alpha_1, opt_alpha_2]
    f_rec_pos_offset = x_lc_vec[:, opt_alpha_1, opt_alpha_2 + index_offset]
    f_rec_neg_offset = x_lc_vec[:, opt_alpha_1, opt_alpha_2 - index_offset]
    fig = plt.figure()
    plt.plot(ground_truth)
    plt.plot(f_rec_grid_opt)
    plt.plot(f_rec_pos_offset)
    plt.plot(f_rec_neg_offset)
    plt.title("MRR Reconstructed Solutions")
    plt.legend(["Ground Truth", "Optimal Grid Search", f"Grid Search Alpha2 offset +{index_offset} index", f"Grid Search Alpha2 offset -{index_offset} index"])
    plt.xlabel("T2")
    plt.ylabel("Amplitude")
    if save == True:
        plt.savefig("".join([save_pathname, f"{date}_vary_alpha2_plot_with_index_offset_{index_offset}_using_MRR_grid_search_{which_MRR_run}"]))
        print("Figure Saved")
    return fig

if __name__ == "main":
    print("Running script")

    #insert path name to your reconstruction values, the indices to your alpha1, alpha2 values, and your ground truth reconstruction
    x_lc_vec = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/1_22_24_Runs/x_lc_vec_NR1_01_22_24_35_alpha1_35_alpha2_discretizations_alpha1log_-5_0_alpha2log_-5_0.npy")
    opt_alpha_ind = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/1_22_24_Runs/opt_alphas_ind_01_22_24_35_alpha1_35_alpha2_discretizations_alpha1log_-5_0_alpha2log_-5_0.npy")
    ground_truth = np.load("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/1_22_24_Runs/ground_truth_01_22_24_35_alpha1_35_alpha2_discretizations_alpha1log_-5_0_alpha2log_-5_0.npy")
    save_pathname = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/Experiments/MRR_grid_search/"

    #Set the date and where you got your MRR_grid data from
    today_date = datetime.today()
    # Format the date as "month_day_lasttwodigits"
    date = today_date.strftime("%m_%d_%y")
    which_MRR_run = "01_22_24_Run"

    #Indicate whether you want to save the plot
    save = True

    print("Running Plot")
    fig = vary_a2_plots(x_lc_vec, opt_alpha_ind, ground_truth, save_pathname, date, which_MRR_run, index_offset = 3, save=True)

    print("Finished Script")
