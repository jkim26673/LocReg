import sys
sys.path.append(".")
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.sim_scripts.peak_resolution.resolutionpeakanalysisfinal import detect_peaks
import os
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_new/est_table_SNR1000_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_new_15Oct24.pkl" #Percentage of Resolved Peaks: 79.9%
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000/est_table_SNR1000_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000_15Oct24.pkl" #Percentage of Resolved Peaks: 85.60%
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300/est_table_SNR300_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300_15Oct24.pkl" # Percentage of Resolved Peaks: 68.00%
# modfile= "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_300_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300/est_table_SNR300_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300_15Oct24.pkl" # Percentage of Resolved Peaks: 91.60%
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50/est_table_SNR50_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50_15Oct24.pkl" #Percentage of Resolved Peaks: 54.40%
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50/est_table_SNR50_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50_15Oct24.pkl" #Percentage of Resolved Peaks: 73.60%

# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000/est_table_SNR1000_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_14Oct24.pkl" #Percentage of Resolved Peaks: 43.20%
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000/est_table_SNR1000_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000_14Oct24.pkl" #Percentage of Resolved Peaks: 40.40%
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300/est_table_SNR300_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300_14Oct24.pkl" #Percentage of Resolved Peaks: 43.60%
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_300_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300/est_table_SNR300_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300_14Oct24.pkl" #Percentage of Resolved Peaks: 67.20%
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50/est_table_SNR50_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50_14Oct24.pkl" # Percentage of Resolved Peaks: 40.00%
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50/est_table_SNR50_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50_14Oct24.pkl" #Percentage of Resolved Peaks: 28.00%


# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000/est_table_SNR1000_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000_18Oct24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-19_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim110_SNR_1000_errtype_Wass. Score/est_table_SNR1000_iter110_lamini_LCurve_dist_narrowL_broadR_parallel_nsim110_SNR_1000_errtype_Wass. Score_19Nov24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-19_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim110_SNR_300_errtype_Wass. Score/est_table_SNR300_iter110_lamini_LCurve_dist_narrowL_broadR_parallel_nsim110_SNR_300_errtype_Wass. Score_19Nov24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-19_SNR_300_lamini_LCurve_dist_broadL_narrowR_parallel_nsim110_SNR_300_errtype_Wass. Score/est_table_SNR300_iter110_lamini_LCurve_dist_broadL_narrowR_parallel_nsim110_SNR_300_errtype_Wass. Score_19Nov24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-19_SNR_50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50_errtype_Wass. Score/est_table_SNR50_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50_errtype_Wass. Score_19Nov24.pkl"
# modfile = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-19_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000_errtype_Wass. Score/est_table_SNR1000_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000_errtype_Wass. Score_19Nov24.pkl"


NLBR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_errtype_Wass. Score/est_table_SNR1000_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_errtype_Wass. Score_22Nov24.pkl"
NLBR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-21_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300_errtype_Wass. Score/est_table_SNR300_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300_errtype_Wass. Score_21Nov24.pkl"
NLBR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-21_SNR_50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50_errtype_Wass. Score/est_table_SNR50_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50_errtype_Wass. Score_21Nov24.pkl"
BLNR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000_errtype_Wass. Score/est_table_SNR1000_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000_errtype_Wass. Score_22Nov24.pkl"
BLNR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_300_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300_errtype_Wass. Score/est_table_SNR300_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300_errtype_Wass. Score_22Nov24.pkl"
BLNR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50_errtype_Wass. Score/est_table_SNR50_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50_errtype_Wass. Score_22Nov24.pkl"


def func(datafile, signalvar):
    with open(datafile, 'rb') as file:
        errors = pickle.load(file)
    df = pd.DataFrame(errors)
    df = df.dropna()
    df['Sigma'] = df['Sigma'].apply(tuple)
    df_sorted = df.sort_values(by=['NR','Sigma', 'RPS_val'], ascending=[True, True, True])
    grouped = df_sorted[df_sorted['RPS_val'] == 1.8250000000000002]
    # grouped = df_sorted[df_sorted['RPS_val'] == 1.75]

    # # Print peaks
    # peak_test = []
    # n = 150
    # m = 200
    print("groouped", grouped)
    LRs = grouped["LR_vect"].to_numpy()
    # signals = grouped["LR_vect"].to_numpy()
    LRsignals = grouped["LR_vect"].to_numpy()
    LCsignals = grouped["LC_vect"].to_numpy()
    GCVsignals = grouped["GCV_vect"].to_numpy()
    oraclesignals = grouped["oracle_vect"].to_numpy()
    DPsignals = grouped["DP_vect"].to_numpy()
    if signalvar == "LR":
        signals = LRsignals
    elif signalvar == "GCV":
        signals = GCVsignals
    elif signalvar == "LC":
        signals = LCsignals
    elif signalvar == "OR":
        signals = oraclesignals
    elif signalvar == "DP":
        signals = DPsignals
    """LRsignals = grouped["LR_vect"].to_numpy()
    LCsignals = grouped["LC_vect"].to_numpy()
    GCVsignals = grouped["LC_vect"].to_numpy()
    oraclesignals = grouped["oracle_vect"].to_numpy()
    DPsignals = grouped["DP_vect"].to_numpy()"""
    T2_values = np.linspace(10, 200, 150)
    # print("peak_test", peaks)
    # non_zero_count  = np.count_nonzero(peaks)
    # resolve_percent = non_zero_count/len(peaks)
    # Initialize resolution stats
    resolution_stats = []

    # Loop through each signal to detect peaks and record resolution
    for idx, signal in enumerate(signals):
        signal1 = np.ravel(signal)
        # print("signal1", signal1.shape)
        resolved_peaks, spurious_peaks, is_resolved = detect_peaks(T2_values, signal1)
        basefilepath = "/home/kimjosy/LocReg_Regularization-1/data/debugfigures/"
        datafilename = f"peakresolution/{datafilenamevar}/{signalvar}"
        # Create the full path by combining base filepath and datafile
        full_filepath = os.path.join(basefilepath, datafilename)
        # Check if the directory exists, if not, create it
        if not os.path.exists(full_filepath):
            os.makedirs(full_filepath)
            print(f"Directory created: {full_filepath}")
        plt.figure()
        plt.plot(T2_values, signal1, label = f"{signalvar} signal[{idx+1}]")
        for peak in resolved_peaks:
            T2_peak, amplitude, min_between = peak  # Unpack the tuple into T2 position and amplitude
            plt.plot(T2_peak, amplitude, 'rx', markersize=10, label = f"resolved peaks {amplitude:3e}")  # Plot "X" marker at each peak
            plt.plot(T2_peak, min_between, 'bx', markersize = 10, label = f"minimum between 2 peaks {min_between:3e}")
        plt.legend()
        plt.savefig(f"{full_filepath}/signal{idx+1}")    
        print(f"Signal {idx+1}:")
        print("Detected Peaks Indices:", resolved_peaks)
        print("Spurious Peaks:", spurious_peaks)
        print(f"Peaks are {'resolved' if is_resolved else 'not resolved'}.\n")
        resolution_stats.append(is_resolved)

    # Calculate and print the percentage of resolved peaks
    percentage_resolved = (sum(resolution_stats) / len(resolution_stats)) * 100
    print(f"Percentage of Resolved Peaks for: {percentage_resolved:.2f}%")
    return percentage_resolved

NLBR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_errtype_Wass. Score/est_table_SNR1000_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_errtype_Wass. Score_22Nov24.pkl"
NLBR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-21_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300_errtype_Wass. Score/est_table_SNR300_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300_errtype_Wass. Score_21Nov24.pkl"
NLBR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-21_SNR_50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50_errtype_Wass. Score/est_table_SNR50_iter50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50_errtype_Wass. Score_21Nov24.pkl"
BLNR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000_errtype_Wass. Score/est_table_SNR1000_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000_errtype_Wass. Score_22Nov24.pkl"
BLNR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_300_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300_errtype_Wass. Score/est_table_SNR300_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300_errtype_Wass. Score_22Nov24.pkl"
BLNR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50_errtype_Wass. Score/est_table_SNR50_iter50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50_errtype_Wass. Score_22Nov24.pkl"

names = np.array(["LR", "OR", "GCV", "DP", "LC"])
files = np.array([NLBR1000, NLBR300, NLBR50, BLNR1000, BLNR300, BLNR50])
namevar = np.array(["NLBR1000", "NLBR300", "NLBR50", "BLNR1000", "BLNR300", "BLNR50"])
filedict = dict(zip(files, namevar))

# datafile = BLNR50
# datafilenamevar = "BLNR50"
results = []
for file, namevar in filedict.items():
    for name in names:
        datafile = file
        datafilenamevar = namevar
        peakresolve = func(datafile, name)
        results.append({
            # 'datafile': datafile,
            'Distribution': datafilenamevar,
            'Reg_Method': name,
            'Peak_Res_Perc': peakresolve
        })
df = pd.DataFrame(results)
# Optionally, display the DataFrame
df_pivoted = df.pivot_table(index=['Distribution'], columns='Reg_Method', values='Peak_Res_Perc')
df_pivoted['SNR'] = [1000, 300, 50, 1000, 300, 50]
df_pivoted.to_pickle("/home/kimjosy/LocReg_Regularization-1/data/debugfigures/peakresolution/peakresoltable.pkl")
print("saved dataframe")
print(df_pivoted)
#func(datafile, "LR")
# func(datafile, "OR")
# func(datafile, "GCV")
# func(datafile, "DP")
# func(datafile, "LC")

# import pickle
# import pandas as pd
# from datetime import datetime
# with open ("/home/kimjosy/LocReg_Regularization-1/data/debugfigures/peakresoltable.pkl", "rb") as file:
#     df = pickle.load(file)
# df = df.rename(columns= {"FileName":"Distribution"})
# df_pivoted = df.pivot_table(index=['Distribution'], columns='Reg_Method', values='Peak_Res_Perc')
# df_pivoted['SNR'] = [1000, 300, 50, 1000, 300, 50]
# filepath = "/home/kimjosy/LocReg_Regularization-1/data/peakresolution"
# today_date = datetime.today().date().strftime("%Y-%m-%d")
# df_pivoted.to_csv(f'{filepath}/peakresolution_{today_date}.csv')

# s = pd.Series([1000, 300, 50, 1000, 300, 50], index=['BLNR1000', 'BLNR300', 'BLNR50', 'NLBR1000', 'NLBR300', 'NLBR50'])
# df_pivoted["SNR"] = s
# # Optional: Plotting
# plt.plot(f_rec_LocReg_LC)
# peaks, _ = find_peaks(f_rec_LocReg_LC)
# plt.plot(peaks, f_rec_LocReg_LC[peaks], "x")
# plt.title("Local Peaks")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.savefig("test.png")  # Save the plot as a PNG file
# plt.show()
# print("f_rec_LocReg_LC[peaks]", f_rec_LocReg_LC[peaks])
# first_peak_amp, second_peak_amp, min_dist = find_min_between_peaks(f_rec_LocReg_LC, ref_ind = -2, sel_ind = -1)
# print(f"{first_peak_amp:.4f}")  # Shows up to 15 decimal places
# print(f"{second_peak_amp:.4f}")
# print(f"min_dist {min_dist:.4f}")
# boolval = check_resolution(first_peak_amp, second_peak_amp, min_dist)
# print(boolval)

# df = pd.DataFrame(errors)
# df['Sigma'] = df['Sigma'].apply(tuple)
# df_sorted = df.sort_values(by=['NR','Sigma', 'RPS_val'], ascending=[True, True, True])

# grouped = df_sorted.groupby(['Sigma', 'RPS_val']).agg({
#     'err_DP': 'sum',
#     'err_LC': 'sum',
#     'err_LR': 'sum',
#     'err_GCV': 'sum',
#     'err_oracle': 'sum'
# })

# filepath = "/home/kimjosy/LocReg_Regularization-1/data/debugfigures/peakresolution"
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(T2_values, signal1, label = "signal1")
# plt.savefig(f"{filepath}/signal1")