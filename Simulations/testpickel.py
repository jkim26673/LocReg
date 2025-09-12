import numpy as np
import matplotlib.pyplot as plt
# noise = np.load("/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-20_SNR_900_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_900_errtype_Wass. Score/est_table_SNR900_iter1_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_900_errtype_Wass. Score_20Nov24noise_arr.npy")
# noise = np.load("/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-20_SNR_900_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_900_errtype_Wass. Scorenoparallel/est_table_SNR900_iter1_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_900_errtype_Wass. Scorenoparallel_20Nov24noise_arr.npy")
# noise = np.load("/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-20_SNR_900_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_900_errtype_Wass. Scorenoparallel2/est_table_SNR900_iter1_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_900_errtype_Wass. Scorenoparallel2_20Nov24noise_arr.npy")
# noise = np.load("/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-20_SNR_900_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_900_errtype_Wass. Scoreparallel2/est_table_SNR900_iter1_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_900_errtype_Wass. Scoreparallel2_20Nov24noise_arr.npy")
noise = np.load("/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-20_SNR_901_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_901_errtype_Wass. Scoreparallel2/est_table_SNR901_iter1_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_901_errtype_Wass. Scoreparallel2_20Nov24noise_arr.npy")
print("noise",noise.shape)
map = noise[0]
print(map.shape)
NR1 = map[0,0,:]
NR2 = map[0,1,:]
NR3 = map[1,1,:]
NR4 = map[1,0,:]
plt.figure()
plt.plot(NR1)
plt.plot(NR2)
plt.plot(NR3)
plt.plot(NR4)
plt.savefig("NR4parallel2901")
print("saved fig")