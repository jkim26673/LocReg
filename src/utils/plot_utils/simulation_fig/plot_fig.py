import matplotlib.pyplot as plt
 
fig, axs = plt.subplots(2, 2, figsize=(12 ,12))
# plt.subplots_adjust(wspace=0.3)

# Plotting the first subplot
# plt.subplot(1, 3, 1)

ymax = np.max(g) * 1.15
axs[0, 0].plot(T2, g, color = "black",  label = "Ground Truth")
axs[0, 0].plot(T2, f_rec_OP_grid, color = "purple",  label = f"Oracle 1P (Error: {round(err_oracle,3)})")
axs[0, 0].plot(T2, LR_mod_rec, color = "blue",  label = f"Ito Derivative (Error: {round(err_LR_Ito,3)})")
axs[0, 0].plot(T2, LR_mod_rec2, color = "gold",  label = f"Ito No Derivative (Error: {round(err_LR_Ito2,3)})")
axs[0, 0].plot(T2, LR_mod_rec3, color = "cyan",  label = f"Ito Gamma No Derivative (Error: {round(err_LR_Ito3,3)})")
axs[0, 0].plot(T2, LR_mod_rec4, color = "brown",  label = f"Ito Gamma Derivative (Error: {round(err_LR_Ito4,3)})")
axs[0, 0].plot(T2, f_rec_LC, color = "orange", label = f"LC 1P (Error: {round(err_Lcurve,3)})")
# axs[0, 0].plot(T2, f_test, color = "cyan", label = "test")
axs[0, 0].plot(T2, f_rec_GCV, color = "green", label = f"GCV 1P (Error:{round(err_GCV,3)}")
axs[0, 0].plot(T2, f_rec_DP, color = "red", label = f"DP 1P (Error:{round(err_DP,3)})")
# axs[0, 0].plot(T2, f_rec_Chuan, color = "red", label = "Chuan")
axs[0, 0].set_xlabel('t', fontsize=20, fontweight='bold')
axs[0, 0].set_ylabel('f(t)', fontsize=20, fontweight='bold')
axs[0, 0].legend(fontsize=10, loc='best')
axs[0, 0].set_ylim(0, ymax)
axs[0, 0].set_title('Reconstruction', fontsize=16, fontweight='bold')  # Add title here

# Plotting the second subplot
# plt.subplot(1, 3, 2)
axs[0, 1].plot(TE, G @ g, linewidth=3, color='black', label='Ground Truth')
axs[0, 1].plot(TE, G @ f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
axs[0, 1].plot(TE, G @ LR_mod_rec, color = "blue",  label = "Ito Derivative")
axs[0, 1].plot(TE, G @ LR_mod_rec2, color = "gold",  label = "Ito No Derivative")
axs[0, 1].plot(TE, G @ LR_mod_rec3, color = "cyan",  label = "Ito Gamma No Derivative")
axs[0, 1].plot(TE, G @ LR_mod_rec4, color = "brown",  label = "Ito Gamma Derivative")
axs[0, 1].plot(TE, G @ f_rec_LC, color = "orange", label = "LC 1P")
axs[0, 1].plot(TE, G @ f_rec_GCV, color = "green", label = "GCV 1P")
axs[0, 1].plot(TE, G @ f_rec_DP, color = "red", label = "DP 1P")

# axs[0, 1].plot(TE, G @ f_rec_Chuan, color = "red", label = "Chuan")
axs[0, 1].legend(fontsize=10, loc='best')
axs[0, 1].set_xlabel('s', fontsize=20, fontweight='bold')
axs[0, 1].set_ylabel('g(s)', fontsize=20, fontweight='bold')
axs[0, 1].set_title('Data', fontsize=16, fontweight='bold')  # Add title here

# plt.subplot(1, 3, 3)
axs[1, 0].semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='orange', label='LC 1P')
axs[1, 0].semilogy(T2, oracle_lam * np.ones(len(T2)), linewidth=3, color='purple', label='Oracle 1P')
axs[1, 0].semilogy(T2, LR_Ito_lams, color = "blue", linewidth=3,  label = "Ito Derivative")
axs[1, 0].semilogy(T2, LR_Ito_lams2, color = "gold", linewidth=3,  label = "Ito No Derivative")
axs[1, 0].semilogy(T2, LR_Ito_lams3, color = "cyan", linewidth=3,  label = "Ito Gamma No Derivative")
axs[1, 0].semilogy(T2, LR_Ito_lams4, color = "brown", linewidth=3,  label = "Ito Gamma Derivative")

# axs[1,0].semilogy(T2, lambda_Chuan, color = "red",  label = "Chuan")
axs[1, 0].semilogy(T2, lambda_GCV * np.ones(len(T2)),linewidth=3,  color = "green", label = "GCV 1P")
axs[1, 0].semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color = "red", label = "DP 1P")
axs[1, 0].legend(fontsize=10, loc='best')
axs[1, 0].set_xlabel('t', fontsize=20, fontweight='bold')
axs[1, 0].set_ylabel('Lambda', fontsize=20, fontweight='bold')
axs[1, 0].set_title('Regularization Distribution', fontsize=16, fontweight='bold')  # Add title here

# ymax2 = 1.5 * np.max(lambda_LC)
# axs[1, 0].set_ylim(0, ymax2)

table_ax = axs[1, 1]
table_ax.axis('off')

# Define the data for the table (This is part of the plot)
data = [
    # ["L-Curve Lambda", lambda_LC.item()],
    # ["Initial Lambda for Ito", LRIto_ini_lam],
    # ["Initial Eta2 for Ito", round(lam_ini, 4)],
    # ["Initial Eta2 for Ito", LRIto_ini_lam],
    # ["Final Eta1 for Ito", fin_etas[0].item()],
    # ["Final Eta2 for Ito", fin_etas[1].item()],
    ["Error 1P DP", err_DP.item()],
    ["Error 1P L-Curve", err_Lcurve.item()],
    ["Error 1P GCV", err_GCV.item()],
    ["Error Ito Derivative No Feedback", err_LR_Ito.item()],
    ["Error Ito No Derivative No Feedback", err_LR_Ito2.item()],
    ["Error Ito Gamma Derivative No Feedback", err_LR_Ito4.item()],
    ["Error Ito Gamma No Derivative No Feedback", err_LR_Ito3.item()],
    # ["error Ito 2P", err_Ito2P.item()],
    ["Error 1P Oracle", err_oracle.item()],
    # ["error test", err_test.item()],
    # ["error Chuan", err_Chuan.item()],
    ["SNR", SNR],
    ["Feedback", feedback],
    ["Exponent", exp]

    # ["Initial Lambdas for Ito Loc", LR_ini_lam],
    # ["Final Lambdas for Ito Loc", LR_Ito_lams]
]

# Create the table
table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.5)
table_ax.set_title('Baart Problem No Feedback Table', fontsize=16, fontweight='bold', y=1.2)  # Adjust the y position

#Save the results in the save results folder
string = "baart"
file_path = create_result_folder(string, SNR)
exp_str = str(int(exp * 100))
plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve_feedback{i}_{lam_ini}_exp_{exp_str}"))
print(f"Saving comparison plot is complete")
plt.close()

fig, axs = plt.subplots(2, 2, figsize=(12 ,12))

ymax = np.max(g) * 1.15
axs[0, 0].plot(T2, g, color = "black",  label = "Ground Truth")
axs[0, 0].plot(T2, f_rec_OP_grid, color = "purple",  label = f"Oracle 1P (Error: {round(err_oracle,3)})")
axs[0, 0].plot(T2, LR_mod_recfeed, color = "blue",  label = f"Ito Derivative feedback (Error: {round(err_LR_Itofeed,3)})")
axs[0, 0].plot(T2, LR_mod_rec2feed, color = "gold",  label = f"Ito No Derivative feedback (Error: {round(err_LR_Ito2feed,3)})")
axs[0, 0].plot(T2, LR_mod_rec3feed, color = "cyan",  label = f"Error Ito Gamma No Derivative feedback (Error: {round(err_LR_Ito3feed,3)})")
axs[0, 0].plot(T2, LR_mod_rec4feed, color = "brown",  label = f"Ito Gamma Derivative feedback (Error: {round(err_LR_Ito4feed,3)})")

# axs[0, 0].plot(T2, f_rec_Chuan, color = "red", label = "Chuan")
axs[0, 0].set_xlabel('t', fontsize=20, fontweight='bold')
axs[0, 0].set_ylabel('f(t)', fontsize=20, fontweight='bold')
axs[0 ,0].legend(fontsize=10, loc='best')
axs[0 ,0].set_ylim(0, ymax)
axs[0 ,0].set_title('Reconstruction', fontsize=16, fontweight='bold')  # Add title here

axs[0 ,1].plot(TE, G @ g, linewidth=3, color='black', label='Ground Truth')
axs[0 ,1].plot(TE, G @ f_rec_OP_grid, color = "purple",  label = "Oracle 1P")
axs[0 ,1].plot(TE, G @ LR_mod_recfeed, color = "blue",  label = "Ito Derivative Feedback")
axs[0 ,1].plot(TE, G @ LR_mod_rec2feed, color = "gold",  label = "Ito No Derivative Feedback")
axs[0 ,1].plot(TE, G @ LR_mod_rec3feed, color = "cyan",  label = "Ito Gamma No Derivative Feedback")
axs[0 ,1].plot(TE, G @ LR_mod_rec4feed, color = "brown",  label = "Ito Gamma Derivative Feedback")
axs[0 ,1].plot(TE, G @ f_rec_LC, color = "orange", label = "LC 1P")
axs[0 ,1].plot(TE, G @ f_rec_GCV, color = "green", label = "GCV 1P")
axs[0 ,1].plot(TE, G @ f_rec_DP, color = "red", label = "DP 1P")

# axs[0, 1].plot(TE, G @ f_rec_Chuan, color = "red", label = "Chuan")
axs[0 ,1].legend(fontsize=10, loc='best')
axs[0 ,1].set_xlabel('s', fontsize=20, fontweight='bold')
axs[0 ,1].set_ylabel('g(s)', fontsize=20, fontweight='bold')
axs[0 ,1].set_title('Data', fontsize=16, fontweight='bold')  # Add title here


axs[1, 0].semilogy(T2, lambda_LC * np.ones(len(T2)), linewidth=3, color='orange', label='LC 1P')
axs[1, 0].semilogy(T2, oracle_lam * np.ones(len(T2)), linewidth=3, color='purple', label='Oracle 1P')
axs[1, 0].semilogy(T2, LR_Ito_lamsfeed, color = "blue", linewidth=3,  label = "Ito Derivative Feedback")
axs[1, 0].semilogy(T2, LR_Ito_lams2feed, color = "gold", linewidth=3,  label = "Ito No Derivative Feedback")
axs[1, 0].semilogy(T2, LR_Ito_lams3feed, color = "cyan", linewidth=3,  label = "Ito Gamma No Derivative Feedback")
axs[1, 0].semilogy(T2, LR_Ito_lams4feed, color = "brown", linewidth=3,  label = "Ito Gamma Derivative Feedback")

# axs[1,0].semilogy(T2, lambda_Chuan, color = "red",  label = "Chuan")
axs[1, 0].semilogy(T2, lambda_GCV * np.ones(len(T2)),linewidth=3,  color = "green", label = "GCV 1P")
axs[1, 0].semilogy(T2, lambda_DP * np.ones(len(T2)), linewidth=3, color = "red", label = "DP 1P")
axs[1, 0].legend(fontsize=10, loc='best')
axs[1, 0].set_xlabel('t', fontsize=20, fontweight='bold')
axs[1, 0].set_ylabel('Lambda', fontsize=20, fontweight='bold')
axs[1, 0].set_title('Regularization Distribution', fontsize=16, fontweight='bold')  # Add title here


table_ax = axs[1, 1]
table_ax.axis('off')

# Define the data for the table (This is part of the plot)
data = [
    # ["L-Curve Lambda", lambda_LC.item()],
    # ["Initial Lambda for Ito", LRIto_ini_lam],
    # ["Initial Eta2 for Ito", round(lam_ini, 4)],
    # ["Initial Eta2 for Ito", LRIto_ini_lam],
    # ["Final Eta1 for Ito", fin_etas[0].item()],
    # ["Final Eta2 for Ito", fin_etas[1].item()],
    ["Error 1P DP", err_DP.item()],
    ["Error 1P L-Curve", err_Lcurve.item()],
    ["Error 1P GCV", err_GCV.item()],
    ["Error Ito Derivative Feedback", err_LR_Itofeed.item()],
    ["Error Ito No Derivative Feedback", err_LR_Ito2feed.item()],
    ["Error Ito Gamma Derivative Feedback", err_LR_Ito4feed.item()],
    ["Error Ito Gamma No Derivative Feedback", err_LR_Ito3feed.item()],
    # ["error Ito 2P", err_Ito2P.item()],
    ["Error 1P Oracle ", err_oracle.item()],
    # ["error test", err_test.item()],
    # ["error Chuan", err_Chuan.item()],
    ["SNR", SNR],
    ["Feedback", True],
    ["Exponent", exp]

    # ["Initial Lambdas for Ito Loc", LR_ini_lam],
    # ["Final Lambdas for Ito Loc", LR_Ito_lams]
]

# Create the table
table = table_ax.table(cellText=data, loc='center', cellLoc='center', colLabels=['Metric', 'Value'])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.5)
table_ax.set_title('Baart Problem with Feedback Table', fontsize=16, fontweight='bold', y=1.2)  # Adjust the y position
string = "baart"
file_path = create_result_folder(string, SNR)
exp_str = str(int(exp * 100))
plt.savefig(os.path.join(file_path, f"Ito_LR_vs_L_curve_feedback{i}_{lam_ini}_exp_{exp_str}"))
print(f"Saving comparison plot is complete")
plt.close()