#This script generate 5 tables:
#1. It generates table comparing low SNR and high SNR against the 11 different methods of parameter selection (DP, GCV, Lcurve, 8 versions of Ito's algo (2 kinds of ito designs x feedback/no feedback loop x derivative matrix/no derivative matrix))
#2. For low SNR, it compares whether ito's starting initial lambda LC, GCV, or DP against 11 different methods of parameter selection (DP, GCV, Lcurve, 8 versions of Ito's algo (2 kinds of ito designs x feedback/no feedback loop x derivative matrix/no derivative matrix))
#3. For high SNR, it compares whether ito's starting initial lambda LC, GCV, or DP against 11 different methods of parameter selection (DP, GCV, Lcurve, 8 versions of Ito's algo (2 kinds of ito designs x feedback/no feedback loop x derivative matrix/no derivative matrix))
#4. For low SNR, it compares whether ito's starting initial lambda LC, GCV, or DP, including various exponent values used for ito's algorithm against 11 different methods of parameter selection (DP, GCV, Lcurve, 8 versions of Ito's algo (2 kinds of ito designs x feedback/no feedback loop x derivative matrix/no derivative matrix))

#Load packages
import os
import pandas as pd
import pickle
import sys
import os
from functools import reduce


#Define save folder
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
cwd = os.getcwd()
cwd_temp = os.getcwd()
base_file = 'LocReg_Regularization-1'
cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'
pat_tag = "classical_prob"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"
output_folder = f"SimulationSets/{pat_tag}"
cwd_full = cwd_cut + output_folder
import os
import pickle
import pandas as pd

# Define functions
def find_files_with_name(path, name_substr):
    matching_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if name_substr in file:
                matching_files.append(os.path.join(root, file))
    return matching_files

def update_path(initial_file_path, s):
    file_path = initial_file_path
    # Check if initial_file_path exists and is a directory
    if os.path.exists(initial_file_path) and os.path.isdir(initial_file_path):
        # Iterate through directories in initial_file_path
        for directory in os.listdir(initial_file_path):
            if s in directory:
                file_path = os.path.join(initial_file_path, directory)
                # print(f"Updated file_path to: {file_path}")
                return file_path  # Exit and return the updated file_path once found
    else:
        print(f"Directory {initial_file_path} does not exist or is not a directory")
    return file_path  # Return original initial_file_path if no update was made

def load_pickle_to_df(pickle_file, exp, lam_ini):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    renamed_columns = {col: f"{col}_{exp}_{lam_ini}" for col in df.columns}
    df = df.rename(columns=renamed_columns)
    return df

def summarytable(fileslist, prob, exp_values, lam_ini_values, level):
    dfs = []
    for file in fileslist:
        filename = os.path.basename(file)
        # Find exp value in filename
        exp = next((exp for exp in exp_values if f"_exp_{exp}_" in filename), None)
        # Find lam_ini value in filename
        lam_ini = next((lam_ini for lam_ini in lam_ini_values if f"_{lam_ini}." in filename), None)
        if exp and lam_ini:
            df = load_pickle_to_df(file, exp, lam_ini)
            dfs.append(df)
            # print(f"Loaded DataFrame for {exp}_{lam_ini} from {file}")
        else:
            print(f"File {file} does not match any exp or lam_ini value. Skipping.")
    
    # Concatenate DataFrames if there are multiple
    if len(dfs) > 1:
        dfs_concatenated = pd.concat(dfs, axis=1)
        # print("Detailed Merged DataFrame:")
        # print(dfs_concatenated)
        duplicated_columns = dfs_concatenated.columns[dfs_concatenated.columns.duplicated()]
        if duplicated_columns.any():
            print(f"Duplicated columns found: {duplicated_columns}")
        else:
            pass
            # print("No duplicated columns found.")

    grouped_by_medians = dfs_concatenated.filter(like='medians').groupby(lambda x: x.split('_')[-1], axis=1).mean()
    # Group by ranks and calculate average
    grouped_by_medians.columns = [f'{col}_Avg_Errors' for col in grouped_by_medians.columns]
    grouped_by_ranks = dfs_concatenated.filter(like='ranks').groupby(lambda x: x.split('_')[-1], axis=1).mean()
    grouped_by_ranks.columns = [f'{col}_Avg_Ranks' for col in grouped_by_ranks.columns]
    # Concatenate averages into a single DataFrame
    averages_df = pd.concat([grouped_by_medians, grouped_by_ranks], axis=1)
    # Rename columns for clarity
    # averages_df.columns = [f'Average Median {col.split("_")[-1]}' if 'medians' in col else f'Average Rank {col.split("_")[-1]}' for col in averages_df.columns]
    # print("Averages for Medians and Ranks:")
    # print(averages_df)

    final_avg_errors = averages_df.filter(like='Avg_Errors').mean(axis=1)

    # Compute the final average across Avg_Ranks
    final_avg_ranks = averages_df.filter(like='Avg_Ranks').mean(axis=1)

    # Combine into a single DataFrame
    final_average_df = pd.DataFrame({
        'Final Average Errors': final_avg_errors,
        'Final Average Ranks': final_avg_ranks
    })
    # print(f"Data for problem: {prob}")
    # print("Final Summarized DataFrame:")
    # print(final_average_df)
    if level == "all":
        return dfs_concatenated
    elif level == "groupbylambda":
        return averages_df
    elif level == "summarized":
        return final_average_df
    # elif level == "":


    # dfs2 = reduce(lambda left, right: pd.concat([left, right], axis=1), dfs)
    # dfs2 = dfs2.loc[:, ~dfs2.columns.str.contains('median')]
    # print("merged", dfs2)


    # dfs = [load_pickle_to_df(file) for file in matching_files]
    # print("df1", dfs[0])
    # print("df2", dfs[1])
    # merge_df = pd.concat([dfs[0],dfs[1]])
    # print("merged", dfs[1])



# Define the path to problem and SNR folder
prob = ["baart", "blur", "deriv2", "foxgood", "gravity", "heat", "phillips", "shaw", "wing"]
# prob = ["baart", "blur"]
SNR = ["30", "300"]
exp = ["0","1","0.66"]
lam_ini = ["LC", "GCV", "DP"]

# Process each problem and SNR value
summary_dict = {}
lambda_dict = {}
all_dict = {}

for p in prob:
    initial_file_path = os.path.join(cwd_full, p)
    # print("Completed initial path")
    # print("Initial_file_path:", initial_file_path)

    for s in SNR:
        updated_file_path = update_path(initial_file_path, s)
        # print("Updated_file_path:", updated_file_path)

        if updated_file_path:
            # print("Successfully updated path")
            
            # Find all files containing 'rankings' in their name within updated_file_path
            matching_files = find_files_with_name(updated_file_path, "rankings")
            
            if matching_files:
                # print(f"Found {len(matching_files)} files containing 'rankings' in {updated_file_path}:")
                # for file in matching_files:
                #     print(f"- {file}")
                finaltable = summarytable(matching_files, p, exp, lam_ini, level = "summarized")
                print(f"Data for problem: {p} for SNR {s}")
                print("Final Summarized DataFrame:")
                # print(finaltable)
                # print(finaltable)
                summary_dict[(p, s)] = finaltable
                lambda_dict[(p, s)] = summarytable(matching_files, p, exp, lam_ini, level = "groupbylambda")

                all_dict[(p, s)] = summarytable(matching_files, p, exp, lam_ini, level = "all")
                # finaltables.append(finaltable)
                # Optionally load these files into DataFrame using load_pickle_to_df function
                    # df = load_pickle_to_df(file)
                    # print(df)
                    #Create summary table here

                    # Process the DataFrame as needed
            else:
                print(f"No files containing 'rankings' found in {updated_file_path}")
        else:
            print("Path update failed, skipping further processing.")
# print(finaltables)
# print(len(finaltables))

# print(summary_dict)
# print("summary_dict")
# print(summary_dict)

# print("lambda_dict")
# print(lambda_dict)

# print("all_dict")
# print(all_dict)
# mean_ranks = {}

# # Iterate over summary_dict to calculate mean 'Final Average Ranks' for each SNR value
# for (prob, snr), df in summary_dict.items():
#     if snr not in mean_ranks:
#         mean_ranks[snr] = df['Final Average Ranks'].copy()  # Initialize with the first dataframe's ranks
#     else:
#         mean_ranks[snr] += df['Final Average Ranks']  # Add ranks from subsequent dataframes

# # Divide by the number of different problem types to get the mean
# for snr in mean_ranks:
#     mean_ranks[snr] /= len(summary_dict) // 2

# # Display the mean ranks for each SNR value
# for snr, ranks in mean_ranks.items():
#     print(f"Mean Final Average Ranks for SNR = {snr}:")
#     print(ranks)
#     print()



# mean_values = {}

# # Iterate over summary_dict to calculate mean 'Final Average Errors' and 'Final Average Ranks' for each SNR value
# for (prob, snr), df in summary_dict.items():
#     if snr not in mean_values:
#         mean_values[snr] = {
#             'Mean Final Average Errors': df['Final Average Errors'].copy(),  # Initialize with the first dataframe's errors
#             'Mean Final Average Ranks': df['Final Average Ranks'].copy()  # Initialize with the first dataframe's ranks
#         }
#     else:
#         mean_values[snr]['Mean Final Average Errors'] += df['Final Average Errors']  # Add errors from subsequent dataframes
#         mean_values[snr]['Mean Final Average Ranks'] += df['Final Average Ranks']  # Add ranks from subsequent dataframes

# # Divide by the number of different problem types to get the mean
# for snr in mean_values:
#     mean_values[snr]['Mean Final Average Errors'] /= len(summary_dict) // 2
#     mean_values[snr]['Mean Final Average Ranks'] /= len(summary_dict) // 2

# # Display the mean values for each SNR value
# for snr, values in mean_values.items():
#     print(f"Mean Values for SNR = {snr}:")
#     print(values)
#     print()

mean_ranks = {}

# Iterate over summary_dict to calculate mean 'Final Average Ranks' for each SNR value
for (prob, snr), df in summary_dict.items():
    if snr not in mean_ranks:
        mean_ranks[snr] = df['Final Average Ranks'].copy()  # Initialize with the ranks from the first dataframe
    else:
        mean_ranks[snr] += df['Final Average Ranks']  # Add ranks from subsequent dataframes

# Divide by the number of different problem types to get the mean
for snr in mean_ranks:
    mean_ranks[snr] /= len(summary_dict) // 2

# Display the mean ranks for each SNR value
# for snr, ranks in mean_ranks.items():
#     print(f"Mean Final Average Ranks for SNR = {snr}:")
#     print(ranks)
#     print()

# print(mean_ranks)
merged_df = pd.DataFrame(mean_ranks)

# Rename the columns to include "SNR"
merged_df.columns = ['SNR ' + col for col in merged_df.columns]

# Display the merged DataFrame
print("Merged DataFrame:")
print(merged_df)
# specific_df = summary_dict[('wing', '30')]

# # Printing the specific DataFrame
# print("DataFrame for 'wing' and SNR '300':")
# specific_df = summary_dict[('blur', '300')]
# print(specific_df)

# # Printing the specific DataFrame
# print("DataFrame for 'wing' and SNR '300':")
# print(specific_df)

# Example logic to find rows where 'ito' method has lower rank than 'oracle' method

# Initialize results list
results = []

# Iterate over summary_dict to find rows where 'ito' has lower rank than 'oracle'
for (prob, snr), df in summary_dict.items():
    # Filter rows where method contains 'ito'
    ito_rows = df[df.index.str.contains('ito')]
    
    # Filter rows where method contains 'oracle'
    oracle_rows = df[df.index.str.contains('oracle')]
    
    # Flag to track if any 'ito' row has lower rank than any 'oracle' row
    ito_lower_than_oracle = False
    
    # Iterate over each row in ito_rows and check against each row in oracle_rows
    for ito_index, ito_row in ito_rows.iterrows():
        for oracle_index, oracle_row in oracle_rows.iterrows():
            if ito_row['Final Average Ranks'] < oracle_row['Final Average Ranks']:
                results.append((prob, snr))
                ito_lower_than_oracle = True
                break  # Break inner loop if condition is met
        if ito_lower_than_oracle:
            break  # Break outer loop if condition is met
    
# Print the results
print("Problem and SNR values where 'ito' method has lower rank than 'oracle' method:")
for prob, snr in results:
    print(f"Problem: {prob}, SNR: {snr}")

# Initialize results list
# results = []

# # Methods to compare against 'ito'
# methods_to_compare = ['lc', 'gcv', 'dp']

# # Iterate over summary_dict to find rows where 'ito' has lower rank than all specified methods
# for (prob, snr), df in summary_dict.items():
#     # Filter rows where method contains 'ito'
#     ito_rows = df[df.index.str.contains('ito')]
    
#     # Track if 'ito' is better than all methods
#     ito_better_than_all = True
    
#     # Iterate over each method to compare against
#     for method in methods_to_compare:
#         # Filter rows where method contains the current comparison method
#         method_rows = df[df.index.str.contains(method)]
        
#         # Ensure both ito_rows and method_rows have common indices
#         common_indices = ito_rows.index.intersection(method_rows.index)
        
#         # Compare 'Final Average Ranks' where indices are common
#         if not (ito_rows.loc[common_indices, 'Final Average Ranks'] < method_rows.loc[common_indices, 'Final Average Ranks']).all():
#             ito_better_than_all = False
#             break  # No need to continue if 'ito' is not better than any method
    
#     # If 'ito' is better than all methods, append to results
#     if ito_better_than_all:
#         results.append((prob, snr))

# # Print the results
# print("Problem and SNR values where 'ito' method has lower rank than all specified methods (lc, gcv, dp):")
# for prob, snr in results:
#     print(f"Problem: {prob}, SNR: {snr}")
