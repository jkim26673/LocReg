
import pickle
with open("/home/kimjosy/LocReg_Regularization-1/Wing_20240624_SNR_3002/Wing_rankings.pkl", 'rb') as f:
    wing = pickle.load(f)
print(wing)

with open("Shaw_20240623_SNR_3003/Shaw_rankings.pkl", 'rb') as f:
    shaw = pickle.load(f)
print(shaw)

with open("baart_20240623_SNR_3002/baart_rankings.pkl", 'rb') as f:
    baart = pickle.load(f)
print(baart)

with open("Phillips_20240625_SNR_3005/Phillips_rankings.pkl", 'rb') as f:
    phillip = pickle.load(f)
print(phillip)

with open("Heat_20240624_SNR_3002/Heat_rankings.pkl", 'rb') as f:
    heat = pickle.load(f)
print(heat)

with open("Gravity_20240624_SNR_3002/Gravity_rankings.pkl", 'rb') as f:
    gravity = pickle.load(f)
print(gravity)

with open("Foxgood_20240624_SNR_3002/Foxgood_rankings.pkl", 'rb') as f:
    foxgood = pickle.load(f)
print(foxgood)

with open("Deriv2_20240623_SNR_3002/Deriv2_rankings.pkl", 'rb') as f:
    deriv2 = pickle.load(f)
print(deriv2)


with open("/home/kimjosy/LocReg_Regularization-1/SimulationSets/classical_prob/baart/baart_20240702_SNR_30/baart_rankings_SNR_30_exp_0_DP.pkl", 'rb') as f:
    deriv2 = pickle.load(f)
print(deriv2)

with open("SimulationSets/classical_prob/baart/baart_20240702_SNR_30/baart_rankings_SNR_30_exp_0_LC.pkl", 'rb') as f:
    deriv2 = pickle.load(f)
print(deriv2)


with open("SimulationSets/classical_prob/baart/baart_20240702_SNR_30/baart_rankings_SNR_30_exp_0_GCV.pkl", 'rb') as f:
    deriv2 = pickle.load(f)
print(deriv2)


import pickle
with open("SimulationSets/MRR/oldtest/comparison_20240714_SNR_2001/comparison_rankings_SNR_200.pkl", 'rb') as f:
    blur = pickle.load(f)
print(blur)

import pickle
with open("SimulationSets/MRR/Laplace/2Bump/L/R/mu1_40.0_mu2_160.0/SNR_200/expval_0.5/Laplace_20240714_SNR200/Laplace_rankings_SNR_200_exp_0.5_LC.pkl", 'rb') as f:
    blur = pickle.load(f)
print(blur)


import os
import pandas as pd
import pickle

# Define a function to load pickle file into DataFrame
def load_pickle_to_df(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return pd.DataFrame(data)

# Define paths to all pickle files
pickle_files = [
    "/home/kimjosy/LocReg_Regularization-1/Wing_20240624_SNR_3002/Wing_rankings.pkl",
    "Shaw_20240623_SNR_3003/Shaw_rankings.pkl",
    "baart_20240623_SNR_3002/baart_rankings.pkl",
    "Phillips_20240625_SNR_3005/Phillips_rankings.pkl",
    "Heat_20240624_SNR_3002/Heat_rankings.pkl",
    "Gravity_20240624_SNR_3002/Gravity_rankings.pkl",
    "Foxgood_20240624_SNR_3002/Foxgood_rankings.pkl",
    "Deriv2_20240623_SNR_3002/Deriv2_rankings.pkl",
    "blur_20240623_SNR_3002/blur_rankings.pkl"
]

# Load all pickle files into a list of DataFrames
dfs = [load_pickle_to_df(file) for file in pickle_files]
import pandas as pd

# Assuming dfs contains all the DataFrames loaded from pickle files as shown
# and concatenated_df is already defined

# Concatenate DataFrames along axis 1 (columns)
concatenated_df = pd.concat(dfs, axis=1)

# Initialize empty lists to store results
# Initialize empty lists to store results
averages_medians = []
averages_ranks = []

# Iterate over each row in the concatenated DataFrame
for index, row in concatenated_df.iterrows():
    # Calculate the average of 'medians' and 'ranks' for the current row
    average_medians = row['medians'].median()
    average_ranks = row['ranks'].median()
    
    # Append averages to the respective lists
    averages_medians.append(average_medians)
    averages_ranks.append(average_ranks)

# Create a DataFrame to display the results
average_df = pd.DataFrame({

    # 'Average Errors': averages_medians,
    'Median Ranks': averages_ranks,
    "SNR": 300
}, index=concatenated_df.index)

print(average_df)




import pandas as pd
import pickle

# Define a function to load pickle file into DataFrame
def load_pickle_to_df(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return pd.DataFrame(data)

# Define paths to all pickle files
pickle_files = [
    "/home/kimjosy/LocReg_Regularization-1/Wing_20240624_SNR_3002/Wing_rankings.pkl",
    "Shaw_20240623_SNR_3003/Shaw_rankings.pkl",
    "baart_20240623_SNR_3002/baart_rankings.pkl",
    "Phillips_20240625_SNR_3005/Phillips_rankings.pkl",
    "Heat_20240624_SNR_3002/Heat_rankings.pkl",
    "Gravity_20240624_SNR_3002/Gravity_rankings.pkl",
    "Foxgood_20240624_SNR_3002/Foxgood_rankings.pkl",
    "Deriv2_20240623_SNR_3002/Deriv2_rankings.pkl",
    "blur_20240623_SNR_3002/blur_rankings.pkl"
]

problem_dfs = {}

# Load all pickle files into a dictionary of DataFrames
for pickle_file in pickle_files:
    problem_name = os.path.basename(pickle_file).split('_rankings.pkl')[0]
    problem_dfs[problem_name] = load_pickle_to_df(pickle_file)

# Display or manipulate the DataFrames as needed
for problem_name, df in problem_dfs.items():
    print(f"Data for problem: {problem_name}")
    print(df)

    print("\n")