import pickle
import os
from src.utils.helper_func.paths.paths_funcs import gen_results_dir
import pandas as pd
import glob 
from src.utils.file_io import load_est_table

def load_est_table(est_filepath:str):
    try:
        os.path.exists(est_filepath)
    except:
        AssertionError("Estimates file path does not exist.")
    with open(est_filepath, 'rb') as f:
        df = pickle.load(f)
    return df

def combine_files(folder_path: str):
    dfs = []
    txt_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    for file in txt_files:
        if os.path.basename(file).startswith("temp_"):
            df = load_est_table(file)  # assuming it returns a DataFrame
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()
    

