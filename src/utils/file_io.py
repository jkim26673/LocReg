import pickle
import os
from src.utils.helper_func.paths.paths_funcs import run_dir

def load_est_table(est_filepath:str):
    try:
        os.path.exists(est_filepath)
    except:
        AssertionError("Estimates file path does not exist.")
    with open(est_filepath, 'rb') as f:
        df = pickle.load(f)
    return df
