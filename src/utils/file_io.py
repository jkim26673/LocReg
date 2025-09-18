import pickle
import os

def load_est_table(est_filepath):
    try:
        os.path.exists(est_filepath)
    except:
        AssertionError("Estimates file path does not exist.")
    with open(est_filepath, 'rb') as f:
        df = pickle.load(f)
    return df