# utils/paths.py
import os
from src.utils.helper_func.paths.paths_funcs import run_dir, get_filepath
# Project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
##### Stable project directories

#Data Directory
DATA_DIR_PATH = run_dir(ROOT_DIR, "data")

#Results Directory
RES_DIR_PATH = run_dir(ROOT_DIR, "results")
print()
#Subdirectories
BRAIN_RES_DIR = run_dir(ROOT_DIR, "results", "brain")
CLASS_RES_DIR = run_dir(ROOT_DIR, "results", "classical")
SC_RES_DIR = run_dir(ROOT_DIR, "results", "spinal_cord")

TOOLS_DIR = run_dir(ROOT_DIR, "tools")
SIM_SCRIPTS_DIR = run_dir(ROOT_DIR, "src", "sim_scripts")
REG_METHODS_DIR = run_dir(ROOT_DIR, "src", "regularization", "reg_methods")
CLASSICAL_DIR = run_dir(ROOT_DIR, "src", "regularization", "classical")

# MOSEK license
MOSEK_LICENSE =get_filepath("mosek.lic")

#Common Brain Paths
RAW_BR_DATA_PATH = get_filepath("mew_cleaned_brain_data_unfiltered.mat")
CLEAN_BR_DATA_PATH = get_filepath("cleaned_brain_data.mat")
MASK_PATH = get_filepath("new_mask.mat")
FILT_SNR_MAP_PATH = get_filepath("new_SNR_Map.mat")
UNFILT_SNR_MAP_PATH = get_filepath("new_SNR_Map_unfiltered.mat")

