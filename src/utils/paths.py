# utils/paths.py
import os
from src.utils.helper_func.paths.paths_funcs import run_dir
# Project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

##### Stable project directories

#Data Directory
DATA_DIR_PATH = run_dir(ROOT_DIR, "data")

#Results Directory
RES_DIR_PATH = run_dir(ROOT_DIR, "results")
#Subdirectories
BRAIN_RES_DIR = run_dir(ROOT_DIR, "results", "brain")
CLASS_RES_DIR = run_dir(ROOT_DIR, "results", "classical")
SC_RES_DIR = run_dir(ROOT_DIR, "results", "spinal_cord")

TOOLS_DIR = run_dir(ROOT_DIR, "tools")
SIM_SCRIPTS_DIR = run_dir(ROOT_DIR, "src", "sim_scripts")
REG_METHODS_DIR = run_dir(ROOT_DIR, "src", "regularization", "reg_methods")
CLASSICAL_DIR = run_dir(ROOT_DIR, "src", "regularization", "classical")

# MOSEK license
MOSEK_LICENSE =run_dir(TOOLS_DIR, "mosek", "mosek.lic")



