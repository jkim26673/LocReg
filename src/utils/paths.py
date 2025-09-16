# utils/paths.py
import os

# Project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Stable project directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
SIM_SCRIPTS_DIR = os.path.join(ROOT_DIR, "src", "sim_scripts")
REG_METHODS_DIR = os.path.join(ROOT_DIR, "src", "regularization", "reg_methods")
CLASSICAL_DIR = os.path.join(ROOT_DIR, "src", "regularization", "classical")

# MOSEK license
MOSEK_LICENSE = os.path.join(TOOLS_DIR, "mosek", "mosek.lic")

# Helper functions for run-specific paths
def run_dir(run_name: str):
    """
    Returns a results subdirectory for a specific experiment/run.
    Creates the directory if it doesn't exist.
    """
    path = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(path, exist_ok=True)
    return path

def data_file(filename: str):
    """
    Returns full path to a file in the data directory.
    """
    return os.path.join(DATA_DIR, filename)
