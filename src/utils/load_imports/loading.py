# -----------------------------
# System & Utility Libraries
# -----------------------------
import os
import sys
import logging
import time
import timeit
from datetime import datetime, date
import functools
import random
import cProfile
import pstats
import subprocess
import unittest
import pickle


# Add project root to sys.path (if needed)
print("Setting system path")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -----------------------------
# Environment Variables
# -----------------------------
mosek_license_path = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/tools/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------
# Scientific Libraries
# -----------------------------
import numpy as np
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from scipy.stats import norm as norm, wasserstein_distance, entropy
from scipy.linalg import norm as linalg_norm, svd
from scipy.optimize import nnls
from scipy.integrate import simpson
from scipy import sparse
import cvxpy as cp
import scipy

# -----------------------------
# Parallel Processing
# -----------------------------
import multiprocess as mp
from multiprocessing import Pool, freeze_support, set_start_method

# -----------------------------
# Progress Bar
# -----------------------------
from tqdm import tqdm

# -----------------------------
# MOSEK (after setting license)
# -----------------------------
import mosek
mosek_license_path = r"/Users/kimjosy/Downloads/LocReg/tools/mosek/mosek.lic"

# -----------------------------
# Common Names
# -----------------------------
__all__ = [
    # Scientific libs
    'np', 'pd', 'plt', 'sns', 'table',
    'norm', 'wasserstein_distance', 'entropy',
    'linalg_norm', 'svd', 'nnls', 'simpson',
    'cp', 'scipy',

    # System/util
    'os', 'sys', 'logging', 'time', 'datetime', 'date',
    'functools', 'random', 'cProfile', 'pstats',
    'subprocess', 'unittest', 'timeit', 'pickle',

    # Parallel
    'mp', 'Pool', 'freeze_support', 'set_start_method',

    # MOSEK
    'mosek',

    # Progress bar
    'tqdm',

    # Plotting
    'ticker'
]
