# -----------------------------
# 🔧 System & Utility Libraries
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

# Add custom paths (if needed)
print("Setting system path")
sys.path.append(".")

# -----------------------------
# 📦 Set Up Environment Variables
# -----------------------------
mosek_license_path = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/tools/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------
# 📊 Scientific Libraries
# -----------------------------
import numpy as np
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from scipy.stats import norm as normsci, wasserstein_distance, entropy
from scipy.linalg import norm as linalg_norm, svd
from scipy.optimize import nnls
from scipy.integrate import simpson
import cvxpy as cp
import scipy

# -----------------------------
# ⚙️ Parallel Processing
# -----------------------------
import multiprocess as mp
from multiprocessing import Pool, freeze_support, set_start_method

# -----------------------------
# ⏳ Progress Bar
# -----------------------------
from tqdm import tqdm

# -----------------------------
# ✅ MOSEK (after setting license)
# -----------------------------
import mosek

# -----------------------------
# 🔬 Regularization Modules
# -----------------------------
from regularization.reg_methods.dp.discrep_L2 import discrep_L2
from regularization.reg_methods.dp.discrep import discrep
from regularization.reg_methods.gcv.GCV_NNLS import GCV_NNLS
from regularization.reg_methods.gcv.gcv import gcv
from regularization.reg_methods.lcurve.Lcurve import Lcurve
from regularization.reg_methods.lcurve import l_curve

from regularization.reg_methods.locreg.LocReg import LocReg, LocReg as Chuan_LR
from regularization.reg_methods.locreg.LocReg_unconstrainedB import LocReg_unconstrainedB
from regularization.reg_methods.locreg.LocReg_NEW_NNLS import LocReg_NEW_NNLS
from regularization.reg_methods.locreg.Ito_LocReg import *  # ⚠️ careful with *
from regularization.reg_methods.spanreg.Multi_Reg_Gaussian_Sum1 import Multi_Reg_Gaussian_Sum1
from regularization.reg_methods.nnls.tikhonov_vec import tikhonov_vec
from regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm
from regularization.reg_methods.nnls.lsqnonneg import lsqnonneg
from regularization.reg_methods.spanreg.generate_gaussian_regs_L2_old import generate_gaussian_regs_L2_old
from sim_scripts.fivebyfiveMRR_script.gen_spanreg_heatmap_copyreference import heatmap_unequal_width_All
from regularization.reg_methods.locreg.LocReg_unconstrained import LocReg_unconstrained
from regularization.reg_methods.locreg.LocReg_v2 import LocReg_v2
from regularization.reg_methods.locreg.LocReg_NE import LocReg_unconstrained_NE
from regularization.reg_methods.locreg.TwoParam_LR import Multi_Param_LR
from regularization.reg_methods.nnls_multiparameter.tikhonov_multi_param import tikhonov_multi_param

# -----------------------------
# 🧩 Subfunctions
# -----------------------------
from regularization.subfunc.lcurve_functions import l_cuve, csvd, l_corner
from regularization.subfunc.l_curve_corner import l_curve_corner
# csvd imported again — no need to re-import

# -----------------------------
# 🔧 Tools and Utilities
# -----------------------------
from tools.trips_py.pasha_gcv import Tikhonov
from sim_scripts.peak_resolution_scripts.resolutionanalysis import find_min_between_peaks, check_resolution

# -----------------------------
# 🧠 Expose Common Names
# -----------------------------
__all__ = [
    # Scientific libs
    'np', 'pd', 'plt', 'sns', 'table',
    'normsci', 'wasserstein_distance', 'entropy',
    'linalg_norm', 'svd', 'nnls', 'simpson',
    'cp', 'scipy',

    # System/util
    'os', 'sys', 'logging', 'time', 'datetime', 'date',
    'functools', 'random', 'cProfile', 'pstats', 'subprocess', 'unittest', 'timeit',

    # Parallel
    'mp', 'Pool', 'freeze_support', 'set_start_method',

    # MOSEK
    'mosek',

    # Progress bar
    'tqdm',

    # Regularization
    'discrep_L2', 'discrep', 'GCV_NNLS', 'gcv',
    'Lcurve', 'l_curve',
    'LocReg', 'Chuan_LR', 'LocReg_unconstrainedB', 'LocReg_NEW_NNLS',
    'Multi_Reg_Gaussian_Sum1', 'tikhonov_vec', 'nonnegtik_hnorm',
    'upen_param_setup', 'upen_setup', 'UPEN_Zama', 'UPEN_Zama0th', 'UPEN_Zama1st',
    'lsqnonneg', 'generate_gaussian_regs_L2_old',
    'heatmap_unequal_width_All','Multi_Param_LR', 'tikhonov_multi_param',

    # Subfunctions
    'l_cuve', 'csvd', 'l_corner', 'l_curve_corner',

    # Tools
    'Tikhonov', 'find_min_between_peaks', 'check_resolution',

    # Plotting
    'ticker'
]
