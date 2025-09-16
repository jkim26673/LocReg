# common.py

# System and path
import os
import sys
import logging
import time
from datetime import datetime, date
import functools
import random
import cProfile
import pstats

# Append current directory or custom path if needed
print("Setting system path")
sys.path.append(".")

# Math and science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm as normsci, wasserstein_distance, entropy
from scipy.linalg import norm as linalg_norm, svd
from scipy.optimize import nnls
import cvxpy as cp
import scipy

# Parallel processing
import multiprocess as mp
from multiprocessing import Pool, freeze_support, set_start_method

# Progress bar
from tqdm import tqdm

# Optimization / MOSEK
import mosek

# Regularization methods
from regularization.reg_methods.dp.discrep_L2 import discrep_L2
from regularization.reg_methods.dp.discrep import discrep
from regularization.reg_methods.gcv.GCV_NNLS import GCV_NNLS
from regularization.reg_methods.lcurve.Lcurve import Lcurve
from regularization.reg_methods.lcurve import l_curve
from regularization.reg_methods.locreg.LocReg import LocReg as Chuan_LR
from regularization.reg_methods.locreg.Ito_LocReg import *  # use carefully
from regularization.reg_methods.locreg.LocReg import LocReg
from regularization.reg_methods.locreg.LocReg_unconstrainedB import LocReg_unconstrainedB
from regularization.reg_methods.locreg.LocReg_NEW_NNLS import LocReg_NEW_NNLS
from regularization.reg_methods.spanreg.Multi_Reg_Gaussian_Sum1 import Multi_Reg_Gaussian_Sum1
from regularization.reg_methods.nnls.tikhonov_vec import tikhonov_vec
from regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm
from regularization.reg_methods.upen.upencode import upen_param_setup, upen_setup
from regularization.reg_methods.upen.upenzama import UPEN_Zama, UPEN_Zama0th, UPEN_Zama1st

# Subfunctions
from regularization.subfunc.lcurve_functions import l_cuve, csvd, l_corner
from regularization.subfunc.l_curve_corner import l_curve_corner
from regularization.subfunc.csvd import csvd  # may be duplicate of above

# Tools and utils
from tools.trips_py.pasha_gcv import Tikhonov
from sim_scripts.peak_resolution_scripts.resolutionanalysis import find_min_between_peaks, check_resolution

# Plotting tweaks
import matplotlib.ticker as ticker

#import mosekpath
mosek_license_path = r"/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/tools/mosek/mosek.lic"
os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path


# IO
import pickle
__all__ = [
    'np', 'pd', 'plt', 'sns', 'normsci', 'linalg_norm', 'nnls',
    'cp', 'svd', 'discrep_L2', 'discrep', 'GCV_NNLS', 'Lcurve',
    'l_curve', 'Chuan_LR', 'LocReg', 'LocReg_unconstrainedB', 'LocReg_NEW_NNLS',
    'Multi_Reg_Gaussian_Sum1', 'tikhonov_vec', 'nonnegtik_hnorm',
    'upen_param_setup', 'upen_setup', 'UPEN_Zama', 'UPEN_Zama0th', 'UPEN_Zama1st',
    'l_cuve', 'csvd', 'l_corner', 'l_curve_corner',
    'Tikhonov', 'find_min_between_peaks', 'check_resolution',
    'wasserstein_distance', 'entropy', 'tqdm', 'mp', 'Pool', 'freeze_support',
    'set_start_method', 'mosek', 'ticker', 'datetime', 'date', 'time',
    'functools', 'random', 'cProfile', 'pstats', 'os', 'sys', 'logging', 'pickle'
]
