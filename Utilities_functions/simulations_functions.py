import numpy as np
import os
import pickle
import sys
sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')  # Replace this path with the actual path to the parent directory of Utilities_functions
from Utilities_functions.generate_gaussian_regs_L2_old import generate_gaussian_regs_L2_old
from Simulations.heatmap_unequal_width_All import heatmap_unequal_width_All
import concurrent.futures
import numpy as np
from scipy.stats import norm
from scipy.linalg import norm as linalg_norm
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, entropy
import pickle
from tqdm import tqdm
from Utilities_functions.GCV_NNLS import GCV_NNLS
from Utilities_functions.Lcurve import Lcurve
from Utilities_functions.LocReg import LocReg
from Utilities_functions.LocReg_unconstrainedB import LocReg_unconstrainedB
from Utilities_functions.discrep_L2 import discrep_L2
from Utilities_functions.Multi_Reg_Gaussian_Sum1 import Multi_Reg_Gaussian_Sum1
from Utilities_functions.LocReg_NEW_NNLS import LocReg_NEW_NNLS
import concurrent.futures


import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ... (existing code)

