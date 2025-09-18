# -----------------------------
# Base loader (safe imports only)
# -----------------------------
from src.utils.load_imports.loading import *

# -----------------------------
#Regularization Methods
# -----------------------------
from src.regularization.reg_methods.dp.discrep import discrep
from src.regularization.reg_methods.dp.discrep_L2 import discrep_L2
from src.regularization.reg_methods.dp.discrep_L2 import discrep_L2_brain
from src.regularization.reg_methods.gcv.GCV_NNLS import GCV_NNLS
from src.regularization.reg_methods.gcv.gcv import gcv
from src.regularization.reg_methods.lcurve.Lcurve import Lcurve
from src.regularization.reg_methods.lcurve.l_curve import l_curve
from src.regularization.reg_methods.oracle.oracle import oracle

from src.regularization.reg_methods.locreg.LocReg import LocReg
from src.regularization.reg_methods.locreg.prototypes.LocReg_unconstrainedB import LocReg_unconstrainedB
from src.regularization.reg_methods.locreg.prototypes.LocReg_NEW_NNLS import LocReg_NEW_NNLS
from src.regularization.reg_methods.locreg.prototypes.LocReg_unconstrained import LocReg_unconstrained
from src.regularization.reg_methods.locreg.LRalgo import LocReg_Ito_mod, LocReg_Ito_mod_deriv, LocReg_Ito_mod_deriv2
from src.regularization.reg_methods.locreg.prototypes.LocReg_v2 import LocReg_v2
from src.regularization.reg_methods.locreg.prototypes.LocReg_NE import LocReg_unconstrained_NE
from src.regularization.reg_methods.locreg.Ito_LocReg import *
from src.regularization.reg_methods.spanreg.Multi_Reg_Gaussian_Sum1 import Multi_Reg_Gaussian_Sum1
from src.regularization.reg_methods.nnls.tikhonov_vec import tikhonov_vec
from src.regularization.reg_methods.nnls.nonnegtik_hnorm import nonnegtik_hnorm
from src.regularization.reg_methods.nnls.lsqnonneg import lsqnonneg
from src.regularization.reg_methods.spanreg.generate_gaussian_regs_L2_old import generate_gaussian_regs_L2_old
from src.sim_scripts.fivebyfiveMRR.gen_spanreg_heatmap_copyreference import heatmap_unequal_width_All
from src.regularization.reg_methods.locreg.prototypes.TwoParam_LR import Multi_Param_LR
from src.regularization.reg_methods.nnls_multiparameter.tikhonov_multi_param import tikhonov_multi_param
from src.regularization.reg_methods.upen.upenzama import UPEN_Zama
from src.regularization.reg_methods.upen.upenzama import UPEN_Zama0th
from src.regularization.reg_methods.upen.upenzama import UPEN_Zama1st

# -----------------------------
# Subfunctions
# -----------------------------
from src.regularization.subfunc.lcurve_functions import l_cuve, l_corner
from src.regularization.subfunc.csvd import csvd
from src.regularization.subfunc.l_curve_corner import l_curve_corner

# -----------------------------
#  Tools
# -----------------------------
from tools.trips_py.pasha_gcv import Tikhonov
from src.sim_scripts.peak_resolution.resolutionanalysis import (
    find_min_between_peaks, check_resolution
)

# -----------------------------
# Regularization Names
# -----------------------------
__all__ = [
    'discrep', 'discrep_L2','discrep_L2_brain','GCV_NNLS', 'gcv',
    'Lcurve', 'l_curve',
    'LocReg', 'LocReg_unconstrainedB', 'LocReg_NEW_NNLS',
    'LocReg_unconstrained', 'LocReg_Ito_mod', 'LocReg_Ito_mod_deriv', 'LocReg_Ito_mod_deriv2',
    'LocReg_v2', 'LocReg_unconstrained_NE',
    'Multi_Reg_Gaussian_Sum1', 'tikhonov_vec', 'nonnegtik_hnorm',
    'lsqnonneg', 'generate_gaussian_regs_L2_old',
    'heatmap_unequal_width_All', 'Multi_Param_LR', 'tikhonov_multi_param', 'UPEN_Zama', 'UPEN_Zama0th', 'UPEN_Zama1st',

    # Subfunctions
    'l_cuve', 'csvd', 'l_corner', 'l_curve_corner',

    # Tools
    'Tikhonov', 'find_min_between_peaks', 'check_resolution'
]
