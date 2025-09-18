# -----------------------------
# Base loader (safe imports only)
# -----------------------------
from src.utils.load_imports.loading import *

# -----------------------------
# Hansen Classical Test Problems
# -----------------------------
from src.regularization.classical.hansen.baart import baart
from src.regularization.classical.hansen.blur import blur
from src.regularization.classical.hansen.deriv2 import deriv2
from src.regularization.classical.hansen.foxgood import foxgood
from src.regularization.classical.hansen.gravity import gravity
from src.regularization.classical.hansen.heat import heat
from src.regularization.classical.hansen.i_laplace import i_laplace
from src.regularization.classical.hansen.phillips import phillips
from src.regularization.classical.hansen.shaw import shaw
from src.regularization.classical.hansen.wing import wing

# -----------------------------
#  Classical Names
# -----------------------------
__all__ = [
    'baart', 'blur', 'deriv2', 'foxgood',
    'gravity', 'heat', 'i_laplace',
    'phillips', 'shaw', 'wing'
]
