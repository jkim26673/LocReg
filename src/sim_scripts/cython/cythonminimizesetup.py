# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


# List all the .pyx files you want to include
# pyx_files = [
#     "/home/kimjosy/LocReg_Regularization-1/Simulations/locreg_ito.pyx"
#     # "/home/kimjosy/LocReg_Regularization-1/Simulations/funcs.pyx"  # Add your second .pyx file here
# ]

# setup(
#     ext_modules=cythonize(pyx_files),
#     include_dirs=[np.get_include()]
# )
extensions = [
    Extension(
        "cythonminimize",
        sources=["/home/kimjosy/LocReg_Regularization-1/Simulations/cythonminimize.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],  # Disable deprecated NumPy API
    )
]

setup(
    ext_modules=cythonize(extensions),
)