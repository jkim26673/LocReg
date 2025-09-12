#!/usr/bin/env python3
# from setuptools import setup
# from Cython.Build import cythonize
# import numpy

# setup(
#     ext_modules=cythonize("/home/kimjosy/LocReg_Regularization-1/Simulations/cy_abs_sum.pyx"),
#     include_dirs=[numpy.get_include()],
# )

# setup(
#     ext_modules=cythonize("/home/kimjosy/LocReg_Regularization-1/Simulations/cy_abs_sum.pyx", compiler_directives={'language_level': "3"}),
#     include_dirs=[numpy.get_include()],
# )


from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

extensions = [
    Extension(
        "cython_demo",
        ["/home/kimjosy/LocReg_Regularization-1/Simulations/cythondemo.pyx"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
        include_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]  # Add this line
    )
]

setup(
    ext_modules=cythonize(extensions)
)