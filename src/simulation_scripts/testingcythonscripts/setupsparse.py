from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "Simulations.sparse_solver",
        sources=["Simulations/sparse_solver.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],  # Disable deprecated NumPy API
    )
]

setup(
    ext_modules=cythonize(extensions),
)

