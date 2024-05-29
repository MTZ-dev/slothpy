from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="compute_diagonal",
        sources=["compute_diagonal.pyx"],
        include_dirs=[np.get_include()],
        libraries=["blas"],
        extra_compile_args=["-fopenmp", "-O3", "-march=native", "-ffast-math", "-funroll-loops", "-flto"],
        extra_link_args=["-fopenmp", "-flto"],
    )
]

setup(
    name="slothpy",
    version="0.2.0",
    packages=find_packages(include=["slothpy", "slothpy.*"]),
    ext_modules=cythonize(extensions, compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
    }
        ),
    zip_safe=False,
)
