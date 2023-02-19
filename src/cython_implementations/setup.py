import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(
        "l0_region_smoothing.pyx",
        annotate=True,
    ),
    include_dirs=[numpy.get_include()],
)
