from distutils.core import setup
from Cython.Build import cythonize
import numpy
from time import time
import matplotlib.pyplot as plt
setup(ext_modules=cythonize("cythonfn.pyx", compiler_directives={"language_level":"3"}), include_dirs=[numpy.get_include()])

# python3 setup.py build_ext --inplace