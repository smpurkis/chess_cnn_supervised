import numpy
import pyximport

pyximport.install(setup_args={"include_dirs": numpy.get_include()}, reload_support=True)
from data_utils_cy import *
