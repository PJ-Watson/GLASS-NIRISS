"""
Grism-related codes.
"""

# from niriss_tools.grism.fitting_tools import *
# from niriss_tools.grism.multiregion import *
# from niriss_tools.grism.specgen import *
# from niriss_tools.grism.utils import *

import os

import numpy as np

# Allow for environment variable override if necessary
float_dtype = np.dtype(os.getenv("MULTIREGION_FLOAT_DTYPE", "f8")).type
