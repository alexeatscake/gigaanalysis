"""GigaAnalysis - Mathematics Functions - :mod:`gigaanalysis.mfunc`
-------------------------------------------------------------------

This module contains a large collections of simple functions that are for 
performing basic maths using :class:`.Data` objects. The functions are 
currently in three types. There are the basic functions that act on the 
dependent variables and return a new :class:`.Data` object. There is the 
make functions that produce data in a certain form given parameters and 
the x values. There are also a few more functions to do with FFTs, 
differentiation and integration.
"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .mfunc_ufunc import *
from .mfunc_make import *
from .mfunc_fft import *
from .mfunc_trans import *

