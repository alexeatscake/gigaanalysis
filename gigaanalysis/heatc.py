"""GigaAnalysis - Heat Capacity - :mod:`gigaanalysis.heatc`
--------------------------------------------------------------

Here are a few functions for equations that are useful for heat capacity 
measurements. They can be made to produce a Data object or just a 
:class:`numpy.ndarray`. This works well with the fitting module.
"""

from .data import *
from . import fit, const

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def schottky_anomaly(temps, num, gap, as_Data=False):
    """The `Schottky anomaly
    <https://en.wikipedia.org/wiki/Schottky_anomaly>`_

    The function which describes the heat capacity of a two state system. 

    Parameters
    ----------
    temps : float or numpy.ndarray
        The value or values of the temperature in Kelvin.
    num : float
        The number of moles of states contributing.
    gap : float
        The energy gap between the two states in Joules.
    as_Data : bool, optional
        If False returns a :class:`numpy.ndarray` which is the default
        behaviour. If True returns a :class:`gigaanalysis.data.Data` object
        with the fields values given and the cosponsoring magnetisation.

    Returns
    -------
    Heat Capacity : numpy.ndarray, Data
        The heat capacity in units of J/K/mol.
    """
    if not isinstance(temps, np.ndarray):
        temps = np.array(temps)
    where0 = temps==0  # where0 is to deal with division by zero errors
    temps[where0] = np.nan
    u = gap/temps/const.kb()
    cp = num*const.R()*u*u*np.exp(u)*np.power(1 + np.exp(u), -2)
    cp[where0] = 0
    if as_Data:
        return Data(temps, cp)
    else:
        return cp

