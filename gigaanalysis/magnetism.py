"""GigaAnalysis - Magnetism - :mod:`gigaanalysis.magnetism`
--------------------------------------------------------------

Here are a few functions for equations that are useful for magnetism 
science. They can be made to produce a Data object or just a 
:class:`numpy.ndarray`. This works well with the fitting module.
"""

from .data import *
from . import fit, const

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def brillouin_function(fields, n_ion, g, j, temp, as_Data=False):
    """The `Brillouin function
    <https://en.wikipedia.org/wiki/Brillouin_and_Langevin_functions>`_ 

    This function which describes the magnetisation of an ideal paramagnet 
    composed of ions with a certain spin J.

    Parameters
    ----------
    fields : float or numpy.ndarray
        The value or values of the applied magnetic filed in Tesla.
    n_ion : float
        The number of contributing ions to the magnetism.
    g : float
        The ions g factor or dimensionless magnetic moment.
    i : float
        Is a positive integer or half integer which is the spin of the ions. 
        This function does not but constrains on the value of j. 
    temp : float
        Temperature in Kelvin.
    as_Data : bool, optional
        If False returns a :class:`numpy.ndarray` which is the default
        behaviour. If True returns a :class:`gigaanalysis.data.Data` object
        with the fields values given and the cosponsoring magnetisation.

    Returns
    -------
    Magnetisation : numpy.ndarray, Data
        The magnetisation produced in units of J/T.
    """
    x = np.array(g*const.muB()*fields/temp/const.kb())
    where0 = x==0
    x[where0] = np.nan
    bj = (2*j+1)/2/j/np.tanh((2*j+1)/2/j*x) - 1/2/j/np.tanh(x/2)
    bj[where0] = 0
    if as_Data:
        return Data(fields, n_ion*g*const.muB()*j*bj)
    else:
        return n_ion*g*const.muB()*j*bj


def langevin_function(fields, n_ion, g, temp, as_Data=False):
    """The `Langevin function
    <https://en.wikipedia.org/wiki/Brillouin_and_Langevin_functions>`_

    This is the classical limit of the Brillouin function which describes the 
    magnetisation of an ideal paramagnet.
    
    Parameters
    ----------
    fields : float or numpy.ndarray
        The value or values of the applied magnetic filed in Tesla
    n_ion : float
        The number of contributing ions to the magnetism
    g : float
        The ions g factor or dimensionless magnetic moment
    temp : float
        Temperature in Kelvin
    as_Data : bool, optional
        If False returns a :class:`numpy.ndarray` which is the default
        behaviour. If True returns a :class:`gigaanalysis.data.Data` object
        with the fields values given and the cosponsoring magnetisation.

    Returns
    -------
    Magnetisation : numpy.ndarray, Data
        The magnetisation produced in units of J/T
    """
    x = np.array(g*const.muB()*fields/temp/const.kb())
    where0 = x==0
    x[where0] = np.nan
    mag = n_ion*g*const.muB()*(1/np.tanh(x) - 1/(x))
    mag[where0] = 0
    if as_Data:
        return Data(fields, mag)
    else:
        return mag

