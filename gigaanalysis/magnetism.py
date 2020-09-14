"""Giga Analysis - Magnetism

"""

from .data import *
from . import fit, const

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def brillouin_function(fields, n_ion, g, j, temp):
    """The `Brillouin function
    <https://en.wikipedia.org/wiki/Brillouin_and_Langevin_functions>`_ is 
    the function which describes the magnetisation of an ideal paramagnet 
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

    Returns
    -------
    Magnetisation : float
        The magnetisation produced in units of J/T.
    """
    x = g*const.muB()*fields/temp/const.kb()
    bj = (2*j+1)/2/j/np.tanh((2*j+1)/2/j*x) - 1/2/j/np.tanh(x/2)
    return n_ion*g*const.muB()*j*bj

def langevin_function(fields, n_ion, g, temp):
    """The `Langevin function
    <https://en.wikipedia.org/wiki/Brillouin_and_Langevin_functions>`_ is 
    the classical limit of the Brillouin function which describes the 
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

    Returns
    -------
    Magnetisation : float
        The magnetisation produced in units of J/T
    """
    x = g*const.muB()*fields/temp/const.kb()
    return n_ion*g*const.muB()*(1/np.tanh(x) - 1/(x))


