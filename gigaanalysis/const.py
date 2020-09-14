"""Giga Analysis - Constants

Here is contained a collection of functions with when called return values 
of physical constants. They always return floats and all have one optional 
parameter 'unit' which default is 'SI' for the International System of Units 
values for these parameters.
"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kb(unit='SI'):
    """`Boltzmann constant 
    <http://en.wikipedia.org/wiki/Boltzmann_constant>`_

    Parameters
    ----------
    unit : {'SI', 'eV', 'CSG'}, optional
        If default 'SI' units are J/K. 
        If 'eV' units are in eV/K. 
        If 'CSG' units are in erg/K. 

    Returns
    -------
    Value of the Boltzmann constant : float
    """
    if unit == 'SI':
        return 1.380649e-23
    elif unit == 'eV':
        return 8.617333262145e-5
    elif unit == 'CSG':
        return 1.380649e-16
    else:
        raise ValueError("unit must be 'SI', 'eV', or 'CSG'.")

def qe():
    """Elementary Charge in C"""
    return 1.602176634e-19

def c():
    """Speed of light in m/s"""
    return 2.99792458e8

def Na():
    """Avogadro constant in mol−1"""
    return 6.02214076e23

def muB():
    """Bohr Magnetron in J/T"""
    return 9.274009994e-24


