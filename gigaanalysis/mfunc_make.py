"""**Make Functions**

Here are a few functions that are for producing mathematical functions of a 
certain form. These are also used by the plotting functions in the module 
:mod:`.fit`. Other functions more specific to the certain areas of physics 
are included in the relevant modules.
"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_poly(x_data, *p_vals, as_Data=True):
    """Generates a polynomial from the coefficients. 

    The point of this function is to generate the values expected from a
    linear fit. It is designed to take the values obtained from
    :func:`numpy.polyfit`.
    For a set of p_vals of length n+1 ``y_data = p_vals[0]*x_data**n + 
    p_vals[0]*x_data**(n-1) + ... + p_vals[n]``

    Parameters
    ----------
    x_data :  numpy.ndarray
        The values to compute the y values of.
    p_vals : float
        These are a series of floats that are the coefficients of the
        polynomial starting with with the highest power.
    as_Data : bool, optional
        If the default of `True` returns a :class:`.Data` object with the x 
        values given and the cosponsoring y values. If False returns a 
        class:`numpy.ndarray`.

    Returns
    -------
    results : Data or numpy.ndarray
        The values expected from a polynomial with the 
        specified coefficients.
    """
    results = x_data*0
    for n, p in enumerate(p_vals[::-1]):
        results += p*np.power(x_data, n)
    if as_Data:
        return Data(x_data, results)
    else:
        return results


def make_sin(x_data, amp, wl, phase, offset=0, as_Data=True):
    """This function generates sinusoidal functions.

    The form of the equation is
    ``amp*np.sin(x_data*np.pi*2./wl + phase*np.pi/180.) + offset``
    The offset is a keyword argument that doesn't need to be applied.

    Parameters
    ----------
    x_data :  numpy.ndarray
        The values to compute the y values of.
    amp : float
        Amplitude of the sin wave.
    wl : float
        Wavelength of the sin wave units the same as `x_data`.
    phase : float
        Phase shift of the sin wave in degrees.
    offset : float, optional
        Shift all of the y values by a certain amount, default is for no 
        offset.
    as_Data : bool, optional
        If the default of `True` returns a :class:`.Data` object with the x 
        values given and the cosponsoring y values. If False returns a 
        class:`numpy.ndarray`.

    Returns
    -------
    results : Data or numpy.ndarray
        The values expected from the sinusoidal with the given parameters
    """
    results = amp*np.sin(x_data*np.pi*2./wl + phase*np.pi/180.) + offset
    if as_Data:
        return Data(x_data, results)
    else:
        return results


def make_gaussian(x_data, amp, mean, std, offset=0, as_Data=True):
    """This function generates Gaussian functions

    The form of the equation is
    ``amp*np.exp(-0.5*np.power((x_data - mean)/std, 2)) + offset``
    The offset is a keyword argument that doesn't need to be applied. The 
    amplitude refers the the maximum value at the top of the peak.

    Parameters
    ----------
    x_data :  numpy.ndarray
        The values to compute the y values of.
    amp : float
        The maxiumal value of the Gaussian function.
    mean : float
        The centre of the Gaussian function.
    std : float
        The width of the Gaussian given as the standard deviation in the 
        same units as the `x_data`.
    offset : float, optional
        Shift all of the y values by a certain amount, default is for no 
        offset.
    as_Data : bool, optional
        If the default of `True` returns a :class:`.Data` object with the x 
        values given and the cosponsoring y values. If False returns a 
        class:`numpy.ndarray`.

    Returns
    -------
    results : Data or numpy.ndarray
        The values expected from the Gaussian with the given parameters
    """
    results = amp*np.exp(-0.5*np.power((x_data - mean)/std, 2)) + offset
    if as_Data:
        return Data(x_data, results)
    else:
        return results

