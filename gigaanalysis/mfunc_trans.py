"""**Transformation Functions**

These are all functions that are applied to a :class:`.Data` class and 
return a similar object. These are either simple filters, transforms, or 
differentiation or intergeneration functions.
"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def invert_x(data):
    """Inverts the x values and re-interpolates them.

    This is useful for quantum oscillations because the periodicity in 
    inverse magnetic field.

    Parameters
    ----------
    data : Data
        The data to invert

    Returns
    -------
    inverted_data : Data
        A data object with points evenly separated in the new inverted x 
        values.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")
    elif not np.all(data.x[:-1] <= data.x[1:]):
        raise ValueError('Array to invert not sorted!')

    interp = interp1d(*data.both, bounds_error=False,
        fill_value=(data.y[0], data.y[1]))
    new_x = np.linspace(1./data.x.max(), 1./data.x.min(), len(data))
    return Data(new_x, interp(1/new_x))


def loess(data, x_window, polyorder):
    """This applies a LOESS filter to the data.

    The LOESS filter is applied using :func:`scipy.signal.savgol_filter`. 

    Parameters
    ----------
    data_set : Data
        The data to be smoothed. The points must be evenly spaced in x for 
        this to be applied.
    x_window : float
        The length of the window to apply in the same units as the x values.
    polyorder : int
        The order of the polynomial to apply.

    Returns
    -------
    smoothed_data : Data
        A data object after the data has been smoothed with LOESS filter.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")
    elif (np.max(data.x) - np.min(data.x)) < x_window:
        raise ValueError(
            "The LOESS window is longer than the given data range")

    spacing = data.x[1] - data.x[0]
    if not np.isclose(spacing, np.diff(data.x)).all():
        raise ValueError(
            "The data needs to be evenly spaced to smooth.")

    smooth_data = savgol_filter(data.y,
        2*int(round(0.5*x_window/spacing)) + 1, 
        polyorder=polyorder)
    return Data(data.x, smooth_data)


def poly_reg(data, polyorder):
    """This applied a polynomial fit to data and returns the fit result.

    This uses :func:`numpy.polyfit` and is used for subtracting polynomial 
    background from data sets.

    Parameters
    ----------
    data : Data
        The data to apply the fit too.
    polyorder : int
        The order of the polynomial to use in the fit.

    Returns
    -------
    poly_smoothed_data : Data
        The polynomial which was the best fit to the data.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")

    fit = np.polyfit(*data.both, deg=polyorder)
    y_vals = 0.
    for n, p in enumerate(fit):
        y_vals += p*np.power(data.x, polyorder-n)

    return Data(data.x, y_vals)