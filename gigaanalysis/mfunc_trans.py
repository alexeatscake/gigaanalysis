"""**Transformation Functions**

These are all functions that are applied to a :class:`.Data` class and 
return a similar object. These are either simple filters, transforms, or 
differentiation or intergeneration functions.
"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d  # For invert_x
from scipy.signal import savgol_filter  # For loess and deriv
from scipy.integrate import cumulative_trapezoid  # For integrate


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


def loess(data, x_window, polyorder, **kwargs):
    """This applies a LOESS filter to the data.

    The LOESS filter is applied using :func:`scipy.signal.savgol_filter`. 
    This fits a polynomial to many sections of the data and uses the central 
    value as the value of the transformed data. Keyword arguments can be 
    given and will be passed to :func:`scipy.signal.savgol_filter`.

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
        polyorder=polyorder, **kwargs)
    return Data(data.x, smooth_data)


def poly_reg(data, polyorder, sigma=None):
    """This applied a polynomial fit to data and returns the fit curve.

    This uses :func:`numpy.polyfit` and filters the data down to a simple 
    polynomial. This can be used for subtracting polynomial background from 
    data sets.

    Parameters
    ----------
    data : Data
        The data to apply the fit too.
    polyorder : int
        The order of the polynomial to use in the fit.
    sigma : numpy.ndarray, optional
        The standard divation of the points of the data. Needs to be a 
        :class:`numpy.ndarray` the same length as the data. Default is None 
        where every point is evenly weighted.

    Returns
    -------
    poly_smoothed_data : Data
        The polynomial which was the best fit to the data with the original 
        x values and the generated values.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")
    
    if sigma is not None:
        weight = 1/np.array(sigma)
        if weight.size != data.x.size:
            raise ValueError(
                f"The sigma values need to be an array the same size as the "
                f"data object. Was length {weight.size} where the data was "
                f"of length {data.x.size}.")
    else:
        weight = None

    fit = np.polyfit(*data.both, deg=polyorder, w=weight)
    y_vals = 0.
    for n, p in enumerate(fit):
        y_vals += p*np.power(data.x, polyorder-n)

    return Data(data.x, y_vals)


def deriv(data, x_window, deriv_order, polyorder=None, **kwargs):
    """This applies a LOESS filter to the data.

    The LOESS filter is applied using :func:`scipy.signal.savgol_filter`. 
    This fits a polynomial to many sections of the data and uses the central 
    value as the value of the transformed data. Keyword arguments can be 
    given and will be passed to :func:`scipy.signal.savgol_filter`.

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
            "The x window is longer than the given data range")

    spacing = data.x[1] - data.x[0]
    if not np.isclose(spacing, np.diff(data.x)).all():
        raise ValueError(
            "The data needs to be evenly spaced to smooth.")

    if not isinstance(deriv_order, (int, np.int_)) or deriv_order < 0:
        raise TypeError(
            f"deriv_order must be an non negative integer but was of type "
            f"{type(deriv_order)}.")
    
    if polyorder is not None:
        if not isinstance(polyorder, (int, np.int_)):
            raise TypeError(
                f"polyorder needs to be an integer but was of type "
                f"{type(polyorder)}")
        elif deriv_order > polyorder:
            raise ValueError(
                f"polyorder if given needs to be bigger than the "
                f"deriv_order value. deriv_order was {deriv_order} and "
                f"polyorder was {polyorder}")
    else:
        polyorder = deriv_order

    smooth_data = savgol_filter(data.y, 
        2*int(round(0.5*x_window/spacing)) + 1, polyorder=polyorder, 
        deriv=deriv_order, delta=spacing, **kwargs)

    return Data(data.x, smooth_data)


def integrate(data):
    """Integrate a :class:`.Data` object cumulatively.

    This uses :func:`scipy.integrate.cumulative_trapezoid` to preform a 
    cumulative integration over the data set. This will sort the data set in 
    the process.

    Parameters
    ----------
    data : Data
        The :class:`.Data` object to preform the integration on.

    Returns
    -------
    integral : Data
        The integral the of the data and will be the same size.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")

    d_sort = data.sort()
    y_int = cumulative_trapezoid(d_sort.y, d_sort.x, initial=0.)

    return Data(d_sort.x, y_int)

