"""**ufunc Functions**

Here is the collection of simple functions that applies numpy ufuncs to the 
:class:`.Data` objects. These all work the same and are based around the 
function :func:`apply_func`. This functionality can be achieved with methods 
on the :class:`.Data` class but this was added to provide more readability.
"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def apply_func(data, func, act_x=False, as_Data=True):
    """Applies numpy ufuncs to GigaAnalysis Data objects.

    This applies the function to the relevant variable and returns the 
    object in the format specified.
    The methods :meth:`.Data.apply_x` and :meth:`.Data.apply_y` perform a 
    very similar function.

    Parameters
    ----------
    data : Data
        The Data object to act on with the function.
    func : numpy.ufunc
        The function to act with on the data.
    act_x : bool, optional
        If default of `False` act on the y values, otherwise if `True` act 
        on the x values.
    as_Data : bool, optional
        If the default of `True` returns a :class:`.Data` object, otherwise 
        if `False` returns a :class:`numpy.ndarray` of the values acted on 
        instead.

    Returns
    -------
    out_data : Data, numpy.ndarray
        The data with the chosen variable acted on, and returned either as 
        a Data object with the other variable unchanged or as a 
        :class:`numpy.ndarray` without the data from the other variable.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data was not a Data object but was {type(data)} instead.")
    if not isinstance(func, np.ufunc):
        raise TypeError(
            f"function provided was not a numpy ufunc but was {type(func)} "
            f"instead.")

    if as_Data and not act_x:
        return Data(data.x, func(data.y))
    elif as_Data and act_x:
        return Data(func(data.x), data.y)
    elif not as_Data and not act_x:
        return func(data.y)
    elif not as_Data and act_x:
        return func(data.x)


def sin(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.sin` using :func:`apply_func`.
    """
    return apply_func(data, np.sin, act_x=act_x, as_Data=as_Data)


def cos(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.cos` using :func:`apply_func`.
    """
    return apply_func(data, np.cos, act_x=act_x, as_Data=as_Data)


def tan(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.tan` using :func:`apply_func`.
    """
    return apply_func(data, np.tan, act_x=act_x, as_Data=as_Data)


def arcsin(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.arcsin` using :func:`apply_func`.
    """
    return apply_func(data, np.arcsin, act_x=act_x, as_Data=as_Data)


def arccos(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.arccos` using :func:`apply_func`.
    """
    return apply_func(data, np.arccos, act_x=act_x, as_Data=as_Data)


def arctan(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.arctan` using :func:`apply_func`.
    """
    return apply_func(data, np.arctan, act_x=act_x, as_Data=as_Data)


def sinh(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.sinh` using :func:`apply_func`.
    """
    return apply_func(data, np.sinh, act_x=act_x, as_Data=as_Data)


def cosh(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.cosh` using :func:`apply_func`.
    """
    return apply_func(data, np.cosh, act_x=act_x, as_Data=as_Data)


def tanh(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.tanh` using :func:`apply_func`.
    """
    return apply_func(data, np.tanh, act_x=act_x, as_Data=as_Data)


def arcsinh(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.arcsinh` using :func:`apply_func`.
    """
    return apply_func(data, np.arcsinh, act_x=act_x, as_Data=as_Data)


def arccosh(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.arccosh` using :func:`apply_func`.
    """
    return apply_func(data, np.arccosh, act_x=act_x, as_Data=as_Data)


def arctanh(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.arctanh` using :func:`apply_func`.
    """
    return apply_func(data, np.arctanh, act_x=act_x, as_Data=as_Data)


def log(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.log` using :func:`apply_func`.
    """
    return apply_func(data, np.log, act_x=act_x, as_Data=as_Data)


def log2(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.log2` using :func:`apply_func`.
    """
    return apply_func(data, np.log2, act_x=act_x, as_Data=as_Data)


def log10(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.log10` using :func:`apply_func`.
    """
    return apply_func(data, np.log10, act_x=act_x, as_Data=as_Data)


def exp(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.exp` using :func:`apply_func`.
    """
    return apply_func(data, np.exp, act_x=act_x, as_Data=as_Data)


def exp2(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.exp2` using :func:`apply_func`.
    """
    return apply_func(data, np.exp2, act_x=act_x, as_Data=as_Data)


def exp10(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.exp10` using :func:`apply_func`.
    """
    return apply_func(data, np.exp10, act_x=act_x, as_Data=as_Data)


def reciprocal(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.reciprocal` using :func:`apply_func`.
    """
    return apply_func(data, np.reciprocal, act_x=act_x, as_Data=as_Data)


def sqrt(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.sqrt` using :func:`apply_func`.
    """
    return apply_func(data, np.sqrt, act_x=act_x, as_Data=as_Data)


def square(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.square` using :func:`apply_func`.
    """
    return apply_func(data, np.square, act_x=act_x, as_Data=as_Data)

def abs(data, act_x=False, as_Data=True):
    """Applies :func:`numpy.abs` using :func:`apply_func`.
    """
    return apply_func(data, np.abs, act_x=act_x, as_Data=as_Data)

