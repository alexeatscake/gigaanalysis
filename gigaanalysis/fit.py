"""GigaAnalysis - Fitting - :mod:`gigaanalysis.fit`
------------------------------------------------------

This module contains the :class:`Fit_result` class which is used for fitting 
functions to the GigaAnalysis Data objects. It also contains some common 
expressions that are needed as well as functions that fit a Data class with 
them.
"""

from .data import *
from . import mfunc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit as sp_curve_fit  # For curve_fit and
# curve_fit_y


class Fit_result():
    """This class is to hold the results of the fits on data objects.

    Parameters
    ----------
    func : function
        The function used in the fitting.
    popt : numpy.ndarray
        The optimum values for the parameters.
    pcov : numpy.ndarray
        The estimated covariance.
    results : Data
        The optimal values obtained from the fit, will be
        none if `full`=`False` when performing the fit.
    residuals : Data
        The residuals of the fit, will be none
        if `full`=`False` when performing the fit.

    Attributes
    ----------
    func : function
        The function used in the fitting.
    popt : numpy.ndarray
        The optimum values for the parameters.
    pcov : numpy.ndarray
        The estimated covariance.
    pstd : numpy.ndarray
        The estimated standard deviation
    results : gigaanalysis.data.Data
        The optimal values obtained from the fit, will be
        none if `full`=`False` when performing the fit.
    residuals : gigaanalysis.data.Data
        The residuals of the fit, will be none
        if `full`=`False` when performing the fit.
    """
    def __init__(self, func, popt, pcov, results, residuals):
        """The __init__ method to produce the fit_result class  
        The point of this class is to store the results from a gigaanalysis
        fit so the arguments are the same as the attributes.
        """
        self.func = func
        self.popt = popt
        self.pcov = pcov
        self.pstd = np.sqrt(np.diag(pcov))
        self.results = results
        self.residuals = residuals

    def __str__(self):
        return np.array2string(self.popt)

    def __repr__(self):
        return 'GA fit results:{}'.format(self.popt)

    def _repr_html_(self):
        return 'GA fit results:{}'.format(self.popt)

    def __dir__(self):
        return ['func', 'popt', 'pcov', 'pstd', 'results', 'residuals', 
        'predict', 'sample_parameters']

    def __len__(self):
        return self.popt.size

    def predict(self, x_vals):
        """This takes a value or an array of x_values and calculates the
        predicated y_vales.

        Parameters
        ----------
        x_vals : numpy.ndarray
            An array of x_vales.
        
        Returns
        -------
        y_vals : Data
            An Data object with the predicted y_values.
        """
        if isinstance(x_vals, (float, int, np.float_, np.int_)):
            x_vals = np.array([x_vals])

        return Data(x_vals, self.func(x_vals, *self.popt))

    def sample_parameters(self, size, **kwargs):
        """This samples values of the parameters from a multivariate normal 
        distribution using the fitted values and variance.

        This uses the function :func:`numpy.random.multivariate_normal` and 
        keyword arguments are passed to it.

        Parameters
        ----------
        size : int
            The number of samples to return. (More complicated behaviour is 
            possible, see: :func:`numpy.random.multivariate_normal`)

        Returns
        -------
        samples : numpy.ndarray
            A numpy array in the shape `(s, n)` where `s` is the number of 
            samples and `n` is the number of parameters.

        """

        return np.random.multivariate_normal(self.popt, self.pcov, size,
            **kwargs)


def curve_fit(data, func, p0, full=True, **kwargs):
    """Fit curves to Data objects with functions that produce Data objects.

    This is an implementation of :func:`scipy.optimize.curve_fit`
    for acting on :class:`.Data` objects. This performs
    a least squares fit to the data of a function.

    Parameters
    ----------
    data : Data
        The data to perform the fit on.
    func : function
        The model function to fit. It must take the x values as
        the first argument and the parameters to fit as separate remaining
        arguments. It must also return a :class:`.Data` object.
    p0 : numpy.ndarray
        Initial guess for the parameters. Is passed to
        :func:`scipy.optimize.curve_fit` included so it can be addressed
        positionally.
    full : bool, optional
        If `True`, `fit_result` will include residuals, and if `False`
        they will not be calculated and only results included.
    kwargs :
        Keyword arguments are passed to :func:`scipy.optimize.curve_fit`.
    
    Returns
    -------
    fit_result : Fit_result
        A GigaAnalysis Fit_result object containing the results.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")

    try:
        func_out = func(data.x, *p0)
        assert isinstance(func_out, Data)
    except:
        raise TypeError(
            f"The func parameter needs to take a function that receives "
            f"an array of x values and the arguments in p0 and returns "
            f"a Data object. This was not what was given.")

    def func_y(x_data, *p0):
        return func(x_data, *p0).y

    popt, pcov = sp_curve_fit(func_y, data.x, data.y, p0=p0, **kwargs)
    if full:
        results = func(data.x, *popt)
        residuals = data - results
    else:
        results, residuals = None, None

    return Fit_result(func_y, popt, pcov, results, residuals)


def curve_fit_y(data, func, p0=None, full=True, **kwargs):
    """Fit curves to Data objects with functions that produce y values.

    This is an implementation of :func:`scipy.optimize.curve_fit`
    for acting on :class:`.Data` objects. This performs
    a least squares fit to the data of a function.

    Parameters
    ----------
    data : Data
        The data to perform the fit on.
    func : function
        The model function to fit. It must take the x values as
        the first argument and the parameters to fit as separate remaining
        arguments.
    p0 : numpy.ndarray, optional
        Initial guess for the parameters. Is passed to
        :func:`scipy.optimize.curve_fit` included so it can be addressed
        positionally. If `None` unity will be used for every parameter.
    full : bool, optional
        If `True`, `fit_result` will include residuals, and if `False`
        they will not be calculated and only results included.
    kwargs:
        Keyword arguments are passed to :func:`scipy.optimize.curve_fit`.
    
    Returns
    -------
    fit_result : Fit_result
        A GigaAnalysis Fit_result object containing the results.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")

    popt, pcov = sp_curve_fit(func, data.x, data.y, p0=p0, **kwargs)
    if full:
        results = Data(data.x, func(data.x, *popt))
        residuals = data - results
    else:
        results, residuals = None, None

    return Fit_result(func, popt, pcov, results, residuals)


def poly_fit(data, order, full=True):
    """Fit a polynomial of a certain order to a given data set.

    It uses :func:`numpy.polyfit` for the fitting. The function
    which is to produce the data is :func:`.mfunc.make_poly`.

    Parameters
    ----------
    data : Data
        The data set to perform the fit on.
    order : int
        The order of the polynomial.
    full : bool, optional
        If True fit_result will include residuals if False they will
        not be calculated and only results included.
    
    Returns
    -------
    fit_result : Fit_result
        A GigaAnalysis Fit_result object containing the results where the
        fit parameters are the coefficients of the polynomial. Follows the
        form of :func:`gigaanalysis.fit.any_poly`.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")

    if len(data) > order + 1:
        popt, pcov = np.polyfit(data.x, data.y, order, cov=True)
    elif len(data) == order + 1:
        popt = np.polyfit(data.x, data.y, order, cov=False)
        pcov = np.full((order+1, order+1), np.nan)
    else:
        raise ValueError(
            f"The order of the polynomial needs to be more than 2 larger "
            f"than the number of data points. There are {len(data)} points "
            f"and is fitting a polynomial of order {order}.")

    func = lambda *args: mfunc.make_poly(*args, as_Data=False)
    if full:
        results = mfunc.make_poly(data.x, *popt)
        residuals = data - results
    else:
        results, residuals = None, None
    return Fit_result(func, popt, pcov, results, residuals)


def sin_fit(data, p0=None, offset=False, full=True, **kwargs):
    """Fits a sinusoid to a given Data object.

    This uses :func:`mfunc.make_sin` and has the option to either include an 
    offset from zero or not to. It then uses :func:`curve_fit_y` that 
    makes use of :func:`scipy.optermize.curve_fit` which `kwargs` are passed 
    to.

    Parameters
    ----------
    data : Data
        The Data to fit the sinusoid to.
    p0 : numpy.ndarray, optional
        The initial values to begin the optimisation from. These are in 
        order, the amplitude of the sin, the wavelength, the phase, and if 
        included the offset from zero.
    offset : bool, optional
        If `True` an offset from zero is also included in the fit. The 
        default is `False`.
    full : bool, optional
        If `True`, which is the default, the results and residuals are 
        included in the returned :class:`Fit_result`.
    kwargs :
        The keyword arguments are passed to 
        :func:`scipy.optermize.curve_fit`.

    Returns
    -------
    fit_result, Fit_result
        A GigaAnalysis Fit_result object containing the results where the
        fit parameters are same as specified in :func:`.mfunc.make_sin`.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")

    if p0 is None:
        pass
    elif len(p0) != 3 and not offset:
        raise ValueError(
            f"There are 3 variables if offset is False and {len(p0)} were "
            f"given in p0.")
    elif len(p0) != 4 and offset:
        raise ValueError(
            f"There are 4 variables if offset is True and {len(p0)} were "
            f"given in p0.")

    if offset:
        func = lambda x_data, amp, wl, phase, offset: mfunc.make_sin(
            x_data, amp, wl, phase, offset, as_Data=False)
    else:
        func = lambda x_data, amp, wl, phase: mfunc.make_sin(
            x_data, amp, wl, phase, as_Data=False)

    return curve_fit_y(data, func, p0=p0, full=full, **kwargs)


def gaussian_fit(data, p0=None, offset=False, full=True, **kwargs):
    """Fits a Gaussian to a given Data object.

    This uses :func:`mfunc.make_gaussian` and has the option to either 
    include an offset from zero or not to. It then uses 
    :func:`curve_fit_y` that makes use of :func:`scipy.optermize.curve_fit` 
    which `kwargs` are passed to.

    Parameters
    ----------
    data : Data
        The Data to fit the sinusoid to.
    p0 : numpy.ndarray, optional
        The initial values to begin the optimisation from. These are in 
        order, the amplitude of the Gaussian, the central point, the 
        standard deviation, and if included the offset from zero.
    offset : bool, optional
        If `True` an offset from zero is also included in the fit. The 
        default is `False`.
    full : bool, optional
        If `True`, which is the default, the results and residuals are 
        included in the returned :class:`Fit_result`.
    kwargs :
        The keyword arguments are passed to 
        :func:`scipy.optermize.curve_fit`.

    Returns
    -------
    fit_result, Fit_result
        A GigaAnalysis Fit_result object containing the results where the
        fit parameters are same as specified in :func:`.mfunc.make_gaussian`.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")

    if p0 is None:
        pass
    elif len(p0) != 3 and not offset:
        raise ValueError(
            f"There are 3 variables if offset is False and {len(p0)} were "
            f"given in p0.")
    elif len(p0) != 4 and offset:
        raise ValueError(
            f"There are 4 variables if offset is True and {len(p0)} were "
            f"given in p0.")

    if offset:
        func = lambda x_data, amp, mean, std, offset: mfunc.make_gaussian(
            x_data, amp, mean, std, offset, as_Data=False)
    else:
        func = lambda x_data, amp, mean, std: mfunc.make_gaussian(
            x_data, amp, mean, std, as_Data=False)

    return curve_fit_y(data, func, p0=p0, full=full, **kwargs)
