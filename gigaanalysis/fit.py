"""GigaAnalysis - Fitting

"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as sp_curve_fit



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
    results : gigaanalysis.data.Data
        The optimal values obtained from the fit, will be
        none if `full`=`False` when performing the fit.
    residuals : gigaanalysis.data.Data
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
        self.results = results
        self.residuals = residuals

    def __str__(self):
        return np.array2string(self.popt)

    def __repr__(self):
        return 'GA fit results:{}'.format(self.popt)

    def _repr_html_(self):
        return 'GA fit results:{}'.format(self.popt)

    def __dir__(self):
        return ['func', 'popt', 'pcov', 'results', 'residuals', 'predict']

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
        y_vals : gigaanalysis.data.Data
            An Data object with the predicted y_values.

        """
        return Data(x_vals, self.func(x_vals, *self.popt))


def curve_fit(data_set, func, p0=None, full=True, **kwargs):
    """This is an implementation of :func:`scipy.optimize.curve_fit`
    for acting on :class:`gigaanalysis.data.Data` objects. This performs
    a least squares fit to the data of a function.

    Parameters
    ----------
    data_set : gigaanalysis.data.Data
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
    fit_result : gigaanalysis.fit.Fit_result
        A gigaanalysis Fit_result object containing the results

    """
    popt, pcov = sp_curve_fit(func, data_set.x, data_set.y, p0=p0, **kwargs)
    if full:
        results = Data(data_set.x, func(data_set.x, *popt))
        residuals = data_set - results
    else:
        results, residuals = None, None
    return Fit_result(func, popt, pcov, results, residuals)


def any_poly(x_data, *p_vals, as_Data=False):
    """The point of this function is to generate the values expected from a
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
        If False returns a :class:`numpy.ndarray` which is the default
        behaviour. If True returns a :class:`gigaanalysis.data.Data` object
        with the x values given and the cosponsoring y values.

    Returns
    -------
    results : numpy.ndarray or gigaanalysis.data.Data
        The values expected from a polynomial with the 
        specified coefficients.

    """
    results = x_data*0
    for n, p in enumerate(p_vals[::-1]):
        results += p*np.power(x_data, n)
    if as_Data:
        return ga.Data(x_data, results)
    else:
        return results

def poly_fit(data_set, order, full=True):
    """This function fits a polynomial of a certain order to a given
    data set. It uses :func:`numpy.polyfit` for the fitting. The function
    which is to produce the data is :func:`gigaanalysis.fit.any_poly`.

    Parameters
    ----------
    data_set : gigaanalysis.data.Data
        The data set to perform the fit on.
    order : int
        The order of the polynomial.
    full : bool, optional
        If True fit_result will include residuals if False they will
        not be calculated and only results included.
    
    Returns
    -------
    fit_result : gigaanalysis.fit.Fit_result
        A gigaanalysis Fit_result object containing the results the
        fit parameters are the coefficients of the polynomial. Follows the
        form of :func:`gigaanalysis.fit.any_poly`.
    
    """
    popt, pcov = np.polyfit(data_set.x, data_set.y, order, cov=True)
    func = any_poly
    if full:
        results = Data(data_set.x, func(data_set.x, *popt))
        residuals = data_set - results
    else:
        results, residuals = None, None
    return Fit_result(func, popt, pcov, results, residuals)


def make_sin(x_data, amp, wl, phase, offset, as_Data=False):
    """This function generates sinusoidal signals
    The form of the equation is
    ``amp*np.sin(x_data*np.pi*2./wl + phase*np.pi/180.) + offset``

    Parameters
    ----------
    x_data :  numpy.ndarray
        The values to compute the y values of.
    amp : float
        Amplitude of the sin wave.
    wl : float
        Wavelength of the sin wave units the same as `x_data`.
    phase : float
        Phase shift of the sin wave in deg
    offset : float
        Shift of the y values
    as_Data : bool, optional
        If False returns a :class:`numpy.ndarray` which is the default
        behaviour. If True returns a :class:`gigaanalysis.data.Data` object
        with the x values given and the cosponsoring y values.

    Returns
    -------
    results : numpy.ndarray or gigaanalysis.data.Data
        The values expected from the sinusoidal with the given parameters
    
    """
    results = amp*np.sin(x_data*np.pi*2./wl + phase*np.pi/180.) + offset
    if as_Data:
        return ga.Data(x_data, results)
    else:
        return results


def sin_fit(data_set, full=True):
    """This function fits a polynomial of a certain order to a given
    data set. It uses :func:`numpy.polyfit` for the fitting. The function
    which is to produce the data is :func:`gigaanalysis.fit.any_poly`.

    Parameters
    ----------
    data_set : gigaanalysis.data.Data
        The data set to perform the fit on.
    full : bool, optional
        If True fit_result will include residuals if False they will
        not be calculated and only results included.
    
    Returns
    -------
    fit_result : gigaanalysis.fit.Fit_result
        A gigaanalysis Fit_result object containing the results the
        fit parameters are the coefficients of the polynomial. Follows the
        form of :func:`gigaanalysis.fit.any_poly`.
    
    """
    popt, pcov = np.polyfit(data_set.x, data_set.y, order, cov=True)
    func = any_poly
    if full:
        results = Data(data_set.x, func(data_set.x, *popt))
        residuals = data_set - results
    else:
        results, residuals = None, None
    return Fit_result(func, popt, pcov, results, residuals)
