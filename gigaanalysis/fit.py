"""
GigaAnalysis - Fitting

"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as sp_curve_fit



class Fit_result():
    """
    This class is to hold the results of the fits on data objects.

    Parameters
    ----------
    func : function
        The function used in the fitting
    popt : numpy.ndarray
        The optimum values for the parameters
    pcov : numpy.ndarray
        The estimated covariance
    results : gigaanalysis.data.Data
        The optimal values obtained from the fit, will be
        none if full=False when performing the fit
    residuals : gigaanalysis.data.Data
        The residuals of the fit, will be none
        if full=False when performing the fit
    """
    def __init__(self, func, popt, pcov, results, residuals):
        """
        The __init__ method to produce the fit_result class  
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
        """
        This takes a value or an array of x_values and calculates the
        predicated y_vales

        Parameters
        ----------
        x_vals : numpy.ndarray
            An array of x_vales
        
        Returns
        -------
        y_vals : numpy.ndarray
            An array of predicted y_values
        """
        return Data(x_vals, self.func(x_vals, *self.popt))


def curve_fit(data_set, func, p0=None, full=True, **kwargs):
    """
    This is an implementation of :func:`scipy.optimize.curve_fit`
    for acting on :class:`gigaanalysis.data.Data` objects. This performs
    a least squares fit to the data of a function.

    Parameters
    ----------
    data_set : gigaanalysis.data.Data
        The data to perform the fit on
    func : function
        The model function to fit. It must take the x_values as
        the first argument and the parameters to fit as separate remaining
        arguments
    p0 : numpy.ndarray
        Initial guess for the parameters. Is passed to
        scipy.optimize.curve_fit included so it can be addressed positionally
    full : bool, optional
        If True fit_result will include residuals if False they will
        not be calculated and only results included
    kwargs:
        Keyword arguments are passed to :func:`scipy.optimize.curve_fit`
    
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


def any_poly(x_data, *p_vals):
    """
    This can be used to calculate things

    """
    # Pass data via numpy.array to prep for ga.Data
    x_data = np.array(x_data)
    x_data = np.reshape(x_data, (x_data.size))
    # Calculate values
    results = Data(x_data, x_data)*0
    for n, p in enumerate(p_vals[::-1]):
        results += p*np.power(x_data, n)
    return results

def poly_fit(data_set, order, full=True):
    popt, pcov = np.polyfit(data_set.x, data_set.y, order, cov=True)
    func = lambda x_data: any_poly(x_data, *popt)
    if full:
        results = Data(data_set.x, func(data_set.x, *popt))
        residuals = data_set - results
    else:
        results, residuals = None, None
    return Fit_result(func, popt, pcov, results, residuals)
