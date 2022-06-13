"""**FFT Functions**

This module contains a large collections of simple functions that are for 
performing basic maths using :class:`.Data` objects. These are for 
performing a Fast Fourier Transform (FFT) mostly using the function 
:func:`fft`. There are also some functions for identifying the FFT peaks.
"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import (get_window,  # For ftt
    find_peaks)  # For get_peaks


def fft(data, n=65536, window='hanning', freq_cut=0.):
    """Performs an Fast Fourier Transform on the given data.

    This assumes that the data is real, and the data provided needs to be 
    evenly spaced. Makes use of :func:`numpy.fft.rfft`. This takes into 
    account the x values to provide the frequencies in the correct units.

    Parameters
    ----------
    data : Data
        The data to be FFT. This must be evenly spaced in x.
    n : int, optional
        The number of points to make the FFT extra points will be zero 
        padded. The default is ``2**16 = 65536``. If the data is longer 
        than the value of n, n is rounded up to the next power of 2.
    window : str, optional
        The type of windowing function to use taken from 
        :func:`scipy.signal.get_window`. The default is 'hanning'.
    freq_cut : float, optional
        The frequency to drop all the higher from. The default is 0 which 
        means that all the frequencies are kept.

    Returns
    -------
    fft_result : Data
        A data object with the FFT frequencies in the x values and the 
        amplitudes in the y values.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")

    spacing = data.x[1] - data.x[0]
    if not np.isclose(spacing, np.diff(data.x)).all():
        raise ValueError(
            "The data needs to be evenly spaced to FFT.")


    window_func = get_window(window, len(data))

    if n < len(data):
        n = int(2**np.ceil(np.log2(len(data))))

    data /= np.average(window_func)*len(data)/2  # Normalise the amplitude

    fft_vals = np.abs(
        np.fft.rfft(data.y*window_func, n=n))
    fft_freqs = np.fft.rfftfreq(n, d=spacing)
    freq_arg = None
    if freq_cut > 0:
        freq_arg = np.searchsorted(fft_freqs, freq_cut)

    return Data(fft_freqs[0:freq_arg], fft_vals[0:freq_arg])


def get_peaks(data, n_peaks=4, as_Data=False, **kwargs):
    """This returns the peaks in a data object as a numpy array.

    Using :func:`scipy.signal.find_peaks` the peaks in the data object are 
    found. ``**kwargs`` are passed to that function. This is most commonly 
    used when examining FFT data.

    Parameters
    ----------
    data : Data
        The data to look for peaks in.
    n_peaks : int, optional
        The number of peaks to return, the default is 4.
    as_Data : bool, optional
        If `True` the peak info is returned as a :class:`.Data` object. The 
        default is `False`. 

    Returns
    -------
    peak_info : numpy.ndarray
        A two column numpy array with the with the location and the 
        amplitude of each peak.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")

    peak_args = find_peaks(data.y, **kwargs)[0]
    if len(peak_args) < n_peaks:
        print('Few peaks were found, try reducing restrictions.')

    peak_args = peak_args[data.y[peak_args].argsort()
                         ][:-(n_peaks+1):-1]
    peak_info = np.concatenate([data.x[peak_args, None],
                           data.y[peak_args, None]], axis=1)

    if as_Data:
        return Data(peak_info)
    else:
        return peak_info


def peak_height(data, position, x_range, x_value=False):
    """Gives the info on highest peak in a given region.

    It achieves this my trimming the data to a region and then returning the 
    maximum y_value. This is useful for extracting peak heights from an FFT.

    Parameters
    ----------
    data : Data
        The data to extract the peak hight from.
    position : float
        The central position to search for the peak.
    x_range : float
        The range in x to look for the peak. This is the total range so will 
        extend half of this from ``position``.
    x_value : bool, optional
        If `True` the x value and the y value is returned. The default is 
        `False` which only returns the y value.

    Returns
    -------
    x_peak : float
        If x_value is `True` the x position of the peak is returned.
    y_peak : float
        The y_value of the peak. Which is the highest y value in a given 
        range of the data.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data need to be a Data object but instead was {type(data)}.")

    trimmed = data.x_cut(position - x_range/2,  position + x_range/2)
    if len(trimmed) == 0:
        raise ValueError(
            f"The x_range given to look for a peak doesn't contain any "
            f"data.")

    peak_arg = np.argmax(trimmed.y)

    if x_value:
        return trimmed.x[peak_arg], trimmed.y[peak_arg]
    else:
        return trimmed.y[peak_arg]

