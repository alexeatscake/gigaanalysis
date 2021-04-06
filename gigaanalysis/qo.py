"""Giga Analysis - Quantum Oscillations

"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, get_window, find_peaks


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


def FFT(data, n=65536, window='hanning', freq_cut=0.):
    """Performs an Fast Fourier Transform on the given data.

    This assumes that the data is real, and the data provided needs to be 
    evenly spaced. Makes use of :func:`numpy.fft.rfft`.

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


def counting_freq(start_field, end_field, number_peaks):
    """Counting quantum oscillation to obtain a frequency.

    This returns the frequency of a quantum oscillation given a range 
    of field and the number of osscilations that occur in that range.
    Performs ``n*B1*B2/(B2 - B1)``

    Parameters
    ----------
    start_field : float
        The magnetic field to start the counting range.
    end_field : float
        The magnetic field to end the counting range.
    number_peaks : float
        The number of peaks in the given range.

    Returns
    -------
    frequency : float
        The frequency of the expected quantum oscillation in Tesla.
    """
    return number_peaks*start_field*end_field/(end_field-start_field)


def counting_num(start_field, end_field, frequency):
    """Expected count of quantum oscillation at a given frequency.

    This returns the number of quantum oscillations given a range 
    of field and frequency of quantum oscillations in question.
    Performs ``Freq*(B2 - B1)/(B1*B2)``

    Parameters
    ----------
    start_field : float
        The magnetic field to start the counting range.
    end_field : float
        The magnetic field to end the counting range.
    frequency : float
        The quantum oscillation frequency in Tesla.

    Returns
    -------
    number : float
        The number of expected peaks in this field range.
    """
    return frequency*(end_field-start_field)/(start_field*end_field)


class QO():
    """Quantum Oscillation object

    This takes a sweep in magnetic field and analyses the sweep to check the 
    presence and properties of quantum oscillations. It uses the 
    :class:`.Data` objects to hold the information.

    This class has an arbitrary subtraction function which is used to remove 
    the background signal and leave the quantum oscillations. Other ready 
    made classes exist that have a certain subtraction function 
    incorporated. This class functions as their parent.

    The analysis happens in 4 stages and the data is assessable at each 
    stage as a :class:`.Data` attribute. The first stage interpolates the 
    data evenly across the field window of interest. The second stage 
    performs the background subtraction. The third stage inverts the data in 
    in field. The final stage Fourier transforms the inverted signal.

    Parameters
    ----------
    data : Data
        The raw data of the field sweep to look for quantum oscillations in.
    min_field : float
        The lowest field value of the field range to inspect.
    max_field : float
        The highest field value of the field range to inspect.
    subtract_func : calculable
        This should take one :class:`.Data` object and return one 
        :class:`.Data` of the same length. The input data is the 
        interpolated sweep, and the output should be the data after the 
        background has been subtracted.
    step_size : float, optional
        The size of field steps to interpolate. The default is 4 times the 
        average step size in the raw data.
    fft_cut : float, optional
        The maximum frequency to consider in FFT, higher frequencies are 
        dropped. THe default is to keep all the data.
    strip_nan : bool, optional
        If `True` non finite values are removed from the raw data. The 
        default is `False` and this will raise an error is non finite values 
        are in the raw data.

    Attributes
    ----------
    raw : Data
        The data originally given to the class.
    interp : Data
        The sweep cut to the field range and interpolated.
    sub : Data
        The sweep after the background subtraction.
    invert : Data
        The subtracted sweep after inverting the field values.
    fft : Data
        The Fourier transform of the inverted signal.
    min_field : float
        The minimum field in the range in consideration.
    max_field : float
        The maximum field in the range in consideration.
    step_size : float
        The steps in field calculated in the interpolation.
    """
    def __init__(self, data, min_field, max_field, subtract_func,
                step_size=None, fft_cut=0, strip_nan=False):
        if type(data) != Data:
            try:
                data = Data(data)
            except:
                raise TypeError(
                    f"data was not a Data object nor could it be cast to "
                    f"one was of type {type(data)}")
        
        self.raw = data
        self.min_field = min_field
        self.max_field = max_field

        if np.min(self.raw.x) > min_field:
            raise ValueError(
                "max_field value to interpolate is below data")
        if np.max(self.raw.x) < max_field:
            raise ValueError(
                "max_field value to interpolate is above data")

        if strip_nan:
            self.raw = self.raw.strip_nan()
        else:
            if not np.all(np.isfinite(self.raw)):
                raise ValueError(
                    f"The data contained non finite values and strip_nan"
                    f"was set to False.")

        if step_size == None:
            self.step_size = (self.raw.max_x() - self.raw.min_x()
                )/len(self.raw)/4
        else:
            self.step_size = step_size
        
        self.interp = self.raw.interp_range(min_field, max_field,
                                            self.step_size)
        self.sub = subtract_func(self.interp)
        self.invert = invert_x(self.sub)
        self.fft = FFT(self.invert, freq_cut=fft_cut)

    def __len__(self):
        return self.interp.x.size

    def __repr__(self):
        return  (
            f"Quantum Oscillation object:\n"
            f"Field Range {self.min_field:.2f} T to {self.max_field:.2f} T\n"
            f"Number of points {self.interp.x.size}")

    def peaks(self, n_peaks=4, as_Data=False, **kwargs):
        """Finds the largest Fourier Transform peaks.

        This makes use of :func:`get_peaks` and the ``**kwargs`` are passed 
        to :func:`scipy.signal.find_peaks`.

        Parameters
        ----------
        n_peaks : int, optional
            The number of peaks to identify, the default is 4.
        as_Data : bool, optional
            If `True` the peak info is returned as a :class:`.Data` object. 
            The default is `False`. 

        Returns
        -------
        peak_info : numpy.ndarray
            A two column numpy array with the with the location and the 
            amplitude of each peak.
        """
        return get_peaks(self.fft, n_peaks, as_Data=as_Data, **kwargs)

    def peak_height(self, position, x_range, x_value=False):
        """Provides the hight of the highest FFT in a given range.

        Makes use of the function :func:`peak_height`.
    
        Parameters
        ----------
        position : float
            The central position to search for the peak.
        x_range : float
            The range in x to look for the peak. This is the total range so 
            will extend half of this from ``position``.
        x_value : bool, optional
            If `True` the x value and the y value is returned. The default 
            is `False` which only returns the y value.

        Returns
        -------
        x_peak : float
            If x_value is `True` the x position of the peak is returned.
        y_peak : float
            The y_value of the peak. Which is the highest y value in a given 
            range of the data.
        """
        return peak_height(self.fft, position, x_range, x_value=x_value)

    def FFT_again(self, n=65536, window='hanning', freq_cut=0):
        """Recalculates the FFT.

        After recalcuating the FFT the new FFT is returned and also the 
        new FFT is saved to the :attr:`fft` attribute. This makes use of 
        :func:`FFT`.

        Parameters
        ----------
        n : int, optional
            The number of points to make the FFT extra points will be zero 
            padded. The default is ``2**16 = 65536``. If the data is longer 
            than the value of n, n is rounded up to the next power of 2.
        window : str, optional
            The type of windowing function to use taken from 
            :func:`scipy.signal.get_window`. The default is 'hanning'.
        freq_cut : float, optional
            The frequency to drop all the higher from. The default is 0 
            which means that all the frequencies are kept.

        Returns
        -------
        fft_result : Data
            A data object with the FFT frequencies in the x values and the 
            amplitudes in the y values.
        """
        self.fft = FFT(self.invert, n=n, window=window, freq_cut=freq_cut)
        return self.fft

    def to_csv(self, file_name, sep=','):
        """This saves the data contained to a .csv file.

        This saves the data in a csv file. It includes the interpolated,
        subtracted, inverse signals as well as the FFT. The FFT is
        interpolated to be the same length as the interpolated data.

        Parameters
        ----------
        file_name : str
            The name of the file to save the data to. If the file type is 
            not one of '.csv', '.txt', or '.dat'; then '.csv' will be 
            appended on to the end of the name.
        sep : str, optional
            The character used to the delimitate between the data, the 
            default is ','.
        """
        if file_name[-4:] not in ['.csv', '.txt', '.dat']:
            file_name += '.csv'

        output_data = np.concatenate([
                        self.interp.values,
                        self.sub.values,
                        self.invert.values,
                        self.fft.interp_number(len(self)).values,
                            ], axis=1)

        header_line = 'Field_Interp{0:s}Interp_Signal{0:s}' \
                        'Field_Sub{0:s}Sub_Signal{0:s}' \
                        'Inverse_Field{0:s}Inverse_Signal{0:s}' \
                        'FFT_freq{0:s}FFT_amp'.format(sep)

        np.savetxt(file_name, output_data,
                   delimiter=sep, comments='',
                   header=header_line)


class QO_loess(QO):
    """Quantum Ossilation object with LOESS subtraction

    This is a example of the :class:`QO` which the subtraction using 
    :func:`loess`. The form is the name but with the initialising function 
    takes the arguments to define the LOESS background subtraction.

    Parameters
    ----------
    data : Data
        The raw data of the field sweep to look for quantum oscillations in.
    min_field : float
        The lowest field value of the field range to inspect.
    max_field : float
        The highest field value of the field range to inspect.
    loess_win : float
        The length of the window in Tesla to use for the LOESS subtraction.
    loess_poly : int
        The order of the polynomial to use for the LOESS subtraction.
    step_size : float, optional
        The size of field steps to interpolate. The default is 4 times the 
        average step size in the raw data.
    fft_cut : float, optional
        The maximum frequency to consider in FFT, higher frequencies are 
        dropped. THe default is to keep all the data.
    strip_nan : bool, optional
        If `True` non finite values are removed from the raw data. The 
        default is `False` and this will raise an error is non finite values 
        are in the raw data.

    Attributes
    ----------
    :
        This class has the same attributes as the :class:`QO` calss but also 
        with the information about the LOESS subtraction.

    loess_win : float
        The length of the window in Tesla to use for the LOESS subtraction.
    loess_poly : int
        The order of the polynomial to use for the LOESS subtraction.
    """
    """
This class is designed to keep all the information for one sweep together
The data given needs to be a ga.Data class or objects that can be passed
to make one.

This class is a subclass of ga.QO
This class is using LOESS background fitting to perform the subtraction. 

The first set of attributes are the same as the parameters passed to the
class in initialisation. The remaining are mostly ga.Data objects that 
are produced in the steps of analysis the quantum oscillations.

Attributes:
    raw (ga.Data): Original data passed to the class
    min_field (float): The minimum field to be considered
    max_field (float): The maximum field to be considered
    loess_win (float): The window length to be passed to ga.loess
    loess_poly (float): The polynomial order to be
        passed to ga.loess_poly
    step_size (float): The spacing between the field points to
        be interpolated
    interp (ga.Data): The sweep cut to the field range and interpolated
        evenly in field
    sub (ga.Data): The sweep with the background subtracted
    invert (ga.Data): The background subtracted sweep evenly interpolated
        in inverse field
    fft (ga.Data): The flourier transform of the inverse
"""
    def __init__(self, data, min_field, max_field, loess_win, loess_poly,
                step_size=None, fft_cut=0, strip_nan=False):

        def bg_sub(interp):
            return interp - loess(interp, loess_win, loess_poly)

        QO.__init__(self, data, min_field, max_field, bg_sub,
            step_size=step_size, fft_cut=fft_cut, strip_nan=strip_nan)

        self.loess_win = loess_win
        self.loess_poly = loess_poly


    def __dir__(self):
        return [*QO.__dir__(self), 'loess_poly', 'loess_win']


    def _repr_html_(self):
        return print('Quantum Oscillation object:\n' \
                     'LOESS Background Subtraction\n' \
                     'Field Range {:.2f} to  {:.2f} \n' \
                     'LOESS polynomial {:.2f}\n' \
                     'LOESS window {:.2f}\n'.format(
                        self.min_field, self.max_field,
                        self.loess_poly, self.loess_win))


class QO_loess_av(QO_loess):
    """
This class is designed to keep all the information for one sweep together
The data given needs to be a ga.Data class or objects that can be passed
to make one.

This class is a subclass of ga.QO_loess
This class is using LOESS background fitting to perform the subtraction.

The first set of attributes are the same as the parameters passed to the
class in initialisation. The remaining are mostly ga.Data objects that 
are produced in the steps of analysis the quantum oscillations.

Attributes:
    raw ([ga.Data]): Original data passed to the class in the from of a list
        of ga.Data objects
    min_field (float): The minimum field to be considered
    max_field (float): The maximum field to be considered
    loess_win (float): The window length to be passed to ga.loess
    loess_poly (float): The polynomial order to be
        passed to ga.loess_poly
    step_size (float): The spacing between the field points to
        be interpolated
    interp (ga.Data): The sweep cut to the field range and interpolated
        evenly in field
    sub (ga.Data): The sweep with the background subtracted
    invert (ga.Data): The background subtracted sweep evenly interpolated
        in inverse field
    fft (ga.Data): The flourier transform of the inverse
"""
    def __init__(self, data_list, min_field, max_field, loess_win,
            loess_poly, step_size=None, fft_cut=0):
        if type(data_list) != list:
            raise TypeError('Not given a list of ga.Data class\n' \
                            'Was given {}'.format(type(data_list)))
        self.raw = []
        for data in data_list:
            if type(data) != Data:
                try:
                    self.raw.append(Data(data))
                except:
                    raise TypeError('Not given data class in list!\n' \
                                    'Was given {}'.format(type(data)))
            else:
                self.raw.append(data)
        
        self.min_field = min_field
        self.max_field = max_field
        self.loess_win = loess_win
        self.loess_poly = loess_poly
        
        if step_size == None:
            self.step_size = np.min(
                [np.abs(np.average(np.diff(data.x)))/4 for data in self.raw])
        else:
            self.step_size = step_size
        
        interp_list = []
        for data in self.raw:
            interp_list.append(data.interp_range(min_field, max_field,
                                                    self.step_size))
        def bg_sub(interp):
            return interp - loess(interp, loess_win, loess_poly)

        sub_list = []
        for interp in interp_list:
            sub_list.append(bg_sub(interp))

        self.interp = mean(interp_list)
        self.sub = mean(sub_list)
        self.invert = invert_x(self.sub)
        self.fft = FFT(self.invert, freq_cut=fft_cut)





