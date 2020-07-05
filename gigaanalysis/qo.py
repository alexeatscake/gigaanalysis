"""

Giga Analysis - Quantum Oscillations

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, get_window, find_peaks

from .data import *


def invert_x(data_set):
    """
    This inverts the x data and then reinterpolates the y points so the x
    data is evenly spread.
    Args:
        data_set (Data): The data object to invert x
    Returns:
        A data object with inverted x and evenly spaced x
    """
    if not np.all(data_set.x[:-1] <= data_set.x[1:]):
        raise ValueError('Array to invert not sorted!')
    interp = interp1d(*data_set.both, bounds_error=False,
                     fill_value=(data_set.y[0], data_set.y[1])
                     )  # Needs fill_value for floating point errors
    new_x = np.linspace(1./data_set.x.max(), 1./data_set.x.min(),
                       len(data_set))
    return Data(new_x, interp(1/new_x))

def loess(data_set, x_window, polyorder):
    """
This performs a Local regression smooth on the data and outputs a new
data object. It uses scipy.signal.savgol_filter
Args:
    data_set (Data): The data set to be smoothed
    x_range (float): The window length to use for the smoothing
    polyorder (int): The order of polynomial to fit
Returns:
    A data object of smoothed data.
"""
    spacing = data_set.x[1] - data_set.x[0]
    if not np.isclose(spacing, np.diff(data_set.x)).all():
        raise ValueError('The data needs to be evenly spaced to smooth')
    smooth_data = savgol_filter(data_set.y,
                                2*int(round(0.5*x_window/spacing)) + 1,
                                polyorder=polyorder)
    return Data(data_set.x, smooth_data)


def poly_reg(data_set, polyorder):
    """
This performs a polynomial fit to the data to smooth it and outputs the
smoothed data. It uses numpy.polyfit
Args:
    data_set (Data): The data set to be smoothed
    polyorder (int): The order of the polynomial to fit
Returns:
    A data set of the fitted values
"""
    fit = np.polyfit(*data_set.both, deg=polyorder)
    y_vals = 0.
    for n, p in enumerate(fit):
        y_vals += p*np.power(data_set.x, polyorder-n)
    return Data(data_set.x, y_vals)


def FFT(data_set, n=65536, window='hanning', freq_cut=0):
    """
This performs an FFT on the data set and outputs another.
Args:
    data_set (Data): The data to perform the FFT on
    n (int default:65536): The number of points to FFT, extra points
        will be added with zero pading
    window (str default:'hanning'): The windowing function to use the
        list is given in scipy.signal.get_window
    freq_cut (float default:0): If given the frequencies higher than this
        are not included with the FFT
Returns:
    A data set of the resulting FFT
"""
    spacing = data_set.x[1] - data_set.x[0]
    if not np.isclose(spacing,
                      np.diff(data_set.x)).all():
        raise ValueError('The data needs to be evenly spaced to smooth')
    fft_vals = np.abs(np.fft.rfft(data_set.y*get_window(window,
                                                        len(data_set)),
                                  n=n))
    fft_freqs = np.fft.rfftfreq(n, d=spacing)
    freq_arg = None
    if freq_cut > 0:
        freq_arg = np.searchsorted(fft_freqs, freq_cut)
    return Data(fft_freqs[0:freq_arg], fft_vals[0:freq_arg])


def get_peaks(data_set, n_peaks=4, **kwargs):
    """
Finds the four largest peaks in the data set and output a two column
numpy array with the frequencies and amplitudes.
Args:
    data_set (Data): The FFT data to find the peaks in
    n_peaks (int): The number of peaks to find
    **kwargs will be passed to scipy.singal.find_peaks
Returns:
    A two column numpy array with frequencies and amplitudes
"""
    peak_args = find_peaks(data_set.y, **kwargs)[0]
    if len(peak_args) < n_peaks:
        print('Few peaks were found, try reducing restrictions.')
    peak_args = peak_args[data_set.y[peak_args].argsort()
                         ][:-(n_peaks+1):-1]
    return np.concatenate([data_set.x[peak_args, None],
                           data_set.y[peak_args, None]], axis=1)


def peak_height(data_set, position, x_range, x_value=False):
    """
This takes a data set around a positions and in a certain range takes
the largest value and outputs that values hight. This is useful for
extracting peak heights from an FFT.
Args:
    data_set (Data): The data object to get the peak height from
    position (float): The central location of the peak in x
    x_range (float): The range in x to look for the peak
    x_value (bool, default:False): If true x value also produced
Returns:
    If x_value is false a float is returned with the hight of the peak
    If x_value is true a np array with x and y values of peak
"""
    trimmed = data_set.x_cut(position - x_range/2,  position + x_range/2)
    peak_arg = np.argmax(trimmed.y)
    if x_value:
        return np.array([trimmed.x[peak_arg], trimmed.y[peak_arg]])
    else:
        return trimmed.y[peak_arg]


def counting_freq(start_field, end_field, number_peaks):
    """
This can give you the frequency from counting peaks.
Performs n*B1*B2/(B2-B1)
Args:
    start_field (float): The lowest field in the counting range
    end_field (float): The highest field in the counting range
    number_peaks (float): The number of peaks in the range
Returns
    The frequency in Tesla as a float
"""
    return number_peaks*start_field*end_field/(end_field-start_field)


def counting_freq(start_field, end_field, frequency):
    """
This provides the number of peaks you expect to count for a frequency.
Performs Freq*(B2-B1)/(B1*B2)
Args:
    start_field (float): The lowest field in the counting range
    end_field (float): The highest field in the counting range
    frequency (float): The frequency in Tesla
Returns
    The number of peaks as a float
"""
    return frequency*(end_field-start_field)/(start_field*end_field)


class QO():
    """
This class is designed to keep all the information for one sweep together
The data given needs to be a ga.Data class or objects that can be passed
to make one.

The first set of attributes are the same as the parameters passed to the
class in initialisation. The remaining are mostly ga.Data objects that 
are produced in the steps of analysis the quantum oscillations.

Attributes:
    raw (ga.Data): Original data passed to the class
    min_field (float): The minimum field to be considered
    max_field (float): The maximum field to be considered
    step_size (float): The spacing between the field points to
        be interpolated
    interp (ga.Data): The sweep cut to the field range and interpolated
        evenly in field
    sub (ga.Data): The sweep with the background subtracted
    invert (ga.Data): The background subtracted sweep evenly interpolated
        in inverse field
    fft (ga.Data): The flourier transform of the inverse
"""
    def __init__(self, data, min_field, max_field, subtract_func,
                step_size=None, fft_cut=0):
        if type(data) != Data:
            try:
                data = Data(data)
            except:
                raise TypeError('Not given data class!\n' \
                                'Was given {}'.format(type(data)))
        
        self.raw = data
        self.min_field = min_field
        self.max_field = max_field
        
        if step_size == None:
            self.step_size = np.abs(np.average(np.diff(data.x)))/4
        else:
            self.step_size = step_size
        
        self.interp = self.raw.interp_range(min_field, max_field,
                                            self.step_size)
        self.sub = subtract_func(self.interp)
        self.invert = invert_x(self.sub)
        self.fft = FFT(self.invert, freq_cut=fft_cut)

    def __dir__(self):
        return ['raw', 'min_field', 'max_field', 'step_size',
                'interp', 'sub', 'invert', 'fft', 'peaks',
                'peak_hight', 'FFT_again']

    def __len__(self):
        return self.interp.x.size

    def _repr_html_(self):
        return print('Quantum Oscillation object:\n' \
                     'Field Range {:.2f} to  {:.2f} \n'.format(
                        self.min_field, self.max_field))

    def peaks(self, n_peaks=4, **kwargs):
        """
Calls ga.get_peaks on the FTT
Finds the four largest peaks in the data set and output a two column
numpy array with the frequencies and amplitudes.
Args:
    n_peaks (int): The number of peaks to find
    **kwargs will be passed to scipy.singal.find_peaks
Returns:
    A two column numpy array with frequencies and amplitudes
"""
        return get_peaks(self.fft, n_peaks, **kwargs)

    def peak_hight(self, position, x_range, x_value=False):
        """
Calls ga.peak_hight on the FFT
This takes a data set around a positions and in a certain range takes
the largest value and outputs that values hight. This is useful for
extracting peak heights from an FFT.
Args:
    position (float): The central location of the peak in x
    x_range (float): The range in x to look for the peak
    x_value (bool, default:False): If true x value also produced
Returns:
    If x_value is false a float is returned with the hight of the peak
    If x_value is true a np array with x and y values of peak
"""
        return peak_height(self.fft, position, x_range, x_value=False)

    def FFT_again(self, n=65536, window='hanning', freq_cut=0):
        """
Recalculates the FTT and returns it, also saved to self.fft
This is so the extra settings can be used. Makes use of ga.FFT
Args:
    n (int default:65536): The number of points to FFT, extra points
        will be added with zero pading
    window (str default:'hanning'): The windowing function to use the
        list is given in scipy.signal.get_window
    freq_cut (float default:0): If given the frequencies higher than this
        are not included with the FFT
"""
        self.fft = FFT(self.invert, n=n, window=window, freq_cut=freq_cut)
        return self.fft

    def to_csv(self, file_name, sep=','):
        """
This saves the data in a csv file. It includes the interpolated,
subtracted, inverse signals as well as the FFT. The FFT is
interpolated to be the same length as the interpolated data.
Args:
    file_name (str): The file name to save the data
    sep (str default:','): The character to delimitate the data
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
                step_size=None, fft_cut=0):

        def bg_sub(interp):
            return interp - loess(interp, loess_win, loess_poly)

        QO.__init__(self, data, min_field, max_field, bg_sub,
                    step_size=step_size, fft_cut=fft_cut)

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
    def __init__(self, data_list, min_field, max_field, loess_win, loess_poly,
                step_size=None, fft_cut=0):
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





