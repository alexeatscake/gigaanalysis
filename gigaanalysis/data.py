
"""GigaAnalysis - Data Type

This holds the data class and the functions that will manipulate them.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, get_window, find_peaks


class Data():
    """
The Data Class

Data object holds the data in the measurements. It works as a simple
wrapper of a two column numpy array. The point is that operations apply
to the y values and interpretation happens to compare the cosponsoring
data points.

The initialisation is documented in the __init__ method.

Attributes:
    values (np array): Two column numpy array with the x and y data in
    x (np array): The x data in a 1D numpy array
    y (np array): The y data in a 1D numpy array
    both (two np arrays): The x data then the y data in a tuple

"""
    def __init__(self, values, split_y=None, strip_sort=False,
                 interp_full=0.):
        """
The __init__ method to produce a incidence of the Data class
Args:
    values (np array): A two column numpy array with the x data in
        the first column and the y data in the second. If a second 
        no array is given then the first corresponds to the x data.
    split_y (np array default:None): A 1D numpy array containing the
        y data. If default all the data should be contained in
        first array.
"""
        if type(values) in [pd.core.frame.DataFrame,
                               pd.core.series.Series]:
            values = values.values

        if split_y is not None:
            if type(split_y) in [pd.core.frame.DataFrame,
                                 pd.core.series.Series]:
                split_y = split_y.values
            elif type(values) is not np.ndarray:
                raise ValueError(
                    "If x and y data are split both need to " + \
                    "be a 1D numpy array.\n" +\
                    "x is not a numpy array.")
            elif type(split_y) is not np.ndarray:
                raise ValueError(
                    "If x and y data are split both need to " + \
                    "be a 1D numpy array.\n" +\
                    "y is not a numpy array.")
            elif values.ndim != 1 or split_y.ndim != 1:
                raise ValueError(
                    "If x and y data are split both need to " + \
                    "be a 1D numpy array.\n" +\
                    "x or y is not 1D.")
            elif values.size != split_y.size:
                raise ValueError(
                    "If x and y data are split both need to " + \
                    "be the same size.")
            values = np.concatenate([values[:, None],
                                     split_y[:, None]], axis=1)

        if type(values) is not np.ndarray:
            raise TypeError('values is not a numpy array. \n' +\
                            'Needs to be a two column numpy array.')
        elif len(values.shape) != 2 or values.shape[1] != 2:
            raise ValueError('values dose not have two columns. \n' +\
                            'Needs to be a two column numpy array.')

        if strip_sort:
            values = values[~np.isnan(values).any(axis=1)]
            values = values[np.argsort(values[:, 0]), :]


        self.values = values.astype(float)   # All the data
        self.x = values[:, 0]  # The x data
        self.y = values[:, 1]  # The y data
        self.both = values[:, 0], values[:, 1]  # A tuple of the data

        if interp_full != 0.:
            self.to_even(interp_full)


    def __str__(self):
        return np.array2string(self.values)

    def __repr__(self):
        return 'GA Data object:\n{}'.format(self.values)

    def _repr_html_(self):
        return 'GA Data object:\n{}'.format(self.values)

    def __dir__(self):
        return ['values', 'x', 'y', 'both',
                'y_from_x', 'x_cut',
                'interp_full', 'interp_number', 'interp_range',
                'plot']

    def __len__(self):
        return self.x.size

    def __mul__(self, other):
        """
The Data class can be multiplied and this just effects the y values,
the x values stay the same.
This can be multiplied to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return Data(self.x, self.y*other)
        elif type(other) is np.ndarray:
            if other.size == 1:
                return Data(self.x, self.y*float(other))
            elif self.x.shape == other.shape:
                return(Data(self.x, self.y*other))
            else:
                raise ValueError('Numpy array to multiply to data '\
                                'is wrong shape.')
        elif type(other) == type(self):
            if np.array_equal(self.x, other.x):
                return(Data(self.x, self.y*other.y))
            else:
                raise ValueError('The two Data class need to have the same '\
                                    'x values to be multiplied.')
        else:
            raise TypeError('Cannot multiple Data class with this type')

    __rmul__ = __mul__

    def __truediv__(self, other):
        """
The Data class can be divided and this just effects the y values,
the x values stay the same.
This can be divided to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return Data(self.x, self.y/other)
        elif type(other) is np.ndarray:
            if other.size == 1:
                return Data(self.x, self.y/float(other))
            elif self.x.shape == other.shape:
                return(Data(self.x, self.y/other))
            else:
                raise ValueError('Numpy array to divide to data '\
                                'is wrong shape.')
        elif type(other) == type(self):    
            if np.array_equal(self.x, other.x):
                return(Data(self.x, self.y/other.y))
            else:
                raise ValueError('The two Data class need to have the same '\
                                    'x values to be divided.')
        else:
            raise TypeError('Cannot divide Data class with this type')

    def __rtruediv__(self, other):
        """
The Data class can be divided and this just effects the y values,
the x values stay the same.
This can be divided to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return Data(self.x, other/self.y)
        elif type(other) is np.ndarray:
            if other.size == 1:
                return Data(self.x, float(other)/self.y)
            elif self.x.shape == other.shape:
                return(Data(self.x, other/self.y))
            else:
                raise ValueError('Numpy array to divide to data '\
                                'is wrong shape.')
        elif type(other) == type(self):  
            if np.array_equal(self.x, other.x):
                return(Data(self.x, other.y/self.y))
            else:
                raise TypeError('The two Data class need to have the same '\
                                    'x values to be divided.')
        else:
            raise ValueError('Cannot divide Data class with this type')

    def __add__(self, other):
        """
The Data class can be added and this just effects the y values, the
x values stay the same.
This can be added to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return(Data(self.x, self.y + other))
        elif type(other) is np.ndarray:
            if other.size == 1:
                return(Data(self.x, self.y + other))
            elif self.x.shape == other.shape:
                return(Data(self.x, self.y + other))
            else:
                raise ValueError('Numpy array to add to data '\
                                'is wrong shape.')
        elif type(other) == type(self):   
            if np.array_equal(self.x, other.x):
                return(Data(self.x, self.y + other.y))
            else:
                raise ValueError('The two Data class need to have the same '\
                                    'x values to be added.')
        else:
            raise TypeError('Cannot add Data class with this type')

    __radd__ = __add__

    def __sub__(self, other):
        """
The Data class can be subtracted and this just effects the y values,
the x values stay the same.
This can be subtracted to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return(Data(self.x, self.y - other))
        elif type(other) is np.ndarray:
            if other.size == 1:
                return(Data(self.x, self.y - other))
            elif self.x.shape == other.shape:
                return(Data(self.x, self.y - other))
            else:
                raise ValueError('Numpy array to subtract to data '\
                                'is wrong shape.')        
        elif type(other) == type(self):    
            if np.array_equal(self.x, other.x):
                return(Data(self.x, self.y - other.y))
            else:
                raise ValueError('The two Data class need to have the same '\
                                    ' x values to be subtracted.')
        else:
            raise ValueError('Cannot subtract Data class with this type')

    def __rsub__(self, other):
        """
The Data class can be subtracted and this just effects the y values,
the x values stay the same.
This can be subtracted to a float, int, numpy array with 1 value,
or a numpy array with same length, or a other Data object
with the same length.
"""
        if type(other) in [float, int, np.float_, np.int_]:
            return(Data(self.x, other - self.y))
        elif type(other) is np.ndarray:
            if other.size == 1:
                return(Data(self.x, other - self.y))
            elif self.x.shape == other.shape:
                return(Data(self.x, other - self.y))
            else:
                raise ValueError('Numpy array to subtract to data '\
                                'is wrong shape.')
        elif type(other) == type(self):   
            if np.array_equal(self.x, other.x):
                return(Data(self.x, other.y - self.y))
            else:
                raise ValueError('The two Data class need to have the same '\
                                    'x values to be subtracted.')
        else:
            raise TypeError('Cannot subtract Data class with this type')

    def __abs__(self):
        """
The abs function takes the absolute value of the y values.
"""
        return Data(self.x, abs(self.y))

    def __pow__(self, power):
        """
Takes the power of the y values and leaves the x-values unchanged.
"""
        return Data(self.x, pow(self.y, power))

    def __eq__(self, other):
        """
The Data class is only equal to other data classes with the same data.
"""
        if type(other) != type(self):
            return False
        else:
            return np.array_equal(self.values, other.values)

    def __iter__(self):
        """
The iteration happens on the values, like if was numpy array.
"""
        return iter(self.values)
    
    def y_from_x(self, x_val):
        """
This function gives the y value for a certain x value or
set of x values.
Args:
    x_val (float): X values to interpolate y values from
Returns:
    y values corresponding to the requested x values in nd array
"""
        y_val = interp1d(self.x, self.y, bounds_error=False,
                        fill_value=(self.y.min(), self.y.max()))(x_val)
        if y_val.size != 1:
            return y_val
        else:
            return float(y_val)

    def x_cut(self, x_min, x_max):
        """
This cuts the data to a region between x_min and x_max
Args:
    x_min (float): The minimal x value to cut the data
    x_max (float): The maximal x value to cut the data
Returns:
    An data object with the values cut to the given x range
"""
        if x_min > x_max:
            raise ValueError('x_min should be smaller than x_max')
        return Data(self.values[
                    np.searchsorted(self.x, x_min, side='left'):
                    np.searchsorted(self.x, x_max, side='right'), :])

    def interp_range(self, min_x, max_x, step_size, **kwargs):
        '''
This evenly interpolates the data points between a min
and max x value. This is used so that the different
sweeps can be combined with the same x-axis.
It uses scipy.interpolate.interp1d and **kwargs can be pass to it.
Args:
    data_set (Data): The data set to be interpolated
    min_x (float): The minimum x value in the interpolation
    max_y (float): The maximum x value in the interpolation
    step_size (float): The step size between each point
Returns:
    A new data set with evenly interpolated points.
'''
        if np.min(self.x) > min_x:
            raise ValueError('min_x value to interpolate is below data')
        if np.max(self.x) < max_x:
            raise ValueError('max_x value to interpolate is above data')
        x_vals = np.arange(min_x, max_x, step_size)
        return Data(x_vals,
                interp1d(self.x, self.y, **kwargs)(x_vals))

    def to_range(self, min_x, max_x, step_size, **kwargs):
        '''
This evenly interpolates the data points between a min
and max x value. This is used so that the different
data objects can be combined with the same x-axis. This changes
the object.
It uses scipy.interpolate.interp1d and **kwargs can be pass to it.
Args:
    data_set (Data): The data set to be interpolated
    min_x (float): The minimum x value in the interpolation
    max_y (float): The maximum x value in the interpolation
    step_size (float): The step size between each point
'''
        if np.min(self.x) > min_x:
            raise ValueError('min_x value to interpolate is below data')
        if np.max(self.x) < max_x:
            raise ValueError('max_x value to interpolate is above data')
        x_vals = np.arange(min_x, max_x, step_size)
        y_vals = interp1d(self.x, self.y, **kwargs)(x_vals)
        self.values = np.concatenate((x_vals[:, None], y_vals[:, None]),
                                     axis=1)
        self.x = x_vals
        self.y = y_vals
        self.both = x_vals, y_vals


    def interp_full(self, step_size, **kwargs):
        """
This interpolates the data to give an even spacing. This is useful
for combining different data sets.
It uses scipy.interpolate.interp1d and **kwargs can be pass to it.
Args:
    step_size (float): The spacing of the data along x.
Return:
    A Data class with the interpolated data.
"""
        x_start = np.ceil(self.x.min()/step_size)*step_size
        x_stop = np.floor(self.x.max()/step_size)*step_size
        x_vals = np.linspace(x_start, x_stop,
                           int(round((x_stop - x_start)/step_size)) + 1)
        y_vals = interp1d(self.x, self.y, **kwargs)(x_vals)
        return Data(x_vals, y_vals)

    def interp_number(self, point_number, **kwargs):
        """
This interpolates the data to give an even spacing. This is useful
for saving data of different types together
It uses scipy.interpolate.interp1d and **kwargs can be pass to it.
Args:
    point_number (int): The spacing of the data along x.
Return:
    A Data class with the interpolated data.
"""

        x_vals = np.linspace(self.x.min(), self.x.max(),
                             int(point_number))
        y_vals = interp1d(self.x, self.y, **kwargs)(x_vals)
        return Data(x_vals, y_vals)

    def to_even(self, step_size, **kwargs):
        """
This interpolates the data to give an even spacing, and changes
the data file.
It uses scipy.interpolate.interp1d and **kwargs can be pass to it.
Args:
    step_size (float): The spacing of the data along x.
"""
        x_start = np.ceil(self.x.min()/step_size)*step_size
        x_stop = np.floor(self.x.max()/step_size)*step_size
        x_vals = np.linspace(x_start, x_stop,
                           int(round((x_stop - x_start)/step_size)) + 1)
        y_vals = interp1d(self.x, self.y, **kwargs)(x_vals)
        self.values = np.concatenate((x_vals[:, None], y_vals[:, None]),
                                     axis=1)
        self.x = x_vals
        self.y = y_vals
        self.both = x_vals, y_vals

    def sort(self):
        """
This sorts the data set in x and returns the new array.
Returns:
    A Data class with the sorted data.
"""
        return Data(self.values[np.argsort(self.x), :])

    def strip_nan(self):
        """
This removes any row which has a nan value in.
Returns:
    Data class without any nan in.
"""
        return Data(self.values[~np.isnan(self.values).any(axis=1)])

    def min_x(self):
        """
This provides the lowest value of x
Returns:
    A float of the minimum x value
"""
        return np.min(self.x)

    def max_x(self):
        """
This provides the lowest value of x
Returns:
    A float of the minimum x value
"""
        return np.max(self.x)

    def spacing_x(self):
        """
Provides the average separation of the x values
(max_x - min_x)/num_points 
Returns:
    A float of the average spacing in x
"""
        return (self.max_x() - self.min_x())/len(self)

    def apply_x(self, function):
        """
This takes a function and applies it to the x values.
Args:
    function (func): THe function to apply to the x values
Returns:
    Data class with new x values
"""
        return Data(function(self.x), self.y)

    def apply_y(self, function):
        """
This takes a function and applies it to the y values.
Args:
    function (func): THe function to apply to the y values
Returns:
    Data class with new x values
"""
        return Data(self.x, function(self.y))

    def plot(self, *args, axis=None, **kwargs):
        """
Simple plotting function that runs
matplotlib.pyplot.plot(self.x, self.y, *args, **kwargs)
Added a axis keyword which operates so that if given
axis.plot(self.x, self.y, *args, **kwargs)
"""
        if axis == None:
            plt.plot(self.x, self.y, *args, **kwargs)
        else:
            axis.plot(self.x, self.y, *args, **kwargs)


def sum_data(data_list):
    """
Preforms the sum of the y data a set of Data class objects.
Args:
    data_list (list of Data): List of Data objects to sum together.
Returns:
    A Data object which is the sum of the y values of the original
        data sets.
"""
    total = data_list[0]
    for data_set in data_list[1:]:
        total += data_set.y
    return total


def mean(data_list):
    """
Preforms the mean of the y data a set of Data class objects.
Args:
    data_list (list of Data): List of Data objects to combine together.
Returns:
    A Data object which is the average of the y values of the original
        data sets.
"""
    return sum_data(data_list)/len(data_list)