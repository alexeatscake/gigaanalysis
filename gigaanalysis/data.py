
"""GigaAnalysis - Data Type - :mod:`gigaanalysis.data`
--------------------------------------------------------

This one module is imported directly into the :mod:`gigaanalysis` namespace,
so that the classes and functions here can be accessed directly.

This holds the :class:`Data` class and the functions that will manipulate 
them. This forms the backbone of the rest of the GigaAnalysis. The point of 
the :class:`Data` object is to hold sweeps. These are data sets with one 
independent and one dependant variable, which are super common in 
experimental physics research. By assuming the data is of this type more 
assumptions and error checking can be facilitated, and this is what 
GigaAnalysis aims to take advantage of.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d  # Used often to interpolate values


def _pick_float_dtype(to_check):
    """Return np.complex128 for complex dtypes, np.float64 otherwise.
    Adapted from scipy.interpolate"""
    if isinstance(to_check, np.ndarray):
        dtype = to_check.dtype
    else:
        dtype = type(to_check)
    if np.issubdtype(dtype, np.complexfloating):
        return np.complex_
    else:
        return np.float_


def _as_float_array(x):
    """Convert the input into a C contiguous float array.
    Adapted from scipy.interpolate
    NB: Upcasts half- and single-precision floats to double precision.
    """
    x = np.ascontiguousarray(x)
    x = x.astype(_pick_float_dtype(x), copy=False)
    return x


class Data():
    """
    The Data Class

    Data object holds the data in the measurements. It works as a simple
    wrapper of a two column numpy array (:class:`numpy.ndarray`). The data 
    stored in the object is meant to be interpreted as x is a independent 
    variable and y is dependant variable.


    Parameters
    ----------
    values : numpy.ndarray 
        A two column numpy array with the x data in the first column and 
        the y data in the second. If a second no array is given then the 
        first corresponds to the x data.
    split_y : numpy.ndarray, optional
        A 1D numpy array containing the y data. If None all the data 
        should be contained in first array.
    strip_sort : bool or {'strip', 'sort'}, optional
        If true the data points with NaN are removed using 
        :func:`numpy.isfinite` and the data is sorted by the x values.
        If 'strip' is given NaNs are removed but the data isn't sorted.
        If 'sort' is given the data is sorted but NaNs are left in.
        Default is False so the data isn't changed.
    interp_full : float, optional
        This interpolates the data to give an even spacing using the 
        inbuilt method :meth:`to_even`. The default is None and the 
        interpolation isn't done.

    Attributes
    ----------
    values : numpy.ndarray
        Two column numpy array with the x and y data in.
    x : numpy.ndarray
        x data in a 1D numpy array.
    y : numpy.ndarray
        The y data in a 1D numpy array.
    both : (numpy.ndarray, numpy.ndarray)
        A two value tuple with the :attr:`x` and :attr:`y` in.

    Notes
    -----
    Mathematical operations applied to the Data class just effects 
    the :attr:`y` values, the :attr:`x` values stay the same. To 
    act two :class:`Data` objects together the :attr:`x` values need to 
    agree. :class:`Data` objects also be mathematically acted to 
    array_like objects (:func:`numpy.asarray`) of length 1 or equal to the 
    length of the Data.

    """
    def __init__(self, values, split_y=None, strip_sort=False,
            interp_full=None):
        # Set up Class
        if isinstance(values, Data):
            values = values.values  # If you pass a Data object to the class

        values = np.asarray(values)

        if split_y is not None:
            split_y = np.asarray(split_y)
            if values.ndim != 1:
                raise ValueError(
                    f"If x and y data are split both need to be a "
                    f"1D numpy array. values has shape {values.shape}")
            elif split_y.ndim != 1:
                raise ValueError(
                    f"If x and y data are split both need to be a "
                    f"1D numpy array. split_y has shape {split_y.shape}")
            elif values.size != split_y.size:
                raise ValueError(
                    f"If x and y data are split both need to be the same "
                    f"size. values has size {values.size} and split_y has "
                    f"size {split_y.size}")
            values = np.concatenate(
                [values[:, None], split_y[:, None]], axis=1)

        if values.ndim != 2:
            raise ValueError(
                f"values needs to be a two column numpy array."
                f"values has the shape {values.shape}")
        elif values.shape[1] != 2:
            raise ValueError(
                f"values needs to be a two column numpy array."
                f"values has the shape {values.shape}")

        if strip_sort:
            if strip_sort == 'strip':
                values = values[np.isfinite(values).all(axis=1)]
            elif strip_sort == 'sort':
                values = values[np.argsort(values[:, 0]), :]
            else:
                values = values[np.isfinite(values).all(axis=1)]
                values = values[np.argsort(values[:, 0]), :]


        # all data in hidden attribute
        self.__values = _as_float_array(values)

        if interp_full is not None:
            self.to_even(interp_full)


    # Set up the attributes
    __slots__ = ("__values",)

    def __attribute_set(self, value):
        raise AttributeError(
            f"Can't set the attributes of a Data object directly. "
            f"Use .set_x, .set_y, .set_data functions.")

    def values(self):
        return self.__values

    values = property(values, __attribute_set, None,
        "Two column numpy array with the x and y data in.")

    def x(self):
        return self.__values[:, 0]

    x = property(x, __attribute_set, None,
        "x data in a 1D numpy array.")

    def y(self):
        return self.__values[:, 1]

    y = property(y, __attribute_set, None,
        "y data in a 1D numpy array.")

    def both(self):
        return self.__values[:, 0], self.__values[:, 1]

    both = property(both, __attribute_set, None,
        "A two value tuple with the :attr:`x` and :attr:`y` in.")

    # standard python methods
    def __str__(self):
        return np.array2string(self.values)

    def __repr__(self):
        return f"GA Data object:\n {str(self.values)[1:-1]}"

    def __len__(self):
        return self.x.size

    def __bool__(self):
        if self.values.size == 0:
            return False
        else:
            return True

    __array_ufunc__ = None
    # This is so that the user need to specify .x or .y when acting 
    # with numpy functions.


    # For mathematical operations
    def __maths_check(self, other, operation,):
        """This performs the error checking on the standard operators

        Parameters
        ----------
        other : :class:`Data` or array_like
            The feature that the data object maths acts on.
        operation : str
            The name of the operation being applied.

        Returns
        -------
            Array like object to calculate with
        """
        if isinstance(other, Data):
            if np.array_equal(self.x, other.x):
                return other.y
            else:
                raise ValueError(
                    f"The two Data classes do not have the same x "
                    f"values, so cannot be {operation}")
        try:
            other = np.asarray(other, dtype=_pick_float_dtype(other))
        except:
            raise TypeError(
                f"Data cannot be {operation} with object of "
                f"type {type(other)}.")
        if other.size == 1:
            return other
        elif other.ndim != 1:
            raise ValueError(
                f"Array to {operation} Data object with is of the wrong "
                f"dimension. Its shape is {other.shape}")
        elif other.size != self.x.size:
            raise ValueError(
                f"Array to {operation} Data object with is of the wrong "
                f"length. Its length is {other.size} while the Data "
                f"is {self.x.size}")
        else:
            return other

    def __mul__(self, other):
        """Multiplication of the y values. """
        other = self.__maths_check(other, "multiplied")
        return Data(self.x, self.y*other)
     
    def __rmul__(self, other):
        other = self.__maths_check(other, "multiplied")
        return Data(self.x, other*self.y)

    def __truediv__(self, other):
        """Division of the y values."""
        other = self.__maths_check(other, "divided")
        return Data(self.x, self.y/other)

    def __rtruediv__(self, other):
        other = self.__maths_check(other, "divide by")
        return Data(self.x, other/self.y)

    def __add__(self, other):
        """Addition of the y values."""
        other = self.__maths_check(other, "added")
        return Data(self.x, self.y + other)

    def __radd__(self, other):
        other = self.__maths_check(other, "added")
        return Data(self.x, other + self.y)

    def __sub__(self, other):
        """Subtraction of the y values."""
        other = self.__maths_check(other, "subtracted")
        return Data(self.x, self.y - other)

    def __rsub__(self, other):
        other = self.__maths_check(other, "subtracted")
        return Data(self.x, other - self.y)

    def __mod__(self, other):
        """Performs the modulus with the y values."""
        other = self.__maths_check(other, "divided mod")
        return Data(self.x, self.y % other)

    def __rmod__(self, other):
        other = self.__maths_check(other, "divided mod")
        return Data(self.x, other % self.y)

    def __floordiv__(self, other):
        """Floor division on the y values."""
        other = self.__maths_check(other, "floor division")
        return Data(self.x, self.y // other)

    def __rfloordiv__(self, other):
        other = self.__maths_check(other, "floor division")
        return Data(self.x, other // self.y)

    def __pow__(self, other):
        """Takes the power of the y values."""
        other  = self.__maths_check(other, "exponentiated")
        return Data(self.x, self.y ** other)

    def __rpow__(self, other):
        other = self.__maths_check(other, "exponentiated")
        return Data(self.x, other ** self.y)

    def __abs__(self):
        """Calculates the absolute value of the y values.
        """
        return Data(self.x, abs(self.y))

    def __neg__(self):
        """Negates the y values"""
        return Data(self.x, -self.y)

    def __pos__(self):
        """Performs a unity operation on y values"""
        return Data(self.x, self.y)

    def __eq__(self, other):
        """Data class is only equal to other Data classes with the same data.
        """
        if type(other) != type(self):
            return False
        else:
            return np.array_equal(self.values, other.values)

    def __iter__(self):
        """The iteration happens on the values, like if was numpy array.
        """
        return iter(self.values)

    # For indexing behaviour
    def __index_check(self, k):
        """Check an index given if it is correct type and size.
        
        Raises errors if it is the wrong type or shape. Also returns a bool
        which is true if only one item is called.

        Parameters
        ----------
        k : slice or can be passed to :func:slice
            A object obtained from index calls

        Returns
        -------
        individual : bool
            Is the index call only for one item?
        """
        if isinstance(k, tuple):
            raise IndexError(
                "Data object only accepts one index")
        elif isinstance(k, slice):
            return False
        try:
            k = np.asarray(k)
        except:
            raise IndexError(
                "Data cannot index with this type.")
        if k.size == 1:
            return True
        elif k.ndim != 1:
            raise IndexError(
                "Data objec can only Index is one dimension.")
        elif k.dtype == np.int_:
            return False
        elif k.size != self.x.size:
            raise IndexError(
                f"Index given was wrong length. The length of index was "
                f"{k.size} and the Data is length {self.x.size}")
        else:
            return False

    def __getitem__(self, k):
        """Indexing returns a subset of the Data object.

        If given a slice or and array of boolean a new Data object is 
        produced. If given a int a length two array with [x, y] is returned. 
        """
        if self.__index_check(k):
            return self.values[k]
        else:
            return Data(self.values[k])

    def __setitem__(self, k, v):
        """Item assignment is not allowed in Data objects.

        This kind of action is possible with the functions :meth:`set_x`, 
        :meth:`set_y`, and :meth:`set_data`.
        """
        raise Warning(
            "Data objects do not allow item assignment. For this "
            "functionality see .set_x, .set_y, and .set_data.")

    def set_x(self, idx, val):
        """This is used for setting x values.

        Works similarly to ``Data.x[idx] = val`` but with more error 
        checking. The previous code would also work (and be faster) but 
        more care should be taken. The built in function 
        :func:`slice(start, end, step)` maybe useful.

        Parameters
        ----------
        idx : slice, int
            Objects that can be passed to a :class:`numpy.ndarray` as 
            an index.
        val : numpy.ndarray
            The values to assign to the indexed x values. 
        """
        if isinstance(val, Data):
            raise TypeError(
                "Cannot set the object type with a Data object.")
        self.__index_check(idx)
        new_x = self.x
        new_x[idx] = val
        self.__init__(new_x, self.y)

    def set_y(self, idx, val):
        """This is used for setting y values.

        Works similarly to ``Data.y[idx] = val`` but with more error 
        checking. The previous code would also work (and be faster) but 
        more care should be taken. The built in function 
        :func:`slice(start, end, step)` maybe useful.

        Parameters
        ----------
        idx : slice, int
            Objects that can be passed to a :class:`numpy.ndarray` as 
            an index.
        val : numpy.ndarray
            The values to assign to the indexed y values. 
        """
        if isinstance(val, Data):
            raise TypeError(
                "Cannot set the object type with a Data object.")
        self.__index_check(idx)
        new_y = self.y
        new_y[idx] = val
        self.__init__(self.x, new_y)

    def set_data(self, idx, val):
        """This is used for setting x and y values.

        Works similarly to ``Data.values[idx] = val`` but with more error 
        checking. The previous code would also work (and be faster) but 
        more care should be taken. The built in function 
        :func:`slice(start, end, step)` maybe useful.

        Parameters
        ----------
        idx : slice, int
            Objects that can be passed to a :class:`numpy.ndarray` as 
            an index.
        val : numpy.ndarray, Data
            The values to assign to the indexed values. This can only be a
            two column :class:`numpy.ndarray` or a :class:`Data` object.
        """
        if self.__index_check(idx):
            size = 2
        else:
            size = self.values[idx].size
        if not isinstance(val, (Data, np.ndarray)):
            raise TypeError(
                f"The value to assign data must be a data object or a two "
                f"column numpy array. The type give was {type(val)}.")
        elif isinstance(val, Data):
            if size != val.values.size:
                raise ValueError(
                    f"The Data to set is a different size to the Data "
                    f"object given. The size to index was {size/2} "
                    f"while the data to set was {val.values.size/2}.")
            else:
                new_data = self.values
                new_data[idx] = val.values
        elif val.ndim != 2:
            raise ValueError(
                f"The dimension of the numpy array to set to is not "
                f"the correct shape. Needs to be a two column array shape "
                f"given was {val.shape}.")
        elif val.shape[1] != 2:
            raise ValueError(
                f"The dimension of the numpy array to set to does not "
                f"have two columns. Needs to be a two column array shape "
                f"given was {val.shape}.")
        elif val.size != size:
            raise ValueError(
                f"The Data to set is a different size to the numpy "
                f"array given. The size to index was {size/2} "
                f"while the data to set was {val.size/2}.")
        else:
            new_data = self.values
            new_data[idx] = val
        self.__init__(new_data)

    # Simple useful methods
    def strip_nan(self):
        """This removes any row which has a nan or infinite values in.
        
        Returns
        -------
        stripped_data : Data
            Data class without non-finite values in.
        """
        return Data(self.values[np.isfinite(self.values).all(axis=1)])

    def sort(self):
        """Sorts the data set in x and returns the new array.

        Returns
        -------
        sorted_data : Data
            A Data class with the sorted data.
        """
        return Data(self.values[np.argsort(self.x), :])

    def min_x(self):
        """This provides the lowest value of x

        Returns
        -------
        x_min : float
            The minimum value of x
        """
        return np.min(self.x)

    def max_x(self):
        """This provides the highest value of x

        Returns
        -------
        x_max : float
            The maximum value of x
        """
        return np.max(self.x)

    def spacing_x(self):
        """Returns the average spacing in x

        Returns
        -------
        x_max : float
            The average spacing in the x data
        """
        return (self.max_x() - self.min_x())/len(self)

    def x_cut(self, x_min, x_max):
        """This cuts the data to a region between x_min and x_max.
        
        Parameters
        ----------    
        x_min : float
            The minimal x value to cut the data.
        x_max : float
            The maximal x value to cut the data.
        
        Returns
        -------
        cut_data : Data    
            A data object with the values cut to the given x range.
        """
        if x_min > x_max:
            raise ValueError('x_min should be smaller than x_max')
        return Data(self.values[(self.x > x_min) & (self.x < x_max)])

    def y_from_x(self, x_val, bounds_error=True, kind='linear'):
        """Gives the y value for a certain x value or set of x values.

        Parameters
        ----------
        x_val : float
            X values to interpolate y values from.
        bounds_error : bool, optional
            If an error should thrown in x value is out of range, 
            default True.
        kind : str or int, optional
            The type of interpolation to use. Passed to 
            :func:`scipy.interpolate.interp1d`, default is `linear`.
        
        Returns
        -------
        y_val : float or numpy.ndarray
            Corresponding to the requested x values in an array if only one 
            value is given a float is returned.
        """
        if bounds_error and \
            (np.max(x_val)>self.max_x() or np.min(x_val)<self.min_x()):
            raise ValueError(
                f"The given x_values are out side of the range of data "
                f"which is between {self.min_x()} and {self.max_x()}")

        y_val = interp1d(self.x, self.y, bounds_error=False,
            fill_value=(self.y.min(), self.y.max()), kind=kind)(x_val)
        if y_val.size != 1:
            return y_val
        else:
            return float(y_val)

    def apply_x(self, function):
        """This takes a function and applies it to the x values.

        Parameters
        ----------
        function : Callable
            The function to apply to the x values.
        
        Returns
        -------
        transformed_data
            Data class with new x values.
        """
        return Data(function(self.x), self.y)

    def apply_y(self, function):
        """This takes a function and applies it to the y values.

        Parameters
        ----------
        function : Callable
            The function to apply to the y values.
        
        Returns
        -------
        transformed_data
            Data class with new y values.
        """
        return Data(self.x, function(self.y))

    def append(self, new_data, in_place=False):
        """This adds values to the end of the data object.

        Parameters
        ----------
        new_data : Data
            These are the values to add onto the end of the data object
        in_place : bool, optional
            Weather to edit the object or to return a new one. The default 
            is `False` which returns a new object.

        Returns
        -------
        combined_data : Data
            If in_place is `False` then a new Data object is returned.
        """
        if isinstance(new_data, Data):
            pass
        else:
            try:
                new_data = Data(new_data)
            except:
                raise ValueError(
                    f"The new_data to append was not a Data object or "
                    f"could be cast to one. Was of type {type(new_data)}")
        new_vals = np.append(self.values, new_data.values, axis=0)
        if in_place:
            self.__init__(new_vals)
        else:
            return Data(new_vals)

    # Methods for Interpolation of Data 
    def interp_range(self, min_x, max_x, 
        step_size=None, num_points=None, shift_step=True,
        kind='linear'):
        """Evenly interpolates in x the data between a min and max value.

        This is used for combining datasets with corresponding but different 
        x values. Either `step_size` or `num_points` can be defined. If 
        `step_size` is defined :func:`numpy.arange` is used. If `num_points` 
        is defined :func:`numpy.linspace` is used.
        If using `step_size` it rounds `min_x` to the next integer value of 
        the steps, unless `shift_step` is `False`.

        If values outside the range of the original data need to be 
        passed to be interpolated, this is possible with 
        :func:`Data.interp_values`.
        It uses :func:`scipy.interpolate.interp1d`.
        
        Parameters
        ----------
        min_x :float
            The minimum x value in the interpolation.
        max_y : float
            The maximum x value in the interpolation.
        step_size : float, optional
            The step size between each point. Either this or num_points must 
            be defined.
        num_points : int, optional
            The number of points to interpolate. Either this or step_size 
            must be defined.
        shift_step: bool, optional
            If the `min_x` value should be rounded to the next whole step. 
            The default is True.
        kind : str or int, optional
            The type of interpolation to use. Passed to 
            :func:`scipy.interpolate.interp1d`, default is `linear`.

        
        Returns
        -------
        interpolated_data : Data
            A Data object with evenly interpolated points.
        """
        if step_size is None and num_points is None:
            raise ValueError(
                f"Must define either step_size or num_points.")
        if min_x > max_x:
            min_x, max_x = max_x, min_x  # order min and max
        if np.min(self.x) > min_x:
            raise ValueError("min_x value to interpolate is below data")
        if np.max(self.x) < max_x:
            raise ValueError("max_x value to interpolate is above data")
        if step_size is not None:
            if shift_step:
                min_x = np.ceil(min_x/step_size)*step_size
            x_vals = np.arange(min_x, max_x, step_size)
        elif num_points is not None:
            x_vals = np.linspace(min_x, max_x, num_points)
        # The bounds are used in the rare case of floating point issues.
        min_y = self.y[self.x.argmin()]
        max_y = self.y[self.x.argmax()]
        y_vals = interp1d(self.x, self.y, kind=kind,
            bounds_error=False, fill_value=(min_y, max_y))(x_vals)
        return Data(x_vals, y_vals)

    def interp_step(self, step_size, shift_step=True, kind='linear'):
        """Evenly interpolates in x the data between a min and max value.

        This uses :meth:`Data.interp_range` specifying `step_size` and 
        giving the maximum range of x points.
        If using `step_size` it rounds `min_x` to the next integer value of 
        the steps, unless `shift_step` is `False`.
        
        Parameters
        ----------
        step_size : float, optional
            The step size between each point. Either this or num_points must 
            be defined.
        shift_step: bool, optional
            If the `min_x` value should be rounded to the next whole step. 
            The default is True.
        kind : str or int, optional
            The type of interpolation to use. Passed to 
            :func:`scipy.interpolate.interp1d`, default is `linear`.
        
        Returns
        -------
        interpolated_data : Data
            A Data object with evenly interpolated points.
        """
        return self.interp_range(self.min_x(), self.max_x(),
            step_size=step_size, shift_step=shift_step, kind=kind,)

    def interp_number(self, num_points, kind='linear'):
        """Evenly interpolates in x the data for a fixed point number.

        This uses :meth:`Data.interp_range` specifying `num_points` and 
        giving the maximum range of x points.
        
        Parameters
        ----------
        num_points : int
            The number of points to interpolate.
        kind : str or int, optional
            The type of interpolation to use. Passed to 
            :func:`scipy.interpolate.interp1d`, default is `linear`.
        
        Returns
        -------
        interpolated_data : Data
            A Data object with evenly interpolated points.
        """
        return self.interp_range(self.min_x(), self.max_x(),
            num_points=num_points, kind=kind,)

    def interp_values(self, x_vals, kind='linear', bounds_error=True,
            fill_value=np.nan, strip_sort=False):
        """Produce Data object from interpolating x values.

        This uses :func:`scipy.interpolate.interp1d` to produce a Data 
        object by interpolating y values from given x values.

        Parameters
        ----------
        x_vals : array_like
            The x values to interpolate which will be the x values.
        kind : str or int, optional
            The type of interpolation to use. Passed to 
            :func:`scipy.interpolate.interp1d`, default is `linear`.
        bounds_error : bool, optional
            If default of `True` data outside the existing range will throw 
            an error. If `False` then the value is set by `fill_value`.
        fill_value : float or (float, float) or `extrapolate`, optional
            If bounds_error is `False` then this value will be used outside 
            the range. Passed to :func:`scipy.interpolate.interp1d`.
        strip_sort : bool, optional
            The default is `False`, where to sort and remove NaNs from the 
            Data object before returning.

        Returns
        -------
        interpolated_data : Data
            A Data object with the given x values and interpolated y values.
        """
        x_vals  = np.asarray(x_vals)
        if x_vals.ndim != 1:
            raise ValueError(
                f"x_vals had shape {x_vals.shape} where as it need to be 1D")

        if bounds_error:
            if self.min_x() > np.min(x_vals):
                raise ValueError(
                    "min_x value to interpolate is below data and "
                    "bounds_error is True.")
            if self.max_x() < np.max(x_vals):
                raise ValueError(
                    "max_x value to interpolate is above data and "
                    "bounds_error is True")
            # The bounds are used in the rare case of floating point issues.
            min_y = self.y[self.x.argmin()]
            max_y = self.y[self.x.argmax()]
            y_vals = interp1d(self.x, self.y, kind=kind,
                bounds_error=False, fill_value=(min_y, max_y))(x_vals)
        else:
            y_vals = interp1d(self.x, self.y, kind=kind,
                bounds_error=False, fill_value=fill_value)(x_vals)

        return Data(x_vals, y_vals, strip_sort=strip_sort)

    def to_even(self, step_size, shift_step=True, kind='linear'):
        """Evenly interpolates the data and updates the data object.

        This uses :meth:`Data.interp_range` specifying `step_size` and 
        giving the maximum range of x points.
        If using `step_size` it rounds `min_x` to the next integer value of 
        the steps, unless `shift_step` is `False`.
        
        Parameters
        ----------
        step_size : float, optional
            The step size between each point. Either this or num_points must 
            be defined.
        shift_step: bool, optional
            If the `min_x` value should be rounded to the next whole step. 
            The default is True.
        kind : str or int, optional
            The type of interpolation to use. Passed to 
            :func:`scipy.interpolate.interp1d`, default is `linear`.
        """
        self.__init__(self.interp_range(self.min_x(), self.max_x(),
            step_size=step_size, shift_step=shift_step,
            kind=kind,).values)

    # Plotting Method
    def plot(self, *args, axis=None, **kwargs):
        """Simple plotting utility
        
        Makes use of matplotlib function :func:`matplotlib.pyplot.plot`.
        Runs ``matplotlib.pyplot.plot(self.x, self.y, *args, **kwargs)``
        If provided an axis keyword which operates so that if given
        ``axis.plot(self.x, self.y, *args, **kwargs)``.
        """
        if axis is None:
            plt.plot(self.x, self.y, *args, **kwargs)
        else:
            axis.plot(self.x, self.y, *args, **kwargs)

    # Saving Method
    def to_csv(self, filename, columns=["X", "Y"], **kwargs):
        """Saves the data as a simple csv

        Uses :func:`pandas.DataFrame.to_csv` and kwargs are pass to it. The 
        index keyword is set to False by default.

        Parameters
        ----------
        filename : str
            Filename to save the data as.
        columns : [str, str]
            The title of the two columns.
        """
        if 'index' not in kwargs:
            kwargs['index'] = False

        pd.DataFrame(self.values, columns=columns
            ).to_csv(filename, **kwargs)


def swap_xy(data, **kwargs):
    """Interchange the independent and dependent variables.
    
    This takes a :class:`.Data` object and returns a new one with the x and 
    y variables swapped around. Keyword arguments are pass to the 
    :class:`.Data` class.

    Parameters
    ----------
    data : Data
        The data to switch the x and y values.

    Returns
    -------
    swapped_data : Data
        A new :class:`.Data` object with x and y values switched.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data needs to be a Data object but was instead {type(data)}")

    return Data(data.y, data.x, **kwargs)


def empty_data():
    """Generates an empty :class:`.Data` object.

    This is useful for place holding, and takes no parameters.

    Returns
    -------
    empty_data : Data
        A Data object that contains no data points.
    """
    return Data(np.array([], dtype=np.float_).reshape(0, 2))


def sum_data(data_list):
    """Preforms the sum of the y data a set of Data class objects.
    
    Parameters
    ----------
    data_list : [Data]
        List of Data objects to sum together.
    
    Returns
    -------
    summed_data : Data
        A Data object with the summed y values of the original data sets.
    """
    if isinstance(data_list, list):
        pass
    elif isinstance(data_list, dict):
        data_list = list(data_list.values())
    elif isinstance(data_list, Data):
        return data_list
    else:
        raise TypeError(
            f"The data_list was of type {type(data_list)} where as it "
            f"needs to be either a list or a dict with Data objects as the "
            f"values.")
    if not isinstance(data_list[0], Data):
            raise TypeError(
                f"List contained type {type(data_list[0])} where is must "
                f"only contain gigaanalysis.data.Data types.")
    if len(data_list) == 1:
        return data_list[0]
    total = data_list[0]
    for dat in data_list[1:]:
        if not isinstance(dat, Data):
            raise TypeError(
                f"List contained type {type(dat)} where is must "
                f"only contain gigaanalysis.data.Data types.")
        total += dat.y
    return total


def mean(data_list):
    """Preforms the mean of the y data a set of Data class objects.
    
    Parameters
    ----------
    data_list : [Data]
        List of Data objects to sum together can also be a dictionary.
    
    Returns
    -------
    averaged_data : Data
        A Data object with the summed y values of the original data sets.
    """
    return sum_data(data_list)/len(data_list)


def _fit_one_y(data, x_value, x_range, poly_deg, std=False):
    """A function used by :func:`y_from_fit` that calculates the y value 
    from one x value.
    """
    xs, ys = data.x_cut(x_value - x_range/2, x_value + x_range/2).both
    xs = xs - x_value
    if len(xs) + 1 <= poly_deg:
        raise ValueError(
            f"There was only {len(xs)} in the provided range which is not "
            f"enough to fit a {poly_deg} order polynomial.")
    if std:
        if std == 'fit':
            val, err = np.polyfit(xs, ys, poly_deg, cov=True)
            err = np.sqrt(err[-1, -1])
        elif std == 'residual':
            val = np.polyfit(xs, ys, poly_deg)
            y_res = ys.copy()
            for n in range(len(val)):
                y_res = y_res - (xs**n)*val[-n-1]
            err = np.std(y_res)
        else:
            raise ValueError("std was not either 'fit' or 'residual'.")
        return val[-1], err
    else:
        return np.polyfit(xs, ys, poly_deg)[-1]


def y_from_fit(data, x_value, x_range, poly_deg=1, as_Data=False, 
        std=False):
    """Fits a polynomial over a range to interpolate a given value.

    This makes use of :func:`numpy.polyfit` to find an interpolated value of 
    y form a data object and a given x value.

    Parameters
    ----------
    data : Data
        The data to interpolate the value from. Should be a sorted data 
        object.
    x_value : float or numpy.ndarray
        The value of the independent to obtain the associated dependent 
        variable.
    x_range : float
        The range of independent variables to perform the fit over.
    poly_deg : int, optional
        The order of the polynomial to use when fitting to find the result. 
        The default is `1` which is a linear fit.
    as_Data : bool, optional
        If default of False y values are given as an float or an array. If 
        `True` then a Data object is returned.
    std : bool, optional
        If `fit` or 'residual' then the standard deviation is returned after 
        the values. The standard deviation can either be calculated from the 
        error in the fit (using 'fit') or from the distribution of the 
        residuals of the fit (using 'residual'). The default value is 
        `False`, where only the value will be returned.

    Returns
    -------
    y_value : float, numpy.ndarray, or Data
        The y values obtained at the associated value of x for the fit 
        performed. The type depends if multiple points are requested and if 
        'as_Data` is set. If `std` is `True` then the standard deviation is 
        followed in the same format.
    """
    if not isinstance(data, Data):
        raise TypeError(
            f"data needs to be a Data object but was a {type(data)}.")
    elif not isinstance(x_range, (int, float, np.int_, np.float_)):
        raise TypeError(
            f"x_range needs to be a float but was a {type(x_range)}")
    elif not isinstance(poly_deg, (int, np.int_)):
        raise TypeError(
            f"poly_deg needs to be a int but was a {type(poly_deg)}")

    x_value = np.asarray(x_value)
    if x_value.dtype != np.float_ and x_value.dtype != np.int_:
        raise TypeError(
            f"x_value needs to be of float type but was a {x_value.dtype}")
    elif x_value.ndim > 1:
        raise TypeError(
            f"x_value can a float or a 1D array like of floats but was of "
            f"shape {x_value.shape}")

    if std:
        if std not in ['fit', 'residual']:
            raise ValueError("std must either False or be 'fit' or "
                f"'residual' but was {std}")

    if x_value.size == 1:
        if not as_Data:
            return _fit_one_y(data, x_value, x_range, poly_deg, std=std)
        else:
            if std:
                val, err = _fit_one_y(data, x_value, x_range, poly_deg, 
                    std=std)
                return Data([[x_value, val]]), Data([[x_value, err]])
            else:
                return Data([[x_value, 
                    _fit_one_y(data, x_value, x_range, poly_deg)]])
    elif as_Data:
        if std:
            val, err = np.array(
                [_fit_one_y(data, xv, x_range, poly_deg, std=std) \
                    for xv in x_value]).T
            return Data(x_value, val), Data(x_value, err)
        else:
            return Data(x_value, np.array(
                [_fit_one_y(data, xv, x_range, poly_deg) for xv in x_value]))
    else:
        if std:
            val, err = np.array(
                [_fit_one_y(data, xv, x_range, poly_deg, std=std) \
                    for xv in x_value]).T
            return val, err
        else:
            return np.array([_fit_one_y(data, xv, x_range, poly_deg) \
                for xv in x_value])


def collect_y_values(data_list):
    """Collates the y values into a array from a collection of Data objects.

    This takes either a list or dictionary of Data objects and collects the
    y values into one array. This can be useful of special comparisons such 
    as trimmed means and standard deviations.

    Parameters
    ----------
    data_list : list or dict
        A list of Data objects or a dictionary where the values are Data 
        objects. The x values or all of these need to be the same.

    Returns
    -------
    x_vals : numpy.ndarray
        One copy of the x values of the arrays.
    all_data : numpy.ndarray
        All the y data from the different data objects each on in it's own
        column.
    """
    if isinstance(data_list, list):
        pass
    elif isinstance(data_list, dict):
        data_list = list(data_list.values())
    elif isinstance(data_list, Data):
        return data_list.y[:, None]
    else:
        raise TypeError(
            f"The data_list was of type {type(data_list)} where as it "
            f"needs to be either a list or a dict with Data objects as the "
            f"values.")
    if not isinstance(data_list[0], Data):
            raise TypeError(
                f"List contained type {type(data_list[0])} where is must "
                f"only contain gigaanalysis.data.Data types.")
    if len(data_list) == 1:
        return data_list[0].y[:, None]
    all_data = data_list[0].y[:, None]
    x_vals = data_list[0].x
    for dat in data_list[1:]:
        if not isinstance(dat, Data):
            raise TypeError(
                f"List contained type {type(dat)} where is must "
                f"only contain gigaanalysis.data.Data types.")
        elif not np.array_equal(dat.x, x_vals):
            raise TypeError(
                f"The x values in the arrays do not match.")
        all_data = np.concatenate([all_data, dat.y[:, None]], axis=1)
    return x_vals, all_data


def __make_x_vals(min_x, max_x, 
    step_size=None, num_points=None, shift_step=True):
    """This generates a set of evenly spaced x values.
    """
    if step_size is not None:
        if shift_step:
            min_x = np.ceil(min_x/step_size)*step_size
        x_vals = np.arange(min_x, max_x, step_size)
    elif num_points is not None:
        x_vals = np.linspace(min_x, max_x, num_points)
    else:
        raise ValueError(
            f"Must define either step_size or num_points.")
    return x_vals


def __match_x_list(data_list,
    step_size=None, num_points=None, shift_step=True):
    """This interpolates all the data sets in a list to the same x values.

    It takes the values from :func:`match_x` and each set to  
    :func:`__match_x_list`, after working out the largest range of x_values
    that can be interpolated across every dataset.
    """
    min_x = -np.inf
    max_x = np.inf
    max_len = 0
    if not isinstance(data_list, list):
        raise TypeError(
            f"data_list need to be list it was a {type(data_list)}")
    for dat in data_list:
        if not isinstance(dat, Data):
            raise TypeError(
                f"data_list needs to be a list of Data objects but "
                f"contained the type {type(dat)}.")
        min_x = min_x if min_x > dat.min_x() else dat.min_x()
        max_x = max_x if max_x < dat.max_x() else dat.max_x()
        max_len = max_len if max_len > len(dat) else len(dat)
    if step_size is None and num_points is None:
        num_points = max_len
    x_vals = __make_x_vals(min_x, max_x, step_size=step_size,
        num_points=num_points, shift_step=shift_step)
    new_data_list = [
        dat.interp_values(x_vals)
        for dat in data_list
    ]
    return new_data_list
        

def match_x(data_list,
    step_size=None, num_points=None, shift_step=True):
    """This transform a collection of dataset to have the same x values.

    This applies :meth:`Data.interp_values` to every data object with the 
    largest possible range of x values to produce the new set of data. This 
    is useful if the data object want to be combined arithmetically.

    Parameters
    ----------
    data_list : list or dict of Data
        A list of data objects or dictionary with data objects as the values.
    step_size : float, optional
        Sets the spacing in the x values to a fixed amount if given  
        :func:`numpy.arange` is called.
    num_points : int, optional
        The number of points to generates for the x values only used if 
        `step_size` is not given or None. If used :func:`numpy.linspace` 
        is called.
    shift_step : bool, optional 
        Only valid if step_size is not new. The default is True and then 
        the first value is an integer number of the steps. If False the 
        lowest x value is used as the first value of the step.

    Returns
    -------
    new_data_list : list or dict of Data objects
        The new data objects with the x values that are interoperated to be 
        all the same. If a dict is provided a dict is returned with the 
        same keys as before.
    """
    if isinstance(data_list, Data):
        return __match_x_list([data_list], step_size=step_size,
            num_points=num_points, shift_step=shift_step)
    elif isinstance(data_list, list):
        return __match_x_list(data_list, step_size=step_size,
            num_points=num_points, shift_step=shift_step)
    elif isinstance(data_list, dict):
        key, vals = zip(*data_list.items())
        new_vals = __match_x_list(list(vals), step_size=step_size,
            num_points=num_points, shift_step=shift_step)
        return dict(zip(key, new_vals))
    elif isinstance(data_list, tuple):
         return __match_x_list(list(data_list), step_size=step_size,
            num_points=num_points, shift_step=shift_step)
    else:
        raise TypeError(
            f"data_list need to be either a list or a dictionary "
            f"but was of the type {type(data_list)}.")


def __interp_list(data_list, x_vals, kind='linear'):
    """Interpolates all the Data objects in a list. Is used by `interp_set`.
    """
    new_list = []
    for dat in data_list:
        if not isinstance(dat, Data):
            raise TypeError(
                f"One of the objects in the list was not a Data "
                f"object. It was of type {type(dat)}")
        new_list.append(dat.interp_values(x_vals, kind=kind))
    return new_list


def interp_set(data_list, x_vals, kind='linear'):
    """Interpolates all Data objects in list or dictionary.

    This applied :meth:`Data.interp_values` to every item in the set and 
    returns a new set.

    Parameters
    ----------
    data_list : list or dict
        A list or dictionary of Data objects to interpolate.
    x_vals : :class:`numpy.ndarray`
        The x values to interpolate to produce the new set.
    kind : str or int, optional
        The type of interpolation to use. Passed to 
        :func:`scipy.interpolate.interp1d`, default is `linear`.

    Returns
    -------
    interpolated_set : list or dict
        The new set of Data is the same form but with interpolated values.
    """
    if isinstance(data_list, Data):
        return [data_list.interp_values(x_vals, kind=kind)]
    elif isinstance(data_list, list):
        return __interp_list(data_list, x_vals, kind=kind)
    elif isinstance(data_list, dict):
        key, vals = zip(*data_list.items())
        return dict(zip(key, __interp_list(vals, x_vals, kind=kind)))
    elif isinstance(data_list, tuple):
        return __interp_list(list(data_list), x_vals, kind=kind)
    else:
        raise TypeError(
            f"The data_list needs to be a dictionary or a list but was "
            f"instead of type {type(data_list)}.")


def concatenate(data_list, strip_sort=False):
    """Combines our collection of Data objects into one.

    This takes either a list, dictionary, or tuple of Data objects or arrays 
    and concatenates their values into one data object.
    This makes use of :func:`numpy.concatenate`.

    Parameters
    ----------
    data_list : list or dict
        The collection of Data objects to combine.
    strip_sort : bool or {'strip', 'sort'}, optional
        This will pass to the strip_sort keyword argument when producing 
        the final Data object.

    Returns
    -------
    concatenated_data : Data
        The data combined into one Data object.
    """
    if isinstance(data_list, Data):
        if strip_sort:
            return Data(data_list, strip_sort=strip_sort)
        else:
            return data_list
    elif isinstance(data_list, list):
        pass
    elif isinstance(data_list, dict):
        data_list = list(data_list.values())
    elif isinstance(data_list, tuple):
         data_list = list(data_list)
    else:
        raise TypeError(
            f"data_list need to be either a list or a dictionary "
            f"but was of the type {type(data_list)}.")
    all_vals = []
    for dat in data_list:
        if isinstance(dat, Data):
            all_vals.append(dat.values)
        elif isinstance(dat, np.ndarray):
            if len(dat.shape) == 2 and dat.shape[1] == 2:
                all_vals.append(dat)
            else:
                raise ValueError(
                    f"The values to concatenate in the form of a "
                    f"numpy array are the wrong shape {dat.shape}")
        elif isinstance(dat, list) and len(dat) == 2:
            all_vals.append(np.asarray([dat]))
        else:
            raise TypeError(
                f"The list contains objects which are not Data objects "
                f"one of the objects was a {type(dat)}")
    return Data(np.concatenate(all_vals, axis=0), strip_sort=strip_sort)


def save_arrays(array_list, column_names, file_name, **kwargs):
    """Writes a list of arrays to csv.

    This saves a collection of one dimensional :class:`numpy.ndarray` 
    stored in a list into a .csv file. It does this by passing it to a 
    :class:`pandas.DataFrame` object and using the method `to_csv`. If the 
    arrays are different lengths the values are padded with NaNs.
    kwargs are passed to :meth:`pandas.DataFrame.to_csv`.

    Parameters
    ----------
    array_list : [numpy.ndarray]
        A list of 1d numpy.ndarrays to save to the .csv file.
    columns_names : [str]
        A list of column names for the .csv file the same length as the list 
        of data arrays.
    file_name : str
        The file name to save the file as.
    """
    if not isinstance(array_list, list):
        raise TypeError("array_list is not a list.")
    elif not isinstance(column_names, list):
        raise TypeError("column_names is not a list.")
    elif len(array_list) != len(column_names):
        raise ValueError("array_list and column_names are not "
            "the same lenght.")
    max_length = 0
    for arr in array_list:
        if not isinstance(arr, np.ndarray):
            raise ValueError("array_list contains objects that are not "
                "numpy arrays.")
        elif len(arr.shape) != 1:
            raise ValueError("array_list arrays are not 1D.")
        elif max_length < arr.size:
            max_length = arr.size
    to_concat = []
    for arr in array_list:
        to_concat.append(np.pad(arr, (0, max_length - arr.size),
            mode='constant',
            constant_values=np.nan)[:, None])
    to_save = np.concatenate(to_concat, axis=1)
    if 'index' not in kwargs.keys():
        kwargs['index'] = False
    pd.DataFrame(to_save, columns=column_names).to_csv(file_name, **kwargs)


def save_data(data_list, data_names, file_name, 
    x_name='X', y_name='Y', name_space='/', no_sapce=True, **kwargs):
    """Saves a list of data objects in to a .csv file.

    This works by passing to :func:`save_arrays` and subsequently to 
    :meth:`pandas.DataFrame.to_csv`. kwargs are passed to 
    :meth:`pandas.DataFrame.to_csv`

    Parameters
    ----------
    data_list : [gigaanalysis.data.Data]
        A list of Data objects to be saved to a .csv file
    data_names : [str]
        A list the same length as the data list of names of each of the data 
        objects. These will make the first half of the column name in the 
        .csv file.
    file_name : str
        The name the file will be saved as.
    x_name : str, optional
        The string to be append to the data name to indicate the x column in 
        the file. Default is 'X'.
    y_name : str, optional
        The string to be append to the data name to indicate the y column in 
        the file. Default is 'Y'.
    name_space : str, optional
        The string that separates the data_name and the x or y column name 
        in the column headers in the .csv file. The default is '/'.
    """
    if not isinstance(data_list, list):
        raise TypeError("data_list is not a list.")
    elif not isinstance(data_names, list):
        raise TypeError("data_names is not a list.")
    elif len(data_list) != len(data_names):
        raise ValueError("data_list and data_names are not "
            "the same lenght.")
    array_list = []
    for dat in data_list:
        if not isinstance(dat, Data):
            raise ValueError("data_list contains objects that are not "
                "Data objects.")
        array_list.append(dat.x)
        array_list.append(dat.y)
    column_names = []
    striping = ',' + name_space
    if no_sapce:
        striping += ' '
    x_name = str(x_name).strip(striping)
    y_name = str(y_name).strip(striping)
    for name in data_names:
        name = str(name).strip(striping)
        column_names.append(name + name_space + x_name)
        column_names.append(name + name_space + y_name)
    save_arrays(array_list, column_names, file_name, **kwargs)


def save_dict(data_dict, file_name,
    x_name='X', y_name='Y', name_space='/', **kwargs):
    """Saves a dictionary of data objects in to a .csv file.

    This works by passing to :func:`save_data` and subsequently to 
    :meth:`pandas.DataFrame.to_csv`. The names of the data objects are taken 
    from  the keys of the data_dict. kwargs are passed to 
    :meth:`pandas.DataFrame.to_csv`

    Parameters
    ----------
    data_list : [gigaanalysis.data.Data]
        A dictionary of Data objects to be saved to a .csv file. The keys of 
        the dictionary will be used as the data names when passed to 
        :func:`save_data`.
    file_name : str
        The name the file will be saved as.
    x_name : str, optional
        The string to be append to the data name to indicate the x column in 
        the file. Default is 'X'.
    y_name : str, optional
        The string to be append to the data name to indicate the y column in 
        the file. Default is 'Y'.
    name_space : str, optional
        The string that separates the data_name and the x or y column name 
        in the column headers in the .csv file. The default is '/'.
    """
    if not isinstance(data_dict, dict):
        raise TypeError("data_dict is not a dictionary.")
    for dat in data_dict.values():
        if not isinstance(dat, Data):
            raise ValueError("data_dict contains values which are not "
                "Data objects.")
    save_data(list(data_dict.values()), list(data_dict.keys()), file_name,
        x_name=x_name, y_name=y_name, name_space=name_space, **kwargs)


def load_dict(file_name, name_space='/',
    strip_sort=False, interp_full=None, **kwargs):
    """Loads from a file a dictionary full of Data objects.

    The type of file it loads is the default produced by :func:`save_dict`. 
    It assumes there is one line for the headers and they are used for the 
    keys of the dictionary. It also removes NaNs at the end of each sweep to 
    undo what is produced by uneven length of data objects.
    It makes use of :func:`pandas.read_csv`, and extra keyword arguments are 
    passed to there.

    Parameters
    ----------
    file_name : str
        The name and location of the file.
    name_space : str
        The string that separates the key from the x and y names.
    strip_sort : bool or {'strip', 'sort'}, optional
        Passed to :class:`Data`.
        If true the data points with NaN are removed using 
        :func:`numpy.isfinite` and the data is sorted by the x values.
        If 'strip' is given NaNs are removed but the data is not sorted.
        If 'sort' is given the data is sorted but NaNs are left in.
        Default is False so the data isn't changed.
    interp_full : float, optional
        Passed to :class:`Data`.
        This interpolates the data to give an even spacing using the 
        inbuilt method :meth:`to_even`. The default is None and the 
        interpolation isn't done.

    Returns
    -------
    data_dict : {str: Data}
        The data contained in the file in the form of a dictionary where the 
        keys are obtained from the header of the data file.
    """
    def to_key(column_name):
        """Makes the key from the column name."""
        if name_space in column_name:
            return name_space.join(column_name.split(name_space)[:-1])
        else:
            return column_name

    data_df = pd.read_csv(file_name, **kwargs)
    if len(data_df.columns)%2 == 1:
        raise ValueError(
            f"There needs to be an even number of columns. "
            f"The csv had {len(data.columns)}.")
    data_dict = {}
    for i in range(int(len(data_df.columns)/2)):
        to_read = data_df.iloc[:, [2*i, 2*i+1]]
        key = to_key(to_read.columns[0])
        if key != to_key(to_read.columns[1]):
            raise ValueError(
                f"The columns names did not match for X and Y data. "
                f"The columns were {to_read.columns}.")
        # This next bit removes nans added to pad the data.
        last_val = None
        for n, x in enumerate(to_read.values[-1::-1, 0]):
            if x == x:  # False if NaN
                last_val = n
                break
        
        if last_val == None:  # No data found
            vals = np.array([]).reshape(0, 2) 
        elif last_val == 0:  # All cells had data
            vals = to_read.values
        else:
            vals = to_read.values[:-last_val, :]
        data_dict[key] = Data(vals,
            strip_sort=strip_sort, interp_full=interp_full)
    return data_dict


def gen_rand(n, func=None, seed=None, interp_full=None):
    """Produces Data object with random values.

    This uses :meth:`numpy.random.Generator.random` to produce a
    :class:`Data` object. The numbers in both x and y values are continually 
    increasing in steps between 0 and 1. A function can be applied to the y 
    values.

    Parameters
    ----------
    n : int
        Number of data point to have in the object.
    func : function
        A function with one parameter to transform the y values.
    seed : float
        Seed to be passed to :func:`numpy.random.default_rng`
    interp_full : float, optional
        If given the data is evenly interpolated, passed to :class:`Data`. 
        The default is `None` which doesn't interpolate the data.

    Returns
    -------
    data : Data
        The generated data object. 
    """
    if not isinstance(n, (int, np.int_)):
        raise TypeError(
            f"n needs to be an int, but was a {type(n)}")
    elif n < 1:
        raise ValueError(
            f"n need to be a positive integer, but was {n}")

    gen_data = Data(
        np.cumsum(np.random.default_rng(seed).random((n, 2)), axis=0),
        interp_full=interp_full)

    if func is not None:
        gen_data = gen_data.apply_y(func)

    return gen_data

