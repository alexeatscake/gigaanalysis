"""GigaAnalysis - Parsing - :mod:`gigaanalysis.parse`
--------------------------------------------------------

This module contains functions for parsing datasets. Now it includes 
functions for identifying measurements clustered in groups and taking the 
average of them. This can be useful for plotting datasets from instruments 
that take multiple measurements at each point in a sweep.
"""

from .data import *
from . import dset, mfunc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cluster_group(data, normalise='constant', threshold=None,
        relative_threshold=False):
    """This identifies clusters of points close together and produces an
    array with each of these points indexed by their cluster.

    Parameters
    ----------
    data : numpy.ndarray
        The values to check if they clustered
    normalise : {'constant', 'value', 'log'} optional
        This normalises the difference between the values of the data set to
        better identify the clusters. 'constant' dose not perform any
        normalisation. 'value' divides the difference by the first value.
        'log' takes the log of all the data values before preforming the
        difference.
    threshold : float optional
        The value the difference needs to exceed to be considered a new
        cluster. If no value is given then the average of the differences are
        used. If :param:relative_threshold is True this value is multiplied 
        by the averages of the differences.
    relative_threshold : bool optional
        If True the given threshold is multiplied by the averages of the
        differences. The default is False.

    Returns
    -------
    groups : numpy.ndarray
        A numpy.ndarray the same length as the dataset containing the indexes
        of the groups each datum corresponds to.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data needs to be a numpy.ndarray.")
    if normalise == 'constant':
        steps = np.abs(np.diff(data))
    elif normalise == 'value':
        steps = np.abs(np.diff(data)/data[:-1])
    elif normalise == 'log':
        steps = np.abs(np.diff(np.log(data)))
    else:
        raise ValueError("Normalise needs to be 'constant', 'value', " 
            "or 'log'.")
    if threshold is None:
        threshold = np.average(steps)
    elif relative_threshold:
        threshold = np.average(steps)*threshold
    else:
        pass  # Just the value of the threshold is used
    split = np.zeros(len(data) -1)
    split[steps>threshold] = 1  # 0 if no jump 1 if jump in T
    split = np.cumsum(split).astype(int)
    groups = np.concatenate([[0], split])
    return groups


def group_average(data, groups, error=False, std_factor=True,
        not_individual=False):
    """This takes a set of data that has a corresponding indexed groups and 
    produces a new set of the averages of those groups. This can also 
    produce a corresponding set with the standard deviation of the groups.

    Parameters
    ----------
    data : numpy.ndarray
        The data set to preform the averages of the groups on.
    groups : numpy.ndarray
        The array with the corresponding index of the groups. Is required to 
        be the same size as the data array.
    error : bool, optional
        Whether to produce the array of the standard deviations. Default is
        False
    std_factor : bool, optional
        If True which is default will output the expectation value of the 
        standard deviation. If False will only output the standard deviation.
        If a group has one datum the standard deviation is given as 0 as
        opposed to infinite.
    not_individual : bool optional
        If True and if error is True the groups with only one datum will be
        dropped.

    Returns
    -------
    averages : numpy.ndarray
        An array the length of the number of groups with the average of the
        values in the data array for the data points in each group.
    errors : numpy.ndarray
        If :param:error is True errors are returned. An array the length of 
        the number of groups with the standard deviation of the
        values in the data array for the data points in each group.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data needs to be a numpy.ndarray.")
    elif not isinstance(groups, np.ndarray):
        raise ValueError("groups needs to be a numpy.ndarray.")
    elif data.size != groups.size:
        raise ValueError("data and groups need to be the same size")
    num_groups = groups[-1] + 1
    averages = np.zeros(num_groups)
    if not_individual:
        skip_individual = np.zeros(num_groups)
    if error:
        errors = np.zeros(num_groups)
    for g in range(num_groups):
        averages[g] = np.average(data[groups==g])
        if error:
            if not_individual or std_factor:
                p_num = np.sum(groups==g)
            if not_individual:
                if p_num > 1:
                    skip_individual[g] = 1
            if std_factor:
                factor = 0 if p_num <= 1 else np.sqrt(p_num)/(p_num - 1)
            else:
                factor = 1
            errors[g] = factor*np.std(data[groups==g])
    if error:
        if not_individual:
            skip_individual = skip_individual.astype(bool)
            return averages[skip_individual], errors[skip_individual]
        else:
            return averages, errors
    else:
        return averages
 

def unroll_dataset(data_set, look_up=None):
    """This unpacks all the values in a data set into 3 arrays.

    This splits the data from a data set into three, the x and y values and 
    the values from the key. To covert the keys into something useful a 
    dict can be provided as a look up table.

    Parameters
    ----------
    data_set : dict of Data
        The data set to unroll all the values from.
    look_up : dict or pandas.Series, optional
        This is a dictionary that converts the keys in the data_set into 
        something to place in the variable array.

    Returns
    -------
    independent : numpy.ndarray
        The x values from all the :class:`.Data` objects.
    dependent : numpy.ndarray
        The y values from all the :class:`.Data` objects.
    variable : numpy.ndarray
        The corresponding keys from the data_set or values produced from 
        passing them into the look up dictionary.
    """
    if look_up is None:
        class self_dict():
            def __getitem__(self, get): return get
        
        look_up = self_dict()
    elif isinstance(look_up, (dict, pd.Series)):
        pass
    else:
        raise TypeError(
            f"If look up is provided need to be a dict was a "
            f"{type(look_up)} instead.")
    
    if dset.check_set(data_set) != 1:
        raise TypeError(
            f"The data_set had multiple nested dictionaries instead "
            f"of only one.")
    
    independent, dependent, variable = [], [], []
    for key, dat in data_set.items():
        independent.append(dat.x)
        dependent.append(dat.y)
        variable.append(np.full(len(dat), look_up[key]))
    
    independent = np.concatenate(independent)
    dependent = np.concatenate(dependent)
    variable = np.concatenate(variable)
    
    return independent, dependent, variable


def roll_dataset(independent, dependent, variable, look_up=None,
        strip_sort=True, drop_empty=False):
    """This packs data from three arrays into a dataset.

    This takes three one dimensional :class:`numpy.ndarray` and uses the 
    last one to group the first two into data objects. The first array is 
    used to for the independent variable and the second is used for the 
    independent variable. A dictionary can also be provided as a look up to 
    change the dataset keys.

    Parameters
    ----------
    independent : numpy.ndarray
        The x values to form all the :class:`.Data` objects.
    dependent : numpy.ndarray
        The y values to  form all the :class:`.Data` objects.
    variable : numpy.ndarray
        The corresponding values to group the values to the different 
        :class:`Data` objects to form the data_set.
    look_up : dict or pandas.Series, optional
        This is a dictionary that converts the values in the variable array 
        into keys that will be used in the dictionary. The default behaviour 
        uses the values in the variable for the keys.
    strip_sort : bool, optional
        This is `True` by default and is passed to the `strip_sort` argument 
        of the :class:`Data` when they are produced.
    drop_empty : bool, optional
        This is `False` by default and if `True` :class:`Data` objects are 
        removed if they contain no data points. This would happen if all the 
        values retrieved were NaNs and then strip_sort was applied.

    Returns
    -------
    data_set : dict of Data
        The data set produced by combining the three data sets.
    """
    if look_up is None:
        class self_dict():
            def __getitem__(self, get): return get
        
        look_up = self_dict()
    elif isinstance(look_up, (dict, pd.Series)):
        pass
    else:
        raise TypeError(
            f"If look up is provided need to be a dict was a "
            f"{type(look_up)} instead.")
    
    if not isinstance(variable, np.ndarray):
        raise TypeError(
            f"variable needs to be an numpy.ndarray but was a "
            f"{type(variable)} instead.")
    elif not isinstance(independent, np.ndarray):
        raise TypeError(
            f"independent needs to be an numpy.ndarray but was a "
            f"{type(variable)} instead.")
    elif not isinstance(dependent, np.ndarray):
        raise TypeError(
            f"dependent needs to be an numpy.ndarray but was a "
            f"{type(dependent)} instead.")
    elif independent.ndim != 1 or \
            independent.shape != dependent.shape or \
            independent.shape != variable.shape:
        raise ValueError(
            f"The three arrays need to be 1 dimensional and of the "
            f"same shape. They had shapes {independent.shape}, "
            f"{dependent.shape}, and {variable.shape}.")
    
    dataset = {}
    for key in np.unique(variable):
        dataset[look_up[key]] = Data(
            independent[variable==key],
            dependent[variable==key],
            strip_sort=strip_sort)

    if drop_empty:
        dataset = dict(filter(lambda elm: len(elm[1])!=0, dataset.items()))

    return dataset


def read_wpd(file_name, parse_keys=None, sort_keys=False, strip_sort=True):
    """Read Web Plot Digitizer output csv files.
    
    `Web Plot Digitizer <https://automeris.io/WebPlotDigitizer>`_ is a 
    program that can extract the data from images of scientific figures. 
    When the program exports the data as a csv file it is in a certain 
    format, which this function reads. The output is a gigaanaylsis dataset 
    with the names of the web plot digitizer datasets as the keys.

    Parameters
    ----------
    file_name : str
        The location of csv file that Web Plot Digitizer produced.
    parse_keys : callable, optional
        If a function is given the keys are passed to it and the output is 
        used as the new key.
    sort_keys : bool, optional
        If `True` then the keys are sorted, the default is `False` where 
        they will be in the order in the csv file.
    strip_sort : bool, optional
        If default of `True` the option of the same name is given for each 
        of the :class:`.Data` objects in the set.

    Returns
    -------
    dataset : dict of Data
        A dictionary where the values are :class:`.Data` objects containing 
        the data in the csv file.
    """
    df = pd.read_csv(file_name, header=0)
    keys = np.array(df.columns)[::2]
    data = df.values[1:, :].astype(np.float_)
    
    if callable(parse_keys):
        keys = [parse_keys(key) for key in keys]
        
    if sort_keys:
        if callable(sort_keys):
            keys = sorted(keys, key=sort_keys)
        else:
            keys = sorted(keys)
    
    dataset = {}
    for n, key in enumerate(keys):
        dataset[key] = Data(data[:, 2*n:2*n+2], strip_sort=strip_sort)
    
    return dataset


def end_of_dataset(data_set, minimum=True, look_up=None,
        interp_step=None, loess_window=None, loess_poly=2):
    """Produces a Data object from the x value extent of a dataset.

    This produces a Data object where each datum is composed of a value made 
    from the key of the dataset, and the other is either the minimum or 
    maximum x value. This is use for finding the x extent of two dimensional 
    maps.

    Parameters
    ----------
    data_set : dict of Data
        The dataset to obtain the values from.
    minimum : bool, optional
        If to take the minimum or maximum x values from the dataset. The 
        default value if `True` and this takes the minimum.
    look_up : dict, optional
        A dictionary that converts the keys from the dataset into floats to 
        be returned. The default is `None` where the values of the keys 
        themselves is used.
    interp_step : float, optional 
        This applies the method :meth:`.Data.interp_step` to the data object 
        after it is produced. This is useful as the data needs to be evenly
        interpreted before it can be smoothed. The default is `None` which 
        does not apply the method.
    loess_window : float, optional 
        This can be used to smooth the data. The default is `None` where no 
        smoothing is applied. The value sets the range to be used for the 
        loess window in :func:`mfunc.loess`. This is useful for smoothly 
        masking the bottom of contour maps.
    loess_poly : int, optional
        The default is `2`. This this is the order of the polynomial to be 
        used by the loess function :func:`mfunc.loess`.

    Returns
    -------
    end_data : Data
        The data object which is composed of the minimum or maximum x values 
        in the dataset.
    """
    if look_up is None:
        class self_dict():
            def __getitem__(self, get): return get
        
        look_up = self_dict()
    elif isinstance(look_up, (dict, pd.Series)):
        pass
    else:
        raise TypeError(
            f"If look up is provided need to be a dict was a "
            f"{type(look_up)} instead.")
    
    if dset.check_set(data_set) != 1:
        raise TypeError(
            f"The data_set had multiple nested dictionaries instead "
            f"of only one.")

    if minimum:
        val_func = np.min
    else:
        val_func = np.max

    end = Data([[look_up[key], val_func(dat.x)] \
        for key, dat in data_set.items()], strip_sort=True)

    if interp_step is not None:
        end = end.interp_step(interp_step)
    elif loess_window is not None:
        end = end.interp_number(len(end))  # Need to interp for loess

    if loess_window is not None:
        end = mfunc.loess(end, loess_window, loess_poly)

    return end

