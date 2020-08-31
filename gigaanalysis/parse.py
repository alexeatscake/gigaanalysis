"""GigaAnalysis - Parsing

"""

from .data import *

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
    if threshold == None:
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
    produces a new set of the averages of those groups. This can also produce
    a corresponding set with the standard deviation of the groups.

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

