"""GigaAnalysis - TDMS

This is a simple thing that I needed in the high field labs to read TDMS 
files. It isn't included in the main library as it requires the very useful 
package `nptdms <https://nptdms.readthedocs.io/en/stable/>`_. All in all I 
would recommend just using this as an example, but given the difficultly of 
opening TDMS files, this could be a good starting point.
"""

import gigaanalysis as ga  # This file is extra to gigaanalysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nptdms as tdms  # For read_ISSP


def read_ISSP(file, fieldCH, currentCH, voltageCH, group='Untitled'):
    """Takes data from TDMS file.

    Requires group to be '名称未設定' which is untitled in Japanese.
    Requires field to be labelled 'Field'
    Makes use of :class:`nptdms.tdms`

    Parameters
    ----------
    file : str
        The file name of the .tdms file with the data.
    fieldCH : str
        The name of the channel that contains the field. This as standard
        is called 'Field'
    current : str
        The name of the channel the current is measured on.
    voltage : str
        The name of the channel the voltage is measured on.
    group : str, optional
        The name of the group of the the .tdms file. LabView as standard
        makes this 'Untitled', but if the set language is something other
        than English this will change. It can also be set by the user.
    
    Returns
    -------
    Field : numpy.ndarray
        1d numpy array with field values in
    current : numpy.ndarray
        1d numpy array with current readings
    voltage : numpy.ndarray
        1d numpy array with voltage readings
    """
    tdms_file = tdms.TdmsFile(file).as_dataframe()
    return [x for x in tdms_file[["/'{}'/'{}'".format(group, fieldCH),
                                     "/'{}'/'{}'".format(group, currentCH),
                                     "/'{}'/'{}'".format(group, voltageCH)]
                                     ].values.T]


def save_to_tdms(data, file_name,
        group='Untitled', x_name='X', y_name='Y', **kwargs):
    """Saves a Data object to TDMS.
    
    Makes use of :class:`nptdms.TdmsWriter`.
    ``**kwargs`` are passed to :class:`nptdms.ChannelObject`.

    Parameters
    ----------
    data : Data
        The Data object to save to .tdms
    file_name : str
        The name of the file to save. If it doesn't end in '.tdms' that will 
        be added at the end.
    group : str, optional
        The TDMS group name.
    x_name : str, optional
        The TDMS channel name for the x data.
    y_name : str, optional
        The TDMS channel name for the y data.
    """
    if not isinstance(data, ga.Data):
        raise TypeError(
            f"data needs to be a Data object but was "
            f"insted of the type {type(data)}.")
    for name in [group, x_name, y_name]:
        if not isinstance(name, str):
            raise TypeError(
                "Group and Channel names need to be strings.")
    x_tdms_obj = tdms.ChannelObject(group, x_name, data.x, **kwargs)
    y_tdms_obj = tdms.ChannelObject(group, y_name, data.y, **kwargs)
    if not isinstance(file_name, str):
        raise TypeError(
            f"file_name needs to be a string but was "
            f"{type(file_name)}")
    if file_name[-5:] != ".tdms":
        file_name += ".tdms"
    with tdms.TdmsWriter() as tdms_writer:
        tdms_writer.write_segment(
            [x_tdms_obj, y_tdms_obj])


def read_tdms(file_name,
        group='Untitled', x_name='X', y_name='Y'):
    """Loads a Data object from TDMS.
    
    Makes use of :class:`nptdms.TdmsFile`.

    Parameters
    ----------
    file_name : str
        The name of the file to read.
    group : str, optional
        The TDMS group name.
    x_name : str, optional
        The TDMS channel name for the x data.
    y_name : str, optional
        The TDMS channel name for the y data.

    Returns
    -------
    data : Data
        A data object with the loaded data in.
    """
    for name in [group, x_name, y_name]:
        if not isinstance(name, str):
            raise TypeError(
                "Group and Channel names need to be strings.")
    with tdms.TdmsFile.read(file_name) as tdms_file:
        x_vals = tdms_file[group][x_name].data
        y_vals = tdms_file[group][y_name].data
    return ga.Data(x_vals, y_vals)

