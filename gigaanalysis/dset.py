"""GigaAnalysis - Data Set Management - :mod:`gigaanalysis.dset`
-------------------------------------------------------------------

This module has functions to save nested dictionaries with :class:`Data` 
objects as the values. It provides the functionality to save and read from 
HDF5 files using `h5py <https://www.h5py.org/>`_ and also .csv files. It 
also can create and use :class:`pandas.DataFrame` to store and display 
associated meta data.

"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import h5py  # For interacting with HDF5 files


def check_set(data_set, meta_df=None, higher_key=()):
    """Checks the data_set and metadata data frame is the correct from.

    This goes through the nested dictionaries and checks that the values 
    contained are either :class:`dict` or :class:`gigaanalysis.data.Data` 
    objects. If objects other than these are found errors are thrown. The 
    metadata dictionary ``meta_df`` is checked that every 
    :class:`Data` has a row that is describing it in the ``meta_df``.

    Parameters
    ----------
    data_set : dict of {str: dict or Data}
        A dictionary containing either nested dictionaries or 
        :class:`gigaanalysis.data.Data` objects.
    meta_df : pandas.DataFrame
        Metadata held in a :class:`pandas.DataFrame` where the indexes are a 
        the keys of ``data_set`` and the columns provide information 
        about the :class:`Data` objects. For nested 
        dictionaries hierarchical indexing is used 
        (:class:`pandas.MultiIndex`).
    higher_key : tuple, optional 
        Tuple with keys in used for the regression to start in a nested 
        dictionary.

    Returns
    -------
    count : int
        The number of layers of the data_set.
    """
    if not isinstance(data_set, dict):
        raise TypeError(
            f"data_set was not a dict but instead a {type(data_set)}.")
    for key, val in data_set.items():
        new_key = (*higher_key, key)
        if isinstance(val, dict):
            count = 1 + check_set(val, meta_df, new_key)
        elif not isinstance(val, Data):
            raise TypeError(
                f"The dictioaries contain objects which are "
                f"not dictionaires or Data objects. The object in "
                f"key:{new_key}, was a {type(val)}.")
        elif meta_df is not None:
            count = 1
            if isinstance(new_key, tuple) and len(new_key) == 1:
                new_key = new_key[0]
            if new_key not in meta_df.index:
                raise ValueError(
                    f"The meta DataFrame given did not have items "
                    f"which where in the data_set. The key missing "
                    f"was: {new_key}.")
        else:
            count = 1
    return count


def __label_hdf5_as_ga_set(file, location):
    """Add a attribute to a hdf5 group called "ga_data_set" and create 
    it if it doesn't exist.

    Parameters
    ----------
    file : h5py.File
        A file object that references the `.hdf5` file where the data is 
        being saved.
    location : str
        The location of the group to consider in the `.hdf5` file.
    """
    if location not in file:
        file.create_group(location)
    file[location].attrs['ga_data_set'] = True


def __set_attrs_from_df(dset, data, meta_df, key):
    """Sets the attributes of a :class:`h5py.Dataset` to hold the metadata.

    Used by :func:`set_to_hdf5`.

    Parameters
    ----------
    dset : h5py.Dataset
        The dataset to set the attributes of.
    data : gigaanalysis.data.Data
        The data that is being saved to the dataset.
    meta_df : pandas.DataFrame or None
        Where the meta data is stored
    key : tuple
        The key that refers to the row in the matadata table
    """
    if meta_df is not None:
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        attrs_to_set = dict(meta_df.loc[key].items())
    else:
        attrs_to_set = dict()
    # set some values that should be in every metadata table
    attrs_to_set.update({
        'size':len(data),
        'min_x':data.min_x() if len(data) != 0 else np.nan,
        'max_x':data.max_x() if len(data) != 0 else np.nan,
    })
    for prop, val in attrs_to_set.items():
        if not pd.isnull(val):
            dset.attrs[prop] = val

        
def __hdf5_set(file, data_set, meta_df, higher_key=(),
    location="/"):
    """Goes though a data_set and save the values to a :class:`h5py.File`.

    This is called in :func:`set_to_hdf5`.

    Parameters
    ----------
    file : h5py.File
        A file object that references the HDF5 file where the data is 
        being saved.
    data_set : dict of {str: dict or Data}
        A dictionary containing either nested dictionaries or 
        :class:`gigaanalysis.data.Data` objects.
    meta_df : pandas.DataFrame
        Metadata held in a :class:`pandas.DataFrame` where the indexes are 
        the keys of the `data_set` dict.
    higher_key : tuple, optional 
        Tuple with keys in used for the regression to start in a nested 
        dictionary.
    location : str, optional
        The location of the hdf5 Group to save the data to.
    """
    for key, val in data_set.items():
        # remove non allowed characters ' ' and '/'
        # and sort keys and locations
        key = str(key).replace(' ', '').replace('/', '')
        new_key = (*higher_key, key)
        new_loc = f"{location}{'/'.join(new_key)}"
        if isinstance(val, dict):  # if dict in dict call self
            __label_hdf5_as_ga_set(file, new_loc)
            __hdf5_set(file, val, meta_df, new_key, location)
        elif not isinstance(val, Data):
            raise TypeError(
                f"The dictioaries contain objects which are "
                f"not dictionaires or Data objects. The object in "
                f"key:{new_key}, was a {type(val)}.")
        else:
            file.create_dataset(new_loc, data=val.values)
            __set_attrs_from_df(
                file[new_loc], val, meta_df, new_key)


def set_to_hdf5(data_set, file_name, meta_df=None,
                location="/", overwrite=False, info_attr=None):
    """This saves a data set to a HDF5 file.

    This saves a data set of made of nested :class:`dict` of 
    :class:`gigaanalysis.data.Data` to a HDF5 file, using 
    :class:`h5py.File`. This can also take a :class:`pandas.DataFrame` 
    containing the associated meta_data.

    Parameters
    ----------
    data_set : dict of {str: dict or Data}
        A dictionary containing either nested dictionaries or 
        :class:`gigaanalysis.data.Data` objects.
    file_name : str
        The file name to save the HDF5 file with
    meta_df : pandas.DataFrame, optional
        Metadata held in a :class:`pandas.DataFrame` where the indexes are a 
        the keys of the `data_set` dict and the columns provide information 
        about the :class:`Data` objects. For nested 
        dictionaries hierarchical indexing is used 
        (:class:`pandas.MultiIndex`).
    location : str, optional
        The location of the HDF5 group to save the data to. The default is 
        the root group.
    overwrite : bool, optional
        If the function should overwrite existing HDF5 file. The default is 
        to not overwrite.
    info_attr : str, optional
        If a string is given this is set as an HDF5 attribute to group. This 
        can hold a description of data if required.
    """
    if location != "/":  # In case the user forgets the "/" at the end
        if location[-1] != "/":
            location += "/"
    if not isinstance(data_set, dict):
        raise TypeError(
            f"data_set needs to be a dict but is a {type(data_set)}")
    if meta_df is not None:
        if not isinstance(meta_df, pd.DataFrame):
            raise TypeError(
                f"meta_table needs to be a pandas.DataFrame but is "
                f"a {type(meta_df)}")
    check_set(data_set, meta_df)
    read_write = 'w' if overwrite else 'a'
    with h5py.File(file_name, read_write) as file:
        __label_hdf5_as_ga_set(file, location)
        __hdf5_set(file, data_set, meta_df, location=location)
        if isinstance(info_attr, str):
            file[location].attrs['info'] = info_attr


def __print_hdf5_group(group):
    """A recursive function to call used by :func:`print_hdf5`.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group to print.
    """
    for val in group.values():
        if isinstance(val, h5py.Group):
            print(
                f"{val.name} - Group\n"
                f"   {list(val.attrs.items())}")
            __print_hdf5_group(val)
        else:
            print(
                f"{val.name} - Data Set\n"
                f"   {list(val.attrs.items())}")


def print_hdf5(file_name):
    """Prints the names and attributes of the contents of a HDF5 file.

    Parameters
    ----------
    file_name : str
        The name of the HDF5 file to read.
    """
    with h5py.File(file_name, 'r') as file:
        __print_hdf5_group(file['/'])


def __read_hdf5_group(group, data_set, meta_df):
    """Build up a data set and metadata table recursively from a HDF5 file.

    Parameters
    ----------
    group : h5py.Group
        The group to read in the HDF5 file.
    data_set : dict
        The dictionary to save the :class:`gigaanalysis.data.Data` in.
    meta_df : pandas.DataFrame
        A metadata table to fill with the attributes of the 
        :class:`h5py.Dataset` in the group.

    Returns
    -------
    data_set : dict
        The dictionary after it is populated.
    meta_df : pandas.DataFrame
        A metadata table after it is populated.
    """
    try:
        assert group.attrs['ga_data_set']
    except:
        raise ValueError(
            f"The groups given to extract gigaanalyis data from "
            f"did not have the tag ga_data_set in the location "
            f"{group.name}.")
    for val in group.values():
        this_key = val.name.split('/')[-1]
        if isinstance(val, h5py.Group):
            data_set[this_key] = {}
            data_set[this_key], meta_df = __read_hdf5_group(
                val, data_set[this_key], meta_df)
        else:
            data_set[this_key] = Data(val[:])
            new_row = pd.DataFrame(
                [dict(val.attrs.items())],
                index=[val.name])
            meta_df = meta_df.append(new_row)
    return data_set, meta_df


def __count_layer(dict_to_check, count=0):
    """Counts the layers of :class:`dict` in a nested dictionary.

    This only counts the depth first item and assumes the rest is the same.

    Parameters
    ----------
    dict_to_check : dict
        The dictionary to check how many layers of dictionaries are inside.
    count : int
        The count which is used recursively.
    """
    count +=1
    for val in dict_to_check.values():
        if isinstance(val, dict):
            count = __count_layer(val, count)
        break
    return count


def __reindex_meta(meta_df, layers):
    """Swaps the indexes in a table from HDF5 locations to multiindex tupels.

    Parameters
    ----------
    meta_df : pandas.DataFrame
        The metadata table to reindex.
    layers : int
        The number of layers in the data set to use as an index.

    Returns
    -------
    meta_df : pandas.DataFrame
        The metadata table with the new indexes.
    """
    meta_df = meta_df.reset_index()
    capture = '/([^/]*)'*layers + '$'
    key_list = [f'key{x+1}' for x in range(layers)]
    meta_df[key_list] = meta_df['index'].str.extract(capture)
    meta_df = meta_df.drop('index', axis=1)
    meta_df = meta_df.set_index(key_list)
    return meta_df


def set_from_hdf5(file_name, location='/'):
    """Reads a HDF5 file and returns a dataset and a metadata table.

    This reads a HDF5 file using :class:`h5py.File`, and produces a dataset 
    comprising of a nested :class:`dict` which contains 
    :class:`gigaanalysis.data.Data` objects. The dataset is accompanied by a 
    metadata table in the form of a :class:`pandas.DataFrame` with the 
    indexes are the same as the keys of the dictionaries.

    Parameters
    ----------
    file_name : str
        The name of the HDF5 file to read.
    location : str, optional
        The location of the group in the HDF5 file which contains the 
        dataset to be read. The default is the root group.

    Returns
    -------
    data_set : dict of {str: dict or Data}
        A dictionary containing either nested dictionaries or 
        :class:`gigaanalysis.data.Data` objects.
    meta_df : pandas.DataFrame, optional
        Metadata held in a :class:`pandas.DataFrame` where the indexes are a 
        the keys of ``data_set`` and the columns provide information 
        about the :class:`Data` objects. For nested 
        dictionaries hierarchical indexing is used 
        (:class:`pandas.MultiIndex`).
    """
    data_set = {}
    meta_df = pd.DataFrame(columns=[
        'size', 'min_x', 'max_x'])
    with h5py.File(file_name, 'r') as file:
        data_set, meta_df = __read_hdf5_group(
            file[location], data_set, meta_df)
    meta_df = __reindex_meta(
        meta_df, __count_layer(data_set))
    return data_set, meta_df


def array_to_hdf5(data, file_name, location, attributes=None, 
        overwrite=False):
    """Saves a numpy array to a HDF5 file.

    This is for saving a plane :class:`numpy.ndarray` to a HDF5 using 
    :class:`h5py.File`. This is meant to work in the same style as 
    :func:`set_to_hdf5`. It also can save a set of attributes in the form of 
    a dictionary.

    Parameters
    ----------
    data : numpy.ndarray
        The data to save to the file in a numpy array.
    file_name : str
        The name of the HDF5 file to save the data to.
    location : str
        The location of the :class:`h5py.Dataset`, which is a string with 
        the groups and the data set name separated by "/".
    attributes : dict of {str: val}, optional
        A dictionary of meta data to attach to the data set. The keys of the 
        dictionary need to be `str`. Default is None and attaches no 
        attributes to the data set.
    overwrite : bool, optional
        If default of `False` the existing file is not overwritten and is 
        instead added to. This will throw an error if trying to save to a 
        location of an already existing dataset.
    """
    # Start with checking location specifier
    if not isinstance(location, np.str):
        raise TypeError(
            f"location needs to be a string but was a "
            f"{type(location)}")
    location = location.replace(" ", "")
    if location[-1] == "/":
        raise ValueError(
            "The location and the data set name need to spesified "
            "no data set name was given after the last '/'")
    if location[0] != "/":  # Same if they spesify the root group
        location = "/" +location
    # Check the data is a correct kind of np.array
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"data needs to be a numpy array but is a {type(data)}")
    if data.dtype == 'O':
        raise TypeError(
                f"The array contained python object type values but "
                f"these cannot be saved to HDF5 files.")
    # Check the attributes are the correct type
    if attributes is not None:
        if not isinstance(attributes, dict):
            raise TypeError(
                f"attributes needs to be a dict but was type "
                f"{type(attributes)}")
        if not all(isinstance(key, np.str) for key in attributes.keys()):
            raise TypeError(
                f"The keys for the attributes need to be all strings")
    # Parse the location into groups and dset names
    locs = location.split('/')
    dset_name = locs[-1]
    if len(locs) ==  2:
        group_name = None
    else:
        group_name = "/".join(locs[:-1]) 
    # Open File
    read_write = 'w' if overwrite else 'a'
    with h5py.File(file_name, read_write) as file:
        if group_name is None:
            file.create_dataset(dset_name, data=data)
            dset = file[dset_name]
        elif group_name in file:
            file[group_name].create_dataset(dset_name, data=data)
            dset = file[group_name][dset_name]
        else:
            file.create_group(group_name)
            file[group_name].create_dataset(dset_name, data=data)
            dset = file[group_name][dset_name]
        if attributes is not None:
            for prop, val in attributes.items():
                dset.attrs[prop] = val


def array_from_hdf5(file_name, location):
    """This reads a dataset in a HDF5 file to a numpy array.

    This function is to read the data saved using the :func:`array_to_hdf5`. 
    It reads the data and the attributes using :class:`h5py.File` and 
    returns the result.

    Parameters
    ----------
    file_name : str
        The name of the HDF5 file to be read.
    location : str
        The location of the dataset with the groups and dataset name 
        separated by "/".

    Returns
    -------
    data : numpy.ndarray
        A numpy array containing the data in the data set.
    attributes: dict
        A dictionary containing the attributes of the data set that was 
        read. If there was no attributes then the dictionary will be empty.
    """
    if not isinstance(location, np.str):
        raise TypeError(
            f"location need to be a string but was a {type(location)}")
    if location[0] != "/":
        location = "/" + location
    with h5py.File(file_name, 'r') as file:
        if location not in file:
            raise ValueError(
                f"This is not a valid location of a data set.")
        if not isinstance(file[location], h5py.Dataset):
            raise ValueError(
                f"The spesified location is not a data set")
        data = file[location][:]
        attributes = dict(file[location].attrs.items())
    return data, attributes


def sort_dset(dataset, apply_key=None, sort_key=None, check_data=True):
    """This sorts and formats the keys in a dataset.

    This is useful after loading a dataset from a HDF5 file, as keys that 
    were floats will have been set to strings and then loaded by the leading 
    digit. This function can apply a function to each key and then sort them.

    Parameters
    ----------
    dataset : recursive dict of Data
        This is the dataset to sort which are nested dictionaries of Data 
        objects.
    apply_key : function or list of function, optional
        This is a function that will be applied to the keys to reformat them 
        before the are reordered. If a list is given then each function will 
        be applied on each layer of the dataset in turn. If `None` then no 
        function is applied. The default is `None`.
    sort_key : function or list of function, optional
        This is the key that is passed to `sorted` to sort the dataset 
        based on its keys. If a list is given then each function will be 
        applied to each layer of the dataset. If `None` then no key is passed 
        and the default sorting behaviour is used. If the string 'pass' is 
        given then no sorting is applied. The default is `None`.
    check_data : bool, optional
        Whether to check if the objects in the dict are :class:`Data` 
        objects. The default is True.
    
    Returns
    -------
    dataset : recursive dict of Data
        The dataset after the functions have been applied to the keys and the
        keys and then they have been sorted.
    """
    if isinstance(dataset, dict):  # deal with dataset
        pass
    elif check_data and not isinstance(dataset, Data):
        raise TypeError(
            f"The dataset contains values other than gigaanalyis.Data "
            f"objects. Contained {type(dataset)}. To turn off this check "
            f"set check_data to False.")
    else:
        return dataset  # For recursion

    if isinstance(apply_key, list):  # if apply_key a list
        if len(apply_key) == 0:
            raise ValueError(
                f"The apply_key list was not as deep as the nested dataset.")
        apply_list = apply_key[1:]
        apply_key = apply_key[0]
    else:
        apply_list = None

    if apply_key is None:  # check apply_key
        apply_key = lambda x: x
        if apply_list is None:
            apply_list = apply_key
    elif callable(apply_key):
        if apply_list is None:
            apply_list=apply_key
    else:
        raise TypeError(
            f"apply_key needs to be callable or a list of callable "
            f"functions but was {type(apply_key)}")

    if isinstance(sort_key, list):  # if sort_ley a list
        if len(sort_key) == 0:
            raise ValueError(
                f"The sort_key list was not as deep as the nested dataset.")
        sort_list = sort_key[1:]
        sort_key = sort_key[0]
    else:
        sort_list = None

    if sort_key is None:  # check sort_key
        pass
    elif sort_key == "pass":
        sort_key = lambda x: 0
        if sort_list is None:
            sort_list = sort_key
    elif callable(sort_key):
        if sort_list is None:
            sort_list = sort_key
    else:
        raise TypeError(
            f"sort_key needs to be callable or a list of callable "
            f"functions but was {type(sort_key)}")
    
    # edit key and preform recursion on included data
    dataset = {
        apply_key(key):sort_dset(dat, apply_list, sort_list, check_data) \
            for key, dat in dataset.items()}
    # Sort by key and return
    sorted_keys = sorted(dataset.keys(), key=sort_key) 
    dataset = {key:dataset[key] for key in sorted_keys}
    return dataset

