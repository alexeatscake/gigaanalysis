Introduction
============

This library provides a collection of classes and functions for analysing 
datasets which are of the from of one independent and one dependent 
variable. This is very common in condensed mater physics experiment and 
gigaanalysis was produced for use in high magnetic field facilities.


Layout
------

It broken into a collection of modules for different uses

*   :mod:`gigaanalysis.data` This contains the Data class which gigaanalysis 
    is built around. It also contains a few functions for very common uses.
*   :mod:`gigaanalysis.mfunc` - This contains mathematical functions that 
    are useful to act on data objects. This is broken into four sections 
    applying numpy ufuncs, making Data objects, performing FFTs, and 
    transforming Data 
    objects.
*   :mod:`gigaanalysis.dset` - For saving, loading, and manipulating 
    collections of Data objects which are referred to datasets.
*   :mod:`gigaanalysis.fit` - For fitting forms to the data contained in 
    Data objects.
*   :mod:`gigaanalysis.parse` - Contains functions for collecting all the 
    data contained in datasets together, or distribution of data into Data 
    objects.
*   :mod:`gigaanalysis.qo` - Functions and classes for analysing quantum 
    osculations experiments. 
*   :mod:`gigaanalysis.contour` - Class for producing contour maps from a 
    data set, using Gaussian processes. 
*   :mod:`gigaanalysis.htsc` - Functions which are useful for studying 
    superconductivity. 
*   :mod:`gigaanalysis.magnetism` - Functions for studying magnetism.
*   :mod:`gigaanalysis.heatc` - Functions for studying heat capacity of 
    materials.
*   :mod:`gigaanalysis.diglock` - An implementation of a digital lock in.
*   :mod:`gigaanalysis.highfield` - A class for processing the data from 
    pulsed magnetic field facilities. 
*   :mod:`gigaanalysis.const` - A few useful scientific constants in 
    different systems of units.


Installation
------------

In an environment with ``python >= 3.7`` the following should install 
gigaanalysis. ::

    pip install gigaanalysis


Requirements
------------

This was developed mostly using

* python 3.7.7
* numpy 1.21.2
* pandas 1.3.4
* matplotlib 3.4.3
* h5py 2.10.0

I haven't found any problems with using newer versions of these same 
dependencies.