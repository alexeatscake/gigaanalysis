# gigaanalysis

version 0.5.1

This library provides a collection of classes and functions for analysing datasets which are of the form of one independent and one dependent variable.
This is very common in condensed matter physics experiments and gigaanalysis was produced for use in high magnetic field facilities.

Documentation: https://gigaanalysis.readthedocs.io/en/latest/

##  Layout

It is broken into a collection of modules for different uses

* `data` - This contains the Data class which gigaanalysis is built around. It 
also contains a few functions for common manipulations.
* `mfunc` - This contains mathematical functions that are useful to manipulate 
data objects. This is broken into four sections applying numpy ufuncs, 
making Data objects, performing FFTs, and transforming Data objects.
* `dset` - For saving, loading, and manipulating collections of Data objects which are referred to datasets.
* `fit` - For fitting forms to the data contained in Data objects.
* `parse` - Contains functions for collecting all the data contained in 
datasets together or distribution of data into Data objects.
* `qo` - Functions and classes for analysing quantum osculations 
experiments. 
* `contour` - Class for producing contour maps from a data set, using 
Gaussian processes. 
* `htsc` - Functions which are useful for studying superconductivity. 
* `magnetism` - Functions for studying magnetism.
* `heatc` - Functions for studying the heat capacity of materials.
* `diglock` - An implementation of a digital lock-in.
* `highfield` - A class for processing the data from pulsed magnetic field 
facilities. 
* `const` - A few useful scientific constants in different systems of units.


## Requirements

In a recent update, I have tried to incorporate changes to make it valid for `Numpy 2` and `Pandas 2`.
I am working for it also to be valid on `Numpy 1.21.1`, and on python 3.9 up to the newest version.

