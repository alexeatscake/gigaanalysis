"""GigaAnalysis - Contour Mapping - :mod:`gigaanalysis.contour`
---------------------------------------------------------------

Here is a class and a few functions for contour mapping datasets to produce 
the gridded data for figures. This mostly makes use of a statistical 
technique called `Gaussian Processes
<https://en.wikipedia.org/wiki/Gaussian_process>`_. In the simplest form 
this technique assumes that the surface being mapped is a combination of 
many normal distributions. While this is a crude approximation it works 
surprisingly well and sets a solid foundation for more complicated 
assumptions.
"""

from .data import *
from . import mfunc, dset, parse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull

def _norm_array(to_change, reference):
    """Normalise an array by the range of values in a different array. Takes 
    two numpy.ndarray and returns one numpy.ndarray the same size as the 
    first.
    """
    return (to_change - reference.min())/(reference.max() - reference.min())


class GP_map():
    """Gaussian Process Contour Mapping

    This takes a gigaanalysis dataset (a :class:`dict` of :class:`.Data` 
    objects), and two :class:`numpy.ndarray` of x and y values to 
    interpolate over in a grid. It then uses the method of Gaussian 
    Processes to interpolate from the provided data into the generated 
    values on the gird.

    The class requires the kernel for the Gaussian process to be set using 
    one of two methods. For most applications it is sufficient to use the 
    method :meth:`GP_map.set_distance_kernel` as this only needs one 
    argument. This does assume that the kernel can be expressed as the 
    euclidean distance between the point of interest and the known data. 
    The method :meth:`GP_map.set_xy_kernel` can be used for more 
    complicated kernel application.

    Finally :meth:`GP_map.predict_z` can be run to generate the 
    interpolated points. This is separated into its own method as it 
    contains the computationally heavy part of the calculation. If desired 
    the generated data can be cut to the region contained in a convex hull 
    of the provided data to avoid unintentional extrapolation.

    Parameters
    ----------
    dataset : dict of {float:Data}
        The data to perform the interpolation on. This can be in the form of 
        a gigaanalysis dataset where the keys of the dictionaries are 
        floats. This will then be unrolled using 
        :func:`parse.unroll_dataset`.
    gen_x : numpy.ndarray
        A 1D numpy array with the x values to interpolate.
    gen_y : numpy.ndarray
        A 1D numpy array with the y values to interpolate.
    key_y : bool, optional
        If default of `False` then the keys of the array are used as the x 
        values, and the x values of the :class:`.Data` objects are used as 
        the y values. To swap these two roles set key_y to `True`.
    normalise_xy : bool, optional
        If default of 'True' then the x y values are normalised to the range 
        0 to 1. This is useful for the kernel to deal with the probable 
        disparity of units in the two directions. If 'False' then this is 
        not done.

    Attributes
    ----------
    input_x : numpy.ndarray
        A 1D array of x values of the provided data to process. These will 
        be normalised if `normalise_xy` is `True`.
    input_y : numpy.ndarray
        A 1D array of y values of the provided data to process. These will 
        be normalised if `normalise_xy` is `True`.
    input_z : numpy.ndarray
        A 1D array of z values of the provided data to process.
    gen_x : numpy.ndarray
        A 1D array of the x values to interpolate.
    gen_y : numpy.ndarray
        A 1D array of the y values to interpolate.
    gen_xx : numpy.ndarray
        A 2D array of the x values for all the interpolated points. These 
        will be normalised if `normalise_xy` is `True`.
    gen_yy : numpy.ndarray
        A 2D array of the y values for all the interpolated points. These 
        will be normalised if `normalise_xy` is `True`.
    kernel : Callable
        A function which takes four 1D numpy arrays. The first is a set of 
        x values and then corresponding y values and then another similar 
        pair. These are then used to produce a 2D array of the kernel 
        weights.
    white_noise : float
        The amplitude of the white noise term in the kernel function.
    kmat_inv : numpy.ndarray
        A 2D array which is the independent part of the covariance matrix.
    predict_z : numpy.ndarray
        A 2D array with the interpolated values produced.
    """
    def __init__(self, dataset, gen_x, gen_y, 
            key_y=False, normalise_xy=True):
        # Set up class
        self.gen_x, self.gen_y = gen_x, gen_y
        
        self.input_y, self.input_z, self.input_x = \
            ga.parse.unroll_dataset(dataset)
        
        if key_y:
            self.input_x, self.input_y = self.input_y, self.input_x
        
        self.gen_xx, self.gen_yy = np.meshgrid(gen_x, gen_y)
        
        if normalise_xy:
            self.input_x = _norm_array(self.input_x, self.gen_x)
            self.input_y = _norm_array(self.input_y, self.gen_y)
            self.gen_xx = _norm_array(self.gen_xx, self.gen_x)
            self.gen_yy = _norm_array(self.gen_yy, self.gen_y)
        
        self.kernel = None
        self.white_noise = None
        self.kmat_inv = None
        self.predict_z = None
        
    def set_distance_kernel(self, dis_kernel, white_noise, **kwargs):
        """Set a kernel which is a function of euclidean distance.

        Here you can set a kernel which is a function of distance between 
        the point to interpolate and the known data. The kernel is a 
        calculable function with one argument.
        The keyword arguments are passed to the distance kernel function.

        Parameters
        ----------
        dis_kernel : Callable
            A function with one argument which takes a 
            :class:`numpy.ndarray` of distance values and returns the same 
            shaped :class:`numpy.ndarray` of kernel weights. Keyword 
            arguments will also be passed to this function.
        white_noise : float
            The value of the white noise term which takes into account 
            stochastic error in the sample. It also insures the success of 
            the matrix inversion, so even with perfect data a small white 
            noise term is preferable. 
        """
        self.white_noise = white_noise
        
        def kernel(x1, y1, x2, y2):
            
            xx1, xx2 = np.meshgrid(x1, x2)
            yy1, yy2 = np.meshgrid(y1, y2)
            
            dis = np.sqrt((xx1 - xx2)**2 + (yy1 - yy2)**2)
            
            return dis_kernel(dis, **kwargs)
        
        self.kernel = kernel
        
    def set_xy_kernel(self, xy_kernel, white_noise, **kwargs):
        """Set a kernel which is a function of the x and y values.

        Here you can set a kernel which is a function of the x and y value 
        of both terms to compare. The kernel is a calculable function with 
        four arguments.
        The keyword arguments are passed to the distance kernel function.

        Parameters
        ----------
        dis_kernel : Callable
            A function with four arguments which takes four
            :class:`numpy.ndarray` of x and y values of the same shape and 
            returns the same shaped :class:`numpy.ndarray` of kernel weights.
            The arguments are x1, y1, x2, y2 where x1 and y1 are the values 
            of the first coordinates to compare, and x2 and y1 are the 
            second.
            Keyword arguments will also be passed to this function.
        white_noise : float
            The value of the white noise term which takes into account 
            stochastic error in the sample. It also insures the success of 
            the matrix inversion, so even with perfect data a small white 
            noise term is preferable. 
        """
        self.white_noise = white_noise
        
        def kernel(x1, y1, x2, y2):
            
            xx1, xx2 = np.meshgrid(x1, x2)
            yy1, yy2 = np.meshgrid(y1, y2)
            
            return xy_kernel(xx1, yy1, xx2, yy2, **kwargs)
        
        self.kernel = kernel
        
    
    def _make_kmat_inv(self):
        """Calculates the kernel matrix inverse.
        """
        if self.kernel is None:
            raise AttributeError("Need to set kernel before hand.")
            
        self.kmat_inv = np.linalg.inv(
            self.kernel(self.input_x, self.input_y, 
                self.input_x, self.input_y)  + \
            self.white_noise*np.identity(self.input_x.size))
    
    def predict(self, cut_outside=False):
        """Calculates the interpolated z values.

        Runs the calculation and returns the result of interpolating the z 
        values using the Gaussian processes technique.

        Parameters
        ----------
        cut_outside : bool, optional
            If default of 'False' returns all the data for the grid to 
            interpolate. If 'True' the values that require extrapolation are 
            set to numpy.nan. This is done using 
            :meth:`cut_outside_hull`. If float is given then that 
            is used as the tolerance and the cut is preformed.

        Returns
        -------
        predict_z : numpy.ndarray
            A 2D array of the values of the calculated z values in the 
            locations of :attr:`gen_x` and :attr:`gen_y`. This function 
            also sets the attribute :attr:`predict_z`.

        """
        if self.kernel is None:
            raise AttributeError("Need to set a kernel before hand.")
        
        self._make_kmat_inv()
        
        self.predict_z = self.input_z @ self.kmat_inv @ \
            self.kernel(self.input_x, self.input_y, 
                self.gen_xx.flatten(), self.gen_yy.flatten()).T
        
        self.predict_z = self.predict_z.reshape(self.gen_y.size, 
            self.gen_x.size)
        
        if cut_outside is not False:
            if isinstance(cut_outside, float):
                self.cut_outside_hull(tol=cut_outside)
            else:
                self.cut_outside_hull()
        
        return self.predict_z
    
    def cut_outside_hull(self, tol=1e-9):
        """Removes data that requires extrapolation

        When called this function sets all the values to interpolate that 
        are not surrounded by three points from the input data to 
        `numopy.nan`. This means that the result doesn't extrapolate. This 
        is done using :class:`scipy.spatial.ConvexHull`.

        This changes the value of :attr:`predict_z`.

        Parameters
        ----------
        tol : float, optional
            The tolerance when comparing points to see if they are inside 
            the convex hull. A higher tolerance means more points will be 
            included. The default is ``10**(-9)``.
        """
        
        hull = ConvexHull(np.concatenate(
            [self.input_x[:, None], self.input_y[:, None]], axis=1))
        hx, hy, ho = hull.equations.T
        
        out_hull = np.any(
            np.outer(self.gen_xx, hx) + \
            np.outer(self.gen_yy, hy) + \
            np.outer(np.ones(self.gen_xx.shape), ho) >= -tol, axis=1)
        
        out_hull = out_hull.reshape(self.gen_xx.shape)
        
        self.predict_z[out_hull] = np.nan
    
    def plotting_arrays(self):
        """Produces the three arrays need for plotting

        This makes use of ``numpy.meshgrid(gen_x, gen_y)``, and is in the 
        from needed for :func:`matplotlib.pyplot.contorf`.

        Returns
        -------
        r_gen_xx : numpy.ndarray
            A 2D array of the x values. These are not normalised even if a 
            normalisation is set.
        r_gen_yy : numpy.ndarray
            A 2D array of the y values. These are not normalised even if a 
            normalisation is set.
        predict_z : numpy.ndarray
            A 2D array of the z values. This is the same as 
            :attr:`predict_z`.
        """
        r_gen_xx, r_gen_yy = np.meshgrid(self.gen_x, self.gen_y)
        
        return r_gen_xx, r_gen_yy, self.predict_z
    
    def plot_contorf(self, colorbar_kwargs={}, **kwargs):
        """Plot the generated data as a contour map

        This makes use of :func:`matplotlib.pyplot.contorf` and keyword 
        arguments are passed to it. 
        Keyword arguments can be passed to the 
        colour bar by setting the keyword argument `colorbar_kwargs` to a 
        dictionary. This uses :func:`matplotlib.pyplot.colorbar`.
        """
        plt.contourf(*self.plotting_arrays(), **kwargs)
        plt.colorbar(**colorbar_kwargs)
        
    def plot_contor(self, **kwargs):
        """Plot the generated data as a contour map

        This makes use of :func:`matplotlib.pyplot.contor` and keyword 
        arguments are passed to it. 
        Keyword arguments can be passed to the 
        colour bar by setting the keyword argument `colorbar_kwargs` to a 
        dictionary. This uses :func:`matplotlib.pyplot.colorbar`.
        """
        plt.contour(*self.plotting_arrays(), **kwargs)
        plt.colorbar(**colorbar_kwargs)

    def plot_input_scatter(self, **kwargs):
        """Plots the input data as a scatter graph

        This is useful for debugging and getting an idea of what the data is 
        like before applying Gaussian processes. Makes use of 
        :func:`matplotlib.pyplot.scatter`, and keyword arguments are passed 
        to it.
        """
        plt.scatter(self.input_x, self.input_y, c=self.input_z, **kwargs)
        plt.colorbar()

