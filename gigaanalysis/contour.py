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

from scipy.spatial import ConvexHull  # For GP_map.cut_outside_hull
from scipy.optimize import minimize  # For GP_map.optermise_argument


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
    dataset : dict of {float:Data} or numpy.ndarray
        The data to perform the interpolation on. This can be in the form of 
        a gigaanalysis dataset where the keys of the dictionaries are 
        floats (unless 'look_up' is used). This will then be unrolled using 
        :func:`parse.unroll_dataset`. A three column numpy array can also be 
        provided with the x, y, and z values in each respective column.
    gen_x : numpy.ndarray
        A 1D numpy array with the x values to interpolate.
    gen_y : numpy.ndarray
        A 1D numpy array with the y values to interpolate.
    key_y : bool, optional
        If default of `False` then the keys of the array are used as the x 
        values, and the x values of the :class:`.Data` objects are used as 
        the y values. To swap these two roles set key_y to `True`.
    normalise_xy : bool or tuple, optional
        If 'True' then the x y values are normalised to the range 
        0 to 1. This is useful for the kernel to deal with the probable 
        disparity of units in the two directions. If default of 'False' then 
        this is not done. A tuple of two floats can be provided instead 
        which the x and y values will be normalised to instead of range 
        unity. This can be useful for weighting the units in an euclidean 
        kernel.
    look_up : dict, optional
        This is a dictionary with all the keys that match the keys in the 
        dataset and values which are floats to be used for the x values. 
        This is passed to :func:`parse.unroll_dataset`. Default is `None` 
        and then keys are used as the values.
    even_space_y : float, optional
        If a float is provided then the independent data in the gigaanalysis 
        data objects is evenly spaced using :meth:`Data.interp_step`. This 
        is useful if more finely spaced data points shouldn't be given more 
        weight in the calculation. The default is none and then the original 
        data is used.

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
    kernel_args : dict
        This is a dictionary of keyword argument to be passed to the 
        provided kernel function.
    white_noise : float
        The amplitude of the white noise term in the kernel function.
    kmat_inv : numpy.ndarray
        A 2D array which is the independent part of the covariance matrix.
    predict_z : numpy.ndarray
        A 2D array with the interpolated values produced.
    """
    def __init__(self, dataset, gen_x, gen_y, key_y=False, 
            normalise_xy=False, look_up=None, even_space_y=None):
        # Set up class
        try:
            gen_x = np.asarray(gen_x)
        except:
            raise ValueError(
                f"gen_x need to be a 1D numpy array but was of type "
                f"{type(gen_x)}")
        if gen_x.ndim != 1:
            raise ValueError(
                f"gen_x needs to be a 1D numpy array, but was of shape "
                f"{gen_x.shape}")

        try:
            gen_y = np.asarray(gen_y)
        except:
            raise ValueError(
                f"gen_y need to be a 1D numpy array but was of type "
                f"{type(gen_y)}")
        if gen_y.ndim != 1:
            raise ValueError(
                f"gen_y needs to be a 1D numpy array, but was of shape "
                f"{gen_y.shape}")


        self.gen_x, self.gen_y = gen_x, gen_y

        if isinstance(dataset, dict) and \
                np.all([isinstance(dat, Data) for dat in dataset.values()]):
            if even_space_y is not None:
                dataset = {key:dat.interp_step(even_space_y) \
                    for key, dat in dataset.items()}

            self.input_y, self.input_z, self.input_x = \
                parse.unroll_dataset(dataset, look_up=look_up)
            self.input_x = self.input_x.astype(np.float_)
            if key_y:
                self.input_x, self.input_y = self.input_y, self.input_x
        elif isinstance(dataset, np.ndarray) and dataset.ndim == 2 \
                and dataset.shape[1] == 3:
            self.input_x, self.input_y, self.input_z = dataset.T
        else:
            raise ValueError(
                f"dataset needs to be a dict of Data objects or a 3 column "
                f"numpy array but was of type {type(dataset)}")

        
        self.gen_xx, self.gen_yy = np.meshgrid(gen_x, gen_y)
        
        if normalise_xy:
            if isinstance(normalise_xy, (tuple, list)) and \
                    len(normalise_xy) == 2:
                norx, nory = normalise_xy
            else:
                norx, nory = 1., 1.
            
            self.input_x = norx*_norm_array(self.input_x, self.gen_x)
            self.input_y = nory*_norm_array(self.input_y, self.gen_y)
            self.gen_xx = norx*_norm_array(self.gen_xx, self.gen_x)
            self.gen_yy = nory*_norm_array(self.gen_yy, self.gen_y)
        
        self.kernel = None
        self._input_kernel = None
        self._kernel_type = None
        self._new_kernel = True
        self.kernel_args = {}
        self.white_noise = None
        self.kmat_inv = None
        self.predict_z = None
    
    def _make_dis_kernel(self):
        """Make the self.kernel function using the current attributes.
        """
        def kernel(x1, y1, x2, y2):
            
            xx1, xx2 = np.meshgrid(x1, x2)
            yy1, yy2 = np.meshgrid(y1, y2)
            
            dis = np.sqrt((xx1 - xx2)**2 + (yy1 - yy2)**2)
            
            return self._input_kernel(dis, **self.kernel_args)
        
        self.kernel = kernel


    def _make_xy_kernel(self):
        """Make the self.kernel function using the current attributes.
        """
        def kernel(x1, y1, x2, y2):
            
            xx1, xx2 = np.meshgrid(x1, x2)
            yy1, yy2 = np.meshgrid(y1, y2)
            
            return self._input_kernel(xx1, yy1, xx2, yy2, **self.kernel_args)
        
        self.kernel = kernel

    def _make_kernel(self):
        if self._kernel_type == 'dis':
            self._make_dis_kernel()
        elif self._kernel_type == 'xy':
            self._make_xy_kernel()
        else:
            raise ValueError(
                f"_kernel_type was not 'xy' or 'dis' but was "
                f"{self._kernel_type}.")

    def set_distance_kernel(self, dis_kernel, white_noise, **kernel_args):
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
        self._input_kernel = dis_kernel
        self._kernel_type = 'dis'
        self.white_noise = white_noise
        self.kernel_args = kernel_args
        self._new_kernel = True
        
        self._make_dis_kernel()
        
    def set_xy_kernel(self, xy_kernel, white_noise, **kernel_args):
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
        self._input_kernel = xy_kernel
        self._kernel_type = 'xy'
        self.kernel_args = kernel_args
        self.white_noise = white_noise
        self._new_kernel = True
        
        self._make_xy_kernel()
    
    def _make_kmat_inv(self):
        """Calculates the kernel matrix inverse.
        """
        if self.kernel is None:
            raise AttributeError(
                "A kernel needs to set a kernel before the prediction can "
                "be preformed. This can be done with either the method "
                "set_distance_kernel or set_xy_kernel.")

        self._new_kernel = False
            
        self.kmat_inv = np.linalg.inv(
            self.kernel(self.input_x, self.input_y, 
                self.input_x, self.input_y)  + \
            self.white_noise*np.identity(self.input_x.size))

    def set_kernel_args(self, **kernel_args):
        """Set keyword arguments for the kernel function.

        This allows new kernel keyword values to be set without resupplying 
        the kernel and all the other keyword values. This keeps the values 
        already set unless they are written over. The values are supplied 
        by providing keyword arguments to this function.
        """

        for key, val in kernel_args.items():
            self.kernel_args[key] = val
        self._new_kernel = True 

    def calculate_log_mlh(self):
        """Calculate and return the negative log marginal likelihood.

        This is the scaler that needs to be minimised to compare the values 
        of kernel parameters. This recalculates the values in a way to try 
        and speed things in the minimisation process.

        Returns
        -------
        neg_log_marg_lh : float
            The negative log marginal likelihood for the current kernel and 
            data provided.
        """
        k_mat = self.kernel(self.input_x, self.input_y, 
            self.input_x, self.input_y)
        k_mat_inv = np.linalg.inv(k_mat + \
            self.white_noise*np.identity(self.input_x.size))
        z_diff = self.input_z - k_mat.T @ k_mat_inv @ self.input_z
        s_det, log_det = np.linalg.slogdet(k_mat + 
            self.white_noise*np.identity(self.input_x.size))
        like = z_diff.T @ k_mat_inv @ z_diff

        return log_det/2. + like/2. + z_diff.size*np.log(2.*np.pi)/2.

    def optermise_argument(self, arguments, **kwargs):
        """Minimise the negative log marginal likelihood.

        This uses :func:`scipy.optimize.minimize` to change the value of 
        keyword arguments to minimise the value from 
        :meth:`calculate_log_mlh`. This should take both into account the 
        model complexity and the quality of the kit to the data.
        Keyword arguments are passed to :func:`scipy.optimize.minimize`, a 
        very useful one is `bounds` which is a list of tuples of the lower 
        then upper bounds.

        Parameters
        ----------
        arguments : dict
            A dictionary of the keywords for the kernel function and the 
            initial values to start the optimisation from.

        Returns
        -------
        minimize_result : scipy.optimize.OptimizeResult
            The result from the running of :func:`scipy.optimize.minimize`, 
            the `x` argument of the result is also set to the 
            :attr:`kernel_args` attribute.
        """
        x0 = [x for x in arguments.values()]
        keys = [k for k in arguments.keys()]

        def to_min(x_temp):
            self.set_kernel_args(**{
                keys[n]:x_temp[n] for n in range(len(x_temp))
                })
            return self.calculate_log_mlh()
        
        min_res = minimize(to_min, x0, **kwargs)
        self.set_kernel_args(**{
            keys[n]:min_res.x[n] for n in range(len(x0))
            })

        return min_res
    
    def predict(self, cut_outside=False, new_invert=False, no_return=False,
            cap_z=None):
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
        new_invert : bool, optional
            If 'True' then the kernel will be inverted again. If the default 
            of 'False' then the kernel will only be recalculated if it has 
            been updated. If the kernel is updated by addressing the 
            attribute then to be recalculated this need to be set to 'True' 
            for the new kernel to be used.
        no_return : bool, optional
            If the default of 'False' the prediction will be returned. If 
            'True' then nothing will be.
        cap_z, tuple
            If a tuple of floats is given then the :attr:`predict_z` is 
            capped between the two values given using :meth:`cap_min_max`. 
            If None is given then the cap isn't performed. 

        Returns
        -------
        predict_z : numpy.ndarray
            A 2D array of the values of the calculated z values in the 
            locations of :attr:`gen_x` and :attr:`gen_y`. This function 
            also sets the attribute :attr:`predict_z`.

        """
        if self.kernel is None:
            raise AttributeError(
                "A kernel needs to set a kernel before the prediction can "
                "be preformed. This can be done with either the method "
                "set_distance_kernel or set_xy_kernel.")
        
        if self._new_kernel or new_invert:
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

        if cap_z is not None:
            if len(cap_z) != 2:
                raise ValueError(
                    f"cap_z should be a tuple of length 2 but was a type "
                    f"{type(cap_z)} .")
            self.cap_min_max(cap_z[0], cap_z[1])
        
        if not no_return:
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
        if self.predict_z is None:
            raise AttributeError(
                "The result needs to be generated using the predict "
                "method before the result can be cut to a hull.")

        hull = ConvexHull(np.concatenate(
            [self.input_x[:, None], self.input_y[:, None]], axis=1))
        hx, hy, ho = hull.equations.T
        
        out_hull = np.any(
            np.outer(self.gen_xx, hx) + \
            np.outer(self.gen_yy, hy) + \
            np.outer(np.ones(self.gen_xx.shape), ho) >= -tol, axis=1)
        
        out_hull = out_hull.reshape(self.gen_xx.shape)
        
        self.predict_z[out_hull] = np.nan

    def cap_min_max(self, z_min, z_max):
        """Caps the z values between a minimum and maximum values

        This changed the :attr:`predict_z` attribute so that values above 
        and bellow a range are caped to the values. This can be useful for 
        trimming unphysical values or cutting out extremes from 
        extrapolation.

        Parameters
        ----------
        z_min : float, None
            If a float is given then the all the values bellow this value 
            are changed to equal this value. If `None` is given then a cap 
            isn't preformed.
        z_max : float, None
            If a float is given then the all the values above this value 
            are changed to equal this value. If `None` is given then a cap 
            isn't preformed.
        """
        if self.predict_z is None:
            raise AttributeError(
                "The result needs to be generated using the predict "
                "method before the result can be cap between z values.")

        if z_min is not None and z_max is not None and z_min > z_max:
            raise ValueError(
                f"The value of z_min ({z_min}) was larger than the value "
                f"of z_max ({z_max}). Therefore capping of the z_values is "
                f"not possible.")

        if z_min is not None:
            self.predict_z[self.predict_z < z_min] = z_min
        if z_max is not None:
            self.predict_z[self.predict_z > z_max] = z_max

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
        if self.predict_z is None:
            raise AttributeError(
                "The result needs to be generated using the predict "
                "method before the result can be plotted.")
        r_gen_xx, r_gen_yy = np.meshgrid(self.gen_x, self.gen_y)
        
        return r_gen_xx, r_gen_yy, self.predict_z
    
    def plot_contourf(self, colorbar_kwargs={}, **kwargs):
        """Plot the generated data as a contour map

        This makes use of :func:`matplotlib.pyplot.contorf` and keyword 
        arguments are passed to it. 
        Keyword arguments can be passed to the 
        colour bar by setting the keyword argument `colorbar_kwargs` to a 
        dictionary. This uses :func:`matplotlib.pyplot.colorbar`.
        """
        plt.contourf(*self.plotting_arrays(), **kwargs)
        plt.colorbar(**colorbar_kwargs)
        
    def plot_contour(self, colorbar_kwargs={}, **kwargs):
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


def gaussian_kernel(x1, y1, x2, y2, const=0., amp=1., length=1.):
    """A Gaussian kernel for contour fitting.

    This is a simple Gaussian kernel for use with 
    :meth:`GP_map.set_xy_kernel`. It has an equation of the form
    ``K = const + amp*exp(((x1 - x2)**2 + (y1 - y2)**2)/length**2)``. 
    The keyword arguments can be set when they are passed through 
    :meth:`GP_map.set_xy_kernel`.

    Parameters
    ----------
    x1: numpy.ndarray
        The four arguments are arrays which contain the x and y values from 
        the points to generate the appropriate kernel matrix. These arrays 
        are all the same size.
    y1 : numpy.ndarray
        See above.
    x2 : numpy.ndarray
        See above.
    y2 : numpy.ndarray
        See above.
    const : float, optional
        A constant term that changes the background level. Default is ``0``
    amp : float, optional
        The amplitude of the Gaussian. The default is ``1``
    length : float, optional
        The length scale of the Gaussian. The default is ``1``

    Returns
    -------
    kernel_mat : numpy.ndarray
        A :class:`numpy.ndarray` the same size as the input arrays with the 
        kernel matrix elements.

    """
    return const + amp*np.exp(-((x1 - x2)**2 + (y1 - y2)**2)/2/length**2)


def elliptical_gaussian_kernel(x1, y1, x2, y2, const=0., amp=1., 
        x_length=1., y_length=1., angle=0.):
    """An elliptical Gaussian kernel for contour fitting.

    This is a simple Gaussian kernel for use with 
    :meth:`GP_map.set_xy_kernel`.  
    The keyword arguments can be set when they are passed through 
    :meth:`GP_map.set_xy_kernel`.

    Parameters
    ----------
    x1: numpy.ndarray
        The four arguments are arrays which contain the x and y values from 
        the points to generate the appropriate kernel matrix. These arrays 
        are all the same size.
    y1 : numpy.ndarray
        See above.
    x2 : numpy.ndarray
        See above.
    y2 : numpy.ndarray
        See above.
    const : float, optional
        A constant term that changes the background level. Default is ``0``
    amp : float, optional
        The amplitude of the Gaussian. The default is ``1``
    x_length : float, optional
        The length scale of the x component Gaussian. The default is ``1``
    y_length : float, optional
        The length scale of the y component Gaussian. The default is ``1``
    angle : float, optional
        The angle in radians to rotate the x and y contributions the default 
        is ``0`` which keeps the x and y values independent.

    Returns
    -------
    kernel_mat : numpy.ndarray
        A :class:`numpy.ndarray` the same size as the input arrays with the 
        kernel matrix elements.

    """
    x_dis = (x1 - x2)
    y_dis = (y1 - y2)

    x_dis2 = np.power(
        (x_dis*np.cos(angle) + y_dis*np.sin(angle))/x_length, 2)/2.
    y_dis2 = np.power(
        (y_dis*np.cos(angle) - x_dis*np.sin(angle))/y_length, 2)/2.


    return const + amp*np.exp(-(x_dis2 + y_dis2))


def linear_kernel(x1, y1, x2, y2, const=1., amp=1., x_scale=1., y_scale=1.):
    """A linear kernel for contour fitting.

    This is a simple linear kernel for use with 
    :meth:`GP_map.set_xy_kernel`. It has an equation of the form
    ``K = const + amp*(x1*x2/x_scale**2+ y1*y2/y_scale**2)``. The keyword 
    arguments can be set when they are passed through 
    :meth:`GP_map.set_xy_kernel`. There are much faster ways of doing this 
    than with Gaussian processes the utility of this function is to be 
    combined with others. Also amp, x_scale, and y_scale over define the 
    function so don't optimise on all at once; they are included to help
    to relate to physical properties.

    Parameters
    ----------
    x1: numpy.ndarray
        The four arguments are arrays which contain the x and y values from 
        the points to generate the appropriate kernel matrix. These arrays 
        are all the same size.
    y1 : numpy.ndarray
        See above.
    x2 : numpy.ndarray
        See above.
    y2 : numpy.ndarray
        See above.
    const : float, optional
        A constant term that changes the background level. The default 
        is ``0``
    amp : float, optional
        The amplitude of the linear term. The default is ``1``
    x_scale : float, optional
        The scaling of the x values in the same units as x. The default is 
        ``1``.
    y_scale : float, optional
        The scaling of the y values in the same units as y. The default is 
        ``1``.

    Returns
    -------
    kernel_mat : numpy.ndarray
        A :class:`numpy.ndarray` the same size as the input arrays with the 
        kernel matrix elements.

    """
    return const + amp*(x1*x2/x_scale**2+ y1*y2/y_scale**2)


def rational_quadratic_kernel(x1, y1, x2, y2, const=0., amp=1., length=1.,
    scale=1.):
    """A rational quadratic kernel for contour fitting.

    This is a rational quadratic kernel for use with 
    :meth:`GP_map.set_xy_kernel`. It has an equation of the form
    It has an equation of the form
    ``K = const + amp*(1 + ((x1 - x2)**2 + 
    (y1 - y2)**2)/2/length**2/scale)**scale``. The keyword arguments can be 
    set when they are passed through :meth:`GP_map.set_xy_kernel`. This can 
    be thought of a combination of many different Gaussian kernels to 
    different powers. These are the same when the scale goes to infinity.

    Parameters
    ----------
    x1: numpy.ndarray
        The four arguments are arrays which contain the x and y values from 
        the points to generate the appropriate kernel matrix. These arrays 
        are all the same size.
    y1 : numpy.ndarray
        See above.
    x2 : numpy.ndarray
        See above.
    y2 : numpy.ndarray
        See above.
    const : float, optional
        A constant term that changes the background level. The default 
        is ``0``
    amp : float, optional
        The amplitude of the rational quadratic term. The default is ``1``
    length : float, optional
        The length scale of the kernel. The default is ``1``
    scale : float, optional
        The scaling function between order terms. The default is ''1''

    Returns
    -------
    kernel_mat : numpy.ndarray
        A :class:`numpy.ndarray` the same size as the input arrays with the 
        kernel matrix elements.

    """
    return const + \
        amp*np.power(1. + ((x1 - x2)**2 + (y1 - y2)**2)/2./length**2/scale, 
            -scale)

