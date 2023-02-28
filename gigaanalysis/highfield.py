"""GigaAnalysis - High Field - :mod:`gigaanalysis.highfield`
---------------------------------------------------------------

This program has a series of useful tools for conducting experiments in
certain high field labs.
"""

from .data import *
from . import diglock

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter  # For PulsedLockIn.lockin_Volt_smooth
from scipy.interpolate import interp1d # For example_field


def example_field(max_field, peak_time, length, sample_rate, as_Data=False):
    """Produces a data set with a field profile.

    This field profile matches the KS3 magnet in HLD. The pulse time and
    maximum field has been normalised, and is get by the user. Originally 
    they where 68.9 Tesla and 0.0336 seconds. I find this useful for 
    simulating test measurements.

    Parameters
    ----------
    max_field : float
        The maximum field value that the data will reach.
    peak_time : float
        The time that the magnet reaches peak field.
    length : int
        The number of data points in the file.
    sample_rate: float
        The sample frequency of the data.
    as_Data : bool, optional
        If true a :class:`.Data` class is returned.

    Returns
    -------
    field_profile : numpy.ndarray or Data
        The field values simulated for the parameters given.
    """
    field = np.array([
        -0.000e+00,  0.000e+00,  4.660e-02,  9.440e-02,  1.376e-01,
        1.785e-01,  2.182e-01,  2.569e-01,  2.949e-01,  3.322e-01,
        3.687e-01,  4.046e-01,  4.397e-01,  4.740e-01,  5.076e-01,
        5.402e-01,  5.719e-01,  6.027e-01,  6.326e-01,  6.614e-01,
        6.892e-01,  7.160e-01,  7.416e-01,  7.662e-01,  7.896e-01,
        8.118e-01,  8.328e-01,  8.527e-01,  8.713e-01,  8.887e-01,
        9.048e-01,  9.197e-01,  9.333e-01,  9.456e-01,  9.567e-01,
        9.665e-01,  9.750e-01,  9.823e-01,  9.883e-01,  9.931e-01,
        9.966e-01,  9.989e-01,  9.999e-01,  9.998e-01,  9.984e-01,
        9.958e-01,  9.920e-01,  9.871e-01,  9.810e-01,  9.738e-01,
        9.655e-01,  9.561e-01,  9.457e-01,  9.343e-01,  9.221e-01,
        9.091e-01,  8.954e-01,  8.810e-01,  8.660e-01,  8.505e-01,
        8.345e-01,  8.180e-01,  8.012e-01,  7.840e-01,  7.666e-01,
        7.490e-01,  7.311e-01,  7.131e-01,  6.950e-01,  6.769e-01,
        6.587e-01,  6.405e-01,  6.223e-01,  6.042e-01,  5.862e-01,
        5.683e-01,  5.505e-01,  5.329e-01,  5.154e-01,  4.982e-01,
        4.812e-01,  4.644e-01,  4.479e-01,  4.316e-01,  4.156e-01,
        3.999e-01,  3.845e-01,  3.693e-01,  3.545e-01,  3.400e-01,
        3.259e-01,  3.120e-01,  2.985e-01,  2.854e-01,  2.725e-01,
        2.600e-01,  2.479e-01,  2.360e-01,  2.245e-01,  2.134e-01,
        2.026e-01,  1.921e-01,  1.819e-01,  1.721e-01,  1.626e-01,
        1.534e-01,  1.445e-01,  1.360e-01,  1.277e-01,  1.197e-01,
        1.121e-01,  1.047e-01,  9.760e-02,  9.080e-02,  8.430e-02,
        7.800e-02,  7.200e-02,  6.620e-02,  6.070e-02,  5.550e-02,
        5.040e-02,  4.560e-02,  4.110e-02,  3.670e-02,  3.260e-02,
        2.860e-02,  2.490e-02,  2.140e-02,  1.800e-02,  1.480e-02,
        1.180e-02,  9.000e-03,  6.300e-03,  3.800e-03,  1.400e-03,
        -8.000e-04, -2.900e-03, -4.800e-03, -6.600e-03, -8.300e-03,
        -9.800e-03, -1.130e-02, -1.260e-02, -1.390e-02, -1.500e-02,
        -1.600e-02, -1.700e-02, -1.790e-02, -1.860e-02, -1.930e-02,
        -2.000e-02, -2.050e-02, -2.100e-02, -2.140e-02, -2.180e-02,
        -2.210e-02, -2.230e-02, -2.250e-02, -2.260e-02, -2.270e-02,
        -2.280e-02, -2.280e-02, -2.280e-02, -2.270e-02, -2.260e-02,
        -2.250e-02, -2.230e-02, -2.210e-02, -2.190e-02, -2.170e-02,
        -2.140e-02, -2.120e-02, -2.090e-02, -2.060e-02, -2.020e-02,
        -1.990e-02, -1.950e-02, -1.910e-02, -1.870e-02, -1.830e-02,
        -1.790e-02, -1.740e-02, -1.700e-02, -1.650e-02, -1.600e-02,
        -1.550e-02, -1.500e-02, -1.450e-02, -1.400e-02, -1.340e-02,
        -1.290e-02, -1.230e-02, -1.180e-02, -1.120e-02, -1.070e-02,
        -1.010e-02, -9.500e-03, -8.900e-03, -8.400e-03, -7.800e-03,
        -7.200e-03, -6.600e-03, -6.000e-03, -5.400e-03, -4.900e-03,
        -4.300e-03, -3.700e-03, -3.100e-03, -2.600e-03, -2.000e-03,
        -1.500e-03, -9.000e-04, -4.000e-04, -2.000e-04, -2.000e-04,
        -1.000e-04, -1.000e-04, -1.000e-04, -0.000e+00, -0.000e+00,
        -0.000e+00, -0.000e+00, -0.000e+00, -0.000e+00, -0.000e+00,])
    time = np.arange(225)/42.
    field_interp = interp1d(peak_time*time, max_field*field, 
        kind='linear', fill_value=0, bounds_error=False, assume_sorted=True)
    sim_time = np.arange(length)/sample_rate
    sim_field = field_interp(sim_time)

    if as_Data:
        return Data(sim_time, sim_field)
    else:
        return sim_field


def PUtoB(PU_signal, field_factor, fit_points, to_fit='PU'):
    """Converts the voltage from the pick up coil to field.

    This is used for pulsed field measurements, where to obtain the filed
    the induced voltage in a coil is integrated. A fit is also applied 
    because slight differences in the grounding voltage can cause a large 
    change in the field so this needs to be corrected for.

    Parameters
    ----------
    PU_signal : numpy.ndarray, Data
        The signal from pick up coil.
    field_factor : float
        Factor to convert integral to magnetic field. Bare in mind this will 
        change if the acquisition rate changes, for the same coil.
    fit_points : int
        Number of point at each end to remove offset.
    to_fit : {'PU', 'field'} optional
        If to correct an offset voltage the PU signal is fit or the field.
    
    Returns
    -------
    field : numpy.ndarray, Data
        An array of magnetic field the same length as PU_signal. If a 
        :class:`.Data` is given then the y values are processed and a 
        :class:`.Data` is returned.
    
    """
    as_data = isinstance(PU_signal, Data)
    if as_data:
        x_vals, PU_signal = PU_signal.both

    count = np.arange(len(PU_signal))
    ends = np.concatenate([count[:fit_points], count[-fit_points:]])

    if to_fit == 'PU':
        a, b = np.polyfit(ends, PU_signal[ends], 1)
        PU_flat = PU_signal - a*count - b
        field_output = np.cumsum(PU_flat*field_factor)
    elif to_fit == 'field':
        field = np.cumsum(PU_signal*field_factor)
        a, b = np.polyfit(ends, field[ends], 1)
        field_output = field - a*count - b
    else:
        raise ValueError(
            f"to_fit needs to be either 'PU' or 'field' but was {to_fit}.")

    if as_data:
        return Data(x_vals, field_output)
    else:
        return field_output


def pick_pulse_side(field, B_min, side, skip_num=1, give_slice=True):
    """Produces a slice that selects a certain section of a filed pulse.

    This takes a field profile and produces a slice that has one side or 
    both in.

    Parameters
    ----------
    field : numpy.ndarray
        Field values in a 1D numpy array, the field needs to be pulsed in 
        the positive direction. If you want to analyse a negative sweep 
        first take the negative of it.
    B_min : float or None
        The value of field to cut all the lower data off. This is used as 
        sometimes the tails of the pulses can be very long. If it is set to 
        None the full range is kept.
    side : {'up', 'down', 'both'}
        Which side of the the pulse to take. 'up' takes the first side, 
        'down' takes the second, and 'both' includes both sides of the 
        pulse.
    skip_num : int, optional
        The ratio of points to skip to reduce the size of the data set. The 
        default is `1`, which doesn't slip any points.
    give_slice : bool, optional
        If the default of `True` a slice is returned as described. If 
        `False` then the field is returned with the slice applied to it.

    Returns
    -------
    B_slice : slice, numpy.ndarray
        The slice to apply to take one field side. If `give_slice` is 
        `False` then the filed array is returned with the slice applied.
    """
    if abs(np.max(field)) < abs(np.min(field)):
        raise ValueError(
            f"The field goes in the negative direction this function only "
            f"takes positive field pulses.")

    maxB_arg = np.argmax(field)
    peak_field = field[maxB_arg]

    if B_min is not None and B_min > peak_field:
        raise ValueError(
            f"The maximum value of the field is smaller than the given "
            f"value of B_min.")

    if side not in ['up', 'down', 'both']:
        raise ValueError(
            f"side was given as {side} but can only be 'up', 'down', or "
            f"'both'.")

    if side == 'down' and maxB_arg == len(field) - 1:
        raise ValueError(
            f"down was selected but the highest field was at the end of the "
            f"field profile given.")
    elif side == 'up' and maxB_arg == 0:
        raise ValueError(
            f"up was selected but the highest field was at the start of the "
            f"field profile given.")

    if B_min is None:
        pass
    elif side == 'down' and B_min > np.min(field[maxB_arg:]):
        pass
    elif side == 'up' and B_min > np.min(field[:maxB_arg]):
        pass
    elif side == 'both' and B_min > np.min(field):
        pass
    else:
        raise ValueError(
            f"The value of B_min was below the lowest field value in the "
            f"side you selected.")

    if side == 'down':
        startB_arg = maxB_arg
        if B_min is None:
            endB_arg = None
        else:
            endB_arg = maxB_arg + \
                np.where(field[maxB_arg:]<B_min)[0][0]

        B_slice = slice(endB_arg, startB_arg, -skip_num)
    elif side == 'up':
        if B_min is None:
            startB_arg = None
        else:
            startB_arg = np.where(field>B_min)[0][0]

        endB_arg = maxB_arg
        B_slice = slice(startB_arg, endB_arg, skip_num)
    elif side == 'both':
        if B_min is None:
            startB_arg, endB_arg = None, None
        else:
            startB_arg, endB_arg = np.where(field>B_min)[0][[0, -1]]

        B_slice = slice(startB_arg, endB_arg, skip_num)
    else:
        raise ValueError("side must be either 'up', 'down', or 'both'.")

    if give_slice:
        return B_slice
    else:
        return field[B_slice]


class PulsedLockIn():
    """Performs a digital lock in on pulse field magnetotransport data.

    This class is used to process data from pulsed field measurements using 
    digital phase sensitive detection. The class is designed and named for 
    it to be used for magnetotransport measurements, it can and has also 
    been used for other types of experiments such as torque magnetometry. 
    The type lock-in process it uses is convolution with a Hamming window.

    As well as the simple phase sensitive detection functionality it also 
    has tools for finding the phase shift, smoothing signal, and filtering 
    out voltage spikes. The data produces can be accessed from the 
    attributes or output as a :class:`.Data` object using one of the 
    methods.
    

    Parameters
    ----------
    field : numpy.ndarray
        Field values in Tesla sorted in a 1D numpy array. The field will be
        changed to positive field is a negative sweep is given.
    current : numpy.ndarray
        Voltage reading across a shunt resistor in the form of a 1D numpy 
        array.
    voltage : numpy.ndarray
        Voltage readings in the measurement channel in the form of a 1D
        numpy array.
    sample_freq : float, optional
        The rate of the data acquisition in Hertz. The default is `15e6` 
        which is a common pulse field sample frequency.
    R_shunt : float, optional
        Value of the shunt resistor to measure the current in Ohms. The 
        default value is `100`.
    preamp : float, optional
        Value of the amplification of the voltage signal before being 
        measured. The default is `1`, so assumes no amplification.
    skip_num : int, optional
        The ratio of points to skip when outputting the data. This is used 
        because the object sizes can become unwieldy if all the data is 
        saved. The default is `200`, which drops 199 points for every one 
        it keeps.
    B_min : float or None, optional
        The minimum value of the field to keep points lower in field to this 
        will be dropped. If set to `None` all of the field range is kept.
        The default value is `0`, which only drops negative field values.
    side : {'up', 'down', 'both'}, optional
        The side of the pulse to produce the data for. 'up' is the first 
        side, 'down' is the second, and 'both' takes both sides. The default 
        is 'down'.
    
    Attributes
    ----------
    field : numpy.ndarray
        The given numpy array containing the field values, if this is a 
        negative field pulse then the sign of the field values are inverted.
    Iv : numpy.ndarray
        The numpy array given for the measurement current voltage.
    Volt : numpy.ndarray
        The numpy array given for the measurement voltage.
    time : numpy.ndarray
        The time values in milliseconds the same size as the given arrays.
    fs : float
        The sample frequency given in Hertz.
    R_shunt : float
        The given shunt voltage which is used to converted the measurement 
        current voltage into current.
    preamp : float
        The given amplification that is used to convert the measured voltage 
        into the generated voltage.
    field_direction : bool
        `True` if it is a positive pulse, `False` if it is a negative pulse.
    peak_field : float
        The maximum field value reached in the magnet pulse.
    slice : slice
        The slice that selects the data of interest out of the complete 
        measurement. This is set by the `B_min`, `side`, and `step_size` 
        keywords.
    freq : float
        The frequency of the applied measurement current voltage.
    phase : float
        The phase shift from the start of the file of the measurement 
        current voltage in degrees.
    Irms : float
        The average applied current in root mean squared Amps.
    time_const : float
        The given time constant used for the voltage lock in seconds.
    phase_shift : float
        The given phase shift between the current measurement voltage and the 
        experiment measurement voltage in degrees.
    loc_Volt : numpy.ndarray
        The experimental voltage after the lock in process considering the 
        amplification in root mean squared Volts.
    loc_Volt_out : numpy.ndarray
        Equivalent to `loc_Volt` but for the out of phase component of the 
        experimental voltage.
    loc_I : numpy.ndarray
        The applied current after a lock in process in root mean squared 
        Amps.
    """
    def __init__(self, field, current, voltage, sample_freq=15e6,
            R_shunt=100., preamp=1., skip_num=200, B_min=0., side='down'):
        for name, array in {'field':field, 'current':current, 
                'voltage':voltage}.items():
            if not isinstance(array, np.ndarray):
                raise TypeError(
                    f"{name} needs to be a numpy array but was of type "
                    f"{type(array)}.")
            elif not np.isfinite(array).all():
                raise ValueError(
                    f"The array {name} contains non-finite values.")
            elif array.ndim != 1:
                raise ValueError(
                    f"{name} needs to be a 1D array but had shape "
                    f"{array.shape}.")

        if field.size != current.size or field.size != voltage.size:
            raise ValueError(
                f"field, current, and voltage all need to be the same "
                f"size but had sizes {field.size}, {current.size}, and "
                f"{voltage.size}.")

        # Save the original data
        self.field = field
        self.Iv = current
        self.Volt = voltage
        self.preamp = preamp
        self.R_shunt = R_shunt

        # Generate time
        self.fs = sample_freq
        self.time = 1000*np.arange(len(self.field))/self.fs

        # Make field positive,and save direction
        self.field_direction = (abs(np.max(self.field)) > 
            abs(np.min(self.field)))
        if not self.field_direction:
            self.field = -self.field

        self.peak_field = np.max(self.field)

        # Make the slice
        self.slice = pick_pulse_side(self.field, B_min, side,
            skip_num=skip_num)

        # Find frequencies and phases
        self.freq = diglock.find_freq(self.Iv, self.fs, padding=10)
        self.phase = diglock.find_phase(self.Iv, self.fs, self.freq)
        self.Irms = np.average((self.Iv*diglock.gen_ref(self.freq, self.fs,
            self.phase, len(self.Iv)))[1000:-1000])*np.sqrt(2)/R_shunt

        # Attributes to be set
        self.time_const = None
        self.phase_shift = 0
        self.loc_Volt = None
        self.loc_Volt_out = None
        self.loc_I = None

    def _has_locked(self):
        """This is just for checking that the lock in process has happened 
        before trying to output or manipulate the locked in data.
        """
        if self.loc_Volt is None:
            raise AttributeError(
                f"The lock in process has not yet been performed. Please "
                f"call the method lockin_Volt first.")

    def _has_locked_I(self):
        """This is just for checking that the lock in process has happened 
        before trying to output or manipulate the locked in data.
        """
        if self.loc_I is None:
            raise AttributeError(
                f"The lock in process on the current has not yet been "
                f"performed. Please call the method lockin_current first.")

    def lockin_Volt(self, time_const, phase_shift=None):
        """This preforms a lock in process on the measurement signal.

        This method sets the attributes :attr:`loc_Volt` and 
        :attr:`loc_Volt_out`. The lock in process is performed using 
        :func:`.diglock.ham_lock_in`. Does not return anything.
        
        Parameters
        ----------
        time_const : float
            Time for averaging in seconds.
        phase_shift : float, optional
            Phase difference between the current voltage and the measurement 
            voltage, this defaults to the attribute :attr:`phase_shift`. 
            This is in degrees.
        """
        self.time_const = time_const
        if phase_shift is not None:
            self.phase_shift = phase_shift
        self.loc_Volt = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + self.phase_shift,
            )/self.preamp
        self.loc_Volt_out = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + self.phase_shift + 90.,
            )/self.preamp

    def lockin_current(self, time_const, phase=0,):
        """This preforms a lock in process on the current signal.

        This uses :func:`.diglock.ham_lock_in` to perform the lock in and 
        sets the attribute :attr:`loc_I`.
        
        Parameters
        ----------
        time_const : float
            Time for averaging in seconds.
        phase : float, optional
            An applied phase shift for the current, in degrees. The default 
            value is 0.
        """
        self.loc_I = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + phase,)/self.R_shunt

    def spike_lockin_Volt(self, time_const, phase_shift=None,
            sdl=2, region=1001):
        """Spike removing lock in process applied to the measurement signal.
        
        This preforms a lock in process on the measurement signal, with the
        aim of removing spikes in the raw first. This can be useful as some 
        magnets can see high voltage spikes.
        This uses :func:`.diglock.spike_lock_in`.
        Nothing is returned but the following attributes are updated, 
        :attr:`time_const`, :attr:`phase_shift`, :attr:`loc_Volt`, and 
        :attr:`loc_Volt_out`.
        
        Parameters
        ----------
        time_const : float
            Time for averaging in seconds.
        phase_shift : float, optional
            Phase difference between the current voltage and the measurement 
            voltage in degrees. This defaults to the attribute 
            :attr:`phase_shift`.
        sdl : float, optional
            The number of standard deviations the data to be deviate from to 
            be considered a outlier. Outliers are identified as spikes. 
            The default is 2.
        region : int, optional
            The number of points around the outlier that are also considered 
            compromised. The default is 1001.
        """
        self.time_const = time_const
        if phase_shift is not None:
            self.phase_shift = phase_shift
        self.loc_Volt = diglock.spike_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + self.phase_shift,
            sdl=sdl, region=region,)/self.preamp
        self.loc_Volt_out = diglock.spike_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + self.phase_shift + 90.,
            sdl=sdl, region=region,)/self.preamp

    def smooth_Volts(self, smooth_time, smooth_order=2):
        """This smooths the measurement signal.
        
        This must be applied after the lock in process. It changes the 
        attributes :attr:`loc_Volt` and :attr:`loc_Volt_out`. The smoothing 
        is done with a pass of a Savitzky-Golay filter from
        :func:`scipy.signal.savgol_filter`. This is particularly useful to 
        remove small aliasing issues that can arise when using a short 
        lock in window.
        
        Parameters
        ----------
        smooth_points : float
            The time window to fit the polynomial for smoothing, in seconds.
        smooth_order : int, optional
            The order of the poly to fit for the smoothing, the default is 2.
        """
        self._has_locked()
        smooth_points = int(np.ciel(smooth_time/self.fs/2)*2 + 1)
        self.loc_Volt = savgol_filter(self.loc_Volt, smooth_points,
            smooth_order,)
        self.loc_Volt_out = savgol_filter(self.loc_Volt_out, smooth_points,
            smooth_order,)

    def rephase(self, phase_shift, trial=False):
        """Rephases the signal to the new phase.
        
        This changes the attributes :attr:`loc_Volt` and 
        :attr:`loc_Volt_out` to shift the phase by a certain amount. If the 
        `trial` is set to `True` then the result is returned instead 
        of updating the attributes. The phase shift given is absolute and 
        :attr:`phase_shft` is also updated.

        Parameters
        ----------
        phase_shift : float
            The new phase shift to use in degrees. This is absolute so the 
            result is independent to the current phase shift.
        trial : bool, optional
            Whether to keep the new rephasing or to return the result 
            instead. The default value is `False` which updates the 
            attributes and returns nothing.
        """
        self._has_locked()
        phase_difference = phase_shift - self.phase_shift
        v_in_new  = self.loc_Volt*np.cos(phase_difference*np.pi/180) + \
            self.loc_Volt_out*np.sin(phase_difference*np.pi/180)
        v_out_new = self.loc_Volt_out*np.cos(phase_difference*np.pi/180) - \
            self.loc_Volt*np.sin(phase_difference*np.pi/180)
        if trial == False:
            self.phase_shift = phase_shift
            self.loc_Volt = v_in_new
            self.loc_Volt_out = v_out_new
        else:
            return v_in_new, v_out_new

    def auto_phase(self, aim='change'):
        """Finds the value of the phase_shift to achieve a certain result.

        This finds the phase which makes the out of phase a flat as possible
        and also has the in phase be majority positive. It can also be set 
        to move the majority of the signal into the in-phase channel using 
        the `aim` parameter. Uses :func:`.diglock.phase_in`.
        
        Parameters
        ----------
        aim : {'change', 'value'}
            Weather to minimise the change in signal or the signal total in 
            the out of phase channel.

        Returns
        -------
        phase : float
            A value between 0 and 360 which produces which most achieves 
            the set goal in degrees.
        """
        self._has_locked()
        v_in = self.loc_Volt_out[self.slice]
        v_out = self.loc_Volt_out[self.slice]
        return (self.phase_shift +
            diglock.phase_in(v_in, v_out, aim=aim)) % 360

    def find_phase(self, skip_num=10, start_auto='change', to_zero=False):
        """Returns a function that makes a graph for phasing.
        
        This produces a function which when called plots a graph showing the 
        in and out of phase signal, the one argument is the phase. The 
        default value of the one argument is set by the start_phase 
        argument. 

        One way to use this is with the library 
        `ipywidgets  <https://ipywidgets.readthedocs.io/en/latest>`_ which 
        can make a slider in notebooks by running ::
        
            find_phase_function = PulsedLockIn.find_phase()
            ipywidgets.interact(find_phase_function, 
               phase=ipywidgets.FloatSlider(min=0, max=360, step=1))

        Parameters
        ----------
        skip_num : int, optional
            The ratio of points to skip when plotting. As this requires a 
            lot of calculation it can be beneficial to only plot a fraction 
            speed up the process. The default value is 10.
        start_auto : int, float, {'change', 'value'}, optional
            Decides in what phase to start the graph at. If a string is 
            given it is passed to :meth:`auto_phase`. If an number is given 
            the phase is set to that value.
        to_zero : bool
            If `True` the in phase and out of phase components are set to 
            zero at the lowest field. This can make the changes easier to 
            inspect. The default is `False`.

        Returns
        -------
        plotting : Calculable
            A function with one keyword argument of `phase` which plots the 
            in and out of phase signal when called.
        """
        self._has_locked()

        if isinstance(start_auto, (int, float, np.int_, np.float_)):
            start_phase = start_auto
        elif start_auto == 'change':
            start_phase = self.auto_phase(aim='change')
        elif start_auto == 'value':
            start_phase = self.auto_phase(aim='value')
        else:
            raise ValueError("start_auto must either be a value to start"
                " or 'change' or 'value' to be passed to auto_phase.")
        # Gets data to plot
        v_in, v_out = self.rephase(0, trial=True)
        v_in = v_in[self.slice][::skip_num]
        v_out = v_out[self.slice][::skip_num]
        if to_zero:
            v_in -= v_in[0]
            v_out -= v_out[0]
        field = self.field[self.slice][::skip_num]
        # Make plotting function
        def plotting(phase=start_phase):
            v_in_new  = v_in*np.cos(phase*np.pi/180) + \
                v_out*np.sin(phase*np.pi/180)
            v_out_new = v_out*np.cos(phase*np.pi/180) - \
                v_in*np.sin(phase*np.pi/180)
            plt.plot(field[[0, -1]], [0, 0], color='0.5')
            plt.plot(field, v_in_new, 'b', label='V In')
            plt.plot(field, v_out_new, 'r', label='V Out')
            plt.ylabel('Voltage (V)')
            plt.xlabel('Field (T)')
            plt.legend(loc='upper left', title=f"phase: {phase:.1f}")
            plt.show()

        return plotting

    def reset_slice(self, skip_num='No', B_min='No', side='No', trial=False):
        """This reproduces the slice which selects the data of interest.

        This is used to change the attribute :attr:`slice`. It also has a 
        trial option that will return a new slice instead of updating the 
        existing attribute. The parameters will try to all default to the 
        values to reproduce the current slice, this maybe not be exacltly 
        the same with `B_min`.

        Parameters
        ----------
        skip_num : int, optional
            The ratio of points to skip when outputting the data. This is 
            used because the object sizes can become unwieldy if all the 
            data is saved.
            it keeps.
        B_min : float or None, optional
            The minimum value of the field to keep points lower in field to 
            this will be dropped. If set to `None` all of the field range is 
            kept.
        side : {'up', 'down', 'both'}, optional
            The side of the pulse to produce the data for. 'up' is the first 
            side, 'down' is the second, and 'both' takes both sides.
        trial : bool, optional
            If `True` the slice is not saved and instead returned. The 
            default is `False` which updates :attr:`slice`.
        """
        if skip_num == 'No':
            skip_num = np.abs(self.slice.step)

        if B_min == 'No':
            B_min = np.min(self.field[[self.slice.start, self.slice.stop]])

        if side == 'No':
            if self.slice.step < 0:
                side = 'down'
            elif np.argmax(self.field[self.slice][::-1]) == 0:
                side = 'up'
            else:
                side = 'both'

        if trial:
            return pick_pulse_side(self.field, B_min, side, 
                skip_num=skip_num)
        else:
            self.slice = pick_pulse_side(self.field, B_min, side, 
                skip_num=skip_num)

    def _make_Data(self, y_values, as_Data, x_axis):
        """This is used to return the data after the lock in process in a 
        certain form.
        """
        if as_Data:
            if x_axis == 'field':
                return Data(self.field[self.slice], y_values)
            elif x_axis == 'time':
                return Data(np.arange(len(self.field))[self.slice]/self.fs,
                    y_values)
            else:
                raise ValueError("x_axis must either be 'field' or 'time'.")
        else:
            return y_values

    def volts_in(self, as_Data=True, x_axis='field'):
        """The signal from the locked in, in phase voltage.

        Parameters
        ----------
        as_Data : bool, optional
            If `True`, which is the default, the data is returned as a 
            :class:`.Data` object. If `False` it is returned as a 
            :class:`numpy.ndarray`.
        x_axis : {'field', 'time'}
            For the :class:`.Data` object whether the independent variable 
            should be the applied field or the time. The default is the 
            field.

        Returns
        -------
        volts_in : Data, numpy.ndarray
            The locked in measurement signal from the in phase channel.
        """
        self._has_locked()
        return self._make_Data(self.loc_Volt[self.slice],
            as_Data=as_Data, x_axis=x_axis)

    def volts_out(self, as_Data=True, x_axis='field'):
        """The signal from the locked in, out of phase voltage.

        Parameters
        ----------
        as_Data : bool, optional
            If `True`, which is the default, the data is returned as a 
            :class:`.Data` object. If `False` it is returned as a 
            :class:`numpy.ndarray`.
        x_axis : {'field', 'time'}
            For the :class:`.Data` object whether the independent variable 
            should be the applied field or the time. The default is the 
            field.

        Returns
        -------
        volts_out : Data, numpy.ndarray
            The locked in measurement signal from the out of phase channel.
        """
        self._has_locked()
        return self._make_Data(self.loc_Volt_out[self.slice],
            as_Data=as_Data, x_axis=x_axis)

    def res_in(self, as_Data=True, x_axis='field'):
        """The locked in voltage signal, in phase in units of Ohms.

        Parameters
        ----------
        as_Data : bool, optional
            If `True`, which is the default, the data is returned as a 
            :class:`.Data` object. If `False` it is returned as a 
            :class:`numpy.ndarray`.
        x_axis : {'field', 'time'}
            For the :class:`.Data` object whether the independent variable 
            should be the applied field or the time. The default is the 
            field.

        Returns
        -------
        res_in : Data, numpy.ndarray
            The locked in voltage signal, from the in phase channel divided 
            by the average current to obtain the units in Ohms.
        """
        self._has_locked()
        return self._make_Data(self.loc_Volt[self.slice]/self.Irms,
            as_Data=as_Data, x_axis=x_axis)

    def res_out(self, as_Data=True, x_axis='field'):
        """The locked in voltage signal, out of phase in units of Ohms.

        Parameters
        ----------
        as_Data : bool, optional
            If `True`, which is the default, the data is returned as a 
            :class:`.Data` object. If `False` it is returned as a 
            :class:`numpy.ndarray`.
        x_axis : {'field', 'time'}
            For the :class:`.Data` object whether the independent variable 
            should be the applied field or the time. The default is the 
            field.

        Returns
        -------
        res_out : Data, numpy.ndarray
            The locked in voltage signal, from the out of phase channel 
            divided by the average current to obtain the units in Ohms.
        """
        self._has_locked()
        return self._make_Data(self.loc_Volt_out[self.slice]/self.Irms,
            as_Data=as_Data, x_axis=x_axis)

    def volts_over_current(self, as_Data=True, x_axis='field'):
        """The locked in voltage signal over the current signal.

        The is for the same purpose as :meth:`res_in` but if the applied 
        current is for some reason not stable.

        Parameters
        ----------
        as_Data : bool, optional
            If `True`, which is the default, the data is returned as a 
            :class:`.Data` object. If `False` it is returned as a 
            :class:`numpy.ndarray`.
        x_axis : {'field', 'time'}
            For the :class:`.Data` object whether the independent variable 
            should be the applied field or the time. The default is the 
            field.

        Returns
        -------
        res_in : Data, numpy.ndarray
            The locked in voltage signal from the in phase channel divided 
            by the locked in current signal. This also obtains the value in 
            units of Ohms but allows to take into consideration variable 
            current flow.
        """
        self._has_locked()
        self._has_locked_I()
        v_over_i = self.loc_Volt[self.slice]/self.loc_I[self.slice]
        return self._make_Data(v_over_i, as_Data=as_Data, x_axis=x_axis)

    def current_in(self, as_Data=True, x_axis='field'):
        """The locked in current signal in units of Amps rms.

        Parameters
        ----------
        as_Data : bool, optional
            If `True`, which is the default, the data is returned as a 
            :class:`.Data` object. If `False` it is returned as a 
            :class:`numpy.ndarray`.
        x_axis : {'field', 'time'}
            For the :class:`.Data` object whether the independent variable 
            should be the applied field or the time. The default is the 
            field.

        Returns
        -------
        current_in : Data, numpy.ndarray
            The locked in current signal in units of Amps rms.
        """
        self._has_locked()
        self._has_locked_I()
        return self._make_Data(self.loc_I[self.slice],
            as_Data=as_Data, x_axis=x_axis)

