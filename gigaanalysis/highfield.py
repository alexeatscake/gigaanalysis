"""GigaAnalysis - High Field

This program has a series of useful tools for conducting experiments in
certain high field labs.
"""

from .data import *
from . import diglock

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider # For PulsedLockIn.find_phase
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
    """Converts the voltage from the pick up coil to field. This is 
    used for pulsed field measurements.

    Parameters
    ----------
    PU_signal : numpy.ndarray
        The signal from pick up coil
    field_factor : float
        Factor to convert integral to magnetic field
    fit_points : int
        Number of point at each end to remove offset
    to_fit : {'PU', 'field'} optional
        If to correct an offset voltage the PU signal is fit or the field.
    
    Returns
    -------
    field : numpy.ndarray
        An array of magnetic field the same length as PU_signal
    
    """
    count = np.arange(len(PU_signal))
    ends = np.concatenate([count[:fit_points], count[-fit_points:]])
    if to_fit == 'PU':
        a, b = np.polyfit(ends, PU_signal[ends], 1)
        PU_flat = PU_signal - a*count - b
        return np.cumsum(PU_flat*field_factor)
    elif to_fit == 'field':
        field = np.cumsum(PU_signal*field_factor)
        a, b = np.polyfit(ends, field[ends], 1)
        return field - a*count - b
    else:
        raise ValueError("to_fit needs to be either 'PU' or 'field'.")


class PulsedLockIn():
    '''
    This class is for locking in to pulsed field data for measurements
    where the raw signal is taken for post phase sensitive detection
    processing. The initialise class processes the data to find a selection
    of properties that will be used for the following processing.
    The phasing and time constant is asked for when performing the lock in
    process. The lock-in process is done using the hamming window.
    These arguments are taken from the __init__ function.
    Args:
        field (1d np.array): field values in Tesla will be corrected to make
            peak field positive
        current (1d np.array): voltage readings across a shut
            resistor in Volts
        voltage (1d np.array): voltage readings in the measurement
            channel in Volts
        sample_freq (float default=15e6): The rate of measurements in hertz
        R_shunt (float default=100.): The values of the shut resistor in ohms
        preamp (float default=100.): The value of pre-amplification of the
            measurement channel
        skip_num (int default=200): The number of points to skip per point
            used after the lock in process. This is reduce the amount of data
            that is stored
        B_min (float default=0.): The minimum value of field to use. This
            should be increased if the coil is set up to not reach zero
            before the measurement finishes
    Attributes:
        As well as the inputs being stored as attributes the following
        are made
        time (1d np.array): The values of the time points of the measurement
            in milliseconds
        maxB (int): The array index where max field is reached
        endB (int): The array index where the measurement ends
        peak_field (float): The maximum field value reached in Tesla
        field_direction (bool): True if positive direction False if negative
        freq (float): The frequency in hertz from the current channel
        phase (float): The phase of the current measurement
        Irms (float): The average current readings in amps
        slice (slice): The array values of the measurement points to use
        sfield (1d np.array): The field values only with the slice points
    '''
    def __init__(self, field, current, voltage, sample_freq=15e6,
            R_shunt=100., preamp=1., skip_num=200, B_min=0., side='down'):
        # Take the original data
        self.field = field
        self.Iv = current
        self.Volt = voltage
        # generate info
        self.fs = sample_freq
        self.time = 1000*np.arange(len(self.field))/self.fs
        self.field_direction = (abs(np.max(self.field)) > 
            abs(np.min(self.field)))
        if not self.field_direction:
            self.field = -self.field 
        self.maxB = np.argmax(self.field)
        self.peak_field = self.field[self.maxB]
        if side == 'down':
            self.startB = self.maxB
            self.endB = self.maxB + \
                np.where(self.field[self.maxB:]<B_min)[0][0]
            self.slice = slice(self.endB, self.startB, -skip_num)
        elif side == 'up':
            self.startB = np.where(self.field[:self.maxB]>B_min)[0][0]
            self.endB = self.maxB
            self.slice = slice(self.startB, self.endB, skip_num)
        elif side == 'both':
            self.startB = np.where(self.field[:self.maxB]>B_min)[0][0]
            self.endB = self.maxB + \
                np.where(self.field[self.maxB:]<B_min)[0][0]
            self.slice = slice(self.startB, self.endB, skip_num)
        else:
            raise ValueError("side must be either 'up', 'down', or 'both'.")
        # Find frequencies and phases
        self.freq = diglock.find_freq(self.Iv, self.fs, padding=10)
        self.phase = diglock.find_phase(self.Iv, self.fs, self.freq)
        self.Irms = np.average((self.Iv*diglock.gen_ref(self.freq, self.fs,
            self.phase, len(self.Iv)))[1000:-1000])*np.sqrt(2)/R_shunt
        self.preamp = preamp
        self.R_shunt = R_shunt
        # Attributes to be set
        self.time_const = None
        self.phase_shift = 0
        self.loc_Volt = None
        self.loc_Volt_out = None

    def lockin_Volt(self, time_const, phase_shift=None):
        """
        This preforms a lock in process on the measurement signal.
        This uses :func:`.diglock.ham_lock_in`.
        
        Parameters
        ----------
        time_const : float
            Time for averaging in seconds
        phase_shift : float
            Phase difference between the current voltage
            and the measurement voltage

        Attributes
        ----------
        As well as the inputs being stored as attributes the following
        are made
        loc_Volt : 1d np.array
            Measurement voltage after lock in
            process corrected for preamp amplitude in Volts rms
        loc_Volt_out : 1d numpy.ndarray
            Measurement voltage after lock in the same as
            `loc_Volt` but with the out of phase signal
        """
        self.time_const = time_const
        if phase_shift != None:
            self.phase_shift = phase_shift
        self.loc_Volt = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + self.phase_shift,
            )/self.preamp
        self.loc_Volt_out = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + self.phase_shift + 90.,
            )/self.preamp

    def lockin_current(self, time_const, phase=0,):
        """
        This preforms a lock in process on the measurement signal.
        This uses :func:`.diglock.ham_lock_in`.
        
        Parameters
        ----------
            time_const : float
                Time for averaging in seconds

        Attributes
        ----------
            As well as the inputs being stored as attributes the following
            are made
            loc_I : 1d np.array
                Current after lock in
                process corrected for preamp amplitude in Volts rms
        """
        self.loc_I = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + phase,)/self.R_shunt

    def smooth_Volts(self, smooth_time, smooth_order=2):
        """
        This preforms a lock in process on the measurement signal, it then
        performs a light smoothing to remove aliasing.
        This uses :func:`.diglock.ham_lock_in`.
        The smoothing is done with a pass of a Savitzky-Golay filter from
        :func:`scipy.signal.savgol_filter`.
        
        Parameters
        ----------
            smooth_points : float
                The time over to fit for smoothing.
            smooth_order : int, optional
                The order of the poly to fit for the smoothing.

        Attributes
        ----------
            As well as the inputs being stored as attributes the following
            are made
            loc_Volt : 1d np.array
                Measurement voltage after lock in
                process corrected for preamp amplitude in Volts rms
            loc_Volt_out : 1d numpy.ndarray
                Measurement voltage after lock in the same as
                `loc_Volt` but with the out of phase signal
        """
        smooth_points = int(np.ciel(smooth_time/self.fs/2)*2 + 1)
        self.loc_Volt = savgol_filter(self.loc_Volt, smooth_points,
            smooth_order,)
        self.loc_Volt_out = savgol_filter(presmooth_out, smooth_points,
            smooth_order,)

    def rephase(self, phase_shift, trial=False):
        """
        Rephases the signal to the new phase
        """
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

    def auto_phase(self, aim='change', **kwargs):
        """
        This finds the phase which makes the out of phase a flat as possible
        and also has the in phase be majority positive.
        Uses :func:`.diglock.ham_lock_in` and
        :func:`.diglock.phase_in`
        
        Args:
            time_const (float): The time constant to be used to average by
                the lock in program.
            aim : ('change', 'value')
                Weather to minimise the change in signal or the signal total
        Returns:
            phase (float): A value between 0 and 360 which produces the
                best out of phase signal.
        """
        # Perform lock in to get data to fit
        v_in = self.loc_Volt_out[self.slice]
        v_out = self.loc_Volt_out[self.slice]
        return (self.phase_shift +
            diglock.phase_in(v_in, v_out, aim=aim)) % 360

    def find_phase(self, skip_num=10, start_auto='change', to_zero=True):
        """
        This produces a graph with a slider that can be used
        Args:
            time_const (float): The averaging time in seconds
            skip_num (int): The number of points to skip passed when plotting
            start_auto (bool): If true will use auto_phase to find the best
                phase and start the graph at that location
        """

        if isinstance(start_auto, (np.int, np.float)):
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
            plt.legend(loc='upper right')
            plt.show()
        # Generate interactive window
        interact(plotting, phase=FloatSlider(min=0, max=360, step=1,
            value=start_phase, continuous_update=False))

    def spike_lockin_Volt(self, time_const, phase_shift,
            sdl=2, region=1001):
        """
        This preforms a lock in process on the measurement signal. With the
        aim of removing spikes in the raw first.
        This uses :func:`.diglock.spike_lock_in`.
        
        Parameters
        ----------
            time_const : float
                Time for averaging in seconds
            phase_shift : float
                Phase difference between the current voltage
                and the measurement voltage
            sdl : float, optional
                The number of standard deviations the data
                to be deviate from to be considered a outlier.
            region : int, optional
                The number of points around the outlier
                that are also considered bad.

        Attributes
        ----------
            As well as the inputs being stored as attributes the following
            are made
            loc_Volt : 1d np.array
                Measurement voltage after lock in
                process corrected for preamp amplitude in Volts rms
                Only includes the data points listed in slice
            loc_Volt_out : 1d numpy.ndarray
                Measurement voltage after lock in the same as
                `loc_Volt` but with the out of phase signal
            Res : 1d numpy.ndarray
                `loc_Volt` divided by current in Ohms
        """
        self.time_const = time_const
        self.phase_shift = phase_shift
        self.loc_Volt = diglock.spike_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + phase_shift,
            sdl=sdl, region=region,)/self.preamp
        self.loc_Volt_out = diglock.spike_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + phase_shift + 90.,
            sdl=sdl, region=region,)/self.preamp

    def _make_Data(self, y_values, as_Data, x_axis):
        """
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
        """
        The signal from the locked in, in phase voltage.
        """
        return self._make_Data(self.loc_Volt[self.slice],
            as_Data=as_Data, x_axis=x_axis)

    def volts_out(self, as_Data=True, x_axis='field'):
        """
        The signal from the locked in, in phase voltage.
        """
        return self._make_Data(self.loc_Volt_out[self.slice],
            as_Data=as_Data, x_axis=x_axis)

    def res_in(self, as_Data=True, x_axis='field'):
        """
        The signal from the locked in, in phase voltage.
        """
        return self._make_Data(self.loc_Volt[self.slice]/self.Irms,
            as_Data=as_Data, x_axis=x_axis)

    def res_out(self, as_Data=True, x_axis='field'):
        """
        The signal from the locked in, in phase voltage.
        """
        return self._make_Data(self.loc_Volt_out[self.slice]/self.Irms,
            as_Data=as_Data, x_axis=x_axis)

