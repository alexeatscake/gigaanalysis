"""
GigaAnalysis - High Field

This program has a series of useful tools for conducting experiments in
certain high field labs.

"""

from .data import *
from . import diglock

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nptdms as tdms  # For read_ISSP
from ipywidgets import interact, FloatSlider # For PulsedLockIn.find_phase


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
        The name of the group of the the .tdms file, lab view as standard
        makes this 'Untitled', but if the language is something other than
        English this will change. It can also be set by the user.
    
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

def PUtoB(PU_signal, field_factor, fit_points):
    """Converts the voltage from the pick up coil to field. This is 
    used for pulsed field measurements.

    Parameters
    ----------
    PU_signal : numpy.ndarray
        The signal from pick up coil
    field_factor : float
        Factor to convert integral to magnetic field
    fit_points :int
        Number of point at each end to remove offset
    
    Returns
    -------
    field : numpy.ndarray
        An array of magnetic field the same length as PU_signal
    
    """
    count = np.arange(len(PU_signal))
    ends = np.concatenate([count[:fit_points], count[-fit_points:]])
    a, b = np.polyfit(ends, PU_signal[ends], 1)
    PU_flat = PU_signal - a*count - b
    return np.cumsum(PU_flat*field_factor)


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
            self.endB = self.maxB + \
                np.where(self.field[self.maxB:]<B_min)[0][0]
            self.slice = slice(self.endB, self.maxB, -skip_num)
        elif side == 'up':
            self.endB = np.where(self.field[:self.maxB]>B_min)[0][0]
            self.slice = slice(self.endB, self.maxB, skip_num)
        else:
            raise ValueError("side must be either 'up' or 'down'.")
        # Find frequencies and phases
        self.freq = diglock.find_freq(self.Iv, self.fs, padding=10)
        self.phase = diglock.find_phase(self.Iv, self.fs, self.freq)
        self.Irms = np.average((self.Iv*diglock.gen_ref(self.freq, self.fs,
            self.phase, len(self.Iv)))[1000:-1000])*np.sqrt(2)/R_shunt
        self.preamp = preamp
        # For cutting down data
        self.sfield = self.field[self.slice]
        

    def lockin_Volt(self, time_const, phase_shift):
        """
        This preforms a lock in process on the measurement signal.
        This uses :func:gigaanalysis.diglock.ham_lock_in
        
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
                Only includes the data points listed in slice
            loc_Volt_out : 1d numpy.ndarray
                Measurement voltage after lock in the same as
                :param:`loc_Volt` but with the out of phase signal
            Res : 1d numpy.ndarray
                :param:`loc_Volt` divided by current in Ohms
        """
        self.time_const = time_const
        self.phase_shift = phase_shift
        self.loc_Volt = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + phase_shift,
            )[self.slice]/self.preamp
        self.loc_Volt_out = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + phase_shift + 90.,
            )[self.slice]/self.preamp
        self.Res = self.loc_Volt/self.Irms

    def lockin_Volt_test(self, time_const, phase_shift):
        """
        This preforms a lock in process on the measurement signal and saves
        everything as opposed to performing the slice reduction 
        This uses diglock.ham_lock_in
        Args:
            time_const (float): time for averaging in seconds
            phase_shift (float): phase difference between the current
                voltage and the measurement voltage
        Returns:
            loc_Volt (1d np.array): Measurement voltage after lock in
                process corrected for preamp amplitude in Volts rms
        """
        return diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + phase_shift)

    def auto_phase(self, time_const, **kwargs):
        """
        This finds the phase which makes the out of phase a flat as possible
        and also has the in phase be majority positive.
        Uses :func:`gigaanalysis.diglock.ham_lock_in` and
        :func:`gigaanalysis.diglock.phase_in`
        
        Args:
            time_const (float): The time constant to be used to average by
                the lock in program.
        Returns:
            phase (float): A value between 0 and 360 which produces the
                flattest out of phase signal.
        """
        # Perform lock in to get data to fit
        v_in = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase)[self.slice]
        v_out = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + 90)[self.slice]
        return diglock.phase_in(v_in, v_out)


    def find_phase(self, time_const, skip_num=10,
                   start_auto=False):
        """
        This produces a graph with a slider that can be used
        Args:
            time_const (float): The averaging time in seconds
            skip_num (int): The number of points to skip passed when plotting
            start_auto (bool): If true will use auto_phase to find the best
                phase and start the graph at that location
        """
        if start_auto:
            start_phase = self.auto_phase(time_const)
        else:
            start_phase = 0
        # Perform lock in to get data to plot
        v_in = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase)[self.slice][::skip_num]
        v_out = diglock.ham_lock_in(self.Volt, time_const,
            self.fs, self.freq, self.phase + 90)[self.slice][::skip_num]
        v_in -= v_in[0]
        v_out -= v_out[0]
        field = self.field[self.slice][::skip_num]
        # Make plotting function
        def plotting(phase=start_phase):
            v_in_new  = v_in*np.cos(phase*np.pi/180) + \
                v_out*np.sin(phase*np.pi/180)
            v_out_new = v_out*np.cos(phase*np.pi/180) - \
                v_in*np.sin(phase*np.pi/180)
            plt.plot(field, v_in_new, 'b', label='V In')
            plt.plot(field, v_out_new, 'r', label='V Out')
            plt.ylabel('Voltage (V)')
            plt.xlabel('Field (T)')
            plt.legend(loc='upper right')
            plt.show()
        # Generate interactive window
        interact(plotting, phase=FloatSlider(min=0, max=360, step=1,
            value=start_phase, continuous_update=False))

    def plot_Res(self, *args, axis=None, **kwargs):
        """
        This plots the Rxx data
        Includes all the standard arguments from matplotlib.pyplot.plot

        Parameters
        ----------
        axis : matplotlib.axes.Axes, optional
            An axis that the line is plotted on if not given
            calls :func:`plt.plot`
        """
        if axis == None:
            axis = plt
        axis.plot(self.sfield, self.Res, *args, **kwargs)

    def save_Res(self, filename, **kwargs):
        """
        Saves the resistance vs field as a csv
        uses pandas.DataFrame.to_csv and kwargs are pass to it
        Args:
            filename (str): filename to save the data as
        """
        pd.DataFrame(np.concatenate(
            [self.sfield[:, None], self.Res[:, None]], axis=1),
            columns=['Field(T)', 'Resistance(Ohm)']
            ).to_csv(filename, **kwargs)

