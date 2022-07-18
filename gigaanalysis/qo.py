"""GigaAnalysis - Quantum Oscillations - :mod:`gigaanalysis.qo`
------------------------------------------------------------------

Here is a set of functions and classes that are useful for analysing 
quantum oscillation data. The general form that I assume when processing 
magnetic field sweeps to look for quantum oscillation are performing a 
background subtraction and then Fourier transforming that inverse field.
"""

from .data import *
from . import mfunc, const

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lifshitz_kosevich(temps, field, e_mass, amp=1., as_Data=True):
    """The Lifshitz-Kosevich formula
    
    This formula describes the change the the amplitude of quantum 
    oscillations as the temperature is changed. This is most commonly used 
    to calculate the effective mass of a carrier if a temperature dependence 
    of quantum oscillations are taken.
    The equation is of the form
    ``r_lk = amp*chi/sinh(chi)`` where
    ``chi = 2*pi*pi*kb*temps*me*e_mass/(hbar*qe*field)``.

    Parameters
    ----------
    temps : numpy.ndarray
        The value of temperatures to use to produce the LK curve.
    field : float
        The magnetic field value in Tesla of the applied field.
    e_mass : float
        The effective mass of the carriers in units of the electron mass.
    amp : float, optional
        The amplitude of the lowest temperature oscillations, the default 
        is unity.
    as_Data : bool, optional
        If the default of `True` the result is returned as a :class:`.Data` 
        object with the temps as the dependant variable. If false only the 
        LK amplitudes are returned as a :class:`numpy.ndarray`.

    Returns
    -------
    r_lk : Data, numpy.ndarray
        The amplitude of the quantum oscillations as the temperature is 
        changed.
    """
    try:
        temps = np.asarray(temps)
    except:
        raise TypeError(
            f"temps need to be an array or be able to be broadcast to one "
            f"but was of type {type(temps)}.")

    where0 = temps == 0
    temps[where0] = 1.

    chi = 2*np.pi*np.pi*const.kb()*temps*const.me()*e_mass/(
        const.hbar()*const.qe()*field)

    r_lk = amp*chi/np.sinh(chi)

    temps[where0] = 0.
    r_lk[where0] = amp


    if as_Data:
        return Data(temps, r_lk)
    else:
        r_lk


def dingle_damping(fields, frequency, scatting, amp=1., as_Data=True):
    """The Dingle Damping term from the LK formulas

    This describes how the amplitude of quantum oscillations changes with 
    applied field due to the scattering of electrons. The equation is of the 
    form 
    ``r_d = amp*exp(-sqrt(2*pi*pi*hbar*frequency/qe)/(fields*scatting))``

    Parameters
    ----------
    fields : numpy.ndarray
        The values of magnetic field to be used when calculating the 
        amplitude.
    frequency : float
        The frequency of the quantum oscillation in Tesla.
    scatting : float
        The scatting given by the mean free path in meters.
    amp : float, optional
        The amplitude at infinite field, the default is unity.
    as_Data : bool, optional
        If the default of `True` the result is returned as a :class:`.Data` 
        object with the fields as the dependant variable. If `False` only 
        the amplitudes are returned as a :class:`numpy.ndarray`.

    Returns
    -------
    r_d : Data, numpy.ndarray
        The amplitude of the quantum oscillations as the field is changed.
    """
    try:
        fields = np.asarray(fields)
    except:
        raise TypeError(
            f"fields need to be an array or be able to be broadcast to one "
            f"but was of type {type(fields)}.")

    where0 = fields == 0
    fields[where0] = 1.

    r_d = amp*np.exp(
        -np.sqrt(2*np.pi*np.pi*const.hbar()*frequency/const.qe())/
        (fields*scatting))

    fields[where0] = 0.
    r_d[where0] = 0.

    if as_Data:
        return Data(fields, r_d)
    else:
        return r_d


def quantum_oscilation(fields, frequency, amp, phase, damping, 
        as_Data=True):
    """Example Quantum Oscillation

    This is a simple example quantum oscillation for fitting and such like. 
    I say simple because the amplitude and the damping term has no frequency 
    or temperature dependence. This means you need to be more careful when 
    thinking about the units but also makes fitting easier. The equation is 
    of the form 
    ``quant_osc = 
    amp*exp(-damping/fields)*sin(360*frequency/fields + phase)``

    Parameters
    ----------
    fields : numpy.ndarray
        The fields to produce the form quantum oscillation over.
    frequency : float
        The frequency of the quantum oscillation in Tesla.
    amp : float
        The amplitude of the quantum oscillation at infinite field.
    phase : float
        The phase of the quantum oscillation in degrees.
    damping : float
        The scatting damping of the quantum oscillation in Tesla.
    as_Data : bool, optional
        If the default of `True` the result is returned as a :class:`.Data` 
        object with the fields as the dependant variable. If `False` only 
        the amplitudes are returned as a :class:`numpy.ndarray`.

    Returns
    -------
    quant_osc : Data, numpy.ndarray
        The amplitude of the quantum oscillations as the field is changed.
    """
    try:
        fields = np.asarray(fields)
    except:
        raise TypeError(
            f"fields need to be an array or be able to be broadcast to one "
            f"but was of type {type(fields)}.")
    
    where0 = fields == 0
    fields[where0] = 1.
    
    amplitudes = amp*np.exp(-damping/fields)
    wave = np.sin(np.radians(360*frequency/fields + phase))
    
    quant_osc = amplitudes*wave
    
    fields[where0] = 0.
    quant_osc[where0] = 0.
    
    if as_Data:
        return Data(fields, quant_osc)
    else:
        return quant_osc


def counting_freq(start_field, end_field, number_peaks):
    """Counting quantum oscillation to obtain a frequency.

    This returns the frequency of a quantum oscillation given a range 
    of field and the number of osscilations that occur in that range.
    Performs ``n*B1*B2/(B2 - B1)``

    Parameters
    ----------
    start_field : float
        The magnetic field to start the counting range.
    end_field : float
        The magnetic field to end the counting range.
    number_peaks : float
        The number of peaks in the given range.

    Returns
    -------
    frequency : float
        The frequency of the expected quantum oscillation in Tesla.
    """
    return number_peaks*start_field*end_field/(end_field-start_field)


def counting_num(start_field, end_field, frequency):
    """Expected count of quantum oscillation at a given frequency.

    This returns the number of quantum oscillations given a range 
    of field and frequency of quantum oscillations in question.
    Performs ``Freq*(B2 - B1)/(B1*B2)``

    Parameters
    ----------
    start_field : float
        The magnetic field to start the counting range.
    end_field : float
        The magnetic field to end the counting range.
    frequency : float
        The quantum oscillation frequency in Tesla.

    Returns
    -------
    number : float
        The number of expected peaks in this field range.
    """
    return frequency*(end_field-start_field)/(start_field*end_field)


class QO():
    """Quantum Oscillation object

    This takes a sweep in magnetic field and analyses the sweep to check the 
    presence and properties of quantum oscillations. It uses the 
    :class:`.Data` objects to hold the information.

    This class has an arbitrary subtraction function which is used to remove 
    the background signal and leave the quantum oscillations. Other ready 
    made classes exist that have a certain subtraction function 
    incorporated. This class functions as their parent.

    The analysis happens in 4 stages and the data is assessable at each 
    stage as a :class:`.Data` attribute. The first stage interpolates the 
    data evenly across the field window of interest. The second stage 
    performs the background subtraction. The third stage inverts the data in 
    in field. The final stage Fourier transforms the inverted signal.

    Parameters
    ----------
    data : Data
        The raw data of the field sweep to look for quantum oscillations in.
    min_field : float
        The lowest field value of the field range to inspect.
    max_field : float
        The highest field value of the field range to inspect.
    subtract_func : calculable
        This should take one :class:`.Data` object and return one 
        :class:`.Data` of the same length. The input data is the 
        interpolated sweep, and the output should be the data after the 
        background has been subtracted.
    step_size : float, optional
        The size of field steps to interpolate. The default is 4 times the 
        average step size in the raw data.
    fft_cut : float, optional
        The maximum frequency to consider in FFT, higher frequencies are 
        dropped. THe default is to keep all the data.
    strip_nan : bool, optional
        If `True` non finite values are removed from the raw data. The 
        default is `False` and this will raise an error is non finite values 
        are in the raw data.

    Attributes
    ----------
    raw : Data
        The data originally given to the class.
    interp : Data
        The sweep cut to the field range and interpolated.
    sub : Data
        The sweep after the background subtraction.
    invert : Data
        The subtracted sweep after inverting the field values.
    fft : Data
        The Fourier transform of the inverted signal.
    min_field : float
        The minimum field in the range in consideration.
    max_field : float
        The maximum field in the range in consideration.
    step_size : float
        The steps in field calculated in the interpolation.
    """
    def __init__(self, data, min_field, max_field, subtract_func,
            step_size=None, fft_cut=0, strip_nan=False):
        # Set up Class
        if type(data) != Data:
            try:
                data = Data(data)
            except:
                raise TypeError(
                    f"data was not a Data object nor could it be cast to "
                    f"one was of type {type(data)}")
        
        self.raw = data
        self.min_field = min_field
        self.max_field = max_field
        self._bg_sub_func = subtract_func

        if np.min(self.raw.x) > min_field:
            raise ValueError(
                "max_field value to interpolate is below data")
        if np.max(self.raw.x) < max_field:
            raise ValueError(
                "max_field value to interpolate is above data")

        if strip_nan:
            self.raw = self.raw.strip_nan()
        else:
            if not np.all(np.isfinite(self.raw.values)):
                raise ValueError(
                    f"The data contained non finite values and strip_nan"
                    f"was set to False.")

        if step_size is None:
            self.step_size = (self.raw.max_x() - self.raw.min_x()
                )/len(self.raw)/4
        else:
            self.step_size = step_size
        
        self.interp = self.raw.interp_range(min_field, max_field,
                                            self.step_size)
        self.sub = subtract_func(self.interp)
        self.invert = mfunc.invert_x(self.sub)
        self.fft = mfunc.fft(self.invert, freq_cut=fft_cut)

    def __len__(self):
        return self.interp.x.size

    def __repr__(self):
        return  (
            f"Quantum Oscillation object:\n"
            f"Field Range {self.min_field:.2f} T to {self.max_field:.2f} T\n"
            f"Number of points {self.interp.x.size}")

    def peaks(self, n_peaks=4, as_Data=False, **kwargs):
        """Finds the largest Fourier Transform peaks.

        This makes use of :func:`.mfunc.get_peaks` and the ``**kwargs`` are 
        passed to :func:`scipy.signal.find_peaks`.

        Parameters
        ----------
        n_peaks : int, optional
            The number of peaks to identify, the default is 4.
        as_Data : bool, optional
            If `True` the peak info is returned as a :class:`.Data` object. 
            The default is `False`. 

        Returns
        -------
        peak_info : numpy.ndarray
            A two column numpy array with the with the location and the 
            amplitude of each peak.
        """
        return mfunc.get_peaks(self.fft, n_peaks, as_Data=as_Data, **kwargs)

    def peak_height(self, position, x_range, x_value=False):
        """Provides the hight of the highest FFT in a given range.

        Makes use of the function :func:`.mfunc.peak_height`.
    
        Parameters
        ----------
        position : float
            The central position to search for the peak.
        x_range : float
            The range in x to look for the peak. This is the total range so 
            will extend half of this from ``position``.
        x_value : bool, optional
            If `True` the x value and the y value is returned. The default 
            is `False` which only returns the y value.

        Returns
        -------
        x_peak : float
            If x_value is `True` the x position of the peak is returned.
        y_peak : float
            The y_value of the peak. Which is the highest y value in a given 
            range of the data.
        """
        return mfunc.peak_height(self.fft, position, x_range, 
            x_value=x_value)

    def FFT_again(self, n=65536, window='hanning', freq_cut=0):
        """Recalculates the FFT.

        After recalcuating the FFT the new FFT is returned and also the 
        new FFT is saved to the :attr:`fft` attribute. This makes use of 
        :func:`.mfunc.fft`.

        Parameters
        ----------
        n : int, optional
            The number of points to make the FFT extra points will be zero 
            padded. The default is ``2**16 = 65536``. If the data is longer 
            than the value of n, n is rounded up to the next power of 2.
        window : str, optional
            The type of windowing function to use taken from 
            :func:`scipy.signal.get_window`. The default is 'hanning'.
        freq_cut : float, optional
            The frequency to drop all the higher from. The default is 0 
            which means that all the frequencies are kept.

        Returns
        -------
        fft_result : Data
            A data object with the FFT frequencies in the x values and the 
            amplitudes in the y values.
        """
        self.fft = mfunc.fft(self.invert, n=n, window=window, 
            freq_cut=freq_cut)
        return self.fft

    def to_csv(self, file_name, sep=','):
        """This saves the data contained to a .csv file.

        This saves the data in a csv file. It includes the interpolated,
        subtracted, inverse signals as well as the FFT. The FFT is
        interpolated to be the same length as the interpolated data.

        Parameters
        ----------
        file_name : str
            The name of the file to save the data to. If the file type is 
            not one of '.csv', '.txt', or '.dat'; then '.csv' will be 
            appended on to the end of the name.
        sep : str, optional
            The character used to the delineate between the data, the 
            default is ','.
        """
        if file_name[-4:] not in ['.csv', '.txt', '.dat']:
            file_name += '.csv'

        output_data = np.concatenate([
                        self.interp.values,
                        self.sub.values,
                        self.invert.values,
                        self.fft.interp_number(len(self)).values,
                            ], axis=1)

        header_line = 'Field_Interp{0:s}Interp_Signal{0:s}' \
                        'Field_Sub{0:s}Sub_Signal{0:s}' \
                        'Inverse_Field{0:s}Inverse_Signal{0:s}' \
                        'FFT_freq{0:s}FFT_amp'.format(sep)

        np.savetxt(file_name, output_data,
                   delimiter=sep, comments='',
                   header=header_line)


class QO_av(QO):
    """Average Quantum Oscillation Class

    This class applies a similar process to a set of sweeps in a list as 
    the :class:`QO` class. For each sweep the background in individually 
    subtracted. They are then averaged to produce the FFT. The attributes 
    for this class are also :class:`.Data` objects and these are the average 
    of all the separately interpolated and subtracted sweeps.

    This class is useful as the average of a collection of background 
    subtractions are not necessarily the same as the subtract of their 
    average.

    One important point with this class is that the :attr:`raw` will be 
    the given list as opposed to the average of this list. For the average 
    it is best to use the :attr:`interp` attribute. The step_size if not 
    given will also use the smallest of the generated step sizes form the 
    list of raw sweeps.
    
    Parameters
    ----------
    data : list
        A list of raw data of the field sweep in the form of a 
        list of :class:`.Data` objects to look for quantum oscillations in.
    min_field : float
        The lowest field value of the field range to inspect.
    max_field : float
        The highest field value of the field range to inspect.
    subtract_func : calculable
        This should take one :class:`.Data` object and return one 
        :class:`.Data` of the same length. The input data is the 
        interpolated sweep, and the output should be the data after the 
        background has been subtracted.
    step_size : float, optional
        The size of field steps to interpolate. The default is 4 times the 
        average step size in the raw data.
    fft_cut : float, optional
        The maximum frequency to consider in FFT, higher frequencies are 
        dropped. THe default is to keep all the data.
    strip_nan : bool, optional
        If `True` non finite values are removed from the raw data. The 
        default is `False` and this will raise an error is non finite values 
        are in the raw data.

    Attributes
    ----------
    raw : list
        The list of data originally given to the class.
    interp : Data
        The average of the sweeps cut to the field range and interpolated.
    sub : Data
        The average of the sweeps after the background subtraction.
    invert : Data
        The average of the subtracted sweeps after inverting the field 
        values.
    fft : Data
        The Fourier transform of the inverted average signal.
    min_field : float
        The minimum field in the range in consideration.
    max_field : float
        The maximum field in the range in consideration.
    step_size : float
        The steps in field calculated in the interpolation.
    """
    def __init__(self, data_list, min_field, max_field, subtract_func,
            step_size=None, fft_cut=0, strip_nan=False):
        # Set up class
        if type(data_list) != list:
            raise TypeError(
                f"data_list was not in the form of a list. "
                f"It was instead of the type {type(data_list)}.")
        self.raw = []
        raw_steps = []
        for n, raw_sweep in enumerate(data_list):
            if type(raw_sweep) != Data:
                try:
                    self.raw.append(Data(raw_sweep))
                except:
                    raise TypeError(
                        f"data in list positions {n} was not a Data object "
                        f"nor could it be cast to one. It was of the "
                        f"type {type(raw_sweep)}.")
            else:
                self.raw.append(raw_sweep)

            if np.min(raw_sweep.x) > min_field:
                raise ValueError(
                    f"max_field value to interpolate is below data."
                    f"Was was seen in data number {n}")
            if np.max(raw_sweep.x) < max_field:
                raise ValueError(
                    f"max_field value to interpolate is above data"
                    f"Was was seen in data number {n}")

            if strip_nan:
                self.raw[n] = raw_sweep.strip_nan()
            else:
                if not np.all(np.isfinite(raw_sweep.values)):
                    raise ValueError(
                        f"The data contained non finite values and "
                        f"strip_nan was set to False."
                        f"Was was seen in data number {n}")

            if step_size is None:
                raw_steps.append(
                    (raw_sweep.max_x() - raw_sweep.min_x()
                    )/len(raw_sweep)/4)
        
        self.min_field = min_field
        self.max_field = max_field
        self._bg_sub_func = subtract_func
  
        if step_size is None:
            self.step_size = np.min(raw_steps)
        else:
            self.step_size = step_size
        
        interp_list = []
        for data in self.raw:
            interp_list.append(data.interp_range(
                min_field, max_field, self.step_size))

        sub_list = []
        for interp in interp_list:
            sub_list.append(subtract_func(interp))

        self.interp = mean(interp_list)
        self.sub = mean(sub_list)
        self.invert = mfunc.invert_x(self.sub)
        self.fft = mfunc.fft(self.invert, freq_cut=fft_cut)

    def __repr__(self):
        return  (
            f"Average Quantum Oscillation object:\n"
            f"Number of sweeps averaged {len(self.raw)}\n"
            f"Field Range {self.min_field:.2f} T to {self.max_field:.2f} T\n"
            f"Number of points {self.interp.x.size}")

    def make_QO(self, raw_num, 
            step_size=None, fft_cut=None, strip_nan=False):
        """Make a Quantum Oscillation object form a certain sweep.

        This return a new quantum oscillation object from a particular 
        sweep in the raw list given. This will use the same field range 
        and background subtraction as used in this class.

        Parameters
        ----------
        raw_num : int 
            The number of the sweep to pass to :class:`QO`. Like all python 
            lists the counting starts at 0.
        step_size : float, optional
            If given this will be the step size used. If not given the 
            step_size in the original class is used.
        fft_cut : float, optional
            If given this will be the FFT cut to used. If not given the 
            fft_cut in the original class is used. To see the full range of 
            frequencies set the fft_cut to 0.
        strip_nan : bool, optional
            This does nothing as in order to make this class there cannot be 
            any NaNs left in the raw data. It is included for completeness.

        Returns
        -------
        single_QO : QO
            A quantum oscillation object with the same parameters as used in 
            this class but only the data from one of the sweeps.
        """
        if not isinstance(raw_num, (int, np.int_)):
            raise TypeError(
                    f"raw_num needs to be a int but is of type "
                    f"{type(raw_num)}")
        elif np.abs(raw_num) >= len(self.raw):
            raise ValueError(
                f"The raw_num is out of range. It was {raw_num} but there "
                f"are only {len(self.raw)} sweeps in the raw data.")

        if step_size is not None:
            to_step = step_size
        else:
            to_step = self.step_size

        if fft_cut is None:
            fft_cut = self.fft.x[-1]

        return QO(self.raw[raw_num], self.min_field, self.max_field,
            self._bg_sub_func, 
            step_size=to_step, fft_cut=fft_cut, strip_nan=strip_nan)


class QO_loess(QO):
    """Quantum Oscillation object with LOESS subtraction

    This is a example of the :class:`QO` which the subtraction using 
    :func:`.mfunc.loess`. The form is the same but with the initialising 
    function takes the arguments to define the LOESS background subtraction.

    Parameters
    ----------
    data : Data
        The raw data of the field sweep to look for quantum oscillations in.
    min_field : float
        The lowest field value of the field range to inspect.
    max_field : float
        The highest field value of the field range to inspect.
    loess_win : float
        The length of the window in Tesla to use for the LOESS subtraction.
    loess_poly : int
        The order of the polynomial to use for the LOESS subtraction.
    step_size : float, optional
        The size of field steps to interpolate. The default is 4 times the 
        average step size in the raw data.
    fft_cut : float, optional
        The maximum frequency to consider in FFT, higher frequencies are 
        dropped. THe default is to keep all the data.
    strip_nan : bool, optional
        If `True` non finite values are removed from the raw data. The 
        default is `False` and this will raise an error is non finite values 
        are in the raw data.

    Attributes
    ----------
    :
        This class has the same attributes as the :class:`QO` class but also 
        with the information about the LOESS subtraction.

    loess_win : float
        The length of the window in Tesla to use for the LOESS subtraction.
    loess_poly : int
        The order of the polynomial to use for the LOESS subtraction.
    """
    def __init__(self, data, min_field, max_field, loess_win, loess_poly,
                step_size=None, fft_cut=0, strip_nan=False):

        def bg_sub(interp):
            return interp - mfunc.loess(interp, loess_win, loess_poly)

        QO.__init__(self, data, min_field, max_field, bg_sub,
            step_size=step_size, fft_cut=fft_cut, strip_nan=strip_nan)

        self.loess_win = loess_win
        self.loess_poly = loess_poly

    def __repr__(self):
        return  (
            f"Quantum Oscillation object:\n"
            f"LOESS Background Subtraction\n"
            f"Field Range {self.min_field:.2f} T to {self.max_field:.2f} T\n"
            f"Number of points {self.interp.x.size}\n"
            f"LOESS window {self.loess_win:.2f} T\n"
            f"LOESS polynomial {self.loess_poly}")


class QO_loess_av(QO_av):
    """Average Quantum Oscillation object with LOESS subtraction

    This is a example of the :class:`QO_av` which the subtraction using 
    :func:`.mfunc.loess`. The form is the same but with the initialising 
    function takes the arguments to define the LOESS background subtraction.

    Parameters
    ----------
    data : list
        A list of raw data of the field sweep in the form of a 
        list of :class:`.Data` objects to look for quantum oscillations in.
    min_field : float
        The lowest field value of the field range to inspect.
    max_field : float
        The highest field value of the field range to inspect.
    loess_win : float
        The length of the window in Tesla to use for the LOESS subtraction.
    loess_poly : int
        The order of the polynomial to use for the LOESS subtraction.
    step_size : float, optional
        The size of field steps to interpolate. The default is 4 times the 
        average step size in the raw data.
    fft_cut : float, optional
        The maximum frequency to consider in FFT, higher frequencies are 
        dropped. THe default is to keep all the data.
    strip_nan : bool, optional
        If `True` non finite values are removed from the raw data. The 
        default is `False` and this will raise an error is non finite values 
        are in the raw data.

    Attributes
    ----------
    :
        This class has the same attributes as the :class:`QO` class but also 
        with the information about the LOESS subtraction.

    loess_win : float
        The length of the window in Tesla to use for the LOESS subtraction.
    loess_poly : int
        The order of the polynomial to use for the LOESS subtraction.
    """
    def __init__(self, data_list, min_field, max_field, 
            loess_win, loess_poly,
            step_size=None, fft_cut=0, strip_nan=False):

        def bg_sub(interp):
            return interp - mfunc.loess(interp, loess_win, loess_poly)

        QO_av.__init__(self, data_list, min_field, max_field, bg_sub,
            step_size=step_size, fft_cut=fft_cut, strip_nan=strip_nan)

        self.loess_win = loess_win
        self.loess_poly = loess_poly

    def __repr__(self):
        return (
            f"Average Quantum Oscillation object:\n"
            f"LOESS Background Subtraction\n"
            f"Number of sweeps averaged {len(self.raw)}\n"
            f"Field Range {self.min_field:.2f} T to {self.max_field:.2f} T\n"
            f"Number of points {self.interp.x.size}\n"
            f"LOESS window {self.loess_win:.2f} T\n"
            f"LOESS polynomial {self.loess_poly}")


class QO_poly(QO):
    """Quantum Oscillation object with polynomial subtraction

    This is a example of the :class:`QO` which the subtraction using 
    :func:`.mfunc.poly_reg`. The form is the same but with the initialising 
    function takes the arguments to define the polynomial background 
    subtraction.

    Parameters
    ----------
    data : Data
        The raw data of the field sweep to look for quantum oscillations in.
    min_field : float
        The lowest field value of the field range to inspect.
    max_field : float
        The highest field value of the field range to inspect.
    poly_order : int
        The order of the polynomial to use for the subtraction.
    step_size : float, optional
        The size of field steps to interpolate. The default is 4 times the 
        average step size in the raw data.
    fft_cut : float, optional
        The maximum frequency to consider in FFT, higher frequencies are 
        dropped. THe default is to keep all the data.
    strip_nan : bool, optional
        If `True` non finite values are removed from the raw data. The 
        default is `False` and this will raise an error is non finite values 
        are in the raw data.

    Attributes
    ----------
    :
        This class has the same attributes as the :class:`QO` class but also 
        with the information about the polynomial subtraction.

    poly_order : int
        The order of the polynomial to use for the subtraction.
    """
    def __init__(self, data, min_field, max_field, poly_order,
                step_size=None, fft_cut=0, strip_nan=False):

        def bg_sub(interp):
            return interp - mfunc.poly_reg(interp, poly_order)

        QO.__init__(self, data, min_field, max_field, bg_sub,
            step_size=step_size, fft_cut=fft_cut, strip_nan=strip_nan)

        self.poly_order = poly_order

    def __repr__(self):
        return  (
            f"Quantum Oscillation object:\n"
            f"Polynomial Background Subtraction\n"
            f"Field Range {self.min_field:.2f} T to {self.max_field:.2f} T\n"
            f"Number of points {self.interp.x.size}\n"
            f"Polynomial order {self.poly_order}")


class QO_poly_av(QO_av):
    """Average Quantum Oscillation object with polynomial subtraction

    This is a example of the :class:`QO_av` which the subtraction using 
    :func:`.mfunc.poly_reg`. The form is the same but with the initialising 
    function takes the arguments to define the polynomial background 
    subtraction.

    Parameters
    ----------
    data : list
        A list of raw data of the field sweep in the form of a 
        list of :class:`.Data` objects to look for quantum oscillations in.
    min_field : float
        The lowest field value of the field range to inspect.
    max_field : float
        The highest field value of the field range to inspect.
    poly_order : int
        The order of the polynomial to use for the subtraction.
    step_size : float, optional
        The size of field steps to interpolate. The default is 4 times the 
        average step size in the raw data.
    fft_cut : float, optional
        The maximum frequency to consider in FFT, higher frequencies are 
        dropped. THe default is to keep all the data.
    strip_nan : bool, optional
        If `True` non finite values are removed from the raw data. The 
        default is `False` and this will raise an error is non finite values 
        are in the raw data.

    Attributes
    ----------
    :
        This class has the same attributes as the :class:`QO` class but also 
        with the information about the polynomial subtraction.

    poly_order : int
        The order of the polynomial to use for the subtraction.
    """
    def __init__(self, data_list, min_field, max_field, 
            poly_order,
            step_size=None, fft_cut=0, strip_nan=False):

        def bg_sub(interp):
            return interp - mfunc.poly_reg(interp, poly_order)

        QO_av.__init__(self, data_list, min_field, max_field, bg_sub,
            step_size=step_size, fft_cut=fft_cut, strip_nan=strip_nan)

        self.poly_order = poly_order

    def __repr__(self):
        return (
            f"Average Quantum Oscillation object:\n"
            f"Polynomial Background Subtraction\n"
            f"Number of sweeps averaged {len(self.raw)}\n"
            f"Field Range {self.min_field:.2f} T to {self.max_field:.2f} T\n"
            f"Number of points {self.interp.x.size}\n"
            f"Polynomial order {self.poly_order}")

