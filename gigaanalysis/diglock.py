"""GigaAnalysis - Digital Lock In - :mod:`gigaanalysis.diglock`
------------------------------------------------------------------

This program is to recreate what a lock in would do for slower measurements
but for our high field experiments. This is based around what the program in 
DRS and WUH did. This module also includes the :func:`scanning_fft` which is 
used for PDO and TDO measurements.
"""

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import (blackmanharris,  # for find_freq
    hamming,  # for hamming_window
    butter, filtfilt,  # for butter_bandpass and butter_bandpass_filter
    get_window)  # For sfft
from numpy.fft import rfft, rfftfreq # for find_freq
from scipy.optimize import minimize  # for find_phase, phase_in


def polypeak(signal, fit_point=3, low_f_skip=0):
    """Finds the largest value in a data set by fitting a parabola.

    It picks the largest point in the dataset and fits a quadratic parabola 
    to it. It then uses that to get the interpolated maximum.

    Parameters
    ----------
    signal : numpy.ndarray
        The data to interpolate the highest value of.
    fit_point : int, optional
        The number of points to use in the fit. Needs to be odd.
    low_f_skip : int, optional
        The number of points to disregard at the start of the data.

    Returns
    -------
    x : float
        The x position as a rational number in relation to the index of the
        maximal value.
    y : float
        The y position of the interpolated maximum value.
    """

    if fit_point%2 != 1:
        ValueError(
            f"fit_point needs to be odd, was {fit_point} .")
    m = int((fit_point - 1)/2)  # Num of points either side
    i = low_f_skip + np.argmax(signal[low_f_skip:])  # Find highest point
    # Fit a quadratic and get the x and y of the highest point
    a, b, c = np.polyfit(np.arange(i-m,i+m+1), signal[i-m:i+m+1], 2) 
    x = -0.5*b/a  
    y = a*x**2 + b*x + c
    return x, y


def find_freq(data, samp_freq,
        padding=1, fit_point=3, plot=False, amp=False, skip_start=40):
    """Finds the dominate frequency in oscillatory data.

    It performs an FFT and then finds the maximal frequency using 
    :func:`polypeak`.
    
    Parameters
    ----------
    data : numpy.ndarray
        The signal in evenly spaced points
    samp_freq : float
        The measurement frequency of the data points
    padding : float, optional
        Pads the data my multiplying it before the FFT, default is 1.
    fit_point : int, optional
        Number of fit points to be used in :func:`polypeak`, default is 3.
    plot : bool, optional
        If `True` plots a figure to check the identification of the peak.
    amp : bool, optional
        If `True` returns the FFT amplitude of the frequency as well.
    skip_start : int, optional
        The number of points to skip the low frequency tail of the FFT, the
        default is 40.

    Returns
    -------
    peak_freq : float
        The value of the dominate frequency.
    peak_amp : float
        If amp is `True` also returns the amplitude of the dominate 
        frequency.
    """
    windowed = data*blackmanharris(len(data))
    n_fft = round(padding*len(windowed))
    fft = abs(rfft(windowed, n=n_fft))

    peak_freq, peak_amp = polypeak(
        fft[skip_start:], fit_point=fit_point)
    peak_freq = samp_freq*(peak_freq+skip_start)/n_fft

    if plot:  # This plots the FFT it uses to find the frequency
        f_freq = rfftfreq(padding*len(windowed), 1/samp_freq)
        plt.plot(f_freq, fft, '.')
        plt.plot(peak_freq, peak_amp, '.')
        plt.show()

    if amp:
        return peak_freq, peak_amp
    else:
        return peak_freq


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Produces the polynomial values for the Butterworth bandpass.

    This make use of :func:`scipy.signal.butter`, and supplies values for 
    :func:`scipy.signal.filtfilt`.

    Parameters
    ----------
    lowcut : float
        The low frequency cut off.
    highcut : float
        The high frequency cut off.
    fs : float
        The sample frequency of the data.
    order : int, optional
        The order of the Butterworth filter, default is 5.
    
    Returns
    -------
    b, numpy.ndarray, 
        The numerator of the polynomials of the IIR filter.
    a, numpy.ndarray, 
        The denominator of the polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter to a set of data.

    This makes use of :func:`butter_bandpass` and applied that filter to a
    given signal.

    Parameters
    ----------
    data : numpy.ndarray
        A array containing the signal.
    lowcut : float
        The low frequency cut off.
    highcut : float
        The high frequency cut off.
    fs : float
        The sample frequency of the data points.
    order : int, optional
        The order of the filter to apply, default is 5.

    Returns
    -------
    filtered : numpy.ndarray
        The filtered data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def gen_ref(freq, fs, phase, number):
    """Produces the reference signal for the digital lock in.

    Parameters
    ----------
    freq : float
        Frequency of the signal.
    fs : float
        Sample frequency of the data.
    phase : float
        Phase of the signal in degrees.
    number : int
        The number of points to generate.

    Returns
    -------
    ref_signal : numpy.ndarray
        An array containing the reference signal.
    """
    return np.sin(
                2*np.pi*freq*np.arange(0, number, 1)/fs 
                + phase*np.pi/180)


def find_phase(data, fs, freq):
    """Finds the phase of a oscillatory signal.

    Parameters
    ----------
    data : numpy.ndarray
        An array containing the signal.
    fs : float
        Sample frequency of the data points.
    freq : float
        The frequency of the oscillatory signal.

    Returns
    -------
    phase : float
        The phase in degrees of the oscillatory signal.
    """
    def to_min(p):
        """Produces a reference signal and sums the product with the data. 
        This is used in the optimisation routine."""
        return -np.sum(data*gen_ref(freq, fs, p, len(data)))

    return float(minimize(to_min, 180, bounds=[(0, 360)]).x)


def round_oscillation(time_const, freq):
    """Rounds to nearest number of whole oscillations.

    Used for minimising aliasing issues.

    Parameters
    ----------
    time_const : float
        The averaging time
    freq : float
        Frequency of the signal

    Returns
    -------
    number_osc : int
        The closet number of oscillations in that time window.
    """
    return round(time_const*freq)


def flat_window(time_const, fs, freq):
    """Produces a flat window for averaging.

    Uses :func:`round_oscillation` to set the window as the same length as 
    a whole number of oscillations.

    Parameters
    ----------
    time_const : float
        The time for averaging
    fs : float
        The sample frequency of the signal
    freq : float
        The frequency of the oscillatory signal.

    Returns
    -------
    flat_window : numpy.ndarray
        An array to convolve with the signal that has a unit total.
    """
    window = np.full(int(round_oscillation(time_const, freq)*fs/freq), 1.0)
    return window/np.sum(window)


def hamming_window(time_const, fs):
    """Produces a hamming window for averaging the signal.

    Uses a Hamming filter shape :func:`scipy.signal.hamming`.

    Parameters
    ----------
    time_const : float
        The time for averaging, this is like the 'mean' time.
    fs : float
        The sample frequency of the data.
    
    Returns
    -------
    hamming_window : numpy.ndarray
        An array to convolve with the signal that has unit total.
    """
    window = hamming(round(time_const*fs/0.54))
    return window/np.sum(window)


def ham_lock_in(signal, time_const, fs, freq, phase):
    """Performs a lock in of the signal and averages with a hamming window.

    The window from :func:`hamming_window` is convolved with the signal that 
    has been multiplied by the reference from :func:`gen_ref`.

    Parameters
    ----------
    signal : numpy.ndarray
        The oscillatory signal to lock in to.
    time_const : float
        The time constant for the averaging.
    fs : float
        The sample frequency of the signal data.
    freq : float
        The frequency of the oscillatory signal to lock in to.
    phase : float
        The phase of the signal is degrees.

    Returns
    -------
    signal_amp : numpy.ndarray
        The signal after the lock in processes which is equal to the 
        amplitude of the oscillatory signal at the given frequency. 
    """
    if time_const*fs > len(signal):
        raise ValueError(
            f"The averaging window is longer than signal.")

    window = hamming_window(time_const, fs)
    lock_ref = gen_ref(freq, fs, phase, len(signal))
    return np.convolve(signal*lock_ref, window, mode='same')*np.sqrt(2)


def flat_lock_in(signal, time_const, fs, freq, phase):
    """Performs a lock in of the signal and averages with a flat window.

    The window from :func:`flat_window` is convolved with the signal that 
    has been multiplied by the reference from :func:`gen_ref`.

    Parameters
    ----------
    signal : numpy.ndarray
        The oscillatory signal to lock in to.
    time_const : float
        The time constant for the averaging.
    fs : float
        The sample frequency of the signal data.
    freq : float
        The frequency of the oscillatory signal to lock in to.
    phase : float
        The phase of the signal is degrees.

    Returns
    -------
    signal_amp : numpy.ndarray
        The signal after the lock in processes which is equal to the 
        amplitude of the oscillatory signal at the given frequency. 
    """
    if time_const*fs > len(signal):
        raise ValueError(
            f"The averaging window is longer than signal.")

    window = flat_window(time_const, fs, freq)
    lock_ref = gen_ref(freq, fs, phase, len(signal))
    return np.convolve(signal*lock_ref, window, mode='same')*np.sqrt(2)


def phase_in_change(signal_in, signal_out, **kwargs):
    """Picks a phase to capture the majority of the change of the signal.

    This given an in and out of phase signal returns the phase shift to
    move the majority of the change in signal into the in phase component.
    It also chooses the phase shift so the signal is positive and between
    0 and 360 deg.
    Uses :func:`scipy.optimize.minimize` and keyword arguments are passed 
    to it.

    Parameters
    ---------
    signal_in : numpy.ndarray
        The values containing the in phase signal.
    signal_out : numpy.ndarray
        The values containing the out of phase signal needs to be the same
        shape as `signal_in`.

    Returns
    -------
    max_phase : float
        The phase in degrees between 0 deg and 360 deg where the change in
        signal in the out of phase is minimised.
    """
    if signal_in.shape != signal_out.shape:
        raise ValueError(
            f"The singal_in and singal_out arrays need to be the same "
            f"shape. They are {signal_in.shape} and {signal_out.shape}")

    scaling = 1./np.sum(
        np.sqrt(signal_in*signal_in + signal_out*signal_out))
    
    def to_min(phase):
        """This returns the average of the square of the out of phase signal 
        minus its mean. Used for the optimisation."""
        new_out = signal_out*np.cos(phase*np.pi/180) - \
            signal_in*np.sin(phase*np.pi/180)
        resisual = (new_out - np.average(new_out))*scaling
        return np.average(resisual*resisual)

    # Find the minimum
    max_phase = np.asscalar(
        minimize(to_min, x0=0., method='Nelder-Mead', **kwargs).x
        ) % 180
    # Pick double value that makes signal positive
    if np.sum(signal_in*np.cos(max_phase*np.pi/180) + \
            signal_out*np.sin(max_phase*np.pi/180)) < 0:
        return max_phase + 180
    else:
        return max_phase


def phase_in_value(signal_in, signal_out, **kwargs):
    """Picks a phase to capture the majority of the amplitude of the signal.

    This given an in and out of phase signal returns the phase shift to
    move the majority of the signal into the in phase component.
    It also chooses the phase shift so the signal is positive and between
    0 and 360 deg.
    Uses :func:`scipy.optimize.minimize` and keyword arguments are passed 
    to it.

    Parameters
    ---------
    signal_in : numpy.ndarray
        The values containing the in phase signal.
    signal_out : numpy.ndarray
        The values containing the out of phase signal needs to be the same
        shape as `signal_in`.

    Returns
    -------
    max_phase : float
        The phase in degrees between 0 deg and 360 deg where amplitude of 
        the signal is maximised.
    """
    if signal_in.shape != signal_out.shape:
        raise ValueError(
            f"The singal_in and singal_out arrays need to be the same "
            f"shape. They are {signal_in.shape} and {signal_out.shape}")

    scaling = 1./np.sum(
        np.sqrt(signal_in*signal_in + signal_out*signal_out))

    def to_min(phase):
        """This returns the average of the square of the out of phase signal.
        Used for the optimisation. Scaled to help optimisation."""
        new_out = signal_out*np.cos(phase*np.pi/180) - \
            signal_in*np.sin(phase*np.pi/180)
        out_abs = np.abs(new_out*scaling)
        return np.average(out_abs)

    # Find the minimum
    max_phase = np.asscalar(
        minimize(to_min, x0=0., method='Nelder-Mead', **kwargs).x) % 180
    # Pick double value that makes signal positive
    if np.sum(signal_in*np.cos(max_phase*np.pi/180) + \
            signal_out*np.sin(max_phase*np.pi/180)) < 0:
        return max_phase + 180
    else:
        return max_phase


def phase_in(signal_in, signal_out, aim='change', **kwargs):
    """Picks a phase that maximises something about the signal.

    This makes use of either :func:`phase_in_change` or 
    :func:`phase_in_value', depending on the aim keyword.
    Uses :func:`scipy.optimize.minimize` and keyword arguments are passed 
    to it.

    Parameters
    ---------
    signal_in : numpy.ndarray
        The values containing the in phase signal.
    signal_out : numpy.ndarray
        The values containing the out of phase signal needs to be the same
        shape as `signal_in`.
    aim : str, {'change', 'value'}, optional
        What to maximise. The default is 'change'.

    Returns
    -------
    max_phase : float
        The best phase in degrees between 0 deg and 360 deg.
    """
    if signal_in.shape != signal_out.shape:
        raise ValueError(
            f"The singal_in and singal_out arrays need to be the same "
            f"shape. They are {signal_in.shape} and {signal_out.shape}")
    if aim == 'change':
        return phase_in_change(signal_in, signal_out, **kwargs)
    elif aim == 'value':
        return phase_in_value(signal_in, signal_out, **kwargs)
    else:
        raise ValueError(
            f"Aim must be either 'change' or 'value', but was '{aim}'.")


def select_not_spikes(data, sdl=2., region=1001):
    """Identifies spikes in the data and returns a boolean array.

    This finds spikes in a set of data an excludes the region around them 
    too. It does this by looking at where the value changes unusually 
    quickly.

    Parameters
    ----------
    data : numpy.ndarray
        The signal to check for spikes.
    sdl : float, optional
        The number of standard deviations the data need to deviate by to be 
        considered an outlier.
    region : int, optional
        The number of points around an outlier to exclude. Default is 1001.

    Returns
    -------
    good_vals : numpy.ndarray
        A boolean array with the same shape as the signal data with points 
        near spikes labelled `False` and the unaffected points labelled 
        `True`.
    """
    change = np.append(np.diff(data), 0)
    spikes = abs(change - np.mean(change)) > sdl*np.std(change)
    good_vals = np.convolve(spikes, np.ones(region), mode='same') == 0
    return good_vals


def spike_lock_in(signal, time_const, fs, freq, phase, sdl=2., region=1001):
    """Performs a lock in of the signal but with also spike removal.

    This lock in makes use of :func:`ham_lock_in`. It also removes the 
    points effected by spikes by using :func:`select_not_spikes`. It tries 
    to interpolate between the points.
    The spike removal works better with a smaller time constant.

    Parameters
    ----------
    signal : numpy.ndarray
        The AC signal to lock in to.
    time_const : float 
        The time for averaging with the hamming window.
    fs : float
        The sample frequency.
    freq : float
        The frequency of the AC signal.
    phase : float
        The phase of the AC signal in degrees.
    sdl : float, optional
        The number of standard deviations that will trigger a spike 
        detection. The default is 2.

    Returns
    -------
    locked_signal : numpy.ndarray
        The signal after the spike removal and lock in process. 

    """
    window = hamming_window(time_const, fs)
    lock_ref = gen_ref(freq, fs, phase, len(signal))
    times = np.arange(len(signal))
    good_points = select_not_spikes(signal*lock_ref, sdl, region)
    locked_in = np.convolve((signal*lock_ref)[good_points],
                            window, mode='same')*np.sqrt(2)
    # step_size = int(np.ceil(len(window)/pp_window))
    return np.interp(times, times[good_points], locked_in)


def scanning_fft(signal, fs, tseg, tstep, 
        nfft=None, window='hamming', fit_point=5, low_f_skip=100,
        tqdm_bar=None):
    """Finds the changing dominate frequency of a oscillatory signal.

    Finds how the frequency of a oscillatory signal changes with time. This 
    is achieved by performing many FFTs over a small window of signal which 
    is slid along the complete signal. This is useful for extracting the 
    measurement from PDO and TDO experiments.

    Parameters
    ----------
    signal : numpy.ndarray
        The data to extract the signal from in the form of a 1d array.
    fs : float
        The sample frequency of the measurement signal in Hertz.
    tseg : float
        The length in time to examine for each FFT in seconds.
    tstep : float
        How far to shift the window between each FFT in seconds.
    nfft : None, optional
        The number of points to use for the FFT extra points will be zero 
        padded. The number of points used by default is ``20*tseg*fs``, 
        where ``tseg*fs`` is the length of the unpadded signal.
    window : str, optional
        The windowing function to used for the FFT that will be passed to 
        :func:`scipy.signal.get_window`. THe default is 'hamming'.
    fit_points : int, optional
        The number of points to fit a parabola to identify the peak of the 
        FFT. THe default is `5` and this is passed to :func:`polypeak`.
    low_f_slip : int, optional
        The number of points to skip when identifying the peak at the 
        beginning of the FFT to ignore the low freq upturn. The default is 
        `100` and this is passed to :func:`polypeak`.
    tqdm_bar : `tqdm.tqdm`, optional
        This function can be slow so a tqdm progress bar can be passed using 
        this keyword which will be updated to show the progress of the 
        calculation. This is done by::

            from tqdm import tqdm
            with tqdm() as bar: 
                res = scanning_fft(signal, fs, tseg, tstep, tqdm_bar=bar)

    Returns
    -------
    times : numpy.ndarray
        The midpoint of the time windows which the FFTs where taken at in 
        seconds.
    freqs : numpy.ndarray
        The frequencies of the dominate oscillatory signal against time in 
        Hertz.
    amps : numpy.ndarray
        The amplitude of the oscillatory signal from the FFT. This should be 
        in the units of the signal.
    """
    nperseg = int(tseg*fs)
    nstep = int(tstep*fs)
    if nfft is None:
        nfft = nperseg*20


    ntimepoint = int((len(signal)-nperseg+nstep)/nstep)
    times = (nperseg/2+np.arange(ntimepoint)*nstep)/fs
    freqs_i = np.zeros(ntimepoint)
    amps = np.zeros(ntimepoint)

    window = get_window('hamming', nperseg)

    if tqdm_bar is not None:
        tqdm_bar.reset(total=ntimepoint)

    for x in range(ntimepoint):
        if tqdm_bar is not None:
            tqdm_bar.update()
        freqs_i[x], amps[x] = polypeak(
            np.abs(
                np.fft.rfft(signal[x*nstep:nperseg+x*nstep]*window, n=nfft)),
            fit_point=fit_point, low_f_skip=low_f_skip)

    freqs = freqs_i*fs/nfft
    amps = 2*amps/np.average(window)/nperseg

    return times, freqs, amps

