'''GigaAnalysis - Digital Lock In

This program is to recreate what a lock in would do for slower measurements
but for our high field experiments. This is based around what the program in 
DRS and WUH did.

'''

from .data import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import (blackmanharris,  # for find_freq
    hamming,  # for hamming_window
    butter, filtfilt)  # for butter_bandpass and butter_bandpass_filter
from numpy.fft import rfft, rfftfreq # for find_freq
from scipy.optimize import minimize  # for find_phase, phase_in


def polypeak(data, fit_points=3):
    '''
    This finds the highest point in the data set.
    It fits a quadratic polynomial to the data to 
    interpolate the highest point.
    Args:
        data (numpy array): the data to find the max peak of
        fit_points (int default=3): the number of points to use to fit
    Returns:
        A tuple of the coordinate of the highest point
    '''
    if fit_points%2 != 1:
        ValueError('fit_points needs to be odd')
    m = int((fit_points - 1)/2)  # Num of points either side
    i = np.argmax(data)  # Find highest point in data
    # Fit a quadratic and get the x and y of the highest point
    a, b, c = np.polyfit(np.arange(i-m,i+m+1), data[i-m:i+m+1], 2) 
    x = -0.5*b/a  
    y = a*x**2 + b*x + c
    return x, y


def find_freq(data, samp_freq, padding=1, fit_point=3, plot=False, amp=False):
    '''
    Find the dominate frequency in oscillatory data.
    This is achieved by applying a FFT and looking for the peak.
    To get a better estimation of the peak polypeak is used.
    Args:
        data (numpy array): The signal in evenly spaced points
        samp_freq (float): The frequency of the data points
        padding (float default=3):The amount of time to expand the data 
            for padding
        fit_points (int default=3): Number of points polypeak uses to fit
        plot (bool default=False): If true a plot of the fft is shown
        amp (bool default=False): If true amplitude included in output 
    Returns:
        Returns a float of the dominate frequency
    '''
    windowed = data * blackmanharris(len(data))
    n_fft = round(padding*len(windowed))
    f = abs(rfft(windowed, n=n_fft))
    f_freq = rfftfreq(padding*len(windowed), 1/samp_freq)
    skip = 40 # Points to skip at the start
    peak_freq = samp_freq*(polypeak(f[skip:],
        fit_points=fit_point)[0]+skip)/n_fft
    peak_amp = polypeak(f[skip:], fit_points=fit_point)[1]
    if plot:  # This plots the FFT it uses to find the frequency
        plt.plot(f_freq, f, '.')
        plt.plot(peak_freq, peak_amp, '.')
        plt.show()
    if amp:
        return peak_freq, peak_amp
    return peak_freq


def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    Produces the polynomial values for a Butterworth bandpass.
    Uses scipy.signal.butter
    Args:
        lowcut (float): The lowest frequency cut off.
        highcut (float): The high frequency cut off.
        fs (float): The frequency of the data points.
        order (int default=5): The order of the filter.
    Returns:
        The values b and a for the filtfilt function.
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    Applies a Butterworth bandpass filter to some data.
    Uses diglock.butter_bandpass
    Args:
        data (numpy array): A array containing the signal
        lowcut (float): The lowest frequency cut off
        highcut (float): The high frequency cut off
        fs (float): The sample frequency of the data points
        order (int default=5): The order of the filter
    Returns:
        Returns the filtered data.
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def gen_ref(freq, fs, phase, number):
    '''
    This produces the reference signal for the digital lock in.
    Args:
        freq (float): The frequency of the signal
        fs (float): Sample frequency of the data points
        phase (float): Phase of the signal in degrees
        number (int): Number of data points to generate
    Returns:
        A numpy array containing the reference signal
    '''
    return np.sin(2*np.pi*freq*np.arange(0, number, 1)/fs 
                          + phase*np.pi/180)


def find_phase(data, fs, freq):
    '''
    Finds the phase of the a signal.
    The phase is given in degrees between 0 and 360
    Args:
        data (numpy array): A array containing the signal
        fs (float): The sample frequency of the data points
        freq (float): The frequency of the signal
    Returns:
        The phase in degrees
    '''
    def to_min(p):  # Makes the signal to compare and sums their product
        return -np.sum(data*gen_ref(freq, fs, p, len(data)))
    return float(minimize(to_min, 180, bounds=[(0, 360)]).x)


def round_oscillation(time_const, freq):
    '''
    Args:
        time_const (float): Time for averaging
        freq (float): Frequency of the signal
    Returns:
        A int of the closest whole number of oscillations.
    '''
    return round(time_const*freq)


def flat_window(time_const, fs, freq):
    '''
    Produces a window for averaging using convolve.
    Makes the window the same length as a whole number of oscillations.
    Uses diglock.round_ossillation
    Args:
        time_const (float): Time for averaging
        fs (float): Frequency of samples
        freq (float): Frequency of the signal
    Returns:
        A numpy array to be convolved with the signal.
    '''
    window = np.full(int(round_oscillation(time_const, freq)*fs/freq), 1.0)
    return window/np.sum(window)


def hamming_window(time_const, fs, freq):
    '''
    Produces a window for averaging using convolve.
    The window is the shape of a hamming filter.
    Uses scipy.signal.hamming
    Args:
        time_const (float): Time for averaging
        fs (float): Frequency of samples
        freq (float): Frequency of the signal
    Returns:
        A numpy array to be convolved with the signal.
    '''
    window = hamming(round(time_const*fs/0.54))
    return window/np.sum(window)


def ham_lock_in(signal, time_const, fs, freq, phase):
    '''
    Prefroms a lock in of the signal and avrages useing a hamming window.
    The window is convolved with the signal times a refrence wave.
    Uses diglock.hamming_window and diglock.gen_ref
    Args:
        signal: (numpy array): AC signal to lock in
        time_const (float): Time for averaging
        fs (float): Frequency of samples
        freq (float): Frequency of the signal
        phase (float): Phase of signal
    Returns:
        A numpy array of the signal after lock in should be in RMS.
    '''
    if time_const*fs > len(signal):
        raise ValueError('Averaging window longer than signal')
    window = hamming_window(time_const, fs, freq)
    lock_ref = gen_ref(freq, fs, phase, len(signal))
    return np.convolve(signal*lock_ref, window, mode='same')*np.sqrt(2)


def flat_lock_in(signal, time_const, fs, freq, phase):
    '''
    Prefroms a lock in of the signal and avrages useing a hamming window.
    The window is convolved with the signal times a refrence wave.
    Uses diglock.hamming_window and diglock.gen_ref
    Args:
        signal: (numpy array): AC signal to lock in
        time_const (float): Time for averaging
        fs (float): Frequency of samples
        freq (float): Frequency of the signal
        phase (float): Phase of signal
    Returns:
        A numpy array of the signal after lock in.
    '''
    window = flat_window(time_const, fs, freq)
    lock_ref = gen_ref(freq, fs, phase, len(signal))
    return np.convolve(signal*lock_ref, window, mode='same')*np.sqrt(2)


def phase_in(signal_in, signal_out, **kwargs):
    """
    This given an in and out of phase signal returns the phase shift to
    move the majority of the change in signal into the in phase component.
    It also chooses the phase shift so the change is positive and between
    0 and 360 deg.
    Uses :func:`scipy.optimize.minimize` and kwargs are passed to it

    Parameters
    ---------
    signal_in : 1d numpy.ndarray
        The values containing the in phase signal
    signal_in : 1d numpy.ndarray
        The values containing the out of phase signal needs to be the same
        shape as signal_in
    kwargs:
        Keyword arguments are passed to :func:`scipy.optimize.minimize`

    Returns
    -------
    max_phase : float
        The phase in degrees between 0 deg and 360 deg where the change in
        signal in the out of phase is minimised.
    """
    if signal_in.shape != signal_out.shape:
        raise(ValueError("The singal_in and singal_out arrays need to be "
            "the same shape. They are {} and {}".format(
                signal_in.shape, signal_out.shape)))
    scaling = 1./np.sum(
        np.sqrt(signal_in*signal_in + signal_out*signal_out))
    # Produce function to minimise
    def to_min(phase):
        new_out = signal_out*np.cos(phase*np.pi/180) - \
            signal_in*np.sin(phase*np.pi/180)
        resisual = (new_out - np.average(new_out))*scaling
        return np.average(resisual*resisual)
    # Find the minimum
    max_phase = np.asscalar(minimize(to_min, x0=0.,
        method='Nelder-Mead', **kwargs).x) % 180
    # Pick double value that makes signal positive
    if np.sum(signal_in*np.cos(max_phase*np.pi/180) + \
            signal_out*np.sin(max_phase*np.pi/180)) < 0:
        return max_phase + 180
    else:
        return max_phase


def select_not_spikes(data, sdl=2., region=1001):
    '''
    Finds the spikes inside the data set and excludes the region around them.
    It does this by looking at when the values change unusually quickly.
    Args:
        data (np array): The signal to examine
        sdl (float default=2.): The number of standard deviations the data
            to be deviate from to be considered a outlier.
        region (int default=1001): The number of points around the outlier
            that are also considered bad.
    Returns:
        A np array of boolean values where true means good values.
    '''
    change = np.append(np.diff(data), 0)
    spikes = abs(change - np.mean(change)) > sdl*np.std(change)
    good_vals = np.convolve(spikes, np.ones(region), mode='same') == 0
    return good_vals


def spike_lock_in(signal, time_const, fs, freq, phase, sdl=2, region=1001):
    '''
    Preforms a lock in of the signal and averages using a hamming window, the
    same way as diglock.ham_lock_in. It also removes the points effected by
    the spikes and interpolates between them using diglock.select_not_spikes.
    Args:
        signal: (numpy array): AC signal to lock in
        time_const (float): Time for averaging
        fs (float): Frequency of samples
        freq (float): Frequency of the signal
        phase (float): Phase of signal
        sdl (float default=2.): The number of standard deviations the data
            to be deviate from to be considered a outlier.
        region (int default=1001): The number of points around the outlier
            that are also considered bad.
    Returns:
        A numpy array of the signal after lock in.
    '''
    window = hamming_window(time_const, fs, freq)
    lock_ref = gen_ref(freq, fs, phase, len(signal))
    times = np.arange(len(signal))
    good_points = select_not_spikes(signal*lock_ref, sdl, region)
    locked_in = np.convolve((signal*lock_ref)[good_points],
                            window, mode='same')*np.sqrt(2)
    # step_size = int(np.ceil(len(window)/pp_window))
    return np.interp(times, times[good_points], locked_in)

