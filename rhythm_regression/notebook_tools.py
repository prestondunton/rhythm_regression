from rhythm_regression.unit_conversion import MILLISECONDS_PER_SECOND, SECONDS_PER_MINUTE
from rhythm_regression.audio_processing import amplitude_envolope, transients

from rhythm_regression.audio_processing import SAMPLING_RATE, FRAME_SIZE, HOP_LENGTH, AMPLITUDE_THRESHOLD

import librosa
import numpy as np
import matplotlib.pyplot as plt


def plot_signal(signal, samples=None, sampling_rate=SAMPLING_RATE, time_range=None, units='s', title='', axs=None, **kwargs):
    """
    Plots an signal using matplotlib and allows for time slicing and 
    different units

    Arguments:
    signal (list-like): The signal (y data) to be plotted
    samples (list-like): The samples (x data) to plot the signal against
    sampling_rate (int): The number of samples taken per second
    time_range (tuple of length 2): A tuple specifying start and end times to plot in the signal.  Uses units specified by the units argument.
    units (['s', 'ms', 'samples']): Units to use for slicing time with time_range and to plot with.
    title (str): The title of the plot
    axs (matplotlib.axes.Axes): Axes to plot the signal on

    Retruns
    axs (matplotlib.axes.Axes): The axes that the signal was plotted on

    
    """
    num_samples = len(signal)
    if samples is None:
        samples = np.array(range(num_samples))

    if units == 'samples':
        x_data = samples
        x_axis_label = 'Samples'
    elif units == 's':
        x_data = np.array([sample / sampling_rate for sample in samples])
        x_axis_label = 'Time (s)'
    elif units == 'ms':
        x_data = np.array([(sample * MILLISECONDS_PER_SECOND) / sampling_rate for sample in samples])
        x_axis_label = 'Time (ms)'
    else:
        raise ValueError(f'{units} is not a valid unit.  Use one of [\'s\', \'ms\', \'samples\'].')

    if time_range is not None:
        if len(time_range) != 2:
            raise ValueError(f'Invalid time range.  Please pass a tuple of length two of a start and end time.')
        if time_range[0] > time_range[1]:
            raise ValueError(f'Invalid time range.  Make sure the start time is less than or equal to the end time.')
            
        in_time_range = (time_range[0] <= x_data) & (x_data <= time_range[1])
    else:
        in_time_range = [True] * num_samples

    if axs is None:
        plt.figure(figsize=(20,10))
        axs = plt.gca()

    if sampling_rate != 22050:
        title += f' (Sampled at {sampling_rate} Hz)'
    plt.title(title, fontsize=24)
    plt.xticks(fontsize=14)
    plt.xlabel(x_axis_label, fontsize=18)

    if 'fmt' in kwargs.keys():
        format = kwargs.pop('fmt')
        axs.plot(x_data[in_time_range], signal[in_time_range], format, **kwargs)
    else:
        axs.plot(x_data[in_time_range], signal[in_time_range], **kwargs)
    
    return axs


def plot_amplitude_envelope(signal, original_signal=True, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, axs=None, **kwargs):
    """
    Plots the amplitude envelope of a signal, which is a sliding window maximum.  
    See https://www.youtube.com/watch?v=rlypsap6Wow&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&index=9

    Arguments
    signal (np.ndarray): The original signal to calculate the amplitude envelope on
    original_signal (bool): Whether or not to plot the original signal as well as the amplitude envelope
    frame_size (int): The frame (window) size for the amplitude envelope
    hop_length (int): The hop length (stride) for sliding the frame for the amplitude envelope
    **kwargs: Arguments for plot_signal()

    Returns
    axs (matplotlib.axes.Axes): The axes that the signals were plotted on
    """

    samples, ae = amplitude_envolope(signal, frame_size, hop_length)
    
    if original_signal:
        axs = plot_signal(signal, axs=axs, **kwargs)

    axs = plot_signal(ae, samples, axs=axs, **kwargs)

    return axs


def plot_rms_energy(signal, original_signal=True, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, axs=None, **kwargs):
    """
    Plots the Root Mean Square Energy of a signal, which is a sliding window root mean square.  
    See https://www.youtube.com/watch?v=EycaSbIRx-0&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&index=9

    Arguments
    signal (np.ndarray): The original signal to calculate the amplitude envelope on
    original_signal (bool): Whether or not to plot the original signal as well as the amplitude envelope
    frame_size (int): The frame (window) size for the amplitude envelope
    hop_length (int): The hop length (stride) for sliding the frame for the amplitude envelope
    **kwargs: Arguments for plot_signal()

    Returns
    axs (matplotlib.axes.Axes): The axes that the signals were plotted on
    """

    rmse = librosa.feature.rms(y=signal, frame_length=frame_size, hop_length=hop_length).flatten()
    frames = range(len(rmse))
    samples = librosa.frames_to_samples(frames, hop_length=hop_length)
    
    if original_signal:
        axs = plot_signal(signal, axs=axs, **kwargs)

    axs = plot_signal(rmse, samples, axs=axs, **kwargs)

    return axs


def plot_rmse_transients(signal, time_range=None, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, 
                         amplitude_threshold=AMPLITUDE_THRESHOLD, title='', axs=None):
    """
    Plots the transients detected from local maxima of the RMSEnergy signal

    Arguments
    signal (np.ndarray): The signal to calculate transients on and plot
    time_range (tuple of length 2): The time range to slice the signal with
    frame_size (int): The frame (window) size for the rmse signal calculation
    hop_length (int): The hop length (stride) for sliding the frame for the rmse signal calculation
    amplitude_threshold (float): The amplitude threshold for the rmse signal calculation
    title (str): The title for the plot
    axs (matplotlib.axes.Axes): Axes to plot the signals and transients on

    Returns
    axs (matplotlib.axes.Axes): The axes that the signals and transients are drawn on
    """

    axs = plot_signal(signal, time_range=time_range, axs=axs)

    axs = plot_rms_energy(signal, frame_size=frame_size, hop_length=hop_length, time_range=time_range, 
                            fmt='-', original_signal=False, axs=axs, title=title)

    transients = transients(signal, frame_size=frame_size, hop_length=hop_length, 
                                        amplitude_threshold=amplitude_threshold)

    if time_range is None:
        time_range = (transients.min(), transients.max())
    for transient in transients:
        if time_range[0] <= transient <= time_range[1]:
            axs.axvline(transient, color='black')

    return axs


def plot_midi_vector(x, bpm, time_range=None, draw_beats=True, x_label=None, y_level=0.5, 
                    subdivisions_per_beat=1, units='beats', axs=None, figsize=(12.8, 1), title='', **kwargs):
    """
    Plots the MIDI vector as a series of points along a time axis.

    Arguments
    x (np.array): The times of the MIDI notes in seconds.
    bpm (float): The tempo of the MIDI used for ploting beats, subdivisions, and x ticks.
    time_range (tuple of length 2): A tuple specifying start and end times to plot in the MIDI vector.  Uses units specified by the units argument.
    draw_beats (bool): Whether or not to draw the beats and subdivisions as vertical lines.
    x_label (str): The string to label the x axis with.  Defaults to 'Time (units)'.
    y_level (float): The height to plot the points at.  Must be in range [0,1].
    subdivision_per_beat (float): The number of subdivisions per beat to plot if draw_beats is True.
    units (['s', 'ms', 'beats']): The units to plot on the x axis and to interpret time_range with.
    axs (matplotlib.axes.Axes): The axes to draw the MIDI vector on.
    figsize (tuple of length 2): The figure size.
    title (str): The title of the plot.
    **kwargs: Keyword arguments to pass to axs.plot().

    Returns 
    axs (matplotlib.axes.Axes): The axes that the MIDI vector was drawn on
    """

    if axs is None:
        plt.figure(figsize=figsize)
        axs = plt.gca()

    if x_label is None:
        x_label = f'Time ({units})'

    if y_level < 0 or y_level > 1:
        raise ValueError('y_level must be in the range [0,1]')

    seconds_per_beat = SECONDS_PER_MINUTE / bpm
    if units == 's':
        beat_lines = np.arange(0, np.nanmax(x), seconds_per_beat)
        subdivision_lines = np.arange(0, np.nanmax(x), seconds_per_beat / subdivisions_per_beat)
    elif units == 'ms':
        x = x * MILLISECONDS_PER_SECOND
        beat_lines = np.arange(0, max(x), seconds_per_beat) * MILLISECONDS_PER_SECOND
        subdivision_lines = np.arange(0, max(x), seconds_per_beat / subdivisions_per_beat) * MILLISECONDS_PER_SECOND
    elif units == 'beats':
        x = x * bpm / SECONDS_PER_MINUTE
        beat_lines = np.arange(0, max(x) * bpm / SECONDS_PER_MINUTE)
        subdivision_lines = np.arange(0, max(x) * bpm / SECONDS_PER_MINUTE, 1 / subdivisions_per_beat)
    else:
        raise ValueError(f'{units} is not a valid unit.  Use one of [\'s\', \'ms\', \'beats\'].')

    if time_range is None:
        time_range = (min(x), max(x))
    in_time_range = (time_range[0] <= x) & (x <= time_range[1])
    
    axs.plot(x[in_time_range], y_level * np.ones_like(x[in_time_range]), 'o', **kwargs)
    plt.xlabel(x_label, fontsize=16)
    plt.yticks([])
    plt.ylim([0,1])
    plt.title(title, fontsize=18)

    if draw_beats:
        axs = draw_beat_lines(axs, beat_lines, subdivision_lines, time_range)
    
    return axs

def draw_beat_lines(axs, beat_lines, subdivision_lines, time_range):
    """
    Draws the vertical lines of beats and their subdivisions on the provided axes.

    Arguments
    axs (matplotlib.axes.Axes): Axes to plot the lines on
    beat_lines (list-like): The x coordinates of the beat lines
    subdivision_lines (list-like): The x coordinates of the subdivision lines
    time_range (tuple of length 2): A tuple specifying start and end times to plot in the signal.

    Returns
    axs (matplotlib.axes.Axes): The axes that the lines were drawn on
    """

    for line in beat_lines:
        if time_range[0] <= line <= time_range[1]:
            axs.axvline(line, color='black')

    for line in subdivision_lines:
        if time_range[0] <= line <= time_range[1]:
            axs.axvline(line, color='black', lw=0.5)

    return axs