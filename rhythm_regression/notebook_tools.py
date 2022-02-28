from rhythm_regression.unit_conversion import MILLISECONDS_PER_SECOND
from rhythm_regression.audio_processing import amplitude_envolope

import numpy as np
import matplotlib.pyplot as plt


def plot_signal(signal, samples=None, sampling_rate=22050, time_range=None, units='s', title='', axs=None):
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

    title += f' (Sampled at {sampling_rate} Hz)'
    plt.title(title, fontsize=24)
    plt.xticks(fontsize=14)
    plt.xlabel(x_axis_label, fontsize=18)

    axs.plot(x_data[in_time_range], signal[in_time_range])
    
    return axs


def plot_amplitude_envelope(audio, original_signal=True, frame_size=1024, hop_length=512, **kwargs):
    """
    Plots the amplitude envelope of a signal, which is a sliding window maximum.  
    See https://www.youtube.com/watch?v=rlypsap6Wow&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&index=9

    Arguments
    audio ():
    original_signal (bool): Whether or not to plot the original signal as well as the amplitude envelope
    frame_size (int): The frame (window) size for the amplitude envelope
    hop_length (int): The hop length (stride) for sliding the frame for the amplitude envelope
    **kwargs: Arguments for plot_signal()

    Returns
    axs (matplotlib.axes.Axes): The axes that the signals were plotted on
    """

    samples, ae = amplitude_envolope(audio, frame_size, hop_length)
    
    if original_signal:
        axs = plot_signal(audio, **kwargs)
    else:
        axs = None
    axs = plot_signal(ae, samples, axs=axs, **kwargs)

    return axs
