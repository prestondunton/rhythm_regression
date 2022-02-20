from unit_conversion import MILLISECONDS_PER_SECOND

import numpy as np
import matplotlib.pyplot as plt

def plot_audio_sample(audio, sampling_rate=None, time_range=None, units='s', title=''):
    """
    Plots an audio sample using matplotlib and allows for time slicing and 
    different units

    Arguments:
    audio (list-like): The waveform to be plotted
    title (str): The title of the plot
    sampling_rate (int): The number of samples taken per second
    time_range (tuple of length 2): A tuple specifying start and end times to plot in the audio.  Uses units specified by the units argument.
    units (['s', 'ms', 'samples']): Units to use for slicing time with time_range and to plot with.
    
    """

    plt.figure(figsize=(20,10))

    if sampling_rate is None:
        if units == 's' or units == 'ms':
            raise ValueError('Sampling rate is None, but is needed for calculating x axis ticks.  '
            'Pass a sampling rate or set units to \'samples\'.')
    else:
        title += f' (Sampled at {sampling_rate} Hz)'
        plt.title(title, fontsize=24)

    num_samples = len(audio)
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

    plt.plot(x_data[in_time_range], audio[in_time_range])
    plt.xticks(fontsize=14)
    plt.xlabel(x_axis_label, fontsize=18)
