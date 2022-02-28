import librosa
import numpy as np


def amplitude_envolope(signal, frame_size=1024, hop_length=512):
    """
    Calculates the amplitude envelope of a signal.
    See https://www.youtube.com/watch?v=SRrQ_v-OOSg&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&index=8

    Arguments
    signal (list-like): The signal to calculate on
    frame_size (int): The size of the frame, or window to slide on the signal
    hop_length (int): The hop length (stride) of the frame

    Returns
    samples (np.ndarray): The sample indices into signal for each corresponding point in the amplitude envelope
    ae (np.ndarray): The amplitude envelope of the signal
    """


    ae = np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])
    frames = range(len(ae))
    samples = librosa.frames_to_samples(frames, hop_length=hop_length)

    return samples, ae
