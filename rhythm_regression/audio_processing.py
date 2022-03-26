import librosa
import numpy as np

SAMPLING_RATE = 22050
HOP_LENGTH = 350
FRAME_SIZE = 700
AMPLITUDE_THRESHOLD = 0.02


def amplitude_envolope(signal, frame_size=FRAME_SIZE, hop_length=HOP_LENGTH):
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


def rms_energy_transients(signal, sampling_rate=SAMPLING_RATE, frame_size=FRAME_SIZE, 
                            hop_length=HOP_LENGTH, amplitude_threshold=AMPLITUDE_THRESHOLD):
    """
    Returns the times of transients in the signal based on
    local maxima of the RMSEnergy of the signal.

    Arguments
    signal (list-like): The signal to get transients of
    sampling_rate (int): The sampling rate to use for conversion to time
    frame_size (int): The number of samples per frame
    hop_length (int): The number of samples to shift frames by
    amplitude_theshold (int): The lowest rms_energy that counts as a transient

    Returns
    rmse_transients (np.ndarray): A numpy array of the times (in seconds) of transients
    """

    rmse = librosa.feature.rms(signal, frame_length=frame_size, hop_length=hop_length).flatten()
    rmse_transient_frames = arg_where_local_max(rmse)

    # filter out transients that are below amplitude_threhsold
    rmse_transients_frames = np.array([frame for frame in rmse_transient_frames if rmse[frame] > amplitude_threshold])

    rmse_transients = librosa.frames_to_time(rmse_transients_frames, sr=sampling_rate, hop_length=hop_length)

    return rmse_transients


def arg_where_local_max(signal):
    """
    Returns the indices of local maxima in the signal

    Arguments
    signal (list-like): The signal to process

    Returns 
    (np.ndarray): The indices of local maxima in the signal
    """

    indices = []
    for i in range(1, len(signal)-1):
        if (signal[i-1] <= signal[i]) and (signal[i+1] < signal[i]):
            indices.append(i)

    return np.array(indices)