from rhythm_regression.unit_conversion import uspb_to_bpm

import numpy as np

MAX_BPM_ROUNDING_ERROR = 0.0013333377777939859  # See MIDI.ipynb for computation.  Assumes a max tempo of 400.


def get_midi_vector(midi):
    """
    Takes a Mido midi object and returns the timestamps in seconds of the note onsets
    Adapted from https://www.programcreek.com/python/?code=jongwook%2Fonsets-and-frames%2Fonsets-and-frames-master%2Fonsets_and_frames%2Fmidi.py
    
    Arguments
    midi (mido.midifiles.midifiles.MidiFile): The MIDI object to parse

    Returns
    note_onsets (np.ndarray): The onsets of the notes
    """

    time = 0
    onset_times = []
    for message in midi:
        time += message.time
        if message.type == 'note_on' and message.velocity > 0:
            onset_times.append(time)        

    return np.array(onset_times) 


def get_bpm(midi):
    """
    Gets the tempo in bpm from the Mido MIDI object.
    Assumes one tempo per MIDI object.

    Arguments
    midi (mido.midifiles.midifiles.MidiFile): The MIDI object to parse

    Returns 
    bpm (float): The tempo in bpm
    """

    for message in midi.tracks[0]:
        if message.type == 'set_tempo':
            microseconds_per_beat = message.tempo
            bpm = uspb_to_bpm(microseconds_per_beat)

            if abs(bpm - round(bpm)) <= MAX_BPM_ROUNDING_ERROR:
                bpm = round(bpm)

            return bpm
