from pymusicxml import *

EIGHTH = 'eighth'

HALF_NOTE_DURATION = 2
QUARTER_NOTE_DURATION = 1
EIGHT_NOTE_DURATION = 0.5
SIXTEENTH_NOTE_DURATION = 0.25

# ##########################################################################################################

def _validate_pitches(pitches, num_expected):
    if not isinstance(pitches, list):
        pitches = [pitches for _ in range(num_expected)]
    if len(pitches) != num_expected:
        raise ValueError(f'Got more/less pitches than expected.  Expected {num_expected}, got {len(pitches)}')

    return pitches

# ##########################################################################################################

def quarter_rest(pitches):
    return Rest(QUARTER_NOTE_DURATION)


# ##########################################################################################################

def one(pitches):
    pitches = _validate_pitches(pitches, 1)
    
    return BeamedGroup([
        Note(pitches[0], QUARTER_NOTE_DURATION)
    ])


def e(pitches):
    pitches = _validate_pitches(pitches, 1)
    
    return BeamedGroup([
        Rest(SIXTEENTH_NOTE_DURATION),
        Note(pitches[0], SIXTEENTH_NOTE_DURATION),
        Rest(EIGHT_NOTE_DURATION)
    ])


def aand(pitches):
    pitches = _validate_pitches(pitches, 1)
    
    return BeamedGroup([
        Rest(EIGHT_NOTE_DURATION),
        Note(pitches[0], EIGHT_NOTE_DURATION)
    ])


def a(pitches):
    pitches = _validate_pitches(pitches, 1)
    
    return BeamedGroup([
        Rest(Duration(EIGHTH, num_dots=1)),
        Note(pitches[0], SIXTEENTH_NOTE_DURATION),
    ])

# ##########################################################################################################

def one_e(pitches):
    pitches = _validate_pitches(pitches, 2)
    
    return BeamedGroup([
        Note(pitches[0], SIXTEENTH_NOTE_DURATION),
        Note(pitches[1], SIXTEENTH_NOTE_DURATION),
        Rest(EIGHT_NOTE_DURATION)
    ])


def one_and(pitches):
    pitches = _validate_pitches(pitches, 2)
    
    return BeamedGroup([
        Note(pitches[0], EIGHT_NOTE_DURATION),
        Note(pitches[1], EIGHT_NOTE_DURATION)
    ])


def one_a(pitches):
    pitches = _validate_pitches(pitches, 2)
    
    return BeamedGroup([
        Note(pitches[0], Duration(EIGHTH, num_dots=1)),
        Note(pitches[1], SIXTEENTH_NOTE_DURATION)
    ])


def e_and(pitches):
    pitches = _validate_pitches(pitches, 2)
    
    return BeamedGroup([
        Rest(SIXTEENTH_NOTE_DURATION),
        Note(pitches[0], SIXTEENTH_NOTE_DURATION),
        Note(pitches[1], EIGHT_NOTE_DURATION)
    ])


def e_a(pitches):
    pitches = _validate_pitches(pitches, 2)
    
    return BeamedGroup([
        Rest(SIXTEENTH_NOTE_DURATION),
        Note(pitches[0], SIXTEENTH_NOTE_DURATION),
        Rest(SIXTEENTH_NOTE_DURATION),
        Note(pitches[1], SIXTEENTH_NOTE_DURATION)
    ])


def and_a(pitches):
    pitches = _validate_pitches(pitches, 2)
    
    return BeamedGroup([
        Rest(EIGHT_NOTE_DURATION),
        Note(pitches[0], SIXTEENTH_NOTE_DURATION),
        Note(pitches[1], SIXTEENTH_NOTE_DURATION)
    ])


# ##########################################################################################################

def one_e_and(pitches):
    pitches = _validate_pitches(pitches, 3)
    
    return BeamedGroup([
        Note(pitches[0], SIXTEENTH_NOTE_DURATION),
        Note(pitches[1], SIXTEENTH_NOTE_DURATION),
        Note(pitches[2], EIGHT_NOTE_DURATION)
    ])


def one_e_a(pitches):
    pitches = _validate_pitches(pitches, 3)
    
    return BeamedGroup([
        Note(pitches[0], SIXTEENTH_NOTE_DURATION),
        Note(pitches[1], EIGHT_NOTE_DURATION),
        Note(pitches[2], SIXTEENTH_NOTE_DURATION)
    ])


def one_and_a(pitches):
    pitches = _validate_pitches(pitches, 3)
    
    return BeamedGroup([
        Note(pitches[0], EIGHT_NOTE_DURATION),
        Note(pitches[1], SIXTEENTH_NOTE_DURATION),
        Note(pitches[2], SIXTEENTH_NOTE_DURATION)
    ])


def e_and_a(pitches):
    pitches = _validate_pitches(pitches, 3)
    
    return BeamedGroup([
        Rest(SIXTEENTH_NOTE_DURATION),
        Note(pitches[0], SIXTEENTH_NOTE_DURATION),
        Note(pitches[1], SIXTEENTH_NOTE_DURATION),
        Note(pitches[2], SIXTEENTH_NOTE_DURATION)
    ])

# ##########################################################################################################

def one_e_and_a(pitches):
    pitches = _validate_pitches(pitches, 4)
    
    return BeamedGroup([
        Note(pitches[0], SIXTEENTH_NOTE_DURATION),
        Note(pitches[1], SIXTEENTH_NOTE_DURATION),
        Note(pitches[2], SIXTEENTH_NOTE_DURATION),
        Note(pitches[3], SIXTEENTH_NOTE_DURATION)
    ])

# ##########################################################################################################

def release_measure(pitch):
    release_measure = Measure(None)
    release_measure.append(Note(pitch, QUARTER_NOTE_DURATION))
    release_measure.append(Rest(QUARTER_NOTE_DURATION))
    release_measure.append(Rest(HALF_NOTE_DURATION))
    return release_measure