MICROSECONDS_PER_SECOND = 1000000
MILLISECONDS_PER_SECOND = 1000
SECONDS_PER_MINUTE = 60

SIXTEENTH_NOTES_PER_BEAT = 4
SEXTUPLETS_PER_BEAT = 6
THIRTY_SECOND_NOTES_PER_BEAT = 8


def uspb_to_bpm(uspb):
    return SECONDS_PER_MINUTE * MICROSECONDS_PER_SECOND / uspb


def bpm_to_usbp(bpm):
    return SECONDS_PER_MINUTE * MICROSECONDS_PER_SECOND / bpm


def hz_to_ms_per_note(h):
    return MILLISECONDS_PER_SECOND / h


def hz_to_8rbpm(h):
    return h * SECONDS_PER_MINUTE / SIXTEENTH_NOTES_PER_BEAT


def hz_to_12rbpm(h):
    return h * SECONDS_PER_MINUTE / SEXTUPLETS_PER_BEAT


def hz_to_16rbpm(h):
    return h * SECONDS_PER_MINUTE / THIRTY_SECOND_NOTES_PER_BEAT


def ms_per_note_to_hz(mspn):
    return MILLISECONDS_PER_SECOND / mspn

def ms_per_note_to_8rbpm(mspn):
    return MILLISECONDS_PER_SECOND * SECONDS_PER_MINUTE / SIXTEENTH_NOTES_PER_BEAT / mspn


def ms_per_note_to_12rbpm(mspn):
    return MILLISECONDS_PER_SECOND * SECONDS_PER_MINUTE / SEXTUPLETS_PER_BEAT / mspn


def ms_per_note_to_16rbpm(mspn):
    return MILLISECONDS_PER_SECOND * SECONDS_PER_MINUTE / THIRTY_SECOND_NOTES_PER_BEAT / mspn


def _8rbpm_to_hz(erbpm):
    return erbpm * SIXTEENTH_NOTES_PER_BEAT / SECONDS_PER_MINUTE


def _8rbpm_to_ms_per_note(erbpm):
    return MILLISECONDS_PER_SECOND * SECONDS_PER_MINUTE / (SIXTEENTH_NOTES_PER_BEAT * erbpm)


def _8rbpm_to_12rbpm(erbpm):
    return erbpm * SIXTEENTH_NOTES_PER_BEAT / SEXTUPLETS_PER_BEAT


def _8rbpm_to_16rbpm(erbpm):
    return erbpm * SIXTEENTH_NOTES_PER_BEAT / THIRTY_SECOND_NOTES_PER_BEAT


def _12rbpm_to_hz(trbpm):
    return trbpm * SEXTUPLETS_PER_BEAT / SECONDS_PER_MINUTE


def _12rbpm_to_ms_per_note(trbpm):
    return MILLISECONDS_PER_SECOND * SECONDS_PER_MINUTE / (SEXTUPLETS_PER_BEAT * trbpm)


def _12rbpm_to_8rbpm(trbpm):
    return trbpm * SEXTUPLETS_PER_BEAT / SIXTEENTH_NOTES_PER_BEAT


def _12rbpm_to_16rbpm(trbpm):
    return trbpm * SEXTUPLETS_PER_BEAT / THIRTY_SECOND_NOTES_PER_BEAT    


def _16rbpm_to_hz(srbpm):
    return srbpm * THIRTY_SECOND_NOTES_PER_BEAT / SECONDS_PER_MINUTE


def _16rbpm_to_ms_per_note(srbpm):
    return MILLISECONDS_PER_SECOND * SECONDS_PER_MINUTE / (THIRTY_SECOND_NOTES_PER_BEAT * srbpm)


def _16rbpm_to_8rbpm(srbpm):
    return srbpm * THIRTY_SECOND_NOTES_PER_BEAT / SIXTEENTH_NOTES_PER_BEAT


def _16rbpm_to_12rbpm(srbpm):
    return srbpm * THIRTY_SECOND_NOTES_PER_BEAT / SEXTUPLETS_PER_BEAT
