import numpy as np


def center_transients_on_midi(t, m):
    """
    Modifies the transient vector in place so that 
    the errors between the transients and midi have mean 0.

    Arguments
    t (np.ndarray): The transient vector
    m (np.ndarray): The midi vector
    """

    error = m - t
    t += np.nanmean(error)


def get_tempo_vector(t, m, bpm):
    """
    Computes the tempo vector (in bpm) given the 
    transient vector, the midi vector, and the
    correct bpm.

    Arguments
    t (np.ndarray): The transient vector
    m (np.ndarray): The midi vector
    bpm (float): The bpm of the original MIDI

    returns
    tempo_vector (np.ndarray): The tempo vector
    """

    midi_note_durations = np.diff(m)
    transient_note_durations = np.diff(t)

    tempo_vector = (bpm * midi_note_durations) / transient_note_durations
    return tempo_vector


def delete_transients(t, m):

    if m.min() > 0:
        m = m - m.min()

    min_tae = np.inf
    q_best = None

    for q in range(0, len(t) - len(m) + 1):
        tae = np.abs((m+t[q]) - t[q : q+len(m)]).sum()

        if tae < min_tae:  # less than gives us the first occurance, because solutions are non-unique
            min_tae = tae
            q_best = q

    m_star = m + t[q_best]
    t_star = [None] * len(m_star)

    # each midi note selects its nearest transient
    claimed_indices = []
    for i in range(len(m_star)):
        # we only have to check a neighborhood of 3 since m_star is shifted and both lists are sorted
        left_index = max(q_best + i - 1, 0)
        middle_index = q_best + i
        right_index = min(q_best + i + 1, len(t)-1)

        neighbors = [left_index, middle_index, right_index]
        neighbors.sort(key=lambda index: abs(m_star[i] - t[index]))
        for j in range(len(neighbors)):
            if neighbors[j] not in claimed_indices:
                nearest_transient = neighbors[j]
                claimed_indices.append(neighbors[j])
                break

        t_star[i] = t[nearest_transient]

    return np.array(t_star)


def add_nan_transients(t, m):
    m_star = delete_transients(m,t - t.min())
    m_star_set = set(m_star)  # used for O(1) "in" operator on next line.
    deleted_indices = [i for i in range(len(m)) if m[i] not in m_star_set]

    t_star = t.copy()
    for i in deleted_indices:
        if i >= len(t_star):
            t_star = np.append(t_star, np.nan)
        else:
            t_star = np.insert(t_star, i, np.nan)
    
    return t_star
