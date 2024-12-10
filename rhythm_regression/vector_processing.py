import numpy as np


def numpy_snap(arr, snap_resolution):
    return (arr / snap_resolution).round() * snap_resolution


def center_midi_on_transients(m, t, method='median'):
    """
    Returns a copy of m so that 
    m and t have the same mean

    Arguments
    t (np.ndarray): The transient vector
    m (np.ndarray): The midi vector
    """

    new_m = m.copy()

    if method == 'mean':
        new_m += (t.mean() - m.mean())
    elif method == 'median':
        new_m += (np.median(t) - np.median(m))
    else:
        raise ValueError(f'Method for centering must either be mean or median.  Got {method}')

    return new_m


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


def validate_matching(m, t, matching):
    m_indexes = sorted([mi for (mi, _) in matching if mi is not None])
    t_indexes = sorted([ti for (_, ti) in matching if ti is not None])

    if m_indexes != list(range(len(m))):
        raise RuntimeError(f'm indexes are not valid {m_indexes} for vectors and matchings \n {len(m)=} \n {matching=}')

    if t_indexes != list(range(len(t))):
        raise RuntimeError(f't indexes are not valid {t_indexes} for vectors and matchings \n {len(t)=} \n {matching=}')



def score_matching(pred_match, actual_match):
    pred_match_set = set(pred_match)    
    actual_match_set = set(actual_match)    

    # return the jaccard index
    return len(pred_match_set & actual_match_set) / len(pred_match_set | actual_match_set)


def match_rhythms(m, t, **kwargs):
    """
    A wrapper function for the current version of rhythm matching.
    When rhythm matching changes, this is the only method that should
    have to change.  All calling methods of rhythm matching should
    call this method.

    Arguments
    m (np.ndarray): The midi vector of timestamps
    t (np.ndarray): The transient vector of timestamps

    Returns
    matchings (list of tuples of length 2): Indices (or None for insertion / deletion) into the m and t vectors.
    matched_m (np.ndarray): A list of m timestamps or None that correspond to elements of matched_t
    matched_t (np.ndarray): A list of t timestamps or None that correspond to elements of matched_m
    """

    matchings = min_cost_match(m, t, **kwargs)
    matched_m = None
    matched_t = None

    matchings.sort(key=lambda tup: tup[0] if tup[0] is not None else tup[1])
    validate_matching(m, t, matchings)

    return matchings, matched_m, matched_t 
   

def min_cost_match(m, t):

    cost_matrix = get_cost_matrix(m, t)
    matchings = get_min_cost_dp(cost_matrix)

    # resolve duplicates
    matchings.sort(key=lambda tup: tup[0] if tup[0] is not None else tup[1])
    matchings = resolve_duplicate_matchings(cost_matrix, matchings)

    return matchings


def get_cost_matrix(m, t):
    cost_matrix = np.empty(shape=(len(m), len(t)))

    m_diff = np.diff(m)
    t_diff = np.diff(t)

    m_diff = np.append(m_diff, m_diff[-1])
    t_diff = np.append(t_diff, t_diff[-1])

    for i in range(len(m)):
        for j in range(len(t)):
            left_diff = abs((m_diff[i] if i == 0 else m_diff[i-1]) - (t_diff[j] if j == 0 else t_diff[j-1]))
            right_diff = abs(m_diff[i] - t_diff[j])
            cost_matrix[i][j] = left_diff + right_diff

    return cost_matrix


def get_min_cost_dp(cost_matrix):

    width = cost_matrix.shape[0]
    height = cost_matrix.shape[1]

    min_cost_sum = [[None for i in range(height)] for j in range(width)]
    matchings = [[None for i in range(height)] for j in range(width)]

    # initialize corner
    min_cost_sum[0][0] = cost_matrix[0][0]
    matchings[0][0] = [(0,0)]

    # initialize the first row of min_cost_sum and matchings
    for i in range(1, width):
        min_cost_sum[i][0] = min_cost_sum[i-1][0] + cost_matrix[i][0]
        matchings[i][0] = matchings[i-1][0].copy() + [(i,0)]

    # initialize the first column of min_cost_sum and matchings
    for j in range(1, height):
        min_cost_sum[0][j] = min_cost_sum[0][j-1] + cost_matrix[0][j]
        matchings[0][j] = matchings[0][j-1].copy() + [(0,j)]


    for i in range(1, width):
        for j in range(1, height):
            diagonal_option = min_cost_sum[i-1][j-1]
            up_option = min_cost_sum[i][j-1]
            left_option = min_cost_sum[i-1][j]

            min_index = np.argmin([diagonal_option, up_option, left_option])
            if min_index == 0:
                matchings[i][j] = matchings[i-1][j-1].copy() + [(i,j)]
            elif min_index == 1:
                matchings[i][j] = matchings[i][j-1].copy() + [(i,j)]
            elif min_index == 2:
                matchings[i][j] = matchings[i-1][j].copy() + [(i,j)]

            min_cost_sum[i][j] = min([diagonal_option, up_option, left_option]) + cost_matrix[i][j]


    return matchings[width-1][height-1]


def resolve_duplicate_matchings(cost_matrix, matchings):

    new_matchings = []
    copy_matchings = matchings.copy()

    contiguous_matches = [[copy_matchings.pop(0)]]
    while len(copy_matchings) > 0:

        current_match = copy_matchings[0]
        if current_match[0] == contiguous_matches[-1][-1][0] or current_match[1] == contiguous_matches[-1][-1][1]:
            contiguous_matches[-1].append(copy_matchings.pop(0))
        else:
            contiguous_matches.append([copy_matchings.pop(0)])

    for contiguous_match in contiguous_matches:
        if len(contiguous_match) == 1:
            new_matchings.append(contiguous_match[0])
        else:
            unique_mis = set([mi for mi, ti in contiguous_match])
            unique_tis = set([ti for mi, ti in contiguous_match])
            assert(len(unique_mis) == 1 or len(unique_tis) == 1)

            if len(unique_mis) == 1:
                mi = list(unique_mis)[0]
                unique_tis = list(unique_tis)

                match_costs = [cost_matrix[mi][ti_] for ti_ in unique_tis]

                best_match_ti = unique_tis.pop(np.argmin(match_costs))
                new_matchings.append((mi, best_match_ti))
                for ti in unique_tis:
                    new_matchings.append((None, ti))

            else:
                ti = list(unique_tis)[0]
                unique_mis = list(unique_mis)

                match_costs = [cost_matrix[mi_][ti] for mi_ in unique_mis]

                best_match_mi = unique_mis.pop(np.argmin(match_costs))
                new_matchings.append((best_match_mi, ti))
                for mi in unique_mis:
                    new_matchings.append((mi, None))

    return new_matchings

