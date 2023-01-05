import sys
# Allows loading of rhythm_regression_module
sys.path.append('..')

import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from rhythm_regression.vector_processing import validate_matching

OUTPUT_DIR = '../data/matchings/'

TEMPO = 120

QUARTER_NOTE = 60 / TEMPO
HALF_NOTE = QUARTER_NOTE * 2
WHOLE_NOTE = QUARTER_NOTE * 4
EIGHTH_NOTE = QUARTER_NOTE / 2
SIXTEENTH_NOTE = QUARTER_NOTE / 4

DOTTED_EIGHTH_NOTE = SIXTEENTH_NOTE * 3
DOTTED_QUARTER_NOTE = EIGHTH_NOTE * 3
DOTTED_HALF_NOTE = QUARTER_NOTE * 3

TRIPLET = QUARTER_NOTE / 3
SEXTUPLET = QUARTER_NOTE / 6
QUARTER_NOTE_TRIPLET = TRIPLET * 2

RHYTHM_BANK = [
    QUARTER_NOTE,
    HALF_NOTE,
    WHOLE_NOTE,
    EIGHTH_NOTE,
    SIXTEENTH_NOTE, 
    DOTTED_EIGHTH_NOTE,
    DOTTED_QUARTER_NOTE,
    DOTTED_HALF_NOTE,
    TRIPLET,
    SEXTUPLET,
    QUARTER_NOTE_TRIPLET,
]
RHYTHM_BANK_PMF = [
    0.1, #QUARTER_NOTE,
    0.025, #HALF_NOTE,
    0.025, #WHOLE_NOTE,
    0.15, #EIGHTH_NOTE,
    0.25, #SIXTEENTH_NOTE, 
    0.1, #DOTTED_EIGHTH_NOTE,
    0.1, #DOTTED_QUARTER_NOTE,
    0.05, #DOTTED_HALF_NOTE,
    0.1, #TRIPLET,
    0.05, #SEXTUPLET,
    0.05, #QUARTER_NOTE_TRIPLET,
]
AVERAGE_RHYTHM = sum([rhythm*p for rhythm, p in zip(RHYTHM_BANK,RHYTHM_BANK_PMF)])

MONOTONICITY_CRUCNH_FACTOR = 0.9
MIN_INSERTION_TIMESTAMP = -1.5

REPITITIONS_PER_CONFIG = 10

RANDOM_SEED = 7202001  # Release Date of Spirited Away


def generate_true_matching(m, t):
    matching = []
    for i in range(len(m)):
        for j in range(len(t)):
            if m[i] == t[j]:
                matching.append((i, j))

    # deletion of notes
    for i in range(len(m)):
        if i not in [mi for mi, ti in matching]:
            matching.append((i, None))

    # insertion of notes
    for j in range(len(t)):
        if j not in [ti for mi, ti in matching]:
            matching.append((None, j))

    matching.sort(key=lambda tup: tup[0] if tup[0] is not None else tup[1])

    return matching


def delete_notes(t, deletion_rate):
    deletion_indices = np.argwhere(np.random.random(len(t)) < deletion_rate).flatten()
    t = np.delete(t, deletion_indices)
    return t


def insert_notes(t, insertion_rate):
    insertion_indices = np.argwhere(np.random.random(len(t)) < insertion_rate).flatten()
    insertion_indices += np.arange(len(insertion_indices))
    for i in insertion_indices:
        if i != 0:
            insertion_interval = (max(0, i-1), i)
            new_note = np.random.uniform(t[insertion_interval[0]], t[insertion_interval[1]])
        else:
            new_note = np.random.uniform(MIN_INSERTION_TIMESTAMP, 0)
        t = np.insert(t, i, new_note)

    return t


def augment_space(t, space_augmentation_rate):
    space_augmentation_indices = np.argwhere(np.random.random(len(t)) < space_augmentation_rate).flatten()
    augmentation_amounts = np.random.choice(RHYTHM_BANK, p=RHYTHM_BANK_PMF, size=len(space_augmentation_indices))

    for i in range(len(space_augmentation_indices)):
        t[space_augmentation_indices[i]:] += augmentation_amounts[i]

    return t


def reduce_space(t, space_reduction_rate):
    space_reduction_indices = np.argwhere(np.random.random(len(t)) < space_reduction_rate).flatten()
    reduction_amounts = np.random.choice(RHYTHM_BANK, p=RHYTHM_BANK_PMF, size=len(space_reduction_indices))

    for i in range(len(space_reduction_indices)):
        if space_reduction_indices[i] != 0:
            maximum_reduction = t[space_reduction_indices[i]] - t[space_reduction_indices[i]-1]
            if reduction_amounts[i] >= maximum_reduction:
                # in order to keep t strictly increasing, we cannot subtract 
                # equal to or more space than was originally there.
                reduction_amounts[i] = MONOTONICITY_CRUCNH_FACTOR * maximum_reduction

        t[space_reduction_indices[i]:] -= reduction_amounts[i]

    return t


def add_noise(t, std):
    for i in range(len(t)):
        
        noise = np.random.normal(0, std)

        if i > 0:
            min_noise = t[i-1] - t[i] # signed distance
            if noise < min_noise:
#                print('low clipping noise')
                noise = MONOTONICITY_CRUCNH_FACTOR * min_noise

        if i < len(t) - 1:
            max_noise = t[i+1] - t[i]
            if noise > max_noise:
#                print('high clipping noise')
                noise = MONOTONICITY_CRUCNH_FACTOR * max_noise

        t[i] += noise

    return t
            

def assert_monotonicly_increasing(a):
    assert(np.all(a[1:] >= a[:-1]))


def generate_example(id, len_m, deletion_rate, insertion_rate, space_augmentation_rate, space_reduction_rate):
    m_diff = np.random.choice(RHYTHM_BANK, p=RHYTHM_BANK_PMF, size=len_m-1)
    m = np.cumsum(m_diff)
    m = np.insert(m, 0, 0)

    t = np.copy(m)

    t = delete_notes(t, deletion_rate)
    assert_monotonicly_increasing(t)

    t = insert_notes(t, insertion_rate)
    assert_monotonicly_increasing(t)

    matchings = generate_true_matching(m, t)
    validate_matching(m, t, matchings)

    t = augment_space(t, space_augmentation_rate)
    assert_monotonicly_increasing(t)

    t = reduce_space(t, space_reduction_rate)
    assert_monotonicly_increasing(t)

    t = add_noise(t, SIXTEENTH_NOTE / 2)
    assert_monotonicly_increasing(t)

    return {
            'id': id,
            'm': m, 
            't': t, 
            'matchings': matchings,
            }


def main():

    np.random.seed(RANDOM_SEED)

    param_options = {
        'len_m':                  [50, 100, 200, 300],
        'deletion_rate':            [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5],
        'insertion_rate':           [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5],
        'space_augmentation_rate':  [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
        'space_reduction_rate':     [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
        'observation_num': list(range(REPITITIONS_PER_CONFIG)),
        }

    """
    param_options = {
        'len_m':                  [50],
        'deletion_rate':            [0, 0.01],
        'insertion_rate':           [0, 0.01],
        'space_augmentation_rate':  [0, 0.001],
        'space_reduction_rate':     [0, 0.001],
        'observation_num': list(range(REPITITIONS_PER_CONFIG)),
        }
    """

    grid = ParameterGrid(param_options)

    example_set = [None] * len(grid)

    for i in tqdm(range(len(grid))):
        params = grid[i]
        example_set[i] = generate_example(
            i,
            params['len_m'], 
            params['deletion_rate'], 
            params['insertion_rate'], 
            params['space_augmentation_rate'], 
            params['space_reduction_rate'])

    with open(os.path.join(OUTPUT_DIR, 'example_set.pickle'), 'wb' ) as f:
        pickle.dump(example_set, f)


    parameter_table = pd.DataFrame(grid)
    parameter_table.drop('observation_num', axis=1, inplace=True) 
    parameter_table.to_csv(os.path.join(OUTPUT_DIR, 'generated_data_params.csv'))



if __name__ == '__main__':
    main()