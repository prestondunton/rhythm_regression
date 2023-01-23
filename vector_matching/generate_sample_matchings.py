import sys
# Allows loading of rhythm_regression_module
sys.path.append('..')

import math
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import ParameterGrid, train_test_split
from tqdm import tqdm

from rhythm_regression.vector_processing import validate_matching
from data_gen_constants import *

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

    m_diff = np.diff(m)
    t_diff = np.diff(t)
    m_diff2 = np.diff(m_diff)
    t_diff2 = np.diff(t_diff)

    return {
            'id': id,
            'm': m, 
            't': t, 
            'm_diff': m_diff,
            't_diff': t_diff,
            'm_diff2': m_diff2,
            't_diff2': t_diff2,
            'matchings': matchings,
            }


def train_val_test_split(examples, parameter_table):
    if len(examples) != len(parameter_table):
        raise ValueError(f'Length of example set must be the same as the length of the parameter table.  Got lengths {len(examples)} {len(parameter_table)}')

    train_examples, test_examples, train_params, test_params = train_test_split(examples, parameter_table, 
                                                                                train_size=TRAIN_SPLIT + VAL_SPLIT, random_state=RANDOM_SEED, shuffle=False)
    train_examples, val_examples, train_params, val_params = train_test_split(train_examples, train_params, 
                                                                                train_size=TRAIN_SPLIT / (TRAIN_SPLIT + VAL_SPLIT), random_state=RANDOM_SEED, shuffle=False)

    return train_examples, val_examples, test_examples, train_params, val_params, test_params


def save_data(train_examples, val_examples, test_examples, train_params, val_params, test_params):

    with open(os.path.join(DATA_DIR, TRAIN_EXAMPLES_FILENAME), 'wb') as f:
        pickle.dump(train_examples, f)
    with open(os.path.join(DATA_DIR, VAL_EXAMPLES_FILENAME), 'wb') as f:
        pickle.dump(val_examples, f)
    with open(os.path.join(DATA_DIR, TEST_EXAMPLES_FILENAME), 'wb') as f:
        pickle.dump(test_examples, f)

    train_params.to_csv(os.path.join(DATA_DIR, TRAIN_PARAMS_FILENAME))
    val_params.to_csv(os.path.join(DATA_DIR, VAL_PARAMS_FILENAME))
    test_params.to_csv(os.path.join(DATA_DIR, TEST_PARAMS_FILENAME))


def main():

    np.random.seed(RANDOM_SEED)

    population_grid = np.array(ParameterGrid(POPULATION_CONFIGS))
    shuffle_permutation = np.random.permutation(len(population_grid))
    shuffled_population_grid = population_grid[shuffle_permutation]
    sample_grid = shuffled_population_grid[0:EXAMPLE_SAMPLE_SIZE]

    example_set = [None] * len(sample_grid)

    for i in tqdm(range(len(sample_grid))):
        params = sample_grid[i]
        example_set[i] = generate_example(
            i,
            params['len_m'], 
            params['deletion_rate'], 
            params['insertion_rate'], 
            params['space_augmentation_rate'], 
            params['space_reduction_rate'])

    parameter_table = pd.DataFrame([dict for dict in sample_grid])
    parameter_table.drop('observation_num', axis=1, inplace=True) 
    parameter_table.index.name = 'example_id'

    save_data(*train_val_test_split(example_set, parameter_table))

    
if __name__ == '__main__':
    main()