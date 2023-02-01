import sys
sys.path.append('..')

import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

from data_gen_constants import *
import rhythm_regression.vector_processing as vp

from pdb import set_trace as bp

def total_num_blocks(len_m, len_t):
   len_shorter_list = int(min(len_m, len_t))
   len_longer_list = int(max(len_m, len_t))
   return sum([(len_shorter_list - block_length + 1) * (len_longer_list - block_length + 1) 
               for block_length in range(1, len_shorter_list + 1)])


def get_all_blocks(len_m, len_t):

    len_smaller_list = min(len_m, len_t)
    len_larger_list = max(len_m, len_t)
    m_is_shorter = (len_m <= len_t)

    block_pairs = [(block1_start, block1_start + block_length, block2_start, block2_start + block_length) if m_is_shorter else \
                   (block2_start, block2_start + block_length, block1_start, block1_start + block_length)
                    for block_length in range(len_smaller_list, MIN_BLOCK_SIZE - 1, -1) \
                    for block1_start in range(0, len_smaller_list - block_length + 1) \
                    for block2_start in range(0, len_larger_list - block_length + 1) \
                    ] 

    return block_pairs


def validate_block_slices(len_m, len_t, m_start, m_end, t_start, t_end):

    if m_start < 0 or m_end < 0 or t_start < 0 or t_end < 0:
        raise ValueError(f'Cannot have negative indices.  Got indices {m_start} {m_end} {t_start} {t_end}')

    if m_start >= len_m or m_end > len_m:
        raise ValueError(f'One m index {m_start}, {m_end} is out of bounds in range [0, {len_m - 1}]. {len_m} {len_t} {t_start} {t_end}') 
    if t_start >= len_t or t_end > len_t:
        raise ValueError(f'One t index {t_start}, {t_end} is out of bounds in range [0, {len_t - 1}]') 

    if m_start >= m_end:
        raise ValueError(f'Starting m index must be less than ending m index.  Got indices {m_start} {m_end}')
    if t_start >= t_end:
        raise ValueError(f'Starting t index must be less than ending t index.  Got indices {t_start} {t_end}')

    if m_end - m_start != t_end - t_start:
        raise ValueError(f'Blocks are unequal sizes.  m block length is {m_end - m_start}.  t block length is {t_end - t_start}')

    if m_end - m_start < MIN_BLOCK_SIZE:
        raise ValueError(f'Block sizes must be at least {MIN_BLOCK_SIZE}.  Got block size {m_end - m_start}')


def true_is_block_match(matchings, m_start, m_end, t_start, t_end):
    """
    m_end and t_end are exclusive
    """

    mi = m_start
    ti = t_start

    while mi < m_end and ti < t_end:
        if (mi, ti) not in matchings:
            return False
        mi += 1
        ti += 1
    
    return True


def compute_block_features(m_block, t_block, m_start, t_start):

    if len(m_block) != len(t_block):
        raise ValueError(f'Blocks are of different lengths.  Got lengths {len(m_block)=} and {len(t_block)=}')

    stats = {}

    m_diff_block = np.diff(m_block) 
    t_diff_block = np.diff(t_block)
    m_diff2_block = np.diff(m_diff_block) 
    t_diff2_block = np.diff(t_diff_block)

    stats['block_length'] = len(m_block)

    timestamp_residuals = np.abs(m_block - t_block)
    stats['sum_timestamp_residual'] = timestamp_residuals.sum()
    stats['mean_timestamp_residual'] = timestamp_residuals.mean()
    stats['max_timestamp_residual'] = timestamp_residuals.max()
    #stats['min_timestamp_residual'] = timestamp_residuals.min()
    #stats['std_timestamp_residual'] = timestamp_residuals.std()

    rhythm_residuals = np.abs(m_diff_block - t_diff_block) if stats['block_length'] > 1 else None 
    stats['sum_rhythm_residual'] = rhythm_residuals.sum() if stats['block_length'] > 1 else 0
    stats['mean_rhythm_residual'] = rhythm_residuals.mean() if stats['block_length'] > 1 else 0
    stats['max_rhythm_residual'] = rhythm_residuals.max() if stats['block_length'] > 1 else 0
    #stats['min_rhythm_residual'] = rhythm_residuals.min() if stats['block_length'] > 1 else 0
    #stats['std_rhythm_residual'] = rhythm_residuals.std() if stats['block_length'] > 1 else 0

    accel_residuals = np.abs(m_diff2_block - t_diff2_block) if stats['block_length'] > 2 else None
    stats['sum_accel_residual'] = accel_residuals.sum() if stats['block_length'] > 2 else 0
    stats['mean_accel_residual'] = accel_residuals.mean() if stats['block_length'] > 2 else 0
    stats['max_accel_residual'] = accel_residuals.max() if stats['block_length'] > 2 else 0
    #stats['min_accel_residual'] = accel_residuals.min() if stats['block_length'] > 2 else 0
    #stats['std_accel_residual'] = accel_residuals.std() if stats['block_length'] > 2 else 0

    stats['index_distance'] = abs(m_start - t_start)
    stats['distance_length_product'] = stats['index_distance'] * stats['block_length']

    return stats 


def get_num_blocks_of_length(length, len_m, len_t):

    len_smaller_list = min(len_m, len_t)
    len_larger_list = max(len_m, len_t)

    if length > len_smaller_list or length < MIN_BLOCK_SIZE:
        return 0

    return (len_smaller_list - length + 1) * (len_larger_list - length + 1)


def sample_blocks(num_blocks, len_m, len_t, upsample_large_blocks = False):

    len_smaller_list = min(len_m, len_t)
    len_larger_list = max(len_m, len_t)
    m_is_shorter = (len_m <= len_t)

    # randomly select block lengths
    block_length_options = list(range(MIN_BLOCK_SIZE, len_smaller_list + 1))
    if upsample_large_blocks:
        block_length_pmf = None # default from numpy is a uniform distribution over block lengths
    else:
        block_length_pmf = np.array([get_num_blocks_of_length(length, len_m, len_t) for length in block_length_options])
        block_length_pmf = block_length_pmf / block_length_pmf.sum()

    block_lengths = np.random.choice(block_length_options, 
                                    size=num_blocks,
                                    replace=True, 
                                    p=block_length_pmf)

    # select a pattern of that block length
    pattern_start_options = [list(range(0, len_smaller_list - block_length + 1)) for block_length in block_lengths]
    pattern_starts = np.array([np.random.choice(options, size=1) for options in pattern_start_options]).flatten()

    # select a test region of that block length
    test_start_options = [list(range(0, len_larger_list - block_length + 1)) for block_length in block_lengths]
    test_starts = np.array([np.random.choice(options, size=1) for options in test_start_options]).flatten()

    # select blocks
    blocks = []
    for i in range(len(block_lengths)):
        block1_start, block1_end = pattern_starts[i], pattern_starts[i] + block_lengths[i]
        block2_start, block2_end = test_starts[i], test_starts[i] + block_lengths[i]

        new_block = (block1_start, block1_end, block2_start, block2_end) if m_is_shorter else \
                    (block2_start, block2_end, block1_start, block1_end)

        validate_block_slices(len_m, len_t, new_block[0], new_block[1], new_block[2], new_block[3])
        blocks.append(new_block)

    return blocks


def generate_block_stats(examples):

    rows = []
    headers = None

    for example in tqdm(examples):

        vp.center_midi_on_transients(example['m'], example['t'])
        assert(abs(example['m'].mean() - example['t'].mean()) < 0.00001)
        assert(example['m'].mean() > 0)

        blocks = sample_blocks(BLOCK_SAMPLE_SIZE, len(example['m']), len(example['t']), upsample_large_blocks=False)
        
        for m_start, m_end, t_start, t_end in blocks:

                validate_block_slices(len(example['m']), len(example['t']), m_start, m_end, t_start, t_end)
                stats = compute_block_features(example['m'][m_start : m_end], example['t'][t_start : t_end], m_start, t_start)

                new_row = {'example_id': example['id'],
                        'm_start': m_start, 
                        'm_end': m_end, 
                        't_start': t_start, 
                        't_end': t_end, 
                        'true_is_block_match': int(true_is_block_match(example['matchings'], 
                                                                    m_start, 
                                                                    m_end, 
                                                                    t_start, 
                                                                    t_end)),
                        } | \
                        stats

                if headers is None:
                    headers = new_row.keys()

                rows.append(list(new_row.values()))

    return pd.DataFrame(rows, columns=headers)
    

def load_example_data():

    with open(os.path.join(DATA_DIR, TRAIN_EXAMPLES_FILENAME), "rb" ) as f:
        train_examples = np.array(pickle.load(f), dtype=object)
        print(f'Number of training examples: {len(train_examples)}')
    with open(os.path.join(DATA_DIR, VAL_EXAMPLES_FILENAME), "rb" ) as f:
        val_examples = np.array(pickle.load(f), dtype=object)
        print(f'Number of validation examples: {len(val_examples)}')
    with open(os.path.join(DATA_DIR, TEST_EXAMPLES_FILENAME), "rb" ) as f:
        test_examples = np.array(pickle.load(f), dtype=object)
        print(f'Number of testing examples: {len(test_examples)}')

    return train_examples, val_examples, test_examples


def main():

    np.random.seed(RANDOM_SEED)

    train_examples, val_examples, test_examples = load_example_data()

    train_block_stats = generate_block_stats(train_examples)
    train_block_stats.to_csv(os.path.join(DATA_DIR, TRAIN_STATS_FILENAME))

    val_block_stats = generate_block_stats(val_examples)
    val_block_stats.to_csv(os.path.join(DATA_DIR, VAL_STATS_FILENAME))

    test_block_stats = generate_block_stats(test_examples)
    test_block_stats.to_csv(os.path.join(DATA_DIR, TEST_STATS_FILENAME))


if __name__ == '__main__':
    main()