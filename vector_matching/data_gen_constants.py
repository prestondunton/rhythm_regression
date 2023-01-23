# #########################################################################################################
# MISC AND DIRECTORY STRUCTURE

RANDOM_SEED = 7202001  # Release Date of Spirited Away

TRAIN_SPLIT = 0.5
VAL_SPLIT = 0.3
TEST_SPLIT = 0.2

TRAIN_EXAMPLES_FILENAME = 'train_examples.pickle'
VAL_EXAMPLES_FILENAME = 'val_examples.pickle'
TEST_EXAMPLES_FILENAME = 'test_examples.pickle'

TRAIN_PARAMS_FILENAME = 'train_params.csv'
VAL_PARAMS_FILENAME = 'val_params.csv'
TEST_PARAMS_FILENAME = 'test_params.csv'

TRAIN_STATS_FILENAME = 'train_stats.csv'
VAL_STATS_FILENAME = 'val_stats.csv'
TEST_STATS_FILENAME = 'test_stats.csv'

DATA_DIR = '../data/matchings/'


# #########################################################################################################
# DATA QUANTITY

EXAMPLE_SAMPLE_SIZE = 10000
BLOCK_SAMPLE_SIZE = 500


# #########################################################################################################
# EXAMPLE GENERATION

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

POPULATION_CONFIGS = {
    'len_m':                  [50, 100, 200, 300],
    'deletion_rate':            [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5],
    'insertion_rate':           [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5],
    'space_augmentation_rate':  [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
    'space_reduction_rate':     [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
    'observation_num': list(range(REPITITIONS_PER_CONFIG)),
    }

"""
POPULATION_CONFIGS = {
    'len_m':                  [50],
    'deletion_rate':            [0, 0.01],
    'insertion_rate':           [0, 0.01],
    'space_augmentation_rate':  [0, 0.001],
    'space_reduction_rate':     [0, 0.001],
    'observation_num': list(range(REPITITIONS_PER_CONFIG)),
    }
"""

# #########################################################################################################
# BLOCK STATS GENERATION

MIN_BLOCK_SIZE = 1




"""
Hyperparameters: CURVE_CONSTANT, NN hyperparams, PR tradeoff coefficient

I train the 
I need to optimize all of the hyperparameters at once (fed by the same data: validation_examples.pickle, validation_block_stats.csv)
I will then evaluate the algorithm (p99 time, p01 jaccard) with the test data: test_examples.pickle

"""