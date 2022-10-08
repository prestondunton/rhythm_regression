from rhythms import *

import numpy as np
import os
from pymusicxml import *


RANDOM_SEED = 12271963 # Start of the Los Angeles Surf Fair, 1963
SNARE_DRUM_PITCH = 'c5'
NUM_PARTS = 4
NUM_MEASURES = 32
# Average beats between mutation is ~ (1-mut_rate) / mut_rate
MUTATION_RATE = 0.11
RHYTHM_BANK = [quarter_rest,                                # 0 note per beat
              one, e, aand, a,                              # 1 note per beat
              one_e, one_and, one_a, e_and, e_a, and_a,     # 2 notes per beat
              one_e_and, one_e_a, one_and_a, e_and_a,       # 3 notes per beat
              one_e_and_a]                                  # 4 notes per beat


def main():

    print('Generating random score!')

    np.random.seed(RANDOM_SEED)

    measures = [[] for _ in range(NUM_PARTS)]

    for i in range(NUM_MEASURES):
        for j in range(NUM_PARTS):
            measures[j].append(Measure(time_signature=(4, 4) if i == 0 else None))
        for _ in range(4):
            rhythm_choice = np.random.choice(RHYTHM_BANK)(SNARE_DRUM_PITCH)
            measures[0][-1].append(rhythm_choice)
            for j in range(1, NUM_PARTS):
                measures[j][-1].append(rhythm_choice if np.random.random() > MUTATION_RATE else np.random.choice(RHYTHM_BANK)(SNARE_DRUM_PITCH))
        
    
    score = Score(title=f'Random Mutations (Seed: {RANDOM_SEED}, Rate: {MUTATION_RATE})', composer=f'{os.getlogin()} via Python')
    parts = [Part('Snare Drum') for _ in range(NUM_PARTS)]
    for i in range(NUM_PARTS):
        measures[i].append(release_measure(SNARE_DRUM_PITCH))
        parts[i].extend(measures[i])
        score.append(parts[i])

    
    score.export_to_file('Random_Mutations.musicxml')

    print('Finished generating score!')


if __name__ == '__main__':
    main()