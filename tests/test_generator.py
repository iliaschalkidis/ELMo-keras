import os
import numpy as np
from data import DATA_SET_DIR
from elmo.lm_generator import LMDataGenerator

parameters = {
    'valid_dataset': 'wikitext-2/wiki.valid.tokens',
    'vocab': 'wikitext-2/wiki.vocab',
}

val_generator = LMDataGenerator(os.path.join(DATA_SET_DIR, parameters['valid_dataset']),
                                os.path.join(DATA_SET_DIR, parameters['vocab']),
                                sentence_maxlen=50,
                                token_maxlen=50,
                                batch_size=1,
                                token_encoding='word')

for i in range(0, 100):
    inputs, _ = val_generator[i]
    print('\n{:>15} {:>15} {:>15}'.format('TOKEN', 'NEXT_TOKEN', 'PREV_TOKEN'))
    print('='*50)
    for token, next_token, previous_token in zip(inputs[0][0], inputs[1][0], inputs[2][0]):
        if token == next_token[0] == previous_token[0] == 0:
            break
        print('{:>15} {:>15} {:>15}'.format(token, next_token[0], previous_token[0]))


val_generator = LMDataGenerator(os.path.join(DATA_SET_DIR, parameters['valid_dataset']),
                                os.path.join(DATA_SET_DIR, parameters['vocab']),
                                sentence_maxlen=50,
                                token_maxlen=50,
                                batch_size=1,
                                token_encoding='char')

for i in range(0, 100):
    inputs, _ = val_generator[i]
    print('\n{:>40} {:>40} {:>40}'.format('TOKEN', 'NEXT_TOKEN', 'PREV_TOKEN'))
    print('='*150)
    for token, next_token, previous_token in zip(inputs[0][0], inputs[1][0], inputs[2][0]):
        if previous_token[0] == next_token[0] == 0:
            break
        print('{:>40} {:>40} {:>40}'.format(np.array2string(token[:5], separator=','), next_token[0], previous_token[0]))
