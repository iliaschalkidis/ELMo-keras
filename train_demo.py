import os
import keras.backend as K

from data import DATA_SET_DIR
from elmo.lm_generator import LMDataGenerator
from elmo.model import ELMo

parameters = {
    'multi_processing': False,
    'n_threads': 4,
    'cuDNN': True if len(K.tensorflow_backend._get_available_gpus()) else False,
    'train_dataset': 'wikitext-2/wiki.train.tokens',
    'valid_dataset': 'wikitext-2/wiki.valid.tokens',
    'test_dataset': 'wikitext-2/wiki.test.tokens',
    'vocab': 'wikitext-2/wiki.vocab',
    'vocab_size': 28914,
    'num_sampled': 1000,
    'charset_size': 262,
    'sentence_maxlen': 100,
    'token_maxlen': 50,
    'token_encoding': 'word',
    'epochs': 10,
    'patience': 2,
    'batch_size': 1,
    'clip_value': 1,
    'cell_clip': 5,
    'proj_clip': 5,
    'lr': 0.2,
    'shuffle': True,
    'n_lstm_layers': 2,
    'n_highway_layers': 2,
    'cnn_filters': [[1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 512]
                    ],
    'lstm_units_size': 400,
    'hidden_units_size': 200,
    'char_embedding_size': 16,
    'dropout_rate': 0.1,
    'word_dropout_rate': 0.05,
    'weight_tying': True,
}

# Set-up Generators
train_generator = LMDataGenerator(os.path.join(DATA_SET_DIR, parameters['train_dataset']),
                                  os.path.join(DATA_SET_DIR, parameters['vocab']),
                                  sentence_maxlen=parameters['sentence_maxlen'],
                                  token_maxlen=parameters['token_maxlen'],
                                  batch_size=parameters['batch_size'],
                                  shuffle=parameters['shuffle'],
                                  token_encoding=parameters['token_encoding'])

val_generator = LMDataGenerator(os.path.join(DATA_SET_DIR, parameters['valid_dataset']),
                                os.path.join(DATA_SET_DIR, parameters['vocab']),
                                sentence_maxlen=parameters['sentence_maxlen'],
                                token_maxlen=parameters['token_maxlen'],
                                batch_size=parameters['batch_size'],
                                shuffle=parameters['shuffle'],
                                token_encoding=parameters['token_encoding'])

test_generator = LMDataGenerator(os.path.join(DATA_SET_DIR, parameters['test_dataset']),
                                os.path.join(DATA_SET_DIR, parameters['vocab']),
                                sentence_maxlen=parameters['sentence_maxlen'],
                                token_maxlen=parameters['token_maxlen'],
                                batch_size=parameters['batch_size'],
                                shuffle=parameters['shuffle'],
                                token_encoding=parameters['token_encoding'])

# Compile ELMo
elmo_model = ELMo(parameters)
elmo_model.compile_elmo(print_summary=True)

# Train ELMo
elmo_model.train(train_data=train_generator, valid_data=val_generator)

# Persist ELMo Bidirectional Language Model in disk
elmo_model.save(sampled_softmax=False)

# Evaluate Bidirectional Language Model
elmo_model.evaluate(test_generator)

# Build ELMo meta-model to deploy for production and persist in disk
elmo_model.wrap_multi_elmo_encoder(print_summary=True, save=True)
