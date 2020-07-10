import warnings
warnings.filterwarnings('ignore')
import os
import keras.backend as K

from data import DATA_SET_DIR
from elmo.lm_generator import LMDataGenerator
from elmo.model import ELMo

parameters = {
    'multi_processing': True,
    'n_threads': os.cpu_count(),
    'cuDNN': True if len(K.tensorflow_backend._get_available_gpus()) else False,
    'train_dataset': 'txt/advertiser_id.train.tokens',
    'valid_dataset': 'txt/advertiser_id.valid.tokens',
    'test_dataset': 'txt/advertiser_id.test.tokens',
    'vocab': 'txt/advertiser_id.vocab',
    'vocab_size': 54837,
    'num_sampled': 1000,
    'charset_size': 262,
    'sentence_maxlen': 100,
    'token_maxlen': 50,
    'token_encoding': 'word',
    'epochs': 10,
    'patience': 2,
    'batch_size': 4,
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
print('compile')
elmo_model = ELMo(parameters)
elmo_model.compile_elmo(print_summary=True)

# Train ELMo
print('train')
elmo_model.train(train_data=train_generator, valid_data=val_generator)

# Persist ELMo Bidirectional Language Model in disk
print('save')
elmo_model.save(sampled_softmax=False)

# Evaluate Bidirectional Language Model
print('evaluate')
elmo_model.evaluate(test_generator)

# Build ELMo meta-model to deploy for production and persist in disk
print('??')
elmo_model.wrap_multi_elmo_encoder(print_summary=True, save=True)

# Load ELMo encoder
print('load')
elmo_model.load_elmo_encoder()

# Get ELMo embeddings to feed as inputs for downstream tasks
elmo_embeddings = elmo_model.get_outputs(test_generator, output_type='word', state='mean')

# BUILD & TRAIN NEW KERAS MODEL FOR DOWNSTREAM TASK (E.G., TEXT CLASSIFICATION)

import joblib
joblib.dump(elmo_embeddings, 'advertiser_id_400_elmo.jl.z')
