# ELMo-keras
Re-implementation of ELMo in Keras based on the tensorflow implementation presented by Allen NLP (https://github.com/allenai/bilm-tf), based on Peters et al. article in NAACL 2018 (https://arxiv.org/abs/1802.05365):

_Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee and Luke Zettlemoyer. 2018. Deep contextualized word representations_

Notice: The project includes WikiText-2 datasets for experimentation as published in (https://einstein.ai/research/blog/the-wikitext-long-term-dependency-language-modeling-dataset), presented in Merity et al. 2016 (https://arxiv.org/abs/1609.07843):

_Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer Sentinel Mixture Models_

## Why in the heck did you do that?

- This was the easiest way to understand ELMo deeply, find its pros and cons and also consider improvements (e.g., make it more computational efficient). 
- I also consider Keras as the most user-friendly and industry-ready library to work with.
- Now we are also able to integrate ELMo for practical use at Cognitiv+, where we rely on Keras for our NLP engine.
- It was really fun! This took me more than a month, in which period I had to learn many things and vastly improve my understading and skills around Keras and Tensorflow, so be kind.

## How to use it?

```
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
    'clip_value': 5,
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

# Load ELMo encoder
elmo_model.load_elmo_encoder()

# Get ELMo embeddings to feed as inputs for downstream tasks
elmo_embeddings = elmo_model.get_outputs(test_generator, output_type='word', state='mean')

# BUILD & TRAIN NEW KERAS MODEL FOR DOWNSTREAM TASK (E.G., TEXT CLASSIFICATION)

```

## What is missing?

- Turn sampled softmax into full softmax dynamically in evaluation mode (TODO) ([Read comment](https://github.com/iliaschalkidis/ELMo-keras/commit/35fa4f9b3245a9c1078d4c7975064b19bd9742f4#commitcomment-31314484))
- More testing (TODO)
- Options to build a unidirectional LM (TODO)
- Proof-reading, you're all welcome!

## Credits for proof-reading and reporting so far...

[@seolhokim](https://github.com/seolhokim)
[@geneva0901](https://github.com/geneva0901)
[@masepehr](https://github.com/masepehr)
[@dilaratorunoglu](https://github.com/dilaratorunoglu)
[@Adherer](https://github.com/Adherer)
