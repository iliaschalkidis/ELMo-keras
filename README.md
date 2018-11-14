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

from data import DATA_SET_DIR, MODELS_DIR
from elmo.lm_generator import LMDataGenerator
from elmo.model import ELMo

parameters = {
    'multi_processing': False,
    'n_threads': 4,
    'cuDNN': True if len(K.tensorflow_backend._get_available_gpus()) else False,
    'train_dataset': 'wikitext-2/wiki.train.tokens',
    'valid_dataset': 'wikitext-2/wiki.valid.tokens',
    'vocab': 'wikitext-2/wiki.vocab',
    'vocab_size': 28915,
    'num_sampled': 1000,
    'charset_size': 262,
    'sentence_maxlen': 100,
    'token_maxlen': 50,
    'token_encoding': 'word',
    'epochs': 1,
    'batch_size': 32,
    'clip_value': 5,
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
    'weight_tying': True
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

# Compile ELMo
elmo_model = ELMo(parameters)
elmo_model.compile_elmo()
elmo_model.summary()

# Train ELMo
elmo_model.train(train_data=train_generator, valid_data=val_generator)

# Build ELMo meta-model to deploy for production and persist in disk
elmo_model.wrap_multi_elmo_encoder()
elmo_model.save(os.path.join(MODELS_DIR, 'ELMo_Encoder.hd5'))
```

## What is missing?

- Full softmax projection layer in evaluation mode (TODO)
- More testing (TODO)
- Options to build a unidirectional LM
- Proof-reading, you're all welcome!
