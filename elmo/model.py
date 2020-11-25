import os
import time

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Input, SpatialDropout1D
from tensorflow.keras.layers import LSTM, Activation
from tensorflow.keras.layers import Lambda, Embedding, Conv2D, GlobalMaxPool1D
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.utils import to_categorical

from data import MODELS_DIR
from .custom_layers import TimestepDropout, Camouflage, Highway, SampledSoftmax


class ELMo(object):
    def __init__(self, parameters):
        self._model = None
        self._elmo_model = None
        self.parameters = parameters
        self.compile_elmo()

    def __del__(self):
        K.clear_session()
        del self._model

    def char_level_token_encoder(self):
        charset_size = self.parameters['charset_size']
        char_embedding_size = self.parameters['char_embedding_size']
        token_embedding_size = self.parameters['hidden_units_size']
        n_highway_layers = self.parameters['n_highway_layers']
        filters = self.parameters['cnn_filters']
        token_maxlen = self.parameters['token_maxlen']

        # Input Layer, word characters (samples, words, character_indices)
        inputs = Input(shape=(None, token_maxlen,), dtype='int32')
        # Embed characters (samples, words, characters, character embedding)
        embeds = Embedding(input_dim=charset_size, output_dim=char_embedding_size)(inputs)
        token_embeds = []
        # Apply multi-filter 2D convolutions + 1D MaxPooling + tanh
        for (window_size, filters_size) in filters:
            convs = Conv2D(filters=filters_size, kernel_size=[window_size, char_embedding_size], strides=(1, 1),
                           padding="same")(embeds)
            convs = TimeDistributed(GlobalMaxPool1D())(convs)
            convs = Activation('tanh')(convs)
            convs = Camouflage(mask_value=0)(inputs=[convs, inputs])
            token_embeds.append(convs)
        token_embeds = concatenate(token_embeds)
        # Apply highways networks
        for i in range(n_highway_layers):
            token_embeds = TimeDistributed(Highway())(token_embeds)
            token_embeds = Camouflage(mask_value=0)(inputs=[token_embeds, inputs])
        # Project to token embedding dimensionality
        token_embeds = TimeDistributed(Dense(units=token_embedding_size, activation='linear'))(token_embeds)
        token_embeds = Camouflage(mask_value=0)(inputs=[token_embeds, inputs])

        token_encoder = Model(inputs=inputs, outputs=token_embeds, name='token_encoding')
        return token_encoder

    def compile_elmo(self, print_summary=False):
        """
        Compiles a Language Model RNN based on the given parameters
        """

        if self.parameters['token_encoding'] == 'word':
            # Train word embeddings from scratch
            word_inputs = Input(shape=(None,), name='word_indices', dtype='int32')
            embeddings = Embedding(self.parameters['vocab_size'], self.parameters['hidden_units_size'], trainable=True, name='token_encoding')
            inputs = embeddings(word_inputs)

            # Token embeddings for Input
            drop_inputs = SpatialDropout1D(self.parameters['dropout_rate'])(inputs)
            lstm_inputs = TimestepDropout(self.parameters['word_dropout_rate'])(drop_inputs)

            # Pass outputs as inputs to apply sampled softmax
            next_ids = Input(shape=(None, 1), name='next_ids', dtype='float32')
            previous_ids = Input(shape=(None, 1), name='previous_ids', dtype='float32')
        elif self.parameters['token_encoding'] == 'char':
            # Train character-level representation
            word_inputs = Input(shape=(None, self.parameters['token_maxlen'],), dtype='int32', name='char_indices')
            inputs = self.char_level_token_encoder()(word_inputs)

            # Token embeddings for Input
            drop_inputs = SpatialDropout1D(self.parameters['dropout_rate'])(inputs)
            lstm_inputs = TimestepDropout(self.parameters['word_dropout_rate'])(drop_inputs)

            # Pass outputs as inputs to apply sampled softmax
            next_ids = Input(shape=(None, 1), name='next_ids', dtype='float32')
            previous_ids = Input(shape=(None, 1), name='previous_ids', dtype='float32')

        # Reversed input for backward LSTMs
        re_lstm_inputs = Lambda(function=ELMo.reverse)(lstm_inputs)
        mask = Lambda(function=ELMo.reverse)(drop_inputs)

        # Forward LSTMs
        for i in range(self.parameters['n_lstm_layers']):
            lstm = LSTM(units=self.parameters['lstm_units_size'], return_sequences=True, activation="tanh",
                        recurrent_activation='sigmoid',
                        kernel_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                     self.parameters['cell_clip']),
                        recurrent_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                        self.parameters['cell_clip'])
                        )(lstm_inputs)
            lstm = Camouflage(mask_value=0)(inputs=[lstm, drop_inputs])
            # Projection to hidden_units_size
            proj = TimeDistributed(Dense(self.parameters['hidden_units_size'], activation='linear',
                                         kernel_constraint=MinMaxNorm(-1 * self.parameters['proj_clip'],
                                                                      self.parameters['proj_clip'])
                                         ))(lstm)
            # Merge Bi-LSTMs feature vectors with the previous ones
            lstm_inputs = add([proj, lstm_inputs], name='f_block_{}'.format(i + 1))
            # Apply variational drop-out between BI-LSTM layers
            lstm_inputs = SpatialDropout1D(self.parameters['dropout_rate'])(lstm_inputs)

        # Backward LSTMs
        for i in range(self.parameters['n_lstm_layers']):
            re_lstm = LSTM(units=self.parameters['lstm_units_size'], return_sequences=True, activation='tanh',
                           recurrent_activation='sigmoid',
                           kernel_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                        self.parameters['cell_clip']),
                           recurrent_constraint=MinMaxNorm(-1 * self.parameters['cell_clip'],
                                                           self.parameters['cell_clip'])
                           )(re_lstm_inputs)
            re_lstm = Camouflage(mask_value=0)(inputs=[re_lstm, mask])
            # Projection to hidden_units_size
            re_proj = TimeDistributed(Dense(self.parameters['hidden_units_size'], activation='linear',
                                            kernel_constraint=MinMaxNorm(-1 * self.parameters['proj_clip'],
                                                                         self.parameters['proj_clip'])
                                            ))(re_lstm)
            # Merge Bi-LSTMs feature vectors with the previous ones
            re_lstm_inputs = add([re_proj, re_lstm_inputs], name='b_block_{}'.format(i + 1))
            # Apply variational drop-out between BI-LSTM layers
            re_lstm_inputs = SpatialDropout1D(self.parameters['dropout_rate'])(re_lstm_inputs)

        # Reverse backward LSTMs' outputs = Make it forward again
        re_lstm_inputs = Lambda(function=ELMo.reverse, name="reverse")(re_lstm_inputs)

        # Project to Vocabulary with Sampled Softmax
        sampled_softmax = SampledSoftmax(num_classes=self.parameters['vocab_size'],
                                         num_sampled=int(self.parameters['num_sampled']),
                                         tied_to=embeddings if self.parameters['weight_tying']
                                         and self.parameters['token_encoding'] == 'word' else None)
        outputs = sampled_softmax([lstm_inputs, next_ids])
        re_outputs = sampled_softmax([re_lstm_inputs, previous_ids])

        self._model = Model(inputs=[word_inputs, next_ids, previous_ids],
                            outputs=[outputs, re_outputs])
        self._model.compile(optimizer=Adagrad(lr=self.parameters['lr'], clipvalue=self.parameters['clip_value']),
                            loss=None)
        if print_summary:
            self._model.summary()

    def train(self, train_data, valid_data):

        # Add callbacks (early stopping, model checkpoint)
        weights_file = os.path.join(MODELS_DIR, "elmo_best_weights.hdf5")
        save_best_model = ModelCheckpoint(filepath=weights_file, monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(patience=self.parameters['patience'], restore_best_weights=True)

        t_start = time.time()

        # Fit Model
        self._model.fit_generator(train_data,
                                  validation_data=valid_data,
                                  epochs=self.parameters['epochs'],
                                  workers=self.parameters['n_threads']
                                  if self.parameters['n_threads'] else os.cpu_count(),
                                  use_multiprocessing=True
                                  if self.parameters['multi_processing'] else False,
                                  callbacks=[save_best_model])

        print('Training took {0} sec'.format(str(time.time() - t_start)))

    def evaluate(self, test_data):

        def unpad(x, y_true, y_pred):
            y_true_unpad = []
            y_pred_unpad = []
            for i, x_i in enumerate(x):
                for j, x_ij in enumerate(x_i):
                    if x_ij == 0:
                        y_true_unpad.append(y_true[i][:j])
                        y_pred_unpad.append(y_pred[i][:j])
                        break
            return np.asarray(y_true_unpad), np.asarray(y_pred_unpad)

        # Generate samples
        x, y_true_forward, y_true_backward = [], [], []
        for i in range(len(test_data)):
            test_batch = test_data[i][0]
            x.extend(test_batch[0])
            y_true_forward.extend(test_batch[1])
            y_true_backward.extend(test_batch[2])
        x = np.asarray(x)
        y_true_forward = np.asarray(y_true_forward)
        y_true_backward = np.asarray(y_true_backward)

        # Predict outputs
        y_pred_forward, y_pred_backward = self._model.predict([x, y_true_forward, y_true_backward])

        # Unpad sequences
        y_true_forward, y_pred_forward = unpad(x, y_true_forward, y_pred_forward)
        y_true_backward, y_pred_backward = unpad(x, y_true_backward, y_pred_backward)

        # Compute and print perplexity
        print('Forward Langauge Model Perplexity: {}'.format(ELMo.perplexity(y_pred_forward, y_true_forward)))
        print('Backward Langauge Model Perplexity: {}'.format(ELMo.perplexity(y_pred_backward, y_true_backward)))

    def wrap_multi_elmo_encoder(self, print_summary=False, save=False):
        """
        Wrap ELMo meta-model encoder, which returns an array of the 3 intermediate ELMo outputs
        :param print_summary: print a summary of the new architecture
        :param save: persist model
        :return: None
        """

        elmo_embeddings = list()
        elmo_embeddings.append(concatenate([self._model.get_layer('token_encoding').output, self._model.get_layer('token_encoding').output],
                                           name='elmo_embeddings_level_0'))
        for i in range(self.parameters['n_lstm_layers']):
            elmo_embeddings.append(concatenate([self._model.get_layer('f_block_{}'.format(i + 1)).output,
                                                Lambda(function=ELMo.reverse)
                                                (self._model.get_layer('b_block_{}'.format(i + 1)).output)],
                                               name='elmo_embeddings_level_{}'.format(i + 1)))

        camos = list()
        for i, elmo_embedding in enumerate(elmo_embeddings):
            camos.append(Camouflage(mask_value=0.0, name='camo_elmo_embeddings_level_{}'.format(i + 1))([elmo_embedding,
                                                                                                         self._model.get_layer(
                                                                                                             'token_encoding').output]))

        self._elmo_model = Model(inputs=[self._model.get_layer('word_indices').input], outputs=camos)

        if print_summary:
            self._elmo_model.summary()

        if save:
            self._elmo_model.save(os.path.join(MODELS_DIR, 'ELMo_Encoder.hd5'))
            print('ELMo Encoder saved successfully')

    def save(self, sampled_softmax=True):
        """
        Persist model in disk
        :param sampled_softmax: reload model using the full softmax function
        :return: None
        """
        if not sampled_softmax:
            self.parameters['num_sampled'] = self.parameters['vocab_size']
        self.compile_elmo()
        self._model.load_weights(os.path.join(MODELS_DIR, 'elmo_best_weights.hdf5'))
        self._model.save(os.path.join(MODELS_DIR, 'ELMo_LM_EVAL.hd5'))
        print('ELMo Language Model saved successfully')

    def load(self):
        self._model = load_model(os.path.join(MODELS_DIR, 'ELMo_LM.h5'),
                                 custom_objects={'TimestepDropout': TimestepDropout,
                                                 'Camouflage': Camouflage})

    def load_elmo_encoder(self):
        self._elmo_model = load_model(os.path.join(MODELS_DIR, 'ELMo_Encoder.hd5'),
                                      custom_objects={'TimestepDropout': TimestepDropout,
                                                      'Camouflage': Camouflage})

    def get_outputs(self, test_data, output_type='word', state='last'):
        """
       Wrap ELMo meta-model encoder, which returns an array of the 3 intermediate ELMo outputs
       :param test_data: data generator
       :param output_type: "word" for word vectors or "sentence" for sentence vectors
       :param state: 'last' for 2nd LSTMs outputs or 'mean' for mean-pooling over inputs, 1st LSTMs and 2nd LSTMs
       :return: None
       """
        # Generate samples
        x = []
        for i in range(len(test_data)):
            test_batch = test_data[i][0]
            x.extend(test_batch[0])

        preds = np.asarray(self._elmo_model.predict(np.asarray(x)))
        if state == 'last':
            elmo_vectors = preds[-1]
        else:
            elmo_vectors = np.mean(preds, axis=0)

        if output_type == 'words':
            return elmo_vectors
        else:
            return np.mean(elmo_vectors, axis=1)

    @staticmethod
    def reverse(inputs, axes=1):
        return K.reverse(inputs, axes=axes)

    @staticmethod
    def perplexity(y_pred, y_true):

        cross_entropies = []
        for y_pred_seq, y_true_seq in zip(y_pred, y_true):
            # Reshape targets to one-hot vectors
            y_true_seq = to_categorical(y_true_seq, y_pred_seq.shape[-1])
            # Compute cross_entropy for sentence words
            cross_entropy = K.categorical_crossentropy(K.tf.convert_to_tensor(y_true_seq, dtype=K.tf.float32),
                                                       K.tf.convert_to_tensor(y_pred_seq, dtype=K.tf.float32))
            cross_entropies.extend(cross_entropy.eval(session=K.get_session()))

        # Compute mean cross_entropy and perplexity
        cross_entropy = np.mean(np.asarray(cross_entropies), axis=-1)

        return pow(2.0, cross_entropy)
