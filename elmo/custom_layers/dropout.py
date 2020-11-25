# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout


class TimestepDropout(Dropout):
    """Word Dropout.

    This version performs the same function as Dropout, however it drops
    entire timesteps (e.g., words embeddings) instead of individual elements (features).

    # Arguments
        rate: float between 0 and 1. Fraction of the timesteps to drop.

    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`

    # Output shape
        Same as input

    # References
        - N/A
    """

    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape
