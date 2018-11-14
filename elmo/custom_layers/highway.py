from keras.layers.core import Layer
from keras import initializers, regularizers, constraints, activations
from keras.initializers import Constant
from keras import backend as K


class Highway(Layer):
    """Highway network, a natural extension of LSTMs to feedforward networks.

    # Arguments
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        transform_activation: Activation function to use
            for the transform unit
            (see [activations](../activations.md)).
            Default: sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).x
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        transform_initializer: Initializer for the `transform` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        transform_bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
            Default: -2 constant.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        transform_regularizer: Regularizer function applied to
            the `transform` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        transform_bias_regularizer: Regularizer function applied to the transform bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # References
        - [Highway Networks](http://arxiv.org/pdf/1505.00387v2.pdf)
    """

    def __init__(self,
                 activation='relu',
                 transform_activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 transform_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 transform_bias_initializer=-2,
                 kernel_regularizer=None,
                 transform_regularizer=None,
                 bias_regularizer=None,
                 transform_bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.transform_activation = activations.get(transform_activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.transform_initializer = initializers.get(transform_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        if isinstance(transform_bias_initializer, int):
            self.transform_bias_initializer = Constant(value=transform_bias_initializer)
        else:
            self.transform_bias_initializer = initializers.get(transform_bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.transform_regularizer = regularizers.get(transform_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.transform_bias_regularizer = regularizers.get(transform_bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[-1]

        self.W = self.add_weight(shape=(input_dim, input_dim),
                                 name='{}_W'.format(self.name),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.W_transform = self.add_weight(shape=(input_dim, input_dim),
                                           name='{}_W_transform'.format(self.name),
                                           initializer=self.transform_initializer,
                                           regularizer=self.transform_regularizer,
                                           constraint=self.kernel_constraint)

        self.bias = self.add_weight(shape=(input_dim,),
                                 name='{}_bias'.format(self.name),
                                 initializer=self.bias_initializer,
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.bias_transform = self.add_weight(shape=(input_dim,),
                                           name='{}_bias_transform'.format(self.name),
                                           initializer=self.transform_bias_initializer,
                                           regularizer=self.transform_bias_regularizer)

        self.built = True

    def call(self, x, mask=None):
        x_h = self.activation(K.dot(x, self.W) + self.bias)
        x_trans = self.transform_activation(K.dot(x, self.W_transform) + self.bias_transform)
        output = x_h * x_trans + (1 - x_trans) * x
        return output

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'transform_activation': activations.serialize(self.transform_activation),
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'transform_initializer': initializers.serialize(self.transform_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'transform_bias_initializer': initializers.serialize(self.transform_bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'transform_regularizer': regularizers.serialize(self.transform_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'transform_bias_regularizer': regularizers.serialize(self.transform_bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
                  }
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
