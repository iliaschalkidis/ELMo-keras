from keras.layers import Layer
import keras.backend as K


class SampledSoftmax(Layer):
    """Sampled Softmax, a faster way to train a softmax classifier over a huge number of classes.

    # Arguments
        num_classes: number of classes
        num_sampled: number of classes to be sampled at each batch
        tied_to: layer to be tied with (e.g., Embedding layer)
        kwargs:
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # References
        - [Tensorflow code](tf.nn.sampled_softmax_loss)
        - [Sampled SoftMax](https://www.tensorflow.org/extras/candidate_sampling.pdf)
    """
    def __init__(self, num_classes=50000, num_sampled=1000, tied_to=None, **kwargs):
        super(SampledSoftmax, self).__init__(**kwargs)
        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.tied_to = tied_to
        self.sampled = (self.num_classes != self.num_sampled)

    def build(self, input_shape):
        if self.tied_to is None:
            self.softmax_W = self.add_weight(shape=(self.num_classes, input_shape[0][-1]), name='W_soft', initializer='lecun_normal')
        self.softmax_b = self.add_weight(shape=(self.num_classes,), name='b_soft', initializer='zeros')
        self.built = True

    def call(self, x, mask=None):
        lstm_outputs, next_token_ids = x

        def sampled_softmax(x):
            lstm_outputs_batch, next_token_ids_batch = x
            batch_losses = K.tf.nn.sampled_softmax_loss(
                self.softmax_W if self.tied_to is None else self.tied_to.weights[0], self.softmax_b,
                next_token_ids_batch, lstm_outputs_batch,
                num_classes=self.num_classes,
                num_sampled=self.num_sampled,
                partition_strategy='div')
            batch_losses = K.tf.reduce_mean(batch_losses)
            return [batch_losses, batch_losses]

        def softmax(x):
            lstm_outputs_batch, next_token_ids_batch = x
            logits = K.tf.matmul(lstm_outputs_batch,
                                 K.tf.transpose(self.softmax_W) if self.tied_to is None else K.tf.transpose(self.tied_to.weights[0]))
            logits = K.tf.nn.bias_add(logits, self.softmax_b)
            batch_predictions = K.tf.nn.softmax(logits)
            labels_one_hot = K.tf.one_hot(K.tf.cast(next_token_ids_batch, dtype=K.tf.int32), self.num_classes)
            batch_losses = K.tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
            return [batch_losses, batch_predictions]

        losses, predictions = K.tf.map_fn(sampled_softmax if self.sampled else softmax, [lstm_outputs, next_token_ids])
        self.add_loss(0.5 * K.tf.reduce_mean(losses[0]))
        return lstm_outputs if self.sampled else predictions

    def compute_output_shape(self, input_shape):
        return input_shape[0] if self.sampled else (input_shape[0][0], input_shape[0][1], self.num_classes)
