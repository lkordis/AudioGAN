from keras import backend as K
from keras.layers import Layer

from keras.layers import Lambda
import tensorflow as tf

class SubPixel(Layer):

    def __init__(self, I,r, **kwargs):
        _, a, r = I.get_shape().as_list()
        bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
        self.ratio = r
        self.out_size = (None, a * r, 1)
        super(SubPixel, self).__init__(**kwargs)

    def call(self, I):
        _, a, r = I.get_shape().as_list()
        bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
        X = tf.split(I, a, 1)  # a, [bsize, 1, r]

        X = tf.concat([tf.squeeze(x, axis=1) for x in X], 1)  # bsize, a*r
        return tf.reshape(X, (bsize, a * r, 1))

    def compute_output_shape(self, input_shape):
        return self.out_size