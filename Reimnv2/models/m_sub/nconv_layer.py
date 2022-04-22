# import the necessary packages
from tensorflow import keras
import tensorflow as tf


class nconv_layer(keras.layers.Layer):
    """
    Model sub-class
    """

    def __init__(self, kernel_size, n_channels, n_filters, strides, kernel_initializer, padding="SAME", wname="kernel"
                 , trainable = True):
        # Call the parent constructor
        super(nconv_layer, self).__init__()
        self.n_filters = n_filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.n_channels = n_channels
        self.w = None
        self.padding = padding
        self.wname = wname
        self.trainable = trainable
        self.w = tf.Variable(
            initial_value=self.kernel_initializer(shape=(*self.kernel_size, self.n_channels, self.n_filters),
                                                  dtype='float32'), trainable=self.trainable, name=self.wname, dtype=tf.float32)

        self.scale1 = tf.Variable(tf.ones([self.n_filters]),trainable=self.trainable)
        self.beta1 = tf.Variable(tf.zeros([self.n_filters]),trainable=self.trainable)
        ##----------------------------------------------------------------------------------------
    def call(self, X,training=False, epsilon=1e-3):
        """
        """
        conv1 = tf.nn.conv2d(X, self.w, self.strides, padding=self.padding)
        if training:
            batch_mean2, batch_var2 = tf.nn.moments(conv1, [0])
            bn1 = tf.nn.batch_normalization(conv1, batch_mean2, batch_var2, self.beta1, self.scale1, epsilon)
            conv1_a = tf.nn.relu(bn1)
        else:
            conv1_a = tf.nn.relu(conv1)
        return conv1_a
