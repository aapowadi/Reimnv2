# import the necessary packages
from tensorflow import keras
import tensorflow as tf


class conv_layer(keras.layers.Layer):
    """
    Model sub-class
    """

    def __init__(self, kernel_size, n_channels, n_filters, strides, kernel_initializer, padding="SAME", wname="kernel"
                 , trainable = True):
        # Call the parent constructor
        super(conv_layer, self).__init__()
        self.n_filters = n_filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.n_channels = n_channels
        self.w = None
        self.padding = padding
        self.wname = wname
        self.trainable = trainable
        ##----------------------------------------------------------------------------------------
        self.w = tf.Variable(initial_value=self.kernel_initializer(shape=(*self.kernel_size, self.n_channels, self.n_filters),
                             dtype='float32'),trainable=self.trainable, name=self.wname, dtype=tf.float32)

    def call(self, X, trainable = True):
        """
        """
        #self.w.trainable = trainable
        x = tf.nn.conv2d(X, self.w, self.strides, padding=self.padding)
        return x
