# import the necessary packages
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dropout
import tensorflow as tf


class nconv_layer(keras.Model):
    """
    Model sub-class
    """

    def __init__(self, kernel_size, n_channels, n_filters, strides, kernel_initializer, padding="SAME", wname="kernel"):
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
        self.w = tf.Variable(
            initial_value=self.kernel_initializer(shape=(*self.kernel_size, self.n_channels, self.n_filters),
                                                  dtype='float32'), trainable=True, name=self.wname, dtype=tf.float32)

        ##----------------------------------------------------------------------------------------
    def call(self, X, drop_conv, beta2, scale2, training=False, epsilon=1e-3):
        """
        """
        conv1 = tf.nn.conv2d(X, self.w, self.strides, padding=self.padding)
        if training:
            batch_mean2, batch_var2 = tf.nn.moments(conv1, [0])
            bn1 = tf.nn.batch_normalization(conv1, batch_mean2, batch_var2, beta2, scale2, epsilon)
            conv1_a = tf.nn.relu(bn1)
        else:
            conv1_a = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, drop_conv)

        return conv1
