# import the necessary packages
from tensorflow import keras
import numpy as np
import tensorflow as tf


class b_filter(keras.layers.Layer):
    """
    Model sub-class
    """

    def __init__(self,filter_shape,
                 upscale_factor, wname = "kernel"):
        # Call the parent constructor
        super(b_filter, self).__init__()
        kernel_size = filter_shape[1]
        self.wname = wname
        ### Centre location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5

        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location) / upscale_factor)) * (
                        1 - abs((y - centre_location) / upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights)
        self.bilinear_weights = tf.Variable(initial_value = init(weights.shape,dtype=tf.float32), trainable=True,
                                            name=self.wname, dtype=tf.float32)

    ##----------------------------------------------------------------------------------------
    def call(self,inputs):
        """
        """
        return self.bilinear_weights


