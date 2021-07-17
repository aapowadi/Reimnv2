# import the necessary packages
#from models.m_sub.bilinear_filter import *
import tensorflow as tf
from tensorflow import keras
import numpy as np

class upsample_layer(keras.layers.Layer):
    """
    Model sub-class
    """

    def __init__(self,n_channels,
                 upscale_factor, wname = "kernel"):
        # Call the parent constructor
        super(upsample_layer, self).__init__()
        self.n_channels = n_channels
        self.wname = wname
        self.upscale_factor = upscale_factor
        self.kernel_size = 2 * self.upscale_factor - self.upscale_factor % 2
        self.stride = self.upscale_factor
        self.strides = [1, self.stride, self.stride, 1]
        ##----------------------------------------------------------------------------------------

    def call(self, inputs,training = False):
        """
        """
        in_shape = tf.shape(inputs)

        h = ((in_shape[1] - 1) * self.stride) + 2
        w = ((in_shape[2] - 1) * self.stride) + 2
        new_shape = [in_shape[0], h, w, self.n_channels]
        output_shape = tf.stack(new_shape)
        #self.filter_shape = [self.kernel_size, self.kernel_size, self.n_channels, self.n_channels]
        #self.bilinear_filter = b_filter(filter_shape,self.upscale_factor,self.wname)
        #_bil_fil = self.bilinear_filter(inputs)
        deconv = tf.nn.conv2d_transpose(inputs, self.bilinear_weights, output_shape,
                                        strides=self.strides, padding='SAME')
        return deconv

    def build(self, input_shape):
        filter_shape = [self.kernel_size, self.kernel_size, self.n_channels, self.n_channels]
        kernel_size = filter_shape[1]
        ### Centre location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            centre_location = self.upscale_factor - 1
        else:
            centre_location = self.upscale_factor - 0.5

        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location) / self.upscale_factor)) * (
                        1 - abs((y - centre_location) / self.upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights)
        self.bilinear_weights = tf.Variable(initial_value = init(weights.shape,dtype=tf.float32), trainable=True,
                                            name=self.wname, dtype=tf.float32)



