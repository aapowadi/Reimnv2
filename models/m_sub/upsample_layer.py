# import the necessary packages
from models.m_sub.bilinear_filter import *
import tensorflow as tf


class upsample_layer(keras.Model):
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

    def call(self, inputs):
        """
        """
        in_shape = tf.shape(inputs)

        h = ((in_shape[1] - 1) * self.stride) + 2
        w = ((in_shape[2] - 1) * self.stride) + 2
        new_shape = [in_shape[0], h, w, self.n_channels]
        output_shape = tf.stack(new_shape)
        filter_shape = [self.kernel_size, self.kernel_size, self.n_channels, self.n_channels]
        self.bilinear_filter = b_filter(filter_shape,self.upscale_factor,self.wname)
        _bil_fil = self.bilinear_filter(inputs)
        deconv = tf.nn.conv2d_transpose(inputs, _bil_fil, output_shape,
                                        strides=self.strides, padding='SAME')
        return deconv


