# import the necessary packages
from tensorflow import keras
import tensorflow as tf
from models.m_sub.m_tools import *
class upsample_layer(keras.Model):
    """
    Model sub-class
    """
    def __init__(self,filters,padding,
                 kernel_initializer, bias_initializer, use_bias):
        #Call the parent constructor
        super(upsample_layer,self).__init__()
        self.upscale = 2
        self.filters = filters
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.w = None
        self.b = None
        ##----------------------------------------------------------------------------------------
    def build(self, input_shape):
        *_, n_channels = input_shape
        self.w = tf.Variable(initial_value=self.kernel_initializer(shape=(*self.kernel_size,
                                                                          n_channels,self.filters)),
                             dtype='float32', trainable=True)

    def call(self,bottom, n_channels):
        """
        """
        kernel_size = 2 * self.upscale - self.upscale % 2
        stride = self.upscale
        strides = [1, stride, stride, 1]

        # Shape of the bottom tensor
        in_shape = tf.shape(bottom)

        h = ((in_shape[1] - 1) * stride) + 2
        w = ((in_shape[2] - 1) * stride) + 2
        new_shape = [in_shape[0], h, w, self.filters]
        output_shape = tf.stack(new_shape)

        filter_shape = [kernel_size, kernel_size, self.filters, self.filters]

        self.weights = get_bilinear_filter(filter_shape, self.upscale)
        deconv = tf.nn.conv2d_transpose(bottom, self.w, output_shape,
                                        strides=strides, padding='SAME')

        return deconv
