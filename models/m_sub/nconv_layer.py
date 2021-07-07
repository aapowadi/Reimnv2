# import the necessary packages
from tensorflow import keras
import tensorflow as tf
class nconv_layer(keras.Model):
    """
    Model sub-class
    """
    def __init__(self,filters,kernel_size,strides,activation,padding,
                 kernel_initializer, bias_initializer, use_bias):
        #Call the parent constructor
        super(nconv_layer,self).__init__()
        self.epsilon = 1e-3
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.w = None
        self.b = None
    def build(self, input_shape):
        *_, n_channels = input_shape
        self.w = tf.Variable(initial_value=self.kernel_initializer(shape=(*self.kernel_size,
                                                                          n_channels,self.filters)),
                             dtype='float32', trainable=True)
        if(self.use_bias):
            self.b = tf.Variable(
            initial_value=self.bias_initializer(shape=(self.filters,),
                                                dtype='float32'),trainable=True)
        ##----------------------------------------------------------------------------------------


    def call(self,X,drop_conv,scale2,beta2, phase_train = False):
        """
        Convolution Layer with batch normalization.
        Args:
            X = Input
            w = Filters
            drop_conv = dropout rate
            beta2 = offset
            scale2 = scale
        """
        x = tf.nn.conv2d(X, filters=self.w, strides=self.strides, padding=self.padding)
        if phase_train == True:
            batch_mean2, batch_var2 = tf.nn.moments(x, [0])
            BN1 = tf.nn.batch_normalization(x, batch_mean2, batch_var2, beta2, scale2, self.epsilon)
            if self.use_bias:
                x = BN1 + self.b
            x = self.activation(BN1)
        else:
            if self.use_bias:
                x = x + self.b
            x = self.activation(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.dropout(x, drop_conv)

        return x

