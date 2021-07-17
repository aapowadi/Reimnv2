import sys
sys.dont_write_bytecode = True
import tensorflow as tf
import numpy as np
import os
from enum import Enum
epsilon = 1e-3
class DNNMODE(Enum):
    TRAIN = 1
    VALID = 2


def init_weights(shape, name):
    return tf.Variable(tf.random.normal(shape, name=name, stddev = 0.01),trainable=True)



def conv_layer(X, w, p_keep_conv,scale2,beta2, phase_train):

    conv1 = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')
    if phase_train == True:
        batch_mean2, batch_var2 = tf.nn.moments(conv1, [0])
        BN1 = tf.nn.batch_normalization(conv1, batch_mean2, batch_var2, beta2, scale2, epsilon)
        # conv_norm = tf.layers.BatchNormalization()(conv1, training = phase_train)
        conv1_a = tf.nn.relu(BN1)
    else:
        conv1_a = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, p_keep_conv)

    return conv1

def get_bilinear_filter(filter_shape, upscale_factor):
    """
    Upscales the weight values.
    :param filter_shape:  filter_shape is [width, height, num_in_channels, num_out_channels]
    :param upscale_factor:
    :return:
    """

    kernel_size = filter_shape[1]
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
    bilinear_weights = tf.Variable(initial_value = init(weights.shape,dtype=tf.float32),
                                   name="decon_bilinear_filter",trainable=True)
    return bilinear_weights



def upsample_layer(bottom, n_channels, name, upscale_factor):

    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    # Shape of the bottom tensor
    in_shape = tf.shape(bottom)

    h = ((in_shape[1] - 1) * stride) + 2
    w = ((in_shape[2] - 1) * stride) + 2
    new_shape = [in_shape[0], h, w, n_channels]
    output_shape = tf.stack(new_shape)

    filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

    weights = get_bilinear_filter(filter_shape, upscale_factor)
    deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                    strides=strides, padding='SAME')

    return deconv


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)