# import the necessary packages
import tensorflow as tf
import numpy as np
def init_weights(shape, name):
    return tf.Variable(tf.random.normal(shape, name=name, stddev = 0.01))

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
    return weights