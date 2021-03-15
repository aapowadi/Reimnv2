import tensorflow as tf
import numpy as np
from models.cnntools import *
from models.plottools import *
import cv2

class Model_RGBD_6DoF:
    """
    This file implements a two-stage convolutional neural network for 6DoF pose estimation

    The first stage implements a fully connected network (FCN) for semantic segmentation.
    It predicts the location of an object pixel-wise. The second stage are two regression network.

    The network processes RGB and depth data.
    The first stage only works on an RGB image and segments the object of interest in the images.
    The segmentation mask is used to cut the depth portion of the object from the depth mask.
    The segmented depth part is feed into the translation regressor and the orientation regressor.

                                      depth
                                        |
    RGB -> FCN -> segmentation mask -> segmented depth -> CNN-regressor of the translation
                                                   |---> CNN-regressor of the orientation

    Note that stage 1 and stage 2 are disconnected, their graphs are disconnected. The results from stage 1
    are funneled into stage 2 via a placeholder.

    Input:
    - RGB images in the range [0,255] per channel
    - Depth images in the range [0,1] per channel.

    Output:
    The outcome is a pose [t|q]
    - t = (x, y, z)
    - q = (qx, qy, qz, qw)



    Rafael Radkowski
    Iowa State University
    rafael@iastate.edu
    May 4, 2018
    MIT License
    -----------------------------------------------------
    last edit:

    July 18, 2019, RR
    - Changed the second stage loss to tf.norm (euclidian distance)

    July 19, 2019, AP
    - changed the kernel number cascade from 32->64->128->128->64->32 to 32->64->128->256->256->128->64->32

    """

    # the number of output classes, note that one class must be the background
    number_of_classes = 2

    # number of outputs for translation and quaternione
    number_trans_outputs = 3
    number_rot_outputs = 4

    # reference to placeholders for the input data.
    X_rgb = []  # input data RGB
    X_pred = []  # prediction output of the first network
    X_depth = []  # input data depth

    # Dropout ratio placeholder
    p_keep_conv = []
    p_keep_hidden = []

    # model weights for the conv part
    w1 = []
    w2 = []
    w3 = []

    # for the translation regression network
    w4 = []
    w5 = []
    wf1 = []
    bf1 = []
    wout = []
    bout = []
    FC_layer = []

    # for the rotation regression network
    w4r = []
    w5r = []
    wf1_r = []
    bf1_r = []
    wout_r = []
    bout_r = []
    FC_layer_r = []

    # Model endpoints
    segmentation_logits = [] # segmentation activations
    fcn_predictions = [] # segmentation results from the first network (argmax)
    trans_predictions = [] # translation prediction
    rot_predictions = [] # Quaternion prediction


    def __init__(self, number_classes=2, number_trans_outputs = 3, number_rot_outputs = 4):
        """
        Constructor to initialize the object
        :param number_classes: (int)
            Number of classes to segment
        :param number_trans_outputs: (int)
            Number of translation outputs. This is usually three but one can reduce the number for
            testing and other experiments
        :param number_rot_outputs: (int)
            Number of quaternion outputs, usually four.
        """
        self.number_of_classes = number_classes
        self.number_trans_outputs = number_trans_outputs
        self.number_rot_outputs = number_rot_outputs


    def createModel(self, X, Xd, Xpred, p_keep_conv, p_keep_hidden, img_width, img_height):
        """
        Create the CNN model.
        :param X: (array) Placeholder for RGB image input as array of size [N, width, height, 3]. Pixel range is [0, 255]
        :param Xd: (array) Placeholder for depth image input as array of size  [N, width, height, 1], with pixel range [0, 1]
        :param Xpred: (array) Placeholder to funnel the predictions from Stage 1 into Stage 2 of size [N, width, height, C]
        :param p_keep_conv: (float) Dropout, probability to keep the values. For stage 1 only.
        :param p_keep_hidden: (float) Dropout, probability to keep the values, For stage 2 only.
        :return: The four graph endpoints (tensorflow nodes)
                self.segmentation_logits - The activation outputs of stage 1 of size [N, width, height, C]
                self.fcn_predictions - The prediction output of stage 1 of size [N, width, height, C], each pixel contains a class label.
                self.trans_predictions - The translation prediction graph
                self.rot_predictions - The rotation prediction graph
        """

        # upscale factor for the FCN
        upscale = 2

        self.X_rgb = X
        self.X_depth = Xd
        self.X_pred = Xpred
        self.p_keep_conv = p_keep_conv
        self.p_keep_hidden = p_keep_hidden

        #------------------------------------------------------------------------
        # First stage

        ## weight for the network
        self.w1 = init_weights([3, 3, 3, 32], "w1")
        self.w2 = init_weights([3, 3, 32, 64], "w2")
        self.w3 = init_weights([3, 3, 64, 128], "w3")
        self.w4 = init_weights([3, 3, 128, 256], "w4")

        # convolution
        conv1 = conv_layer(X, self.w1, self.p_keep_conv)
        conv2 = conv_layer(conv1, self.w2, self.p_keep_conv)
        conv3 = conv_layer(conv2, self.w3, self.p_keep_conv)
        conv4 = conv_layer(conv3, self.w4, self.p_keep_conv)

        # deconvolution
        deconv1 = upsample_layer(conv4, 256, "deconv1", upscale)
        pconv1 = tf.nn.conv2d(deconv1, init_weights([1, 1, 256, 128], "pw1"), strides=[1, 1, 1, 1], padding='SAME')
        deconv2 = upsample_layer(pconv1, 128, "deconv2", upscale)
        pconv2 = tf.nn.conv2d(deconv2, init_weights([1, 1, 128, 64], "pw2"), strides=[1, 1, 1, 1], padding='SAME')
        deconv3 = upsample_layer(pconv2, 64, "deconv3", upscale)
        pconv3 = tf.nn.conv2d(deconv3, init_weights([1, 1, 64, 32], "pw3"), strides=[1, 1, 1, 1], padding='SAME')
        deconv4 = upsample_layer(pconv3, 32, "deconv4", 2)
        result = tf.nn.conv2d(deconv4, init_weights([1, 1, 32, self.number_of_classes], "pw4"), strides=[1, 1, 1, 1],
                              padding='SAME')

        # Flatten the predictions, so that we can compute cross-entropy for
        # each pixel and get a sum of cross-entropies.
        # For image size (64x64), the last layer is of size [N,64,64,2] -> [N, 4096, 2]
        # with number_of_classes = 2
        self.segmentation_logits = tf.reshape(tensor=result, shape=(-1, img_width*img_height, self.number_of_classes))

        ##----------------------------------------------------------------------------------------
        ## First to second stage transition

        result_max = tf.argmax(result, 3)  # max result along axis 3
        result_max_labels = tf.equal(result_max, 0)  # object class is 0
        result_max = tf.cast(result_max_labels, tf.float32)
        self.fcn_predictions = tf.reshape(tensor=result_max, shape=(-1, img_width, img_height, 1))

        # segment the results from the network and
        depth_in = tf.multiply(Xpred, Xd)

        ##----------------------------------------------------------------------------------------
        ## Second stage translation
        self.w4t = init_weights([3, 3, 1, 32], "w4t") # 64 x 64 x 1 -> 32 x 32 x 32
        self.w5t = init_weights([3, 3, 32, 4], "w5t") # 32 x 32 x 32 -> 16 x 16 x 4
        self.wf1 = init_weights([16 * 16 * 4, 625], "wf1")
        # self.wf1 = init_weights([32 * 32 * 4, 625], "w6") #For 64x64 images
        self.bf1 = tf.Variable(tf.zeros(625))
        self.wout = init_weights([625, self.number_trans_outputs], "w_out")
        self.bout = tf.Variable(tf.zeros(self.number_trans_outputs))

        conv4t = conv_layer(depth_in, self.w4t, self.p_keep_conv)
        conv5 = conv_layer(conv4t, self.w5t, self.p_keep_conv)

        FC_layer = tf.reshape(conv5, [-1, self.wf1.get_shape().as_list()[0]])
        FC_layer = tf.nn.dropout(FC_layer, self.p_keep_hidden)

        output_layer = tf.nn.relu(tf.add(tf.matmul(FC_layer, self.wf1), self.bf1))
        output_layer = tf.nn.dropout(output_layer, self.p_keep_hidden)

        self.trans_predictions = tf.add(tf.matmul(output_layer, self.wout), self.bout)

        ##----------------------------------------------------------------------------------------
        ## Second stage roation
        self.w4r = init_weights([3, 3, 1, 32], "w4r")  # 64 x 64 x 1 -> 32 x 32 x 32
        self.w5r = init_weights([3, 3, 32, 4], "w5r")  # 32 x 32 x 32 -> 16 x 16 x 4
        self.wf1_r = init_weights([16 * 16 * 4, 825], "wf1_r")
        #self.wf1_r = init_weights([16 * 16 * 4, 825], "w6r") # For 64x64 images
        self.bf1_r = tf.Variable(tf.zeros(825))
        self.wout_r = init_weights([825, self.number_rot_outputs], "w_out_r")
        self.bout_r = tf.Variable(tf.zeros(self.number_rot_outputs))

        conv4r = conv_layer(depth_in, self.w4r, self.p_keep_conv)
        conv5r = conv_layer(conv4r, self.w5r, self.p_keep_conv)

        FC_layer_r = tf.reshape(conv5r, [-1, self.wf1_r.get_shape().as_list()[0]])
        FC_layer_r = tf.nn.dropout(FC_layer_r, self.p_keep_hidden)

        output_layer_r = tf.nn.relu(tf.add(tf.matmul(FC_layer_r, self.wf1_r), self.bf1_r))
        output_layer_r = tf.nn.dropout(output_layer_r, self.p_keep_hidden)

        self.rot_predictions = tf.add(tf.matmul(output_layer_r, self.wout_r), self.bout_r)


        return  self.segmentation_logits,  self.fcn_predictions, self.trans_predictions, self.rot_predictions


