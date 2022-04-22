# import the necessary packages
from tensorflow import keras
from models.cnntools_norm import *
class M8n_tf2(keras.layers.Layer):
    """
    Model sub-class
    """
    def __init__(self, number_classes=2,img_height=128,chanDim=-1):
        #Call the parent constructor
        super(M8n_tf2,self).__init__()
        self.number_of_classes = number_classes
        self.img_height = img_height
        self.upscale = 2
        self.w1 = init_weights([3, 3, 3, 32], "w1")
        self.scale1 = tf.Variable(tf.ones([32]),trainable=True)
        self.beta1 = tf.Variable(tf.zeros([32]),trainable=True)
        self.w2 = init_weights([3, 3, 32, 64], "w2")
        self.scale2 = tf.Variable(tf.ones([64]),trainable=True)
        self.beta2 = tf.Variable(tf.zeros([64]),trainable=True)
        self.w3 = init_weights([3, 3, 64, 128], "w3")
        self.scale3 = tf.Variable(tf.ones([128]),trainable=True)
        self.beta3 = tf.Variable(tf.zeros([128]),trainable=True)
        self.w4 = init_weights([3, 3, 128, 256], "w4")
        self.scale4 = tf.Variable(tf.ones([256]),trainable=True)
        self.beta4 = tf.Variable(tf.zeros([256]),trainable=True)

    def call(self,X,drop_conv,training=False):
        """
        """
        # convolution
        conv1 = conv_layer(X, self.w1,drop_conv, self.scale1,
                           self.beta1, training)
        conv2 = conv_layer(conv1, self.w2, drop_conv, self.scale2,
                           self.beta2, training)
        conv3 = conv_layer(conv2, self.w3, drop_conv, self.scale3,
                           self.beta3, training)
        conv4 = conv_layer(conv3, self.w4, drop_conv, self.scale4,
                           self.beta4, training)

        # deconvolution
        deconv1 = upsample_layer(conv4, 256, "deconv1", self.upscale)
        pconv1 = tf.nn.conv2d(deconv1, init_weights([1, 1, 256, 128], "pw1"), strides=[1, 1, 1, 1], padding='SAME')
        deconv2 = upsample_layer(pconv1, 128, "deconv2", self.upscale)
        pconv2 = tf.nn.conv2d(deconv2, init_weights([1, 1, 128, 64], "pw2"), strides=[1, 1, 1, 1], padding='SAME')
        deconv3 = upsample_layer(pconv2, 64, "deconv3", self.upscale)
        pconv3 = tf.nn.conv2d(deconv3, init_weights([1, 1, 64, 32], "pw3"), strides=[1, 1, 1, 1], padding='SAME')
        deconv4 = upsample_layer(pconv3, 32, "deconv4", 2)
        result = tf.nn.conv2d(deconv4, init_weights([1, 1, 32, self.number_of_classes], "pw4"), strides=[1, 1, 1, 1],
                              padding='SAME')

        # Flatten the predictions, so that we can compute cross-entropy for
        # each pixel and get a sum of cross-entropies.
        # For image size (64x64), the last layer is of size [N,64,64,2] -> [N, 4096, 2]
        # with number_of_classes = 2
        seg_logits = tf.reshape(tensor=result,
                                              shape=(-1, self.img_height * self.img_height, self.number_of_classes))

        result_max = tf.argmax(result, 3)  # max result along axis 3
        result_max_labels = tf.equal(result_max, 1)  # object class is 0
        result_max = tf.cast(result_max_labels, tf.float32)
        fcn_pred = tf.reshape(tensor=result_max, shape=(-1, self.img_height, self.img_height, 1))

        return seg_logits,fcn_pred
