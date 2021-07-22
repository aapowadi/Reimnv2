# import the necessary packages
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from models.cnntools_norm import *
class Model_8n_tst2(keras.Model):
    """
    Model sub-class
    """
    def __init__(self, number_classes=2,img_height=128,chanDim=-1):
        #Call the parent constructor
        super(Model_8n_tst2,self).__init__()
        self.number_of_classes = number_classes
        self.img_height = img_height
        # Encoder
        self.w1 = init_weights([3, 3, 3, 32], "w1")
        self.scale1 = tf.Variable(tf.ones([32]))
        self.beta1 = tf.Variable(tf.zeros([32]))
        self.w2 = init_weights([3, 3, 32, 64], "w2")
        self.scale2 = tf.Variable(tf.ones([64]))
        self.beta2 = tf.Variable(tf.zeros([64]))
        # Layer 1

        # Layer 3
        self.conv3 = Conv2D(128, (3, 3), (1, 1),activation="relu", padding="same")
        self.bn3 = BatchNormalization(axis=chanDim)
        # Layer 4
        self.conv4 = Conv2D(256, (3, 3), (1, 1),activation="relu", padding="same")
        self.bn4 = BatchNormalization(axis=chanDim)


        # Decoder
        # Layer 5
        self.conv5u = Conv2D(128,(1,1),padding="same")
        # Layer 6
        self.conv6u = Conv2D(64,(1,1),padding="same")
        # Layer 7
        self.conv7u = Conv2D(32,(1,1),padding="same")
        # Layer 8
        self.up_sam8 = Conv2DTranspose(32,(1,1),(2,2))
        self.conv8u = Conv2D(self.number_of_classes, (1, 1), padding="same")
        # Flatten the predictions, so that we can compute cross-entropy for
        # each pixel and get a sum of cross-entropies.
        # For image size (64x64), the last layer is of size [N,64,64,2] -> [N, 4096, 2]
        # with number_of_classes = 2
        ##----------------------------------------------------------------------------------------


    def call(self,inputs,drop_conv,training=False):
        """
        """

        # Layer 1
        #x=self.conv1(inputs)
        x = conv_layer(inputs, self.w1, drop_conv, self.scale1,
                           self.beta1, training)
        x = conv_layer(x, self.w2, drop_conv, self.scale2,
                           self.beta2, training)
        # Layer 7
        x = upsample_layer(x, 64, "deconv3", 2)
        x = tf.nn.conv2d(x, init_weights([1, 1, 64, 32], "pw3"), strides=[1, 1, 1, 1], padding='SAME')
        # Layer 8
        x = upsample_layer(x, 32, "deconv4", 2)
        x = tf.nn.conv2d(x, init_weights([1, 1, 32, self.number_of_classes], "pw4"), strides=[1, 1, 1, 1],
                         padding='SAME')
        seg_logits= tf.reshape(tensor=x,shape=(-1, self.img_height * self.img_height, self.number_of_classes))
        ## First to second stage transition
        result_max = tf.argmax(x, 3)  # max result along axis 3
        result_max_labels = tf.equal(result_max, 1)  # object class is 0
        result_max = tf.cast(result_max_labels, tf.float32)
        fcn_pred = tf.reshape(tensor=result_max, shape=(-1, self.img_height, self.img_height, 1))

        return seg_logits,fcn_pred
