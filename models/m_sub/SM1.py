# import the necessary packages
from models.m_sub.nconv_layer import *
from models.m_sub.conv_layer import *
from models.m_sub.upsample_layer import *
import tensorflow as tf
class SM1(keras.Model):
    """
    Model sub-class
    """
    def __init__(self, number_classes=2,img_height=128,chanDim=-1):
        #Call the parent constructor
        super(SM1,self).__init__()
        self.number_of_classes = number_classes
        self.img_height = img_height

        self.scale1 = tf.Variable(tf.ones([32]))
        self.beta1 = tf.Variable(tf.zeros([32]))
        self.scale2 = tf.Variable(tf.ones([64]))
        self.beta2 = tf.Variable(tf.zeros([64]))
        self.scale3 = tf.Variable(tf.ones([128]))
        self.beta3 = tf.Variable(tf.zeros([128]))
        self.scale4 = tf.Variable(tf.ones([256]))
        self.beta4 = tf.Variable(tf.zeros([256]))
        # Encoder
        # Layer 1
        self.conv1 = nconv_layer((3,3),3,32,(1,1),tf.keras.initializers.RandomNormal(stddev=0.01),wname = "w1")
        # Layer 2
        self.conv2 = nconv_layer((3, 3), 32, 64, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="w2")
        # Layer 3
        self.conv3 = nconv_layer((3, 3), 64, 128, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="w3")
        # Layer 4
        self.conv4 = nconv_layer((3, 3), 128, 256, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="w4")

        # Decoder
        # Layer 5
        self.up_sam5 = upsample_layer(256,2,"deconv1")
        self.conv5u = conv_layer((3, 3), 256, 128, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="pw1")
        # Layer 6
        self.up_sam6 = upsample_layer(128,2,"deconv2")
        self.conv6u = conv_layer((3, 3), 128, 64, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="pw2")
        # Layer 7
        self.up_sam7 = upsample_layer(64,2,"deconv3")
        self.conv7u = conv_layer((3, 3), 64, 32, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="pw3")
        # Layer 8
        self.up_sam8 = upsample_layer(32,2,"deconv4")
        self.conv8u = conv_layer((3, 3), 32, self.number_of_classes, (1, 1),
                                 tf.keras.initializers.RandomNormal(stddev=0.01), wname="pw4")
        ##----------------------------------------------------------------------------------------


    def call(self,inputs,drop_conv,training=False):
        """
        """
        # Layer 1
        x = self.conv1(inputs,drop_conv,self.beta1,self.scale1)

        # Layer 2
        x = self.conv2(x,drop_conv,self.beta2,self.scale2)

        # Layer 3
        x = self.conv3(x,drop_conv,self.beta3,self.scale3)

        # Layer 4
        x = self.conv4(x,drop_conv,self.beta4,self.scale4)

        # Layer 5
        x = self.up_sam5(x)
        x = self.conv5u(x)

        # Layer 6
        x = self.up_sam6(x)
        x = self.conv6u(x)

        # Layer 7
        x = self.up_sam7(x)
        x = self.conv7u(x)

        # Layer 8
        x = self.up_sam8(x)
        x = self.conv8u(x)
        seg_logits= tf.reshape(tensor=x,shape=(-1, self.img_height * self.img_height, self.number_of_classes))
        ## First to second stage transition
        result_max = tf.argmax(x, 3)  # max result along axis 3
        result_max_labels = tf.equal(result_max, 1)  # object class is 0
        result_max = tf.cast(result_max_labels, tf.float32)
        fcn_pred = tf.reshape(tensor=result_max, shape=(-1, self.img_height, self.img_height, 1))

        return seg_logits,fcn_pred
