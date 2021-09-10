# import the necessary packages
from .conva_layer import *
from .conv_layer import *
from .upsample_layer import *
import tensorflow as tf
class SM11(keras.Model):
    """
    Model sub-class
    """
    def __init__(self, number_classes=2,img_height=128, training = True):
        #Call the parent constructor
        super(SM11,self).__init__()
        self.number_of_classes = number_classes
        self.img_height = img_height
        self.trainable = training
        # Encoder
        # Layer 1
        self.conv1 = conva_layer((3,3),3,32,(1,1),tf.keras.initializers.RandomNormal(stddev=0.01),wname = "w1"
                                 ,trainable= self.trainable)
        # Layer 2
        self.conv2 = conva_layer((3, 3), 32, 64, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="w2"
                                 ,trainable= self.trainable)
        # Layer 3
        self.conv3 = conva_layer((3, 3), 64, 128, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="w3"
                                 ,trainable= self.trainable)
        # Layer 4
        self.conv4 = conva_layer((3, 3), 128, 256, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="w4"
                                 ,trainable= self.trainable)

        # Decoder
        # Layer 5
        self.up_sam5 = upsample_layer(256,2,"deconv1",trainable= self.trainable)
        self.conv5u = conv_layer((3, 3), 256, 128, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="pw1"
                                 ,trainable= self.trainable)
        # Layer 6
        self.up_sam6 = upsample_layer(128,2,"deconv2",trainable= self.trainable)
        self.conv6u = conv_layer((3, 3), 128, 64, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="pw2"
                                 ,trainable= self.trainable)
        # Layer 7
        self.up_sam7 = upsample_layer(64,2,"deconv3",trainable= self.trainable)
        self.conv7u = conv_layer((3, 3), 64, 32, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="pw3"
                                 ,trainable= self.trainable)
        # Layer 8
        self.up_sam8 = upsample_layer(32,2,"deconv4",trainable= self.trainable)
        self.conv8u = conv_layer((3, 3), 32, self.number_of_classes, (1, 1),
                                 tf.keras.initializers.RandomNormal(stddev=0.01), wname="pw4",trainable= self.trainable)
        ##----------------------------------------------------------------------------------------


    def call(self,inputs,drop_conv,training=False):
        """
        """
        # Layer 1
        x = self.conv1(inputs)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.dropout(x, drop_conv)


        # Layer 2
        x = self.conv2(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.dropout(x, drop_conv)


        # Layer 3
        x = self.conv3(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.dropout(x, drop_conv)


        # Layer 4
        x = self.conv4(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        x = tf.nn.dropout(x, drop_conv)


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

        return seg_logits, fcn_pred
