# import the necessary packages
from models.m_sub.conv_layer import *
from models.m_sub.nconv_layer import *
from tensorflow.keras.layers import UpSampling2D
from models.m_sub.m_tools import *
class M8(keras.Model):
    """
    Model sub-class
    """
    def __init__(self, number_classes=2,img_height=128):
        #Call the parent constructor
        super(M8,self).__init__()
        self.number_of_classes = number_classes
        self.img_height = img_height
        # Encoder

        self.scale1 = tf.Variable(tf.ones([32]))
        self.beta1 = tf.Variable(tf.zeros([32]))
        self.scale2 = tf.Variable(tf.ones([64]))
        self.beta2 = tf.Variable(tf.zeros([64]))
        # Layer 1
        self.conv1 = nconv_layer(32,(3,3),(1,1),tf.nn.relu,"SAME",
                                tf.initializers.glorot_uniform(seed=42),
                                tf.initializers.Zeros(),use_bias=False)
        self.conv2 = nconv_layer(64,(3,3),(1,1),tf.nn.relu,"SAME",
                                tf.initializers.glorot_uniform(seed=42),
                                tf.initializers.Zeros(),use_bias=False)
        self.upsample7 = UpSampling2D((2,2),interpolation="bilinear")
        self.conv7 = conv_layer(32,(1,1),(1,1),tf.nn.relu,"SAME",
                                tf.initializers.glorot_uniform(seed=42),
                                tf.initializers.Zeros(),use_bias=False)
        self.upsample8 = UpSampling2D((2,2),interpolation="bilinear")
        self.conv8 = conv_layer(self.number_of_classes,(1,1),(1,1),tf.nn.relu,"SAME",
                                tf.initializers.glorot_uniform(seed=42),
                                tf.initializers.Zeros(),use_bias=False)
        ##----------------------------------------------------------------------------------------


    def call(self,inputs,drop_conv,training=False):
        """
        """
        x = self.conv1(inputs, drop_conv, self.scale1,
                           self.beta1, training)
        x = self.conv2(x,drop_conv, self.scale2,
                           self.beta2, training)
        # Layer 7
        x = self.upsample7(x)
        x = self.conv7(x,drop_conv)
        # Layer 8
        x = self.upsample8(x)
        x = self.conv8(x, drop_conv)
        seg_logits= tf.reshape(tensor=x,shape=(-1, self.img_height * self.img_height, self.number_of_classes))
        ## First to second stage transition
        result_max = tf.argmax(x, 3)  # max result along axis 3
        result_max_labels = tf.equal(result_max, 1)  # object class is 0
        result_max = tf.cast(result_max_labels, tf.float32)
        fcn_pred = tf.reshape(tensor=result_max, shape=(-1, self.img_height, self.img_height, 1))

        return seg_logits,fcn_pred
