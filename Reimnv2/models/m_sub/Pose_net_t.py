# import the necessary packages
from .conva_layer import *
import tensorflow as tf
class Pose_net_t(keras.Model):
    """
    Model sub-class
    """
    def __init__(self, number_trans_outputs=3,number_rot_outputs = 4,img_height=128,
                 training = True):
        #Call the parent constructor
        super(Pose_net_t,self).__init__()
        self.number_trans_outputs = number_trans_outputs
        self.number_rot_outputs = number_rot_outputs
        self.img_height = img_height
        self._trainable = training
        self.initializer = tf.random_normal_initializer(stddev=0.01)
        # Layer 1
        self.conv1t = conva_layer((3, 3), 1, 32, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="w1",
                                  trainable=self._trainable)
        # Layer 2
        self.conv2t = conva_layer((3, 3), 32, 16, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01), wname="w2",
                                  trainable=self._trainable)

        self.wf1 = tf.Variable(initial_value=self.initializer(shape = (32 * 32 * 16, 6025), dtype="float32"),
                               trainable=self._trainable, name="wf1")

        self.bf1 = tf.Variable(tf.zeros(6025), trainable=self._trainable)

        self.wout = tf.Variable(initial_value=self.initializer(shape = (6025, number_trans_outputs),dtype="float32"),
                                trainable=self._trainable, name="w_out")

        self.bout = tf.Variable(tf.zeros(number_trans_outputs), trainable=self._trainable)

        ##----------------------------------------------------------------------------------------


    def call(self,fcn_pred,depth_ip,drop_conv,training=False):
        """
        """

        # Translation
        #self.set_states(training)
        depth_in = tf.multiply(fcn_pred, depth_ip)
        xt = self.conv1t(depth_in, training)
        xt = tf.nn.max_pool(xt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        xt = tf.nn.dropout(xt, drop_conv)

        # Layer 2
        xt = self.conv2t(xt, training)
        xt = tf.nn.max_pool(xt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        xt = tf.nn.dropout(xt, drop_conv)

        # Fully Connected Layer
        xt = tf.reshape(xt, [-1, self.wf1.get_shape().as_list()[0]])
        xt = tf.nn.dropout(xt, drop_conv)

        xt = tf.nn.relu(tf.add(tf.matmul(xt, self.wf1), self.bf1))
        xt = tf.nn.dropout(xt, drop_conv)

        xt = tf.add(tf.matmul(xt, self.wout), self.bout)

        return xt

    def set_states(self, trainable):
        self.wf1.trainable = trainable
        self.bf1.trainable = trainable
        self.wout.trainable = trainable
        self.bout.trainable = trainable