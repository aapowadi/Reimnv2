# import the necessary packages
from .conva_layer import *
import tensorflow as tf
class Pose_net_r(keras.Model):
    """
    Model sub-class
    """
    def __init__(self, number_trans_outputs=3,number_rot_outputs = 4,img_height=128,
                 training = True):
        #Call the parent constructor
        super(Pose_net_r,self).__init__()
        self.number_trans_outputs = number_trans_outputs
        self.number_rot_outputs = number_rot_outputs
        self.img_height = img_height
        self.trainable = training
        self.initializer = tf.random_normal_initializer(stddev=0.01)
        # Rotation
        # Layer 1
        self.conv1r = conva_layer((3, 3), 1, 32, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01),
                                  wname="w1_r",trainable=self.trainable)
        # Layer 2
        self.conv2r = conva_layer((3, 3), 32, 16, (1, 1), tf.keras.initializers.RandomNormal(stddev=0.01),
                                  wname="w2_r",trainable=self.trainable)

        self.wf1_r = tf.Variable(initial_value=self.initializer(shape = (32 * 32 * 16, 8025),dtype="float32"),
                                 trainable=self.trainable, name="wf1_r")

        self.bf1_r = tf.Variable(tf.zeros(8025), trainable=self.trainable)

        self.wout_r = tf.Variable(initial_value=self.initializer(shape = (8025, number_rot_outputs),dtype="float32"),
                                  trainable=self.trainable, name="w_out_r")

        self.bout_r = tf.Variable(tf.zeros(number_rot_outputs), trainable=self.trainable)

        ##----------------------------------------------------------------------------------------


    def call(self,fcn_pred,depth_ip,drop_conv,training=False):
        """
        """
        #self.set_states(training)
        depth_in = tf.multiply(fcn_pred, depth_ip)

        # Rotation
        xr = self.conv1r(depth_in, training)
        xr = tf.nn.max_pool(xr, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        xr = tf.nn.dropout(xr, drop_conv)

        # Layer 2
        xr = self.conv2r(xr, training)
        xr = tf.nn.max_pool(xr, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        xr = tf.nn.dropout(xr, drop_conv)

        # Fully Connected Layer
        xr = tf.reshape(xr, [-1, self.wf1_r.get_shape().as_list()[0]])
        xr = tf.nn.dropout(xr, drop_conv)

        xr = tf.nn.relu(tf.add(tf.matmul(xr, self.wf1_r), self.bf1_r))
        xr = tf.nn.dropout(xr, drop_conv)

        xr = tf.add(tf.matmul(xr, self.wout_r), self.bout_r)

        return xr

    def set_states(self, trainable):
        self.wf1_r.trainable = trainable
        self.bf1_r.trainable = trainable
        self.wout_r.trainable = trainable
        self.bout_r.trainable = trainable
