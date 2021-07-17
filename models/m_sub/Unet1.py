# import the necessary packages
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dropout
import tensorflow as tf
class M1(keras.Model):
    """
    Model sub-class
    """
    def __init__(self, number_classes=2,img_height=128,chanDim=-1):
        #Call the parent constructor
        super(M1,self).__init__()
        self.number_of_classes = number_classes
        self.img_height = img_height
        initializer = 'he_normal'
        # Encoder
        # Layer 1
        self.conv1=Conv2D(64, 3, activation='relu',padding="same",kernel_initializer = initializer)
        self.conv1a = Conv2D(64, 3, activation='relu', padding="same", kernel_initializer = initializer)
        self.max1=MaxPooling2D(pool_size=(2, 2))
        self.bn1=BatchNormalization(axis=chanDim,momentum=0.993, epsilon=1e-3)
        # Layer 2
        self.conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)
        self.conv2a = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer = initializer)
        self.max2=MaxPooling2D(pool_size=(2, 2))
        self.bn2 = BatchNormalization(axis=chanDim,momentum=0.993, epsilon=1e-3)
        # Layer 3
        self.conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = initializer)
        self.conv3a = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer = initializer)

        self.max3=MaxPooling2D(pool_size=(2, 2))
        self.bn3 = BatchNormalization(axis=chanDim,momentum=0.993, epsilon=1e-3)
        # Layer 4
        self.conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv4a = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)
        self.max4=MaxPooling2D(pool_size=(2, 2))
        self.bn4 = BatchNormalization(axis=chanDim,momentum=0.993, epsilon=1e-3)

        self.conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)
        self.conv5a = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=initializer)

        self.up_sam6 = UpSampling2D((2,2), interpolation="bilinear")
        self.conv6u = Conv2D(64,(1,1),padding="same",kernel_initializer=initializer)
        # Layer 7
        self.up_sam7 = UpSampling2D((2,2), interpolation="bilinear")
        self.conv7u = Conv2D(32,(1,1),padding="same",kernel_initializer=initializer)
        # Layer 8
        self.up_sam8 = UpSampling2D((2,2), interpolation="bilinear")
        self.conv8u = Conv2D(self.number_of_classes, (1, 1), padding="same",kernel_initializer=initializer)
        # Flatten the predictions, so that we can compute cross-entropy for
        # each pixel and get a sum of cross-entropies.
        # For image size (64x64), the last layer is of size [N,64,64,2] -> [N, 4096, 2]
        # with number_of_classes = 2
        ##----------------------------------------------------------------------------------------


    def call(self,inputs,drop_conv,training=False):
        """
        """
        # Layer 1
        x = self.conv1(inputs)
        if training:
            x = self.bn1(x)
        x = keras.layers.Activation("relu")(x)
        x = self.max1(x)
        x = Dropout(drop_conv)(x)
        # Layer 2
        x = self.conv2(x)
        if training:
            x = self.bn2(x)
        x = keras.layers.Activation("relu")(x)
        x = self.max2(x)
        x = Dropout(drop_conv)(x)
        # Layer 3
        x = self.conv3(x)
        if training:
            x = self.bn3(x)
        x = keras.layers.Activation("relu")(x)
        x = self.max3(x)
        x = Dropout(drop_conv)(x)
        # Layer 4
        x = self.conv4(x)
        if training:
            x = self.bn4(x)
        x = keras.layers.Activation("relu")(x)
        x = self.max4(x)
        x = Dropout(drop_conv)(x)
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
