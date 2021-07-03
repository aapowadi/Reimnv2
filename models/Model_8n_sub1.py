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
class Model_8n_sub1(keras.Model):
    """
    Model sub-class
    """
    def __init__(self, number_classes=2,drop_conv=0, img_height=128,chanDim=-1):
        #Call the parent constructor
        super(Model_8n_sub1,self).__init__()
        self.number_of_classes = number_classes
        self.drop_conv = drop_conv
        self.img_height = img_height
        # Encoder
        # Layer 1
        self.conv1=Conv2D(32,(3,3),(1,1),padding="same")
        self.act1 = Activation("relu")
        self.bn1=BatchNormalization(axis=chanDim)
        self.pool1=MaxPooling2D(pool_size=(2,2))
        self.drop1=Dropout(self.drop_conv)
        # Layer 2
        self.conv2 = Conv2D(64, (3, 3), (1, 1), padding="same")
        self.act2 = Activation("relu")
        self.bn2 = BatchNormalization(axis=chanDim)
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.drop2 = Dropout(self.drop_conv)
        # Layer 3
        self.conv3 = Conv2D(128, (3, 3), (1, 1), padding="same")
        self.act3 = Activation("relu")
        self.bn3 = BatchNormalization(axis=chanDim)
        self.pool3 = MaxPooling2D(pool_size=(2, 2))
        self.drop3 = Dropout(self.drop_conv)
        # Layer 4
        self.conv4 = Conv2D(256, (3, 3), (1, 1), padding="same")
        self.act4 = Activation("relu")
        self.bn4 = BatchNormalization(axis=chanDim)
        self.pool4 = MaxPooling2D(pool_size=(2, 2))
        self.drop4 = Dropout(self.drop_conv)

        # Decoder
        # Layer 5
        self.up_sam5 = Conv2DTranspose(256,(1,1),(2,2))
        self.conv5u = Conv2D(128,(1,1),padding="same")
        # Layer 6
        self.up_sam6 = Conv2DTranspose(128,(1,1),(2,2))
        self.conv6u = Conv2D(64,(1,1),padding="same")
        # Layer 7
        self.up_sam7 = Conv2DTranspose(64,(1,1),(2,2))
        self.conv7u = Conv2D(32,(1,1),padding="same")
        # Layer 8
        self.up_sam8 = Conv2DTranspose(32,(1,1),(2,2))
        self.conv8u = Conv2D(self.number_of_classes, (1, 1), padding="same")
        # Flatten the predictions, so that we can compute cross-entropy for
        # each pixel and get a sum of cross-entropies.
        # For image size (64x64), the last layer is of size [N,64,64,2] -> [N, 4096, 2]
        # with number_of_classes = 2
        ##----------------------------------------------------------------------------------------


    def call(self,inputs,training=None):
        """
        """
    # Layer 1
        x=self.conv1(inputs)
        if training:
            x=self.bn1(x)
            x = self.act1(x)
        else:
            x = self.act1(x)
        x=self.pool1(x)
        x=Dropout(self.drop_conv)
    # Layer 2
        x = self.conv2(x)
        if training:
            x=self.bn2(x)
            x = self.act2(x)
        else:
            x = self.act2(x)
        x = self.pool2(x)
        x = Dropout(self.drop_conv)
    # Layer 3
        x = self.conv3(x)
        if training:
            x=self.bn3(x)
            x = self.act3(x)
        else:
            x = self.act3(x)
        x = self.pool3(x)
        x = Dropout(self.drop_conv)
    # Layer 4
        x = self.conv4(x)
        if training:
            x=self.bn4(x)
            x = self.act4(x)
        else:
            x = self.act4(x)
        x = self.pool4(x)
        x = Dropout(self.drop_conv)
    # Layer 5
        x = UpSampling2D((2, 2),interpolation="bilinear")(x)
        x = self.conv5u(x)
    # Layer 6
        x = UpSampling2D((2, 2),interpolation="bilinear")(x)
        x = self.conv6u(x)
    # Layer 7
        x = UpSampling2D((2, 2),interpolation="bilinear")(x)
        x = self.conv7u(x)
    # Layer 8
        x = UpSampling2D((2, 2),interpolation="bilinear")(x)
        x = self.conv8u(x)
        seg_logits= tf.reshape(tensor=x,shape=(-1, self.img_height * self.img_height, self.number_of_classes))
    ## First to second stage transition
        result_max = tf.argmax(x, 3)  # max result along axis 3
        result_max_labels = tf.equal(result_max, 1)  # object class is 0
        result_max = tf.cast(result_max_labels, tf.float32)
        fcn_pred = tf.reshape(tensor=result_max, shape=(-1, self.img_height, self.img_height, 1))

        return seg_logits,fcn_pred
