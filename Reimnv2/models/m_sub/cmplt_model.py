# import the necessary packages
from .SM11 import *
from .Pose_net_t import *
from .Pose_net_r import *
class cmplt_model(keras.Model):
    """
    Model sub-class
    """
    def __init__(self, number_classes=2,img_height=128,stage2 = False,rot = False):
        #Call the parent constructor
        super(cmplt_model,self).__init__()
        self.number_of_classes = number_classes
        self.img_height = img_height
        self.stage2 = stage2
        self.rot = rot
        self.seg_model = SM11(self.number_of_classes,self.img_height,trainable=not self.stage2)
        self.pose_model_t = Pose_net_t(img_height = self.img_height, trainable = self.stage2)
        self.pose_model_r = Pose_net_r(img_height = self.img_height, trainable = self.stage2)


    def call(self, input_img, drop_conv, depth = None, rot = False, training = False):
        self.rot = rot
        seg_logits, fcn_pred = self.seg_model(input_img, drop_conv)
        if training:
            if self.stage2 and self.rot:
                pose_r = self.get_pose_r(fcn_pred, depth, drop_conv, training=self.rot)
                return pose_r
            elif self.stage2 and not self.rot:
                pose_t = self.get_pose_r(fcn_pred, depth, drop_conv, training = not self.rot)
                return pose_t
            else:
                return seg_logits
        else:
            pose_r = self.get_pose_r(fcn_pred, depth, drop_conv, training=self.rot)
            pose_t = self.get_pose_r(fcn_pred, depth, drop_conv, training=not self.rot)
            return seg_logits, pose_t, pose_r

    def get_pose_t(self,fcn_pred, depth, drop_conv, training = False):

         return self.pose_model_t(fcn_pred, depth, drop_conv, training=False)

    def get_pose_r(self,fcn_pred, depth, drop_conv, training = False):

        return self.pose_model_r(fcn_pred, depth, drop_conv, training = False)