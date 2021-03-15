

import tensorflow as tf
import numpy as np
import pickle
from models.cnntools import *
from models.plottools import *
import cv2



""" ---------------------------------------------
Loading and data preparation
"""
def prepare_mask(logit_mask):
    """
    Split the image mask into a foregoround and background mask
    and prepare foreground and background so that it can be used
    with the softmax loss.
    The result are two colums per image
    :param logit_mask: The mask images with [N, w, h, 1]. Values
            should be scaled [0,1]
    :return: tensor with three columns [N, image h*w, c (fg, bg = 2) ]
                N - number of test samples
                h*w - image size
                c - number of classes - two for foreground - background separation.
    """

    # Obtain boolean values for the mask and for the background
    Ytr_class_labels = np.equal(logit_mask, 1)
    Ytr_bg_labels = np.not_equal(logit_mask, 1)

    # Convert the boolean values into floats -- so that
    # computations in cross-entropy loss is correct
    # np.to_float(Ytr_class_labels)
    # np.to_float(Ytr_bg_labels)
    bit_mask_class = Ytr_class_labels.astype(float)
    bit_mask_background = Ytr_bg_labels.astype(float)

    # combine the images along axis 3
    #combined_mask = tf.concat(axis=3, values=[bit_mask_class,
     #                                         bit_mask_background])
    # bit_mask_class - > [N, w, h, 1]
    # bit_mask_background  - > [N, w, h, 1]
    # combined_mask - > [N, w, h, 2]
    combined_mask = np.concatenate((bit_mask_class, bit_mask_background), axis=3) # the object class became 0 here

    number_of_classes = 2
    # flattem them all
   # Ytr_flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, np.product(logit_mask[0].shape), number_of_classes))
    Ytr_flat_labels = combined_mask.reshape([-1, np.product(logit_mask[0].shape), number_of_classes])

    return Ytr_flat_labels

    # Test to verify that the Ytr_flat_labels really contains the flatten bunny data.
    #
    invTest =  Ytr_flat_labels[0]
    invTest = invTest[:,0]
   # invTest =  tf.reshape(tensor=invTest, shape=(-1,64))
    invTest =  invTest.reshape([-1, 64])
    invTest = invTest * 255
    #with tf.device('/cpu:0'):
    #    sess = tf.InteractiveSession()
     #   vis = cv2.cvtColor(invTest.eval(), cv2.COLOR_GRAY2BGR) # to cv mat
    #cv2.imshow("Mask",invTest)
    #cv2.waitKey()

    return Ytr_flat_labels

def prepare_data(filename):
    """
    Load and prepare the data.
    The function loads data from a pickle file and extracts the labels Xtr, Xtr_mask, Xte, and Xte_mask.
    It expects to find the data the same shape with [N, w, h, channels]
    :param filename:
    :return:
    """
    pickle_in = open(filename, "rb")
    data = pickle.load( pickle_in)

    Xtr = data["Xtr"] # RGB image data [N, w, h, channels]
    Ytr = data["Xtr_mask"] # the mask [N, w, h, 1]
    Ytr = Ytr / 255 # Scale every value to either 0 or 1. The values in the value are in a range [0,255]
    Ytr = prepare_mask(Ytr)


    Xte = data["Xte"]
    Yte = data["Xte_mask"]
    Yte = Yte / 255
    Yte = prepare_mask(Yte)

    return Xtr, Ytr, Xte, Yte


def prepare_data_RGBD_pose(filename):
    """
    Load and prepare the data.
    The function loads data from a pickle file and extracts the labels Xtr, Xtr_mask, Xte, and Xte_mask.
    It expects to find the data the same shape with [N, w, h, channels]
    :param filename:
    :return:
    """
    pickle_in = open(filename, "rb")
    data = pickle.load( pickle_in)

    Xtr = data["Xtr"] # RGB image data [N, w, h, channels]
    Xtr_depth = data["Xtr_depth"]
    Xtr_depth = Xtr_depth / 65536 # normalize [0,1]
    Ytr_pose = data["Ytr_pose"][:,0:3]
    Ytr = data["Xtr_mask"] # the mask [N, w, h, 1]
    Ytr = Ytr / 255 # Scale every value to either 0 or 1. The values in the value are in a range [0,255]
    Ytr = prepare_mask(Ytr)


    #test_depth = Xtr[0]
    #cv2.imshow("test", test_depth)
    #cv2.waitKey(1)
    #rows  = test_depth.shape[0]
    #cols = test_depth.shape[1]
    #file = open( "depth_test.csv", "a")
    #for i in range(0,rows):
    #    for j in range(0, cols):
    #        out = str(test_depth[i,j])
    #        file.write(out)
    #    file.write("\n")
    # file.close()


    Xte = data["Xte"]
    Xte_depth = data["Xte_depth"]
    Xte_depth = Xte_depth  / 65536
    Yte_pose = data["Yte_pose"][:, 0:3]
    Yte = data["Xte_mask"]
    Yte = Yte / 255
    Yte = prepare_mask(Yte)

    return [Xtr, Xtr_depth, Ytr, Ytr_pose, Xte, Xte_depth, Yte, Yte_pose]



def prepare_data_RGBD_6DoF(filename):
    """
    Load and prepare the data.
    The function loads data from a pickle file and extracts the labels Xtr, Xtr_mask, Xte, and Xte_mask.
    It expects to find the data the same shape with [N, w, h, channels]
    :param filename:
    :return:
    """
    pickle_in = open(filename, "rb")
    data = pickle.load( pickle_in)

    Xtr = data["Xtr"] # RGB image data [N, w, h, channels]
    Xtr = Xtr/ 255
    Xtr_depth = data["Xtr_depth"]
    Xtr_depth = Xtr_depth / 65536 # normalize [0,1]
    Ytr_pose = data["Ytr_pose"]
    Ytr = data["Xtr_mask"] # the mask [N, w, h, 1]
    Ytr = Ytr / 255 # Scale every value to either 0 or 1. The values in the value are in a range [0,255]
    Ytr = prepare_mask(Ytr)


    #test_depth = Xtr[0]
    #cv2.imshow("test", test_depth)
    #cv2.waitKey(1)
    #rows  = test_depth.shape[0]
    #cols = test_depth.shape[1]
    #file = open( "depth_test.csv", "a")
    #for i in range(0,rows):
    #    for j in range(0, cols):
    #        out = str(test_depth[i,j])
    #        file.write(out)
    #    file.write("\n")
    # file.close()


    Xte = data["Xte"]
    Xte = Xte/255
    Xte_depth = data["Xte_depth"]
    Xte_depth = Xte_depth  / 65536
    Yte_pose = data["Yte_pose"]
    Yte = data["Xte_mask"]
    Yte = Yte / 255
    Yte = prepare_mask(Yte)

    return [Xtr, Xtr_depth, Ytr, Ytr_pose, Xte, Xte_depth, Yte, Yte_pose]
