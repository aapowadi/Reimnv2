"""
This script provides helper functions to evaluate the accuracy of the pose prediction

Package requirements:
- NumPy
- pyquaternion
- datetime


Iowa State University
May 8, 2019
aapowadi@iastate.edu

June 23, 2019, RR
- Flipped the quaternion vector from (x, y, z, w) -> (w, x, y, z)
  tfquaternion and other python quaternions prefer this way.

"""
import numpy as np
from pyquaternion import Quaternion
import datetime
import matplotlib.pyplot as plt
import math

def eval_pose_accuracy(t_gt, t_pr, q_gt, q_pr, filepath = ""):
    """
    Evaluate the pose accuracy by comparing prediction to ground truth
    :param t_gt: (array)
        An array with the translation ground truth data as vector (x, y, z); the array is of size [N, 3].
    :param t_pr: (array)
        An array with the translation prediction data as vector (x, y, z); the array is of size [N, 3].
    :param q_gt: (array)
        An array with the orientation ground truth data as quaternion (qx, qy, qz, qw); the array is of size [N, 4].
        Note that qw is the last component.
    :param q_pr:
        An array with the orientation predictions as quaternion (qx, qy, qz, qw); the array is of size [N, 4].
        Note that qw is the last component.
    :param path_and_file: (str)
        If a string with a path and filename is given, the function will write all results into a cvs file.
    :return: t_mean (float), the root mean translation error
             t_acc (array), an array with all individual errors.
             q_mean (float), the average orientation error in degree
             q_acc (array), an array with all individual orientation errors.
    """

    # write a header
    file = open(filepath+"test_pose_acc.csv", "w")
    #file.write("pose rms,orient_rms\n")
    file.close()
    t_mean, t_acc, cs_dist, st_errors, pose_prec_rec = eval_translation_accuracy(t_gt, t_pr, filepath)
    q_mean, q_acc, sr_errors, rot_prec_rec = evaluate_orientation_accuracy(q_gt, q_pr, cs_dist, filepath)
    file = open(filepath + "test_pose_acc.csv", "a")
    for i in range(len(sr_errors)):
        file.write(str(st_errors[i]) + "," + str(sr_errors[i]) + "\n")
    file.close()
    # write the results to a file
    #write_results(t_acc, q_acc, path_and_file+"accuracy.csv")

    return t_mean, t_acc, q_mean, q_acc


def eval_translation_accuracy(t_gt, t_pr, filepath):
    """
    Evaluate the accuracy of the predicted translation by comparing it to the ground truth

    The function uses the root mean square error to calcualte the error, which is the average ground truth error.
    :param t_gt: (array)
            The ground truth translation (x, y, z) as array of size [N, 3].
    :param t_pr: (array)
            The predicted translation (x, y, z) as array of size [N, 3].
    :return: (float) the root mean square error as float.
             (array) an array of all individual errors
    """
    # Check the sizes
    if t_gt.shape[0] != t_pr.shape[0]:
        print("ERROR - translation prediction size != ground truth size.\nBoth must have the same size.")
        return 0.0

    size = t_gt.shape[0]
    distance_error = 0
    all_errors = []
    all_accu=0
    all_cdist=[]
    z=[0,0,0]
    prec_rec=[]
    n=0;

    for i in range(0,size):
        gt = t_gt[i]
        pr = t_pr[i]
        dist = np.linalg.norm(gt - pr) # Euclidian distance
        cam_dist = np.linalg.norm(gt - z)
        distance_error = distance_error + dist
        all_errors.append(dist)
        all_cdist.append(cam_dist)
    distance_error = distance_error / size
    s_errors=sorted(all_errors)
    cs_dist=sorted(all_cdist)
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(cs_dist, s_errors)
    plt.xlabel('Camera Distance')
    plt.ylabel('Delta t')
    plt.title('t_error')
    #plt.show()
    plt.savefig(filepath + "t_error.png");
    plt.close()
    file = open(filepath + "pose_prec_rec.csv", "w")
    for i in range(0,size, int(size/10)):
        ss_errors=s_errors[:i]
        for value in ss_errors:
            if value <= 0.3:
                n=n+1;
        ss_errors.clear();
        if i == 0:
            prec = 1;
            n = 0;
            prec_rec.append(prec);

        else:
            prec = n/i;
            n = 0;
            prec_rec.append(prec);
        file.write(str(prec)+"\n")
    file.close()
    t=np.arange(0.0, 1.0, 0.1)

    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(t, prec_rec)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Translation')
    #plt.show()
    plt.savefig(filepath + "t_prec_rec-cutoff=0.3 meters.png");
    plt.close()


    return distance_error, all_accu, cs_dist, s_errors, prec_rec


def evaluate_orientation_accuracy(q_gt, q_pr, cs_dist, filepath):
    """
    Calculate the orientation accuracy.

    The function calculates the quaternion difference between q_gt and q_pr and extracts the angle delta.
    :param q_gt: (array) an array with the ground truth quaternion components (qw, qx, qy, qz) as array of size [N, 4]
    :param q_pr: (array) an array with the prediction quaternion components (qw, qx, qy, qz) as array of size [N, 4]
    :return: float) the average distance error in DEGREE as float.
             (array) an array of all individual errors
    """

    # Check the sizes
    if q_gt.shape[0] != q_pr.shape[0]:
        print("ERROR - orientation prediction size != ground truth size.\nBoth must have the same size.")
        return 0.0

    size = q_gt.shape[0]
    orientation_error = 0
    all_acc=0;
    ac=0;
    ex=0;
    all_errors = []
    s_errors=[]
    ss_errors=[]
    prec_rec=[]

    for i in range(0, size):
        q0 = q_gt[i]
        q1 = q_pr[i]
        if(q0[0]==0 and q0[1] ==0 and q0[2]==0 and q0[3]==0):
            continue
        if (q1[0] == 0 and q1[1]==0 and q1[2]==0 and q1[3]==0):
            continue
        # get quaternions. Note that pyquaternion uses
        # (w, x, y, z), http://kieranwynn.github.io/pyquaternion/#explicitly-by-element
        quat0 = Quaternion(q0[3], q0[0], q0[1], q0[2])
        quat1 = Quaternion(q1[3], q1[0], q1[1], q1[2])

        # Calculate the difference
        delta_q = quat0 * quat1.inverse
        orientation_error = orientation_error + delta_q.degrees
        all_errors.append(abs(delta_q.degrees* math.pi / 180))
    s_errors = sorted(all_errors)
    orientation_error = orientation_error/ size
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(cs_dist, s_errors)
    plt.xlabel('Camera Distance')
    plt.ylabel('Delta Orient')
    plt.title('R_error')
    #plt.show()
    plt.savefig(filepath + "r_error.png");
    plt.close()
    file = open(filepath + "rot_prec_rec.csv", "w")
    for i in range(0,size, int(size/10)):
        ss_errors=s_errors[:i]
        for value in ss_errors:
            if value <= 0.11:
                n=n+1;
        ss_errors.clear();
        if i == 0:
            prec = 1;
            n = 0;
            prec_rec.append(prec);

        else:
            prec = n/i;
            n = 0;
            prec_rec.append(prec);
        file.write(str(prec)+"\n")
    file.close()
    t=np.arange(0.0, 1.0, 0.1)
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(t, prec_rec)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Orientation')
    #plt.show()
    plt.savefig(filepath + "r_prec_rec-cutoff = 0.11 radians.png");
    plt.close()
    all_acc = 1 - all_acc;
    return orientation_error, all_acc, s_errors, prec_rec


def write_results(t_array, q_array, file):
    """
    Write all individual results index aligned into a file.
    :param t_array: (array)
        An array with translation errors of size [N]
    :param q_array: (array)
        A array with orientation errors of size [N]
    :param file: (str)
        A string with a path and a filename. No file will be written if empty
    :return:
    """
    if len(file) == 0:
        return

    file = open(file, "w")
    file.write("idx,t_acc,q_acc\n")

    size = t_array.shape

    out = str(1) + "," + str(t_array) + "," + str(q_array)  + "\n"
    file.write(out)

    file.close()



def write_report(filename, epoch,  seg_loss, seg_pr, seg_re, t_loss, q_loss, t_rms, q_rms):
    """
    Write the results of an evaluation run into a file.

    :param filename: (str), the name of the file as string
    :param epoch: (int) The current training epoch. Indicats after how many epoches of training this evaluation runs.
    :param seg_loss: (float) segmentation loss value
    :param seg_pr: (float) segmentation precision value
    :param seg_re: (float) segmentation recall value
    :param t_loss: (float) translation training loss
    :param q_loss: (float) quaternion training  loss
    :param t_rms: (float) translation root mean square error
    :param q_rms: (float) orientation average error.
    :return:
    """

    if len(filename) == 0:
        return

    try:
        with open(filename, 'r') as fh:
            pass
    except FileNotFoundError:
        # write a header
        file = open(filename, "w")
        file.write("Evaluation results\n")
        file.write("Date,epoch training,seg. loss,seg. pr,seg. re,trans loss,orient loss,trans rms,orient mean\n")
        file.close()

    now = datetime.datetime.now()
    date = str(now.month) +"\\" + str(now.day) + "\\" + str(now.year) + "-" + str(now.hour) + ":" + str(now.minute) + ":" + str(now.second)

    file = open(filename, "a")

    out = date + "," + str(epoch) + "," + str(seg_loss) + "," + str(seg_pr)  + "," +  str(seg_re) + "," +  str(t_loss) \
          + ","  + str(q_loss) + "," + str(t_rms) + "," + str(q_rms) + "\n"

    file.write(out)

    #file.write("--------------------------------------------------------------\n")
    #file.write("Results after epochs training," + str(epoch) + "\n" )
    #file.write("Date," + str(now.month) +"\\" + str(now.day) + "\\" + str(now.year) + "-" + str(now.hour) + "_" + str(now.min) + "_" + str(now.second) + "\n")
    #file.write("seg. loss," + str(seg_loss) + "\n")
    #file.write("seg. precision," + str(seg_pr) + "\n")
    #file.write("seg. recall," + str(seg_re) + "\n")
    #file.write("translation loss," + str(t_loss) + "\n")
    #file.write("orientation loss," + str(q_loss) + "\n")
    #file.write("translation rms," + str(t_rms) + "\n")
    #file.write("orientation mean," + str(q_rms) + "\n")

    file.close()


def write_data_for_3DPoseEval(filename, pose_t, pose_q, pr_t, pr_q, start_idx):
    """
    Write the pose data into a csv file.

    Organized as dx,x,y,z,qx,qy,qz,qw,pr_x,pr_y,pr_z,pr_qx,pr_qy,pr_qz,pr_qw
    This dataset is for the Pose Evaluation 3D renderer, it can read this file and
    show the pose as 3D rendering.
    :param filename: (str)
        A path and filename for the csv file
    :param pose_t: (array)
        An array with translations (x, y, z) as array of size [N, 3]
    :param pose_q: (array)
        An array with orientations as quaternion (qw, qx, qy, qz) as array of size [N, 4]
    :param pr_t: (array)
        An array with translation predictions (x, y, z) as array of size [N, 3]
    :param pr_q: (array)
        An array with orientation predictions as quaternion (qw, qx, qy, qz) as array of size [N, 4]
    :param start_idx: (int)
        The start index for this evaluation batch. It should align with the image index, as long as the
        data is not shuffled.
    :return:
    """
    if len(filename) == 0:
        return

    if start_idx == 0:
        file = open(filename, "w")
           # file = open(filename, "w")
        file.write("idx,x,y,z,qx,qy,qz,qw,pr_x,pr_y,pr_z,pr_qx,pr_qy,pr_qz,pr_qw\n")
        file.close()

    size = pose_t.shape[0]

    file = open(filename, "a")
    for i in range(0,size):
        index = start_idx + i
        out = str(index) + "," + str(pose_t[i][0]) + "," + str(pose_t[i][1]) + "," + str(pose_t[i][2]) + "," + str(pose_q[i][0]) + "," + str(pose_q[i][1]) \
              + "," + str(pose_q[i][2]) + "," + str(pose_q[i][3]) + "," + str(pr_t[i][0])+ "," + str(pr_t[i][1]) + "," + str(pr_t[i][2])\
              + "," + str(pr_q[i][0])  + "," + str(pr_q[i][1])  + "," + str(pr_q[i][2])  + "," + str(pr_q[i][3])+ "\n"
        file.write(out)
    file.close()