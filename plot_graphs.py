import pathlib
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
from get_values import *
def plot_stuff(logs,im_width, im_height,batch = False):
    l = len(logs)
    res_path = logs[0]
    if batch:
        res_path = res_path[:-2] + "/_results/"
    else:
        res_path = res_path + "_results/"
    pathlib.Path(res_path).mkdir(parents=True, exist_ok=True)
    seg_train_prec = []
    seg_train_rec = []
    seg_test_prec = []
    seg_test_rec = []
    pose_train_prec = []
    pose_train_rec = []
    pose_test_prec = []
    pose_test_rec = []
    seg_train_loss = []
    seg_test_loss = []
    pose_train_loss = []
    pose_test_loss = []
    set = 0;
    for i in range(l):
    # Get Precision and recall values
        train_prec,train_rec,test_prec,test_rec = \
            get_prec_rec_values(logs[i],im_width,im_height)
        seg_train_prec.append(train_prec);seg_test_prec.append(test_prec)
        seg_train_rec.append(train_rec);seg_test_rec.append(test_rec)

    # Get Loss values
        train_loss,test_loss = get_loss_values(logs[i],im_width,im_height)
        seg_train_loss.append(train_loss);seg_test_loss.append(test_loss)
        if ((i+1) % 10 == 0 and l>10):
            set+=1
            size = len(seg_train_rec[i])
            tt = np.arange(0.0, size, 1)
        # Precision Plots
            plt.clf()
            plt.cla()
            plt.figure()
            plt.ylim(0, 1.0)
            for j in range (i-10,i):
                plt.plot(tt, seg_train_prec[j], label='precision_train')
                plt.plot(tt, seg_test_prec[j], label='precision_test')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.title('Train-segmentation_precision - 128x128')
            plt.legend()
            plt.savefig(res_path + "seg_prec_set%d.png"%set)
            plt.close()
        # Recall Plots
            plt.clf()
            plt.cla()
            plt.figure()
            plt.ylim(0, 1.0)
            for j in range(i - 10, i):
                plt.plot(tt, seg_train_rec[j], label='recall_train')
                plt.plot(tt, seg_test_rec[j], label='recall_test')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.title('Train-segmentation_recall - 128x128')
            plt.legend()
            plt.savefig(res_path + "seg_rec_set%d.png"%set)
            plt.close()

        # Loss Plots
            plt.clf()
            plt.cla()
            plt.figure()
            #plt.ylim(0, 1.0)
            for j in range(i - 10, i):
                plt.plot(tt, seg_train_loss[j], label='training loss')
                plt.plot(tt, seg_test_loss[j], label='testing loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Segmentation Cost - 128x128')
            plt.legend()
            plt.savefig(res_path + "seg_losses_set%d.png"%set)
            plt.close()
    if l<=10:
        # Precision Plots
        plt.clf()
        plt.cla()
        plt.figure()
        plt.ylim(0, 1.0)
        for j in range(0, l):
            size = len(seg_train_prec[j])
            tt = np.arange(1, size+1, 1)
            plt.plot(tt, seg_train_prec[j], label='precision_train_%d'%j)
            plt.plot(tt, seg_test_prec[j], label='precision_test_%d'%j)
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Train-segmentation_precision - 128x128')
        plt.legend()
        plt.savefig(res_path + "seg_prec_set%d.png" % set)
        plt.close()
        # Recall Plots
        plt.clf()
        plt.cla()
        plt.figure()
        plt.ylim(0, 1.0)
        for j in range(0, l):
            size = len(seg_train_rec[j])
            tt = np.arange(1, size+1, 1)
            plt.plot(tt, seg_train_rec[j], label='recall_train_%d'%j)
            plt.plot(tt, seg_test_rec[j], label='recall_test%d'%j)
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Train-segmentation_recall - 128x128')
        plt.legend()
        plt.savefig(res_path + "seg_rec_set%d.png" % set)
        plt.close()

        # Loss Plots
        plt.clf()
        plt.cla()
        plt.figure()
        #plt.ylim(0, 1.0)
        for j in range(0, l):
            size = len(seg_train_loss[j])
            tt = np.arange(1, size+1, 1)
            plt.plot(tt, seg_train_loss[j], label='training loss%d'%j)
            plt.plot(tt, seg_test_loss[j], label='testing loss%d'%j)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Segmentation Cost - 128x128')
        plt.legend()
        plt.savefig(res_path + "seg_losses_set%d.png" % set)
        plt.close()

    # if (os.path.exists(logs[i] + 'train_pose_results.csv')):
        #     with open(logs[i] + 'train_pose_results.csv', 'r') as csvfile:
        #         data = csv.reader(csvfile, delimiter=',')
        #
        #         # retrieves rows of data and saves it as a list of list
        #         x = [row for row in data]
        #
        #         # forces list as array to type cast as int
        #         int_x = np.array(x, float)
        #
        #         column_index = [1, 2, 3]
        #         col = 2
        #         # loops through array
        #         column_values = np.empty(0)
        #         for array in range(len(int_x)):
        #             # gets correct column number
        #             column_values = np.append(column_values, np.array(int_x[array][col - 1]))
        #         size = len(column_values)
        #         column_values_trp = column_values
        #         tt = np.arange(0.0, size, 1)
        #         s_values = sorted(column_values)
        #         pr_t.clear
        #         if size != 0:
        #             for i in range(0, size, int(size / 10)):
        #                 ss_errors = s_values[:i]
        #                 for value in ss_errors:
        #                     if value <= 0.3:
        #                         n = n + 1
        #                 ss_errors.clear()
        #                 if i == 0:
        #                     prec = 1
        #                     n = 0
        #                     pr_t.append(prec)
        #
        #                 else:
        #                     prec = n / i
        #                     n = 0
        #                     pr_t.append(prec)
        #             prec_rec_tr_t = pr_t
        #             t = np.arange(0.0, 1.0, 0.1)
        #             col = 3
        #             # loops through array
        #             n = 0
        #             column_values = np.empty(0)
        #             for array in range(len(int_x)):
        #                 # gets correct column number
        #                 column_values = np.append(column_values, np.array(int_x[array][col - 1]))
        #                 column_values[array] = column_values[array] * math.pi / 180
        #             # for i in range(len(column_values)):
        #             #     column_values[i] = 1- column_values[i]
        #             column_values_trr = column_values
        #             s_r_values = sorted(column_values)
        #             prec_rec.clear()
        #             for i in range(0, size, int(size / 10)):
        #                 ss_errors = s_r_values[:i]
        #                 for value in ss_errors:
        #                     if value <= 0.11:
        #                         n = n + 1
        #                 ss_errors.clear()
        #                 if i == 0:
        #                     prec = 1
        #                     n = 0
        #                     prec_rec.append(prec)
        #
        #                 else:
        #                     prec = n / i
        #                     n = 0
        #                     prec_rec.append(prec)
        #             prec_rec_tr_r = prec_rec
        #
        # if (os.path.exists(logs[i] + 'train_loss.csv')):
        #     with open(logs[i] + 'train_loss.csv', 'r') as csvfile:
        #         data = csv.reader(csvfile, delimiter=',')
        #         # retrieves rows of data and saves it as a list of list
        #         x = [row for row in data]
        #
        #         # forces list as array to type cast as int
        #         int_x = np.array(x, float)
        #
        #         col = 2
        #         # loops through array
        #         column_values = np.empty(0)
        #         for array in range(len(int_x)):
        #             # gets correct column number
        #             column_values = np.append(column_values, np.array(int_x[array][col - 1]))
        #         size = len(column_values)
        #         clm_val_seg_loss = column_values
        #
        #         col = 3
        #         # loops through array
        #         column_values = np.empty(0)
        #         for array in range(len(int_x)):
        #             # gets correct column number
        #             column_values = np.append(column_values, np.array(int_x[array][col - 1]))
        #         size = len(column_values);
        #         clm_val_tr_loss = column_values;
        #
        #         col = 4
        #         # loops through array
        #         column_values = np.empty(0)
        #         for array in range(len(int_x)):
        #             # gets correct column number
        #             column_values = np.append(column_values, np.array(int_x[array][col - 1]))
        #         size = len(column_values)
        #         clm_val_r_loss = column_values
        # if (os.path.exists(logs[i] + 'test_loss.csv')):
        #     with open(logs[i] + 'test_loss.csv', 'r') as csvfile:
        #         data = csv.reader(csvfile, delimiter=',')
        #         # retrieves rows of data and saves it as a list of list
        #         x = [row for row in data]
        #
        #         # forces list as array to type cast as int
        #         int_x = np.array(x, float)
        #
        #         col = 2
        #         # loops through array
        #         column_values = np.empty(0)
        #         for array in range(len(int_x)):
        #             # gets correct column number
        #             column_values = np.append(column_values, np.array(int_x[array][col - 1]))
        #             column_values[array] = column_values[array] / (im_width * im_height)
        #         size = len(column_values)
        #         clm_val_segt_loss = column_values
        #
        #         col = 3
        #         # loops through array
        #         column_values = np.empty(0)
        #         for array in range(len(int_x)):
        #             # gets correct column number
        #             column_values = np.append(column_values, np.array(int_x[array][col - 1]))
        #         size = len(column_values);
        #         clm_val_trt_loss = column_values
        #
        #         col = 4
        #         # loops through array
        #         column_values = np.empty(0)
        #         for array in range(len(int_x)):
        #             # gets correct column number
        #             column_values = np.append(column_values, np.array(int_x[array][col - 1]))
        #         size = len(column_values);
        #         clm_val_rt_loss = column_values
        #         size = len(column_values);
        #         tt = np.arange(0.0, size, 1)
        #         plt.clf()
        #         plt.cla()
        #         if (len(clm_val_rt_loss) > 0):
        #             plt.figure()
        #             # plt.ylim(0,50)
        #             plt.plot(tt, clm_val_tr_loss, 'r--', label='train_translation')
        #             plt.plot(tt, clm_val_trt_loss, 'g--', label='test_translation')
        #             plt.plot(tt, clm_val_r_loss, 'b--', label='train_rotation')
        #             plt.plot(tt, clm_val_rt_loss, 'y--', label='test_rotation')
        #             plt.xlabel('Epoch')
        #             plt.ylabel('Loss')
        #             plt.title('Cost of training Pose - 128x128')
        #             plt.legend()
        #             plt.savefig(res_path + "Cost_train_pose.png");
        #             plt.close()
        #
        #             tt = np.arange(0.0, size, 1)
        #             plt.clf()
        #             plt.cla()
        #             plt.figure()
        #             # plt.ylim(0,50)
        #             plt.plot(tt, clm_val_segt_loss, 'r--', label='test_segmentation')
        #             plt.plot(tt, clm_val_seg_loss, 'g--', label='train_segmentation')
        #             plt.xlabel('Epoch')
        #             plt.ylabel('Loss')
        #             plt.title('Cost of training Segmentation - 128x128')
        #             plt.legend()
        #             plt.savefig(res_path + "Cost_train_seg.png");
        #             plt.close()
        #
        # if (os.path.exists(logs[i] + 'test_pose_results.csv')):
        #     with open(logs[i] + 'test_pose_results.csv', 'r') as csvfile:
        #         data = csv.reader(csvfile, delimiter=',')
        #
        #         # retrieves rows of data and saves it as a list of list
        #         x = [row for row in data]
        #
        #         # forces list as array to type cast as int
        #         int_x = np.array(x, float)
        #
        #         col = 2
        #         n = 0
        #         # loops through array
        #         column_values = np.empty(0)
        #         for array in range(len(int_x)):
        #             # gets correct column number
        #             column_values = np.append(column_values, np.array(int_x[array][col - 1]))
        #         size = len(column_values)
        #         tt = np.arange(0.0, size, 1)
        #         plt.clf()
        #         plt.cla()
        #         plt.figure()
        #         plt.plot(tt, column_values, label='test')
        #         plt.plot(tt, column_values_trp, label='train')
        #         plt.ylim(top=1.4)
        #         plt.xlim(right=300)
        #         plt.xlabel('Epoch')
        #         plt.ylabel('Accuracy')
        #         plt.title('Train-pose_Accuracy - 128x128')
        #         plt.legend()
        #         plt.savefig(res_path + "Accuracy_train_pose.png")
        #         plt.close()
        #         s_values = sorted(column_values)
        #         pr_t.clear()
        #         if size != 0:
        #             for i in range(0, size, int(size / 10)):
        #                 ss_errors = s_values[:i]
        #                 for value in ss_errors:
        #                     if value <= 0.3:
        #                         n = n + 1
        #                 ss_errors.clear()
        #                 if i == 0:
        #                     prec = 1
        #                     n = 0
        #                     pr_t.append(prec)
        #
        #                 else:
        #                     prec = n / i
        #                     n = 0
        #                     pr_t.append(prec)
        #
        #             t = np.arange(0.0, 1.0, 0.1)
        #             plt.clf()
        #             plt.cla()
        #             plt.figure()
        #             plt.plot(t, pr_t)
        #             plt.plot(t, pr_t, label='test')
        #             plt.plot(t, prec_rec_tr_t, label='train')
        #             plt.xlabel('recall')
        #             plt.ylabel('precision')
        #             plt.title('Precision-recall (train-pose) -Cutoff = 0.3 m - 128x128')
        #             plt.legend()
        #             plt.savefig(res_path + "t_prec_rec.png")
        #             plt.close()
        #             col = 3
        #             # loops through array
        #             n = 0
        #             column_values = np.empty(0)
        #             for array in range(len(int_x)):
        #                 # gets correct column number
        #                 column_values = np.append(column_values, np.array(int_x[array][col - 1]))
        #                 column_values[array] = column_values[array] * math.pi / 180
        #             # for i in range(len(column_values)):
        #             #     column_values[i] = 1- column_values[i]
        #             plt.clf()
        #             plt.cla()
        #             plt.figure()
        #             plt.plot(tt, column_values, label='test')
        #             plt.plot(tt, column_values_trr, label='train')
        #             plt.ylim(top=1.4)
        #             plt.xlim(right=300)
        #             plt.xlabel('Epoch')
        #             plt.ylabel('Accuracy')
        #             plt.title('Accuracy (train-rotation_rms - 128x128)')
        #             plt.legend()
        #             plt.savefig(res_path + "Accuracy_train_rotation.png")
        #             plt.close()
        #             s_r_values = sorted(column_values)
        #             prec_rec.clear()
        #             for i in range(0, size, int(size / 10)):
        #                 ss_errors = s_r_values[:i]
        #                 for value in ss_errors:
        #                     if value <= 0.11:
        #                         n = n + 1
        #                 ss_errors.clear()
        #                 if i == 0:
        #                     prec = 1
        #                     n = 0
        #                     prec_rec.append(prec)
        #
        #                 else:
        #                     prec = n / i
        #                     n = 0
        #                     prec_rec.append(prec)
        #
        #             plt.clf()
        #             plt.cla()
        #             plt.figure()
        #             plt.plot(t, prec_rec, label='test')
        #             plt.plot(t, prec_rec_tr_r, label='train')
        #             plt.xlabel('recall')
        #             plt.ylabel('precision')
        #             plt.title('Precision-recall (train-rotation). Cutoff = 0.11 radian - 128x128')
        #             plt.legend()
        #             plt.savefig(logs[i] + "r_prec_rec.png")
        #             plt.close()

                # col = 4
                # n = 0
                # # loops through array
                # column_values = np.empty(0)
                # for array in range(len(int_x)):
                #     # gets correct column number
                #     column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                # clm_prect_val=column_values
                # col = 5
                # n = 0
                # # loops through array
                # column_values = np.empty(0)
                # for array in range(len(int_x)):
                #     # gets correct column number
                #     column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                # clm_rect_val = column_values

