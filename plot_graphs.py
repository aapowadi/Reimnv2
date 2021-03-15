from ExpList_syn import *
from get_gra import *
import sys, getopt

class plot_graphs:


    def __init__(self, exs):
        self._descrip=[]
        for i in range(len(exs)):
            self._descrip.append(exs[i])

        # check the dictionary for completeness
        for i in range(len(self._descrip)):
            err = self.__check_dict__(self._descrip[i],i)

    def plot_stuff(self):
        l = len(self._descrip)
        for i in range(l):
            path = self.saver_log_folder + "results"
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)
            x = []
            y = []
            pr_t = []
            prec_rec = []
            n = 0;
        if (os.path.exists(self.saver_log_folder + 'train_pose_results.csv')):
            with open(self.saver_log_folder + 'train_pose_results.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # retrieves rows of data and saves it as a list of list
                x = [row for row in data]

                # forces list as array to type cast as int
                int_x = np.array(x, float)

                column_index = [1, 2, 3]
                col = 2
                # loops through array
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                size = len(column_values)
                column_values_trp = column_values
                tt = np.arange(0.0, size, 1)
                s_values = sorted(column_values)
                pr_t.clear
                if size != 0:
                    for i in range(0, size, int(size / 10)):
                        ss_errors = s_values[:i]
                        for value in ss_errors:
                            if value <= 0.3:
                                n = n + 1
                        ss_errors.clear()
                        if i == 0:
                            prec = 1
                            n = 0
                            pr_t.append(prec)

                        else:
                            prec = n / i
                            n = 0
                            pr_t.append(prec)
                    prec_rec_tr_t = pr_t
                    t = np.arange(0.0, 1.0, 0.1)
                    col = 3
                    # loops through array
                    n = 0
                    column_values = np.empty(0)
                    for array in range(len(int_x)):
                        # gets correct column number
                        column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                        column_values[array] = column_values[array] * math.pi / 180
                    # for i in range(len(column_values)):
                    #     column_values[i] = 1- column_values[i]
                    column_values_trr = column_values
                    s_r_values = sorted(column_values)
                    prec_rec.clear()
                    for i in range(0, size, int(size / 10)):
                        ss_errors = s_r_values[:i]
                        for value in ss_errors:
                            if value <= 0.11:
                                n = n + 1
                        ss_errors.clear()
                        if i == 0:
                            prec = 1
                            n = 0
                            prec_rec.append(prec)

                        else:
                            prec = n / i
                            n = 0
                            prec_rec.append(prec)
                    prec_rec_tr_r = prec_rec
        if (os.path.exists(self.saver_log_folder + 'seg_results.csv')):
            with open(self.saver_log_folder + 'seg_results.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')
                # retrieves rows of data and saves it as a list of list
                x = [row for row in data]

                # forces list as array to type cast as int
                int_x = np.array(x, float)

                col = 2
                # loops through array
                n = 0
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                # for i in range(len(column_values)):
                #     column_values[i] = 1- column_values[i]
                clm_prec_values = column_values
                col = 3
                # loops through array
                n = 0
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                # for i in range(len(column_values)):
                #     column_values[i] = 1- column_values[i]
                clm_rec_values = column_values

                col = 4
                # loops through array
                n = 0
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                # for i in range(len(column_values)):
                #     column_values[i] = 1- column_values[i]
                clm_prect_values = column_values

                col = 5
                # loops through array
                n = 0
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                # for i in range(len(column_values)):
                #     column_values[i] = 1- column_values[i]
                clm_rect_values = column_values

                size = len(column_values)
                tt = np.arange(0.0, size, 1)
                plt.clf()
                plt.cla()
                plt.figure()
                plt.ylim(0, 1.0)
                # plt.plot(tt, clm_prect_val, label='precision_test')
                # plt.plot(tt, clm_rect_val, label='recall_test')
                plt.plot(tt, clm_prec_values, label='precision_train')
                plt.plot(tt, clm_prect_values, label='precision_test')
                plt.plot(tt, clm_rec_values, label='recall_train')
                plt.plot(tt, clm_rect_values, label='recall_test')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Train-segmentation_Accuracy - 64x64')
                plt.legend()
                plt.savefig(self.saver_log_folder + "results/Accuracy_segmentation.png")
                plt.close()

        if (os.path.exists(self.saver_log_folder + 'train_loss.csv')):
            with open(self.saver_log_folder + 'train_loss.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')
                # retrieves rows of data and saves it as a list of list
                x = [row for row in data]

                # forces list as array to type cast as int
                int_x = np.array(x, float)

                col = 2
                # loops through array
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                size = len(column_values)
                clm_val_seg_loss = column_values

                col = 3
                # loops through array
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                size = len(column_values);
                clm_val_tr_loss = column_values;

                col = 4
                # loops through array
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                size = len(column_values)
                clm_val_r_loss = column_values
        if (os.path.exists(self.saver_log_folder + 'test_loss.csv')):
            with open(self.saver_log_folder + 'test_loss.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')
                # retrieves rows of data and saves it as a list of list
                x = [row for row in data]

                # forces list as array to type cast as int
                int_x = np.array(x, float)

                col = 2
                # loops through array
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                    column_values[array] = column_values[array] / (self.im_width * self.im_height)
                size = len(column_values)
                clm_val_segt_loss = column_values

                col = 3
                # loops through array
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                size = len(column_values);
                clm_val_trt_loss = column_values

                col = 4
                # loops through array
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                size = len(column_values);
                clm_val_rt_loss = column_values
                size = len(column_values);
                tt = np.arange(0.0, size, 1)
                plt.clf()
                plt.cla()
                if (len(clm_val_rt_loss) > 0):
                    plt.figure()
                    # plt.ylim(0,50)
                    plt.plot(tt, clm_val_tr_loss, 'r--', label='train_translation')
                    plt.plot(tt, clm_val_trt_loss, 'g--', label='test_translation')
                    plt.plot(tt, clm_val_r_loss, 'b--', label='train_rotation')
                    plt.plot(tt, clm_val_rt_loss, 'y--', label='test_rotation')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Cost of training Pose - 64x64')
                    plt.legend()
                    plt.savefig(self.saver_log_folder + "results/Cost_train_pose.png");
                    plt.close()

                    tt = np.arange(0.0, size, 1)
                    plt.clf()
                    plt.cla()
                    plt.figure()
                    # plt.ylim(0,50)
                    plt.plot(tt, clm_val_segt_loss, 'r--', label='test_segmentation')
                    plt.plot(tt, clm_val_seg_loss, 'g--', label='train_segmentation')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Cost of training Segmentation - 64x64')
                    plt.legend()
                    plt.savefig(self.saver_log_folder + "results/Cost_train_seg.png");
                    plt.close()

        if (os.path.exists(self.saver_log_folder + 'test_pose_results.csv')):
            with open(self.saver_log_folder + 'test_pose_results.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # retrieves rows of data and saves it as a list of list
                x = [row for row in data]

                # forces list as array to type cast as int
                int_x = np.array(x, float)

                col = 2
                n = 0
                # loops through array
                column_values = np.empty(0)
                for array in range(len(int_x)):
                    # gets correct column number
                    column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                size = len(column_values)
                tt = np.arange(0.0, size, 1)
                plt.clf()
                plt.cla()
                plt.figure()
                plt.plot(tt, column_values, label='test')
                plt.plot(tt, column_values_trp, label='train')
                plt.ylim(top=1.4)
                plt.xlim(right=300)
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Train-pose_Accuracy - 64x64')
                plt.legend()
                plt.savefig(self.saver_log_folder + "results/Accuracy_train_pose.png")
                plt.close()
                s_values = sorted(column_values)
                pr_t.clear()
                if size != 0:
                    for i in range(0, size, int(size / 10)):
                        ss_errors = s_values[:i]
                        for value in ss_errors:
                            if value <= 0.3:
                                n = n + 1
                        ss_errors.clear()
                        if i == 0:
                            prec = 1
                            n = 0
                            pr_t.append(prec)

                        else:
                            prec = n / i
                            n = 0
                            pr_t.append(prec)

                    t = np.arange(0.0, 1.0, 0.1)
                    plt.clf()
                    plt.cla()
                    plt.figure()
                    plt.plot(t, pr_t)
                    plt.plot(t, pr_t, label='test')
                    plt.plot(t, prec_rec_tr_t, label='train')
                    plt.xlabel('recall')
                    plt.ylabel('precision')
                    plt.title('Precision-recall (train-pose) -Cutoff = 0.3 m - 64x64')
                    plt.legend()
                    plt.savefig(self.saver_log_folder + "results/t_prec_rec.png")
                    plt.close()
                    col = 3
                    # loops through array
                    n = 0
                    column_values = np.empty(0)
                    for array in range(len(int_x)):
                        # gets correct column number
                        column_values = np.append(column_values, np.array(int_x[array][col - 1]))
                        column_values[array] = column_values[array] * math.pi / 180
                    # for i in range(len(column_values)):
                    #     column_values[i] = 1- column_values[i]
                    plt.clf()
                    plt.cla()
                    plt.figure()
                    plt.plot(tt, column_values, label='test')
                    plt.plot(tt, column_values_trr, label='train')
                    plt.ylim(top=1.4)
                    plt.xlim(right=300)
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.title('Accuracy (train-rotation_rms - 64x64)')
                    plt.legend()
                    plt.savefig(self.saver_log_folder + "results/Accuracy_train_rotation.png")
                    plt.close()
                    s_r_values = sorted(column_values)
                    prec_rec.clear()
                    for i in range(0, size, int(size / 10)):
                        ss_errors = s_r_values[:i]
                        for value in ss_errors:
                            if value <= 0.11:
                                n = n + 1
                        ss_errors.clear()
                        if i == 0:
                            prec = 1
                            n = 0
                            prec_rec.append(prec)

                        else:
                            prec = n / i
                            n = 0
                            prec_rec.append(prec)

                    plt.clf()
                    plt.cla()
                    plt.figure()
                    plt.plot(t, prec_rec, label='test')
                    plt.plot(t, prec_rec_tr_r, label='train')
                    plt.xlabel('recall')
                    plt.ylabel('precision')
                    plt.title('Precision-recall (train-rotation). Cutoff = 0.11 radian - 64x64')
                    plt.legend()
                    plt.savefig(self.saver_log_folder + "results/r_prec_rec.png")
                    plt.close()

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

    def __check_dict__(self, dict,n):
        """Check if all keys are present

                The function compares the given keys with a set of expected keys and tries to fix them if
                possible.

                :param dict: (dict)
                    Dictionary with all keys.
                :return: (int)
                    An error code.
                    0 - no errors
                    1 - some solveable errors
                    2 - critical error - abort
                """
        error_level = 0

        expected_keys = ["train_dataset", "test_dataset", "solver", "model", "num_iterations", "debug_output",
                         "log_path","log_file", "train", "eval", "test", "proof", "restore_file", "quat_used", "plot_title",
                         "learning_rate", "label", "stage2", "cont", "drp_cnv", "drp_pose", "batch_size"]

        keys = dict.keys()

        # search for all keys.
        for i in keys:
            for j in expected_keys:
                if i == j:
                    expected_keys.remove(i)
                    break

        # Check the missing variables and try to fix them
        if len(expected_keys) > 0:
            print("WARNING - Not all variables have been set. Miss the following variables")
            for i in expected_keys:
                print(i + ", ")

            for i in expected_keys:
                if i == "train_dataset":
                    print("CRITICAL ERROR - Training dataset is missing!")
                    error_level = 2
                elif i == "test_dataset":
                    print("Test dataset is missing, disable tests")
                    self._descrip["test"] = False
                elif i == "solver":
                    print("CRITICAL ERROR - Solver is missing!")
                    error_level = 2
                elif i == "model":
                    print("CRITICAL ERROR - Model is missing!")
                    error_level = 2
                elif i == "num_iterations":
                    print("num_iterations is missing! Set to 100")
                    self._descrip[n]["num_iterations"] = 100
                elif i == "batch_size":
                    print("batch_size is missing! Set to 128")
                    self._descrip[n]["batch_size"] = 128
                elif i == "debug_output":
                    print("debug_output of iteratrion is missing! Set to False")
                    self._descrip[n]["debug_output"] = False
                elif i == "log_path":
                    print("log_path is missing! Set to ./logs/temp/")
                    self._descrip[n]["log_path"] = "./logs/temp/"
                elif i == "log_file":
                    print("log_file is missing! Set to idiot")
                    self._descrip[n]["log_file"] = "idiot"
                elif i == "stage2":
                    print("Stage2 is not set")
                    self._descrip[n]["stage2"] = False
                elif i == "cont":
                    print("Continuation is not set")
                    self._descrip[n]["cont"] = False
                elif i == "drp_cnv":
                    print("Dropout for 1st stage not set so, it is set to 1.0")
                    self._descrip[n]["drp_cnv"] = 1.0
                elif i == "drp_pose":
                    print("Dropout for 2nd stage not set so, it is set to 1.0")
                    self._descrip[n]["drp_pose"] = 1.0
                elif i == "train":
                    print("train is missing! Set to True")
                    self._descrip[n]["train"] = True
                elif i == "eval":
                    print("eval is missing! Set to True")
                    self._descrip[n]["eval"] = True
                elif i == "test":
                    print("test is missing! Set to False")
                    self._descrip[n]["test"] = False
                elif i == "proof":
                    print("Proof is missing! Set to False")
                    self._descrip[n]["test"] = False
                elif i == "restore_file":
                    print("restore_file is missing! Set empty string")
                    self._descrip[n]["restore_file"] = ""
                elif i == "quat_used":
                    print("quat_used is missing! Set to True")
                    self._descrip[n]["restore_file"] = True
                elif i == "plot_title":
                    print("plot_title is missing! Set to No Title")
                    self._descrip[n]["plot_title"] = "No title"
                elif i == "learning_rate":
                    print("learning_rate is missing! Set to 0.001")
                    self._descrip[n]["learning_rate"] = 0.001
                elif i == "label":
                    self._descrip[n]["label"] = "Unlabeled experiment"

        return error_level


def main(argv):
    ex = []
    try:
      opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
      print('ex_imp.py -i <exp_name>')
      sys.exit(2)
    l = len(args)
    for i in range(l):
        xr=args[i]
        ex.append(exs(xr))
    plot_graphs(ex)
if __name__ == "__main__":
   main(sys.argv[1:])