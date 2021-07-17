import sys
sys.dont_write_bytecode = True
from solvers.Solver_reim_cm import *
from models.filetools import *
from solvers.tools.quattool import *
from plot_graphs import *

class Experiment:
    """Universal class to run CNN model experiments.
    The class gets a description from a dictionary and can run the experiment with this
    description.
    """

    _descrip = []
    _ready = False

    def __init__(self, description):
        """ Init the class with a description of the experiment.

        :param description: dictionary
        The dictionary must contain the following variables
        train_dataset: (string) the training dataset path and file, e.g., "../data/dataset_bunny06.pickle"
        test_dataset: (string) the test dataset e.g,., "../data/dataset_bunny06.pickle"
        solver: (class) the class of the solver to be user, e.g., SolverRGBD_QuatLoss
        model: (class) the class name of the CNN model to be used, e.g.,  Model_RGBD_6DoF_L
        num_iterations: (integer) containing the number of iterations to run, e.g., 100
        learning_rate: (float) Learning rate for the model
        debug_output: (bool) if True, graphical debug windows are display, False hides them.
        log_path: (string) relative or absolute log path. END THE PATH WITH A "/", e.g., "./log/23/"
        log_file: (string) a file name for a log file, e.g., "bunny06_64x64"
        train: (bool) set True to enable training, False will not train the model
        eval: (bool) set True to enable x-evaluation after training, False will not eval the model
        test: (bool) set True to test model with the test_dataset. Note that the variable test_dataset must be set
        restore_file: (string) filename of the model that needs to be restored. Keep empty to train from scratch.
        quat_used: (bool) indicate if the dataset contains quaternions. Set to True, if so.
        plot_title: (string) a string containing a title that should appear on the plot.
        """
        self._descrip = description

        # check the dictionary for completeness
        err = self.__check_dict__( self._descrip )

        if err == 2:
            self._ready = False
            print("ABORT - CRITICAL ERRORS. CNN WILL NOT RUN")
        else:
            self._ready = True

    def start(self):
        """
        Start the experiment
        :return:
        """

        if self._ready == False:
            return

        # Load and prepare the data
        # [Xtr, Xtr_depth, Ytr, Ytr_pose, Xte, Xte_depth, Yte, Yte_pose]
        loaded_data = prepare_data_RGBD_6DoF(self._descrip["train_dataset"])
        # Synthetic data
        if self._descrip["proof"]:
            Xtr_rgb0 = loaded_data[0]
            Xtr_rgb = Xtr_rgb0[510:638]
            Xtr_depth0 = loaded_data[1]
            Xtr_depth = Xtr_depth0[510:638]
            Ytr_mask0 = loaded_data[2]
            Ytr_mask = Ytr_mask0[510:638]
            Ytr_pose0 = loaded_data[3]  # [:,0]
            Ytr_pose = Ytr_pose0[510:638]

        else:
            Xtr_rgb = loaded_data[0]
            Xtr_depth = loaded_data[1]
            Ytr_mask = loaded_data[2]
            Ytr_pose = loaded_data[3]  # [:,0]
        # convert all quaternions into axis-angle transformation
        Ytr_pose_aa = []
        size = Ytr_pose.shape[0]
        for i in range(0, size):
            pose = Ytr_pose[i]
            t = pose[0:3]
            q = pose[3:7]
            aa = Quaternion.quat2AxisAngle(q)
            new_pose = np.array([t[0], t[1], t[2], float(aa[0]), float(aa[1]), float(aa[2]), float(aa[3])])
            new_pose = new_pose.reshape([1, new_pose.shape[0]])
            if i == 0:
                Ytr_pose_aa = new_pose
            else:
                Ytr_pose_aa = np.concatenate((Ytr_pose_aa, new_pose), axis=0)

        # swap colums for the quaternion from (x, y, z, w) -> (w, x, y, z)
        # Ytr_pose[:,[3,6]] = Ytr_pose[:,[6,3]]
        if self._descrip["proof"]:
            Xte_rgb0 = loaded_data[0]
            Xte_rgb = Xte_rgb0[510:638]
            Xte_depth0 = loaded_data[1]
            Xte_depth =Xte_depth0[510:638]
            Yte_mask0 = loaded_data[2]
            Yte_mask=Yte_mask0[510:638]
            Yte_pose0 = loaded_data[3]  # [:,0]
            Yte_pose= Yte_pose0[510:638]

        else:
            Xte_rgb = loaded_data[4]
            Xte_depth = loaded_data[5]
            Yte_mask = loaded_data[6]
            Yte_pose = loaded_data[7]  # [:,0]
        Yte_pose_aa = []
        size = Yte_pose.shape[0]
        for i in range(0, size):
            pose = Yte_pose[i]
            t = pose[0:3]
            q = pose[3:7]
            aa = Quaternion.quat2AxisAngle(q)
            new_pose = np.array([t[0], t[1], t[2], float(aa[0]), float(aa[1]), float(aa[2]), float(aa[3])])
            new_pose = new_pose.reshape([1, new_pose.shape[0]])
            if i == 0:
                Yte_pose_aa = new_pose
            else:
                Yte_pose_aa = np.concatenate((Yte_pose_aa, new_pose), axis=0)
        # swap colums for the quaternion from (x, y, z, w) -> (w, x, y, z)
        # Yte_pose[:,[3,6]] = Yte_pose[:,[6,3]]


        # Init the network
        if self._descrip["batch"] == False:
            logs = []
            solver = self._descrip["solver"](self._descrip["model"], 2, 3, 4, self._descrip["learning_rate"])
            solver.setParams(self._descrip["num_iterations"], self._descrip["batch_size"], self._descrip["batch_size"])
            solver.showDebug(self._descrip["debug_output"])
            solver.setLogPathAndFile(self._descrip["log_path"], self._descrip["log_file"]
                                     ,self._descrip["plot_title"],(self._descrip["trained_models"]))
            logs.append(self._descrip["log_path"])

            # start training
            solver.init(Xtr_rgb.shape[1], Xtr_rgb.shape[2], self._descrip["restore_file"],self._descrip["cont"])
            solver.img_dimensions(Xtr_rgb.shape[1], Xtr_rgb.shape[2])
            solver.trainstage(self._descrip["stage2"])
            solver.drp_cnv(self._descrip["drp_cnv"])
            solver.drp_pose(self._descrip["drp_pose"])
            if self._descrip["train"]:
                solver.train( Xtr_rgb, Xtr_depth, Ytr_mask, Ytr_pose, Xte_rgb, Xte_depth, Yte_mask, Yte_pose)
        else:
            j=0;
            logs = []
            for i in decimal_range(self._descrip["learning_rate"],0.1,0.001):
                j=j+1
                solver = self._descrip["solver"](self._descrip["model"], 2, 3, 4, (self._descrip["learning_rate"]+i))
                solver.setParams(self._descrip["num_iterations"], self._descrip["batch_size"], self._descrip["batch_size"])
                solver.showDebug(self._descrip["debug_output"])
                log_path = self._descrip["log_path"][:-1] + str(j) + "/"
                logs.append(log_path)
                solver.setLogPathAndFile((log_path), (self._descrip["log_file"]),
                                         (self._descrip["plot_title"]),(self._descrip["trained_models"]))

                # start training
                solver.init(Xtr_rgb.shape[1], Xtr_rgb.shape[2], self._descrip["restore_file"], self._descrip["cont"])
                solver.img_dimensions(Xtr_rgb.shape[1], Xtr_rgb.shape[2])
                solver.trainstage(self._descrip["stage2"])
                solver.drp_cnv(self._descrip["drp_cnv"])
                solver.drp_pose(self._descrip["drp_pose"])
                if self._descrip["train"]:
                    solver.train(Xtr_rgb, Xtr_depth, Ytr_mask, Ytr_pose, Xte_rgb, Xte_depth, Yte_mask, Yte_pose)

        plot_stuff(logs, Xtr_rgb.shape[1], Xtr_rgb.shape[2],self._descrip["batch"])
        # evaluate
        # if self._descrip["test"]:
        #     solver.eval(Xte_rgb, Xte_depth, Yte_mask, Yte_pose)
        #
        # # evaluation with a second evaluation set.
        if self._descrip["eval"] and len(self._descrip["eval_dataset"]) > 0:
            loaded_eval_data = prepare_data_RGBD_6DoF(self._descrip["eval_dataset"])
            # Real world evaluation data
            Xev_rgb = loaded_eval_data[0]
            Xev_depth = loaded_eval_data[1]
            Yev_mask = loaded_eval_data[2]
            Yev_pose = loaded_eval_data[3]  # [:,0]

            # swap colums for the quaternion from (x, y, z, w) -> (w, x, y, z)
            # Yev_pose[:,[3,6]] = Yev_pose[:,[6,3]]
        #     solver.eval(Xev_rgb, Xev_depth, Yev_mask, Yev_pose)

    def __check_dict__(self, dict):
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

        expected_keys = ["train_dataset", "eval_dataset", "solver", "model", "num_iterations", "debug_output", "log_path",
                         "log_file", "train", "eval", "test","proof", "restore_file", "quat_used", "plot_title",
                         "learning_rate", "label","stage2","cont","drp_cnv","drp_pose","batch_size", "only_r","batch"
                        ,"trained_models"]

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
                elif i == "eval_dataset":
                    print("CRITICAL ERROR - Evaluation dataset is missing!")
                    error_level = 2
                elif i == "test":
                    print("Test is not set. Setting it to False")
                    self._descrip["test"] = False
                elif i == "solver":
                    print("CRITICAL ERROR - Solver is missing!")
                    error_level = 2
                elif i == "model":
                    print("CRITICAL ERROR - Model is missing!")
                    error_level = 2
                elif i == "num_iterations":
                    print("num_iterations is missing! Set to 100")
                    self._descrip["num_iterations"] = 100
                elif i == "batch_size":
                    print("batch_size is missing! Set to 128")
                    self._descrip["batch_size"] = 128
                elif i == "batch":
                    print("batch Learning not set! default = False")
                    self._descrip["batch"] = False
                elif i == "debug_output":
                    print("debug_output of iteratrion is missing! Set to True")
                    self._descrip["debug_output"] = True
                elif i == "log_path":
                    print("log_path is missing! Set to ./logs/temp/")
                    self._descrip["log_path"] = "./logs/temp/"
                elif i == "trained_models":
                    print("path for trained models is missing! Set to ./temp_models/")
                    self._descrip["trained_models"] = "./temp_models/"
                elif i == "log_file":
                    print("log_file is missing! Set to idiot")
                    self._descrip["log_file"] = "idiot"
                elif i == "stage2":
                    print("Stage2 is not set! default = False")
                    self._descrip["stage2"] = False
                elif i == "only_r":
                    print("only_r is not set")
                    self._descrip["only_r"] = False
                elif i == "cont":
                    print("Continuation is not set, default = False")
                    self._descrip["cont"] = False
                elif i == "drp_cnv":
                    print("Dropout for 1st stage not set so, it is set to 0.2")
                    self._descrip["drp_cnv"] = 0.2
                elif i == "drp_pose":
                    print("Dropout for 2nd stage not set so, it is set to 0.0")
                    self._descrip["drp_pose"] = 0.0
                elif i == "train":
                    print("train is missing! Set to True")
                    self._descrip["train"] = True
                elif i == "eval":
                    print("eval is missing! Set to False")
                    self._descrip["eval"] = False
                elif i == "test":
                    print("test is missing! Set to False")
                    self._descrip["test"] = False
                elif i == "proof":
                    print("Proof is missing! Set to False")
                    self._descrip["test"] = False
                elif i == "restore_file":
                    print("restore_file is missing! Set empty string")
                    self._descrip["restore_file"] = ""
                elif i == "quat_used":
                    print("quat_used is missing! Set to True")
                    self._descrip["restore_file"] = True
                elif i == "plot_title":
                    print("plot_title is missing! Set to No Title")
                    self._descrip["plot_title"] = "No title"
                elif i == "learning_rate":
                    print("learning_rate is missing! Set to 0.001")
                    self._descrip["learning_rate"] = 0.001
                elif i == "label":
                    self._descrip["label"] = "Unlabeled experiment"


        return error_level
