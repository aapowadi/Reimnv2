import sys
sys.dont_write_bytecode = True
import pathlib
import cv2
from solvers.tools.reim_loss import *
import tensorflow as tf
from solvers.tools.test_tool import *

class Solver_reim_cm:
    """
    """
    def trainstage(self, stage):
        self.stage2 = stage

    def img_dimensions(self, img_width, img_height):
        self.im_width = img_width;
        self.im_height = img_height;

    def drp_cnv(self, drp_cnv):
        self.drop_conv = drp_cnv

    def drp_pose(self, drp_pose):
        self.keep_hidden = drp_pose

    def __init__(self, model_class, num_classes, num_pose_t_outputs, num_pose_q_output, learning_rate=0.0001):
        """Constructor

        Just takes some essential parameters. Does not do anything else otherwise.

        :param model_class: (class)
            Reference to the class of the model. Note that this solver will create its own instance.
        :param num_classes: (int)
            The number of classes to train
        :param num_pose_t_outputs: (int)
            The number of translation output, typically three, but one can train less.
        :param num_pose_q_output: (int)
            The number of orientation outputs, typically four for a quaternion.
        :param learning_rate: (float)
            The learning rate for all networks.
        """
        self.model_cls = model_class
        self.learning_rate = learning_rate
        self.learning_rate_t = learning_rate
        self.learning_rate_q = learning_rate
        self.number_classes = num_classes
        self.number_pose_t_outputs = num_pose_t_outputs
        self.number_pose_q_outputs = num_pose_q_output

    def init(self, img_width, img_height, restore_from_file="", cont=False):
        """ Initialize the model the the tensorflow graph.

        This function initializes the model, all placeholders, and the solver.
        It also initializes the Tensorflow Saver

        :param img_width: (pixel)
            The width of the training and testing images in pixels.
        :param img_height: (pixel)
            The height of the training and testing images in pixels.
        :param restore_from_file: (str)
            Add a checkpoint filename to restart training from a certain checkpoint.
                                  Keep it empty for a fresh model.
        :return:
        """
        self.cont = cont
        self.restore_from_file = restore_from_file
        #self.epoch = tf.Variable(0, name='epoche', trainable=False)
        self.epoch=0;
        # solver
        #self.__initSolver__()

        # To save the variables.
        # Need to be created after all variables have been created.
        # self.saver = tf.train.Saver()
        #
        # if len(restore_from_file) > 0:
        #     self.restore_model = True
        #     self.restore_from_file = restore_from_file

    def train(self, Xtr_rgb, Xtr_depth, Ytr_mask, Ytr_pose, Xte_rgb, Xte_depth, Yte_mask, Yte_pose):
        """Start to train the model.

        This function starts the training session. It restores a model parameters, if one is given,
        and runs the training session for the specified number of epochs.

        :param Xtr_rgb: (array)
            Training rgb data of size [N, width, height, 3], with N, the number of samples.
        :param Xtr_depth: (array)
            Training depth data of size [N, width, height, 1]
        :param Ytr_mask: (array)
            Training ground truth mask of size [N, width, height, C]
        :param Ytr_pose: (array)
            Training ground truth pose composed as [x, y, z, qx, qy, qz, qw]
        :param Xte_rgb:(array)
            Testing rgb data of size [N, width, height, 3]
        :param Xte_depth: (array)
            Testing depth data of size [N, width, height, 1]
        :param Yte_mask: (array)
            Testing ground truth mask of size [N, width, height, C]
        :param Yte_pose: (array)
            Testing ground truth pose composed as [x, y, z, qx, qy, qz, qw]
        :return:
        """
        self.Xtr_rgb = Xtr_rgb
        self.Xtr_depth = Xtr_depth
        self.Ytr_mask = Ytr_mask
        self.Ytr_pose = Ytr_pose
        self.Xte_rgb = Xte_rgb
        self.Xte_depth = Xte_depth
        self.Yte_mask = Yte_mask
        self.Yte_pose = Yte_pose


        # invokes training
        self.train_seg()

    def eval(self, Xte_rgb, Xte_depth, Yte_mask, Yte_pose):
        """ Start the  evaluation of the current model.

        :param Xte_rgb: (array)
            rgb data of size [N, width, height, 3]
        :param Xte_depth: (array)
            depth data of size [N, width, height, 1]
        :param Yte_mask: (array)
            Evalutation ground truth mask of size [N, width, height, C-1]
        :param Yte_pose: (array)
            Evalutation ground truth pose given as translation and quaterniont as (x, y, z, qx, qy, qz, qw).
            The data needs to come as array of size [N,7}
        :return: None
        """
        self.__start_eval(Xte_rgb, Xte_depth, Yte_mask, Yte_pose)

    def setLogPathAndFile(self, log_path, log_file, plot_title, trained_models):
        """Set a log file path and a log file name.

        The solve logs the tensorflow checkpoints automatically each 10 epochs.
        Set the path and the logfile using this function. The log_path is also used for
        all the other output files. The solver will not log anything if no path is given.

        :param log_path: (str)
            A string with a relative or absolute path.  Complete the log path with a /
        :param log_file: (str)
            A string containing the log file name.
        :return: None
        """
        self.saver_log_folder = log_path
        self.saver_log_file = log_file
        self.plot_title = plot_title
        self.trained_model_path = trained_models

    def setParams(self, num_epochs, batch_size, test_size):
        """Set training parameters.

        :param num_epochs: (int)
            Set the number of epochs to train. Note that is a relative number and not the global, already trained
            epoch. The number set will be added to the total number of epochs to train.
        :param batch_size: (int)
            The batch size for mini-batch training as int
        :param test_size: (int)
            The test size for testing. Note that the test size should be larger than the batch size.
        :return:
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.test_size = test_size

    def showDebug(self, show_plot):
        """Show or hide all debug outputs.

        A debug window showing the predicted image mask and the RGB image
        shows the results of each batch and the test results after each epoch.
        True activates this feature, False deactives it.

        :param show_plot: (bool)
            True shows all debug outputs, False will hide them.
        :return: None
        """
        self.show_sample = show_plot

    def __initSolver__(self):
        """Init the solver for the model.

        This solver uses Adam to optimize the first stage + softmax cross-entropy
        It uses RMSProp for the second stage.
        The learning rate is equal for all of the graphs
        :return:
        """

        # -------------------------------------------------------------------------------------
        # Solver first stage


        ##-------------------------------------------------------------------------------
        ## Loss second model translation
        self.pose_loss_t = tf.norm(self.Y_pre_pose_t - self.pl_Y_pose_t)

        self.train_pose_t = tf.keras.optimizers.RMSprop(self.learning_rate_t, 0.9).minimize(self.pose_loss_t)

        ##-------------------------------------------------------------------------------
        ## Loss second model quaternion
        self.pose_loss_q = reim_loss(self.Y_pre_pose_q,self.pl_Y_pose_q)
        self.pose_grad_q = reim_grad(self.Y_pre_pose_q,self.pl_Y_pose_q)

        self.train_pose_q = tf.keras.optimizers.Adam(self.learning_rate_q, 0.9).minimize(self.pose_loss_q,grad_loss=self.pose_grad_q)

    def train_seg(self):
        """
        Start the training procedure.
        The training procedure runs for num_epochs epochs. If one re-trains the network,
        the number of epochs will be added to the current number.
        :return:
        """
        net=self.model_cls(self.number_classes,self.im_height)

        # Training with adam optimizer.
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.step = 0;
        self.iter = 0;
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer,
                                   net=net)
        manager = tf.train.CheckpointManager(ckpt, self.trained_model_path, max_to_keep=3)
        # ----------------------------
        # -----  Train and Save  -----
        # ----------------------------

        if self.cont:
            ckpt.restore(manager.latest_checkpoint)

            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for i in range(self.num_epochs):
            # Count the steps manually
            start_idx=0
            self.iter = i;
            self.step = ckpt.step
            ## ---------------------------------------------------------------------------------
            # Train
            training_batch = zip(range(0, len(self.Xtr_rgb), self.batch_size),
                                 range(self.batch_size, len(self.Xtr_rgb) + 1, self.batch_size))

            # shuffle the indices
            indices = list(range(0, len(self.Xtr_rgb)))
            shuffled = np.random.permutation(indices)
            overall_precision = 0
            overall_recall = 0
            overall_t_rms = 0
            overall_q_rms = 0
            average_loss = 0
            test_loss = 0
            average_t_loss = 0
            average_q_loss = 0
            j = 0

            # run the batch
            for start, end in training_batch:
                with tf.GradientTape() as tape:
                    train_indices = shuffled[start:end]
                    seg_logits, predictions = net(self.Xtr_rgb[train_indices],self.drop_conv, training=True)
                    loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.Ytr_mask[train_indices],logits=seg_logits)
                    loss = tf.reduce_sum(loss)
                    #loss /= tf.cast(tf.reduce_prod(tf.shape(self.Ytr_mask[train_indices])[1:]), tf.float32)
                grads=tape.gradient(loss, net.trainable_weights)
                optimizer.apply_gradients(zip(grads,net.trainable_weights))
                seg_pred, l2_in = net(self.Xtr_rgb[train_indices], 0, training = False)
                seg_pred = tf.argmax(seg_pred, 2)
                train_prec, train_rec = self.__getAccuracy(self.Ytr_mask[train_indices],
                                                           seg_pred)  # precision and recall
                j=j+1
                overall_precision = overall_precision + train_prec
                overall_recall = overall_recall + train_rec
                average_loss = average_loss + (loss / (self.im_width * self.im_height))

            overall_precision = overall_precision / j;
            overall_recall = overall_recall / j;
            average_loss = average_loss / j;
            average_q_loss = average_q_loss / j;
            average_t_loss = average_t_loss / j;
            overall_t_rms = overall_t_rms / j;
            overall_q_rms = overall_q_rms / j;

            ### ---------------------------------------------------------------------------------
            # Test accuracy
            test_indices = np.arange(len(self.Xte_rgb))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:self.test_size]
            seg_pred, l2_in = net(self.Xte_rgb[test_indices], 0, training = False)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.Yte_mask[test_indices], logits=seg_pred)
            test_loss = tf.reduce_sum(loss)/(self.im_width * self.im_height)
            seg_pred = tf.argmax(seg_pred, 2)
            precision, recall = self.__getAccuracy(self.Yte_mask[test_indices], seg_pred)


            #####
            np.random.shuffle(test_indices)
            idx = test_indices[0]

            # get the test rgb image and reshape it to [1, width, height, 3]
            sample = self.Xte_rgb[idx:idx + 1]
            sample = sample.reshape([1, sample.shape[1], sample.shape[2], sample.shape[3]])

            # execute the test
            prr, l2_test = net(sample, 0, training = False)
            prr = tf.argmax(prr, 2)
            pr = np.array(prr).astype(float)
            pr = pr.reshape([-1, self.im_width]).astype(float)
            pr = pr * 255
            test_mask = self.Yte_mask[idx]
            test_mask = test_mask[:, 1]
            te = test_mask.reshape([-1, self.im_width]).astype(float)
            te = te * 255

            if self.iter % 10 == 0:
                pathlib.Path(self.saver_log_folder + "render").mkdir(parents=True, exist_ok=True)
                file = self.saver_log_folder + "render/result_" + str(float(self.step)) + ".png"
                file2 = self.saver_log_folder + "render/mask_rgb_" + str(float(self.step)) + ".png"
                cv2.imwrite(file, pr)
                cv2.imwrite(file2, te)

            pathlib.Path(self.saver_log_folder).mkdir(parents=True, exist_ok=True)
            if (self.iter == 0 and self.cont == 0):
                file = open(self.saver_log_folder + "accuracy_results.csv", "w")
                file.close()
                file = open(self.saver_log_folder + "losses.csv", "w")
                file.close()

            file = open(self.saver_log_folder + "losses.csv", "a")
            file_str = str(float(self.step)) + "," + str(float(average_loss)) + "," \
                       + str(float(test_loss)) + "\n"
            file.write(file_str)
            file.close()

            file = open(self.saver_log_folder + "accuracy_results.csv", "a")

            out = str(float(self.step)) + "," + str(overall_precision) + "," + str(
                overall_recall) + "," + str(precision) + "," + str(
                recall) + "\n"
            file.write(out)
            if int(ckpt.step) % 10 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("Step %d: loss = %.4f, precision = %.4f, recall = %.4f" % (self.step,average_loss,overall_precision,overall_recall))
            ckpt.step.assign_add(1)

    def __getAccuracy(self, Y, Ypr, validation="", start_index=0):
        """
        Calculate the accuracy of all images as precision and recall values.
        Only for the pose segmentation part.

        :param Y: (array) The ground truth data mask as array of size [N, width, height, C]
        :param Ypr: (array) The prediction, index aligned with the ground truth data
        :param validation: (str), set a filename. If the string length is > 0, the results will be written into this file.
        :param start_index: (int), for the file writer; a batch indes that indicates the number of the current batch.
        :return: the precision (float) and recall (float) values.
        """
        Ypr_arr=np.array(Ypr).astype(float)
        pr = Ypr_arr.reshape([-1, self.im_width, self.im_height]).astype(float)
        pr = pr * 255

        y = Y[:, :, 1]  # this is the background mask. But the bunny model is = 0 here.
        y = y.reshape([-1, self.im_width, self.im_height]).astype(float)
        y = y * 255

        N = Y.shape[0]  # size

        if len(validation) > 0 and start_index == 0:
            file = open(self.saver_log_folder + validation, "w")
            file_str = "idx,precision,recall\n"
            file.write(file_str)
            file.close()

        recall = 0
        precision = 0
        all_precision = []
        prec_rec = []
        ss_precision = []
        s_precision = []
        for i in range(N):
            pr0 = pr[i]
            y0 = y[i]
            this_recall = 0

            # number of positive examples
            relevant = np.sum(np.equal(y0, 0).astype(int))
            tp_map = np.add(pr0, y0)  # all true positive end up as 0 after the additon.
            tp = np.sum(np.equal(tp_map, 0).astype(int))  # count all true positionve

            if relevant.astype(float) > 0:
                this_recall = (tp.astype(float) / relevant.astype(float))  # recall = tp / (tp + fn)
                recall = recall + this_recall

            # get all true predicted = tp and fp
            pr_true = np.sum(np.equal(pr0, 0).astype(int))
            this_precision = 0
            if pr_true > 0:
                this_precision = tp / pr_true
                precision = precision + this_precision

            if len(validation) > 0:
                file = open(self.saver_log_folder + validation, "a")
                file_str = str(start_index + i) + "," + str(this_precision) + "," + str(this_recall) + "\n"
                file.write(file_str)
                file.close()

        recall = recall / float(N)
        precision = precision / float(N)
        # s_precision = sorted(all_precision, reverse=True)
        # for i in range(0, N, int(N / 10)):
        #     ss_precision = s_precision[:i]
        #     for value in ss_precision:
        #         if value >= 0.5:
        #             n = n + 1;
        #     ss_precision.clear();
        #     if i == 0:
        #         prec = 1;
        #         n = 0;
        #         prec_rec.append(prec);
        #
        #     else:
        #         prec = n / i;
        #         n = 0;
        #         prec_rec.append(prec);
        #
        # t = np.arange(0.0, 1.0, 0.1)
        # plt.plot(t, prec_rec)
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.title('Segmentation')
        # plt.show()

        return precision, recall