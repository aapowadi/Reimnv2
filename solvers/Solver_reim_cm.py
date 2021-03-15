import sys
sys.dont_write_bytecode = True
import numpy as np
import os
import cv2
from models.Model_8n_sub1 import *
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

    def set_norm(self, norm):
        self.__norm = norm;

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

    def setLogPathAndFile(self, log_path, log_file, plot_title):
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

    def seg_train_step(self, train_indices):
        """Set up the training parameters
        """
        # Prediction operations
        # Maximum arguments over all logits along dimension 1.
        self.prediction = tf.argmax(self.Y_pre_logits, 2)

        # Probability operation for all logits
        self.probabilities = tf.nn.softmax(self.Y_pre_logits)

    def train_seg(self):
        """
        Start the training procedure.
        The training procedure runs for num_epochs epochs. If one re-trains the network,
        the number of epochs will be added to the current number.
        :return:
        """
################################################################################################################
            # if self.restore_model:
            #     saver = tf.train.import_meta_graph(self.saver_log_folder + "../" + self.restore_from_file)
            #     saver.restore(sess, tf.train.latest_checkpoint(self.saver_log_folder + "../"))
            #     print("Model restored at epoche ", sess.run('epoche:0'))
################################################################################################################
        model_seg=Model_8n_sub1(self.number_classes,self.drop_conv,self.im_height)
        # Softmax and cross-entropy to determine the loss

        # Reduce the sum of all errors. This will sum up all the
        # incorrect identified pixels as loss and reduce this number.


        # Training with adam optimizer.
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        for i in range(self.num_epochs):
            # Count the steps manually
            start_idx=0
            self.step = i + start_idx
            self.iter = i;
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
            average_t_loss = 0
            average_q_loss = 0
            i = 0

            # run the batch
            for start, end in training_batch:
                with tf.GradientTape() as tape:
                    train_indices = shuffled[start:end]
                    seg_logits, predictions = model_seg(self.Xtr_rgb[train_indices], training=True)
                    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(self.Ytr_mask[train_indices],seg_logits)
                    loss = tf.reduce_sum(cross_entropies)
                grads=tape.gradient(loss, model_seg.trainable_weights)
                optimizer.apply_gradients(zip(grads,model_seg.trainable_weights))
                seg_pred, l2_in = model_seg(self.Xtr_rgb[train_indices])
                train_prec, train_rec = self.__getAccuracy(self.Ytr_mask[train_indices],
                                                           seg_pred)  # precision and recall
                i=i+1
                overall_precision = overall_precision + train_prec
                overall_recall = overall_recall + train_rec
                average_loss = average_loss + (loss / (self.im_width * self.im_height))

            overall_precision = overall_precision / i;
            overall_recall = overall_recall / i;
            average_loss = average_loss / i;
            average_q_loss = average_q_loss / i;
            average_t_loss = average_t_loss / i;
            overall_t_rms = overall_t_rms / i;
            overall_q_rms = overall_q_rms / i;

            ### ---------------------------------------------------------------------------------
            # Test accuracy
            test_indices = np.arange(len(self.Xte_rgb))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:self.test_size]
            seg_pred, l2_in = model_seg(self.Xte_rgb[test_indices])
            precision, recall = self.__getAccuracy(self.Yte_mask[test_indices], seg_pred)
            ## ---------------------------------------------------------------------------------
            # Save and test all 10 iterations
            #if i % 10 == 0:
                # self.saver.save(sess, self.saver_log_folder + self.saver_log_file, global_step=self.step)
                # print("Saved at step {self.step}")
                # render and print the current outcome
                # self.__sample_test(self.Xte_rgb, self.Xte_depth, self.Yte_mask, self.Yte_pose, sess)

            if not os.path.exists(self.saver_log_folder):
                os.makedirs(self.saver_log_folder)

            if (self.iter == 0 and self.cont == 0):
                file = open(self.saver_log_folder + "seg_results.csv", "w")
                file.close()
                file = open(self.saver_log_folder + "train_pose_results.csv", "w")
                file.close()
                file = open(self.saver_log_folder + "test_pose_results.csv", "w")
                file.close()
                file = open(self.saver_log_folder + "train_loss.csv", "w")
                file.close()
                file = open(self.saver_log_folder + "test_loss.csv", "w")
                file.close()
            #self.__sample_test(self.Xte_rgb, self.Xte_depth, self.Yte_mask, self.Yte_pose, sess)

            # file = open(self.saver_log_folder + "test_loss.csv", "a")
            # file_str = str(self.step) + "," + str(test_loss / self.im_width * self.im_height) + "," + str(
            #     test_pose_t_loss) + "," + str(test_pose_q_loss) + "\n"
            # file.write(file_str)
            # file.close()

            file = open(self.saver_log_folder + "train_loss.csv", "a")
            file_str = str(self.step) + "," + str(average_loss) + "," + str(
                average_t_loss) + "," + str(average_q_loss) + "\n"
            file.write(file_str)
            file.close()

            file = open(self.saver_log_folder + "seg_results.csv", "a")

            out = str(self.step) + "," + str(overall_precision) + "," + str(
                overall_recall) + "," + str(precision) + "," + str(
                recall) + "\n"
            file.write(out)
            if self.step % 10 ==0:
                print("Step %d: loss = %.4f" % (self.step,loss))
            # if (self.stage2 == True):
            #     file = open(self.saver_log_folder + "train_pose_results.csv", "a")
            #
            #     out = str(self.step) + "," + str(overall_t_rms) + "," + str(overall_q_rms) + "\n"
            #     file.write(out)
            #
            #     file.close()
            #     file = open(self.saver_log_folder + "test_pose_results.csv", "a")
            #
            #     out = str(self.step) + "," + str(testt_rms) + "," + str(testq_mean) + "\n"
            #     file.write(out)
            #
            #     file.close()

        # save the last step
        # self.saver.save(sess, self.saver_log_folder + self.saver_log_file, global_step=self.step)

    def __start_eval(self, Xte_rgb, Xte_depth, Yte_mask, Yte_pose):
        """
        Start the network evaluation.
        :param Xte_rgb: (array) the rgb test dataset of size [N, width, height, 3]
        :param Xte_depth: (array) the depth test dataset of size [N, width, height, 1]
        :param Yte_mask: (array) the ground truth image mask of size [N, width, height, C]
        :param Yte_pose: (array) the ground truth pose as (x, y, z, qx, qy, qz, qw)
        :return:
        """
        # Check if a model has been restored.
        if len(self.restore_from_file) == 0:
            print("ERROR - Validation mode requires a restored model")
            return

        # run the test session
        with tf.Session(graph=self.g) as sess:
            tf.global_variables_initializer().run()

            # Restore the graph
            saver = tf.train.import_meta_graph(self.saver_log_folder + self.restore_from_file)
            saver.restore(sess, tf.train.latest_checkpoint(self.saver_log_folder))
            curr_epoch = sess.run('epoche:0')
            print("Model restored at epoch ", curr_epoch)

            # Check if the number of samples is equal
            if Xte_rgb.shape[0] != Xte_depth.shape[0]:
                print("ERROR - the number of RGB and depth samples must match for validation")
                return

            print("Start validation")
            print("Num test samples {int(Xte_rgb.shape[0])}.")

            # split the evaluation batch in small chunks
            # Note that the evaluation images are not shuffled to keep them aligned with the input images
            test_size = int(Xte_rgb.shape[0] / 4)
            # test_size = int(Xte_rgb.shape[0])
            validation_batch = zip(range(0, len(Xte_rgb), test_size), range(test_size, len(Xte_rgb) + 1, test_size))

            overall_precision = 0
            overall_recall = 0
            overall_t_rms = 0
            overall_q_rms = 0
            average_loss = 0
            average_t_loss = 0
            average_q_loss = 0
            i = 0

            # run the evaluation
            for start, end in validation_batch:
                predict, prob, loss, t_loss, q_loss = self.__test_step(sess, Xte_rgb[start:end], Xte_depth[start:end],
                                                                       Yte_mask[start:end], Yte_pose[start:end][:, 0:3],
                                                                       Yte_pose[start:end][:, 3:7])

                # get precision and recall for the first srage.
                precision, recall = self.__getAccuracy(Yte_mask[start:end], predict, "precision_recall.csv",
                                                       i * test_size)
                print(
                    "Batch {i}, seg. loss {loss / self.im_width*self.im_height}, seg. precison: {precision}, seg. recall: {recall}, pose t loss: {t_loss}, pose q loss: {q_loss}")

                # get rms values for the second stage.
                pr, pr_pose_t, pr_pose_q = self.__exec_sample_test(sess, Xte_rgb[start:end], Xte_depth[start:end],
                                                                   Yte_mask[start:end], Yte_pose[start:end][:, 0:3],
                                                                   Yte_pose[start:end][:, 3:7])
                t_rms, t_all, q_mean, q_all = eval_pose_accuracy(Yte_pose[start:end][:, 0:3], pr_pose_t,
                                                                 Yte_pose[start:end][:, 3:7], pr_pose_q)

                # Write the pose data into a file. This file is for 3D Pose Eval, the 3D pose renderer.
                write_data_for_3DPoseEval(self.saver_log_folder + "pose_eval_input.csv", Yte_pose[start:end][:, 0:3],
                                          Yte_pose[start:end][:, 3:7], pr_pose_t, pr_pose_q, start)

                # Renders the output for a random image.
                self.__sample_test(Xte_rgb, Xte_depth, Yte_mask, Yte_pose, sess)

                i = i + 1
                overall_precision = overall_precision + precision
                overall_recall = overall_recall + recall
                average_loss = average_loss + (loss / self.im_width * self.im_height)
                average_t_loss = average_t_loss + t_loss
                average_q_loss = average_q_loss + q_loss
                overall_t_rms = t_rms + overall_t_rms
                overall_q_rms = overall_q_rms + q_mean

            overall_precision = overall_precision / i
            overall_recall = overall_recall / i
            average_loss = average_loss / i
            average_q_loss = average_q_loss / i
            average_t_loss = average_t_loss / i
            overall_t_rms = overall_t_rms / i
            overall_q_rms = overall_q_rms / i

            print(
                "Final results: \nseg. loss {average_loss}, seg. precison: {overall_precision}, seg. recall: {overall_recall}")
            print(
                "pose t loss: {average_t_loss}, pose q loss {average_q_loss}, t RMS: {overall_t_rms}, q mean: {overall_q_rms}")

            write_report(self.saver_log_folder + "evaluation_rep.csv", curr_epoch, average_loss, overall_precision,
                         overall_recall, average_t_loss, average_q_loss, overall_t_rms, overall_q_rms)

    def __train_step(self, sess, rgb_batch, depth_batch, mask_batch, pose_t_batch, pose_q_batch, p_keep_conv,
                     p_keep_hidden):
        """
        Execute one training step for the entire graph
        :param sess: (tf node) reference to the current tensorflow session
        :param rgb_batch: (array) the rgb training batch as array of size [N, width, height, 3]
        :param depth_batch: (array) the depth image batch as array of size [N, width, height, 1]
        :param mask_batch: (array) the ground truth mask as array of size [N, width, height, 1]
        :param pose_t_batch: (array) the ground truth pose given by (x, y, z) as array of size [N, 3]
        :param pose_q_batch: (array) the ground truth orientation given by (qx, qy, qz, qw) as array of size [N, 4]
        :param p_keep_conv: (float) drouput for the convolutional part, probability to keep the values
        :param p_keep_hidden: (float) dropout for the regression part, probability to keep the values.
        :return:
        """
        test_pose_t_loss = 0
        test_pose_q_loss = 0
        pr_pose_t = 0;
        pr_pose_q = 0
        test_predict = 0;
        test_prob = 0;
        test_loss = 0;
        # Train the first stage.
        if self.stage2 == False:
            sess.run([self.train_fcn, self.prediction,
                      self.probabilities,
                      self.cross_entropy_sum],
                     feed_dict={self.pl_X_rgb: rgb_batch,
                                self.pl_Y_mask: mask_batch,
                                self.pl_keep_conv: p_keep_conv,
                                self.pl_keep_hidden: p_keep_hidden,
                                self.phase_train: self.__norm})

        # Generate the output from the first stage. These are the images with argmax(activations) applied
        output, test_predict, test_prob, test_loss = sess.run([self.fcn_predictions, self.prediction,
                                                               self.probabilities, self.cross_entropy_sum],
                                                              feed_dict={self.pl_X_rgb: rgb_batch,
                                                                         self.pl_Y_mask: mask_batch,
                                                                         self.pl_keep_conv: p_keep_conv,
                                                                         self.pl_keep_hidden: p_keep_hidden,
                                                                         self.phase_train: False})

        # Train the second stage.
        if self.stage2 == True:
            pose_t_gradients, pose_q_gradients, test_pose_t_loss, test_pose_q_loss, pr_pose_t, pr_pose_q = sess.run(
                [self.train_pose_t,
                 self.train_pose_q,
                 self.pose_loss_t,
                 self.pose_loss_q,
                 self.Y_pre_pose_t,
                 self.Y_pre_pose_q],
                feed_dict={self.pl_X_pred: output,
                           self.pl_X_depth: depth_batch,
                           self.pl_Y_pose_t: pose_t_batch,
                           self.pl_Y_pose_q: pose_q_batch,
                           self.pl_keep_conv: p_keep_conv,
                           self.pl_keep_hidden: p_keep_hidden,
                           self.phase_train: self.__norm})

        return test_predict, test_prob, test_loss, test_pose_t_loss, test_pose_q_loss, pr_pose_t, pr_pose_q

    def __test_step(self, sess, Xte_rgb, Xte_depth, Yte_mask, Yte_pose_t, Yte_pose_q):
        """
        Test the trained network.
        All dropouts are set to 1.0
        :param sess: (tf node) reference to the current tensorflow session
        :param Xte_rgb: (array) the rgb test batch as array of size [N, width, height, 3]
        :param Xte_depth: (array) the depth image batch as array of size [N, width, height, 1]
        :param Yte_mask: (array) the ground truth mask as array of size [N, width, height, C-1]
        :param Yte_pose_t: the ground truth pose given as translation (x, y, z) as array of size [N, 3]
        :param Yte_pose_q: the ground truth orientation as quaternion (qx, qy, qz, qw) as array of size [N, 4]
        :return: test_predict (float) - the class predictions
                 test_prob (float) - the class probabilities
                 test_loss (float) - the test loss. Note that this is the sum of all losses
                 test_pose_t_loss (float) - the translation prediction average loss
                 test_pose_q_loss (float) - the orientation prediction average loss.
        """

        # Test the fully connected network, stage 1
        test_predict, test_prob, test_loss = sess.run([self.prediction,
                                                       self.probabilities,
                                                       self.cross_entropy_sum],
                                                      feed_dict={self.pl_X_rgb: Xte_rgb,
                                                                 self.pl_Y_mask: Yte_mask,
                                                                 self.pl_keep_conv: 1.0,
                                                                 self.pl_keep_hidden: 1.0,
                                                                 self.phase_train: False})

        # Generate the output from the first stage. These are the images with argmax(activations) applied
        output = sess.run(self.fcn_predictions, feed_dict={self.pl_X_rgb: Xte_rgb,
                                                           self.pl_Y_mask: Yte_mask,
                                                           self.pl_keep_conv: 1.0,
                                                           self.pl_keep_hidden: 1.0,
                                                           self.phase_train: False})

        # Generate pose output.
        test_pose_t_loss, test_pose_q_loss, test_pose_t, test_pose_q = sess.run([self.pose_loss_t, self.pose_loss_q,
                                                                                 self.Y_pre_pose_t, self.Y_pre_pose_q],
                                                                                feed_dict={self.pl_X_pred: output,
                                                                                           self.pl_X_depth: Xte_depth,
                                                                                           self.pl_Y_pose_t: Yte_pose_t,
                                                                                           self.pl_Y_pose_q: Yte_pose_q,
                                                                                           self.pl_keep_conv: 1.0,
                                                                                           self.pl_keep_hidden: 1.0,
                                                                                           self.phase_train: False})

        return test_predict, test_prob, test_loss, test_pose_t_loss, test_pose_q_loss, test_pose_t, test_pose_q

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
        pr = Ypr.reshape([-1, self.im_width, self.im_height]).astype(float)
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

    def __sample_test(self, Xte, Xte_depth, Yte, Yte_pose, sess):
        """
        Test the network using ONE random image. The function will generate a visual output and
        store the data into a file.
        :param Xte: the RGB test dataset of size [N, width, height, 3]
        :param Xte_depth: the depth images test dataset of size [N, width, height, 3]
        :param Yte: the ground truth mask  of size [N, width, height, 1]
        :param Yte_pose: the ground truth pose as (x, y, z, qx, qy, qz, qw)
        :param sess: a reference to the current session.
        :return:
        """

        # get a random number
        file_index = self.epoch.eval()
        test_indices = np.arange(len(Xte))
        np.random.shuffle(test_indices)
        idx = test_indices[0]

        # get the test rgb image and reshape it to [1, width, height, 3]
        sample = Xte[idx:idx + 1]
        sample = sample.reshape([1, sample.shape[1], sample.shape[2], sample.shape[3]])

        # get the test depth image and reshape it to [1, width, height, 1]
        sample_depth = Xte_depth[idx:idx + 1]
        sample_depth = sample_depth.reshape([1, sample_depth.shape[1], sample_depth.shape[2], sample_depth.shape[3]])

        # get the test mask  and reshape it to [1, width, height, 1]
        good = Yte[idx:idx + 1]
        good = good.reshape([1, good.shape[1], good.shape[2]])

        pose_t = Yte_pose[idx:idx + 1][:, 0:3]
        pose_t = pose_t.reshape([1, pose_t.shape[1]])

        pose_q = Yte_pose[idx:idx + 1][:, 3:7]
        pose_q = pose_q.reshape([1, pose_q.shape[1]])

        # execute the test
        pr, pr_pose_t, pr_pose_q = self.__exec_sample_test(sess, sample, sample_depth, good, pose_t, pose_q)

        pr = pr.reshape([-1, self.im_width]).astype(float)
        pr = pr * 255
        test_mask = Yte[idx]
        test_mask = test_mask[:, 1]
        te = test_mask.reshape([-1, self.im_width]).astype(float)
        te = te * 255

        # vis = np.concatenate((te, pr), axis=1)
        # if self.show_sample:
        #    cv2.imshow("result", vis)
        #    cv2.imshow("testimage", Xte[idx])
        #    if file_index == 0:
        #        cv2.moveWindow('result', 50, 150)
        #        cv2.moveWindow('testimage', 50, 260)
        # cv2.imshow("outcome", pr)
        #    cv2.waitKey(1)
        if self.step % 10 == 0:
            if not os.path.exists(self.saver_log_folder + "render"):
                os.makedirs(self.saver_log_folder + "render")
            file = self.saver_log_folder + "render/result_" + str(file_index) + ".png"
            file2 = self.saver_log_folder + "render/result_rgb_" + str(file_index) + ".png"
            cv2.imwrite(file, pr)
            # cv2.imwrite(file2, Xte[idx])
            # file_index = file_index + 1

    def __exec_sample_test(self, sess, sample, sample_depth, good, pose_t, pose_q):
        """
        Execute the sample test.
        :param sess:
        :param sample:
        :param sample_depth:
        :param good:
        :param pose_t:
        :param pose_q:
        :return:
        """

        pred = tf.argmax(self.Y_pre_logits, 2)

        pr = sess.run(pred,
                      feed_dict={self.pl_X_rgb: sample,
                                 self.pl_Y_mask: good,
                                 self.pl_keep_conv: 1.0,
                                 self.pl_keep_hidden: 1.0,
                                 self.phase_train: False})

        output = sess.run(self.fcn_predictions, feed_dict={self.pl_X_rgb: sample,
                                                           self.pl_Y_mask: good,
                                                           self.pl_keep_conv: 1.0,
                                                           self.pl_keep_hidden: 1.0,
                                                           self.phase_train: False})

        pr_pose_t, pr_pose_q = sess.run([self.Y_pre_pose_t, self.Y_pre_pose_q],
                                        feed_dict={self.pl_X_pred: output,
                                                   self.pl_X_depth: sample_depth,
                                                   self.pl_Y_pose_t: pose_t,
                                                   self.pl_Y_pose_q: pose_q,
                                                   self.pl_keep_conv: 1.0,
                                                   self.pl_keep_hidden: 1.0,
                                                   self.phase_train: False})

        return pr, pr_pose_t, pr_pose_q

