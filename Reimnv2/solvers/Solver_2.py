import tensorflow as tf
import numpy as np
import pickle
import os
from os import path
import tfquaternion as tfq
from models.cnntools import *
from solvers.tools.eval_tools import *
import cv2

class Solver_2:
    """
    This class implements a 6-DoF pose prediction cnn.

    This file implements a solver for a tensorflow graph for 6-DoF pose prediction.
    It is prepared for 2-stage models that work with RGB-D data.
    The solver implements a minibatch training and uses the Adam optimizer for the first stage,
    and the RMSProp optimizer for the second stage.

    The solver shuffles the images in the train batch. The test and evaluation images are remain in their order.

    Features:
    -----------
     - Tensorflow graphy initialization
     - Manages all placeholders
     - Runs the training process
     - Runs a test procedure
     - Saves and restores all models.

    Output files:
    ---------------
    The class writes several output files
    - datalog.csv: stores the loss values and the segmentation precison and recall values for each epoch.
    - evaluation_rep.csv: For each evaluation run, it stores all loss values, the segmentation precision, and the pose accuracy.
    - precision_recall.csv: During evaluation, it stores the individual segmentation precision and recall values for each image.

    Usage:
    ---------
    # init the network
    solver = Solver2StageRGBD_6DoF(Model_RGBD_6DoF, 2, 3, 4, 0.0001)
    solver.setParams(2000, 128, 256)
    solver.showDebug( True)
    solver.setLogPathAndFile(log_folder, log_file)
    # start training
    solver.init(Xtr_rgb.shape[1], Xtr_rgb.shape[2], restore_file )
    solver.train( Xtr_rgb, Xtr_depth, Ytr_mask, Ytr_pose, Xte_rgb, Xte_depth, Yte_mask, Yte_pose)


    Rafael Radkowski
    Iowa State University
    rafael@iastate.edu
    May 4, 2018
    MIT License
    -----------------------------------------------------
    last edit:
    June 18th, 2019, RR
    - Added some opencv debug windows.
    - Results are written into a file.
    June 21, 2019, RR
    - Added a pose evaluation function into the evaluation step __start_eval
    - Added a csv log writer for the evaluation.
    July 19 2019, AP
    - depth image input to CNN model i.e "createModel" function removed at line 228
    July 20, 2019, AP
    - added img_height and img_width parameters to "createModel" class.
    """

    # training mini-batch size
    batch_size = 128
    # test size
    test_size = 256
    # global step variables
    epoch = []
    step = 0

    # num epochs to train
    num_epochs = 2000

    # models
    model = [] # a model instance -> model = model_cls()
    model_cls = [] # reference to the model class

    # Learning rate
    learning_rate = 0.0001

    # number of classes
    number_classes = 2

    # number of outputs
    number_pose_t_outputs = 3
    number_pose_q_outputs = 4

    # placeholders
    pl_X_rgb = []  # input RGB data of size [N, width, height, 3]
    pl_X_pred = []  # prediction output of the first network of size [N, width, height, C]
    pl_X_depth = []  # input data depth [N, width, height, 1]; values in range [0,1]
    pl_Y_mask = []  # ground truth mask of size [N, width, height, 1]
    pl_Y_pose_t = []  # ground truth pose translation of size [N, 3]
    pl_Y_pose_q = []  # ground truth pose rotation of size [N, 4]

    # Dropout ratio placeholder
    pl_keep_conv = []
    pl_keep_hidden = []

    # References to data
    # All those variables point to the data location
    Xtr_rgb = [] # Training RGB data
    Xtr_depth = [] # Training depth images
    Ytr_mask = [] # Training ground truth mask
    Ytr_pose = [] # Training ground truth pose
    Xte_rgb = [] # Testing RGB data
    Xte_depth = [] # Testing depth data
    Yte_mask = [] # Testing ground truth mask
    Yte_pose = [] # Testing ground truth pose

    # predictions output
    Y_pre_logits = [] # output of stage 1, activations
    Y_pre_pose_t = [] # pose output of stage 2
    Y_pre_pose_q = [] # orientation output of stage 2
    fcn_predictions = [] # output of stage 1, class predictions.

    #---------------------------------------------------------
    # 1-stage solver
    cross_entropies = []
    cross_entropy_sum = []
    optimizer = []
    train_fcn = []
    prediction = []
    probabilities = []

    # 2-stage solver translation
    pose_loss_t = []
    train_pose_t = []

    # 2-stage solver quaternion
    pose_loss_q = []
    train_pose_q = []

    # ---------------------------------------------------------
    # To save the variables.
    # Need to be created after all variables have been created.
    saver = []
    saver_log_file = "model"
    saver_log_folder = "./log/100/"


    # restore the model or train a new one
    restore_model = False
    restore_from_file = ""  # keep this empty to not restore the model

    # for debugging, if enabled, the class will show some opencv
    # debug windows during the trianing process.
    show_sample = True



    def __init__(self, model_class,  num_classes, num_pose_t_outputs, num_pose_q_output, learning_rate=0.0001):
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
        self.number_classes = num_classes
        self.number_pose_t_outputs = num_pose_t_outputs
        self.number_pose_q_outputs = num_pose_q_output


    def init(self, img_width, img_height, restore_from_file = ""):
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

        self.restore_from_file = restore_from_file
        self.epoch = tf.Variable(0, name='epoche', trainable=False)

        #-------------------------------------------------------------------------------------
        # placeholders

        # the input image of size [64, 64, 3]
        self.pl_X_rgb = tf.placeholder("float", [None, img_width, img_height, 3])

        # the input image of size [64, 64, 3]
        self.pl_X_depth = tf.placeholder("float", [None, img_width, img_height, 1])

        # predictions from the first network, input for the second network.
        self.pl_X_pred = tf.placeholder("float", [None, img_width, img_height, 1])

        # vector for the output data.
        # The network generates an output layer for each class.
        # 4096 = 64 * 64
        self.pl_Y_mask = tf.placeholder("float", [None, img_width * img_height, self.number_classes], name="Y_mask")

        # the pose output
        self.pl_Y_pose_t = tf.placeholder("float", [None, self.number_pose_t_outputs], name="Y_pose_t")

        # the pose output
        self.pl_Y_pose_q = tf.placeholder("float", [None, self.number_pose_q_outputs], name="Y_pose_q")

        # Dropout ratio placeholder
        self.pl_keep_conv = tf.placeholder("float", name="keep_conv")
        self.pl_keep_hidden = tf.placeholder("float", name="keep_hidden")

        # -------------------------------------------------------------------------------------
        # model
        self.model = self.model_cls(self.number_classes, self.number_pose_t_outputs, self.number_pose_q_outputs)
        self.Y_pre_logits, self.fcn_predictions, self.Y_pre_pose_t, self.Y_pre_pose_q = \
            self.model.createModel(self.pl_X_rgb, self.pl_X_pred, self.pl_keep_hidden, self.pl_keep_hidden, img_width, img_height)

        # -------------------------------------------------------------------------------------
        # solver
        self.__initSolver__()


        # To save the variables.
        # Need to be created after all variables have been created.
        self.saver = tf.train.Saver()

        if len(restore_from_file) > 0:
            self.restore_model = True
            self.restore_from_file = restore_from_file


    def train(self, Xtr_rgb, Xtr_depth, Ytr_mask, Ytr_pose, Xte_rgb, Xte_depth, Yte_mask, Yte_pose, img_width, img_height):
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
        self.__start_train(img_width, img_height)

    def eval(self,  Xte_rgb, Xte_depth, Yte_mask, Yte_pose, img_width, img_height):
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
        self.__start_eval(Xte_rgb, Xte_depth, Yte_mask, Yte_pose, img_width, img_height)

    def setLogPathAndFile(self, log_path, log_file):
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

        # Softmax and cross-entropy to determine the loss
        self.cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=self.Y_pre_logits,
                                                                       labels=self.pl_Y_mask)
        # Reduce the sum of all errors. This will sum up all the
        # incorrect identified pixels as loss and reduce this number.
        self.cross_entropy_sum = tf.reduce_sum(self.cross_entropies)

        # Training with adam optimizer.
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # reducint the gradients.
        gradients = self.optimizer.compute_gradients(loss=self.cross_entropy_sum)

        for grad_var_pair in gradients:
            current_variable = grad_var_pair[1]
            current_gradient = grad_var_pair[0]

            # Relace some characters from the original variable name
            # tensorboard doesn't accept ':' symbol
            gradient_name_to_save = current_variable.name.replace(":", "_")

            # Let's get histogram of gradients for each layer and
            # visualize them later in tensorboard
            if current_gradient != None:
                tf.summary.histogram(gradient_name_to_save, current_gradient)

        # The training step.
        self.train_fcn = self.optimizer.apply_gradients(grads_and_vars=gradients)

        # Prediction operations
        # Maximum arguments over all logits along dimension 1.
        self.prediction = tf.argmax(self.Y_pre_logits, 2)

        # Probability operation for all logits
        self.probabilities = tf.nn.softmax(self.Y_pre_logits)

        ##-------------------------------------------------------------------------------
        ## Loss second model translation

        # self.pose_loss = tf.reduce_mean(tf.square(self.pose_logits - self.Y_pose))
        self.pose_loss_t = tf.norm(self.Y_pre_pose_t - self.pl_Y_pose_t)

        self.train_pose_t = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.pose_loss_t)

        # self.train_pose = tf.train.GradientDescentOptimizer(0.0001).minimize(self.pose_loss)

        ##-------------------------------------------------------------------------------
        ## Loss second model quaternion

        # self.pose_loss = tf.reduce_mean(tf.square(self.pose_logits - self.Y_pose))
        #self.pose_loss_q = tf.reduce_mean(tf.square(self.Y_pre_pose_q - self.pl_Y_pose_q)) # log/12, /13
        #self.pose_loss_q = tf.reduce_mean( tf.reduce_sum(tf.square(self.Y_pre_pose_q - self.pl_Y_pose_q), axis =1)) # log/14

        self.pose_loss_q = tf.reduce_sum(tf.square(tf.abs(self.Y_pre_pose_q[:, 3] - self.pl_Y_pose_q[:, 3])) + tf.norm(self.Y_pre_pose_q[:, 0:3] - self.pl_Y_pose_q[:, 0:3], axis=1))

        # Quaternion comes in the order (w, x, y, z)
       # q_pred = tfq.Quaternion(self.Y_pre_pose_q)
        #q_truth = tfq.Quaternion(self.pl_Y_pose_q)

        #delta = tfq.quaternion_multiply(q_truth.inverse(), q_pred)

        # The constant depicts the perfect goal (1, 0,0,0) means no difference between quaternions
        #self.pose_loss_q =  tf.reduce_sum( (tf.abs(delta - tf.constant([1.0,0.0,0.0,0.0]))))

        #self.train_pose_q = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.pose_loss_q)
        self.train_pose_q = tf.train.AdamOptimizer(self.learning_rate, 0.9).minimize(self.pose_loss_q)

        return self.train_fcn, self.prediction, self.probabilities, self.train_pose_t, self.train_pose_q


    def __start_train(self, img_width, img_height):
        """
        Start the training procedure.
        The training procedure runs for num_epochs epochs. If one re-trains the network,
        the number of epochs will be added to the current number.
        :return:
        """

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            if self.restore_model:
                saver = tf.train.import_meta_graph(self.saver_log_folder + self.restore_from_file)
                saver.restore(sess, tf.train.latest_checkpoint(self.saver_log_folder))
                print("Model restored at epoche ", sess.run('epoche:0'))

            print(f"Start training")
            start_idx = sess.run('epoche:0') + 1

            for i in range(self.num_epochs):
                # Count the steps manually
                self.step = i + start_idx
                assign_op = self.epoch.assign(i + start_idx)
                sess.run(assign_op)

                ## ---------------------------------------------------------------------------------
                # Train
                training_batch = zip(range(0, len(self.Xtr_rgb), self.batch_size),
                                     range(self.batch_size, len(self.Xtr_rgb) + 1, self.batch_size))

                # shuffle the indices
                indices = list(range(0, len(self.Xtr_rgb)))
                shuffled = np.random.permutation(indices)

                # run the batch
                for start, end in training_batch:

                    train_indices = shuffled[start:end]
                    self.__train_step(sess, self.Xtr_rgb[train_indices], self.Xtr_depth[train_indices], self.Ytr_mask[train_indices],
                                         self.Ytr_pose[train_indices][:,0:3], self.Ytr_pose[train_indices][:,3:7], 0.8, 0.8)

                    #self.__train_step(sess, self.Xtr_rgb[start:end], self.Xtr_depth[start:end],self.Ytr_mask[start:end],
                    #                  self.Ytr_pose[start:end][:, 0:3], self.Ytr_pose[start:end][:, 3:7], 0.8, 0.8)

                ### ---------------------------------------------------------------------------------
                # Test accuracy
                test_indices = np.arange(len(self.Xte_rgb))
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:self.test_size]
                test_predict, test_prob, test_loss, test_pose_t_loss, test_pose_q_loss= self.__test_step(sess,
                    self.Xte_rgb[test_indices], self.Xte_depth[test_indices],
                    self.Yte_mask[test_indices], self.Yte_pose[test_indices][:,0:3],  self.Yte_pose[test_indices][:,3:7])

                precision, recall = self.__getAccuracy(self.Yte_mask[test_indices], test_predict, img_width, img_height)
                print(f"Epoch {self.step}, loss {test_loss / img_width*img_height}, avg. precison: {precision}, avg. recall: {recall}, pose t loss: {test_pose_t_loss}, pose q loss: {test_pose_q_loss}")

                ## ---------------------------------------------------------------------------------
                # Save and test all 10 iterations
                if i % 10 == 0:
                    self.saver.save(sess, self.saver_log_folder + self.saver_log_file, global_step=self.step)
                    print(f"Saved at step {self.step}")
                    # render and print the current outcome
                    self.__sample_test(self.Xte_rgb, self.Xte_depth, self.Yte_mask, self.Yte_pose, sess, img_width, img_height)

                file = open(self.saver_log_folder + "datalog.csv", "a")
                file_str = str(self.step) + "," + str(test_loss/img_width*img_height) + "," + str(precision) + "," + str(recall) + "," + str(test_pose_t_loss)+ "," + str(test_pose_q_loss) + "\n"
                file.write(file_str)
                file.close()

            # save the last step
            self.saver.save(sess, self.saver_log_folder + self.saver_log_file, global_step=self.step)


    def __start_eval(self, Xte_rgb, Xte_depth, Yte_mask, Yte_pose, img_width, img_height):
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
        with tf.Session() as sess:
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

            print(f"Start validation")
            print(f"Num test samples {int(Xte_rgb.shape[0])}.")

            # split the evaluation batch in small chunks
            # Note that the evaluation images are not shuffled to keep them aligned with the input images
            test_size = int(Xte_rgb.shape[0] / 4)
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
                                                        Yte_mask[start:end], Yte_pose[start:end][:,0:3], Yte_pose[start:end][:,3:7])

                # get precision and recall for the first srage.
                precision, recall = self.__getAccuracy(img_width, img_height, Yte_mask[start:end], predict, "precision_recall.csv", i * test_size)
                print(f"Batch {i}, seg. loss {loss / img_height*img_width}, seg. precison: {precision}, seg. recall: {recall}, pose t loss: {t_loss}, pose q loss: {q_loss}")

                # get rms values for the second stage.
                pr, pr_pose_t, pr_pose_q = self.__exec_sample_test(sess, Xte_rgb[start:end], Xte_depth[start:end], Yte_mask[start:end], Yte_pose[start:end][:,0:3], Yte_pose[start:end][:,3:7])
                t_rms, t_all, q_mean, q_all = eval_pose_accuracy(Yte_pose[start:end][:,0:3], pr_pose_t, Yte_pose[start:end][:,3:7], pr_pose_q)

                # Write the pose data into a file. This file is for 3D Pose Eval, the 3D pose renderer.
                write_data_for_3DPoseEval(self.saver_log_folder + "pose_eval_input.csv", Yte_pose[start:end][:,0:3], Yte_pose[start:end][:,3:7], pr_pose_t, pr_pose_q, start)

                # Renders the output for a random image.
                self.__sample_test(Xte_rgb, Xte_depth, Yte_mask, Yte_pose, sess, img_width, img_height)

                i = i + 1
                overall_precision = overall_precision + precision
                overall_recall = overall_recall + recall
                average_loss = average_loss + loss / img_width*img_height
                average_t_loss = average_t_loss + t_loss
                average_q_loss = average_q_loss + q_loss
                overall_t_rms = t_rms + overall_t_rms
                overall_q_rms = overall_q_rms + q_mean

            overall_precision = overall_precision / i
            overall_recall = overall_recall / i
            average_loss = average_loss / i
            average_q_loss = average_q_loss / i
            average_t_loss = average_t_loss / i
            overall_t_rms =  overall_t_rms / i
            overall_q_rms = overall_q_rms / i

            print(f"Final results: \nseg. loss {average_loss}, seg. precison: {overall_precision}, seg. recall: {overall_recall}")
            print(f"pose t loss: {average_t_loss}, pose q loss {average_q_loss}, t RMS: {overall_t_rms}, q mean: {overall_q_rms}")

            write_report(self.saver_log_folder + "evaluation_rep.csv", curr_epoch, average_loss, overall_precision, overall_recall, average_t_loss, average_q_loss, overall_t_rms, overall_q_rms )


    def __train_step(self, sess, rgb_batch, depth_batch, mask_batch, pose_t_batch, pose_q_batch, p_keep_conv, p_keep_hidden):
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

        # Train the first stage.
        sess.run(self.train_fcn, feed_dict={self.pl_X_rgb: rgb_batch,
                                        self.pl_Y_mask: mask_batch,
                                        self.pl_keep_conv: p_keep_conv,
                                        self.pl_keep_hidden: p_keep_hidden})

        # Generate the output from the first stage. These are the images with argmax(activations) applied
        output = sess.run(self.fcn_predictions, feed_dict={self.pl_X_rgb: rgb_batch,
                                                             self.pl_Y_mask: mask_batch,
                                                             self.pl_keep_conv: p_keep_conv,
                                                            self.pl_keep_hidden: p_keep_hidden })

        # Shows an RGB test iamge and the segmentation output.
        if self.show_sample:
            test_img = output[0]
            test_img = test_img * 255
            test_rgb = rgb_batch[0]
            ##vis = np.concatenate((test_img, test_rgb), axis=1)

            cv2.imshow("test_img", test_img)
            cv2.imshow("test_rgb", test_rgb)
            cv2.moveWindow('test_img', 30, 450)
            cv2.moveWindow('test_rgb', 30, 560)
            cv2.waitKey(1)

        # Train the second stage.
        sess.run([self.train_pose_t, self.train_pose_q],
                 feed_dict={self.pl_X_pred: output,
                            self.pl_Y_pose_t: pose_t_batch,
                            self.pl_Y_pose_q: pose_q_batch,
                            self.pl_keep_conv: p_keep_conv,
                            self.pl_keep_hidden: p_keep_hidden})


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
                                                                  self.pl_keep_hidden: 1.0})

        # Generate the output from the first stage. These are the images with argmax(activations) applied
        output = sess.run(self.fcn_predictions, feed_dict={self.pl_X_rgb: Xte_rgb,
                                                             self.pl_Y_mask: Yte_mask,
                                                             self.pl_keep_conv: 1.0,
                                                           self.pl_keep_hidden: 1.0})

        # Generate pose output.
        test_pose_t_loss, test_pose_q_loss = sess.run([self.pose_loss_t, self.pose_loss_q],
                                   feed_dict={self.pl_X_pred: output,
                                              self.pl_Y_pose_t: Yte_pose_t,
                                              self.pl_Y_pose_q: Yte_pose_q,
                                              self.pl_keep_conv: 1.0,
                                              self.pl_keep_hidden: 1.0})

        return test_predict, test_prob, test_loss, test_pose_t_loss, test_pose_q_loss


    def __getAccuracy(self,img_width, img_height, Y, Ypr, validation="", start_index=0):
        """
        Calculate the accuracy of all images as precision and recall values.
        Only for the pose segmentation part.

        :param Y: (array) The ground truth data mask as array of size [N, width, height, C]
        :param Ypr: (array) The prediction, index aligned with the ground truth data
        :param validation: (str), set a filename. If the string length is > 0, the results will be written into this file.
        :param start_index: (int), for the file writer; a batch indes that indicates the number of the current batch.
        :return: the precision (float) and recall (float) values.
        """
        pr = Ypr.reshape([-1, img_width, img_height]).astype(float)
        pr = pr * 255

        y = Y[:, :, 1]  # this is the background mask. But the buny model is = 0 here.
        y = y.reshape([-1, img_width, img_height]).astype(float)
        y = y * 255

        N = Y.shape[0]  # size

        if len(validation) > 0 and start_index == 0:
            file = open(self.saver_log_folder + validation, "w")
            file_str = "idx,precision,recall\n"
            file.write(file_str)
            file.close()

        recall = 0
        precision = 0
        for i in range(N):
            pr0 = pr[i]
            y0 = y[i]
            this_recall = 0

            # number of positive examples
            relevant = np.sum(np.equal(y0, 0).astype(int))
            tp_map = np.add(pr0, y0)  # all true positive end up as 0 after the additon.
            tp = np.sum(np.equal(tp_map, 0).astype(int))  # count all true positionve
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

        return precision, recall


    def __sample_test(self, Xte, Xte_depth, Yte, Yte_pose, sess, img_width, img_height):
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

        pose_t =  Yte_pose[idx:idx + 1][:,0:3]
        pose_t = pose_t.reshape([1, pose_t.shape[1]])

        pose_q = Yte_pose[idx:idx + 1][:, 3:7]
        pose_q = pose_q.reshape([1, pose_q.shape[1]])

        # execute the test
        pr, pr_pose_t, pr_pose_q = self.__exec_sample_test(sess, sample, sample_depth, good, pose_t, pose_q)

        pr = pr.reshape([-1, img_width]).astype(float)
        pr = pr * 255
        test_mask = Yte[idx]
        test_mask = test_mask[:, 1]
        te = test_mask.reshape([-1, img_width]).astype(float)
        te = te * 255

        if pr_pose_t.shape[1] == 3:
            print(f"pose t: {pose_t[0][0]}, {pose_t[0][1]},{pose_t[0][2]}, t pr:  {pr_pose_t[0][0]}, {pr_pose_t[0][1]},{pr_pose_t[0][2]}")
            print(f"pose q: {pose_q[0][0]}, {pose_q[0][1]},{pose_q[0][2]}, {pose_q[0][3]}, q pr:  {pr_pose_q[0][0]}, {pr_pose_q[0][1]}, {pr_pose_q[0][2]}, {pr_pose_q[0][3]}")

            file = open(self.saver_log_folder + "pose_results.csv", "a")
            str_t = "t_gt," +  str(pose_t[0][0]) + "," + str(pose_t[0][1]) + "," + str(pose_t[0][2])  + ",t_pr," + str(pr_pose_t[0][0]) + "," + str(pr_pose_t[0][1])+ "," + str(pr_pose_t[0][2]) + "\n"
            str_q = "q_gt," + str(pose_q[0][0]) + "," + str(pose_q[0][1]) + "," + str(pose_q[0][2]) + "," + str(pose_q[0][3]) + ",q_pr," + str(pr_pose_q[0][0]) + "," + str(pr_pose_q[0][1]) + "," + str(pr_pose_q[0][2])  + "," + str(pr_pose_q[0][3]) +"\n"
            file.write( str(file_index) + str("\n"))
            file.write(str_t)
            file.write(str_q)
            file.close()

        else:
            print(f"pose: {pr_pose_t[0][0]} , pr:  {pr_pose_t[0][0]}")

        vis = np.concatenate((te, pr), axis=1)
        if self.show_sample:
            cv2.imshow("result", vis)
            cv2.imshow("testimage", Xte[idx])
            if file_index == 0:
                cv2.moveWindow('result', 50, 150)
                cv2.moveWindow('testimage', 50, 260)
            # cv2.imshow("outcome", pr)
            cv2.waitKey(1)

        if not os.path.exists(self.saver_log_folder + "render"):
            os.makedirs(self.saver_log_folder + "render")
        file = self.saver_log_folder + "render/result_" + str(file_index) + ".png"
        file2 = self.saver_log_folder + "render/result_rgb_" + str(file_index) + ".png"
        cv2.imwrite(file, vis)
        cv2.imwrite(file2, Xte[idx])
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
                                 self.pl_keep_hidden: 1.0})

        output = sess.run(self.fcn_predictions, feed_dict={self.pl_X_rgb: sample,
                                                             self.pl_Y_mask: good,
                                                             self.pl_keep_conv: 1.0,
                                                             self.pl_keep_hidden: 1.0})

        pr_pose_t, pr_pose_q = sess.run( [self.Y_pre_pose_t,self.Y_pre_pose_q],
                            feed_dict={ self.pl_X_pred: output,
                                        self.pl_Y_pose_t: pose_t,
                                        self.pl_Y_pose_q: pose_q,
                                        self.pl_keep_conv: 1.0,
                                        self.pl_keep_hidden: 1.0})

        return pr, pr_pose_t, pr_pose_q

