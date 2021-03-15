import csv
import matplotlib.pyplot as plt
import math
import numpy as np

class get_gra:

    def plot_train_64(self, _file_loc1, _file_loc2, _file_loc3, _file_loc4, _file_loc5, _file_loc6, _file_loc7, _file_loc8,
                    _file_loc9,saver_folder):
            saver_log_folder = saver_folder;
            saver_log_folder1 = _file_loc1;
            saver_log_folder2 = _file_loc2;
            saver_log_folder3 = _file_loc3;
            saver_log_folder4 = _file_loc4;
            saver_log_folder5 = _file_loc5;
            saver_log_folder6 = _file_loc6;
            saver_log_folder7 = _file_loc7;
            saver_log_folder8 = _file_loc8;
            saver_log_folder9 = _file_loc9;
            label1 ="1: seg layers = 4, lr = 0.001";label2 ="2: seg layers = 4, lr = 0.01";label3 ="3: seg layers = 4, lr = 0.1";
            label4 = "4: seg layers = 8, lr = 0.001";label5 ="5: seg layers = 8, lr = 0.01";label6 ="6: seg layers = 8, lr = 0.1";
            label7 = "7: seg layers = 10, lr = 0.001";label8 ="8: seg layers = 10, lr = 0.01";label9 ="9: seg layers = 10, lr = 0.1";

            x = []
            y = []
            pr_t1 = [];pr_t2 = [];pr_t3 = [];pr_t4 = [];pr_t5 = [];pr_t6 = [];pr_t7 = [];pr_t8 = [];pr_t9 = [];
            pr_r1 = [];pr_r2 = [];pr_r3 = [];pr_r4 = [];pr_r5 = [];pr_r6 = [];pr_r7 = [];pr_r8 = [];pr_r9 = [];
            color1 = 'k';color2 = 'r';color3 = 'y';color4 = 'c';color5 = 'g';color6 = 'b';color7 = 'm';
            color8 = 'peru';color9 = 'indigo';
            # with open(saver_log_folder1 + 'accuracy.csv', 'r') as csvfile:
            #     data = csv.reader(csvfile, delimiter=',')
            #
            #     # retrieves rows of data and saves it as a list of list
            #     train_pose1 = [row for row in data]
            #
            #     # forces list as array to type cast as int
            #     int_tr_pose1 = np.array(train_pose1, float)

            with open(saver_log_folder2 + 'accuracy.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # retrieves rows of data and saves it as a list of list
                train_pose2 = [row for row in data]

                # forces list as array to type cast as int
                int_tr_pose2 = np.array(train_pose2, float)

            with open(saver_log_folder3 + 'accuracy.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # retrieves rows of data and saves it as a list of list
                train_pose3 = [row for row in data]

                # forces list as array to type cast as int
                int_tr_pose3 = np.array(train_pose3, float)

            with open(saver_log_folder4 + 'accuracy.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # retrieves rows of data and saves it as a list of list
                train_pose4 = [row for row in data]

                # forces list as array to type cast as int
                int_tr_pose4 = np.array(train_pose4, float)

            with open(saver_log_folder5 + 'accuracy.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # retrieves rows of data and saves it as a list of list
                train_pose5 = [row for row in data]

                # forces list as array to type cast as int
                int_tr_pose5 = np.array(train_pose5, float)

            with open(saver_log_folder6 + 'accuracy.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # retrieves rows of data and saves it as a list of list
                train_pose6 = [row for row in data]

                # forces list as array to type cast as int
                int_tr_pose6 = np.array(train_pose6, float)

            with open(saver_log_folder7 + 'accuracy.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # retrieves rows of data and saves it as a list of list
                train_pose7 = [row for row in data]

                # forces list as array to type cast as int
                int_tr_pose7 = np.array(train_pose7, float)

            with open(saver_log_folder8 + 'accuracy.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # retrieves rows of data and saves it as a list of list
                train_pose8 = [row for row in data]

                # forces list as array to type cast as int
                int_tr_pose8 = np.array(train_pose8, float)

            with open(saver_log_folder9 + 'accuracy.csv', 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # retrieves rows of data and saves it as a list of list
                train_pose9 = [row for row in data]

                # forces list as array to type cast as int
                int_tr_pose9 = np.array(train_pose9, float)
            column_index = [1, 2, 3]
            col = 2
            # loops through array
            column_values1 = np.empty(0);column_values2 = np.empty(0);column_values3 = np.empty(0);column_values4 = np.empty(0);
            column_values5 = np.empty(0);column_values6 = np.empty(0);column_values7 = np.empty(0);column_values8 = np.empty(0);
            column_values9 = np.empty(0);
            for array in range(len(int_tr_pose2)):
                # gets correct column number
               # column_values1 = np.append(column_values1, np.array(int_tr_pose1[array][col - 1]))
                column_values2 = np.append(column_values2, np.array(int_tr_pose2[array][col - 1]))
                column_values3 = np.append(column_values3, np.array(int_tr_pose3[array][col - 1]))
                column_values4 = np.append(column_values4, np.array(int_tr_pose4[array][col - 1]))
                column_values5 = np.append(column_values5, np.array(int_tr_pose5[array][col - 1]))
                column_values6 = np.append(column_values6, np.array(int_tr_pose6[array][col - 1]))
                column_values7 = np.append(column_values7, np.array(int_tr_pose7[array][col - 1]))
                column_values8 = np.append(column_values8, np.array(int_tr_pose8[array][col - 1]))
                column_values9 = np.append(column_values9, np.array(int_tr_pose9[array][col - 1]))
            size = len(column_values2);
            tt = np.arange(0.0, size, 1)
            plt.clf()
            plt.cla()
            fig = plt.figure()
            ax = plt.subplot(111)
            plt.ylim(0,1.5)
            rgb = np.random.rand(3, )
            #ax.plot(tt, column_values1, marker='', color = color1, linewidth=1, label=label1)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values2, marker='', color = color2, linewidth=1, label=label2)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values3, marker='', color = color3, linewidth=1, label=label3)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values4, marker='', color = color4, linewidth=1, label=label4)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values5, marker='', color = color5, linewidth=1, label=label5)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values6, marker='', color = color6, linewidth=1, label=label6)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values7, marker='', color = color7, linewidth=1, label=label7)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values8, marker='', color = color8, linewidth=1, label=label8)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values9, marker='', color = color9, linewidth=1, label=label9)
            plt.xlabel('Epoch')
            plt.ylabel('Error (meters)')
            plt.title('train-pose-rms - 64x64 images')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.27,
                         box.width, box.height * 0.8])

        # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.13),
                  fancybox=True, shadow=True, ncol=2)
            fig.savefig(saver_log_folder + "Error_train_pose.png");
            plt.close()
            s1_values = sorted(column_values1)
            s2_values = sorted(column_values2)
            s3_values = sorted(column_values3)
            s4_values = sorted(column_values4)
            s5_values = sorted(column_values5)
            s6_values = sorted(column_values6)
            s7_values = sorted(column_values7)
            s8_values = sorted(column_values8)
            s9_values = sorted(column_values9)

            for i in range(0, size, int(size / 10)):
                ss1_errors = s1_values[:i];ss2_errors = s2_values[:i];ss3_errors = s3_values[:i];
                ss4_errors = s4_values[:i];ss5_errors = s5_values[:i];ss6_errors = s6_values[:i];
                ss7_errors = s7_values[:i];ss8_errors = s8_values[:i];ss9_errors = s9_values[:i];
                for value in ss1_errors:
                    if value <= 0.3:
                        n1 = n1 + 1;
                for value in ss2_errors:
                    if value <= 0.3:
                        n2 = n2 + 1;

                for value in ss3_errors:
                    if value <= 0.3:
                        n3 = n3 + 1;
                for value in ss4_errors:
                    if value <= 0.3:
                        n4 = n4 + 1;

                for value in ss5_errors:
                    if value <= 0.3:
                        n5 = n5 + 1;
                for value in ss6_errors:
                    if value <= 0.3:
                        n6 = n6 + 1;
                for value in ss7_errors:
                    if value <= 0.3:
                        n7 = n7 + 1;

                for value in ss8_errors:
                    if value <= 0.3:
                        n8 = n8 + 1;
                for value in ss9_errors:
                    if value <= 0.3:
                        n9 = n9 + 1;
                ss1_errors.clear();ss2_errors.clear();ss3_errors.clear();ss4_errors.clear();ss5_errors.clear();
                ss6_errors.clear();ss7_errors.clear();ss8_errors.clear();ss9_errors.clear();
                if i == 0:
                    prec1 = 1;prec2 = 1;prec3 = 1;prec4 = 1;prec5 = 1;prec6 = 1;prec7 = 1;prec8 = 1;prec9 = 1;
                    n1=0;n2= 0;n3= 0;n4= 0;n5= 0;n6= 0;n7= 0;n8= 0;n9 = 0;
                    pr_t1.append(prec1);pr_t2.append(prec2);pr_t3.append(prec3);pr_t4.append(prec4);
                    pr_t5.append(prec5);pr_t6.append(prec6);pr_t7.append(prec7);pr_t8.append(prec8);pr_t9.append(prec9);
                else:
                    prec1 = n1 / i;prec2 = n2 / i;prec3 = n3 / i;prec4 = n4/ i;prec5 = n5 / i;prec6 = n6 / i;prec7 = n7 / i;prec8 = n8 / i;prec9 = n9 / i;
                    n1=0;n2= 0;n3= 0;n4= 0;n5= 0;n6= 0;n7= 0;n8= 0;n9 = 0;
                    pr_t1.append(prec1);pr_t2.append(prec2);pr_t3.append(prec3);pr_t4.append(prec4);pr_t5.append(prec5);
                    pr_t6.append(prec6);pr_t7.append(prec7);pr_t8.append(prec8);pr_t9.append(prec9);

            t = np.arange(0.0, 1.0, 0.1)
            plt.clf()
            plt.cla()
            fig = plt.figure()
            ax = plt.subplot(111)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_t1,  color = color1, linewidth=1, label=label1)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_t2,  color = color2, linewidth=1, label=label2)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_t3,  color = color3, linewidth=1, label=label3)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_t4,  color = color4, linewidth=1, label=label4)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_t5,  color = color5, linewidth=1, label=label5)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_t6,  color = color6, linewidth=1, label=label6)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_t7,  color = color7, linewidth=1, label=label7)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_t8,  color = color8, linewidth=1, label=label8)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_t9,  color = color9, linewidth=1, label=label9)
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.title('Precision-recall - train-pose - Cutoff = 0.3 m - 64x64 images')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.27,
                         box.width, box.height * 0.8])

                 # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.13),
                  fancybox=True, shadow=True, ncol=2)
            fig.savefig(saver_log_folder + "t_prec_rec.png");
            plt.close()
            col = 3
            # loops through array
            n=0;
            column_values1 = np.empty(0);
            column_values2 = np.empty(0);
            column_values3 = np.empty(0);
            column_values4 = np.empty(0);
            column_values5 = np.empty(0);
            column_values6 = np.empty(0);
            column_values7 = np.empty(0);
            column_values8 = np.empty(0);
            column_values9 = np.empty(0);
            for array in range(len(int_tr_pose2)):
                # gets correct column number
                #column_values1 = np.append(column_values1, np.array(int_tr_pose1[array][col - 1]))
                #column_values1[array] = column_values1[array] * math.pi / 180
                column_values2 = np.append(column_values2, np.array(int_tr_pose2[array][col - 1]))
                column_values2[array] = column_values2[array] * math.pi / 180
                column_values3 = np.append(column_values3, np.array(int_tr_pose3[array][col - 1]))
                column_values3[array] = column_values3[array] * math.pi / 180
                column_values4 = np.append(column_values4, np.array(int_tr_pose4[array][col - 1]))
                column_values4[array] = column_values4[array] * math.pi / 180
                column_values5 = np.append(column_values5, np.array(int_tr_pose5[array][col - 1]))
                column_values5[array] = column_values5[array] * math.pi / 180
                column_values6 = np.append(column_values6, np.array(int_tr_pose6[array][col - 1]))
                column_values6[array] = column_values6[array] * math.pi / 180
                column_values7 = np.append(column_values7, np.array(int_tr_pose7[array][col - 1]))
                column_values7[array] = column_values7[array] * math.pi / 180
                column_values8 = np.append(column_values8, np.array(int_tr_pose8[array][col - 1]))
                column_values8[array] = column_values8[array] * math.pi / 180
                column_values9 = np.append(column_values9, np.array(int_tr_pose9[array][col - 1]))
                column_values9[array] = column_values9[array] * math.pi / 180
            size = len(column_values2);
            tt = np.arange(0.0, size, 1)
            plt.clf()
            plt.cla()
            fig = plt.figure()
            ax = plt.subplot(111)
            plt.ylim(0, 2.0)
            rgb = np.random.rand(3, )
            #ax.plot(tt, column_values1, marker='', color = color1, linewidth=1, label=label1)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values2, marker='', color = color2, linewidth=1, label=label2)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values3, marker='', color = color3, linewidth=1, label=label3)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values4, marker='', color = color4, linewidth=1, label=label4)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values5, marker='', color = color5, linewidth=1, label=label5)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values6, marker='', color = color6, linewidth=1, label=label6)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values7, marker='', color = color7, linewidth=1, label=label7)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values8, marker='', color = color8, linewidth=1, label=label8)
            rgb = np.random.rand(3, )
            ax.plot(tt, column_values9, marker='', color = color9, linewidth=1, label=label9)
            plt.xlabel('Epoch')
            plt.ylabel('Error (Radians)')
            plt.title('train-rot-rms - 64x64 images')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.27,
                         box.width, box.height * 0.8])

        # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.13),
                  fancybox=True, shadow=True, ncol=2)
            fig.savefig(saver_log_folder + "Error_train_rot.png");
            plt.close()
            s1r_values = sorted(column_values1)
            s2r_values = sorted(column_values2)
            s3r_values = sorted(column_values3)
            s4r_values = sorted(column_values4)
            s5r_values = sorted(column_values5)
            s6r_values = sorted(column_values6)
            s7r_values = sorted(column_values7)
            s8r_values = sorted(column_values8)
            s9r_values = sorted(column_values9)

            for i in range(0, size, int(size / 10)):
                ss1_errors = s1r_values[:i];
                ss2_errors = s2r_values[:i];
                ss3_errors = s3r_values[:i];
                ss4_errors = s4r_values[:i];
                ss5_errors = s5r_values[:i];
                ss6_errors = s6r_values[:i];
                ss7_errors = s7r_values[:i];
                ss8_errors = s8r_values[:i];
                ss9_errors = s9r_values[:i];
                for value in ss1_errors:
                    if value <= 0.11:
                        n1 = n1 + 1;
                for value in ss2_errors:
                    if value <= 0.11:
                        n2 = n2 + 1;

                for value in ss3_errors:
                    if value <= 0.11:
                        n3 = n3 + 1;
                for value in ss4_errors:
                    if value <= 0.11:
                        n4 = n4 + 1;

                for value in ss5_errors:
                    if value <= 0.11:
                        n5 = n5 + 1;
                for value in ss6_errors:
                    if value <= 0.11:
                        n6 = n6 + 1;
                for value in ss7_errors:
                    if value <= 0.11:
                        n7 = n7 + 1;

                for value in ss8_errors:
                    if value <= 0.11:
                        n8 = n8 + 1;
                for value in ss9_errors:
                    if value <= 0.11:
                        n9 = n9 + 1;
                ss1_errors.clear();
                ss2_errors.clear();
                ss3_errors.clear();
                ss4_errors.clear();
                ss5_errors.clear();
                ss6_errors.clear();
                ss7_errors.clear();
                ss8_errors.clear();
                ss9_errors.clear();
                if i == 0:
                    prec1 = 1;
                    prec2 = 1;
                    prec3 = 1;
                    prec4 = 1;
                    prec5 = 1;
                    prec6 = 1;
                    prec7 = 1;
                    prec8 = 1;
                    prec9 = 1;
                    n1=0;n2= 0;n3= 0;n4= 0;n5= 0;n6= 0;n7= 0;n8= 0;n9 = 0;
                    pr_r1.append(prec1);
                    pr_r2.append(prec2);
                    pr_r3.append(prec3);
                    pr_r4.append(prec4);
                    pr_r5.append(prec5);
                    pr_r6.append(prec6);
                    pr_r7.append(prec7);
                    pr_r8.append(prec8);
                    pr_r9.append(prec9);
                else:
                    prec1 = n1 / i;
                    prec2 = n2 / i;
                    prec3 = n3 / i;
                    prec4 = n4 / i;
                    prec5 = n5 / i;
                    prec6 = n6 / i;
                    prec7 = n7 / i;
                    prec8 = n8 / i;
                    prec9 = n9 / i;
                    n1=0;n2= 0;n3= 0;n4= 0;n5= 0;n6= 0;n7= 0;n8= 0;n9 = 0;
                    pr_r1.append(prec1);
                    pr_r2.append(prec2);
                    pr_r3.append(prec3);
                    pr_r4.append(prec4);
                    pr_r5.append(prec5);
                    pr_r6.append(prec6);
                    pr_r7.append(prec7);
                    pr_r8.append(prec8);
                    pr_r9.append(prec9);

            t = np.arange(0.0, 1.0, 0.1)
            plt.clf()
            plt.cla()
            fig = plt.figure()
            ax = plt.subplot(111)
            rgb = np.random.rand(3, )
            #ax.plot(t, pr_r1,  color = color1, linewidth=1, label=label1)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_r2,  color = color2, linewidth=1, label=label2)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_r3,  color = color3, linewidth=1, label=label3)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_r4,  color = color4, linewidth=1, label=label4)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_r5,  color = color5, linewidth=1, label=label5)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_r6,  color = color6, linewidth=1, label=label6)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_r7,  color = color7, linewidth=1, label=label7)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_r8,  color = color8, linewidth=1, label=label8)
            rgb = np.random.rand(3, )
            ax.plot(t, pr_r9,  color = color9, linewidth=1, label=label9)
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.title('Precision-recall - train-orient - Cutoff = 0.11 rad - 64x64 images')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.27,
                         box.width, box.height * 0.8])

        # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.13),
                  fancybox=True, shadow=True, ncol=2)
            fig.savefig(saver_log_folder + "r_prec_rec.png");
            plt.close()

    def plot_test_64(self, _file_loc1, _file_loc2, _file_loc3, _file_loc4, _file_loc5, _file_loc6, _file_loc7, _file_loc8,_file_loc9,saver_folder):
        n_exp =9;
        saver_log_folder = saver_folder;saver_log_folder1 = _file_loc1;saver_log_folder2 = _file_loc2;
        saver_log_folder3 = _file_loc3;saver_log_folder4 = _file_loc4;saver_log_folder5 = _file_loc5;
        saver_log_folder6 = _file_loc6;saver_log_folder7 = _file_loc7;saver_log_folder8 = _file_loc8;
        saver_log_folder9 = _file_loc9;
        label1 ="1: seg layers = 4, lr = 0.001";label2 ="2: seg layers = 4, lr = 0.01";label3 ="3: seg layers = 4, lr = 0.1";
        label4 = "4: seg layers = 8, lr = 0.001";label5 ="5: seg layers = 8, lr = 0.01";label6 ="6: seg layers = 8, lr = 0.1";
        label7 = "7: seg layers = 10, lr = 0.001";label8 ="8: seg layers = 10, lr = 0.01";label9 ="9: seg layers = 10, lr = 0.1";
        pr_t1 = [];pr_t2 = [];pr_t3 = [];pr_t4 = [];pr_t5 = [];pr_t6 = [];pr_t7 = [];pr_t8 = [];pr_t9 = [];
        pr_r1 = [];pr_r2 = [];pr_r3 = [];pr_r4 = [];pr_r5 = [];pr_r6 = [];pr_r7 = [];pr_r8 = [];pr_r9 = [];

        color1 = 'k';color2 = 'r'; color3 = 'y';color4 = 'c';color5 = 'g';color6 = 'b';color7 = 'm';
        color8 = 'peru';color9 = 'indigo';

        # with open(saver_log_folder1 + 'test_pose_acc.csv', 'r') as csvfile:
        #     data = csv.reader(csvfile, delimiter=',')
        #
        #     # retrieves rows of data and saves it as a list of list
        #     train_pose1 = [row for row in data]
        #
        #     # forces list as array to type cast as int
        #     int_tr_pose1 = np.array(train_pose1, float)

        with open(saver_log_folder2 + 'test_pose_acc.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')

            # retrieves rows of data and saves it as a list of list
            train_pose2 = [row for row in data]

            # forces list as array to type cast as int
            int_tr_pose2 = np.array(train_pose2, float)

        with open(saver_log_folder3 + 'test_pose_acc.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')

            # retrieves rows of data and saves it as a list of list
            train_pose3 = [row for row in data]

            # forces list as array to type cast as int
            int_tr_pose3 = np.array(train_pose3, float)

        with open(saver_log_folder4 + 'test_pose_acc.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')

            # retrieves rows of data and saves it as a list of list
            train_pose4 = [row for row in data]

            # forces list as array to type cast as int
            int_tr_pose4 = np.array(train_pose4, float)

        # with open(saver_log_folder5 + 'test_pose_acc.csv', 'r') as csvfile:
        #     data = csv.reader(csvfile, delimiter=',')

            # retrieves rows of data and saves it as a list of list
            train_pose5 = [row for row in data]

            # forces list as array to type cast as int
            int_tr_pose5 = np.array(train_pose5, float)

        with open(saver_log_folder6 + 'test_pose_acc.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')

            # retrieves rows of data and saves it as a list of list
            train_pose6 = [row for row in data]

            # forces list as array to type cast as int
            int_tr_pose6 = np.array(train_pose6, float)

        with open(saver_log_folder7 + 'test_pose_acc.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')

            # retrieves rows of data and saves it as a list of list
            train_pose7 = [row for row in data]

            # forces list as array to type cast as int
            int_tr_pose7 = np.array(train_pose7, float)

        with open(saver_log_folder8 + 'test_pose_acc.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')

            # retrieves rows of data and saves it as a list of list
            train_pose8 = [row for row in data]

            # forces list as array to type cast as int
            int_tr_pose8 = np.array(train_pose8, float)

        with open(saver_log_folder9 + 'test_pose_acc.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')

            # retrieves rows of data and saves it as a list of list
            train_pose9 = [row for row in data]

            # forces list as array to type cast as int
            int_tr_pose9 = np.array(train_pose9, float)

        column_index = [1, 2, 3]
        col = 1
        # loops through array
        column_values1 = np.empty(0);column_values2 = np.empty(0);column_values3 = np.empty(0);column_values4 = np.empty(0);
        column_values5 = np.empty(0);column_values6 = np.empty(0);column_values7 = np.empty(0);column_values8 = np.empty(0);
        column_values9 = np.empty(0);
        for array in range(len(int_tr_pose2)):
            # gets correct column number
            #column_values1 = np.append(column_values1, np.array(int_tr_pose1[array][col - 1]))
            column_values2 = np.append(column_values2, np.array(int_tr_pose2[array][col - 1]))
            column_values3 = np.append(column_values3, np.array(int_tr_pose3[array][col - 1]))
            column_values4 = np.append(column_values4, np.array(int_tr_pose4[array][col - 1]))
           # column_values5 = np.append(column_values5, np.array(int_tr_pose5[array][col - 1]))
            column_values6 = np.append(column_values6, np.array(int_tr_pose6[array][col - 1]))
            column_values7 = np.append(column_values7, np.array(int_tr_pose7[array][col - 1]))
            column_values8 = np.append(column_values8, np.array(int_tr_pose8[array][col - 1]))
            column_values9 = np.append(column_values9, np.array(int_tr_pose9[array][col - 1]))
        size = len(column_values2);
        tt = np.arange(0.0, size, 1)
        plt.clf()
        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.ylim(0,1.5)
        rgb = np.random.rand(3, )
        #ax.plot(tt, column_values1, color = color1, linewidth=1, label=label1)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values2, color = color2, linewidth=1, label=label2)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values3, color = color3, linewidth=1, label=label3)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values4, color = color4, linewidth=1, label=label4)
        rgb = np.random.rand(3, )
       # ax.plot(tt, column_values5, color = color5, linewidth=1, label=label5)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values6, color = color6, linewidth=1, label=label6)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values7, color = color7, linewidth=1, label=label7)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values8, color = color8, linewidth=1, label=label8)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values9, color = color9, linewidth=1, label=label9)
        plt.xlabel('Test Images')
        plt.ylabel('Error (meters)')
        plt.title('Test-pose-rms - 64x64 images')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.27,
                         box.width, box.height * 0.8])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.13),
                  fancybox=True, shadow=True, ncol=2)
        fig.savefig(saver_log_folder + "Error_Test_pose.png");
        plt.close()
        s1_values = sorted(column_values1)
        s2_values = sorted(column_values2)
        s3_values = sorted(column_values3)
        s4_values = sorted(column_values4)
        s5_values = sorted(column_values5)
        s6_values = sorted(column_values6)
        s7_values = sorted(column_values7)
        s8_values = sorted(column_values8)
        s9_values = sorted(column_values9)

        for i in range(0, size, int(size / 10)):
            ss1_errors = s1_values[:i];ss2_errors = s2_values[:i];ss3_errors = s3_values[:i];
            ss4_errors = s4_values[:i];ss5_errors = s5_values[:i];ss6_errors = s6_values[:i];
            ss7_errors = s7_values[:i];ss8_errors = s8_values[:i];ss9_errors = s9_values[:i];
            for value in ss1_errors:
                if value <= 0.3:
                    n1 = n1 + 1;
            for value in ss2_errors:
                if value <= 0.3:
                    n2 = n2 + 1;

            for value in ss3_errors:
                if value <= 0.3:
                    n3 = n3 + 1;
            for value in ss4_errors:
                if value <= 0.3:
                    n4 = n4 + 1;

            for value in ss5_errors:
                if value <= 0.3:
                    n5 = n5 + 1;
            for value in ss6_errors:
                if value <= 0.3:
                    n6 = n6 + 1;
            for value in ss7_errors:
                if value <= 0.3:
                    n7 = n7 + 1;

            for value in ss8_errors:
                if value <= 0.3:
                    n8 = n8 + 1;
            for value in ss9_errors:
                if value <= 0.3:
                    n9 = n9 + 1;
            ss1_errors.clear();ss2_errors.clear();ss3_errors.clear();ss4_errors.clear();ss5_errors.clear();
            ss6_errors.clear();ss7_errors.clear();ss8_errors.clear();ss9_errors.clear();
            if i == 0:
                prec1 = 1;prec2 = 1;prec3 = 1;prec4 = 1;prec5 = 1;prec6 = 1;prec7 = 1;prec8 = 1;prec9 = 1;
                n1=0;n2= 0;n3= 0;n4= 0;n5= 0;n6= 0;n7= 0;n8= 0;n9 = 0;
                pr_t1.append(prec1);pr_t2.append(prec2);pr_t3.append(prec3);pr_t4.append(prec4);
                pr_t5.append(prec5);pr_t6.append(prec6);pr_t7.append(prec7);pr_t8.append(prec8);pr_t9.append(prec9);
            else:
                prec1 = n1 / i;prec2 = n2 / i;prec3 = n3 / i;prec4 = n4/ i;prec5 = n5 / i;prec6 = n6 / i;prec7 = n7 / i;prec8 = n8 / i;prec9 = n9 / i;
                n1=0;n2= 0;n3= 0;n4= 0;n5= 0;n6= 0;n7= 0;n8= 0;n9 = 0;
                pr_t1.append(prec1);pr_t2.append(prec2);pr_t3.append(prec3);pr_t4.append(prec4);pr_t5.append(prec5);
                pr_t6.append(prec6);pr_t7.append(prec7);pr_t8.append(prec8);pr_t9.append(prec9);

        t = np.arange(0.0, 1.0, 0.1)
        plt.clf()
        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_t1,  color = color1, linewidth=1, label=label1)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_t2,  color = color2, linewidth=1, label=label2)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_t3,  color = color3, linewidth=1, label=label3)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_t4,  color = color4, linewidth=1, label=label4)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_t5,  color = color5, linewidth=1, label=label5)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_t6,  color = color6, linewidth=1, label=label6)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_t7,  color = color7, linewidth=1, label=label7)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_t8,  color = color8, linewidth=1, label=label8)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_t9,  color = color9, linewidth=1, label=label9)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('Precision-recall - Test-pose - Cutoff = 0.3 m - 64x64 images')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.27,
                         box.width, box.height * 0.8])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.13),
                  fancybox=True, shadow=True, ncol=2)
        fig.savefig(saver_log_folder + "Test_t_p_r.png");
        plt.close()
        col = 2
        # loops through array
        n=0;
        column_values1 = np.empty(0);
        column_values2 = np.empty(0);
        column_values3 = np.empty(0);
        column_values4 = np.empty(0);
        column_values5 = np.empty(0);
        column_values6 = np.empty(0);
        column_values7 = np.empty(0);
        column_values8 = np.empty(0);
        column_values9 = np.empty(0);
        for array in range(len(int_tr_pose2)):
            # gets correct column number
            #column_values1 = np.append(column_values1, np.array(int_tr_pose1[array][col - 1]))
            #column_values1[array] = column_values1[array] * math.pi / 180
            column_values2 = np.append(column_values2, np.array(int_tr_pose2[array][col - 1]))
            #column_values2[array] = column_values2[array] * math.pi / 180
            column_values3 = np.append(column_values3, np.array(int_tr_pose3[array][col - 1]))
            #column_values3[array] = column_values3[array] * math.pi / 180
            column_values4 = np.append(column_values4, np.array(int_tr_pose4[array][col - 1]))
            #column_values4[array] = column_values4[array] * math.pi / 180
            #column_values5 = np.append(column_values5, np.array(int_tr_pose5[array][col - 1]))
            #column_values5[array] = column_values5[array] * math.pi / 180
            column_values6 = np.append(column_values6, np.array(int_tr_pose6[array][col - 1]))
            #column_values6[array] = column_values6[array] * math.pi / 180
            column_values7 = np.append(column_values7, np.array(int_tr_pose7[array][col - 1]))
            #column_values7[array] = column_values7[array] * math.pi / 180
            column_values8 = np.append(column_values8, np.array(int_tr_pose8[array][col - 1]))
            #column_values8[array] = column_values8[array] * math.pi / 180
            column_values9 = np.append(column_values9, np.array(int_tr_pose9[array][col - 1]))
            #column_values9[array] = column_values9[array] * math.pi / 180
        size = len(column_values2);
        tt = np.arange(0.0, size, 1)
        plt.clf()
        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.ylim(0, 2.0)
        rgb = np.random.rand(3, )
        #ax.plot(tt, column_values1, color = color1, linewidth=1, label=label1)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values2, color = color2, linewidth=1, label=label2)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values3, color = color3, linewidth=1, label=label3)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values4, color = color4, linewidth=1, label=label4)
        rgb = np.random.rand(3, )
       # ax.plot(tt, column_values5, color = color5, linewidth=1, label=label5)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values6, color = color6, linewidth=1, label=label6)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values7, color = color7, linewidth=1, label=label7)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values8, color = color8, linewidth=1, label=label8)
        rgb = np.random.rand(3, )
        ax.plot(tt, column_values9, color = color9, linewidth=1, label=label9)
        plt.xlabel('Test Images')
        plt.ylabel('Error (radians)')
        plt.title('Test-rot-rms - 64x64 images')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.27,
                         box.width, box.height * 0.8])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.13),
                  fancybox=True, shadow=True, ncol=2)
        fig.savefig(saver_log_folder + "Error_Test_rot.png");
        plt.close()
        s1r_values = sorted(column_values1)
        s2r_values = sorted(column_values2)
        s3r_values = sorted(column_values3)
        s4r_values = sorted(column_values4)
        s5r_values = sorted(column_values5)
        s6r_values = sorted(column_values6)
        s7r_values = sorted(column_values7)
        s8r_values = sorted(column_values8)
        s9r_values = sorted(column_values9)

        for i in range(0, size, int(size / 10)):
            ss1_errors = s1r_values[:i];
            ss2_errors = s2r_values[:i];
            ss3_errors = s3r_values[:i];
            ss4_errors = s4r_values[:i];
            ss5_errors = s5r_values[:i];
            ss6_errors = s6r_values[:i];
            ss7_errors = s7r_values[:i];
            ss8_errors = s8r_values[:i];
            ss9_errors = s9r_values[:i];
            for value in ss1_errors:
                if value <= 0.11:
                    n1 = n1 + 1;
            for value in ss2_errors:
                if value <= 0.11:
                    n2 = n2 + 1;

            for value in ss3_errors:
                if value <= 0.11:
                    n3 = n3 + 1;
            for value in ss4_errors:
                if value <= 0.11:
                    n4 = n4 + 1;

            for value in ss5_errors:
                if value <= 0.11:
                    n5 = n5 + 1;
            for value in ss6_errors:
                if value <= 0.11:
                    n6 = n6 + 1;
            for value in ss7_errors:
                if value <= 0.11:
                    n7 = n7 + 1;

            for value in ss8_errors:
                if value <= 0.11:
                    n8 = n8 + 1;
            for value in ss9_errors:
                if value <= 0.11:
                    n9 = n9 + 1;
            ss1_errors.clear();
            ss2_errors.clear();
            ss3_errors.clear();
            ss4_errors.clear();
            ss5_errors.clear();
            ss6_errors.clear();
            ss7_errors.clear();
            ss8_errors.clear();
            ss9_errors.clear();
            if i == 0:
                prec1 = 1;
                prec2 = 1;
                prec3 = 1;
                prec4 = 1;
                prec5 = 1;
                prec6 = 1;
                prec7 = 1;
                prec8 = 1;
                prec9 = 1;
                n1=0;n2= 0;n3= 0;n4= 0;n5= 0;n6= 0;n7= 0;n8= 0;n9 = 0;
                pr_r1.append(prec1);
                pr_r2.append(prec2);
                pr_r3.append(prec3);
                pr_r4.append(prec4);
                pr_r5.append(prec5);
                pr_r6.append(prec6);
                pr_r7.append(prec7);
                pr_r8.append(prec8);
                pr_r9.append(prec9);
            else:
                prec1 = n1 / i;
                prec2 = n2 / i;
                prec3 = n3 / i;
                prec4 = n4 / i;
                prec5 = n5 / i;
                prec6 = n6 / i;
                prec7 = n7 / i;
                prec8 = n8 / i;
                prec9 = n9 / i;
                n1=0;n2= 0;n3= 0;n4= 0;n5= 0;n6= 0;n7= 0;n8= 0;n9 = 0;
                pr_r1.append(prec1);
                pr_r2.append(prec2);
                pr_r3.append(prec3);
                pr_r4.append(prec4);
                pr_r5.append(prec5);
                pr_r6.append(prec6);
                pr_r7.append(prec7);
                pr_r8.append(prec8);
                pr_r9.append(prec9);

        t = np.arange(0.0, 1.0, 0.1)
        plt.clf()
        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)
        rgb = np.random.rand(3, )
        #ax.plot(t, pr_r1, color = color1, linewidth=1, label=label1)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_r2,  color = color2, linewidth=1, label=label2)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_r3,  color = color3, linewidth=1, label=label3)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_r4,  color = color4, linewidth=1, label=label4)
        rgb = np.random.rand(3, )
      #  ax.plot(t, pr_r5,  color = color5, linewidth=1, label=label5)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_r6,  color = color6, linewidth=1, label=label6)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_r7,  color = color7, linewidth=1, label=label7)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_r8,  color = color8, linewidth=1, label=label8)
        rgb = np.random.rand(3, )
        ax.plot(t, pr_r9,  color = color9, linewidth=1, label=label9)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('Precision-recall - Test-orient - Cutoff = 0.11 rad - 64x64 images')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.27,
                         box.width, box.height * 0.8])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.13),
                  fancybox=True, shadow=True, ncol=2)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='lower center',mode="expand", borderaxespad=0.)
        fig.savefig(saver_log_folder + "Test_r_p_r.png");
        plt.close()