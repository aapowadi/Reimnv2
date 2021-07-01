import matplotlib.pyplot as plt
import csv
import numpy as np
def main():
    saver_log_folder = "./log_real_128/";
    saver_log_folder1 = "./log_real_128/ex1n";
    saver_log_folder2 = "./log_real_128/ex6n/";
    saver_log_folder3 = "./log_real_128/ex12n/";
    saver_log_folder4 = "./log_real_128/ex13n/";
    saver_log_folder5 = "./log_real_128/ex14n/";
    saver_log_folder6 = "./log_real_128/ex15n/";
    # saver_log_folder7 = _file_loc7;
    # saver_log_folder8 = _file_loc8;
    # saver_log_folder9 = _file_loc9;
    label1 = "1: seg layers = 8n, keep_rate = 0.8";
    label2 = "2: seg layers = 8n,keep_rate = 0.85";
    label3 = "3: seg layers = 8n,keep_rate = 0.8";
    label4 = "4: seg layers = 8n,keep_rate = 0.8";
    label5 = "5: seg layers = 8n,keep_rate = 0.85";
    label6 = "6: seg layers = 8n,keep_rate = 0.85";
    # label7 = "7: seg layers = 10, lr = 0.001";
    # label8 = "8: seg layers = 10, lr = 0.01";
    # label9 = "9: seg layers = 10, lr = 0.1";

    x = []
    y = []
    pr_t1 = [];
    pr_t2 = [];
    pr_t3 = [];
    pr_t4 = [];
    pr_t5 = [];
    pr_t6 = [];
    pr_t7 = [];
    pr_t8 = [];
    pr_t9 = [];
    pr_r1 = [];
    pr_r2 = [];
    pr_r3 = [];
    pr_r4 = [];
    pr_r5 = [];
    pr_r6 = [];
    pr_r7 = [];
    pr_r8 = [];
    pr_r9 = [];
    color1 = 'k';
    color2 = 'r';
    color3 = 'y';
    color4 = 'c';
    color5 = 'g';
    color6 = 'b';
    color7 = 'm';
    color8 = 'peru';
    color9 = 'indigo';
    with open(saver_log_folder1 + 'train_loss.csv', 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')

        # retrieves rows of data and saves it as a list of list
        train_pose1 = [row for row in data]

        # forces list as array to type cast as int
        int_tr_pose1 = np.array(train_pose1, float)

    with open(saver_log_folder2 + 'train_loss.csv', 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')

        # retrieves rows of data and saves it as a list of list
        train_pose2 = [row for row in data]

        # forces list as array to type cast as int
        int_tr_pose2 = np.array(train_pose2, float)

    with open(saver_log_folder3 + 'train_loss.csv', 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')

        # retrieves rows of data and saves it as a list of list
        train_pose3 = [row for row in data]

        # forces list as array to type cast as int
        int_tr_pose3 = np.array(train_pose3, float)

    with open(saver_log_folder4 + 'train_loss.csv', 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')

        # retrieves rows of data and saves it as a list of list
        train_pose4 = [row for row in data]

        # forces list as array to type cast as int
        int_tr_pose4 = np.array(train_pose4, float)

    with open(saver_log_folder5 + 'train_loss.csv', 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')

        # retrieves rows of data and saves it as a list of list
        train_pose5 = [row for row in data]

        # forces list as array to type cast as int
        int_tr_pose5 = np.array(train_pose5, float)

    with open(saver_log_folder6 + 'train_loss.csv', 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')

        # retrieves rows of data and saves it as a list of list
        train_pose6 = [row for row in data]

        # forces list as array to type cast as int
        int_tr_pose6 = np.array(train_pose6, float)

    # with open(saver_log_folder7 + 'train_loss.csv', 'r') as csvfile:
    #     data = csv.reader(csvfile, delimiter=',')
    #
    #     # retrieves rows of data and saves it as a list of list
    #     train_pose7 = [row for row in data]
    #
    #     # forces list as array to type cast as int
    #     int_tr_pose7 = np.array(train_pose7, float)
    #
    # with open(saver_log_folder8 + 'train_loss.csv', 'r') as csvfile:
    #     data = csv.reader(csvfile, delimiter=',')
    #
    #     # retrieves rows of data and saves it as a list of list
    #     train_pose8 = [row for row in data]
    #
    #     # forces list as array to type cast as int
    #     int_tr_pose8 = np.array(train_pose8, float)
    #
    # with open(saver_log_folder9 + 'train_loss.csv', 'r') as csvfile:
    #     data = csv.reader(csvfile, delimiter=',')
    #
    #     # retrieves rows of data and saves it as a list of list
    #     train_pose9 = [row for row in data]
    #
    #     # forces list as array to type cast as int
    #     int_tr_pose9 = np.array(train_pose9, float)
    column_index = [1, 2, 3]
    col = 1
    # loops through array
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
        column_values1 = np.append(column_values1, np.array(int_tr_pose1[array][col - 1]))
        column_values2 = np.append(column_values2, np.array(int_tr_pose2[array][col - 1]))
        column_values3 = np.append(column_values3, np.array(int_tr_pose3[array][col - 1]))
        column_values4 = np.append(column_values4, np.array(int_tr_pose4[array][col - 1]))
        column_values5 = np.append(column_values5, np.array(int_tr_pose5[array][col - 1]))
        column_values6 = np.append(column_values6, np.array(int_tr_pose6[array][col - 1]))
        # column_values7 = np.append(column_values7, np.array(int_tr_pose7[array][col - 1]))
        # column_values8 = np.append(column_values8, np.array(int_tr_pose8[array][col - 1]))
        # column_values9 = np.append(column_values9, np.array(int_tr_pose9[array][col - 1]))
    size = len(column_values2);
    tt = np.arange(0.0, size, 1)
    plt.clf()
    plt.cla()
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.ylim(0, 1.5)
    rgb = np.random.rand(3, )
    ax.plot(tt, column_values1, marker='', color = color1, linewidth=1, label=label1)
    rgb = np.random.rand(3, )
    ax.plot(tt, column_values2, marker='', color=color2, linewidth=1, label=label2)
    rgb = np.random.rand(3, )
    ax.plot(tt, column_values3, marker='', color=color3, linewidth=1, label=label3)
    rgb = np.random.rand(3, )
    ax.plot(tt, column_values4, marker='', color=color4, linewidth=1, label=label4)
    rgb = np.random.rand(3, )
    ax.plot(tt, column_values5, marker='', color=color5, linewidth=1, label=label5)
    rgb = np.random.rand(3, )
    ax.plot(tt, column_values6, marker='', color=color6, linewidth=1, label=label6)
    rgb = np.random.rand(3, )
    # ax.plot(tt, column_values7, marker='', color=color7, linewidth=1, label=label7)
    # rgb = np.random.rand(3, )
    # ax.plot(tt, column_values8, marker='', color=color8, linewidth=1, label=label8)
    # rgb = np.random.rand(3, )
    # ax.plot(tt, column_values9, marker='', color=color9, linewidth=1, label=label9)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('train_loss - 128x128 images')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.27,
                     box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
              fancybox=True, shadow=True, ncol=2)
    fig.savefig(saver_log_folder + "Losses.png");
    plt.close()


if __name__ == "__main__":
    main()