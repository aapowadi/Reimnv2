import pathlib
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
def get_prec_rec_values(logs,im_width, im_height,batch = False):

        if (os.path.exists(logs + 'accuracy_results.csv')):
            with open(logs + 'accuracy_results.csv', 'r') as csvfile:
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


        return clm_prec_values , clm_rec_values , clm_prect_values , clm_rect_values

def get_loss_values(logs,im_width, im_height,batch = False):

    if (os.path.exists(logs+ 'losses.csv')):
        with open(logs + 'losses.csv', 'r') as csvfile:
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
            seg_train_loss = column_values

            col = 3
            # loops through array
            column_values = np.empty(0)
            for array in range(len(int_x)):
                # gets correct column number
                column_values = np.append(column_values, np.array(int_x[array][col - 1]))
            size = len(column_values);
            seg_test_loss = column_values;

        return seg_train_loss,seg_test_loss