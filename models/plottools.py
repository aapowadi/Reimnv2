import matplotlib.pyplot as plt
import numpy as np
import csv



def plotPrecisionReconFromFile(path_and_file, plot_title):


    csv_dataset = []
    try:
        with open(path_and_file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            print("[INFO] - Reading content from ", path_and_file)
            for row in csv_reader:
                if line_count == 0:
                    print('Column names are {", ".join(row)}')
                    line_count += 1
                line_count += 1

                data = {
                    "index": row['idx'],
                    "pr": row['precision'],
                    "rc": row['recall']
                }

                csv_dataset.append(data)

            print('[INFO] - Read {len(csv_dataset)} lines.')  # -1, to not count the header line
    except IOError:
        print('[ERROR] - Cannot open csv file {csv_file}. Check the folder')
        return

    if len(csv_dataset) == 0:
        print('[ERROR] - No data read')
        return

    # sort all items
    def takePr(elem):
        return elem["pr"]
    csv_dataset.sort(key=takePr, reverse=True)

    # split the data
    data_x = [float(d['rc']) for d in csv_dataset]
    data_y = [float(d['pr']) for d in csv_dataset]

    chunk_len = 100
    N = len(data_x)
    chunks = N / chunk_len

    plot_y = []
    plot_x = []

    for i in range(int(np.ceil(chunks))):
        start = i * chunk_len
        stop = start + chunk_len
        if stop > N:
            stop = N
        c_pr = data_y[0: stop]
        c_re = data_x[0: stop]

        avg_pr = np.sum(c_pr) / len(c_pr)
        avg_re = np.sum(c_re) / len(c_re)
        avg_re = avg_re * float(len(c_re)/N)

        plot_y.append(avg_pr)
        plot_x.append(avg_re)


    plt.plot(plot_x, plot_y, '-o')
    plt.ylim([0, 1.2])
    plt.xlim([0, 1.2])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(plot_title)

    path = plot_title + ".png"
    plt.savefig(path)
    plt.show()


#plotPrecisionReconFromFile("./log/1/pr-re.csv", "Bunny experiment_01")