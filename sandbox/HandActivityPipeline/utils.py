import numpy as np
import csv
import glob
from os.path import join
from operator import itemgetter

import math
import matplotlib.pyplot as plt
from numpy import interp

def extract_features(path_features, num_gestures, label_encoder, test_file_idx, file_names):

    path_file_names = glob.glob(join(path_features, '*.csv'))

    num_files = len(path_file_names)
    num_video_files = int(num_files / num_gestures)

    train_files = []
    test_files = []

    data_train = []
    data_test = []

    labels_train = []
    labels_test = []

    frame_counter_train = 0
    frame_counter_test = 0

    labels_range_train = [0]
    labels_range_test = [0]

    file_counter = 0

    for path_file_name in path_file_names:
        gesture_data = []
        if not "false" in path_file_name:
            # print(file_name)
            label_idx = int(path_file_name.rsplit("\\", 1)[-1].split(".")[0][0]) - 1
            file_name = path_file_name.rsplit("\\", 1)[-1].split(".")[0].rsplit('_',1)[0:][0][2:]
            is_other_feature = int(path_file_name.rsplit("\\", 1)[-1].split(".")[0].rsplit('_',1)[0][0]) == 0
            a = 1
            if file_name != file_names[test_file_idx]: #file_counter not in test_file_idx:
                train_files.append(path_file_name)
            else:
                test_files.append(path_file_name)
            with open(path_file_name, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                for row in csv_reader:
                    try:
                        gesture_data.append(row)
                        if file_name != file_names[test_file_idx]: # file_counter%num_video_files not in test_file_idx:
                            data_train.append(row)
                            frame_counter_train += 1
                        else:
                            data_test.append(row)
                            frame_counter_test += 1
                    except IndexError:
                        gesture_data.append(-1)
            if file_name != file_names[test_file_idx]: # file_counter%num_video_files not in test_file_idx:
                labels_range_train.append(frame_counter_train)
                labels_train.append( label_encoder[label_idx] )
            else:
                labels_range_test.append(frame_counter_test)
                labels_test.append( label_encoder[label_idx] )
        file_counter += 1
    data_train = np.array(data_train, dtype=np.float)
    labels_train = np.array(labels_train)
    labels_range_train = np.array(labels_range_train)
    data_test = np.array(data_test, dtype=np.float)
    labels_test = np.array(labels_test)
    labels_range_test = np.array(labels_range_test)

    return data_train, labels_train, labels_range_train, data_test, labels_test, labels_range_test


def spatio_temporal_features(data, labels_range, labels, num_features, label_names, plot_name=[]):
    N = num_features
    STF = []
    if plot_name != []:
        fig = plt.figure()
    for i in range(0, len(labels_range) - 1):
        num_frames = labels_range[i + 1] - labels_range[i]
        step = int(num_frames / N)
        idx = labels_range[i]
        features = np.zeros((2, N))
        features_x = np.zeros((N, 1))
        features_y = np.zeros((N, 1))
        for f in range(0, N):
            features[0, f] = round(interp(math.atan2(-data[idx, 2], data[idx, 3]), [-math.pi, math.pi], [0, 16]), 0)
            features[1, f] = interp(data[idx, 4], [-math.pi, math.pi], [0, 16], 0)
            # features[2, f] = math.sqrt(data[idx, 2]**2 + data[idx,3]**2)
            # features[2,f] = data[idx,0]
            # features[3,f] = data[idx,1]
            features_x[f, 0] = data[idx, 1]
            features_y[f, 0] = -data[idx, 0]
            idx += step
        STF.append(features)
        if plot_name != []:
            plt.subplot(2, 4, labels[i])
            plt.plot(features_x, features_y, marker='x')
            plt.title(label_names[labels[i] - 1])
    if plot_name != []:
        fig.canvas.set_window_title(plot_name)
    STF = np.asarray(STF)
    a = STF.shape
    b = STF[0, :, 0]
    # STF = STF.reshape(STF.shape[0], STF.shape[2], STF.shape[1])
    return STF