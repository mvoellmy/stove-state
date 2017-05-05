import numpy as np
import cv2
import configparser
from os.path import join
import glob
import csv
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from numpy import interp
from sklearn import svm


def extract_features(path_file_names, num_gestures, label_encoder, test_file_idx):
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

    for file_name in path_file_names:
        gesture_data = []
        if not "false" in file_name:
            # print(file_name)
            label_idx = int(file_name.rsplit("\\", 1)[-1].split(".")[0][0]) - 1

            if file_counter not in test_file_idx:
                train_files.append(file_name)
            else:
                test_files.append(file_name)
            with open(file_name, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                for row in csv_reader:
                    try:
                        gesture_data.append(row)
                        if file_counter%num_video_files not in test_file_idx:
                            data_train.append(row)
                            frame_counter_train += 1
                        else:
                            data_test.append(row)
                            frame_counter_test += 1
                    except IndexError:
                        gesture_data.append(-1)
            if file_counter%num_video_files not in test_file_idx:
                labels_range_train.append(frame_counter_train)
                labels_train.append( label_encoder[label_idx] )
            else:
                labels_range_test.append(frame_counter_test)
                labels_test.append( label_encoder[label_idx] )
        file_counter += 1
    data_train = np.array(data_train, dtype=np.float)
    data_test = np.array(data_test, dtype=np.float)

    return data_train, labels_train, labels_range_train, data_test, labels_test, labels_range_test


# [place pan, pour water, place egg, place lid, remove lid, remove egg, remove pan]
# becomes
# [pan, nothing, food, lid, lid, food, pan]
label_names = ['place pan', 'pour water', 'place egg', 'place lid', 'remove lid', 'remove egg', 'remove pan']
begg_labels = [1, -1, 2, 3, 4, 5, 6]
label_names = ['pan-gesture', 'food-gesture', 'lid-gesture']
begg_labels = [1, -1, 2, 3, 3, 2, 1]

segg_labels = [1, 6, 2, 3, 3, 4, 2, 1]
scegg_labels = [1, 6, 2, 2, 5, 4, 1]

path_labels = join('gesture_features/begg/')
file_names = glob.glob(join(path_labels, '*.csv'))
out = extract_features(file_names, num_gestures=6, label_encoder=begg_labels, test_file_idx=[8])
data_train, labels_train, labels_range_train, data_test, labels_test, labels_range_test = out

label_names = ['pan-gesture', 'food-gesture', 'lid-gesture', 'season', 'stirr', 'other']
path_labels = join('gesture_features/segg/')
file_names = glob.glob(join(path_labels, '*.csv'))
data_train, labels_train, labels_range_train, data_test, labels_test, labels_range_test = extract_features(file_names, 8, segg_labels, [3])

fig = plt.figure()
for i in range(0,len(labels_range_train)-1):
    plt.subplot(2,3,labels_train[i])
    a = labels_range_train[i]
    b = labels_range_train[i+1]
    plt.plot(data_train[a:b,1], -data_train[a:b,0])
    plt.title(label_names[labels_train[i]-1])
fig.canvas.set_window_title('Trajectories of Training Data')

def spatio_temporal_features(data, labels_range, labels, num_features, plot_name=[]):
    N = num_features
    STF = []
    if plot_name != []:
        fig = plt.figure()
    for i in range(0,len(labels_range)-1):
        num_frames = labels_range[i+1] - labels_range[i]
        step = int(num_frames/N)
        idx = labels_range[i]
        features = np.zeros((1,N))
        features_x = np.zeros((N,1))
        features_y = np.zeros((N,1))
        for f in range(0,N):
            features[0,f] = round(interp(math.atan2(-data[idx,2], data[idx,3]), [-math.pi, math.pi], [0, 16]),0)
            features_x[f,0] = data[idx,1]
            features_y[f,0] = -data[idx,0]
            idx += step
        STF.append(features)
        if plot_name != []:
            plt.subplot(2, 3, labels[i])
            plt.plot(features_x, features_y, marker='x')
            plt.title(label_names[labels[i]-1])
    if plot_name != []:
        fig.canvas.set_window_title(plot_name)
    STF = np.asarray(STF)
    STF = STF.reshape(STF.shape[0], STF.shape[2])
    return STF

N = 10
STF_train = spatio_temporal_features(data_train, labels_range_train, labels_train, N, 'Keyframes of Training Data')
STF_test = spatio_temporal_features(data_test, labels_range_test, labels_test, N)

labels_train = np.asarray(labels_train)
labels_test = np.asarray(labels_test)
a = STF_train[:,0:2]

fig, ax = plt.subplots()
cmap = plt.cm.get_cmap('jet',N)
color = ['b', 'r', 'y', 'g', 'm', 'c']
for i in range(0,len(labels_range_train)-1):
    plt.subplot(2,3,labels_train[i])
    plt.scatter(range(0,N), STF_train[i,:], c=color[labels_train[i]-1])
    plt.xlabel('Keyframes', fontsize=10)
    plt.ylabel('Direction', fontsize=10)
    plt.title(label_names[labels_train[i]-1])
    plt.tight_layout()
fig.canvas.set_window_title('Spatio Temporal Features (STF)')

parameters = [
        # {'kernel': ['rbf'], 'gamma': [1, 0.1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        #{'kernel': ['linear'], 'C': [10, 100, 1000]},
        {'kernel': ['poly'], 'gamma': [0.2], 'C': [100]}
    ]
clf = GridSearchCV(SVC(), parameters, cv=2, n_jobs=1)
clf.fit(STF_train, labels_train)

predicted_labels = clf.predict(STF_test)
print("Actual Labels")
print(labels_test)
print("Correct predictions \t Predicted Labels")
print("{} \t\t\t {}".format((labels_test==predicted_labels)*1, predicted_labels))
# print("Predicted Labels")
# print(predicted_labels)

predicted = []
counter = labels_test*0
print("Predict each Keyframe")
print("Correct predictions \t Predicted Labels")
for i in range(0,N):
    # clf = GridSearchCV(SVC(), parameters, cv=2, n_jobs=1)
    clf = svm.SVC()
    clf.fit(STF_train[:,i:i+2], labels_train)
    predicted_labels = clf.predict(STF_test[:,i:i+2])
    counter += (labels_test == predicted_labels) * 1
    print("{} \t\t\t {}".format((labels_test == predicted_labels) * 1, predicted_labels))

print("Percentage of correct guesses")
print(counter/N)

plt.show()
