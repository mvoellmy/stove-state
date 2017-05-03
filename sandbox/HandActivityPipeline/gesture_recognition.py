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


# [place pan, place egg, place lid, remove lid, remove egg, remove pan]
# becomes
# [pan, food, lid, lid, food, pan]
begg_labels = [1, 2, 3, 3, 2, 1]

segg_labels = [1, 6, 2, 3, 3, 4, 2, 1]
scegg_labels = [1, 6, 2, 2, 5, 4, 1]
num_labels = 3

train_file_idx = [0, 1, 2, 3, 4, 5]



def extract_features(path_file_names, num_gestures, label_encoder, train_file_idx):
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
        if file_counter in train_file_idx:
            train_files.append(file_name)
        else:
            test_files.append(file_name)
        with open(file_name, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                try:
                    gesture_data.append(row)
                    if file_counter%num_video_files in train_file_idx:
                        data_train.append(row)
                        frame_counter_train += 1
                    else:
                        data_test.append(row)
                        frame_counter_test += 1
                except IndexError:
                    gesture_data.append(-1)
        if file_counter%num_video_files in train_file_idx:
            labels_range_train.append(frame_counter_train)
            labels_train.append( label_encoder[int(file_counter / num_video_files)] )
        else:
            labels_range_test.append(frame_counter_test)
            labels_test.append( label_encoder[int(file_counter / num_video_files)] )
        file_counter += 1
    data_train = np.array(data_train, dtype=np.float)
    data_test = np.array(data_test, dtype=np.float)

    return data_train, labels_train, labels_range_train, data_test, labels_test, labels_range_test


path_labels = join('gesture_features/begg/')
file_names = glob.glob(join(path_labels, '*.csv'))
data_train, labels_train, labels_range_train, data_test, labels_test, labels_range_test = extract_features(file_names, 6, begg_labels, train_file_idx)

# path_labels = join('gesture_features/segg/')
# file_names = glob.glob(join(path_labels, '*.csv'))
# data_train, labels_train, labels_range_train, data_test, labels_test, labels_range_test = extract_features(file_names, 8, segg_labels, [0])

k = 10
kmeans = KMeans(k)
# data_train  = data_train[1:,0:5]
# data_test = data_test[1:,0:5]
kmeans.fit(data_train)

########################################################################################################################
table = np.zeros((num_labels, k))
for i in range(0,len(labels_range_train)-1):
    for j in range(0,k):
        a = labels_range_train[i]
        b = labels_range_train[i+1]
        l = labels_train[i] - 1
        table[l,j] = (np.asarray(kmeans.labels_[a:b]) == j).sum()

np.set_printoptions(precision=3)
row_sums = table.sum(axis=1)
table_norm = (table.transpose()/row_sums).transpose()
table_norm_norm = table_norm / table_norm.sum(axis=0)
print(table)
print("Table norm")
print(table_norm)
print("Table norm norm")
print(table_norm_norm)

g = 5
a = labels_range_test[g]
b = labels_range_test[g+1]
predictions = kmeans.predict(data_test[a:b])
print(predictions)
print(data_train[0])

gesture_distribution = np.zeros((1,num_labels))
for p in predictions:
    gesture_distribution += table_norm_norm[:,p]

gesture_distribution = gesture_distribution / gesture_distribution.sum(axis=1)
print("Gesture_distribution")
print(gesture_distribution)
# gesture_distribution_compact = np.array([gesture_distribution[0,0]+gesture_distribution[0,5],
#                                 gesture_distribution[0,1]+gesture_distribution[0,4],
#                                 gesture_distribution[0,2]+gesture_distribution[0,3]])
# print(gesture_distribution_compact)
########################################################################################################################



def spatio_temporal_features(data, labels_range, num_features):
    N = num_features
    STF = []
    for i in range(0,len(labels_range)-1):
        num_frames = labels_range[i+1] - labels_range[i]
        step = int(num_frames/N)
        idx = labels_range[i]
        features = np.zeros((1,N))
        for f in range(0,N):
            features[0,f] = math.atan2(-data[idx,2], data[idx,3])
            idx += step
        STF.append(features)
    STF = np.asarray(STF)
    STF = STF.reshape(STF.shape[0], STF.shape[2])
    return STF

N = 10
STF_train = spatio_temporal_features(data_train, labels_range_train, N)
STF_test = spatio_temporal_features(data_test, labels_range_test, N)

labels_train = np.asarray(labels_train)
labels_test = np.asarray(labels_test)


fig, ax = plt.subplots()
cmap = plt.cm.get_cmap('jet',N)
color = ['b', 'r', 'y', 'g', 'm', 'c']
for i in range(0,len(labels_range_train)-1):
    plt.subplot(2,3,labels_train[i])
    plt.scatter(range(0,N), STF_train[i,:], c=color[labels_train[i]-1])
plt.show()


parameters = [
        {'kernel': ['rbf'], 'gamma': [1, 0.1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        #{'kernel': ['linear'], 'C': [10, 100, 1000]},
        {'kernel': ['poly'], 'gamma': [0.2, 0.1, 0.05], 'C': [100, 1000]}
    ]
clf = GridSearchCV(SVC(), parameters, cv=2, n_jobs=1)
clf.fit(STF_train, labels_train)

predicted_labels = clf.predict(STF_test)
print("Predicted Labels")
print(predicted_labels)
print("Actual Labels")
print(labels_test)

