import cv2
import configparser
import ast
import numpy as np
import csv
import os.path
import time
import pickle

from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint

import matplotlib.pyplot as plt

plt.ion()

# Options
video_type = '.mp4'
stove_type = 'M'
plate_of_interest = 2
_single_video = True
_plot_patches = True
_use_rgb = False
_perc_jump = 5
# Parameters
threshold = 5

# Read Config data
config = configparser.ConfigParser()
config.read('../cfg/cfg.txt')

# Read corners and reshape them into 2d-Array
corners = np.reshape(ast.literal_eval(config.get("corners", "coordinates")), (-1, 4))

# Read dataset
path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')
has_pan = np.zeros((1, 4))

# Import & parse dataset into labels and frames
list_videos = []
if _single_video:
    # list_videos.append('M_2017-04-06-06_25_00_pan_labeling.mp4')
    # list_videos.append('M_2017-04-07-14_06_53_short_test.mp4')
    # list_videos.append('M_2017-04-07-14_06_53_short_test.mp4')
    list_videos.append('M_2017-04-06-07_06_40_begg.mp4')
else:
    list_videos = [f for f in os.listdir(path_videos) if os.path.isfile(os.path.join(path_videos, f))
                                                         and video_type in f
                                                         and f.startswith(stove_type)]

patch_width = int(np.floor((corners[plate_of_interest-1, 2] - corners[plate_of_interest-1, 0])))
patch_height = int(np.floor((corners[plate_of_interest-1, 3] - corners[plate_of_interest-1, 1])))

print(len(list_videos))
print(list_videos)


def get_HOG(img, orientations=4, pixels_per_cell=(12, 12), cells_per_block=(4, 4), widthPadding=10):
    """
    Calculates HOG feature vector for the given image.

    img is a numpy array of 2- or 3-dimensional image (i.e., grayscale or rgb).
    Color-images are first transformed to grayscale since HOG requires grayscale
    images.

    Reference: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Crop the image from left and right.
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]

    # Note that we are using skimage.feature.
    hog_features = feature.hog(img, orientations, pixels_per_cell, cells_per_block, visualise=True)

    return hog_features


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

features = []
labels = np.array('empty')

for video_count, video in enumerate(list_videos):
    # find corresponding labeling file
    label_file = video.replace(video_type, ".csv")
    path_label = path_labels + label_file

    # Read file
    path_video = path_videos + video
    cap = cv2.VideoCapture(path_video)
    # frame_rate = cap.get(5)
    frame_rate = 30
    nr_of_frames = int(cap.get(7))

    patch_count = 1
    frame_id = 1
    frame_time = frame_id*frame_rate
    frame_labels = np.zeros(nr_of_frames)  # 0: no pan, 1: pan, 2: cover
    next_perc = 0

    # Creat container for plate patches
    if _use_rgb:
        patches = np.zeros((nr_of_frames, patch_height, patch_width, 3))
    else:
        patches = np.zeros((nr_of_frames, patch_height, patch_width))

    patch_labels = []

    # Pan Labeling
    with open(path_label, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ', )
        for row in csv_reader:
            if str(plate_of_interest) == row[1]:
                if 'pan' in row[2] and int(row[3]) == 1:
                    frame_labels[int(frame_time * float(row[0])):] = int(row[2][-1:])

                elif 'pan' in row[2] and int(row[3]) == 0:
                    frame_labels[int(frame_time * float(row[0])):] = 0

                elif 'lid' in row[2] and int(row[3]) == 1:
                    frame_labels[int(frame_time * float(row[0])):] = -int(row[2][-1:])

                elif 'lid' in row[2] and int(row[3]) == 0:
                    frame_labels[int(frame_time * float(row[0])):] = int(row[2][-1:])

        print("Labeling done")

    # for frames in video
    while frame_id < nr_of_frames:
        ret, frame = cap.read()
        if not _use_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # split plates
        patch = frame[corners[plate_of_interest-1, 1]:corners[plate_of_interest-1, 3], corners[plate_of_interest-1, 0]:corners[plate_of_interest-1, 2]]

        if _plot_patches:
            patch_title = 'Label: ' + str(int(frame_labels[frame_id]))
            cv2.imshow(patch_title, patch)
            # cv2.imshow('frame', frame)
            cv2.waitKey(1)

        # Check the mean squared error between two consecutive frames
        if frame_id == 1 or mse(patch, old_patch) > threshold:
            # extract features and save labels
            # print(mse(patch, old_patch))
            hog = get_HOG(patch)
            features.append(hog)
            patch_labels.append(frame_labels[frame_id])
            # cv2.imshow('Hog', hog)
            patch_count = patch_count + 1

        frame_id = frame_id + 1
        if frame_id/nr_of_frames*100 >= next_perc:
            print("{:.2f} %".format(frame_id/nr_of_frames*100))
            next_perc = next_perc + _perc_jump

        old_patch = patch

    if video_count == 0:
        labels = np.asarray(patch_labels)
    else:
        labels = np.concatenate((labels, np.asarray(patch_labels)))
        print(type(labels))

    print("count: {} frame_id: {}".format(patch_count, frame_id))
    print("{}/{} {:.2f}% features extracted...".format(len(features), nr_of_frames, 100 * patch_count / nr_of_frames))


train_data, test_data, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.3, random_state=2)

# Optimize the parameters by cross-validation
parameters = [
    # {'kernel': ['rbf'], 'gamma': [0.1, 1], 'C': [1, 100]},
    {'kernel': ['linear'], 'C': [1000]},
    # {'kernel': ['poly'], 'degree': [2]}
]

# Grid search object with SVM classifier.
clf = GridSearchCV(SVC(), parameters, cv=3, n_jobs=-1, verbose=30)
print("GridSearch Object created")
clf.fit(train_data, train_labels)

print("Best parameters set found on training set:")
print(clf.best_params_)
print()

# save the model to disk
filename_sav = time.strftime("%Y-%m-%d-%H_%M_%S") + ".sav"
pickle.dump(clf, open(filename_sav, 'wb'))
print("Model has been saved.")

print("Starting test dataset...")
labels_predicted = clf.predict(test_data)
for i, image in enumerate(test_data):
    if labels_predicted[i] != test_labels[i]:
        cv2.imshow('{} Predicted: {} Truth: {}'.format(i, labels_predicted[i], test_labels[i]), image)
        cv2.waitKey(1)
        print(image)
        input("hello")

input()
print("Test Accuracy [%0.3f]" % ((labels_predicted == test_labels).mean()))

# Extract Features from images
print("program has terminated.")