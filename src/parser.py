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

# Options
video_type = '.mp4'
stove_type = 'M'
# plates_of_interest = np.array([0, 1, 0, 0])  # Ians kitchen
plate_of_interest = 2

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
list_videos = [f for f in os.listdir(path_videos) if os.path.isfile(os.path.join(path_videos, f))
                                                     and video_type in f
                                                     and f.startswith(stove_type)]

patch_width = int(np.floor((corners[plate_of_interest-1, 2] - corners[plate_of_interest-1, 0])))
patch_height = int(np.floor((corners[plate_of_interest-1, 3] - corners[plate_of_interest-1, 1])))
# frames = np.zeros((np.count_nonzero(plates_of_interest), patch_height, patch_width))

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
    hog_features = feature.hog(img, orientations, pixels_per_cell, cells_per_block)

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
labels = []
video_count = 0

for video in list_videos:
    # Single Video:
    # path_video = path_videos + '2017-03-31-05_01_45_egg_boiled.mp4'

    # find corresponding labeling file
    label_file = video.replace(video_type, ".csv")
    path_label = path_labels + label_file

    # Read file
    path_video = path_videos + video
    cap = cv2.VideoCapture(path_video)
    # frame_rate = cap.get(5)
    frame_rate = 30
    nr_of_frames = int(cap.get(7))

    count = 1
    frame_id = 1
    frame_time = frame_id*frame_rate

    frame_labels = np.zeros(nr_of_frames)  # 0: no pan, 1: pan, 2: cover

    # Creat container for video
    # RGB:
    # train_patches = np.zeros((nr_of_frames, patch_height, patch_width, 3))
    # gray:
    patches = np.zeros((nr_of_frames, patch_height, patch_width))
    patch_labels = []

    # Labeling
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TODO for multiple plates
        # for i, k in enumerate(plates_of_interest):
        #     if k > 0:
        #         cv2.rectangle(frame, tuple(corners[i, 0:2]), tuple(corners[i, 2:4]), 255)
        # cv2.rectangle(frame, tuple(corners[plate_of_interest-1, 0:2]), tuple(corners[plate_of_interest-1, 2:4]), 255)

        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)

        # split plates
        patch = frame[corners[plate_of_interest-1, 1]:corners[plate_of_interest-1, 3], corners[plate_of_interest-1, 0]:corners[plate_of_interest-1, 2]]
        patch_title = 'Label: ' + str(int(frame_labels[frame_id]))
        # cv2.imshow(patch_title, patch)
        # print(int(pan_labels[frame_id]))

        # train_patches[frame_id-1, :, :, :] = patch

        # Check if it is worth extracting features
        if frame_id > 1:
            # print(mse(patch, old_patch))
            if mse(patch, old_patch) > threshold:
                features.append(get_HOG(patch))
                patch_labels.append(frame_labels[frame_id])
                count = count + 1
        else:
            features.append(get_HOG(patch))
            patch_labels.append(frame_labels[frame_id])

        # extract features and save labels
        frame_id = frame_id + 1
        print("{:.2f} %".format(frame_id/nr_of_frames*100))

        old_patch = patch

    labels.append(patch_labels)

    print("count: {} frame_id: {}".format(count, frame_id))
    print("Features extracted...")

    video_count = video_count+1
    if video_count >= 1:
        break

print(len(features))

train_data, test_data, train_labels, test_labels = train_test_split(
        features, labels[0], test_size=0.2, random_state=1)

# Optimize the parameters by cross-validation
parameters = [
    {'kernel': ['rbf'], 'gamma': [2], 'C': [100]},
    # {'kernel': ['linear'], 'C': [1000, 500]},
    # {'kernel': ['poly'], 'degree': [2, 3]}
]

# Grid search object with SVM classifier.
clf = GridSearchCV(SVC(), parameters, cv=3, n_jobs=-1, verbose=20)
print("GridSearch Object created")
clf.fit(train_data, train_labels)

print("Best parameters set found on training set:")
print(clf.best_params_)
print()

# save the model to disk
filename_sav = time.strftime("%Y-%m-%d-%H_%M_%S") + ".sav"
pickle.dump(clf, open('../models/' + filename_sav, 'wb'))
print("Model has been saved.")

print("Starting test dataset...")
labels_predicted = clf.predict(test_data)
print("Test Accuracy [%0.3f]" % ((labels_predicted == test_labels).mean()))

# Extract Features from images
print("program has terminated.")