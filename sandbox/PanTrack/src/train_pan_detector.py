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

# Own Libraries
from locate_pan import locate_pan

# Options
img_type = '.jpg'
stove_type = 'I_test'
cfg_path = '../../../cfg/class_cfg.txt'
features_path = '../features/'
features_name = 'test'

_plot_patches = False
_use_rgb = False
_perc_jump = 0.01
_plot_fails = True
_fit_ellipse = False
_load_features = True
_hog_params = {'orientations': 4,
               'pixels_per_cell': (16, 16),
               'cells_per_block': (4, 4),
               'widthPadding': 10}


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def get_HOG(img, orientations=4, pixels_per_cell=(16, 16), cells_per_block=(4, 4), widthPadding=10):
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


# Read Config data
config = configparser.ConfigParser()
config.read(cfg_path)

# Read corners and reshape them into 2d-Array
corners = np.reshape(ast.literal_eval(config.get(stove_type, "corners")), (-1, 4))

# Read dataset
path_data = config.get('paths', 'data')
path_data = path_data + stove_type + '/'
has_pan = np.zeros((1, 4))
threshold = float(config.get(stove_type, "threshold"))
plate_of_interest = int(config.get(stove_type, "plate_of_interest"))
class_list = [f for f in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, f))]
labels = []
data = []
patches = []
next_perc = 0


# Build dataset
if _load_features:
    print('---------------------------')
    print('Features name: {}'.format(features_name))
    data = np.load(features_path + 'F_' + features_name + '.npy')
    labels = np.load(features_path + 'L_' + features_name + '.npy')

    # load info file
    _params = {}
    print('Parameters: ')
    for key, val in csv.reader(open(features_path + 'I_' + features_name + '.csv')):
        _params[key] = val
        print('     {}: {}'.format(key, val))
else:
    for label_nr, label_name in enumerate(class_list):
        img_list = [f for f in os.listdir(path_data+label_name) if os.path.isfile(os.path.join(path_data+label_name, f)) and img_type in f]
        print('{}:\nExtracting features from {} images...'.format(label_name, len(img_list)))
        for it, img in enumerate(img_list):
            frame = cv2.imread(path_data+label_name+'/'+img, 0)
            patch = frame[corners[plate_of_interest-1, 1]:corners[plate_of_interest-1, 3],
                          corners[plate_of_interest-1, 0]:corners[plate_of_interest-1, 2]]

            # Check the mean squared error between two consecutive frames
            if it == 0 or mse(patch, old_patch) > threshold:
                hog = get_HOG(patch)
                hog = get_HOG(patch, orientations=_hog_params['orientations'],
                                     pixels_per_cell=_hog_params['pixels_per_cell'],
                                     cells_per_block=_hog_params['cells_per_block'],
                                     widthPadding=_hog_params['widthPadding'])
                if _plot_fails:
                    patches.append(patch)
                data.append(hog)
                labels.append(label_nr)
                if _fit_ellipse:
                    locate_pan(patch, _plot_ellipse=True)

            if _plot_patches:
                patch_title = 'Label: ' + label_name
                cv2.imshow(patch_title, patch)
                # cv2.imshow('frame', frame)
                cv2.waitKey(1)

            if it/len(img_list)*100 >= next_perc:
                print("{:.2f} %".format(it/len(img_list)*100))
                next_perc = next_perc + _perc_jump

            old_patch = patch

        print('{} features extracted'.format(len(data)))
        next_perc = 0

    # Save features
    time_name = time.strftime("%Y-%m-%d-%H_%M_%S")
    features_name = features_path + 'F_' + time_name + '.npy'
    labels_name = features_path + 'L_' + time_name + '.npy'
    info_name = features_path + 'I_' + time_name + '.csv'

    np.save(features_name, data)
    np.save(labels_name, labels)
    # save I in csv file
    with open(info_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, val in _hog_params.items():
            writer.writerow([key, val])

        writer.writerow(['nr_features', len(data)])
        writer.writerow(['stove_type', stove_type])
        writer.writerow(['labels', class_list])

    print(labels)
    print('Feature extraction finished!')

print('---------------------------')

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=2)
# Optimize the parameters by cross-validation
parameters = [
    # {'kernel': ['rbf'], 'gamma': [0.1, 1], 'C': [1, 100]},
    {'kernel': ['linear'], 'C': [1000]},
    # {'kernel': ['poly'], 'degree': [2]}
]


# Grid search object with SVM classifier.
clf = GridSearchCV(SVC(), parameters, cv=3, n_jobs=-1, verbose=0)
print("GridSearch Object created")
print("Starting training")
clf.fit(train_data, train_labels)

print("Best parameters set found on training set:")
print(clf.best_params_)
#
# # save the model to disk
# filename_sav = time.strftime("%Y-%m-%d-%H_%M_%S.sav")
# pickle.dump(clf, open(filename_sav, 'wb'))
# print("Model has been saved.")

print("Starting test dataset...")
labels_predicted = clf.predict(test_data)
print("Test Accuracy [%0.3f]" % ((labels_predicted == test_labels).mean()))

if _plot_fails:
    for i, image in enumerate(test_data):
       if labels_predicted[i] != test_labels[i]:
           cv2.imshow('{} Predicted: {} Truth: {}'.format(i, labels_predicted[i], test_labels[i]), patches[i])
           cv2.waitKey(0)