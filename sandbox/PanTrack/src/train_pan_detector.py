import cv2
import configparser
import ast
import numpy as np
import csv
import os.path
import time
import pickle

from random import shuffle

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
from helpers import mse, get_HOG, histogram_equalization

# Hog Params
_params = {'orientations': 4,
           'pixels_per_cell': (16, 16),
           'cells_per_block': (4, 4),
           'widthPadding': 10}

# Options
_params['stove_type'] = 'I'
img_type = '.jpg'
cfg_path = '../../../cfg/class_cfg.txt'
features_path = '../features/'
models_path = '../models/'
features_name = '2017-04-27-15_17_27'

_load_features = True
_train_model = True
_perc_jump = 10
_max_features = 5000

_use_rgb = False
_locate_pan = False
_plot_fails = True
_plot_patches = False

# Read Config data
config = configparser.ConfigParser()
config.read(cfg_path)

# Read corners and reshape them into 2d-Array
corners = np.reshape(ast.literal_eval(config.get(_params['stove_type'], "corners")), (-1, 4))

path_data = config.get('paths', 'data') + _params['stove_type'] + '/'
threshold = float(config.get(_params['stove_type'], "threshold"))
plate_of_interest = int(config.get(_params['stove_type'], "plate_of_interest"))

# get classes
class_list = [f for f in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, f))]

# initialize variables
labels = []
data = []
patches = []
next_perc = 0


# Build or load Features
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

        nr_of_label_features = 0

        img_list = [f for f in os.listdir(path_data+label_name) if os.path.isfile(os.path.join(path_data+label_name, f))
                                                                   and img_type in f]
        print('{}:\nExtracting features from {} images...'.format(label_name, len(img_list)))

        shuffle(img_list)
        for it, img in enumerate(img_list):
            if nr_of_label_features >= _max_features/len(class_list):
                print("Max number of features for class {} has been reached".format(label_name))
                break

            frame = cv2.imread(path_data+label_name+'/'+img, 0)
            patch = frame[corners[plate_of_interest-1, 1]:corners[plate_of_interest-1, 3],
                          corners[plate_of_interest-1, 0]:corners[plate_of_interest-1, 2]]

            # Check the mean squared error between two consecutive frames
            if it == 0 or mse(patch, old_patch) > threshold:

                patch_normalized = histogram_equalization(patch)
                hog = get_HOG(patch_normalized,
                              orientations=_params['orientations'],
                              pixels_per_cell=_params['pixels_per_cell'],
                              cells_per_block=_params['cells_per_block'],
                              widthPadding=_params['widthPadding'])

                data.append(hog)
                labels.append(label_nr)

                nr_of_label_features += 1

                if _locate_pan:
                    locate_pan(patch, _plot_ellipse=True)

                if _plot_fails:
                    patches.append(patch)

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

    _params['labels'] = class_list
    _params['nr_features'] = len(data)

    # Save features
    f_time_name = time.strftime("%Y-%m-%d-%H_%M_%S")
    features_name = features_path + 'F_' + f_time_name + '.npy'
    labels_name = features_path + 'L_' + f_time_name + '.npy'
    info_name = features_path + 'I_' + f_time_name + '.csv'

    np.save(features_name, data)
    np.save(labels_name, labels)
    # save I in csv file
    with open(info_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, val in _params.items():
            writer.writerow([key, val])

    print('Feature extraction finished!')

print('---------------------------')

if _train_model:
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=2)
    # Optimize the parameters by cross-validation
    parameters = [
        # {'kernel': ['rbf'], 'gamma': [0.1, 1], 'C': [1, 100]},
        {'kernel': ['linear'], 'C': [1000]},
        # {'kernel': ['poly'], 'degree': [2]}
    ]

    # Grid search object with SVM classifier.
    clf = GridSearchCV(SVC(), parameters, cv=3, n_jobs=-1, verbose=1)
    print("GridSearch Object created")
    print("Starting training")
    clf.fit(train_data, train_labels)

    print("Best parameters set found on training set:")
    print(clf.best_params_)

    print("Starting test dataset...")
    labels_predicted = clf.predict(test_data)
    _params['model_accuracy'] = (labels_predicted == test_labels).mean()
    print("Test Accuracy [%0.3f]" % (_params['model_accuracy']))

    # save the model to disk
    m_time_name = time.strftime("%Y-%m-%d-%H_%M_%S")
    model_name = models_path + 'M_' + m_time_name + '.sav'
    info_name = models_path + 'I_' + m_time_name + '.csv'

    pickle.dump(clf, open(model_name, 'wb'))
    with open(info_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, val in _params.items():
            writer.writerow([key, val])


    print("Model has been saved.")


    if _plot_fails:
        for i, image in enumerate(test_data):
           if labels_predicted[i] != test_labels[i]:
               cv2.imshow('{} Predicted: {} Truth: {}'.format(i, labels_predicted[i], test_labels[i]), patches[i])
               cv2.waitKey(0)