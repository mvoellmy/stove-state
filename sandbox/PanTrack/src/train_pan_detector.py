import cv2
import configparser
import ast
import numpy as np
import os.path
import time
import pickle

from random import shuffle
# Sk learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Own Libraries
from panlocator import PanLocator
from helpers import mse, get_HOG, histogram_equalization

# Hog Params
_feature_params = {'orientations':      4,
                   'pixels_per_cell':   (12, 12),
                   'cells_per_block':   (4, 4),
                   'widthPadding':      10}

# Features Info Parameters
_params = {'stove_type':        'I',
           'plate_of_interest': 4,
           'feature_type':      'HOG',
           'nr_of_features':    0,
           'feature_params':    _feature_params}


# Paths
img_type = '.jpg'
cfg_path = '../../../cfg/class_cfg.txt'
features_name = '2017-05-11-15_52_55'  # I_4 begg
features_name = '2017-05-18-16_59_00'  # I_2 scegg and segg
features_name = '2017-05-18-18_23_14'  # I_4 begg

# Options
_train_model = True
_load_features = _train_model
_max_features = 3000
_test_size = 0.3

_use_mse = True
_use_rgb = False

# Output Options
_print_update_rate = 500
_plot_fails = False
_plot_patches = False
_locate_pan = False

# Read Config data
config = configparser.ConfigParser()
config.read(cfg_path)

# Read corners and reshape them into 2d-Array
_params['corners'] = ast.literal_eval(config.get(_params['stove_type'], "corners"))
corners = np.reshape(_params['corners'], (-1, 4))
polybox_path = config.get('paths', 'polybox')
threshold = float(config.get(_params['stove_type'], "threshold"))
plate_of_interest = _params['plate_of_interest']

# More Paths
features_path = polybox_path + 'pan_detect/pan_features/'
models_path = polybox_path + 'pan_detect/pan_models/'
data_path = '/Users/miro/Desktop/' + _params['stove_type'] + '_' + str(_params['plate_of_interest']) + '/'
data_path = polybox_path + 'pan_detect/data/' + _params['stove_type'] + '_' + str(_params['plate_of_interest']) + '/pan/'

# get classes
label_types = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

# initialize variables
labels = []
data = []
patches = []
print_update_state = _print_update_rate

# initialize Ellipse fitter
pan_locator = PanLocator(_ellipse_method='CONVEX', _ellipse_smoothing='RAW')

# Build or load Features
if _load_features:
    print('---------------------------')
    print('Features name: {}'.format(features_name))
    data = np.load(features_path + 'F_' + features_name + '.npy')
    labels = np.load(features_path + 'L_' + features_name + '.npy')

    # load info file
    _params = {}

    with open(features_path + 'I_' + features_name + '.txt', 'r') as file:
        _params = eval(file.read())

    print('Parameters: ')
    for key, val in _params.items():
        print('\t{}: {}'.format(key, val))

else:
    for label_nr, label_name in enumerate(label_types):

        nr_of_label_features = 0

        img_list = [f for f in os.listdir(data_path + label_name) if os.path.isfile(os.path.join(data_path + label_name, f))
                    and img_type in f]
        print('Extracting {2} features from {1} images of label {0}'.format(label_name, len(img_list), int(_max_features/len(label_types))))

        # Randomize img list so
        # shuffle(img_list)
        for img_nr, img_name in enumerate(img_list):
            if nr_of_label_features >= _max_features/len(label_types):
                print("Max number of features for class {} has been reached".format(label_name))
                break

            frame = cv2.imread(data_path + label_name + '/' + img_name, 0)
            patch = frame[corners[plate_of_interest-1, 1]:corners[plate_of_interest-1, 3],
                          corners[plate_of_interest-1, 0]:corners[plate_of_interest-1, 2]]

            # Check the mean squared error between two consecutive frames
            if img_nr == 0 or not _use_mse or mse(patch, old_patch)\
                    > threshold:

                patch_normalized = histogram_equalization(patch)
                hog = get_HOG(patch_normalized,
                              orientations=_feature_params['orientations'],
                              pixels_per_cell=_feature_params['pixels_per_cell'],
                              cells_per_block=_feature_params['cells_per_block'],
                              widthPadding=_feature_params['widthPadding'])

                data.append(hog)
                labels.append(label_nr)

                nr_of_label_features += 1

                if _locate_pan:
                    pan_locator.find_pan(patch, _plot_ellipse=True)

                if _plot_fails:
                    patches.append(patch)

            if _plot_patches:
                patch_title = 'Label: ' + label_name
                cv2.imshow(patch_title, patch)
                # cv2.imshow('frame', frame)
                cv2.waitKey(0)

            if img_nr >= print_update_state:
                print("{}/{} features extracted".format(len(labels), _max_features))
                print("{}/{} images processed".format(img_nr, len(img_list)))
                print_update_state = print_update_state + _print_update_rate

            old_patch = patch

        print('{} features extracted'.format(len(data)))
        print_update_state = _print_update_rate

    _params['labels'] = label_types
    _params['nr_of_features'] = len(data)

    print('Feature extraction finished!')
    print('---------------------------')

    # Save features
    if input('Save features? [y/n]').lower() == 'y':
        f_time_name = time.strftime("%Y-%m-%d-%H_%M_%S")
        features_name = features_path + 'F_' + f_time_name + '.npy'
        labels_name = features_path + 'L_' + f_time_name + '.npy'
        info_name = features_path + 'I_' + f_time_name + '.txt'

        np.save(features_name, data)
        np.save(labels_name, labels)
        # save I=Info in txt file
        with open(info_name, 'w') as file:
            file.write(repr(_params))

        print("Features have been saved with name:\n{}".format(f_time_name))


print('---------------------------')
if _train_model:
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=_test_size, random_state=2)
    # Optimize the parameters by cross-validation
    parameters = [
        # {'kernel': ['rbf'], 'gamma': [0.1, 1], 'C': [1, 100]},
        # {'kernel': ['linear'], 'C': [1, 50, 100]},
        {'kernel': ['linear'], 'C': [1]},
        # {'kernel': ['poly'], 'degree': [2]}
    ]

    # Grid search object with SVM classifier.
    clf = GridSearchCV(SVC(), parameters, cv=3, n_jobs=-1, verbose=10)
    print("GridSearch Object created")
    print("Starting training")
    clf.fit(train_data, train_labels)

    print("Best parameters set found on training set:")
    print(clf.best_params_)

    print("Starting test dataset...")
    labels_predicted = clf.predict(test_data)
    _model_info = {'accuracy': (labels_predicted == test_labels).mean(),
                   'best_params': clf.best_params_,
                   'test_size': _test_size}

    _params['model_params'] = _model_info

    print("Test Accuracy {}".format(_model_info['accuracy']))

    if input('Save model? [y/n]\n').lower() == 'y':
        _params['model_params']['notes'] = str(input('Notes about model:\n'))
        # save the model to disk
        m_time_name = time.strftime("%Y-%m-%d-%H_%M_%S")
        model_name = models_path + 'M_' + m_time_name + '.sav'
        info_name = models_path + 'I_' + m_time_name + '.txt'

        pickle.dump(clf, open(model_name, 'wb'))

        with open(info_name, 'w') as file:
            file.write(repr(_params))

        print("Model has been saved with name: \n{}".format(m_time_name))

    if _plot_fails:
        for i, image in enumerate(test_data):
            if labels_predicted[i] != test_labels[i]:
                cv2.imshow('{} Predicted: {} Truth: {}'.format(i, labels_predicted[i], test_labels[i]), patches[i])
                cv2.waitKey(0)
