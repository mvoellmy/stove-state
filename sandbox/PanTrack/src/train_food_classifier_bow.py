import cv2
import configparser
import ast
import numpy as np
import os.path
import time
import pickle

from random import shuffle
from math import pi, log

# Sk learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# Own Libraries
from panlocator import PanLocator
from helpers import *


# Features Info Parameters
_params = {'stove_type':        'I',
           'plate_of_interest': 4,
           'feature_type':      'SIFT',
           'nr_of_features':    0}

if _params['feature_type'] == 'RGB_HIST':
    # RGB Histogram Params
    _feature_params = {'resolution': 32}

elif _params['feature_type'] == 'SIFT':
    _feature_params = {'k': 50,
                       'tf-idf': True}

_params['feature_params'] = _feature_params

# Paths
img_type = '.jpg'
cfg_path = '../../../cfg/class_cfg.txt'
features_name = '2017-05-19-07_34_40'  # I_2 scegg and segg #
features_name = '2017-05-19-09_12_03'

_train_model = True
_load_features = _train_model
_load_features = False
_max_features = 30
_test_size = 0.3

_use_mse = False
_use_img_shuffle = False

# Output Options
_plot_patches = False
_plot_ellipse = True
_plot_hists = False
_print_update_rate = 50
_plot_fails = True

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
features_path = polybox_path + 'pan_detect/food_features/'
models_path = polybox_path + 'pan_detect/food_models/'
data_path = '/Users/miro/Desktop/' + _params['stove_type'] + '_' + str(_params['plate_of_interest']) + '/'
data_path = '/Volumes/SD_128_DOS/' + _params['stove_type'] + '_' + str(_params['plate_of_interest']) + '/'
data_path = polybox_path + 'pan_detect/data/' + _params['stove_type'] + '_' + str(_params['plate_of_interest']) + '/food/'

print(data_path)

# get classes
label_types = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

# initialize variables
sift_accu = []                  # Container storing all img_accus which store the sift features
labels = []
data_s = []
patches = []
print_update_state = _print_update_rate

# initialize Ellipse fitter
pan_locator = PanLocator(_ellipse_method='CONVEX', _ellipse_smoothing='RAW')


# Build or load Features
if _load_features:
    print('---------------------------')
    print('Features name: {}'.format(features_name))
    data_vw = np.load(features_path + 'F_' + features_name + '.npy')
    labels = np.load(features_path + 'L_' + features_name + '.npy')

    # load info file
    _params = {}

    with open(features_path + 'I_' + features_name + '.txt', 'r') as file:
        _params = eval(file.read())

    print('Parameters: ')
    for key, val in _params.items():
        print('\t{}: {}'.format(key, val))

    kmeans = pickle.load(features_path + 'K_' + features_name + '.sav', 'rb')

else:
    for label_nr, label_name in enumerate(label_types):

        nr_of_label_features = 0

        img_list = [f for f in os.listdir(data_path + label_name) if os.path.isfile(os.path.join(data_path + label_name, f))
                    and img_type in f]
        print('Extracting {2} features from {1} images of label {0}'.format(label_name, len(img_list), int(_max_features/len(label_types))))

        # Randomize img list so
        if _use_img_shuffle:
            shuffle(img_list)

        for img_nr, img_name in enumerate(img_list):
            if nr_of_label_features >= _max_features/len(label_types):
                print("Max number of features for class {} has been reached".format(label_name))
                break

            frame = cv2.imread(data_path + label_name + '/' + img_name)
            patch = frame[corners[plate_of_interest-1, 1]:corners[plate_of_interest-1, 3],
                          corners[plate_of_interest-1, 0]:corners[plate_of_interest-1, 2]]

            # Check the mean squared error between two consecutive frames
            if img_nr == 0 or not _use_mse or mse(patch, old_patch) > threshold:

                center, axes, phi = pan_locator.find_pan(patch, _plot_ellipse=False)

                mask = np.zeros_like(patch)
                ellipse_mask = cv2.ellipse(mask, tuple(map(int, center)), tuple(map(int, axes)),
                                           int(-phi * 180 / pi), 0, 360, (255, 255, 255), thickness=-1)

                patch_normalized = histogram_equalization(patch)

                if _params['feature_type'] == 'RGB_HIST':
                    feature = np.zeros((3, _feature_params['resolution']))

                    for i in range(3):
                        hist = cv2.calcHist([patch], [i], ellipse_mask[:, :, 0], [_feature_params['resolution']], [0, 256])
                        feature[i, :] = np.transpose(hist)
                        if _plot_hists:
                            plot_histogram(hist, i)

                    feature = feature.flatten()

                elif _params['feature_type'] == 'SIFT':

                    sift = cv2.xfeatures2d.SIFT_create()
                    kp, descriptors = sift.detectAndCompute(patch, ellipse_mask[:, :, 0])

                    # Create
                    if descriptors is None:
                        print('feature dead')
                    else:
                        feature = np.reshape(descriptors, (-1, 128))

                if _plot_ellipse:
                    plot_patch = cv2.ellipse(patch, tuple(map(int, center)), tuple(map(int, axes)),
                                             int(-phi * 180 / pi), 0, 360, (0, 0, 255), thickness=2)
                else:
                    plot_patch = patch

                data_s.append(feature)
                labels.append(label_nr)

                nr_of_label_features += 1

                if _plot_fails:
                    patches.append(patch)

            if _plot_patches:
                patch_title = 'Label: ' + label_name
                cv2.imshow(patch_title, plot_patch)
                # cv2.imshow('frame', frame)
                cv2.waitKey(1)

            if img_nr + 1 >= print_update_state:
                print("[{}/{}]\tfeatures extracted".format(len(labels), _max_features))
                print("[{}/{}]\timages processed".format(img_nr + 1, len(img_list)))
                print_update_state = print_update_state + _print_update_rate

            old_patch = patch

        print('{} features extracted'.format(len(data_s)))
        print_update_state = _print_update_rate

    _params['labels'] = label_types
    _params['nr_of_features'] = len(data_s)

    print('Feature extraction finished!')
    print('---------------------------')

    data_vw = np.zeros((len(data_s), _feature_params['k']))  # data_vw: Visual Word matrix [nr_of_imgs, k: nr_of_visual_words]

    # BAG OF WORDS
    if _params['feature_type'] == 'SIFT':
        # data_s: List with the sift features of each image in in a combined cell
        # sift_features: ALL sift features used for clustering
        sift_features = np.vstack(data_s)
        # K Means Clustering
        kmeans = KMeans(n_clusters=_feature_params['k'], random_state=0).fit(sift_features)

        if _feature_params['tf-idf']:
            tf = np.zeros(data_vw.shape)
            idf = np.zeros(_feature_params['k'])

        # Index Visual Words
        for i, img_features in enumerate(data_s):
            for sift_feature in img_features:
                data_vw[i, kmeans.predict(sift_feature)] += 1

            if _feature_params['tf-idf']:
                # Term frequency
                for j, visual_word_count in enumerate(data_vw[i, :]):
                    tf[i, j] = visual_word_count/sum(data_vw[i, :])

        if _feature_params['tf-idf']:
            # Inverse Document Frequency
            for i, visual_word_occurances_in_imgs in enumerate(data_vw.T):
                idf[i] = (log(len(visual_word_occurances_in_imgs)/np.count_nonzero(visual_word_occurances_in_imgs)))
                print(np.count_nonzero(visual_word_occurances_in_imgs))
                print(len(visual_word_occurances_in_imgs))

            input(tf)
            input(idf)

            data_vw = data_vw * tf * idf

            _params['visual_word_idf'] = idf.tolist()

    # Save features
    if input('Save features? [y/n]').lower() == 'y':
        f_time_name = time.strftime("%Y-%m-%d-%H_%M_%S")
        features_name = features_path + 'F_' + f_time_name + '.npy'
        labels_name = features_path + 'L_' + f_time_name + '.npy'
        info_name = features_path + 'I_' + f_time_name + '.txt'
        kmeans_name = features_path + 'K_' + f_time_name + '.sav'

        np.save(features_name, data_vw)
        np.save(labels_name, labels)
        # save I=Info in txt file
        with open(info_name, 'w') as file:
            file.write(repr(_params))

        pickle.dump(kmeans, open(kmeans_name, 'wb'))

        print("Features have been saved with name:\n{}".format(f_time_name))


print('---------------------------')
if _train_model:
    train_data, test_data, train_labels, test_labels = train_test_split(data_vw, labels, test_size=_test_size, random_state=2)
    # Optimize the parameters by cross-validation
    parameters = [
        {'kernel': ['rbf'], 'gamma': [0.1, 1], 'C': [1, 100]},
        {'kernel': ['linear'], 'C': [ 50, 100]},
        {'kernel': ['linear'], 'C': [1]},
        {'kernel': ['poly'], 'degree': [2]}
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
        kmeans_name = models_path + 'K_' + m_time_name + '.sav'

        pickle.dump(clf, open(model_name, 'wb'))
        pickle.dump(kmeans, open(kmeans_name, 'wb'))

        with open(info_name, 'w') as file:
            file.write(repr(_params))

        print("Model has been saved with name: \n{}".format(m_time_name))

    if _plot_fails:
        for i, image in enumerate(test_data):
            if labels_predicted[i] != test_labels[i]:

                cv2.imshow('{} Predicted: {} Truth: {}'.format(i, label_types[labels_predicted[i]], label_types[test_labels[i]]), patches[i])
                cv2.waitKey(0)
