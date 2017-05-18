import cv2
import configparser
import numpy as np
import os
import pickle

from math import pi

# Own Libraries
from panlocator import PanLocator
from helpers import *

# Params
_ellipse_smoothing = 'AVERAGE'
_ellipse_smoothing = 'RAW'
_ellipse_smoothing = 'VOTE'

_ellipse_method = 'RANSAC'
_ellipse_method = 'MAX_ARC'
_ellipse_method = 'CONVEX'

_segment = False

_start_frame = 900
_frame_rate = 1

# Read config
cfg_path = '../../../cfg/class_cfg.txt'
config = configparser.ConfigParser()
config.read(cfg_path)

polybox_path = config.get('paths', 'polybox')

features_path = polybox_path + 'pan_detect/pan_features/'
pan_models_path = polybox_path + 'pan_detect/pan_models/'
food_models_path = polybox_path + 'pan_detect/food_models/'
video_path = polybox_path + 'pan_detect/test_videos'


video_name = 'I_2017-04-06-20_08_45_begg.mp4'
video_name = 'I_20170425_205126_scegg.mp4'     # I_scegg
video_name = 'I_20170504_221703_segg.mp4'      # I_scegg
video_name = 'begg_test_2.mp4'      # I_begg1
video_name = 'segg_short.mov'      # I_scegg
video_name = 'scegg_test_1.mp4'
video_path = os.path.join(polybox_path, 'pan_detect', 'test_videos', video_name)

pan_model_name = '2017-05-11-16_44_38'
pan_model_name = '2017-04-27-15_19_51'  # I_begg1
pan_model_name = '2017-05-18-17_03_24'  # I_2 segg/scegg

food_model_name = '2017-05-18-14_19_44'

# Load pan detect model
pan_model = pickle.load(open(pan_models_path + 'M_' + pan_model_name + '.sav', 'rb'))
food_model = pickle.load(open(food_models_path + 'M_' + food_model_name + '.sav', 'rb'))

# Load pan_model info file
with open(pan_models_path + 'I_' + pan_model_name + '.txt', 'r') as file:
    _pan_params = eval(file.read())

print('Pan model parameters: ')
for key, val in _pan_params.items():
    print('\t{}: {}'.format(key, val))

# Load food_model info file
with open(food_models_path + 'I_' + food_model_name + '.txt', 'r') as file:
    _food_params = eval(file.read())

print('Food model parameters: ')
for key, val in _food_params.items():
    print('\t{}: {}'.format(key, val))


# Read corners and reshape them into 2d-Array
corners = np.reshape(_pan_params['corners'], (-1, 4))
plate_of_interest = int(_pan_params['plate_of_interest'])

# import images or videos or video stream
cap = cv2.VideoCapture(video_path)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# init pan_locator
pan_locator = PanLocator(_ellipse_smoothing=_ellipse_smoothing, _ellipse_method=_ellipse_method)

frame_id = 0
_end_frame = int(cap.get(7))

# for image in images
while cap.isOpened():

    ret, frame = cap.read()
    frame_id += 1

    if frame_id < _start_frame or (frame_id % _frame_rate != 0):
        continue
    elif frame_id + _start_frame > _end_frame:
        break

    # Todo: put preprocessing function here:
    patch = frame[corners[plate_of_interest - 1, 1]:corners[plate_of_interest - 1, 3],
                  corners[plate_of_interest - 1, 0]:corners[plate_of_interest - 1, 2]]

    patch_normalized = histogram_equalization(patch)

    pan_feature = get_HOG(patch_normalized,
                          orientations=_pan_params['feature_params']['orientations'],
                          pixels_per_cell=_pan_params['feature_params']['pixels_per_cell'],
                          cells_per_block=_pan_params['feature_params']['cells_per_block'],
                          widthPadding=_pan_params['feature_params']['widthPadding'])

    label_predicted_id = pan_model.predict(pan_feature)
    label_predicted_name = _pan_params['labels'][int(label_predicted_id)]

    if 'pan' in label_predicted_name or 'lid' in label_predicted_name:

        center, axes, phi = pan_locator.find_pan(patch)

        # Run Object Recognition inside pan
        mask = np.zeros_like(patch)
        ellipse_mask = cv2.ellipse(mask, tuple(map(int, center)), tuple(map(int, axes)),
                                   int(-phi*180/pi), 0, 360, (255, 255, 255), thickness=-1)

        if _food_params['feature_type'] == 'RGB_HIST':
            food_feature = np.zeros((3, _food_params['feature_params']['resolution']))

            for i in range(3):
                food_feature[i, :] = np.transpose(
                    cv2.calcHist([patch], [i], ellipse_mask[:, :, 0], [_food_params['feature_params']['resolution']], [0, 256]))

            food_feature = food_feature.flatten()

        food_label_predicted_id = food_model.predict(food_feature)
        food_label_predicted_name = _food_params['labels'][int(food_label_predicted_id)]

        # Plot contures of used edges
        # for x_it, y_it in zip(x, y):
        #    cv2.circle(patch, (y_it, x_it), 2, (0, 255, 0), -1)

        # Plot ellipse of fitted ellipse
        # cv2.ellipse(patch, tuple(map(int, raw_center)), tuple(map(int, raw_axes)),
        #             int(-raw_phi*180/pi), 0, 360, (255, 0, 0), thickness=2)

        # Plot ellipse of voted ellipse
        cv2.ellipse(patch, tuple(map(int, center)), tuple(map(int, axes)),
                    int(-phi*180/pi), 0, 360, (0, 0, 255), thickness=5)

        # # masked_patch = masked_patch[]
        # if _segment:
        #     masked_patch = np.bitwise_and(patch, ellipse_mask)
        #     fgmask = fgbg.apply(patch)
        #     fgmask = np.dstack((fgmask, fgmask, fgmask))
        #     plot_patch = np.bitwise_and(masked_patch, fgmask)
        #     plot_patch = masked_patch
        # else:
        #     plot_patch = patch

        plot_patch = patch

    else:
        plot_patch = patch

    cv2.putText(plot_patch, str(label_predicted_name), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0))
    cv2.putText(plot_patch, str(food_label_predicted_name), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0))
    cv2.imshow('predicted', plot_patch)
    cv2.waitKey(1)

print('Program finished successfully')

# plt.subplot(321), plt.hist(accu_center[0, :],normed=1, facecolor='green', alpha=0.75)
# plt.title('Center Voting'), plt.xticks([]), plt.yticks([])
# plt.grid(True)
# plt.subplot(322), plt.hist(accu_center[1, :],normed=1, facecolor='green', alpha=0.75)
# plt.title('Center Voting'), plt.xticks([]), plt.yticks([])
# plt.grid(True)
# plt.subplot(323), plt.hist(accu_axes[0, :],normed=1, facecolor='green', alpha=0.75)
# plt.title('Axes Voting'), plt.xticks([]), plt.yticks([])
# plt.grid(True)
# plt.subplot(324), plt.hist(accu_axes[1, :],normed=1, facecolor='green', alpha=0.75)
# plt.title('Axes Voting'), plt.xticks([]), plt.yticks([])
# plt.grid(True)
# plt.subplot(325), plt.hist(accu_phi[0, :],normed=1, facecolor='green', alpha=0.75)
# plt.title('Phi Voting'), plt.xticks([]), plt.yticks([])
# plt.grid(True)


# check if pan is detected -> fit patch to model
#   if detected:
#       locate pan
#       display state
#           run object recognition inside pan elipse:
#           display object