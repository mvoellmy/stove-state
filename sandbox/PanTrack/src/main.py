import cv2
import configparser
import numpy as np
import os

import pickle

from math import pi

# Own Libraries
from panlocator import PanLocator
from helpers import mse, get_HOG

# Params
_ellipse_smoothing = 'AVERAGE'
_ellipse_smoothing = 'RAW'
_ellipse_smoothing = 'VOTE'

_ellipse_method = 'RANSAC'
_ellipse_method = 'CONVEX'
_ellipse_method = 'MAX_ARC'

_segment = False

_start_frame = 200

# Read config
cfg_path = '../../../cfg/class_cfg.txt'
config = configparser.ConfigParser()
config.read(cfg_path)

polybox_path = config.get('paths', 'polybox')

features_path = polybox_path + 'pan_detect/features/'
models_path = polybox_path + 'pan_detect/models/'
video_path = polybox_path + 'pan_detect/test_videos'


video_name = 'I_2017-04-06-20_08_45_begg.mp4'
video_name = 'I_20170425_205126_scegg.mp4'     # I_scegg
video_name = 'I_20170504_221703_segg.mp4'      # I_scegg
video_name = 'begg_test_2.mp4'      # I_begg1
video_name = 'segg_short.mov'      # I_scegg
video_path = os.path.join(polybox_path, 'pan_detect', 'test_videos', video_name)

model_name = '2017-05-11-16_44_38'
model_name = '2017-04-27-15_19_51'  # I_begg1
model_name = '2017-05-15-15_27_09'  # I_2 segg/scegg


# Load pan detect model
pan_model = pickle.load(open(models_path + 'M_' + model_name + '.sav', 'rb'))

# Load pan_detect info file
with open(models_path + 'I_' + model_name + '.txt', 'r') as file:
    _params = eval(file.read())

print('Model parameters: ')
for key, val in _params.items():
    print('\t{}: {}'.format(key, val))


# Read corners and reshape them into 2d-Array
corners = np.reshape(_params['corners'], (-1, 4))
plate_of_interest = int(_params['plate_of_interest'])

# import images or videos or video stream
cap = cv2.VideoCapture(video_path)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# init pan_locator
pan_locator = PanLocator(_ellipse_smoothing=_ellipse_smoothing, _ellipse_method=_ellipse_method)

frame_id = 0
ellips_counter = 0
_end_frame = int(cap.get(7))


# for image in images
while True:

    ret, frame = cap.read()
    frame_id += 1

    if frame_id < _start_frame:
        continue
    elif frame_id - _start_frame > _end_frame:
        break

    # Todo: put preprocessing function here:
    patch = frame[corners[plate_of_interest - 1, 1]:corners[plate_of_interest - 1, 3],
                  corners[plate_of_interest - 1, 0]:corners[plate_of_interest - 1, 2]]

    # hog = get_HOG(patch, orientations=int(_params['orientations']),
    #               pixels_per_cell=_params['pixels_per_cell'],
    #               cells_per_block=_params['cells_per_block'],
    #               widthPadding=int(_params['widthPadding']))

    hog = get_HOG(patch,
                  orientations=_params['feature_params']['orientations'],
                  pixels_per_cell=_params['feature_params']['pixels_per_cell'],
                  cells_per_block=_params['feature_params']['cells_per_block'],
                  widthPadding=_params['feature_params']['widthPadding'])

    label_predicted_id = pan_model.predict(hog)
    label_predicted_name = _params['labels'][int(label_predicted_id)]

    if 'pan' in label_predicted_name or 'lid' in label_predicted_name:

        center, axes, phi = pan_locator.find_pan(patch)

        # for x_it, y_it in zip(x, y):
        #    cv2.circle(patch, (y_it, x_it), 2, (0, 255, 0), -1)

        # cv2.ellipse(patch, tuple(map(int, raw_center)), tuple(map(int, raw_axes)),
        #             int(-raw_phi*180/pi), 0, 360, (255, 0, 0), thickness=2)
        cv2.ellipse(patch, tuple(map(int, center)), tuple(map(int, axes)),
                    int(-phi*180/pi), 0, 360, (0, 0, 255), thickness=5)

        # Run Object Segementation/Recognition inside pan
        mask = np.zeros_like(patch)
        ellipse_mask = cv2.ellipse(mask, tuple(map(int, center)), tuple(map(int, axes)),
                                     int(-phi*180/pi), 0, 360, (255, 255, 255), thickness=-1)

        # masked_patch = masked_patch[]
        if _segment:
            masked_patch = np.bitwise_and(patch, ellipse_mask)
            fgmask = fgbg.apply(patch)
            fgmask = np.dstack((fgmask, fgmask, fgmask))
            plot_patch = np.bitwise_and(masked_patch, fgmask)
            plot_patch = masked_patch
        else:
            plot_patch = patch

    else:
        plot_patch = patch

    cv2.putText(plot_patch, str(label_predicted_name), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 100, 0))
    cv2.imshow('predicted', plot_patch)
    cv2.waitKey(1)


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