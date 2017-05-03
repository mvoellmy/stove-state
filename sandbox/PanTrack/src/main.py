import cv2
import configparser
import ast
import numpy as np
import csv
import os.path
import time
import pickle
from matplotlib import pyplot as plt

from math import pi

# Own Libraries
from locate_pan import locate_pan
from helpers import mse, get_HOG

plt.figure()
input('should be a figure here....')
# Params
_ellipse_smoothing = 'ACCU'

# Options
cfg_path = '../../../cfg/class_cfg.txt'
features_path = '../features/'
models_path = '../models/'

video_path = 'I_2017-04-06-20_08_45_begg.mp4'
video_path = 'demovideo.mp4'
model_name = '2017-04-27-15_19_51'

ellipse_method = 'RANSAC'
ellipse_method = 'CONVEX'
ellipse_method = 'MAX_ARCH'

# Load pan detect model
pan_model = pickle.load(open(models_path + 'M_' + model_name + '.sav', 'rb'))

# Load pan_detect info file
_params = {}
for key, val in csv.reader(open(models_path + 'I_' + model_name + '.csv')):
    _params[key] = val
    print('     {}: {}'.format(key, val))

# Read config
config = configparser.ConfigParser()
config.read(cfg_path)

# Read corners and reshape them into 2d-Array
corners = np.reshape(ast.literal_eval(config.get(_params['stove_type'], "corners")), (-1, 4))

plate_of_interest = int(config.get(_params['stove_type'], "plate_of_interest"))



# pan localization model
# object recognition model
#
# import images or videos or video stream
cap = cv2.VideoCapture(video_path)
frame_id = 0
ellips_counter = 0
nr_of_frames = int(cap.get(7))

if _ellipse_smoothing == 'ACCU':
    res_center = 300
    res_phi = 180
    res_axes = 300
    accu_center = np.zeros((2, res_center))
    accu_phi = np.zeros((1, res_phi))
    accu_axes = np.zeros((2, res_axes))

# for image in images
# while frame_id < 300:
while frame_id < nr_of_frames:
    frame_id += 1
    ret, frame = cap.read()

    # Todo: put preprocessing function here:
    patch = frame[corners[plate_of_interest - 1, 1]:corners[plate_of_interest - 1, 3],
            corners[plate_of_interest - 1, 0]:corners[plate_of_interest - 1, 2]]

    # hog = get_HOG(patch, orientations=int(_params['orientations']),
    #               pixels_per_cell=_params['pixels_per_cell'],
    #               cells_per_block=_params['cells_per_block'],
    #               widthPadding=int(_params['widthPadding']))

    hog = get_HOG(patch)
    label_predicted = pan_model.predict(hog)

    if label_predicted == 0:
        ellips_counter += 1

        raw_center, raw_axes, raw_phi, x, y = locate_pan(patch, _plot_ellipse=0, method=ellipse_method)
        raw_center = raw_center[::-1]
        raw_axes = raw_axes[::-1]

        if _ellipse_smoothing == 'AVERAGE':
            if ellips_counter == 1:
                center, axes, phi = raw_center, raw_axes, raw_phi
            else:
                center = (center*(ellips_counter-1) + raw_center)/ellips_counter
                axes = (axes*(ellips_counter-1) + raw_axes)/ellips_counter
                phi = (phi*(ellips_counter-1) + raw_phi)/ellips_counter
        elif _ellipse_smoothing == 'ACCU':

            patch_size = patch.shape
            accu_center[0, int(raw_center[0]/patch_size[0]*res_center)] += 1
            accu_center[1, int(raw_center[1]/patch_size[1]*res_center)] += 1
            accu_axes[0, np.min([res_axes-1, int(raw_axes[0]/(patch_size[0])*res_axes)])] += 1
            accu_axes[1, np.min([res_axes-1, int(raw_axes[1]/(patch_size[1])*res_axes)])] += 1
            accu_phi[0, int(raw_phi/pi*res_phi)] += 1

            if ellips_counter < 3:
                center, axes, phi = raw_center, raw_axes, raw_phi
            else:
                center[0] = np.argmax(accu_center[0, :])*patch_size[0]/res_center
                center[1] = np.argmax(accu_center[1, :])*patch_size[1]/res_center
                axes[0] = np.argmax(accu_axes[0, :])*(patch_size[0])/res_axes
                axes[1] = np.argmax(accu_axes[1, :])*(patch_size[1])/res_axes
                phi = np.argmax(accu_phi)*pi/res_phi

        else:
            center, axes, phi = raw_center, raw_axes, raw_phi

        cv2.ellipse(patch, tuple(map(int, raw_center)), tuple(map(int, raw_axes)), int(-raw_phi*180/pi), 0, 360, (255, 0, 0), thickness=2)
        cv2.ellipse(patch, tuple(map(int, center)), tuple(map(int, axes)), int(-phi*180/pi), 0, 360, (0, 0, 255), thickness=5)
        for x_it, y_it in zip(x, y):
            cv2.circle(patch, (y_it, x_it), 2, (0, 255, 0), -1)

    cv2.putText(patch, str(label_predicted), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 0))
    cv2.imshow('predicted', patch)
    cv2.waitKey(1)


# plt.figure()
# plt.subplot(321), plt.hist(accu_center[0, :])
# plt.title('Center Voting'), plt.xticks([]), plt.yticks([])
# plt.subplot(322), plt.hist(accu_center[1, :])
# plt.title('Center Voting'), plt.xticks([]), plt.yticks([])
# plt.subplot(323), plt.hist(accu_axes[0, :])
# plt.title('Axes Voting'), plt.xticks([]), plt.yticks([])
# plt.subplot(324), plt.hist(accu_axes[1, :])
# plt.title('Axes Voting'), plt.xticks([]), plt.yticks([])
# plt.subplot(325), plt.hist(accu_phi[0, :])
# plt.title('Phi Voting'), plt.xticks([]), plt.yticks([])

# plt.show()
input('Program finished')
# check if pan is detected -> fit patch to model
#   if detected:
#       locate pan
#       display state
#           run object recognition inside pan elipse:
#           display object



