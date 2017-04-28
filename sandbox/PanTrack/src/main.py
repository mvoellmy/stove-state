import cv2
import configparser
import ast
import numpy as np
import csv
import os.path
import time
import pickle


# Own Libraries
from locate_pan import locate_pan
from helpers import mse, get_HOG

# Options
cfg_path = '../../../cfg/class_cfg.txt'
features_path = '../features/'
models_path = '../models/'

video_path = 'I_2017-04-06-20_08_45_begg.mp4'
video_path = 'demovideo.mp4'
model_name = '2017-04-27-15_19_51'

ellipse_method = 'RANSAC'
ellipse_method = 'MAX_ARCH'
ellipse_method = 'CONVEX'

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
nr_of_frames = int(cap.get(7))

# for image in images
while frame_id < nr_of_frames:
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
        center, axes, phi, xx, yy = locate_pan(patch, _plot_ellipse=0, method=ellipse_method)
        center = center[::-1]
        axes = axes[::-1]
        cv2.ellipse(patch, tuple(map(int, center)), tuple(map(int, axes)), int(-phi*180/3.1415), 0, 360, (0, 0, 255), thickness=5)

    cv2.putText(patch, str(label_predicted), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 0))
    #cv2.imshow('predicted', patch)
    #cv2.waitKey(1)

# check if pan is detected -> fit patch to model
#   if detected:
#       locate pan
#       display state
#           run object recognition inside pan elipse:
#           display object



