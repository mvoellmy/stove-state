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

video_path = 'test.mp4'
model_name = 'test'


# Load pan detect model
pan_model = pickle.load(open(models_path + 'M_' + model_name + '.sav', 'rb'))

# Load pan_detect info file
_params = {}
for key, val in csv.reader(open(models_path + 'I_' + model_name + '.csv')):
    _params[key] = val
    print('     {}: {}'.format(key, type(val)))

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

    patch = frame[corners[plate_of_interest - 1, 1]:corners[plate_of_interest - 1, 3],
            corners[plate_of_interest - 1, 0]:corners[plate_of_interest - 1, 2]]

    # hog = get_HOG(patch, orientations=int(_params['orientations']),
    #               pixels_per_cell=_params['pixels_per_cell'],
    #               cells_per_block=_params['cells_per_block'],
    #               widthPadding=int(_params['widthPadding']))

    hog = get_HOG(patch)

    label_predicted = pan_model.predict(hog)
    cv2.imshow('predicted', patch)
    print(label_predicted)
    cv2.waitKey(1)


# check if pan is detected -> fit patch to model
#   if detected:
#       locate pan
#       display state
#           run object recognition inside pan elipse:
#           display object



