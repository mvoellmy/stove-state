import cv2
import configparser
import ast
import numpy as np
import os
import os.path

# Options
video_type = '.mp4'
stove_type = 'M'
plates_of_interest = np.array([0, 1, 0, 0])  # Ians kitchen

# Read Config data
config = configparser.ConfigParser()
config.read('../cfg/cfg.txt')

# Read corners and reshape them into 2d-Array
corners = np.reshape(ast.literal_eval(config.get("corners", "coordinates")), (-1, 2))
print(corners)
print(corners[0, 1])

# Read dataset
path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')

has_pan = np.zeros((1, 4))


# Import & parse dataset into labels and frames
list_videos = []
list_videos = [f for f in os.listdir(path_videos) if os.path.isfile(os.path.join(path_videos, f))
                                                     and video_type in f
                                                     and f.startswith(stove_type)]

# for videos in folder
path_video = path_videos + '2017-03-31-05_01_45_egg_boiled.mp4'

patch_width = int(np.floor((corners[1, 0] - corners[0, 0]) / 2))
patch_height = int(np.floor((corners[2, 1] - corners[0, 1]) / 2))
print(type(patch_width))
print(patch_width)
frames = np.zeros((np.count_nonzero(plates_of_interest), patch_height, patch_width))

print(frames)


for video in list_videos:
    # Read file
    path_video = path_videos + video
    cap = cv2.VideoCapture(path_video)

    # for frames in video
    while cap.isOpened():
        ret, frame = cap.read()

        cv2.rectangle(frame, tuple(corners[0, :]), tuple(corners[3, :]), 255)

        cv2.imshow('frame', frame)

        cv2.waitKey(1)
        input()
        # split plates
        for k in plates_of_interest:
            if k > 0:
                print("extracting patch")
        # extract features and save labels
# Extract Features from images