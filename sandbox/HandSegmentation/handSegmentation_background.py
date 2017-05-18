# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:57:49 2017

@author: ian
"""

import numpy as np
import cv2
import os
import configparser
from os.path import join

# Import Files ---------------------------------------------------------
config = configparser.ConfigParser()
config.read('../../cfg/cfg.txt')

path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')
file_name = 'M_2017-03-31-05_01_45_egg_boiled'
video_format = '.mp4'
file_name = 'M_2017-04-06-07_06_40_begg'
video_format = '.h264'

# path_videos = '../../../../Polybox/Shared/stove-state-data/ssds/test_gestures/'
# path_labels = ''
# file_name = 'test_skin_dark'

# Main Program ---------------------------------------------------------
cap = cv2.VideoCapture(join(path_videos, file_name + video_format))

count = 0
s = 1
d = 0
keyframe_dist = 1
last_frames = []
while (cap.isOpened()):
    ret, frame = cap.read()
    dim = frame.shape
    segmented = frame

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_old = gray_image
    # Hand segmentation
    #frame = frame.astype(int)
    # blur = cv2.GaussianBlur(gray_image, (d, d), s)
    # segmented = blur
    # last_frames.append(gray_image)
    # if count > keyframe_dist:
    #     gray_old = last_frames.pop(0)
    #     blur_old = cv2.GaussianBlur(gray_old, (d, d), s)
    #     segmented = cv2.subtract(blur,blur_old)

    if count > 1:
        segmented = cv2.subtract(gray_image, gray_old)
    gray_old = gray_image

    count += 1
    # Display images
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)
    cv2.resizeWindow("Frame", int(dim[1] / 2), int(dim[0] / 2))
    cv2.namedWindow("Segmented", cv2.WINDOW_NORMAL)
    cv2.imshow("Segmented", segmented)
    cv2.resizeWindow("Segmented", int(dim[1] / 2), int(dim[0] / 2))
    k = cv2.waitKey(30)
    if k == 27: # Exit by pressing escape-key
        break

cv2.destroyAllWindows()