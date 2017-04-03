# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:57:49 2017

@author: ian
"""

import numpy as np
import cv2
import os

#cap = cv2.VideoCapture("../../data/place_noodles.mp4")
cap = cv2.VideoCapture("../../../../Polybox/stove-state-data/salmon_noodles.mp4")

count = 0
s = 1
d = 0
keyframe_dist = 1
last_frames = []
while (cap.isOpened()):
    ret, frame = cap.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_old = gray_image
    # Hand segmentation
    #frame = frame.astype(int)
    blur = cv2.GaussianBlur(gray_image, (d, d), s)
    segmented = blur
    last_frames.append(gray_image)
    if count > keyframe_dist:
        frame_old = last_frames.pop(0)
        blur_old = cv2.GaussianBlur(frame_old, (d, d), s)
        segmented = cv2.subtract(blur,blur_old)

    count += 1
    # Display images
    cv2.imshow("Segmented", segmented)
    k = cv2.waitKey(30)
    if k == 27: # Exit by pressing escape-key
        break

cv2.destroyAllWindows()