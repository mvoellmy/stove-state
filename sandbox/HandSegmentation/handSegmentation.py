# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:57:49 2017

@author: ian
"""

import numpy as np
import cv2
import configparser
from os.path import join

# Function Definitions ---------------------------------------------------
def segmentation_RGB(frame):
     frame_int = frame.astype(int)

     threshold = -40
     r_g = frame_int[:, :, 0] - frame_int[:, :, 1]
     r_b = frame_int[:, :, 0] - frame_int[:, :, 2]
     lowest = cv2.min(r_g, r_b)
     # Compare lowest > threshold
     segmented = cv2.compare(lowest, threshold, cmpop=1)
     segmented = (segmented == np.zeros(segmented.shape))*255
     segmented = segmented.astype(np.uint8)

     return segmented

def segmentation_YCC(frame):
    # http://stackoverflow.com/questions/14752006/computer-vision-masking-a-human-hand
    imgYCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    Y = imgYCC[:, :, 0]
    Cr = imgYCC[:, :, 1]
    Cb = imgYCC[:, :, 2]
    skin_ycrcb_mint = np.array((0, 150, 0))
    skin_ycrcb_maxt = np.array((255, 255, 255))
    skin_ycrcb = cv2.inRange(imgYCC, skin_ycrcb_mint, skin_ycrcb_maxt)

    return skin_ycrcb

def segmentation_HSV(frame):
    # http://stackoverflow.com/questions/14752006/computer-vision-masking-a-human-hand
    skin_min = np.array([0, 40, 150], np.uint8)
    skin_max = np.array([20, 150, 255], np.uint8)
    gaussian_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    blur_hsv = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2HSV)
    segmented = cv2.inRange(blur_hsv, skin_min, skin_max)

    return segmented

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

fgbg = cv2.createBackgroundSubtractorKNN(history=2, dist2Threshold=1000)
# fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=128, detectShadows=True)

ret, background = cap.read()
background_segmented = segmentation_RGB(background)
kernel = np.ones((10,10),np.uint8)
background_segmented = cv2.dilate(background_segmented,kernel,iterations = 1)

count = 0
while (cap.isOpened()):
     ret, frame = cap.read()
     dim = frame.shape

     # Automatic Background Subtraction KNN
     # fgmask = fgbg.apply(frame)

     # Hand segmentation with color
     segmented = segmentation_RGB(frame)

     # Background subtraction of segmented images
     # segmented2 = cv2.subtract(segmented, background_segmented)

     # Movement
     gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     motion = gray_image
     if count > 1:
         motion = cv2.subtract(gray_image, gray_old)
     gray_old = gray_image
     count += 1

     # Combine color segmentation with motion
     segmented2 = cv2.multiply((segmented > 0)*1 , (motion > 10)*1) * 255
     segmented2 = segmented2.astype(np.uint8)

     # Display images
     cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
     cv2.imshow("Frame", frame)
     cv2.resizeWindow("Frame", int(dim[1] / 2), int(dim[0] / 2))
     cv2.namedWindow("Segmented", cv2.WINDOW_NORMAL)
     cv2.imshow("Segmented", segmented)
     cv2.resizeWindow("Segmented", int(dim[1] / 2), int(dim[0] / 2))
     cv2.namedWindow("asd", cv2.WINDOW_NORMAL)
     cv2.imshow("asd", segmented2)
     cv2.resizeWindow("asd", int(dim[1] / 2), int(dim[0] / 2))
     k = cv2.waitKey(1)
     if k == 27: # Exit by pressing escape-key
         break

cv2.destroyAllWindows()