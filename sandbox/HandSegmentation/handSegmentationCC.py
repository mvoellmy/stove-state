# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:57:49 2017

@author: ian
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import configparser
from os.path import join
from skimage import measure


config = configparser.ConfigParser()
config.read('../../cfg/cfg.txt')

path_videos = config.get('paths', 'videos')
path_labels = config.get('paths', 'labels')
file_name = 'I_2017-04-02-10_14_19_schnitzel'
video_format = '.mp4'

path_videos = '../../../../Polybox/Shared/stove-state-data/ssds/test/'
path_labels = ''
file_name = 'place_schnitzel_1'

cap = cv2.VideoCapture(join(path_videos, file_name + video_format))

while (cap.isOpened()):
     ret, frame = cap.read()
     dim = frame.shape
    
     # Hand segmentation
     frame_int = frame.astype(int)
     height, width = frame.shape[:2]

     threshold = -70
     r_g = frame_int[:,:,0] - frame_int[:,:,1]
     r_b = frame_int[:,:,0] - frame_int[:,:,2]
     lowest = cv2.min(r_g, r_b)
     # Compare lowest > threshold
     segmented =  cv2.compare(lowest, threshold, cmpop=1)
     segmented2 = (segmented == np.zeros(segmented.shape)) * 255
     segmented2 = segmented2.astype(np.uint8)

     #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     #blur = cv2.GaussianBlur(gray, (1, 1), 0)
     #ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

     labelnum, labelimg, contours, centroids = cv2.connectedComponentsWithStats(segmented2)
     num_components = labelnum
     img = labelimg
     idx_max = 0
     val_max = 0
     # for i in range(1,10):
     #      val = np.sum(np.sum(img == np.ones(img.shape)*i)*1)
     #      if val > val_max:
     #           val_max = val
     #           idx_max = i

     for label in range(1,labelnum):
          x, y, w, h, size = contours[label]
          if size > val_max:
              val_max = size
              idx_max = label
          segmented2 = cv2.rectangle(segmented2, (x, y), (x + w, y + h), (255, 255, 0), 5)

     #img = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     #all_labels = measure.label(segmented)
     final = (img == np.ones(img.shape)*idx_max) * 1
     final = cv2.multiply(final,255)
     final = final.astype(np.uint8)

     # Display images
     plt.imshow(segmented)
     cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
     cv2.imshow("Frame", segmented2)
     cv2.resizeWindow("Frame", int(dim[1] / 2), int(dim[0] / 2))
     # cv2.namedWindow("Segmented", cv2.WINDOW_NORMAL)
     # cv2.imshow("Segmented", segmented)
     # cv2.resizeWindow("Segmented", int(dim[1] / 2), int(dim[0] / 2))
     cv2.namedWindow("Final", cv2.WINDOW_NORMAL)
     cv2.imshow("Final", final)
     cv2.resizeWindow("Final", int(dim[1] / 2), int(dim[0] / 2))
     k = cv2.waitKey(1)
     if k == 27: # Exit by pressing escape-key
         break

cv2.destroyAllWindows()