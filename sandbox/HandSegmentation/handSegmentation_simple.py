# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:57:49 2017

@author: ian
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import measure

cap = cv2.VideoCapture("../../data/place_noodles.mp4")
cap = cv2.VideoCapture("../../data/test_dark.mp4")
#cap = cv2.VideoCapture("../../../../Polybox/Shared/stove-state-data/ssds/gestures/remove_egg.mp4")
#cap = cv2.VideoCapture("../../data/salmon_noodles.mp4")


while (cap.isOpened()):
     ret, frame = cap.read()
    
     # Hand segmentation
     frame_int = frame.astype(int)
     height, width = frame.shape[:2]

     threshold = -70
     r_g = frame_int[:,:,0] - frame_int[:,:,1]
     r_b = frame_int[:,:,0] - frame_int[:,:,2]
     lowest = cv2.min(r_g, r_b)
     # Compare lowest > threshold
     segmented =  cv2.compare(lowest, threshold, cmpop=1)
     segmented2 = (segmented == np.zeros(segmented.shape)) * 1
     segmented2 = segmented2.astype(np.uint8)



     # Display images
     plt.imshow(segmented)
     cv2.imshow("Frame", frame)
     cv2.imshow("Segmented", segmented)
     k = cv2.waitKey(1)
     if k == 27: # Exit by pressing escape-key
         break

cv2.destroyAllWindows()