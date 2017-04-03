# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:57:49 2017

@author: ian
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("../../data/place_noodles.mp4")
#cap = cv2.VideoCapture("../../../../Polybox/stove-state-data/salmon_noodles.mp4")


while (cap.isOpened()):
     ret, frame = cap.read()
    
     # Hand segmentation
     frame_int = frame.astype(int)
     height, width = frame.shape[:2]
     threshold = -50
     r_g = frame_int[:,:,0] - frame_int[:,:,1]
     r_b = frame_int[:,:,0] - frame_int[:,:,2]
     lowest = cv2.min(r_g, r_b)
     # Compare lowest > threshold
     segmented =  cv2.compare(lowest, threshold, cmpop=1)
    
     # Display images
     plt.imshow(segmented)
     cv2.imshow("Segmented", segmented)
     cv2.imshow("Frame", frame)
     k = cv2.waitKey(30)
     if k == 27: # Exit by pressing escape-key
         break

cv2.destroyAllWindows()