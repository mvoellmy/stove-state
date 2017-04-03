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
#cap = cv2.VideoCapture("../../data/salmon_noodles.mp4")


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
     segmented2 = (segmented == np.zeros(segmented.shape)) * 1
     segmented2 = segmented2.astype(np.uint8)

     #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     #blur = cv2.GaussianBlur(gray, (1, 1), 0)
     #ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

     output = cv2.connectedComponentsWithStats(segmented2)
     num_components = output[0]
     img = output[1]
     idx_max = 0
     val_max = 0
     for i in range(1,10):
          val = np.sum(np.sum(img == np.ones(img.shape)*i)*1)
          if val > val_max:
               val_max = val
               idx_max = i


     #img = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     #all_labels = measure.label(segmented)
     S = (img == np.ones(img.shape)*idx_max) * 1
     S = cv2.multiply(S,255)
     S = S.astype(np.uint8)

     # Display images
     plt.imshow(segmented)
     cv2.imshow("Frame", frame)
     cv2.imshow("Segmented", segmented)
     cv2.imshow("Final", S)
     k = cv2.waitKey(1)
     if k == 27: # Exit by pressing escape-key
         break

cv2.destroyAllWindows()