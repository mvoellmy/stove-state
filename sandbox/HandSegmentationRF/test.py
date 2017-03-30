# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:39:49 2017

@author: ian
"""

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from random import randint
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("../../data/place_noodles.mp4")
mypath='../../data/In-airGestures/Training/gesture1/RGB'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)

tree_depth = 10
n_nodes = 2**(tree_depth-1)
Gamma_modes = np.array([[0,0],[0,1],[1,0],[1,1]])
Gamma = np.empty((n_nodes,2), dtype=int)
u_rand = np.empty((n_nodes,2), dtype=int)
for i in range(0, n_nodes):
    Gamma[i] = Gamma_modes[randint(0,3),:]
    u_rand[i] = np.array([[randint(-5,5), randint(-5,5)]])
    
#%% 
frame = cv2.imread( join(mypath,onlyfiles[0]) )
frame_shifted_all = np.empty((frame.shape[0], frame.shape[1], n_nodes), dtype=int)
for n in range(0, len(onlyfiles)):
    frame = cv2.imread( join(mypath,onlyfiles[n]) )
    
    # Hand segmentation
    frame = frame.astype(int)
    height, width = frame.shape[:2]
    threshold = -50
    r_g = frame[:,:,0] - frame[:,:,1]
    r_b = frame[:,:,0] - frame[:,:,2]
    lowest = cv2.min(r_g, r_b)
    # Compare lowest > threshold
    segmented =  cv2.compare(lowest, threshold, cmpop=1)
    k = 0
    for idx, val in enumerate(u_rand[:]):
        frame_shifted = np.roll(segmented, val[0], 0)
        frame_shifted = np.roll(frame_shifted, val[1], 1)
        frame_shifted_all[:,:,idx] = frame_shifted
    # Display images
    cv2.imshow("Segmented", segmented)
    k = cv2.waitKey(1000)
    if k == 27: # Exit by pressing escape-key
        break
    
    # Train data
    
    
cv2.destroyAllWindows()