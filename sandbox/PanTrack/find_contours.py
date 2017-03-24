import numpy as np
import cv2
from matplotlib import pyplot as plt

im = cv2.imread('../../data/stills/hot_butter_in_pan.PNG')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    cv2.drawContours(im2, contour, 0, (0, 255, 0), 1)

    print("Anotherone...")
    plt.figure()
    plt.imshow(im2)
    plt.show()

