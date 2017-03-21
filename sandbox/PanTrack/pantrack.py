# Find Elipses
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = '../../data/stills/hot_butter_in_pan.PNG'
img = cv2.imread(img_path, 0)
edges = cv2.Canny(img, 100, 300)
plt.subplot(211), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
