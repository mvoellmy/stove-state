# Find Elipses
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = '../../data/stills/stove_left_on.PNG'
img_path = '../../data/stills/boiling_water.PNG'
img_path = '../../data/stills/stove_left_on.PNG'
img_path = '../../data/stills/hot_butter_in_pan.PNG'
img = cv2.imread(img_path, 0)


def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()

    # cdf_normalized = cdf * hist.max() / cdf.max()
    # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(img.flatten(),256,[0,256], color = 'r')
    # plt.xlim([0,256])
    # plt.legend(('cdf','histogram'), loc = 'upper left')
    # plt.show()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    return cdf[img]


img2 = histogram_equalization(img)

edges = cv2.Canny(img2, 200, 300)
plt.subplot(211), plt.imshow(img2, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()