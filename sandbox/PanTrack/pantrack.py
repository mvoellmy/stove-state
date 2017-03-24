# Find Elipses
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = '../../data/stills/stove_left_on.PNG'
img_path = '../../data/stills/boiling_water.PNG'
img_path = '../../data/stills/stove_left_on.PNG'
img_path = '../../data/stills/hot_butter_in_pan.PNG'
img = cv2.imread(img_path)

_use_sift = False
_plot_canny = False

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

# Histogram equalization
img2 = histogram_equalization(img)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Sift
if _use_sift:
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img2, None)
    img = cv2.drawKeypoints(img2, kp, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints.jpg', img2)

# Canny Edges
edges = cv2.Canny(img2, 200, 300)
if _plot_canny:
    plt.subplot(211), plt.imshow(img2, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(212), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

# Find contours
im2, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key = cv2.contourArea, reverse = True)[:50]

# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# cv2.imshow("Contours", img)
# cv2.waitKey(0)

for contour in contours:
    if True:
        print(cv2.isContourConvex(contour))
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 1)
        cv2.imshow("Contours", img)
        print("printed contour")
        cv2.waitKey(0)
