# Find Elipses
import cv2
import numpy as np
from matplotlib import pyplot as plt

from fitEllipse import *
from pylab import *

img_path = '../../data/stills/stove_left_on.PNG'
img_path = '../../data/stills/boiling_water.PNG'
img_path = '../../data/stills/stove_left_on.PNG'
img_path = '../../data/stills/hot_butter_in_pan.PNG'
img = cv2.imread(img_path)

_use_sift = False
_plot_canny = False
_fit_ellipse = True

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
im2, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# cv2.imshow("Contours", img)
# cv2.waitKey(0)

x = []
y = []
edges = []

for contour in contours:
    if True:

        print(cv2.isContourConvex(contour))
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)

        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, contour, -1, 255, -1)
        # cv2.drawContours(img, [contour], -1, (0, 255, 0), 1)
        cv2.imshow("Contours", mask)
        print("printed contour")
        coords = contour[:, 0]
        x = coords[:, 0]
        y = coords[:, 1]

        cv2.waitKey(0)

    if _fit_ellipse:
        print("fitting ellipse")

        shape = contour.shape
        size = shape[1]

        arc = size*0.5
        R = np.arange(0,arc*np.pi, 0.01)

        a = fitEllipse(x, y)

        center = ellipse_center(a)
        #phi = ellipse_angle_of_rotation(a)
        phi = ellipse_angle_of_rotation2(a)
        axes = ellipse_axis_length(a)

        print("center = ",  np.floor(center))
        print("angle of rotation = ",  phi)
        print("axes = ", axes)


        arc_ = 2
        R_ = np.arange(0, arc_ * np.pi, 0.01)
        a, b = axes
        xx = center[0] + a * np.cos(R_) * np.cos(phi) - b * np.sin(R_) * np.sin(phi)
        yy = center[1] + a * np.cos(R_) * np.sin(phi) + b * np.sin(R_) * np.cos(phi)
        #
        # plot(x, y, 'bo', ms=1)
        # plot(xx, yy, color='red')
        # show()

        # cv2.ellipse(img, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), int(phi*180/3.14), 0, 360, 255 , 1)


