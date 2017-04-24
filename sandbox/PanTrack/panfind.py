# Find Elipses
import cv2
import numpy as np
from matplotlib import pyplot as plt

from pylab import *
from fitEllipse import *



# Options
_plot_canny = False
_plot_cnt = False
_plot_ellipse = True

# Read Images
img_path = '../../data/stills/stove_left_on.PNG'
img_path = '../../data/stills/stove_left_on.PNG'
img_path = '../../data/stills/stove_top.PNG'
img_path = '../../data/stills/hot_butter_in_pan.PNG'
img_path = '../../data/stills/frame-000055.jpg'
img_path = '../../data/stills/boiling_water.PNG'

img = cv2.imread(img_path)


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


img = histogram_equalization(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(img, 100, 200)

if _plot_canny:
    plt.subplot(211), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(212), plt.imshow(canny, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


# Find contours
im2, contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)[:20]

edges = []

if _plot_ellipse:
    plt.subplot(211), plt.imshow(canny, cmap='gray')
    plt.title('Original Canny'), plt.xticks([]), plt.yticks([])
    plt.subplot(212), plt.imshow(canny, cmap='gray')
    plt.title('Elipses'), plt.xticks([]), plt.yticks([])
    plt.imshow(canny, cmap='gray', zorder=1)

for cnt in contours:
    mask = np.zeros(canny.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)

    edge = mask * canny
    edges.append(edge)

    if _plot_cnt:
        plt.subplot(211), plt.imshow(mask, cmap='gray')
        plt.title('Mask'), plt.xticks([]), plt.yticks([])
        plt.subplot(212), plt.imshow(edge, cmap='gray')
        plt.title('Contour'), plt.xticks([]), plt.yticks([])
        plt.show()

    print("fitting ellipse")

    pixelpoints = np.transpose(np.nonzero(mask))
    x = pixelpoints[:, 0]
    y = pixelpoints[:, 1]

    a = fitEllipse(x, y)
    center = ellipse_center(a)
    # phi = ellipse_angle_of_rotation(a)
    phi = ellipse_angle_of_rotation2(a)
    axes = ellipse_axis_length(a)

    print("center = ", center)
    print("angle of rotation = ", phi)
    print("axes = ", axes)

    arc_ = 2
    R_ = np.arange(0, arc_ * np.pi, 0.01)
    a, b = axes
    xx = center[0] + a * np.cos(R_) * np.cos(phi) - b * np.sin(R_) * np.sin(phi)
    yy = center[1] + a * np.cos(R_) * np.sin(phi) + b * np.sin(R_) * np.cos(phi)

    if _plot_ellipse:
        plt.scatter(y, x,s=.5, zorder=2)
        plt.scatter(yy, xx, color='red', s=0.5, zorder=3)

plt.show()