import cv2
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import max_clique
from skimage import feature
from math import cos, sin, pi, inf, sqrt
from matplotlib import pyplot as plt


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def get_HOG(img, orientations=4, pixels_per_cell=(16, 16), cells_per_block=(4, 4), widthPadding=10):
    """
    Calculates HOG feature vector for the given image.

    img is a numpy array of 2- or 3-dimensional image (i.e., grayscale or rgb).
    Color-images are first transformed to grayscale since HOG requires grayscale
    images.

    Reference: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Crop the image from left and right.
    if int(widthPadding) > 0:
        img = img[:, widthPadding:-widthPadding]

    # Note that we are using skimage.feature.
    hog_features = feature.hog(img, orientations, pixels_per_cell, cells_per_block)

    return hog_features


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


def point_to_line(point, theta, r):
    '''
    :param point: 
    :param theta: 
    :param r: 
    :return: -1: Point is closer to center than line
             1:  Point is farther to center than line
             0:  Point is on line
    '''
    if (point[1] * cos(theta * pi / 180) + point[0] * sin(theta * pi / 180) - r) < 0:
        return -1
    elif (point[1] * cos(theta * pi / 180) + point[0] * sin(theta * pi / 180) - r) > 0:
        return 1
    else:
        return 0


def points_to_line(pixelpoints, best_theta, best_r, _plot_tangent=False):
    '''
    :param pixelpoints: 
    :param best_theta: 
    :param best_r: 
    :param _plot_tangent:
    :return: ratio: Ratio between points on both sides
             direction: side on which most points are
                        -1: Point is closer to center than line
                        1:  Point is farther to center than line
                        0:  Point is on line
    '''

    # Find Convex contours
    outer = 0
    inner = 0
    online = 0

    # https://math.stackexchange.com/questions/757591/how-to-determine-the-side-on-which-a-point-lies
    for point in pixelpoints:
        if point_to_line(point, best_theta, best_r) == -1:
            inner += 1
            if _plot_tangent:
                plt.scatter(point[1], point[0], color='green', s=5, zorder=4)
        elif point_to_line(point, best_theta, best_r) == 1:
            outer += 1
            if _plot_tangent:
                plt.scatter(point[1], point[0], color='blue', s=5, zorder=4)
        elif point_to_line(point, best_theta, best_r) == 0:
            online += 1
            if _plot_tangent:
                plt.scatter(point[1], point[0], color='yellow', s=5, zorder=4)

    plt.show()

    if outer < inner:
        ratio = outer / inner
        direction = -1
    elif outer > inner:
        ratio = inner / outer
        direction = 1
    else:
        ratio = 1
        direction = 0

    return ratio, direction


def get_max_clique(c):
    G = nx.from_numpy_matrix(c)
    return list(nx.find_cliques(G))


def plot_histogram(hist_item, index):

    # Create window to display image
    cv2.namedWindow('colorhist {}'.format(index+1), 2)

    # Set hist parameters

    hist_height = 64
    hist_width = 256
    nbins = 32
    bin_width = hist_width / nbins

    # Create an empty image for the histogram
    h = np.zeros((hist_height, hist_width))

    # Create array for the bins
    bins = np.arange(nbins, dtype=np.int32).reshape(nbins, 1)

    # Calculate and normalise the histogram
    cv2.normalize(hist_item, hist_item, hist_height, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    pts = np.column_stack((bins, hist))

    # Loop through each bin and plot the rectangle in white
    for x, y in enumerate(hist):
        cv2.rectangle(h, (int(x * bin_width), y), (int(x * bin_width + bin_width - 1), hist_height), (255, 0, 0), -1)

    # Flip upside down
    h = np.flipud(h)

    # Show the histogram
    cv2.imshow('colorhist {}'.format(index+1), h)
    cv2.waitKey(1)